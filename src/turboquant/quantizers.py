from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codebooks import Codebook, CodebookRepository, precompute_codebook
from .packing import pack_indices, pack_signs, unpack_indices, unpack_signs
from .rotation import gaussian_orthogonal_matrix, gaussian_projection_matrix
from .types import PackedArray


def _ensure_2d(vectors: np.ndarray, dim: int | None = None) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("vectors must be a 1D or 2D array")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"expected dim={dim}, got {array.shape[1]}")
    return array


def _l2_norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=1)


def _normalize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = _l2_norms(vectors)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return vectors / safe_norms[:, None], norms


def _choose_centroids(values: np.ndarray, codebook: Codebook) -> np.ndarray:
    boundaries = codebook.boundaries[1:-1]
    return np.digitize(values, boundaries, right=False).astype(np.int32)


@dataclass(frozen=True)
class QuantizedMSE:
    indices: PackedArray
    norms: np.ndarray | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class QuantizedProd:
    indices: PackedArray
    qjl_signs: PackedArray
    residual_norms: np.ndarray
    norms: np.ndarray | None
    shape: tuple[int, int]


class TurboQuantMSE:
    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int,
        *,
        device: str = "cpu",
        store_norms: bool = True,
        codebook_mode: str = "precomputed",
        codebook_dir: str | Path = "artifacts/codebooks",
        codebook: Codebook | None = None,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self.store_norms = store_norms
        self.rotation = gaussian_orthogonal_matrix(dim, seed)
        self.codebook_repo = CodebookRepository(codebook_dir)
        if codebook is not None:
            self.codebook = codebook
        elif codebook_mode == "precomputed":
            self.codebook = self.codebook_repo.get_or_create(dim, bits)
        elif codebook_mode == "compute":
            self.codebook = precompute_codebook(dim, bits)
        else:
            raise ValueError(f"unsupported codebook mode: {codebook_mode}")

    def quantize(self, vectors: np.ndarray) -> QuantizedMSE:
        array = _ensure_2d(vectors, self.dim)
        normalized, norms = _normalize_rows(array)
        rotated = normalized @ self.rotation.T
        indices = _choose_centroids(rotated, self.codebook)
        packed, bit_count = pack_indices(indices.reshape(-1), self.bits)
        return QuantizedMSE(
            indices=PackedArray(packed, bit_count, indices.size),
            norms=norms.astype(np.float32) if self.store_norms else None,
            shape=array.shape,
        )

    def decode_rotated(self, quantized: QuantizedMSE) -> np.ndarray:
        flat = unpack_indices(quantized.indices.data, self.bits, quantized.indices.item_count)
        decoded = self.codebook.centroids[flat].reshape(quantized.shape)
        return decoded

    def dequantize(self, quantized: QuantizedMSE) -> np.ndarray:
        rotated = self.decode_rotated(quantized)
        reconstructed = rotated @ self.rotation
        if quantized.norms is not None:
            reconstructed = reconstructed * quantized.norms[:, None]
        return reconstructed.astype(np.float32)

    def reconstruct_unit(self, quantized: QuantizedMSE) -> np.ndarray:
        rotated = self.decode_rotated(quantized)
        return (rotated @ self.rotation).astype(np.float64)


class TurboQuantProd:
    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int,
        *,
        device: str = "cpu",
        store_norms: bool = True,
        codebook_mode: str = "precomputed",
        codebook_dir: str | Path = "artifacts/codebooks",
    ) -> None:
        if bits < 1:
            raise ValueError("bits must be at least 1")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self.store_norms = store_norms
        self.mse_bits = bits - 1
        self.base = None
        if self.mse_bits > 0:
            self.base = TurboQuantMSE(
                dim=dim,
                bits=self.mse_bits,
                seed=seed,
                device=device,
                store_norms=False,
                codebook_mode=codebook_mode,
                codebook_dir=codebook_dir,
            )
        self.projection = gaussian_projection_matrix(dim, seed + 1_000_003)

    def quantize(self, vectors: np.ndarray) -> QuantizedProd:
        array = _ensure_2d(vectors, self.dim)
        normalized, norms = _normalize_rows(array)
        if self.base is None:
            base_unit = np.zeros_like(normalized)
            packed_idx = np.zeros(0, dtype=np.uint8)
            idx_bits = 0
            idx_items = 0
        else:
            base_codes = self.base.quantize(normalized)
            base_unit = self.base.reconstruct_unit(base_codes)
            packed_idx = base_codes.indices.data.copy()
            idx_bits = base_codes.indices.bit_count
            idx_items = base_codes.indices.item_count
        residual = normalized - base_unit
        projected = residual @ self.projection.T
        signs = np.where(projected >= 0, 1.0, -1.0)
        packed_signs, sign_bits = pack_signs(signs.reshape(-1))
        return QuantizedProd(
            indices=PackedArray(packed_idx, idx_bits, idx_items),
            qjl_signs=PackedArray(packed_signs, sign_bits, signs.size),
            residual_norms=_l2_norms(residual).astype(np.float32),
            norms=norms.astype(np.float32) if self.store_norms else None,
            shape=array.shape,
        )

    def _decode_components(self, quantized: QuantizedProd) -> tuple[np.ndarray, np.ndarray]:
        if self.base is None:
            base_unit = np.zeros(quantized.shape, dtype=np.float64)
        else:
            mse_quant = QuantizedMSE(indices=quantized.indices, norms=None, shape=quantized.shape)
            base_unit = self.base.reconstruct_unit(mse_quant)
        qjl_signs = unpack_signs(quantized.qjl_signs.data, quantized.qjl_signs.item_count).reshape(quantized.shape)
        qjl_unit = math.sqrt(math.pi / 2.0) / self.dim * (qjl_signs @ self.projection)
        qjl_unit = qjl_unit * quantized.residual_norms[:, None]
        return base_unit, qjl_unit

    def dequantize(self, quantized: QuantizedProd) -> np.ndarray:
        base_unit, qjl_unit = self._decode_components(quantized)
        reconstructed = base_unit + qjl_unit
        if quantized.norms is not None:
            reconstructed = reconstructed * quantized.norms[:, None]
        return reconstructed.astype(np.float32)

    def score_inner_product(self, queries: np.ndarray, quantized: QuantizedProd) -> np.ndarray:
        queries_array = _ensure_2d(queries, self.dim)
        base_unit, qjl_unit = self._decode_components(quantized)
        decoded = base_unit + qjl_unit
        if quantized.norms is not None:
            decoded = decoded * quantized.norms[:, None]
        return queries_array @ decoded.T
