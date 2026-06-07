from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .distributions import integration_grid, weighted_interval_stats


@dataclass(frozen=True)
class Codebook:
    dim: int
    bits: int
    centroids: np.ndarray
    boundaries: np.ndarray
    masses: np.ndarray
    mse_cost: float
    grid_resolution: int
    iterations: int

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["centroids"] = self.centroids.tolist()
        payload["boundaries"] = self.boundaries.tolist()
        payload["masses"] = self.masses.tolist()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "Codebook":
        return cls(
            dim=int(payload["dim"]),
            bits=int(payload["bits"]),
            centroids=np.asarray(payload["centroids"], dtype=np.float64),
            boundaries=np.asarray(payload["boundaries"], dtype=np.float64),
            masses=np.asarray(payload["masses"], dtype=np.float64),
            mse_cost=float(payload["mse_cost"]),
            grid_resolution=int(payload["grid_resolution"]),
            iterations=int(payload["iterations"]),
        )


def _symmetric_initial_centroids(dim: int, bits: int) -> np.ndarray:
    levels = 2 ** bits
    sigma = 1.0 / np.sqrt(dim)
    normal_grid = np.linspace(-(levels - 1), levels - 1, levels, dtype=np.float64)
    normal_grid = normal_grid / max(np.abs(normal_grid).max(), 1.0)
    centroids = 2.5 * sigma * normal_grid
    return np.clip(centroids, -1.0, 1.0)


def _boundaries_from_centroids(centroids: np.ndarray) -> np.ndarray:
    mids = (centroids[:-1] + centroids[1:]) / 2.0
    return np.concatenate(([-1.0], mids, [1.0]))


def _compute_mse_cost(
    grid: np.ndarray,
    pdf: np.ndarray,
    centroids: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[float, np.ndarray]:
    total = 0.0
    masses = np.zeros_like(centroids)
    for idx, centroid in enumerate(centroids):
        left = boundaries[idx]
        right = boundaries[idx + 1]
        mask = (grid >= left) & (grid <= right)
        subgrid = grid[mask]
        subpdf = pdf[mask]
        if subgrid.size == 0:
            continue
        masses[idx] = np.trapezoid(subpdf, subgrid)
        total += np.trapezoid(np.square(subgrid - centroid) * subpdf, subgrid)
    mass_sum = masses.sum()
    if mass_sum > 0:
        masses = masses / mass_sum
    return float(total), masses


def precompute_codebook(
    dim: int,
    bits: int,
    *,
    resolution: int = 32769,
    max_iter: int = 256,
    tolerance: float = 1e-10,
) -> Codebook:
    if bits < 1:
        raise ValueError("bits must be at least 1")
    grid, pdf = integration_grid(dim, resolution=resolution)
    centroids = _symmetric_initial_centroids(dim, bits)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        boundaries = _boundaries_from_centroids(centroids)
        updated = np.empty_like(centroids)
        for idx in range(centroids.size):
            updated[idx], _ = weighted_interval_stats(grid, pdf, boundaries[idx], boundaries[idx + 1])
        updated = np.maximum.accumulate(updated)
        if np.max(np.abs(updated - centroids)) <= tolerance:
            centroids = updated
            break
        centroids = updated
    boundaries = _boundaries_from_centroids(centroids)
    mse_cost, masses = _compute_mse_cost(grid, pdf, centroids, boundaries)
    return Codebook(
        dim=dim,
        bits=bits,
        centroids=centroids,
        boundaries=boundaries,
        masses=masses,
        mse_cost=mse_cost,
        grid_resolution=resolution,
        iterations=iterations,
    )


class CodebookRepository:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, dim: int, bits: int) -> Path:
        return self.root / f"codebook_d{dim}_b{bits}.json"

    def save(self, codebook: Codebook) -> Path:
        path = self.path_for(codebook.dim, codebook.bits)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(codebook.to_json_dict(), indent=2))
        return path

    def load(self, dim: int, bits: int) -> Codebook:
        payload = json.loads(self.path_for(dim, bits).read_text())
        return Codebook.from_json_dict(payload)

    def get_or_create(self, dim: int, bits: int, **kwargs: object) -> Codebook:
        path = self.path_for(dim, bits)
        if path.exists():
            return self.load(dim, bits)
        codebook = precompute_codebook(dim, bits, **kwargs)
        self.save(codebook)
        return codebook
