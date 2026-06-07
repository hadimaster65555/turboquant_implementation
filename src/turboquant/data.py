from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MatrixSpec:
    source: str
    key: str | None = None
    split: str | None = None
    column: str | None = None
    limit: int | None = None
    offset: int = 0


def parse_source_spec(
    source: str,
    *,
    key: str | None = None,
    split: str | None = None,
    column: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> MatrixSpec:
    if source.startswith("hf://"):
        payload = source[len("hf://") :]
        dataset_name, _, maybe_column = payload.partition("::")
        if maybe_column and column is None:
            column = maybe_column
        return MatrixSpec(source=dataset_name, split=split or "train", column=column, limit=limit, offset=offset)
    return MatrixSpec(source=source, key=key, limit=limit, offset=offset)


def _coerce_float_matrix(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {matrix.shape}")
    return matrix


def _slice_rows(matrix: np.ndarray, spec: MatrixSpec) -> np.ndarray:
    start = spec.offset
    stop = None if spec.limit is None else start + spec.limit
    return matrix[start:stop]


def _load_local(spec: MatrixSpec) -> np.ndarray:
    path = Path(spec.source)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _slice_rows(_coerce_float_matrix(np.load(path)), spec)
    if suffix == ".npz":
        archive = np.load(path)
        if spec.key is None:
            if len(archive.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; provide --db-key/--query-key")
            key = archive.files[0]
        else:
            key = spec.key
        return _slice_rows(_coerce_float_matrix(archive[key]), spec)
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("h5py is required for .h5/.hdf5 sources") from exc
        key = spec.key or "train"
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise ValueError(f"{path} does not contain key {key!r}")
            matrix = _coerce_float_matrix(handle[key][...])
        return _slice_rows(matrix, spec)
    raise ValueError(f"unsupported local matrix format: {path}")


def _load_hf(spec: MatrixSpec) -> np.ndarray:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required for hf:// sources") from exc
    if not spec.column:
        raise ValueError("hf:// sources require a column name via ::column or --db-column/--query-column")
    dataset = load_dataset(spec.source, split=spec.split)
    if spec.offset:
        dataset = dataset.select(range(spec.offset, len(dataset)))
    if spec.limit is not None:
        dataset = dataset.select(range(min(spec.limit, len(dataset))))
    values = dataset[spec.column]
    return _coerce_float_matrix(np.asarray(values, dtype=np.float32))


def load_matrix(
    source: str,
    *,
    key: str | None = None,
    split: str | None = None,
    column: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> np.ndarray:
    spec = parse_source_spec(
        source,
        key=key,
        split=split,
        column=column,
        limit=limit,
        offset=offset,
    )
    if source.startswith("hf://"):
        return _load_hf(spec)
    return _load_local(spec)
