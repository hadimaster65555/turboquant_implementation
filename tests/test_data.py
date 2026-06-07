from pathlib import Path

import numpy as np

from turboquant.data import load_matrix, parse_source_spec


def test_parse_hf_source_spec() -> None:
    spec = parse_source_spec("hf://org/dataset::embedding", split="train[:100]")
    assert spec.source == "org/dataset"
    assert spec.column == "embedding"
    assert spec.split == "train[:100]"


def test_load_npy_matrix(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npy"
    array = np.arange(20, dtype=np.float32).reshape(5, 4)
    np.save(path, array)
    loaded = load_matrix(str(path), limit=3, offset=1)
    assert loaded.shape == (3, 4)
    assert np.array_equal(loaded, array[1:4])


def test_load_npz_matrix_with_key(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    target = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(path, db=target, other=np.zeros((2, 2), dtype=np.float32))
    loaded = load_matrix(str(path), key="db")
    assert np.array_equal(loaded, target)


def test_load_hdf5_matrix_with_key(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "vectors.h5"
    target = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=target)
        handle.create_dataset("test", data=np.zeros((2, 4), dtype=np.float32))
    loaded = load_matrix(str(path), key="train")
    assert np.array_equal(loaded, target)
