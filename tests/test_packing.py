import numpy as np

from turboquant.packing import pack_indices, pack_signs, unpack_indices, unpack_signs


def test_index_pack_roundtrip() -> None:
    values = np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.int32)
    packed, _ = pack_indices(values, 2)
    unpacked = unpack_indices(packed, 2, len(values))
    assert np.array_equal(values, unpacked)


def test_sign_pack_roundtrip() -> None:
    values = np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    packed, _ = pack_signs(values)
    unpacked = unpack_signs(packed, len(values))
    assert np.array_equal(values, unpacked)
