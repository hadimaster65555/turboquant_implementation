from __future__ import annotations

import math

import numpy as np


def _pack_unsigned(values: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    total_bits = values.size * bits
    output = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    bit_offset = 0
    mask = (1 << bits) - 1
    for value in values:
        raw = int(value) & mask
        byte_index = bit_offset // 8
        offset = bit_offset % 8
        accumulator = raw << offset
        output[byte_index] |= accumulator & 0xFF
        if byte_index + 1 < output.size:
            output[byte_index + 1] |= (accumulator >> 8) & 0xFF
        if bits + offset > 16 and byte_index + 2 < output.size:
            output[byte_index + 2] |= (accumulator >> 16) & 0xFF
        bit_offset += bits
    return output, total_bits


def _unpack_unsigned(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    output = np.zeros(count, dtype=np.uint64)
    mask = (1 << bits) - 1
    bit_offset = 0
    for idx in range(count):
        byte_index = bit_offset // 8
        offset = bit_offset % 8
        accumulator = int(packed[byte_index])
        if byte_index + 1 < packed.size:
            accumulator |= int(packed[byte_index + 1]) << 8
        if byte_index + 2 < packed.size:
            accumulator |= int(packed[byte_index + 2]) << 16
        output[idx] = (accumulator >> offset) & mask
        bit_offset += bits
    return output


def pack_indices(indices: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    return _pack_unsigned(indices, bits)


def unpack_indices(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    return _unpack_unsigned(packed, bits, count).astype(np.int32)


def pack_signs(signs: np.ndarray) -> tuple[np.ndarray, int]:
    bits = (np.asarray(signs) > 0).astype(np.uint8)
    return _pack_unsigned(bits, 1)


def unpack_signs(packed: np.ndarray, count: int) -> np.ndarray:
    bits = _unpack_unsigned(packed, 1, count).astype(np.int8)
    return np.where(bits > 0, 1.0, -1.0).astype(np.float32)


def bits_per_code(byte_count: int, items: int) -> float:
    return 8.0 * byte_count / max(items, 1)


def expected_storage_bytes(count: int, bits: int) -> int:
    return math.ceil(count * bits / 8.0)
