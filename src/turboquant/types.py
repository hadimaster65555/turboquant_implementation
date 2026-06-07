from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedArray:
    data: np.ndarray
    bit_count: int
    item_count: int

    @property
    def byte_count(self) -> int:
        return int(self.data.size)
