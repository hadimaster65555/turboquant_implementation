from __future__ import annotations

import numpy as np


def gaussian_orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dim, dim), dtype=np.float64)
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def gaussian_projection_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dim, dim), dtype=np.float64)
