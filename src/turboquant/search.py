from __future__ import annotations

import numpy as np

from .quantizers import QuantizedProd, TurboQuantProd


def brute_force_inner_products(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    return np.asarray(queries, dtype=np.float64) @ np.asarray(database, dtype=np.float64).T


def score_queries_against_prod(
    quantizer: TurboQuantProd,
    queries: np.ndarray,
    codes: QuantizedProd,
) -> np.ndarray:
    return quantizer.score_inner_product(queries, codes)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.shape[1])
    partial = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    partial_scores = np.take_along_axis(scores, partial, axis=1)
    order = np.argsort(-partial_scores, axis=1)
    return np.take_along_axis(partial, order, axis=1)


def recall_at_k(ground_truth: np.ndarray, approx: np.ndarray) -> float:
    hits = 0
    total = ground_truth.shape[0]
    for gt_row, approx_row in zip(ground_truth, approx, strict=True):
        if np.intersect1d(gt_row, approx_row).size > 0:
            hits += 1
    return hits / max(total, 1)


def recall_one_in_topk(scores: np.ndarray, approx: np.ndarray, k: int) -> float:
    gt = topk_indices(scores, 1)
    approx_top = topk_indices(approx, k)
    return recall_at_k(gt, approx_top)
