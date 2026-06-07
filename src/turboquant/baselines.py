from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def faiss_available() -> bool:
    return faiss is not None


def faiss_pq_topk(
    database: np.ndarray,
    queries: np.ndarray,
    *,
    bits: int,
    topk: int,
) -> np.ndarray:
    if faiss is None:
        raise RuntimeError("faiss is not installed")
    db = np.asarray(database, dtype=np.float32, order="C")
    q = np.asarray(queries, dtype=np.float32, order="C")
    required = 2**bits
    if db.shape[0] < required:
        raise ValueError(
            f"PQ requires at least {required} training vectors for {bits}-bit codebooks; got {db.shape[0]}"
        )
    dim = db.shape[1]
    index = faiss.IndexPQ(dim, dim, bits, faiss.METRIC_INNER_PRODUCT)
    index.train(db)
    index.add(db)
    _, indices = index.search(q, topk)
    return indices.astype(np.int64)
