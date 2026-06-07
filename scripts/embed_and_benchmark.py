from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from turboquant import TurboQuantProd, brute_force_inner_products, faiss_available, faiss_pq_topk
from turboquant.reports import plot_nn_report, write_json
from turboquant.search import recall_at_k, topk_indices


DEFAULT_DOCS = [
    "TurboQuant compresses high-dimensional vectors with randomized rotation and scalar quantization.",
    "Nearest-neighbor retrieval uses inner products between query embeddings and database embeddings.",
    "Residual quantization with a one-bit QJL stage reduces bias in inner-product estimation.",
    "Product quantization is a common baseline for approximate nearest-neighbor search.",
    "Sentence embeddings map text into dense vectors that preserve semantic similarity.",
    "Smaller embedding models are useful for quick local experiments before scaling up to larger datasets.",
    "Random rotations can make coordinate distributions easier to quantize uniformly across inputs.",
    "Vector databases often trade indexing speed, memory footprint, and recall against each other.",
]

DEFAULT_QUERIES = [
    "How does TurboQuant reduce vector compression error?",
    "What baseline is commonly used in vector search?",
    "Why would I use a small embedding model for testing?",
]


def _load_lines(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]


def _encode(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def run_demo(args: argparse.Namespace) -> dict[str, object]:
    docs = _load_lines(args.docs_file, DEFAULT_DOCS)
    queries = _load_lines(args.queries_file, DEFAULT_QUERIES)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    database = _encode(args.model, docs, args.batch_size)
    queries_matrix = _encode(args.model, queries, args.batch_size)
    t1 = time.perf_counter()

    np.save(outdir / "database.npy", database)
    np.save(outdir / "queries.npy", queries_matrix)
    (outdir / "documents.txt").write_text("\n".join(docs) + "\n")
    (outdir / "queries.txt").write_text("\n".join(queries) + "\n")

    gt_scores = brute_force_inner_products(queries_matrix, database)
    report: dict[str, object] = {
        "model": args.model,
        "embedding_seconds": t1 - t0,
        "database_shape": list(database.shape),
        "query_shape": list(queries_matrix.shape),
        "documents_count": len(docs),
        "queries_count": len(queries),
        "documents_preview": docs[: min(10, len(docs))],
        "queries_preview": queries[: min(10, len(queries))],
        "results": [],
    }

    for bits in args.bits:
        q0 = time.perf_counter()
        quantizer = TurboQuantProd(database.shape[1], bits, args.seed, codebook_dir=args.codebook_dir)
        codes = quantizer.quantize(database)
        q1 = time.perf_counter()
        approx_scores = quantizer.score_inner_product(queries_matrix, codes)
        q2 = time.perf_counter()

        result = {
            "bits": bits,
            "storage": {
                "turboquant_bytes": int(
                    codes.indices.byte_count
                    + codes.qjl_signs.byte_count
                    + codes.residual_norms.nbytes
                    + (0 if codes.norms is None else codes.norms.nbytes)
                )
            },
            "methods": {"turboquant": {}},
            "timings": {
                "turboquant": {
                    "quantize_seconds": q1 - q0,
                    "score_seconds": q2 - q1,
                }
            },
        }
        for k in args.topk:
            gt_top = topk_indices(gt_scores, k)
            approx_top = topk_indices(approx_scores, k)
            result["methods"]["turboquant"][str(k)] = recall_at_k(gt_top[:, :1], approx_top)

        if faiss_available():
            try:
                p0 = time.perf_counter()
                pq_top = faiss_pq_topk(database, queries_matrix, bits=bits, topk=max(args.topk))
                p1 = time.perf_counter()
                result["methods"]["pq"] = {}
                result["timings"]["pq"] = {
                    "quantize_seconds": p1 - p0,
                    "score_seconds": 0.0,
                }
                for k in args.topk:
                    gt_top = topk_indices(gt_scores, k)
                    result["methods"]["pq"][str(k)] = recall_at_k(gt_top[:, :1], pq_top[:, :k])
            except ValueError as exc:
                result["methods"]["pq"] = {"skipped": str(exc)}
                result["timings"]["pq"] = {
                    "quantize_seconds": 0.0,
                    "score_seconds": 0.0,
                }

        report["results"].append(result)

    write_json(outdir / "report.json", report)
    plot_nn_report(report, outdir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed text and benchmark TurboQuant")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--docs-file")
    parser.add_argument("--queries-file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--codebook-dir", default="artifacts/codebooks")
    parser.add_argument("--output", default="artifacts/demo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_demo(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
