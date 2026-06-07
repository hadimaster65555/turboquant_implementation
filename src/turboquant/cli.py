from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .baselines import faiss_available, faiss_pq_topk
from .codebooks import CodebookRepository
from .data import load_matrix
from .quantizers import TurboQuantMSE, TurboQuantProd
from .reports import plot_nn_report, write_json
from .search import brute_force_inner_products, recall_at_k, topk_indices


def _cmd_precompute(args: argparse.Namespace) -> int:
    repo = CodebookRepository(args.output)
    for bits in args.bits:
        codebook = repo.get_or_create(args.dim, bits, resolution=args.resolution)
        print(f"saved d={args.dim} b={bits} mse_cost={codebook.mse_cost:.8f}")
    return 0


def _random_sphere(rng: np.random.Generator, samples: int, dim: int) -> np.ndarray:
    raw = rng.standard_normal((samples, dim))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def _cmd_validate(args: argparse.Namespace) -> int:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vectors = _random_sphere(rng, args.samples, args.dim)
    report: dict[str, object] = {"dim": args.dim, "samples": args.samples, "results": []}
    for bits in args.bits:
        mse_quantizer = TurboQuantMSE(args.dim, bits, args.seed, codebook_dir=args.codebook_dir)
        mse_codes = mse_quantizer.quantize(vectors)
        mse_decoded = mse_quantizer.dequantize(mse_codes)
        mse = float(np.mean(np.sum(np.square(vectors - mse_decoded), axis=1)))

        prod_quantizer = TurboQuantProd(args.dim, bits, args.seed, codebook_dir=args.codebook_dir)
        prod_codes = prod_quantizer.quantize(vectors)
        prod_decoded = prod_quantizer.dequantize(prod_codes)
        y = _random_sphere(rng, args.samples, args.dim)
        gt = np.sum(y * vectors, axis=1)
        est = np.sum(y * prod_decoded, axis=1)
        prod_error = float(np.mean(np.square(gt - est)))
        bias = float(np.mean(est - gt))
        report["results"].append(
            {
                "bits": bits,
                "mse": mse,
                "prod_error": prod_error,
                "prod_bias": bias,
            }
        )
    (outdir / "distortion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def _cmd_benchmark_nn(args: argparse.Namespace) -> int:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    database = load_matrix(
        args.db_path,
        key=args.db_key,
        split=args.db_split,
        column=args.db_column,
        limit=args.db_limit,
        offset=args.db_offset,
    )
    queries = load_matrix(
        args.query_path,
        key=args.query_key,
        split=args.query_split,
        column=args.query_column,
        limit=args.query_limit,
        offset=args.query_offset,
    )
    gt_scores = brute_force_inner_products(queries, database)
    report: dict[str, object] = {
        "database_shape": list(database.shape),
        "query_shape": list(queries.shape),
        "results": [],
    }
    for bits in args.bits:
        turbo_t0 = time.perf_counter()
        quantizer = TurboQuantProd(database.shape[1], bits, args.seed, codebook_dir=args.codebook_dir)
        codes = quantizer.quantize(database)
        turbo_t1 = time.perf_counter()
        approx_scores = quantizer.score_inner_product(queries, codes)
        turbo_t2 = time.perf_counter()
        bit_report = {
            "bits": bits,
            "storage": {
                "turboquant_bytes": int(
                    codes.indices.byte_count
                    + codes.qjl_signs.byte_count
                    + codes.residual_norms.nbytes
                    + (0 if codes.norms is None else codes.norms.nbytes)
                ),
            },
            "methods": {"turboquant": {}},
            "timings": {
                "turboquant": {
                    "quantize_seconds": turbo_t1 - turbo_t0,
                    "score_seconds": turbo_t2 - turbo_t1,
                }
            },
        }
        for k in args.topk:
            gt_top = topk_indices(gt_scores, k)
            approx_top = topk_indices(approx_scores, k)
            bit_report["methods"]["turboquant"][str(k)] = recall_at_k(gt_top[:, :1], approx_top)
        if faiss_available():
            try:
                pq_t0 = time.perf_counter()
                pq_top = faiss_pq_topk(database, queries, bits=bits, topk=max(args.topk))
                pq_t1 = time.perf_counter()
                bit_report["methods"]["pq"] = {}
                bit_report["timings"]["pq"] = {
                    "quantize_seconds": pq_t1 - pq_t0,
                    "score_seconds": 0.0,
                }
                for k in args.topk:
                    gt_top = topk_indices(gt_scores, k)
                    bit_report["methods"]["pq"][str(k)] = recall_at_k(gt_top[:, :1], pq_top[:, :k])
            except ValueError as exc:
                bit_report["methods"]["pq"] = {"skipped": str(exc)}
                bit_report["timings"]["pq"] = {
                    "quantize_seconds": 0.0,
                    "score_seconds": 0.0,
                }
        report["results"].append(bit_report)
    write_json(outdir / "nn_report.json", report)
    if args.plot:
        plot_nn_report(report, outdir)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_prepare_synthetic(args: argparse.Namespace) -> int:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    database = rng.standard_normal((args.db_size, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.query_size, args.dim), dtype=np.float32)
    if args.normalize:
        database = database / np.linalg.norm(database, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    np.save(outdir / "database.npy", database)
    np.save(outdir / "queries.npy", queries)
    manifest = {
        "database_path": str(outdir / "database.npy"),
        "queries_path": str(outdir / "queries.npy"),
        "database_shape": list(database.shape),
        "query_shape": list(queries.shape),
        "normalized": bool(args.normalize),
        "seed": args.seed,
    }
    write_json(outdir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurboQuant CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)

    precompute = subparsers.add_parser("precompute-codebooks")
    precompute.add_argument("--dim", type=int, required=True)
    precompute.add_argument("--bits", type=int, nargs="+", required=True)
    precompute.add_argument("--resolution", type=int, default=32769)
    precompute.add_argument("--output", default="artifacts/codebooks")
    precompute.set_defaults(func=_cmd_precompute)

    validate = subparsers.add_parser("validate-distortion")
    validate.add_argument("--dim", type=int, required=True)
    validate.add_argument("--bits", type=int, nargs="+", required=True)
    validate.add_argument("--samples", type=int, default=2048)
    validate.add_argument("--seed", type=int, default=7)
    validate.add_argument("--codebook-dir", default="artifacts/codebooks")
    validate.add_argument("--output", default="artifacts/validation")
    validate.set_defaults(func=_cmd_validate)

    nn = subparsers.add_parser("benchmark-nn")
    nn.add_argument("--db-path", required=True)
    nn.add_argument("--query-path", required=True)
    nn.add_argument("--db-key")
    nn.add_argument("--query-key")
    nn.add_argument("--db-split")
    nn.add_argument("--query-split")
    nn.add_argument("--db-column")
    nn.add_argument("--query-column")
    nn.add_argument("--db-limit", type=int)
    nn.add_argument("--query-limit", type=int)
    nn.add_argument("--db-offset", type=int, default=0)
    nn.add_argument("--query-offset", type=int, default=0)
    nn.add_argument("--bits", type=int, nargs="+", required=True)
    nn.add_argument("--topk", type=int, nargs="+", default=[1, 4, 8, 16])
    nn.add_argument("--seed", type=int, default=7)
    nn.add_argument("--codebook-dir", default="artifacts/codebooks")
    nn.add_argument("--output", default="artifacts/nn")
    nn.add_argument("--plot", action="store_true")
    nn.set_defaults(func=_cmd_benchmark_nn)

    synthetic = subparsers.add_parser("prepare-synthetic")
    synthetic.add_argument("--dim", type=int, required=True)
    synthetic.add_argument("--db-size", type=int, default=100000)
    synthetic.add_argument("--query-size", type=int, default=1000)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--normalize", action="store_true")
    synthetic.add_argument("--output", default="artifacts/data/synthetic")
    synthetic.set_defaults(func=_cmd_prepare_synthetic)

    return parser


def main() -> int:
    parser = build_parser()
    argv0 = Path(sys.argv[0]).name
    if argv0 in {"precompute-codebooks", "validate-distortion", "benchmark-nn", "prepare-synthetic"}:
        args = parser.parse_args([argv0, *sys.argv[1:]])
    else:
        args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
