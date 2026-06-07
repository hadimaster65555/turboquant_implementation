from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd, brute_force_inner_products, faiss_available, faiss_pq_topk
from turboquant.reports import plot_nn_report, write_json
from turboquant.search import recall_at_k, topk_indices


def _load_lines(path: str) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return summed / counts


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.bfloat16
    return torch.float16


def _encode_causal_lm(
    model_name: str,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: str,
    local_files_only: bool,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=_pick_dtype(device),
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            pooled = F.normalize(pooled.float(), dim=1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    docs = _load_lines(args.docs_file)[: args.max_docs]
    queries = _load_lines(args.queries_file)[: args.max_queries]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.perf_counter()
    database = _encode_causal_lm(
        args.model,
        docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        local_files_only=args.local_files_only,
    )
    queries_matrix = _encode_causal_lm(
        args.model,
        queries,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        local_files_only=args.local_files_only,
    )
    t1 = time.perf_counter()

    np.save(outdir / "database.npy", database)
    np.save(outdir / "queries.npy", queries_matrix)
    (outdir / "documents.txt").write_text("\n".join(docs) + "\n")
    (outdir / "queries.txt").write_text("\n".join(queries) + "\n")

    gt_scores = brute_force_inner_products(queries_matrix, database)
    report: dict[str, object] = {
        "model": args.model,
        "device": device,
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
                result["timings"]["pq"] = {"quantize_seconds": 0.0, "score_seconds": 0.0}
        report["results"].append(result)

    write_json(outdir / "report.json", report)
    plot_nn_report(report, outdir)
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant with Mistral-7B hidden-state embeddings")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--docs-file", required=True)
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=128)
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--codebook-dir", default="artifacts/codebooks")
    parser.add_argument("--output", default="artifacts/demo_mistral")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
