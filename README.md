# TurboQuant

Reference implementation of the TurboQuant paper:

- `TurboQuantMSE` for MSE-oriented reconstruction
- `TurboQuantProd` for unbiased inner-product estimation via residual QJL

The implementation is intentionally explicit and research-friendly. It uses a shared random rotation,
precomputed scalar codebooks for Beta-distributed coordinates on the sphere, and packed compressed
representations for centroid indices and QJL sign bits.

## Install

```bash
python -m pip install -e .
python -m pip install -e ".[dev,bench]"
```

## CLI

```bash
precompute-codebooks --dim 1536 --bits 1 2 3 4 --output artifacts/codebooks
validate-distortion --dim 1536 --bits 1 2 3 4 --samples 2048 --output artifacts/validation
benchmark-nn --db-path data/db.npy --query-path data/query.npy --bits 2 4 --output artifacts/nn
prepare-synthetic --dim 1536 --db-size 100000 --query-size 1000 --normalize --output artifacts/data/synth
```

`benchmark-nn` also accepts:

- local `.npz` inputs with `--db-key` / `--query-key`
- local `.h5` / `.hdf5` inputs with `--db-key` / `--query-key` (for ANN-Benchmarks-style files)
- Hugging Face datasets via `hf://dataset-name::column` plus `--db-split` / `--query-split`
- `--plot` to emit a PNG summary beside the JSON report

## Embedding Demo

Install embedding dependencies:

```bash
python -m pip install -e ".[embed]"
```

Run the included `all-MiniLM-L6-v2` demo:

```bash
./.venv/bin/python scripts/embed_and_benchmark.py --bits 2 4 --output artifacts/demo
```

This will:

- download `sentence-transformers/all-MiniLM-L6-v2`
- embed a small document/query set
- save `database.npy` and `queries.npy`
- run TurboQuant and PQ
- write `report.json` and `report.png`
