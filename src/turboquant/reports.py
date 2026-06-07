from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return target


def plot_nn_report(report: dict[str, object], output_dir: str | Path) -> Path | None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "turboquant-mpl"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_recall, ax_time = axes
    results = report.get("results", [])
    topk = sorted(
        {
            int(k)
            for item in results
            for method in item["methods"].values()
            for k in method.keys()
            if str(k).isdigit()
        }
    )
    for method_name in sorted({name for item in results for name in item["methods"].keys()}):
        xs = [item["bits"] for item in results]
        ys = []
        for item in results:
            method = item["methods"].get(method_name, {})
            ys.append(method.get(str(topk[0]), float("nan")) if topk else float("nan"))
        ax_recall.plot(xs, ys, marker="o", label=method_name)
    ax_recall.set_xlabel("bits")
    ax_recall.set_ylabel(f"recall@1 in top-{topk[0]}" if topk else "recall")
    ax_recall.set_title("Recall vs Bitwidth")
    ax_recall.legend()
    for method_name in sorted({name for item in results for name in item["timings"].keys()}):
        xs = [item["bits"] for item in results]
        ys = [item["timings"][method_name]["quantize_seconds"] for item in results]
        ax_time.plot(xs, ys, marker="o", label=method_name)
    ax_time.set_xlabel("bits")
    ax_time.set_ylabel("quantize seconds")
    ax_time.set_title("Indexing Time")
    ax_time.legend()
    figure.tight_layout()
    output = outdir / "nn_report.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
