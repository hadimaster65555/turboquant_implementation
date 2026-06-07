from pathlib import Path

from turboquant.reports import plot_nn_report


def test_plot_nn_report(tmp_path: Path) -> None:
    report = {
        "results": [
            {
                "bits": 2,
                "methods": {"turboquant": {"1": 0.9}, "pq": {"1": 0.8}},
                "timings": {
                    "turboquant": {"quantize_seconds": 0.1, "score_seconds": 0.02},
                    "pq": {"quantize_seconds": 0.3, "score_seconds": 0.0},
                },
            },
            {
                "bits": 4,
                "methods": {"turboquant": {"1": 0.95}, "pq": {"1": 0.9}},
                "timings": {
                    "turboquant": {"quantize_seconds": 0.12, "score_seconds": 0.03},
                    "pq": {"quantize_seconds": 0.35, "score_seconds": 0.0},
                },
            },
        ]
    }
    output = plot_nn_report(report, tmp_path)
    assert output is not None
    assert output.exists()
