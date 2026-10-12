from __future__ import annotations

from pathlib import Path

from src.evaluate_collision import evaluate_baseline


def test_collision_baseline_metrics_are_valid():
    project_root = Path(__file__).resolve().parents[1]
    result = evaluate_baseline(project_root / "data" / "train.csv")

    assert result["train_rows"] == 1500
    assert result["positive_samples"] == 750
    assert result["negative_samples"] == 750
    assert result["metrics"]["precision"] >= 0.0
    assert result["metrics"]["recall"] >= 0.0
    assert result["metrics"]["f1_score"] >= 0.0
    assert result["baseline_name"] == "timestamp_presence_baseline"
