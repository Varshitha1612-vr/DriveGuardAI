from __future__ import annotations

from pathlib import Path

from src.inspect_dataset import inspect_dataset


def test_dataset_inspection_counts_and_outputs(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "results"

    summary = inspect_dataset(project_root, output_dir=output_dir)
    dataset_summary = summary["dataset_summary"]

    assert dataset_summary["train_rows"] == 1500
    assert dataset_summary["test_rows"] == 1344
    assert dataset_summary["positive_samples"] == 750
    assert dataset_summary["negative_samples"] == 750
    assert dataset_summary["missing_video_matches"] == []
    assert dataset_summary["missing_csv_matches"] == []
    assert (output_dir / "dataset_report.json").exists()
    assert (output_dir / "dataset_report.txt").exists()
