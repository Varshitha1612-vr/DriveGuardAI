from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_label_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load the training CSV while preserving sample IDs as strings."""
    frame = pd.read_csv(csv_path, dtype={"id": "string"})
    frame["id"] = frame["id"].astype(str).str.strip()
    return frame


def timestamp_presence_baseline(frame: pd.DataFrame) -> pd.Series:
    """Baseline intentionally uses only label metadata.

    This is not a learned video classifier. It is a sanity-check baseline demonstrating
    how the dataset encodes event/alert presence for positive and negative samples.
    """
    present = (
        pd.to_numeric(frame["time_of_event"], errors="coerce").notna()
        | pd.to_numeric(frame["time_of_alert"], errors="coerce").notna()
    )
    return present.astype(int)


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute standard binary classification metrics for a positive/negative target."""
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    balanced_accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2.0 if (tp + fn) and (tn + fp) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_baseline(csv_path: str | Path) -> dict[str, Any]:
    """Run a naive metadata-driven baseline and evaluate it on the training labels."""
    frame = load_label_frame(csv_path)
    y_true = frame["target"].astype(int)
    y_pred = timestamp_presence_baseline(frame)
    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "baseline_name": "timestamp_presence_baseline",
        "train_rows": int(len(frame)),
        "positive_samples": int((y_true == 1).sum()),
        "negative_samples": int((y_true == 0).sum()),
        "metrics": metrics,
        "notes": [
            "This baseline is intentionally simple and uses only whether event/alert timestamps are present.",
            "It is not a real video classifier; it is a sanity-check for dataset semantics and label validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple collision-label baseline for the Nexar dataset.")
    parser.add_argument("--csv", default="data/train.csv", help="Training CSV file to evaluate.")
    parser.add_argument("--output", default="results/collision_evaluation.json", help="Output JSON path.")
    args = parser.parse_args()

    result = evaluate_baseline(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
