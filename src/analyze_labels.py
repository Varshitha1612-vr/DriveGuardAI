from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_training_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the Nexar training CSV while preserving ID values as strings."""
    path = Path(csv_path)
    frame = pd.read_csv(path, dtype={"id": "string"})
    frame["id"] = frame["id"].astype(str).str.strip()
    return frame


def validate_timestamp_fields(frame: pd.DataFrame) -> dict[str, Any]:
    """Check the numeric semantics of event/alert timestamps.

    The dataset stores event and alert times in seconds relative to each video clip.
    Missing values are represented as NaN. This is not a real-world clock timestamp;
    it is a per-video time offset used for evaluation, so we preserve the raw values
    and only validate that they are numeric and non-negative when present.
    """
    event_series = pd.to_numeric(frame["time_of_event"], errors="coerce")
    alert_series = pd.to_numeric(frame["time_of_alert"], errors="coerce")

    positives = frame["target"].astype(int)
    event_valid = event_series.notna()
    alert_valid = alert_series.notna()
    invalid_event = int(((frame["time_of_event"].notna()) & event_series.isna()).sum())
    invalid_alert = int(((frame["time_of_alert"].notna()) & alert_series.isna()).sum())
    negative_times = int((event_series[frame["time_of_event"].notna()] < 0).sum()) + int((alert_series[frame["time_of_alert"].notna()] < 0).sum())

    return {
        "timestamp_units": "seconds relative to each video clip",
        "missing_values": {"time_of_event": int(frame["time_of_event"].isna().sum()), "time_of_alert": int(frame["time_of_alert"].isna().sum())},
        "invalid_numeric_rows": {"time_of_event": invalid_event, "time_of_alert": invalid_alert},
        "negative_time_values": negative_times,
        "matches_positive_label_pattern": bool((positives == 1).sum() == int((event_valid | alert_valid).sum())),
        "event_values_present": int(event_valid.sum()),
        "alert_values_present": int(alert_valid.sum()),
    }


def summarize_labels(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize label balance and timestamp distribution for the training data."""
    summary = {
        "rows": int(len(frame)),
        "positive_samples": int((frame["target"] == 1).sum()),
        "negative_samples": int((frame["target"] == 0).sum()),
        "missing_target_values": int(frame["target"].isna().sum()),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "id_count_unique": int(frame["id"].nunique()),
        "timestamp_analysis": validate_timestamp_fields(frame),
    }
    return summary


def build_dataset_summary(csv_path: str | Path) -> dict[str, Any]:
    """Create a friendly summary of the Nexar label semantics and validity."""
    frame = load_training_csv(csv_path)
    summary = summarize_labels(frame)
    summary["label_distribution"] = {
        "target_0": int((frame["target"] == 0).sum()),
        "target_1": int((frame["target"] == 1).sum()),
    }
    summary["notes"] = [
        "The target label is preserved as supplied by the dataset.",
        "The event and alert columns are interpreted as time offsets in seconds relative to a video clip, with NaN when unavailable.",
        "This is not a bounding-box annotation file, so no object-level ground truth is fabricated.",
    ]
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the Nexar collision-label CSV for semantic validity.")
    parser.add_argument("--csv", default="data/train.csv", help="Training CSV path.")
    parser.add_argument("--output", default="results/label_summary.json", help="Output JSON file path.")
    args = parser.parse_args()

    summary = build_dataset_summary(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "positive_samples": summary["positive_samples"], "negative_samples": summary["negative_samples"]}, indent=2))


if __name__ == "__main__":
    main()
