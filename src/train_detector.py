from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_label_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load the Nexar CSV with IDs preserved as strings.

    This is intentionally a training-orchestration scaffold. We do not directly train
    a detector on the collision labels, and we do not fabricate object boxes. The
    module is prepared for later detector training once a proper dataset split and
    video-level pipeline are defined.
    """
    path = Path(csv_path)
    frame = pd.read_csv(path, dtype={"id": "string"})
    frame["id"] = frame["id"].astype(str).str.strip()
    return frame


def prepare_training_plan(csv_path: str | Path) -> dict[str, Any]:
    """Create a documented training plan and validation split structure."""
    frame = load_label_frame(csv_path)
    sample_count = len(frame)
    positive_count = int((frame["target"] == 1).sum())
    negative_count = int((frame["target"] == 0).sum())
    return {
        "status": "not_implemented",
        "csv_path": str(csv_path),
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "notes": [
            "Keep train/validation/test splits at the video level, not the frame level.",
            "Never train a YOLO detector directly on collision labels.",
            "Use the detection model for object localization and a separate classifier for video-level risk estimation when needed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the training plan for a future detector/classifier pipeline."
    )
    parser.add_argument("--csv", default="data/train.csv", help="Training CSV used for planned model preparation.")
    parser.add_argument("--output", default="results/training_plan.json", help="Output JSON summary.")
    args = parser.parse_args()

    result = prepare_training_plan(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}, indent=2))


if __name__ == "__main__":
    main()
