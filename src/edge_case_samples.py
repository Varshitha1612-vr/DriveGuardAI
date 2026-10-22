from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.edge_case_monitor import inspect_frame


FRAME_WARNINGS = (
    "Potentially low-light frame",
    "Potentially blurry frame",
)


def _read_labeled_samples(
    train_csv: Path,
    samples_per_class: int,
) -> list[dict[str, Any]]:
    with train_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"id", "target", "time_of_event"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{train_csv} must contain columns: "
                f"{', '.join(sorted(required_columns))}"
            )
        records = list(reader)

    selected = []
    for target in (0, 1):
        class_records = sorted(
            (record for record in records if record["target"] == str(target)),
            key=lambda record: record["id"],
        )
        if len(class_records) < samples_per_class:
            raise ValueError(
                f"Expected at least {samples_per_class} training videos "
                f"with target={target}, found {len(class_records)}."
            )
        selected.extend(class_records[:samples_per_class])

    return selected


def _extract_frame(video_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open dataset video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid video metadata in dataset video: {video_path}")

        target = int(record["target"])
        duration = frame_count / fps
        if target == 1:
            try:
                timestamp = float(record["time_of_event"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Positive sample {video_path.stem} has no valid time_of_event."
                ) from exc
            if not math.isfinite(timestamp):
                raise ValueError(
                    f"Positive sample {video_path.stem} has no valid time_of_event."
                )
        else:
            timestamp = duration / 2

        frame_index = min(max(round(timestamp * fps), 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(
                f"Could not read frame {frame_index} from dataset video: {video_path}"
            )

        return {
            "frame": frame,
            "frame_index": frame_index,
            "timestamp_seconds": round(frame_index / fps, 3),
            "target": target,
        }
    finally:
        capture.release()


def _make_contact_sheet(samples: list[dict[str, Any]]) -> np.ndarray:
    card_width = 700
    card_height = 460
    columns = 2
    rows = math.ceil(len(samples) / columns)
    sheet = np.full(
        (52 + rows * card_height, columns * card_width + 24, 3),
        (24, 27, 33),
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        "DriveGuard AI | Frames extracted from the training dataset",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    for index, sample in enumerate(samples):
        x = 8 + (index % columns) * card_width
        y = 48 + (index // columns) * card_height
        cv2.rectangle(
            sheet,
            (x, y),
            (x + card_width - 10, y + card_height - 10),
            (38, 42, 50),
            -1,
        )
        title = (
            f'{sample["video_id"]}.mp4 | '
            f'{"event" if sample["target"] else "no event"} | '
            f'{sample["timestamp_seconds"]:.2f}s'
        )
        cv2.putText(
            sheet,
            title,
            (x + 16, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

        preview = cv2.resize(
            sample["frame"],
            (660, 371),
            interpolation=cv2.INTER_AREA,
        )
        sheet[y + 40:y + 411, x + 10:x + 670] = preview

        warnings = sample["warnings"]
        alert = "Warnings: " + ("; ".join(warnings) if warnings else "None")
        cv2.putText(
            sheet,
            alert[:85],
            (x + 16, y + 438),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 180, 255) if warnings else (100, 210, 120),
            1,
            cv2.LINE_AA,
        )

    return sheet


def generate_dataset_samples(
    project_root: str | Path = ".",
    output_dir: str | Path = "results/edge_case_dataset",
    samples_per_class: int = 2,
) -> dict[str, Any]:
    """Extract unmodified frames from labeled training videos and report monitor output."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")

    root = Path(project_root).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    records = _read_labeled_samples(root / "data" / "train.csv", samples_per_class)
    samples = []
    for record in records:
        video_id = record["id"]
        video_path = root / "data" / "train" / f"{video_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Dataset video not found: {video_path}")

        extracted = _extract_frame(video_path, record)
        quality = inspect_frame(extracted["frame"])
        frame_name = f'{video_id}_frame_{extracted["frame_index"]:06d}.png'
        frame_path = frames_dir / frame_name
        if not cv2.imwrite(str(frame_path), extracted["frame"]):
            raise OSError(f"Could not save extracted dataset frame: {frame_path}")

        samples.append({
            "video_id": video_id,
            "target": extracted["target"],
            "frame_index": extracted["frame_index"],
            "timestamp_seconds": extracted["timestamp_seconds"],
            "brightness": quality["brightness"],
            "blur_score": quality["blur_score"],
            "warnings": quality["warnings"],
            "frame_path": str(frame_path.relative_to(destination)),
            "frame": extracted["frame"],
        })

    sheet_path = destination / "edge_case_dataset_contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), _make_contact_sheet(samples)):
        raise OSError(f"Could not write dataset contact sheet: {sheet_path}")

    csv_path = destination / "sample_results.csv"
    fields = [
        "video_id",
        "target",
        "frame_index",
        "timestamp_seconds",
        "brightness",
        "blur_score",
        "warnings",
        "frame_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                **{key: sample[key] for key in fields if key != "warnings"},
                "warnings": "; ".join(sample["warnings"]),
            })

    warning_counts = {
        warning: sum(warning in sample["warnings"] for sample in samples)
        for warning in FRAME_WARNINGS
    }
    summary = {
        "source": "data/train.csv and original decoded frames from data/train/",
        "sample_count": len(samples),
        "samples_per_class": samples_per_class,
        "warning_counts": warning_counts,
        "frames_with_warnings": sum(bool(sample["warnings"]) for sample in samples),
        "accuracy_percent": None,
        "accuracy_note": (
            "Not calculated: the dataset provides video-level event labels, "
            "not frame-level ground truth for image-quality warnings or object boxes."
        ),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_dataset_samples(), indent=2))
