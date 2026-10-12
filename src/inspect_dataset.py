from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
site_packages = project_root / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))

import cv2
import pandas as pd

EXPECTED_TRAIN_COLUMNS = ["id", "time_of_event", "time_of_alert", "target"]
EXPECTED_TEST_COLUMNS = ["id"]


def normalize_id(value: Any) -> str:
    """Convert dataset IDs to strings while preserving leading zeros."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def read_label_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file with dataset IDs preserved as strings."""
    dtype = {"id": "string"} if "id" in pd.read_csv(csv_path, nrows=1).columns else None
    frame = pd.read_csv(csv_path, dtype=dtype)
    if "id" in frame.columns:
        frame["id"] = frame["id"].map(normalize_id)
    return frame


def inspect_video(video_path: Path) -> dict[str, Any]:
    """Read basic OpenCV metadata for a video and capture readable-file issues."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return {
            "path": str(video_path),
            "readable": False,
            "fps": None,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration_seconds": None,
            "issue": "Video file could not be opened by OpenCV.",
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 else None

    capture.release()
    return {
        "path": str(video_path),
        "readable": True,
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "issue": None,
    }


def inspect_dataset(
    project_root: str | Path,
    *,
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    train_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the Nexar dataset and write a JSON/TXT report for review."""
    root = Path(project_root).resolve()
    data_dir = root / "data"
    train_csv_path = Path(train_csv) if train_csv else data_dir / "train.csv"
    test_csv_path = Path(test_csv) if test_csv else data_dir / "test.csv"
    train_dir_path = Path(train_dir) if train_dir else data_dir / "train"
    test_dir_path = Path(test_dir) if test_dir else data_dir / "test"
    results_dir = Path(output_dir) if output_dir else root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    missing_files: list[Path] = []
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    train_df = read_label_csv(train_csv_path)
    test_df = read_label_csv(test_csv_path)

    issues: list[str] = []
    if list(train_df.columns) != EXPECTED_TRAIN_COLUMNS:
        issues.append(
            f"Unexpected train CSV columns: {list(train_df.columns)}; expected {EXPECTED_TRAIN_COLUMNS}."
        )
    if list(test_df.columns) != EXPECTED_TEST_COLUMNS:
        issues.append(
            f"Unexpected test CSV columns: {list(test_df.columns)}; expected {EXPECTED_TEST_COLUMNS}."
        )

    train_ids = train_df["id"].dropna().astype(str).tolist()
    test_ids = test_df["id"].dropna().astype(str).tolist()

    duplicate_train_ids = train_df["id"].duplicated(keep=False)
    duplicate_ids = sorted(train_df.loc[duplicate_train_ids, "id"].dropna().unique().tolist())
    if duplicate_ids:
        issues.append(f"Duplicate training IDs detected: {duplicate_ids}")

    missing_in_csv: list[str] = []
    missing_in_files: list[str] = []
    if train_dir_path.exists():
        train_videos = sorted(
            path.stem for path in train_dir_path.glob("*.mp4") if path.is_file()
        )
        csv_ids = set(train_ids)
        video_ids = set(train_videos)
        missing_in_csv = sorted(video_ids - csv_ids)
        missing_in_files = sorted(csv_ids - video_ids)
        if missing_in_csv:
            issues.append(
                f"Video files found without matching CSV IDs: {missing_in_csv[:10]}"
            )
        if missing_in_files:
            issues.append(
                f"CSV IDs without matching video files: {missing_in_files[:10]}"
            )
    else:
        issues.append(f"Training video directory not found: {train_dir_path}")

    if test_dir_path.exists():
        test_videos = sorted(
            path.stem for path in test_dir_path.glob("*.mp4") if path.is_file()
        )
    else:
        issues.append(f"Test video directory not found: {test_dir_path}")
        test_videos = []

    unreadable_videos: list[str] = []
    metadata_by_video: dict[str, dict[str, Any]] = {}
    for video_path in sorted(train_dir_path.glob("*.mp4")) + sorted(test_dir_path.glob("*.mp4")):
        info = inspect_video(video_path)
        metadata_by_video[video_path.stem] = info
        if not info["readable"]:
            unreadable_videos.append(video_path.name)
            issues.append(f"Unreadable video: {video_path.name}")

    train_df = train_df.copy()
    train_df["target_numeric"] = pd.to_numeric(train_df["target"], errors="coerce")
    train_df["time_of_event_numeric"] = pd.to_numeric(
        train_df["time_of_event"], errors="coerce"
    )
    train_df["time_of_alert_numeric"] = pd.to_numeric(
        train_df["time_of_alert"], errors="coerce"
    )

    invalid_timestamp_rows = []
    for index, row in train_df.iterrows():
        if pd.notna(row["time_of_event"]) and not pd.notna(
            pd.to_numeric(row["time_of_event"], errors="coerce")
        ):
            invalid_timestamp_rows.append({"index": int(index), "id": row["id"], "field": "time_of_event"})
        if pd.notna(row["time_of_alert"]) and not pd.notna(
            pd.to_numeric(row["time_of_alert"], errors="coerce")
        ):
            invalid_timestamp_rows.append({"index": int(index), "id": row["id"], "field": "time_of_alert"})

    if invalid_timestamp_rows:
        issues.append(
            "Rows with invalid timestamp values: "
            + ", ".join(f"{entry['id']} ({entry['field']})" for entry in invalid_timestamp_rows[:10])
        )

    positive_samples = int((train_df["target_numeric"] == 1).sum())
    negative_samples = int((train_df["target_numeric"] == 0).sum())
    missing_label_rows = []
    for index, row in train_df.iterrows():
        target_value = row["target_numeric"]
        if pd.isna(target_value):
            missing_label_rows.append({"index": int(index), "id": row["id"], "issue": "missing target label"})
            continue
        if target_value == 1 and (
            pd.isna(row["time_of_event_numeric"]) or pd.isna(row["time_of_alert_numeric"])
        ):
            missing_label_rows.append({
                "index": int(index),
                "id": row["id"],
                "issue": "positive target missing event/alert timestamps",
            })

    if missing_label_rows:
        issues.append(
            "Rows with missing labels or missing timestamps: "
            + ", ".join(f"{entry['id']} ({entry['issue']})" for entry in missing_label_rows[:10])
        )

    summary = {
        "project_root": str(root),
        "train_csv": str(train_csv_path),
        "test_csv": str(test_csv_path),
        "train_video_dir": str(train_dir_path),
        "test_video_dir": str(test_dir_path),
        "dataset_summary": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_videos": len(sorted(train_dir_path.glob("*.mp4"))) if train_dir_path.exists() else 0,
            "test_videos": len(sorted(test_dir_path.glob("*.mp4"))) if test_dir_path.exists() else 0,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "duplicate_csv_ids": duplicate_ids,
            "missing_video_matches": missing_in_csv,
            "missing_csv_matches": missing_in_files,
            "unreadable_videos": unreadable_videos,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "missing_label_rows": missing_label_rows,
            "total_issues": len(issues),
        },
        "video_metadata": metadata_by_video,
        "review_log": issues,
    }

    json_path = results_dir / "dataset_report.json"
    txt_path = results_dir / "dataset_report.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "DriveGuardAI dataset inspection report",
        "===================================",
        f"Project root: {root}",
        f"Training CSV rows: {summary['dataset_summary']['train_rows']}",
        f"Test CSV rows: {summary['dataset_summary']['test_rows']}",
        f"Training videos: {summary['dataset_summary']['train_videos']}",
        f"Test videos: {summary['dataset_summary']['test_videos']}",
        f"Positive samples: {positive_samples}",
        f"Negative samples: {negative_samples}",
        f"Duplicate IDs: {duplicate_ids}",
        f"Video files with no CSV match: {summary['dataset_summary']['missing_video_matches']}",
        f"CSV IDs with no video match: {summary['dataset_summary']['missing_csv_matches']}",
        f"Unreadable videos: {unreadable_videos}",
        f"Invalid timestamps: {len(invalid_timestamp_rows)}",
        f"Missing labels or timestamps: {len(missing_label_rows)}",
    ]
    if issues:
        report_lines.extend(["", "Review log:"])
        report_lines.extend(f"- {issue}" for issue in issues)
    else:
        report_lines.extend(["", "Review log:", "- No structural issues detected."])

    txt_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the DriveGuardAI Nexar dataset and write a human-readable report."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing the data, src, and results folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for the JSON/TXT dataset summary report.",
    )
    args = parser.parse_args()

    report = inspect_dataset(args.project_root, output_dir=args.output_dir)
    print(json.dumps(report["dataset_summary"], indent=2))
    print("\nDataset report saved to results/dataset_report.json and results/dataset_report.txt")


if __name__ == "__main__":
    main()
