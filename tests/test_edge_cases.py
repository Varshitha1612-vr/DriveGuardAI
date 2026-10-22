
import csv

import cv2
import numpy as np

from src.edge_case_samples import generate_dataset_samples
from src.edge_case_monitor import (
    inspect_frame,
    inspect_detection_stability,
)


def test_invalid_frame():
    result = inspect_frame(None)

    assert result["valid"] is False


def test_dark_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = inspect_frame(frame)

    assert result["valid"] is True
    assert "Potentially low-light frame" in result["warnings"]


def test_bright_frame():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = inspect_frame(frame)

    assert result["brightness"] == 200.0
    assert "Potentially low-light frame" not in result["warnings"]


def test_detection_count_drop():
    result = inspect_detection_stability(1, 6)

    assert "Sudden detection-count drop" in result


def test_stable_detection_count():
    result = inspect_detection_stability(5, 6)

    assert result == []


def test_first_frame_has_no_drop_warning():
    result = inspect_detection_stability(0, None)

    assert result == []


def test_dataset_samples_use_decoded_video_frames_and_write_results(tmp_path):
    train_dir = tmp_path / "data" / "train"
    train_dir.mkdir(parents=True)
    csv_path = tmp_path / "data" / "train.csv"

    for video_id in ("00001", "00002"):
        video_path = train_dir / f"{video_id}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (32, 32),
        )
        assert writer.isOpened()
        for index in range(12):
            writer.write(
                np.full((32, 32, 3), 30 + index * 10, dtype=np.uint8)
            )
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("id", "time_of_event", "target"),
        )
        writer.writeheader()
        writer.writerow({"id": "00001", "time_of_event": "0.5", "target": "1"})
        writer.writerow({"id": "00002", "time_of_event": "", "target": "0"})

    output_dir = tmp_path / "results"
    summary = generate_dataset_samples(
        tmp_path,
        output_dir,
        samples_per_class=1,
    )

    assert summary["sample_count"] == 2
    assert summary["accuracy_percent"] is None
    assert "video-level event labels" in summary["accuracy_note"]
    assert (output_dir / "edge_case_dataset_contact_sheet.png").is_file()
    assert (output_dir / "summary.json").is_file()

    with (output_dir / "sample_results.csv").open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2

    for row in rows:
        saved_frame = cv2.imread(
            str(output_dir / row["frame_path"]),
            cv2.IMREAD_COLOR,
        )
        capture = cv2.VideoCapture(
            str(train_dir / f'{row["video_id"]}.mp4')
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        success, source_frame = capture.read()
        capture.release()

        assert success
        np.testing.assert_array_equal(saved_frame, source_frame)
