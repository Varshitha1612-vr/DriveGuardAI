from __future__ import annotations

from pathlib import Path

import cv2

from src.detect_video import process_video


def test_process_video_creates_annotated_output(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    source_video = project_root / "data" / "train" / "00000.mp4"
    output_dir = tmp_path / "annotated_videos"

    summary = process_video(
        source_video,
        model_name="yolov8n.pt",
        output_dir=output_dir,
        max_frames=5,
        display=False,
    )

    assert summary["processed_frames"] == 5
    assert Path(summary["output_video"]).exists()
    assert Path(summary["risk_events_csv"]).exists()
    assert isinstance(summary["tracked_objects_by_class"], dict)
    assert summary["highest_risk"] in {"NORMAL", "CAUTION", "HIGH_RISK"}

    capture = cv2.VideoCapture(summary["output_video"])
    try:
        assert capture.isOpened()
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        assert codec.lower() == "h264"
    finally:
        capture.release()
