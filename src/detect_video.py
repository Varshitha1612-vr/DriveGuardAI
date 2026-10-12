from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.risk_engine import Detection, RiskEngine

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at import time for graceful errors.
    torch = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - Ultralytics is required for this pipeline.
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


def get_device() -> str:
    """Choose the best available device, falling back to CPU when MPS is unavailable."""
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the directory used to save annotated videos."""
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def validate_video(video_path: str | Path) -> Path:
    """Return a validated video path or raise a clear error."""
    candidate = Path(video_path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Video file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Video path is not a file: {candidate}")
    return candidate


def process_video(
    video_path: str | Path,
    *,
    model_name: str = "yolov8n.pt",
    output_dir: str | Path = "outputs/annotated_videos",
    confidence_threshold: float = 0.25,
    display: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run a YOLO video detection pipeline and save an annotated output video."""
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not available. Install the project requirements and retry."
        ) from ULTRALYTICS_IMPORT_ERROR

    source_video = validate_video(video_path)
    target_dir = ensure_output_dir(output_dir)
    output_path = target_dir / f"{source_video.stem}_annotated.mp4"
    events_path = target_dir / f"{source_video.stem}_risk_events.csv"

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {source_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width == 0 or height == 0:
        raise ValueError(f"Video dimensions could not be read for: {source_video}")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not write annotated video to: {output_path}")

    try:
        model = YOLO(model_name)
        model.to(get_device())
        risk_engine = RiskEngine(image_size=(width, height))
        track_classes: dict[int, str] = {}
        risk_events: list[dict[str, Any]] = []
        event_levels: dict[int, str] = {}
        peak_track_warnings: dict[int, dict[str, Any]] = {}
        severity = {"NORMAL": 0, "CAUTION": 1, "HIGH_RISK": 2}

        frame_index = 0
        processed_frames = 0
        start_time = time.perf_counter()

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if max_frames is not None and processed_frames >= max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=confidence_threshold,
                tracker="bytetrack.yaml",
                device=get_device(),
            )

            result = results[0] if results else None
            if result is not None:
                annotated_frame = result.plot()
                boxes = result.boxes
                tracked_detections: list[Detection] = []
                if boxes is not None and boxes.id is not None:
                    coordinates = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    class_ids = boxes.cls.cpu().tolist()
                    track_ids = boxes.id.cpu().tolist()
                    names = result.names

                    for coordinates_for_box, confidence, class_id, track_id in zip(
                        coordinates, confidences, class_ids, track_ids
                    ):
                        name = str(names[int(class_id)])
                        if name not in {"car", "motorcycle", "bus", "truck", "bicycle", "person"}:
                            continue
                        normalized_track_id = int(track_id)
                        track_classes[normalized_track_id] = name
                        tracked_detections.append(
                            Detection(
                                track_id=normalized_track_id,
                                class_name=name,
                                confidence=float(confidence),
                                box=tuple(float(value) for value in coordinates_for_box),
                                timestamp=frame_index / fps if fps > 0 else 0.0,
                                frame_index=frame_index,
                            )
                        )

                warnings = risk_engine.process_detections(tracked_detections)
                warning_by_track = {
                    int(warning.track_id): warning for warning in warnings
                }
                active_warnings = [
                    warning for warning in warnings if warning.level != "NORMAL"
                ]

                for warning in warnings:
                    track_id = int(warning.track_id)
                    previous_peak = peak_track_warnings.get(track_id)
                    if (
                        previous_peak is None
                        or severity[warning.level] > severity[previous_peak["level"]]
                        or (
                            warning.level == previous_peak["level"]
                            and warning.score > previous_peak["score"]
                        )
                    ):
                        peak_track_warnings[track_id] = warning.to_dict()

                for detection in tracked_detections:
                    warning = warning_by_track[detection.track_id]
                    box_color = {
                        "NORMAL": (0, 200, 0),
                        "CAUTION": (0, 165, 255),
                        "HIGH_RISK": (0, 0, 255),
                    }[warning.level]
                    x1, y1, x2, y2 = (int(value) for value in detection.box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    label = (
                        f"{detection.class_name} #{detection.track_id} "
                        f"{detection.confidence:.2f} | {warning.level}"
                    )
                    label_y = max(y1 - 8, 22)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

                if active_warnings:
                    highest_warning = max(
                        active_warnings, key=lambda warning: severity[warning.level]
                    )
                    color = (0, 0, 255) if highest_warning.level == "HIGH_RISK" else (0, 165, 255)
                    cv2.putText(
                        annotated_frame,
                        f'{highest_warning.level}: {highest_warning.class_name} #{highest_warning.track_id}',
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                for warning in warnings:
                    track_id = int(warning.track_id)
                    previous_level = event_levels.get(track_id, "NORMAL")
                    if severity[warning.level] > severity[previous_level]:
                        if warning.level != "NORMAL":
                            risk_events.append(warning.to_dict())
                    event_levels[track_id] = warning.level
            else:
                annotated_frame = frame

            elapsed_seconds = frame_index / fps if fps > 0 else 0.0
            current_fps = 1.0 / max((time.perf_counter() - start_time) / max(processed_frames, 1), 1e-6)

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Time: {elapsed_seconds:.2f}s",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            writer.write(annotated_frame)
            processed_frames += 1
            frame_index += 1

            if display:
                cv2.imshow("DriveGuardAI detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        elapsed_run = time.perf_counter() - start_time
        with events_path.open("w", newline="", encoding="utf-8") as event_file:
            fieldnames = [
                "track_id",
                "class_name",
                "level",
                "score",
                "confidence",
                "explanation",
                "timestamp",
                "ttc_seconds",
            ]
            writer_csv = csv.DictWriter(event_file, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(risk_events)

        class_track_counts: dict[str, int] = {}
        for class_name in track_classes.values():
            class_track_counts[class_name] = class_track_counts.get(class_name, 0) + 1
        highest_risk = max(
            (warning["level"] for warning in peak_track_warnings.values()),
            key=lambda level: severity[level],
            default="NORMAL",
        )
        summary = {
            "source_video": str(source_video),
            "output_video": str(output_path),
            "risk_events_csv": str(events_path),
            "processed_frames": processed_frames,
            "reported_total_frames": total_frames,
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "elapsed_seconds": round(elapsed_run, 3),
            "device": get_device(),
            "model": model_name,
            "confidence_threshold": confidence_threshold,
            "display_mode": display,
            "tracked_object_count": len(track_classes),
            "tracked_objects_by_class": class_track_counts,
            "highest_risk": highest_risk,
            "risk_events": risk_events,
            "peak_track_warnings": list(peak_track_warnings.values()),
        }
        return summary
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO-based video detection pipeline for DriveGuardAI.")
    parser.add_argument("--video", required=True, help="Path to an MP4 video file to analyze.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics detection model name. Example: yolov8n.pt or yolov8s.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/annotated_videos",
        help="Directory where annotated videos are written.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold used for object detection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for testing the pipeline on a small subset of frames.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames in a window while processing.",
    )
    args = parser.parse_args()

    try:
        summary = process_video(
            args.video,
            model_name=args.model,
            output_dir=args.output_dir,
            confidence_threshold=args.conf,
            display=args.display,
            max_frames=args.max_frames,
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"DriveGuardAI detection failed: {exc}", file=sys.stderr)
        return 1

    print(summary)
    print(f"Annotated video saved to: {summary['output_video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
