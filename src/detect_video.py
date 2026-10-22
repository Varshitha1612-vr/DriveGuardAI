
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

from src.edge_case_monitor import (
    inspect_frame,
    inspect_detection_stability,
)
from src.risk_engine import Detection, RiskEngine

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


def get_device() -> str:
    """Use Apple MPS when available; otherwise use CPU."""
    if (
        torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory."""
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def validate_video(video_path: str | Path) -> Path:
    """Validate the input video path."""
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
    """
    Run YOLO + ByteTrack + risk analysis + edge-case monitoring.

    Edge-case checks flag possible image-quality issues and sudden
    changes in raw YOLO detection counts. These are heuristic alerts,
    not proof that an object was missed or that a collision will occur.
    """
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is unavailable. Install project requirements."
        ) from ULTRALYTICS_IMPORT_ERROR

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    source_video = validate_video(video_path)
    target_dir = ensure_output_dir(output_dir)

    output_path = target_dir / f"{source_video.stem}_annotated.mp4"
    events_path = target_dir / f"{source_video.stem}_risk_events.csv"
    edge_events_path = (
        target_dir / f"{source_video.stem}_edge_case_events.csv"
    )

    capture = cv2.VideoCapture(str(source_video))

    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {source_video}")

    writer = None

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Video dimensions could not be read: {source_video}"
            )

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        if not writer.isOpened():
            writer.release()
            # Retry with a commonly available MP4 codec.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )

        if not writer.isOpened():
            raise OSError(
                f"Could not create annotated video: {output_path}"
            )

        device = get_device()
        model = YOLO(model_name)
        model.to(device)

        risk_engine = RiskEngine(image_size=(width, height))

        track_classes: dict[int, str] = {}
        risk_events: list[dict[str, Any]] = []
        edge_events: list[dict[str, Any]] = []
        event_levels: dict[int, str] = {}
        peak_track_warnings: dict[int, dict[str, Any]] = {}

        severity = {
            "NORMAL": 0,
            "CAUTION": 1,
            "HIGH_RISK": 2,
        }

        frame_index = 0
        processed_frames = 0
        previous_detection_count: int | None = None
        start_time = time.perf_counter()

        while True:
            ret, frame = capture.read()

            if not ret:
                break

            if max_frames is not None and processed_frames >= max_frames:
                break

            # 1. Inspect image quality before running YOLO.
            quality = inspect_frame(frame)

            # 2. Run existing YOLO tracking.
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=confidence_threshold,
                tracker="bytetrack.yaml",
                device=device,
            )

            result = results[0] if results else None

            if result is not None:
                annotated_frame = result.plot()
                boxes = result.boxes
            else:
                annotated_frame = frame.copy()
                boxes = None

            # Count raw YOLO output boxes, not only the road-user classes.
            current_detection_count = (
                len(boxes) if boxes is not None else 0
            )

            edge_warnings = list(quality["warnings"])

            edge_warnings.extend(
                inspect_detection_stability(
                    current_detection_count,
                    previous_detection_count,
                )
            )

            # Record edge-case alerts independently from risk events.
            for edge_warning in edge_warnings:
                edge_events.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 3),
                    "warning": edge_warning,
                    "brightness": quality["brightness"],
                    "blur_score": quality["blur_score"],
                    "detection_count": current_detection_count,
                })

            previous_detection_count = current_detection_count

            # Show edge-case alerts on the annotated frame.
            for warning_index, edge_warning in enumerate(edge_warnings):
                y_position = height - 20 - warning_index * 24

                if y_position < 20:
                    break

                cv2.putText(
                    annotated_frame,
                    f"EDGE CASE: {edge_warning}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )

            tracked_detections: list[Detection] = []

            # 3. Prepare tracked road-user detections for the risk engine.
            if (
                result is not None
                and boxes is not None
                and boxes.id is not None
            ):
                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                track_ids = boxes.id.cpu().tolist()
                names = result.names

                for box, confidence, class_id, track_id in zip(
                    coordinates,
                    confidences,
                    class_ids,
                    track_ids,
                ):
                    name = str(names[int(class_id)])

                    if name not in {
                        "car",
                        "motorcycle",
                        "bus",
                        "truck",
                        "bicycle",
                        "person",
                    }:
                        continue

                    normalized_track_id = int(track_id)
                    track_classes[normalized_track_id] = name

                    tracked_detections.append(
                        Detection(
                            track_id=normalized_track_id,
                            class_name=name,
                            confidence=float(confidence),
                            box=tuple(float(value) for value in box),
                            timestamp=frame_index / fps,
                            frame_index=frame_index,
                        )
                    )

            # 4. Run the existing heuristic risk engine.
            warnings = risk_engine.process_detections(
                tracked_detections
            )

            warning_by_track = {
                int(warning.track_id): warning
                for warning in warnings
            }

            active_warnings = [
                warning
                for warning in warnings
                if warning.level != "NORMAL"
            ]

            for warning in warnings:
                track_id = int(warning.track_id)
                previous_peak = peak_track_warnings.get(track_id)

                if (
                    previous_peak is None
                    or severity[warning.level]
                    > severity[previous_peak["level"]]
                    or (
                        warning.level == previous_peak["level"]
                        and warning.score > previous_peak["score"]
                    )
                ):
                    peak_track_warnings[track_id] = warning.to_dict()

            # 5. Draw existing risk annotations.
            for detection in tracked_detections:
                warning = warning_by_track.get(detection.track_id)

                if warning is None:
                    continue

                box_color = {
                    "NORMAL": (0, 200, 0),
                    "CAUTION": (0, 165, 255),
                    "HIGH_RISK": (0, 0, 255),
                }.get(warning.level, (255, 255, 0))

                x1, y1, x2, y2 = (
                    int(value) for value in detection.box
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                label = (
                    f"{detection.class_name} #{detection.track_id} "
                    f"{detection.confidence:.2f} | {warning.level}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

            if active_warnings:
                highest_warning = max(
                    active_warnings,
                    key=lambda warning: severity[warning.level],
                )

                color = (
                    (0, 0, 255)
                    if highest_warning.level == "HIGH_RISK"
                    else (0, 165, 255)
                )

                cv2.putText(
                    annotated_frame,
                    (
                        f'{highest_warning.level}: '
                        f'{highest_warning.class_name} '
                        f'#{highest_warning.track_id}'
                    ),
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # 6. Preserve existing risk-event recording.
            for warning in warnings:
                track_id = int(warning.track_id)
                previous_level = event_levels.get(track_id, "NORMAL")

                if severity[warning.level] > severity[previous_level]:
                    if warning.level != "NORMAL":
                        risk_events.append(warning.to_dict())

                event_levels[track_id] = warning.level

            # 7. Draw frame/time/FPS information.
            elapsed_seconds = frame_index / fps

            average_elapsed = (
                time.perf_counter() - start_time
            ) / max(processed_frames + 1, 1)

            current_fps = 1.0 / max(average_elapsed, 1e-6)

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
                cv2.imshow(
                    "DriveGuardAI detection",
                    annotated_frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        elapsed_run = time.perf_counter() - start_time

        # 8. Save existing risk events.
        risk_fieldnames = [
            "track_id",
            "class_name",
            "level",
            "score",
            "confidence",
            "explanation",
            "timestamp",
            "ttc_seconds",
        ]

        with events_path.open(
            "w", newline="", encoding="utf-8"
        ) as event_file:
            event_writer = csv.DictWriter(
                event_file,
                fieldnames=risk_fieldnames,
                extrasaction="ignore",
            )
            event_writer.writeheader()
            event_writer.writerows(risk_events)

        # 9. Save the new edge-case event report.
        edge_fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "warning",
            "brightness",
            "blur_score",
            "detection_count",
        ]

        with edge_events_path.open(
            "w", newline="", encoding="utf-8"
        ) as edge_file:
            edge_writer = csv.DictWriter(
                edge_file,
                fieldnames=edge_fieldnames,
            )
            edge_writer.writeheader()
            edge_writer.writerows(edge_events)

        class_track_counts: dict[str, int] = {}

        for class_name in track_classes.values():
            class_track_counts[class_name] = (
                class_track_counts.get(class_name, 0) + 1
            )

        highest_risk = max(
            (
                warning["level"]
                for warning in peak_track_warnings.values()
            ),
            key=lambda level: severity[level],
            default="NORMAL",
        )

        summary = {
            "source_video": str(source_video),
            "output_video": str(output_path),
            "risk_events_csv": str(events_path),
            "edge_case_events_csv": str(edge_events_path),
            "edge_case_event_count": len(edge_events),
            "processed_frames": processed_frames,
            "reported_total_frames": total_frames,
            "fps": fps,
            "resolution": {
                "width": width,
                "height": height,
            },
            "elapsed_seconds": round(elapsed_run, 3),
            "device": device,
            "model": model_name,
            "confidence_threshold": confidence_threshold,
            "display_mode": display,
            "tracked_object_count": len(track_classes),
            "tracked_objects_by_class": class_track_counts,
            "highest_risk": highest_risk,
            "risk_events": risk_events,
        }

        return summary

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run DriveGuardAI YOLO detection, risk analysis, "
            "and edge-case monitoring."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model, e.g. yolov8n.pt or yolov8s.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/annotated_videos",
        help="Directory for output videos and CSV reports.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold between 0 and 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated frames in a window.",
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
    except Exception as exc:
        print(
            f"DriveGuardAI detection failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Processing complete.")
    print(f"Annotated video: {summary['output_video']}")
    print(f"Risk events CSV: {summary['risk_events_csv']}")
    print(f"Edge-case CSV: {summary['edge_case_events_csv']}")
    print(
        "Edge-case events recorded:",
        summary["edge_case_event_count"],
    )
    print(f"Summary: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
