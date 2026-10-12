from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.detect_video import process_video


st.set_page_config(page_title="DriveGuard AI", page_icon="🚘", layout="wide")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_uploaded_video(uploaded_file: Any, model_name: str, confidence: float) -> dict[str, Any]:
    """Temporarily save an upload, run detection, and remove the source copy."""
    digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()[:12]
    upload_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="driveguard_", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(upload_bytes)

    try:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "annotated_videos"
        result = process_video(
            temp_path,
            model_name=model_name,
            output_dir=output_dir,
            confidence_threshold=confidence,
        )
        generated_path = Path(result["output_video"])
        final_path = output_dir / f"{digest}_annotated.mp4"
        if generated_path != final_path:
            generated_path.replace(final_path)
        result["output_video"] = str(final_path)
        generated_events_path = Path(result["risk_events_csv"])
        final_events_path = output_dir / f"{digest}_risk_events.csv"
        if generated_events_path != final_events_path:
            generated_events_path.replace(final_events_path)
        result["risk_events_csv"] = str(final_events_path)
        result["upload_name"] = uploaded_file.name
        return result
    finally:
        temp_path.unlink(missing_ok=True)


project_root = Path(__file__).resolve().parents[1]
st.title("DriveGuard AI")
st.caption("Upload a road video, run object detection and tracking, then review the annotated result.")

st.subheader("Analyze a video")
uploaded_file = st.file_uploader(
    "Choose an MP4 video",
    type=["mp4"],
    help="The uploaded source is processed locally and removed after analysis.",
)

if uploaded_file is not None:
    st.video(uploaded_file.getvalue())
    settings_col, button_col = st.columns([2, 1])
    with settings_col:
        model_name = st.selectbox("Detection model", ["yolov8n.pt", "yolov8s.pt"], index=0)
        confidence = st.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    with button_col:
        st.write("")
        st.write("")
        analyze_clicked = st.button("Analyze video", type="primary", width="stretch")

    upload_digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()[:12]
    if analyze_clicked:
        try:
            with st.spinner("Running YOLO detection and tracking. This may take a while for longer videos..."):
                analysis = process_uploaded_video(uploaded_file, model_name, confidence)
            st.session_state["video_analysis"] = analysis
            st.session_state["video_analysis_digest"] = upload_digest
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

    analysis = st.session_state.get("video_analysis")
    if analysis and st.session_state.get("video_analysis_digest") == upload_digest:
        st.success("Video analysis complete — play the annotated video below.")
        output_path = Path(analysis["output_video"])
        if output_path.exists():
            st.subheader("Annotated video — play here")
            st.video(output_path.read_bytes(), format="video/mp4")
            st.download_button(
                "Download annotated MP4",
                data=output_path.read_bytes(),
                file_name=f"{Path(uploaded_file.name).stem}_annotated.mp4",
                mime="video/mp4",
                width="stretch",
            )
        else:
            st.error(f"Annotated output was not found at {output_path}. Run the analysis again.")

        metric_columns = st.columns(5)
        metric_columns[0].metric("Frames analyzed", analysis["processed_frames"])
        metric_columns[1].metric("Video FPS", analysis["fps"])
        metric_columns[2].metric("Resolution", f'{analysis["resolution"]["width"]}×{analysis["resolution"]["height"]}')
        metric_columns[3].metric("Processing time", f'{analysis["elapsed_seconds"]:.1f}s')
        metric_columns[4].metric("Device", analysis["device"].upper())

        highest_risk = analysis.get("highest_risk", "NORMAL")
        if highest_risk == "HIGH_RISK":
            st.error("HIGH RISK heuristic alert observed. Review the annotated video and event details.")
        elif highest_risk == "CAUTION":
            st.warning("CAUTION heuristic alert observed. Review the annotated video and event details.")
        else:
            st.success("No elevated heuristic warning observed in the analyzed frames.")

        st.subheader("Tracked road users")
        tracked_objects_by_class = analysis.get("tracked_objects_by_class", {})
        if tracked_objects_by_class:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Class": class_name.title(), "Unique tracks": count}
                        for class_name, count in sorted(
                            tracked_objects_by_class.items()
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No tracked road users were found in the analyzed frames.")

        st.subheader("Heuristic risk warnings")
        st.warning(
            "These are uncalibrated heuristic indicators based on image-space motion; "
            "they are not trained collision predictions or safety-certified alerts."
        )
        peak_track_warnings = analysis.get("peak_track_warnings", [])
        if peak_track_warnings:
            st.caption("Bounding-box colors: green = normal, orange = caution, red = high risk.")
            st.dataframe(
                pd.DataFrame(peak_track_warnings),
                hide_index=True,
                width="stretch",
            )

        if analysis["risk_events"]:
            st.subheader("Risk event history")
            st.dataframe(
                pd.DataFrame(analysis["risk_events"]),
                hide_index=True,
                width="stretch",
            )
            events_path = Path(analysis["risk_events_csv"])
            if events_path.exists():
                st.download_button(
                    "Download risk event CSV",
                    data=events_path.read_bytes(),
                    file_name=f"{Path(uploaded_file.name).stem}_risk_events.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.info(
                "No CAUTION or HIGH_RISK heuristic events were generated for this video. "
                "This does not guarantee the video is safe."
            )
        st.caption(
            f'Model: {analysis["model"]} · Confidence threshold: {analysis["confidence_threshold"]:.2f} · '
            f'Input frames: {analysis["reported_total_frames"]}'
        )
else:
    st.info("Upload an MP4 above to begin. The YOLO model may download the first time you analyze a video.")
    st.session_state.pop("video_analysis", None)
    st.session_state.pop("video_analysis_digest", None)

with st.expander("Dataset and evaluation summaries"):
    dataset_summary = load_json(project_root / "results" / "dataset_report.json")
    label_summary = load_json(project_root / "results" / "label_summary.json")
    evaluation_summary = load_json(project_root / "results" / "collision_evaluation.json")

    if dataset_summary:
        dataset_stats = dataset_summary.get("dataset_summary", {})
        cols = st.columns(4)
        cols[0].metric("Training videos", dataset_stats.get("train_videos", 0))
        cols[1].metric("Test videos", dataset_stats.get("test_videos", 0))
        cols[2].metric("Positive samples", dataset_stats.get("positive_samples", 0))
        cols[3].metric("Negative samples", dataset_stats.get("negative_samples", 0))
    else:
        st.info("Dataset summary is unavailable. Run src/inspect_dataset.py first.")

    if label_summary:
        st.subheader("Label validation")
        st.json({
            "rows": label_summary.get("rows"),
            "positive_samples": label_summary.get("positive_samples"),
            "negative_samples": label_summary.get("negative_samples"),
            "timestamp_analysis": label_summary.get("timestamp_analysis", {}),
        })

    if evaluation_summary:
        st.subheader("Metadata baseline (not a video predictor)")
        metrics = evaluation_summary.get("metrics", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Precision", round(metrics.get("precision", 0.0), 4))
        metric_cols[1].metric("Recall", round(metrics.get("recall", 0.0), 4))
        metric_cols[2].metric("F1", round(metrics.get("f1_score", 0.0), 4))
        metric_cols[3].metric("Balanced accuracy", round(metrics.get("balanced_accuracy", 0.0), 4))
        st.warning("These metrics evaluate timestamp presence in the labels, not the uploaded video.")
    else:
        st.info("Evaluation metrics are unavailable. Run src/evaluate_collision.py first.")

    if evaluation_summary:
        comparison = pd.DataFrame(
            [{
                "Model": "Timestamp presence baseline",
                "Precision": evaluation_summary["metrics"].get("precision", 0.0),
                "Recall": evaluation_summary["metrics"].get("recall", 0.0),
            }]
        )
        st.bar_chart(comparison.set_index("Model"))

st.caption("DriveGuard AI is an experimental prototype. Detection and heuristic risk outputs are not certified safety predictions.")
