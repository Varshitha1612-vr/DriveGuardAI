from __future__ import annotations

from src.risk_engine import Detection, RiskEngine


def test_close_fast_object_is_high_risk():
    engine = RiskEngine(
        caution_threshold=0.45,
        high_risk_threshold=0.75,
        temporal_confirmation_frames=1,
    )

    detections = [
        Detection(
            track_id=7,
            class_name="car",
            confidence=0.86,
            box=(300.0, 250.0, 420.0, 390.0),
            timestamp=0.0,
            frame_index=0,
        ),
        Detection(
            track_id=7,
            class_name="car",
            confidence=0.85,
            box=(310.0, 300.0, 430.0, 430.0),
            timestamp=0.5,
            frame_index=1,
        ),
        Detection(
            track_id=7,
            class_name="car",
            confidence=0.84,
            box=(320.0, 340.0, 440.0, 470.0),
            timestamp=1.0,
            frame_index=2,
        ),
    ]

    warnings = engine.process_detections(detections)
    final_warning = warnings[-1]

    assert final_warning.level in {"CAUTION", "HIGH_RISK"}
    assert final_warning.score >= 0.45
    assert "Heuristic risk estimate" in final_warning.explanation


def test_warning_levels_respect_thresholds():
    engine = RiskEngine(caution_threshold=0.55, high_risk_threshold=0.8, temporal_confirmation_frames=1)

    mild = Detection(
        track_id="mild",
        class_name="pedestrian",
        confidence=0.72,
        box=(200.0, 220.0, 230.0, 280.0),
        timestamp=0.0,
        frame_index=0,
    )
    mild_second = Detection(
        track_id="mild",
        class_name="pedestrian",
        confidence=0.7,
        box=(205.0, 260.0, 235.0, 320.0),
        timestamp=0.5,
        frame_index=1,
    )

    severe = Detection(
        track_id="severe",
        class_name="motorcycle",
        confidence=0.9,
        box=(350.0, 180.0, 500.0, 440.0),
        timestamp=0.0,
        frame_index=0,
    )
    severe_second = Detection(
        track_id="severe",
        class_name="motorcycle",
        confidence=0.9,
        box=(360.0, 250.0, 520.0, 500.0),
        timestamp=0.5,
        frame_index=1,
    )

    mild_warnings = engine.process_detections([mild, mild_second])
    severe_warnings = engine.process_detections([severe, severe_second])

    assert mild_warnings[-1].level in {"NORMAL", "CAUTION"}
    assert severe_warnings[-1].level in {"CAUTION", "HIGH_RISK"}
    assert severe_warnings[-1].score >= mild_warnings[-1].score


def test_ttc_is_only_estimated_when_track_moves_toward_ego_region():
    toward_engine = RiskEngine(temporal_confirmation_frames=1)
    toward = toward_engine.process_detections([
        Detection(1, "car", 0.9, (250, 250, 350, 350), 0.0),
        Detection(1, "car", 0.9, (300, 250, 400, 350), 1.0),
    ])[-1]

    away_engine = RiskEngine(temporal_confirmation_frames=1)
    away = away_engine.process_detections([
        Detection(2, "car", 0.9, (300, 250, 400, 350), 0.0),
        Detection(2, "car", 0.9, (250, 250, 350, 350), 1.0),
    ])[-1]

    assert toward.ttc_seconds is not None
    assert away.ttc_seconds is None
