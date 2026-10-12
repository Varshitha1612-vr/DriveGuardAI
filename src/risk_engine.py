from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Literal

RiskLevel = Literal["NORMAL", "CAUTION", "HIGH_RISK"]


@dataclass(slots=True)
class Detection:
    """Single tracked detection from the YOLO pipeline."""

    track_id: str | int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    timestamp: float
    frame_index: int = 0

    @property
    def x1(self) -> float:
        return float(self.box[0])

    @property
    def y1(self) -> float:
        return float(self.box[1])

    @property
    def x2(self) -> float:
        return float(self.box[2])

    @property
    def y2(self) -> float:
        return float(self.box[3])

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(self.x2 - self.x1, 0.0)

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 0.0)

    @property
    def area(self) -> float:
        return max(self.width * self.height, 1.0)


@dataclass(slots=True)
class RiskWarning:
    """Structured risk warning emitted for a tracked object."""

    track_id: str | int
    class_name: str
    level: RiskLevel
    score: float
    confidence: float
    explanation: str
    timestamp: float
    ttc_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "level": self.level,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "timestamp": self.timestamp,
            "ttc_seconds": self.ttc_seconds,
        }


class RiskEngine:
    """Heuristic collision-risk analysis for tracked road users.

    The model is intentionally conservative: it never claims to produce real-world
    distance measurements or collision probabilities without calibration. The
    outputs are best interpreted as heuristic risk estimates for alerting and
    dashboarding.
    """

    def __init__(
        self,
        *,
        caution_threshold: float = 0.45,
        high_risk_threshold: float = 0.75,
        temporal_confirmation_frames: int = 2,
        ego_center: tuple[float, float] | None = None,
        image_size: tuple[int, int] = (1280, 720),
        calibration: dict[str, float] | None = None,
    ) -> None:
        self.caution_threshold = caution_threshold
        self.high_risk_threshold = high_risk_threshold
        self.temporal_confirmation_frames = temporal_confirmation_frames
        self.ego_center = ego_center or (image_size[0] * 0.5, image_size[1] * 0.85)
        self.image_width, self.image_height = image_size
        self.calibration = calibration or {}
        self.track_history: dict[str | int, list[Detection]] = {}
        self.track_alert_counts: dict[str | int, int] = {}

    def _lookup_calibration(self, key: str, default: float) -> float:
        return float(self.calibration.get(key, default))

    def _safe_level_from_score(self, score: float) -> RiskLevel:
        if score >= self.high_risk_threshold:
            return "HIGH_RISK"
        if score >= self.caution_threshold:
            return "CAUTION"
        return "NORMAL"

    def _estimate_relative_motion(self, current: Detection, previous: Detection) -> tuple[float, float, float, float | None, float]:
        """Return displacement, radial closing speed, area change, TTC, and motion factor."""
        dt = max(current.timestamp - previous.timestamp, 0.1)
        center_distance = hypot(current.center_x - previous.center_x, current.center_y - previous.center_y)
        velocity_x = (current.center_x - previous.center_x) / dt
        velocity_y = (current.center_y - previous.center_y) / dt
        area_change = (current.area - previous.area) / max(previous.area, 1.0)

        ego_x, ego_y = self.ego_center
        to_ego_x = ego_x - current.center_x
        to_ego_y = ego_y - current.center_y
        distance_to_ego = hypot(to_ego_x, to_ego_y)
        if distance_to_ego <= 1e-6:
            closing_speed_px_per_s = 0.0
        else:
            closing_speed_px_per_s = (
                velocity_x * to_ego_x + velocity_y * to_ego_y
            ) / distance_to_ego

        if closing_speed_px_per_s <= 0:
            ttc_seconds = None
        else:
            ttc_seconds = distance_to_ego / closing_speed_px_per_s

        motion_factor = min(max(closing_speed_px_per_s / 250.0, 0.0), 1.0)
        return center_distance, closing_speed_px_per_s, area_change, ttc_seconds, motion_factor

    def _evaluate_track(self, detection: Detection) -> RiskWarning:
        history = self.track_history.get(detection.track_id, [])
        if not history:
            self.track_history.setdefault(detection.track_id, []).append(detection)
            return RiskWarning(
                track_id=detection.track_id,
                class_name=detection.class_name,
                level="NORMAL",
                score=0.0,
                confidence=detection.confidence,
                explanation="No prior motion history available; risk remains unconfirmed.",
                timestamp=detection.timestamp,
                ttc_seconds=None,
            )

        previous = history[-1]
        _, speed_px_per_s, area_change, ttc_seconds, motion_factor = self._estimate_relative_motion(detection, previous)

        ego_x, ego_y = self.ego_center
        distance_to_ego = hypot(detection.center_x - ego_x, detection.center_y - ego_y)
        distance_ratio = min(distance_to_ego / max(hypot(self.image_width, self.image_height), 1.0), 1.0)
        relative_proximity = 1.0 - distance_ratio
        size_change_risk = min(abs(area_change) * 1.6, 1.0)

        # Heuristic, not calibrated. We interpret a vehicle/pedestrian approaching the ego area
        # as a higher risk signal while avoiding false precision claims.
        approach_bonus = 0.0
        if speed_px_per_s > 0 and detection.center_y > previous.center_y and detection.center_y > self.image_height * 0.35:
            approach_bonus += 0.08
        if speed_px_per_s > 40:
            approach_bonus += 0.06
        if ttc_seconds is not None and ttc_seconds < 8.0:
            approach_bonus += 0.10

        # Objects near the lower-mid image region are treated as more relevant for front-facing risk.
        lower_region_bonus = max(0.0, 1.0 - (detection.center_y / self.image_height)) * 0.12
        scale_bonus = min((detection.area / 25000.0), 1.0) * 0.08

        class_weight_map = {
            "pedestrian": 0.65,
            "person": 0.65,
            "bicycle": 0.75,
            "motorcycle": 1.00,
            "car": 0.82,
            "bus": 0.9,
            "truck": 0.9,
        }
        class_weight = class_weight_map.get(detection.class_name.lower(), 0.7)

        score = (
            relative_proximity * 0.25
            + motion_factor * 0.22
            + size_change_risk * 0.18
            + lower_region_bonus
            + scale_bonus
            + approach_bonus
        ) * class_weight

        score = min(max(score, 0.0), 1.0)
        level = self._safe_level_from_score(score)

        if level != "NORMAL":
            self.track_alert_counts[detection.track_id] = self.track_alert_counts.get(detection.track_id, 0) + 1
        else:
            self.track_alert_counts[detection.track_id] = 0

        if self.track_alert_counts.get(detection.track_id, 0) < self.temporal_confirmation_frames and level != "HIGH_RISK":
            score = max(score * 0.75, 0.0)
            level = self._safe_level_from_score(score)

        explanation_parts = [
            "Heuristic risk estimate",
            f"{detection.class_name} track {detection.track_id}",
            f"relative motion {speed_px_per_s:.1f}px/s",
            f"approach to ego region {distance_to_ego:.1f}px away",
        ]
        if ttc_seconds is not None:
            explanation_parts.append(f"estimated TTC {ttc_seconds:.1f}s")
        if abs(area_change) > 0.05:
            explanation_parts.append(f"size change {area_change:.2f}")
        explanation = "; ".join(explanation_parts) + "."

        warning = RiskWarning(
            track_id=detection.track_id,
            class_name=detection.class_name,
            level=level,
            score=score,
            confidence=detection.confidence,
            explanation=explanation,
            timestamp=detection.timestamp,
            ttc_seconds=ttc_seconds,
        )
        history.append(detection)
        return warning

    def process_detections(self, detections: list[Detection]) -> list[RiskWarning]:
        """Evaluate a frame of detections and return a warning for each tracked object."""
        warnings: list[RiskWarning] = []
        for detection in detections:
            warnings.append(self._evaluate_track(detection))
        return warnings

    def summarize_warning(self, warning: RiskWarning) -> str:
        return warning.explanation


__all__ = ["Detection", "RiskWarning", "RiskEngine", "RiskLevel"]
