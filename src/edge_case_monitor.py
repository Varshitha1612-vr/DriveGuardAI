
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def inspect_frame(frame: Any) -> dict[str, Any]:
    """Check a video frame for possible image-quality problems."""

    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return {
            "valid": False,
            "brightness": None,
            "blur_score": None,
            "warnings": ["Invalid or empty frame"],
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []

    # Preliminary thresholds. Calibrate on representative videos.
    if brightness < 45:
        warnings.append("Potentially low-light frame")

    if blur_score < 50:
        warnings.append("Potentially blurry frame")

    return {
        "valid": True,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "warnings": warnings,
    }


def inspect_detection_stability(
    current_count: int,
    previous_count: int | None,
) -> list[str]:
    """Flag a large detection-count drop; this is not proof of failure."""

    if previous_count is None or previous_count < 3:
        return []

    if current_count < previous_count * 0.5:
        return ["Sudden detection-count drop"]

    return []
