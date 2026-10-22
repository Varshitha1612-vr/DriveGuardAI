
from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_low_light_episodes(
    events: pd.DataFrame,
    brightness_threshold: float = 45.0,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """Group consecutive low-light frames into episodes."""

    if events.empty:
        return []

    required = {"frame_index", "timestamp_seconds", "brightness"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing columns: {required - set(events.columns)}")

    frames = events.sort_values("frame_index").copy()
    frames = frames.drop_duplicates("frame_index", keep="last")
    frames["is_low_light"] = (
        pd.to_numeric(frames["brightness"], errors="coerce")
        < brightness_threshold
    )

    episodes = []
    current = []

    def save_episode(rows):
        if not rows:
            return

        group = pd.DataFrame(rows)
        start = float(group["timestamp_seconds"].iloc[0])
        end = float(group["timestamp_seconds"].iloc[-1])

        episodes.append({
            "start_frame": int(group["frame_index"].iloc[0]),
            "end_frame": int(group["frame_index"].iloc[-1]),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(len(group) / fps, 3),
            "frames_affected": len(group),
            "average_brightness": round(group["brightness"].mean(), 2),
            "minimum_brightness": round(group["brightness"].min(), 2),
        })

    previous_frame = None

    for row in frames.to_dict("records"):
        frame_index = int(row["frame_index"])
        is_consecutive = (
            previous_frame is not None
            and frame_index == previous_frame + 1
        )

        if not row["is_low_light"]:
            save_episode(current)
            current = []
        else:
            if current and not is_consecutive:
                save_episode(current)
                current = []

            if pd.notna(row["brightness"]):
                current.append(row)

        previous_frame = frame_index

    save_episode(current)
    return episodes
