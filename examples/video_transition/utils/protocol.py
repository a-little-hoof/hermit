"""Video-time scheduling shared by preparation, generation, and evaluation."""
from __future__ import annotations

import math
import re

FPS = 16
TEMPORAL_COMPRESSION = 4
BLOCK_LATENTS = 3
METRICS = ("VBS", "MBS", "OSE", "NSE", "SP", "ScP", "SUF", "TP")


def latent_boundary(seconds: float, fps: int = FPS, block: int = BLOCK_LATENTS) -> int:
    """First block whose first newly decoded pixel frame is not before seconds."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("event time must be finite and nonnegative")
    if seconds == 0:
        return 0
    return math.ceil((seconds * fps + 3) / (TEMPORAL_COMPRESSION * block)) * block


def pixel_time(latent_index: int, fps: int = FPS) -> float:
    return max(0, TEMPORAL_COMPRESSION * latent_index - 3) / fps


def validate_task(task: dict) -> None:
    events = task.get("events")
    if not isinstance(task.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task["id"]):
        raise ValueError("task id must be a filename-safe identifier")
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("a task needs an id and at least two instruction events")
    if events[0].get("time_s") != 0:
        raise ValueError("the first instruction must start at video time zero")
    previous = -1
    for event in events:
        time_s = float(event["time_s"])
        if not isinstance(event.get("prompt"), str) or not event["prompt"].strip():
            raise ValueError("each event needs a nonempty prompt")
        position = latent_boundary(time_s)
        if position <= previous:
            raise ValueError("instruction times must advance to distinct generation blocks")
        previous = position
    duration = float(task["duration_s"])
    if not math.isfinite(duration) or duration < events[-1]["time_s"] + 5:
        raise ValueError("the final instruction needs at least five seconds of evaluation context")


def uniform_times(start: float, end: float, count: int) -> list[float]:
    if count <= 0 or not start < end:
        raise ValueError("a sampling interval needs positive length and frame count")
    return [start + (end - start) * j / (count + 1) for j in range(1, count + 1)]


def metric_times(metric: str, switch_s: float) -> list[float]:
    """SEGUE Table 12 sampling in video seconds, before choosing native frames."""
    t = switch_s
    if metric in ("VBS", "MBS"):
        return uniform_times(t - 2, t, 2) + uniform_times(t, t + 2, 2)
    if metric == "OSE":
        return [t - 0.5] + uniform_times(t, t + 1.5, 4)
    if metric == "NSE":
        return [t - 0.5] + uniform_times(t, t + 5, 4)
    if metric in ("SP", "ScP"):
        return uniform_times(t - 1, t, 2) + uniform_times(t, t + 1, 2)
    if metric == "SUF":
        return uniform_times(t - 1, t, 2) + uniform_times(t, t + 5, 4)
    if metric == "TP":
        return uniform_times(t - 2, t, 2) + uniform_times(t, t + 5, 6)
    raise ValueError(f"unknown metric: {metric}")
