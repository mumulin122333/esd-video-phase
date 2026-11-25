from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASES = (
    "Diagnosis",
    "Local injection",
    "Incision / Dissection",
    "Hemostasis",
)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}

_PHASE_ALIASES = {
    "diagnosis": "Diagnosis",
    "local injection": "Local injection",
    "local_injection": "Local injection",
    "incision / dissection": "Incision / Dissection",
    "incision/dissection": "Incision / Dissection",
    "incision and dissection": "Incision / Dissection",
    "hemostasis": "Hemostasis",
    "haemostasis": "Hemostasis",
}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    phase: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class VideoWindow:
    window_id: str
    case_id: str
    video_id: str
    source_video: Path
    clip_path: Path
    start_seconds: float
    source_end_seconds: float
    model_duration_seconds: float


@dataclass(frozen=True)
class ParsedPrediction:
    valid: bool
    segments: tuple[Segment, ...]
    raw: str
    error: str | None = None


def _canonical_phase(value: Any) -> str:
    text = str(value).strip()
    if text in PHASE_INDEX:
        return text
    canonical = _PHASE_ALIASES.get(text.casefold())
    if canonical is None:
        raise ValueError(f"unknown phase: {text!r}")
    return canonical


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _THINK_RE.sub("", text or "").strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last < first:
        raise ValueError("no JSON object found")
    value = json.loads(cleaned[first : last + 1])
    if not isinstance(value, dict):
        raise ValueError("prediction root is not an object")
    return value


def parse_prediction(text: str, duration_seconds: float) -> ParsedPrediction:
    """Parse the model contract with conservative numeric normalization.

    The model must return one or two ordered segments. Small floating-point
    boundary errors are normalized, but unknown labels, reversed intervals,
    phase regressions, and structurally invalid JSON are rejected.
    """

    try:
        payload = _extract_json(text)
        rows = payload.get("segments")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
            raise ValueError("segments must contain one or two items")

        parsed: list[Segment] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("segment is not an object")
            phase = _canonical_phase(row.get("phase"))
            start = float(row.get("start_seconds"))
            end = float(row.get("end_seconds"))
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("segment interval is not finite and positive")
            parsed.append(Segment(phase, start, end))

        parsed.sort(key=lambda item: item.start_seconds)
        if any(
            PHASE_INDEX[left.phase] > PHASE_INDEX[right.phase]
            for left, right in zip(parsed, parsed[1:])
        ):
            raise ValueError("phase order regresses")

        if abs(parsed[0].start_seconds) > 1.0:
            raise ValueError("first segment does not start at zero")
        if abs(parsed[-1].end_seconds - duration_seconds) > 1.0:
            raise ValueError("last segment does not end at clip duration")

        if len(parsed) == 2:
            left, right = parsed
            if abs(left.end_seconds - right.start_seconds) > 1.0:
                raise ValueError("segments contain a gap or overlap")
            boundary = max(0.0, min(duration_seconds, (left.end_seconds + right.start_seconds) / 2))
            if boundary <= 0 or boundary >= duration_seconds:
                raise ValueError("transition boundary is outside the clip")
            parsed = [
                Segment(left.phase, 0.0, boundary),
                Segment(right.phase, boundary, duration_seconds),
            ]
            if parsed[0].phase == parsed[1].phase:
                parsed = [Segment(parsed[0].phase, 0.0, duration_seconds)]
        else:
            parsed = [Segment(parsed[0].phase, 0.0, duration_seconds)]

        return ParsedPrediction(True, tuple(parsed), text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ParsedPrediction(False, (), text, str(exc))


# Prompt shared by every model, training and inference (QwenVideoSegment JSON contract).
def prompt(duration_seconds: float) -> str:
    duration = int(round(duration_seconds))
    return (
        f"<video>\nSegment this {duration}-second ESD clip into one or two "
        "chronologically ordered procedural phases. Allowed phases: Diagnosis, "
        "Local injection, Incision / Dissection, Hemostasis. Cover the complete "
        "clip without gaps or overlaps. Return only a valid JSON object with the "
        'key "segments"; every segment must contain exactly "phase", '
        '"start_seconds", and "end_seconds".'
    )
