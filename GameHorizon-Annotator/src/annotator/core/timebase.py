"""Normalize annotation timestamps into video-relative presentation timestamps.

The delivered game pipelines do not all use the same clock.  This module is
the single conversion point used before Stage 1 and again by Stage 1c, so
model evidence frames and canonical manifest frames always seek to the same
video positions.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .schema import NS_PER_S, atomic_write_json


# A video-relative position this large is almost certainly a Unix-epoch value.
ABSOLUTE_TIMESTAMP_MIN_NS = 100_000_000_000_000_000
# MKV creation_time is container metadata and may be rounded slightly later than
# the first captured annotation. Accept at most two sampled frames of skew, with
# a hard 100 ms ceiling (the benchmark converter samples screen events at 20 Hz).
MAX_NEGATIVE_VIDEO_POSITION_TOLERANCE_NS = 100_000_000


@dataclass(frozen=True)
class TimebaseAudit:
    conversion_source: str
    fallback_source: str | None
    video_start_ns: int | None
    video_start_source: str | None
    scene_start_offset_ns: int | None
    scene_start_offset_source: str | None
    fps: float
    fps_source: str
    preserved_video_pts_count: int
    filled_video_pts_count: int
    unresolved_video_pts_count: int
    synthesized_timestamp_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_fps(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return resolved


def _frame_number(frame: Mapping[str, Any], position: int) -> int:
    value = _integer(frame.get("frame_index"))
    return value if value is not None else position


def infer_annotation_fps(frames: Sequence[Mapping[str, Any]]) -> float | None:
    """Infer FPS while accounting for non-contiguous frame indices."""

    estimates: list[float] = []
    previous: tuple[int, int] | None = None
    for position, frame in enumerate(frames[: min(len(frames), 5000)]):
        timestamp = _integer(frame.get("timestamp_ns"))
        if timestamp is None:
            continue
        frame_number = _frame_number(frame, position)
        if previous is not None:
            frame_delta = frame_number - previous[0]
            timestamp_delta = timestamp - previous[1]
            if frame_delta > 0 and timestamp_delta > 0:
                estimates.append(frame_delta * NS_PER_S / timestamp_delta)
        previous = (frame_number, timestamp)
    return float(median(estimates)) if estimates else None


def probe_video_creation_time_ns(video: Path) -> int | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=creation_time",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * NS_PER_S)


def _metadata_integer(metadata: Mapping[str, Any], key: str) -> int | None:
    value = _integer(metadata.get(key))
    return value


def _resolve_fps(
    frames: Sequence[Mapping[str, Any]],
    *,
    explicit_fps: float | None,
    fallback_fps: float | None,
) -> tuple[float, str]:
    explicit = _positive_fps(explicit_fps, label="--fps")
    if explicit is not None:
        return explicit, "explicit"
    inferred = infer_annotation_fps(frames)
    if inferred is not None:
        return inferred, "timestamp_intervals"
    fallback = _positive_fps(fallback_fps, label="profile fallback_fps")
    if fallback is not None:
        return fallback, "profile_fallback"
    raise ValueError(
        "annotations have no usable time intervals; provide --fps or configure fallback_fps"
    )


def _resolve_video_start(
    metadata: Mapping[str, Any],
    timestamps: Sequence[int | None],
    *,
    explicit: int | None,
    video: Path | None,
) -> tuple[int | None, str | None]:
    if explicit is not None:
        value = _integer(explicit)
        if value is None or value < 0:
            raise ValueError("--video-start-ns must be a non-negative integer")
        return value, "explicit"
    metadata_value = _metadata_integer(metadata, "video_start_ns")
    if metadata_value is not None:
        if metadata_value < 0:
            raise ValueError("annotations video_start_ns must be non-negative")
        return metadata_value, "annotations.video_start_ns"
    has_absolute_timestamp = any(
        value is not None and value >= ABSOLUTE_TIMESTAMP_MIN_NS for value in timestamps
    )
    if has_absolute_timestamp and video is not None:
        probed = probe_video_creation_time_ns(video)
        if probed is not None:
            return probed, "video.creation_time"
    return None, None


def _resolve_scene_offset(
    metadata: Mapping[str, Any], explicit: int | None
) -> tuple[int | None, str | None]:
    candidates = (
        (explicit, "explicit"),
        (metadata.get("scene_start_offset_ns"), "annotations.scene_start_offset_ns"),
        (metadata.get("interval_start_ns"), "annotations.interval_start_ns"),
    )
    for raw, source in candidates:
        value = _integer(raw)
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"{source} must be non-negative")
        # interval_start_ns is sometimes an epoch timestamp.  It cannot safely
        # be interpreted as a video offset in that case.
        if value >= ABSOLUTE_TIMESTAMP_MIN_NS:
            if source == "annotations.interval_start_ns":
                continue
            raise ValueError(f"{source} looks like an absolute timestamp, not a video offset")
        return value, source
    return None, None


def _normalized_document(document: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(document, dict):
        output = copy.deepcopy(document)
        raw_frames = output.get("frames")
    elif isinstance(document, list):
        raw_frames = copy.deepcopy(document)
        output = {"frames": raw_frames}
    else:
        raise ValueError("annotations JSON must be an object or a frame list")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("annotations must contain a non-empty frames list")
    if not all(isinstance(frame, dict) for frame in raw_frames):
        raise ValueError("every annotation frame must be an object")
    return output, raw_frames


def normalize_annotation_timebase(
    document: Any,
    *,
    video_start_ns: int | None = None,
    scene_start_offset_ns: int | None = None,
    fps: float | None = None,
    fallback_fps: float | None = None,
    video: Path | None = None,
    allow_unresolved_video_positions: bool = False,
) -> tuple[dict[str, Any], TimebaseAudit]:
    """Return a copy whose frames all have trustworthy ``video_pts_ns``."""

    output, frames = _normalized_document(document)
    resolved_fps, fps_source = _resolve_fps(
        frames, explicit_fps=fps, fallback_fps=fallback_fps
    )

    frame_numbers = [_frame_number(frame, position) for position, frame in enumerate(frames)]
    if any(right <= left for left, right in zip(frame_numbers, frame_numbers[1:])):
        raise ValueError("annotation frame_index values must be strictly increasing")

    raw_timestamps = [_integer(frame.get("timestamp_ns")) for frame in frames]
    synthesized_timestamp_count = sum(value is None for value in raw_timestamps)
    if synthesized_timestamp_count:
        anchor_position = next(
            (index for index, value in enumerate(raw_timestamps) if value is not None), 0
        )
        anchor_timestamp = raw_timestamps[anchor_position] or 0
        anchor_frame = frame_numbers[anchor_position]
        for index, value in enumerate(raw_timestamps):
            if value is None:
                frame_delta = frame_numbers[index] - anchor_frame
                raw_timestamps[index] = anchor_timestamp + round(
                    frame_delta * NS_PER_S / resolved_fps
                )
                frames[index]["timestamp_ns"] = raw_timestamps[index]

    timestamps = [int(value) for value in raw_timestamps if value is not None]
    if len(timestamps) != len(frames):
        raise ValueError("could not synthesize all timestamp_ns values")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("annotation timestamp_ns values must be strictly increasing")

    resolved_video_start, video_start_source = _resolve_video_start(
        output,
        timestamps,
        explicit=video_start_ns,
        video=video,
    )
    resolved_scene_offset, scene_offset_source = _resolve_scene_offset(
        output, scene_start_offset_ns
    )

    original_pts: list[int | None] = []
    for index, frame in enumerate(frames):
        raw = frame.get("video_pts_ns")
        value = _integer(raw)
        if raw is not None and (value is None or value < 0):
            raise ValueError(f"frame {index} has invalid video_pts_ns")
        original_pts.append(value)

    missing_count = sum(value is None for value in original_pts)
    fallback_source: str | None = None
    if missing_count:
        if resolved_video_start is not None:
            if synthesized_timestamp_count:
                raise ValueError(
                    "--video-start-ns cannot locate frames without timestamps; "
                    "also provide --scene-start-offset-ns"
                )
            fallback_source = "timestamp_minus_video_start"
        elif resolved_scene_offset is not None:
            fallback_source = (
                "scene_offset_plus_fps"
                if synthesized_timestamp_count
                else "scene_offset_plus_timestamp_delta"
            )
        elif synthesized_timestamp_count:
            raise ValueError(
                "frames have no timestamps or video_pts_ns; provide --scene-start-offset-ns "
                "and --fps"
            )
        elif max(timestamps) < ABSOLUTE_TIMESTAMP_MIN_NS:
            fallback_source = "relative_timestamp"
        elif not allow_unresolved_video_positions:
            raise ValueError(
                "absolute annotation timestamps have no video origin; provide "
                "--video-start-ns or --scene-start-offset-ns"
            )

    first_timestamp = timestamps[0]
    first_frame_number = frame_numbers[0]
    normalized_pts: list[int] = []
    unresolved_count = 0
    for index, original in enumerate(original_pts):
        if original is not None:
            position = original
        elif fallback_source == "timestamp_minus_video_start":
            position = timestamps[index] - int(resolved_video_start)
        elif fallback_source == "scene_offset_plus_timestamp_delta":
            position = int(resolved_scene_offset) + timestamps[index] - first_timestamp
        elif fallback_source == "scene_offset_plus_fps":
            position = int(resolved_scene_offset) + round(
                (frame_numbers[index] - first_frame_number) * NS_PER_S / resolved_fps
            )
        elif fallback_source == "relative_timestamp":
            position = timestamps[index]
        elif allow_unresolved_video_positions:
            unresolved_count += 1
            frames[index].pop("video_pts_ns", None)
            continue
        else:
            raise AssertionError("missing PTS without a resolved fallback")
        if position < 0:
            negative_tolerance_ns = min(
                MAX_NEGATIVE_VIDEO_POSITION_TOLERANCE_NS,
                round(2 * NS_PER_S / resolved_fps),
            )
            if (
                fallback_source == "timestamp_minus_video_start"
                and position >= -negative_tolerance_ns
            ):
                position = 0
            else:
                raise ValueError(
                    f"frame {index} resolves to negative video position {position} ns; "
                    "check --video-start-ns and --scene-start-offset-ns"
                )
        if position >= ABSOLUTE_TIMESTAMP_MIN_NS:
            raise ValueError(
                f"frame {index} resolves to {position} ns, which looks like an absolute "
                "timestamp rather than a video position"
            )
        normalized_pts.append(position)
        frame = frames[index]
        frame["video_pts_ns"] = position
        if index + 1 < len(frames):
            current_next = _integer(frame.get("next_frame_timestamp_ns"))
            if current_next is None or current_next <= timestamps[index]:
                frame["next_frame_timestamp_ns"] = timestamps[index + 1]
        elif _integer(frame.get("next_frame_timestamp_ns")) is None:
            frame["next_frame_timestamp_ns"] = timestamps[index] + round(
                NS_PER_S / resolved_fps
            )

    if any(right < left for left, right in zip(normalized_pts, normalized_pts[1:])):
        raise ValueError("resolved video_pts_ns values are not monotonic")

    preserved_count = len(frames) - missing_count
    if unresolved_count:
        conversion_source = "unresolved_external_frames_only"
    else:
        conversion_source = (
            "exact_video_pts"
            if missing_count == 0
            else fallback_source
            if preserved_count == 0
            else "mixed_exact_and_fallback"
        )
    audit = TimebaseAudit(
        conversion_source=str(conversion_source),
        fallback_source=fallback_source,
        video_start_ns=resolved_video_start,
        video_start_source=video_start_source,
        scene_start_offset_ns=resolved_scene_offset,
        scene_start_offset_source=scene_offset_source,
        fps=resolved_fps,
        fps_source=fps_source,
        preserved_video_pts_count=preserved_count,
        filled_video_pts_count=missing_count - unresolved_count,
        unresolved_video_pts_count=unresolved_count,
        synthesized_timestamp_count=synthesized_timestamp_count,
    )
    output["frames"] = frames
    output["num_frames"] = len(frames)
    output["timebase_normalization"] = audit.as_dict()
    return output, audit


def normalize_annotation_file(
    source: Path,
    destination: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], TimebaseAudit]:
    document = json.loads(source.read_text(encoding="utf-8"))
    normalized, audit = normalize_annotation_timebase(document, **kwargs)
    normalized["source_annotations"] = str(source.resolve())
    atomic_write_json(destination, normalized)
    return normalized, audit
