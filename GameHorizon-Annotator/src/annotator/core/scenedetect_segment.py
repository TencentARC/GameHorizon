"""Generic PySceneDetect backend for Stage 1a.

The per-game action segmenters in ``game_assets/<game>/1a_*.py`` cut the
timeline on keybind semantics and label every span with a taxonomy leaf. That
needs a hand-written taxonomy plus tuned hold/lock thresholds, so it only
exists for the games shipped here.

This backend trades labels for portability: boundaries come from visual shot
changes, so it runs without game-specific assets. Every span is emitted as
``unknown`` with ``vision_needed=True`` - the same contract the action
segmenter uses for spans it cannot resolve - so Stage 1b assigns the taxonomy
leaf from vision afterwards.

The output is a drop-in replacement for the action segmenter's
``*_segments.json``.
"""

from __future__ import annotations

import bisect
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .frames import infer_fps, load_annotation_frames
from .profiles import GameProfile
from .schema import NS_PER_S, atomic_write_json
from .timebase import probe_video_creation_time_ns

UNKNOWN_LABEL = "unknown"

# ContentDetector sensitivity carried over from the frame-space prototype these
# profiles were tuned against; lower cuts more aggressively.
DEFAULT_THRESHOLD = 27.0
DEFAULT_MIN_DURATION_S = 1.0


@dataclass
class SceneDetectConfig:
    threshold: float = DEFAULT_THRESHOLD
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    min_scene_frames: int | None = None
    frame_skip: int = 0
    backend: str = "opencv"

    def resolved_min_scene_frames(self, fps: float) -> int:
        """Detector-side floor on scene length, in video frames.

        Defaults to one second of video so the detector itself suppresses the
        sub-second flicker that gunfire and cutscene flashes produce.
        """

        if self.min_scene_frames is not None:
            return max(1, self.min_scene_frames)
        return max(1, int(round(fps * DEFAULT_MIN_DURATION_S)))

    def as_dict(self, *, min_scene_frames: int, fps: float) -> Dict[str, Any]:
        return {
            "segmenter": "scenedetect",
            "detector": "ContentDetector",
            "threshold": self.threshold,
            "min_scene_frames": min_scene_frames,
            "min_duration_s": self.min_duration_s,
            "frame_skip": self.frame_skip,
            "video_backend": self.backend,
            "annotation_fps_estimate": round(fps, 3),
        }


def _unknown_leaf(profile: GameProfile) -> Any | None:
    """Load the taxonomy's ``unknown`` leaf so tier/vision hints match Stage 1b."""

    module_name = f"scenedetect_taxonomy_{profile.game_id}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, profile.taxonomy_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, "LEAF_BY_LABEL", {}).get(UNKNOWN_LABEL)


def detect_scene_cuts(
    video: Path,
    *,
    config: SceneDetectConfig,
    min_scene_frames: int,
    start_s: float | None = None,
    end_s: float | None = None,
) -> List[float]:
    """Return shot-change times in video seconds, restricted to [start_s, end_s].

    Scanning only the annotated span matters here: the recordings are whole
    sessions while the annotations cover one scene, so detecting over the full
    file would decode hours of video to reach a few minutes of interest.
    """

    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "PySceneDetect is required for --segmenter scenedetect; install it with:\n"
            "  pip install 'GameHorizon-Annotator[scenedetect]'"
        ) from exc

    stream = open_video(str(video), backend=config.backend)
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=config.threshold, min_scene_len=min_scene_frames)
    )
    if start_s is not None and start_s > 0:
        stream.seek(start_s)
    manager.detect_scenes(
        stream, end_time=end_s, frame_skip=config.frame_skip, show_progress=False
    )

    cuts: set[float] = set()
    for scene_start, scene_end in manager.get_scene_list():
        cuts.add(round(scene_start.get_seconds(), 4))
        cuts.add(round(scene_end.get_seconds(), 4))
    return sorted(cuts)


def _nearest_index(timestamp_ns: int, timestamps: Sequence[int], indices: Sequence[int]) -> int:
    position = bisect.bisect_left(timestamps, timestamp_ns)
    if position <= 0:
        return indices[0]
    if position >= len(timestamps):
        return indices[-1]
    before, after = timestamps[position - 1], timestamps[position]
    if timestamp_ns - before <= after - timestamp_ns:
        return indices[position - 1]
    return indices[position]


def _boundaries_from_cuts(
    cuts: Sequence[float],
    *,
    video_start_ns: int,
    timestamps: Sequence[int],
    indices: Sequence[int],
) -> List[int]:
    """Map cut times onto the annotation frame indices that start a segment."""

    first, last = indices[0], indices[-1]
    starts = {first}
    for cut in cuts:
        cut_ns = video_start_ns + int(round(cut * NS_PER_S))
        if cut_ns <= timestamps[0] or cut_ns >= timestamps[-1]:
            continue
        index = _nearest_index(cut_ns, timestamps, indices)
        if first < index <= last:
            starts.add(index)
    return sorted(starts)


def _drop_short_spans(
    starts: Sequence[int],
    last_index: int,
    *,
    frame_by_index: Mapping[int, Mapping[str, Any]],
    min_duration_s: float,
) -> List[int]:
    """Drop boundaries that would create spans below ``min_duration_s``.

    The detector's own ``min_scene_len`` counts video frames, which only
    matches wall-clock when capture is uniform; these recordings are not. This
    second pass enforces the floor against annotation timestamps instead.
    Dropping a start merges that span into the preceding one, so coverage
    stays gap-free.
    """

    if min_duration_s <= 0 or len(starts) <= 1:
        return list(starts)

    def stamp(index: int) -> int | None:
        frame = frame_by_index.get(index)
        if frame is None:
            return None
        try:
            return int(frame["timestamp_ns"])
        except (KeyError, TypeError, ValueError):
            return None

    kept = [starts[0]]
    for start in starts[1:]:
        begin, here = stamp(kept[-1]), stamp(start)
        if begin is None or here is None or (here - begin) / NS_PER_S >= min_duration_s:
            kept.append(start)

    # The tail span is bounded by the final frame rather than a following
    # start, so it needs the same check.
    while len(kept) > 1:
        begin, end = stamp(kept[-1]), stamp(last_index)
        if begin is None or end is None or (end - begin) / NS_PER_S >= min_duration_s:
            break
        kept.pop()
    return kept


def _action_stats(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarise raw input over a span, mirroring the action segmenter's fields.

    Purely descriptive - no taxonomy decision is made here - but Stage 1b and
    the Level-1 prompt both read these, so they must be populated.
    """

    keys: set[str] = set()
    buttons: set[str] = set()
    wheel = 0
    motion = 0.0
    for frame in frames:
        for event in frame.get("keyboard_events") or ():
            name = event.get("key_name")
            if name:
                keys.add(str(name))
        for event in frame.get("mouse_events") or ():
            kind = event.get("event_type")
            if kind == "click":
                button = event.get("button")
                if button:
                    buttons.add(str(button))
            elif kind == "scroll":
                wheel += 1
        for event in frame.get("raw_mouse_events") or ():
            try:
                motion += math.hypot(float(event.get("dx", 0)), float(event.get("dy", 0)))
            except (TypeError, ValueError):
                continue
    return {
        "keys_pressed": sorted(keys),
        "mouse_buttons_pressed": sorted(buttons),
        "mouse_motion_total": round(motion, 1),
        "wheel_events": wheel,
    }


def _exclusive_end_ns(frame: Mapping[str, Any]) -> int:
    """End timestamp of a span, matching the action segmenter's convention.

    The action segmenters close a segment at the *next* frame's timestamp, so
    adjacent durations tile the timeline exactly. Levels 2 and 3 size their
    windows by summing ``duration_s``, so using the final frame's own stamp
    here would quietly shorten every span by one frame.
    """

    for key in ("next_frame_timestamp_ns", "timestamp_ns"):
        try:
            value = frame[key]
        except KeyError:
            continue
        if value is not None:
            return int(value)
    raise ValueError("annotation frame has no usable timestamp")


def build_segments(
    starts: Sequence[int],
    last_index: int,
    *,
    frame_by_index: Mapping[int, Mapping[str, Any]],
    leaf: Any | None,
) -> List[Dict[str, Any]]:
    ordered = sorted(frame_by_index)
    segments: List[Dict[str, Any]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] - 1 if position + 1 < len(starts) else last_index
        span = [index for index in ordered if start <= index <= end]
        if not span:
            continue
        frames = [frame_by_index[index] for index in span]
        start_ns = int(frames[0]["timestamp_ns"])
        end_ns = _exclusive_end_ns(frames[-1])
        segments.append(
            {
                "start_frame": span[0],
                "end_frame": span[-1],
                "num_frames": len(span),
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
                "duration_s": round((end_ns - start_ns) / NS_PER_S, 3),
                "label": UNKNOWN_LABEL,
                "tier1": getattr(leaf, "tier1", None),
                "tier2": getattr(leaf, "tier2", None),
                "action_distinguishable": getattr(leaf, "action_distinguishable", "n/a"),
                "vision_needed": True,
                "vision_subtypes": list(getattr(leaf, "vision_subtypes", []) or []),
                **_action_stats(frames),
                "frame_file_start": None,
                "frame_file_end": None,
                "sub_segments": [],
            }
        )
    return segments


def run(
    *,
    profile: GameProfile,
    annotations: Path,
    video: Path,
    out: Path,
    video_name: str,
    config: SceneDetectConfig,
    video_start_ns: int | None = None,
) -> Dict[str, Any]:
    meta, frames, frame_by_index = load_annotation_frames(annotations)
    if not frame_by_index:
        raise SystemExit(f"annotations contain no frames: {annotations}")

    indices = sorted(frame_by_index)
    timestamps = [int(frame_by_index[index]["timestamp_ns"]) for index in indices]
    fps = infer_fps(frames, 30.0)

    if video_start_ns is None:
        video_start_ns = probe_video_creation_time_ns(video)
    if video_start_ns is None:
        raise SystemExit(
            f"cannot resolve the video start timestamp for {video}; "
            "pass --video-start-ns so shot changes can be mapped onto annotation frames"
        )

    start_s = max(0.0, (timestamps[0] - video_start_ns) / NS_PER_S)
    end_s = (timestamps[-1] - video_start_ns) / NS_PER_S
    min_scene_frames = config.resolved_min_scene_frames(fps)
    print(
        f"[scenedetect] video_start_ns={video_start_ns} scan={start_s:.2f}s..{end_s:.2f}s "
        f"threshold={config.threshold} min_scene_frames={min_scene_frames}",
        flush=True,
    )

    cuts = detect_scene_cuts(
        video,
        config=config,
        min_scene_frames=min_scene_frames,
        start_s=start_s,
        end_s=end_s,
    )
    starts = _boundaries_from_cuts(
        cuts, video_start_ns=video_start_ns, timestamps=timestamps, indices=indices
    )
    kept = _drop_short_spans(
        starts,
        indices[-1],
        frame_by_index=frame_by_index,
        min_duration_s=config.min_duration_s,
    )
    segments = build_segments(
        kept, indices[-1], frame_by_index=frame_by_index, leaf=_unknown_leaf(profile)
    )
    total_s = sum(segment["duration_s"] for segment in segments)
    payload = {
        "video": video_name,
        "scene_name": meta.get("scene_name"),
        "num_frames": len(frames),
        "fps_estimate": round(fps, 3),
        "interval_start_ns": meta.get("interval_start_ns", timestamps[0]),
        "interval_end_ns": meta.get("interval_end_ns", timestamps[-1]),
        "config": config.as_dict(min_scene_frames=min_scene_frames, fps=fps),
        "label_counts": {
            UNKNOWN_LABEL: {"segments": len(segments), "total_s": round(total_s, 2)}
        },
        "segments": segments,
        "transitions": [],
        "scenedetect": {
            "video_start_ns": video_start_ns,
            "scan_start_s": round(start_s, 3),
            "scan_end_s": round(end_s, 3),
            "detected_cuts": len(cuts),
            "boundaries_in_scene": len(starts),
            "boundaries_after_min_duration": len(kept),
        },
    }
    atomic_write_json(out, payload)
    print(
        f"[scenedetect] cuts={len(cuts)} boundaries={len(starts)} segments={len(segments)} "
        f"min_duration_s={config.min_duration_s} -> {out}",
        flush=True,
    )
    return payload
