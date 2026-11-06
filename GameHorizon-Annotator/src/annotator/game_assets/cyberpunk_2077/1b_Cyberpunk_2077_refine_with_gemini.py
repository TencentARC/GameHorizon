"""Stage 1b: Gemini-vision refinement of action-segmented Cyberpunk 2077 clips.

Reads a segments JSON produced by
``1a_Cyberpunk_2077_scripts_segment_from_actions.py``, selects segments where
``vision_needed=True`` (for example ``unknown`` or parent buckets such as
``interact`` / ``use_consumable`` / ``vehicle_combat``), samples frames from
the source video, and asks Gemini to choose a more accurate leaf label from the
Cyberpunk 2077 taxonomy.

The result is written back into the segments JSON with new fields:

* ``refined_label`` - the chosen leaf label
* ``refined_tier1`` / ``refined_tier2`` - re-derived from taxonomy
* ``refined_confidence`` - 0..1 from the model
* ``refined_reason`` - short explanation
* ``refined_via`` - model id

For ``unknown`` segments, the default behavior mirrors the Apex pipeline: ask
Gemini to subdivide the segment into one or more contiguous sub-events, then
optionally merge adjacent same-label / same-group results.

CLI::

    python 1b_Cyberpunk_2077_refine_with_gemini.py scene_segments.json \\
        --video scene_video_annotation/recording.mkv \\
        --out scene_segments_refined.json \\
        --frames-per-segment 5 --subdivide-frames-per-segment 10
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Cyberpunk_2077_taxonomy import (
    LEAF_BY_LABEL,
    LEAVES,
    format_taxonomy_for_prompt,
    normalize_label,
    subtype_choices,
    vision_needed as taxonomy_vision_needed,
)

try:
    from Cyberpunk_2077_video_io import (
        JsonCache,
        NS_PER_S,
        build_image_content,
        extract_frame_jpeg,
        extract_json_object,
        gemini_client,
        sample_segment_timestamps,
        with_retry,
    )
except ModuleNotFoundError:
    # Reuse the game-agnostic helper module already present in the Apex
    # pipeline until/unless this project grows a Cyberpunk-local copy.
    apex_dir = (
        Path(__file__).resolve().parents[1]
        / "Apex_Legends_scripts_parrellel"
    )
    sys.path.append(str(apex_dir))
    from Apex_Legends_video_io import (  # type: ignore[no-redef]
        JsonCache,
        NS_PER_S,
        build_image_content,
        extract_frame_jpeg,
        extract_json_object,
        gemini_client,
        sample_segment_timestamps,
        with_retry,
    )


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_S = 300.0
PROMPT_VERSION = 1


def _probe_video_start_ns(video_path: Path) -> Optional[int]:
    """Return the MKV creation time used as the epoch-to-video offset."""
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
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
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


def _resolve_video_start_ns(
    video_path: Path, segments: List[Dict[str, Any]]
) -> int:
    """Return zero for relative timestamps, otherwise the MKV epoch origin."""
    origin_ns = _probe_video_start_ns(video_path) or 0
    if origin_ns and segments:
        max_end_ns = max(
            int(segment.get("end_timestamp_ns") or 0)
            for segment in segments
        )
        if max_end_ns < origin_ns * 0.5:
            return 0
    return origin_ns


_KEYBINDS_JSON_PATH = Path(__file__).resolve().parent / "Cyberpunk_2077_keybinds.json"
_PAREN_EN_RE = re.compile(r"\(([^)]+)\)")


def _load_keybinds_english() -> Dict[str, str]:
    """Read Cyberpunk keybinds as ``raw event name -> English action``."""
    try:
        raw = json.loads(_KEYBINDS_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            continue
        # Compatible with Apex-style bilingual values, but Cyberpunk values are
        # already English.
        match = _PAREN_EN_RE.search(value)
        out[key] = (match.group(1) if match else value).strip()
    return out


KEYBINDS_EN: Dict[str, str] = _load_keybinds_english()


def format_keybinds_for_prompt() -> str:
    if not KEYBINDS_EN:
        return ""
    lines = ["Cyberpunk 2077 key bindings (raw event name -> in-game action):"]
    for key, value in KEYBINDS_EN.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


KEYBINDS_REFERENCE_TEXT = format_keybinds_for_prompt()


def all_non_idle_atomic_labels() -> List[str]:
    """All taxonomy leaves except rule-only idle labels."""
    seen: set[str] = set()
    out: List[str] = []
    for leaf in LEAVES:
        label = leaf.label
        if label.startswith("idle"):
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    if "unknown" not in seen:
        out.append("unknown")
    return out


UNKNOWN_FULL_CANDIDATES: List[str] = all_non_idle_atomic_labels()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    keybind_block = (
        f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    )
    return (
        "You are a fine-grained event classifier for short Cyberpunk 2077 "
        "first-person gameplay clips.\n"
        "You will see 1-10 still frames sampled from a clip plus a short JSON "
        "description of what a rule-based action segmenter inferred from "
        "keyboard / mouse alone. The `keys_pressed` and "
        "`mouse_buttons_pressed` fields use raw RawInput names; map them to "
        "in-game actions using the keybind reference below.\n"
        "Your job is to pick the SINGLE most accurate leaf label from the "
        "candidate list. Use vision to disambiguate context that raw input "
        "cannot know, especially on-foot vs vehicle, braindance, menus, "
        "dialogue, and combat vs scanner overlays. If none of the candidates "
        "fit, return the parent label from `parent_label`.\n\n"
        f"{keybind_block}"
        "Cyberpunk 2077 3-tier taxonomy (descriptions included):\n"
        f"{format_taxonomy_for_prompt(include_descriptions=True)}\n\n"
        "Output STRICTLY a single JSON object with keys:\n"
        '  "label":      one of the candidates listed in the user message\n'
        '  "confidence": float in [0,1]\n'
        '  "reason":     one short clause (<= 20 words)\n'
        "Do not include any other text."
    )


def build_user_prompt(seg: Dict[str, Any], candidates: List[str]) -> str:
    parent = seg.get("label", "unknown")
    fields = {
        "parent_label": parent,
        "candidates": candidates,
        "duration_s": seg.get("duration_s"),
        "tier1": seg.get("tier1"),
        "tier2": seg.get("tier2"),
        "keys_pressed": seg.get("keys_pressed", []),
        "mouse_buttons_pressed": seg.get("mouse_buttons_pressed", []),
        "wheel_events": seg.get("wheel_events", 0),
        "horizontal_wheel_events": seg.get("horizontal_wheel_events", 0),
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    extra = ""
    if parent == "unknown":
        extra = (
            "\nIMPORTANT: do NOT return any `idle_*` label. Idle is handled "
            "by the rule classifier upstream. If the clip truly has no "
            "recognizable event, return `unknown`."
        )
    return (
        "Pick the most appropriate label for this clip from `candidates`. "
        "If none clearly fit, return `parent_label`."
        f"{extra}\n\n"
        f"Segment metadata:\n```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        "Frames are attached below in temporal order (first -> last)."
    )


def build_subdivide_system_prompt() -> str:
    keybind_block = (
        f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    )
    return (
        "You are a fine-grained event SEGMENTER for short Cyberpunk 2077 "
        "gameplay clips.\n"
        "You will see K still frames sampled in temporal order from a segment "
        "that the upstream rule classifier marked as `unknown`. You will also "
        "receive aggregated keyboard / mouse evidence.\n\n"
        "Task: walk through the K frames in order and decide which candidate "
        "event each frame belongs to. Group consecutive frames of the same "
        "event into one `sub_event`. The clip may contain one event or multiple "
        "events in sequence.\n\n"
        "STRICT rules:\n"
        "1. Sub-events must be contiguous and together cover frames 0..K-1.\n"
        "2. Labels MUST come from `candidates`. If no candidate fits a range, "
        "label it `unknown`.\n"
        "3. Do NOT return any `idle_*` label.\n"
        "4. Treat visible vehicle riding/driving as vehicle context even if "
        "the raw keys look like on-foot WASD movement.\n"
        "5. Only split when the visible event class actually changes.\n\n"
        f"{keybind_block}"
        "Cyberpunk 2077 3-tier taxonomy (descriptions included):\n"
        f"{format_taxonomy_for_prompt(include_descriptions=True)}\n\n"
        "Output STRICTLY a single JSON object:\n"
        '{\n'
        '  "sub_events": [\n'
        '    {\n'
        '      "start_frame_idx": int (0..K-1, inclusive),\n'
        '      "end_frame_idx":   int (0..K-1, inclusive, >= start_frame_idx),\n'
        '      "label":           one of `candidates`,\n'
        '      "confidence":      float in [0,1],\n'
        '      "reason":          short clause (<= 20 words)\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Do not include any other text."
    )


def build_subdivide_user_prompt(
    seg: Dict[str, Any], candidates: List[str], frame_count: int
) -> str:
    fields = {
        "K": frame_count,
        "parent_label": "unknown",
        "candidates": candidates,
        "duration_s": seg.get("duration_s"),
        "tier1": seg.get("tier1"),
        "tier2": seg.get("tier2"),
        "keys_pressed": seg.get("keys_pressed", []),
        "mouse_buttons_pressed": seg.get("mouse_buttons_pressed", []),
        "wheel_events": seg.get("wheel_events", 0),
        "horizontal_wheel_events": seg.get("horizontal_wheel_events", 0),
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    return (
        f"This is an `unknown` clip. Walk through the {frame_count} attached "
        f"frames (indices 0..{frame_count - 1}) and output `sub_events` "
        "covering all of them. Do NOT return any `idle_*` label.\n\n"
        "Segment metadata (aggregated over the whole clip):\n"
        f"```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        f"{frame_count} frames are attached below in temporal order."
    )


def _normalize_sub_events(
    raw: Any, candidates: List[str], frame_count: int
) -> List[Dict[str, Any]]:
    fallback = [{
        "start_frame_idx": 0,
        "end_frame_idx": frame_count - 1,
        "label": "unknown",
        "confidence": None,
        "reason": "could not parse sub_events response",
    }]
    if not isinstance(raw, list) or not raw:
        return fallback

    candidate_set = set(candidates)
    parsed: List[Dict[str, Any]] = []
    for event in raw:
        if not isinstance(event, dict):
            continue
        try:
            start_idx = int(event.get("start_frame_idx"))
            end_idx = int(event.get("end_frame_idx"))
        except (TypeError, ValueError):
            continue
        start_idx = max(0, min(start_idx, frame_count - 1))
        end_idx = max(start_idx, min(end_idx, frame_count - 1))
        raw_label = str(event.get("label", "")).strip()
        label = raw_label if raw_label in candidate_set else normalize_label(raw_label)
        if label not in candidate_set:
            label = "unknown"
        try:
            confidence = float(event["confidence"]) if "confidence" in event else None
        except (TypeError, ValueError, KeyError):
            confidence = None
        parsed.append({
            "start_frame_idx": start_idx,
            "end_frame_idx": end_idx,
            "label": label,
            "confidence": confidence,
            "reason": str(event.get("reason", ""))[:300],
        })

    if not parsed:
        return fallback

    parsed.sort(key=lambda event: (event["start_frame_idx"], event["end_frame_idx"]))
    fixed: List[Dict[str, Any]] = []
    cur_start = 0
    for i, event in enumerate(parsed):
        start_idx = max(cur_start, event["start_frame_idx"])
        if i + 1 < len(parsed):
            next_start = max(parsed[i + 1]["start_frame_idx"], start_idx + 1)
            end_idx = min(event["end_frame_idx"], next_start - 1)
        else:
            end_idx = frame_count - 1
        end_idx = max(end_idx, start_idx)
        fixed.append({**event, "start_frame_idx": start_idx, "end_frame_idx": end_idx})
        cur_start = end_idx + 1
        if cur_start > frame_count - 1:
            break

    if not fixed:
        return fallback
    fixed[0]["start_frame_idx"] = 0
    fixed[-1]["end_frame_idx"] = frame_count - 1

    merged: List[Dict[str, Any]] = [fixed[0]]
    for event in fixed[1:]:
        last = merged[-1]
        if event["label"] == last["label"]:
            last["end_frame_idx"] = event["end_frame_idx"]
            if (event.get("confidence") or 0.0) > (last.get("confidence") or 0.0):
                last["confidence"] = event.get("confidence")
                last["reason"] = event.get("reason", "")
        else:
            merged.append(event)
    return merged


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------


@with_retry(max_attempts=4, initial_delay_s=1.5, max_delay_s=20.0)
def _call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    jpeg_blobs: List[bytes],
    image_detail: str = "low",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    content = build_image_content(user_prompt, jpeg_blobs, detail=image_detail)
    content = [{"type": "text", "text": system_prompt}] + content
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        timeout=timeout_s,
    )
    raw = response.choices[0].message.content or ""
    return extract_json_object(raw)


def refine_segment(
    seg: Dict[str, Any],
    *,
    video_path: Path,
    client,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    system_prompt: str,
    video_start_ns: int,
) -> Dict[str, Any]:
    parent = seg.get("label", "unknown")
    strict_candidates = parent == "unknown"
    candidates = (
        list(UNKNOWN_FULL_CANDIDATES)
        if parent == "unknown"
        else list(subtype_choices(parent))
    )

    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": parent,
        "model": model,
        "k": frames_per_segment,
        "detail": image_detail,
        "stage": "cyberpunk_refine",
        "video_start_ns": video_start_ns,
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)
    if cached:
        return cached

    timestamps = sample_segment_timestamps(
        seg["start_timestamp_ns"], seg["end_timestamp_ns"], frames_per_segment
    )
    video_timestamps = [
        max(0.0, timestamp - video_start_ns / NS_PER_S)
        for timestamp in timestamps
    ]
    jpeg_blobs = [
        extract_frame_jpeg(video_path, timestamp, max_width=1024)
        for timestamp in video_timestamps
    ]
    parsed = _call_gemini(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=build_user_prompt(seg, candidates),
        jpeg_blobs=jpeg_blobs,
        image_detail=image_detail,
    )

    raw_label = str(parsed.get("label", "")).strip()
    label = raw_label if raw_label in candidates else normalize_label(raw_label)
    if strict_candidates and label not in candidates:
        label = parent
    elif label not in candidates and label not in LEAF_BY_LABEL:
        label = parent
    leaf = LEAF_BY_LABEL.get(label)
    try:
        confidence = float(parsed["confidence"]) if "confidence" in parsed else None
    except (TypeError, ValueError, KeyError):
        confidence = None

    result = {
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else seg.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else seg.get("tier2"),
        "refined_confidence": confidence,
        "refined_reason": str(parsed.get("reason", ""))[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [
            round(timestamp, 3) for timestamp in video_timestamps
        ],
    }
    cache.set(cache_key, result)
    return result


def _interp_frame_index(seg: Dict[str, Any], t_ns: int) -> int:
    start_ns = int(seg["start_timestamp_ns"])
    end_ns = int(seg["end_timestamp_ns"])
    start_frame = int(seg["start_frame"])
    end_frame = int(seg["end_frame"])
    if end_ns <= start_ns or end_frame <= start_frame:
        return start_frame
    t_ns = max(start_ns, min(t_ns, end_ns))
    frac = (t_ns - start_ns) / (end_ns - start_ns)
    return start_frame + int(round(frac * (end_frame - start_frame)))


def _build_sub_segment(
    *,
    parent: Dict[str, Any],
    sub_event: Dict[str, Any],
    sample_timestamps_s: List[float],
    next_event_start_idx: Optional[int],
    is_first: bool,
    is_last: bool,
    sub_index: int,
    sub_total: int,
    candidates: List[str],
    model: str,
) -> Dict[str, Any]:
    start_idx = int(sub_event["start_frame_idx"])
    end_idx = int(sub_event["end_frame_idx"])
    label = str(sub_event.get("label") or "unknown")

    parent_start_ns = int(parent["start_timestamp_ns"])
    parent_end_ns = int(parent["end_timestamp_ns"])
    parent_dur_ns = max(parent_end_ns - parent_start_ns, 1)

    if is_first:
        start_ns = parent_start_ns
    else:
        start_ns = max(parent_start_ns, int(round(sample_timestamps_s[start_idx] * NS_PER_S)))

    if is_last or next_event_start_idx is None:
        end_ns = parent_end_ns
    else:
        midpoint_s = (sample_timestamps_s[end_idx] + sample_timestamps_s[next_event_start_idx]) / 2.0
        end_ns = max(start_ns + 1, min(parent_end_ns - 1, int(round(midpoint_s * NS_PER_S))))

    if end_ns <= start_ns:
        end_ns = start_ns + 1

    start_frame = _interp_frame_index(parent, start_ns)
    end_frame = max(start_frame, _interp_frame_index(parent, end_ns))
    duration_ns = end_ns - start_ns
    share = duration_ns / parent_dur_ns
    leaf = LEAF_BY_LABEL.get(label)

    new_seg: Dict[str, Any] = dict(parent)
    new_seg.update({
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": max(end_frame - start_frame + 1, 1),
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "duration_s": round(duration_ns / NS_PER_S, 3),
        "label": "unknown",
        "tier1": parent.get("tier1"),
        "tier2": parent.get("tier2"),
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else parent.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else parent.get("tier2"),
        "refined_confidence": sub_event.get("confidence"),
        "refined_reason": str(sub_event.get("reason", ""))[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [
            round(sample_timestamps_s[k], 3) for k in range(start_idx, end_idx + 1)
        ],
        "subdivided_from_unknown": True,
        "sub_event_index": sub_index,
        "sub_event_total": sub_total,
        "sub_event_frame_range": [start_idx, end_idx],
        "mouse_motion_total": round(
            float(parent.get("mouse_motion_total", 0.0) or 0.0) * share, 1
        ),
        "wheel_events": int(round(float(parent.get("wheel_events", 0) or 0) * share)),
        "horizontal_wheel_events": int(
            round(float(parent.get("horizontal_wheel_events", 0) or 0) * share)
        ),
    })
    new_seg.pop("merged", None)
    new_seg.pop("sub_segments", None)
    return new_seg


def subdivide_unknown_segment(
    seg: Dict[str, Any],
    *,
    video_path: Path,
    client,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    system_prompt: str,
    video_start_ns: int,
) -> List[Dict[str, Any]]:
    candidates = list(UNKNOWN_FULL_CANDIDATES)
    frame_count = max(int(frames_per_segment), 2)
    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": "unknown",
        "model": model,
        "k": frame_count,
        "detail": image_detail,
        "stage": "cyberpunk_subdivide_unknown",
        "video_start_ns": video_start_ns,
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)
    timestamps = sample_segment_timestamps(
        seg["start_timestamp_ns"], seg["end_timestamp_ns"], frame_count
    )

    if cached is not None:
        sub_events = cached
    else:
        video_timestamps = [
            max(0.0, timestamp - video_start_ns / NS_PER_S)
            for timestamp in timestamps
        ]
        jpeg_blobs = [
            extract_frame_jpeg(video_path, timestamp, max_width=1024)
            for timestamp in video_timestamps
        ]
        parsed = _call_gemini(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=build_subdivide_user_prompt(seg, candidates, frame_count),
            jpeg_blobs=jpeg_blobs,
            image_detail=image_detail,
        )
        sub_events = _normalize_sub_events(parsed.get("sub_events"), candidates, frame_count)
        cache.set(cache_key, sub_events)

    out: List[Dict[str, Any]] = []
    total = len(sub_events)
    for i, event in enumerate(sub_events):
        next_start = sub_events[i + 1]["start_frame_idx"] if i + 1 < total else None
        out.append(_build_sub_segment(
            parent=seg,
            sub_event=event,
            sample_timestamps_s=timestamps,
            next_event_start_idx=next_start,
            is_first=(i == 0),
            is_last=(i == total - 1),
            sub_index=i,
            sub_total=total,
            candidates=candidates,
            model=model,
        ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def refine_segments(
    segments_payload: Dict[str, Any],
    *,
    video_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
    frames_per_segment: int,
    image_detail: str,
    concurrency: int,
    cache_dir: Optional[Path],
    limit: Optional[int],
    subdivide_unknown: bool = True,
    subdivide_frames_per_segment: Optional[int] = None,
    video_start_ns: Optional[int] = None,
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = segments_payload["segments"]
    scene_name = segments_payload.get("scene_name") or video_path.stem
    if video_start_ns is None:
        video_start_ns = _resolve_video_start_ns(video_path, segments)
    if video_start_ns:
        print(f"video start (creation_time) = {video_start_ns} ns")
    else:
        print("video timestamps appear relative; using origin 0")
    targets = [i for i, seg in enumerate(segments) if seg.get("vision_needed")]
    if limit is not None:
        targets = targets[:limit]

    unknown_targets = sum(1 for i in targets if segments[i].get("label") == "unknown")
    other_targets = len(targets) - unknown_targets
    sub_k = (
        int(subdivide_frames_per_segment)
        if subdivide_frames_per_segment is not None
        else max(int(frames_per_segment), 10)
    )
    print(
        f"refining {len(targets)} / {len(segments)} segments via {model} "
        f"(single-label: {other_targets}, "
        f"{'subdivide' if subdivide_unknown else 'single-label'} unknown: "
        f"{unknown_targets}, K={frames_per_segment}"
        f"{f' / sub_K={sub_k}' if subdivide_unknown else ''})"
    )

    client = gemini_client(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir)
    system_prompt = build_system_prompt()
    subdivide_system_prompt = build_subdivide_system_prompt()

    target_outputs: Dict[int, List[Dict[str, Any]]] = {}
    target_lock = threading.Lock()
    error_lock = threading.Lock()
    error_count = 0
    subdivided_count = 0
    sub_segments_emitted = 0

    def _worker(idx: int) -> None:
        nonlocal error_count, subdivided_count, sub_segments_emitted
        seg = segments[idx]
        try:
            if subdivide_unknown and seg.get("label") == "unknown":
                subs = subdivide_unknown_segment(
                    seg,
                    video_path=video_path,
                    client=client,
                    model=model,
                    frames_per_segment=sub_k,
                    image_detail=image_detail,
                    cache=cache,
                    scene_name=scene_name,
                    system_prompt=subdivide_system_prompt,
                    video_start_ns=video_start_ns,
                )
                with target_lock:
                    target_outputs[idx] = subs
                with error_lock:
                    if len(subs) > 1:
                        subdivided_count += 1
                    sub_segments_emitted += len(subs)
            else:
                result = refine_segment(
                    seg,
                    video_path=video_path,
                    client=client,
                    model=model,
                    frames_per_segment=frames_per_segment,
                    image_detail=image_detail,
                    cache=cache,
                    scene_name=scene_name,
                    system_prompt=system_prompt,
                    video_start_ns=video_start_ns,
                )
                new_seg = dict(seg)
                new_seg.update(result)
                with target_lock:
                    target_outputs[idx] = [new_seg]
        except Exception as exc:  # noqa: BLE001
            with error_lock:
                error_count += 1
            sys.stderr.write(f"[seg {idx}] error: {exc}\n")

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, idx): idx for idx in targets}
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="refine", unit="seg")
        for future in iterator:
            future.result()

    out_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if i in target_outputs:
            out_segments.extend(target_outputs[i])
        else:
            out_segments.append(dict(seg))

    refined_count = sum(1 for seg in out_segments if "refined_label" in seg)
    label_changes: Dict[str, Dict[str, int]] = {}
    for seg in out_segments:
        refined_label = seg.get("refined_label")
        if refined_label and refined_label != seg.get("label"):
            label_changes.setdefault(seg["label"], {}).setdefault(refined_label, 0)
            label_changes[seg["label"]][refined_label] += 1

    payload = dict(segments_payload)
    payload["segments"] = out_segments
    payload["refine_meta"] = {
        "model": model,
        "frames_per_segment": frames_per_segment,
        "subdivide_frames_per_segment": sub_k if subdivide_unknown else None,
        "image_detail": image_detail,
        "video_start_ns": video_start_ns,
        "vision_needed_total": sum(1 for seg in segments if seg.get("vision_needed")),
        "refined_total": refined_count,
        "errors": error_count,
        "label_changes": label_changes,
        "subdivide_unknown_enabled": subdivide_unknown,
        "unknown_inputs_total": unknown_targets,
        "unknown_inputs_subdivided": subdivided_count,
        "unknown_sub_segments_emitted": sub_segments_emitted,
        "segments_in": len(segments),
        "segments_out_after_refine": len(out_segments),
    }
    return payload


# ---------------------------------------------------------------------------
# Post-refine merging
# ---------------------------------------------------------------------------

_REFINED_FIELDS = (
    "refined_label",
    "refined_tier1",
    "refined_tier2",
    "refined_confidence",
    "refined_reason",
    "refined_via",
    "refined_candidates",
    "refined_sampled_seconds",
)

MOVEMENT_GROUP = frozenset({"walking", "sprinting", "crouch_walking", "jumping"})
COMBAT_GROUP = frozenset({
    "aiming_down_sights",
    "firing_hipfire",
    "firing_ads",
    "melee_fast_attack",
    "blocking_or_heavy_attack",
})
VEHICLE_GROUP = frozenset({
    "driving_forward",
    "vehicle_brake_reverse",
    "vehicle_turning",
    "vehicle_handbrake",
    "vehicle_camera",
    "vehicle_radio",
    "vehicle_lights",
    "vehicle_primary_fire",
    "vehicle_secondary_fire",
    "vehicle_combat",
})
BRAINDANCE_GROUP = frozenset({
    "braindance_pause",
    "braindance_rewind",
    "braindance_fast_forward",
    "braindance_restart",
    "braindance_layer_change",
    "braindance_analysis_mode",
    "braindance_zoom",
})
MERGE_GROUPS: Tuple[Tuple[str, frozenset], ...] = (
    ("movement_group", MOVEMENT_GROUP),
    ("combat_group", COMBAT_GROUP),
    ("vehicle_group", VEHICLE_GROUP),
    ("braindance_group", BRAINDANCE_GROUP),
)


def _effective_label(seg: Dict[str, Any]) -> str:
    refined_label = seg.get("refined_label")
    if isinstance(refined_label, str) and refined_label:
        return refined_label
    return seg.get("label", "") or ""


def _seg_components(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = seg.get("sub_segments")
    if existing:
        return [dict(comp) for comp in existing]
    comp: Dict[str, Any] = {
        "label": seg.get("label"),
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "start_timestamp_ns": seg["start_timestamp_ns"],
        "end_timestamp_ns": seg["end_timestamp_ns"],
        "duration_s": seg.get("duration_s"),
    }
    if seg.get("refined_label"):
        comp["refined_label"] = seg.get("refined_label")
    return [comp]


def _merge_pair(left: Dict[str, Any], right: Dict[str, Any], label: str) -> Dict[str, Any]:
    leaf = LEAF_BY_LABEL.get(label)
    duration_ns = right["end_timestamp_ns"] - left["start_timestamp_ns"]
    keys = sorted(set(left.get("keys_pressed", []) or []) | set(right.get("keys_pressed", []) or []))
    mouse_buttons = sorted(
        set(left.get("mouse_buttons_pressed", []) or [])
        | set(right.get("mouse_buttons_pressed", []) or [])
    )
    if left.get("refined_label") == label:
        refined_src: Optional[Dict[str, Any]] = left
    elif right.get("refined_label") == label:
        refined_src = right
    else:
        refined_src = None

    merged: Dict[str, Any] = {
        "start_frame": left["start_frame"],
        "end_frame": right["end_frame"],
        "num_frames": right["end_frame"] - left["start_frame"] + 1,
        "start_timestamp_ns": left["start_timestamp_ns"],
        "end_timestamp_ns": right["end_timestamp_ns"],
        "duration_s": round(duration_ns / NS_PER_S, 3),
        "label": label,
        "tier1": leaf.tier1 if leaf else left.get("tier1") or right.get("tier1"),
        "tier2": leaf.tier2 if leaf else left.get("tier2") or right.get("tier2"),
        "action_distinguishable": leaf.action_distinguishable if leaf else "n/a",
        "vision_needed": taxonomy_vision_needed(label),
        "vision_subtypes": list(leaf.vision_subtypes) if leaf else [],
        "keys_pressed": keys,
        "mouse_buttons_pressed": mouse_buttons,
        "mouse_motion_total": round(
            float(left.get("mouse_motion_total", 0.0) or 0.0)
            + float(right.get("mouse_motion_total", 0.0) or 0.0),
            1,
        ),
        "wheel_events": int(left.get("wheel_events", 0) or 0)
        + int(right.get("wheel_events", 0) or 0),
        "horizontal_wheel_events": int(left.get("horizontal_wheel_events", 0) or 0)
        + int(right.get("horizontal_wheel_events", 0) or 0),
        "frame_file_start": left.get("frame_file_start"),
        "frame_file_end": right.get("frame_file_end"),
        "sub_segments": _seg_components(left) + _seg_components(right),
        "merged": True,
    }
    if (
        left.get("label") == "unknown"
        or right.get("label") == "unknown"
        or left.get("merged_from_unknown")
        or right.get("merged_from_unknown")
    ):
        merged["merged_from_unknown"] = True
    if refined_src is not None:
        for key in _REFINED_FIELDS:
            if key in refined_src:
                merged[key] = refined_src[key]
    return merged


def _recount_labels(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        label = seg.get("label", "unknown")
        count = counts.setdefault(label, {"segments": 0, "total_s": 0.0})
        count["segments"] += 1
        count["total_s"] += float(seg.get("duration_s") or 0.0)
    return {
        label: {"segments": int(value["segments"]), "total_s": round(value["total_s"], 2)}
        for label, value in counts.items()
    }


def _resolve_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[Dict[str, str]]:
    left_label = _effective_label(left)
    right_label = _effective_label(right)
    if left_label == "unknown" and right_label == "unknown":
        return None
    if left_label and left_label == right_label:
        return {"label": left_label, "kind": "same_label"}
    for kind, group in MERGE_GROUPS:
        if left_label in group and right_label in group:
            left_duration = float(left.get("duration_s") or 0.0)
            right_duration = float(right.get("duration_s") or 0.0)
            return {"label": left_label if left_duration >= right_duration else right_label, "kind": kind}
    return None


def merge_refined_unknowns(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = list(payload.get("segments", []))
    if not segments:
        return dict(payload)

    out: List[Dict[str, Any]] = []
    merges_performed = 0
    merges_by_label: Dict[str, int] = {}
    merges_by_kind: Dict[str, int] = {}

    for seg in segments:
        candidate = dict(seg)
        if not out:
            out.append(candidate)
            continue
        prev = out[-1]
        decision = _resolve_merge(prev, candidate)
        if decision is None:
            out.append(candidate)
            continue
        label = decision["label"]
        kind = decision["kind"]
        out[-1] = _merge_pair(prev, candidate, label)
        merges_performed += 1
        merges_by_label[label] = merges_by_label.get(label, 0) + 1
        merges_by_kind[kind] = merges_by_kind.get(kind, 0) + 1

    new_payload = dict(payload)
    new_payload["segments"] = out
    new_payload["label_counts"] = _recount_labels(out)
    refine_meta = dict(new_payload.get("refine_meta") or {})
    refine_meta["merge_unknown_meta"] = {
        "merges_performed": merges_performed,
        "segments_before": len(segments),
        "segments_after": len(out),
        "merges_by_label": dict(sorted(merges_by_label.items(), key=lambda item: -item[1])),
        "merges_by_kind": merges_by_kind,
        "merge_groups": {kind: sorted(group) for kind, group in MERGE_GROUPS},
        "movement_group": sorted(MOVEMENT_GROUP),
    }
    new_payload["refine_meta"] = refine_meta
    return new_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "segments_json",
        type=Path,
        help="Input segments JSON from 1a_Cyberpunk_2077_scripts_segment_from_actions.py.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source MKV / MP4 video to sample frames from. Required unless --merge-only.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Where to write refined JSON.")
    parser.add_argument(
        "--merge-unknowns",
        dest="merge_unknowns",
        action="store_true",
        default=True,
        help="After refine, merge adjacent compatible effective labels (default on).",
    )
    parser.add_argument(
        "--no-merge-unknowns",
        dest="merge_unknowns",
        action="store_false",
        help="Disable post-refine merge step.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip Gemini calls; read input as already-refined JSON and only merge.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model id (default {DEFAULT_MODEL}).")
    parser.add_argument(
        "--frames-per-segment",
        type=int,
        default=5,
        help="Frames sampled for non-unknown vision_needed segments (default 5).",
    )
    parser.add_argument(
        "--subdivide-unknown",
        dest="subdivide_unknown",
        action="store_true",
        default=True,
        help="Ask Gemini to subdivide unknown segments into sub-events (default on).",
    )
    parser.add_argument(
        "--no-subdivide-unknown",
        dest="subdivide_unknown",
        action="store_false",
        help="Use single-label refinement for unknown segments too.",
    )
    parser.add_argument(
        "--subdivide-frames-per-segment",
        type=int,
        default=10,
        help="Frames sampled for unknown subdivision (default 10).",
    )
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel Gemini calls (default 8).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".gemini_cache"),
        help="JSON cache directory (default .gemini_cache). Use empty string to disable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N vision_needed segments.")
    parser.add_argument("--api-key", default=None, help="Override env API key for this run.")
    parser.add_argument("--base-url", default=None, help="Override Gemini OpenAI-compatible base URL.")
    parser.add_argument(
        "--video-start-ns",
        type=int,
        default=None,
        help=(
            "epoch-ns timestamp of the video's first frame; default: "
            "auto-detect MKV creation_time. Pass 0 for relative timestamps"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.segments_json.exists():
        sys.exit(f"segments JSON not found: {args.segments_json}")
    if not args.merge_only:
        if args.video is None:
            sys.exit("--video is required unless --merge-only is set")
        if not args.video.exists():
            sys.exit(f"video not found: {args.video}")

    with args.segments_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cache_dir: Optional[Path] = args.cache_dir if str(args.cache_dir) else None

    if args.merge_only:
        print("--merge-only set: skipping Gemini refine, running merge step only")
        refined = payload
    else:
        refined = refine_segments(
            payload,
            video_path=args.video,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            frames_per_segment=args.frames_per_segment,
            image_detail=args.image_detail,
            concurrency=args.concurrency,
            cache_dir=cache_dir,
            limit=args.limit,
            subdivide_unknown=args.subdivide_unknown,
            subdivide_frames_per_segment=args.subdivide_frames_per_segment,
            video_start_ns=args.video_start_ns,
        )

    if args.merge_unknowns:
        refined = merge_refined_unknowns(refined)

    if args.out is not None:
        out_path = args.out
    elif args.merge_only:
        out_path = args.segments_json.with_name(args.segments_json.stem + "_merged.json")
    else:
        out_path = args.segments_json.with_name(args.segments_json.stem + "_refined.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    meta = refined.get("refine_meta") or {}
    if not args.merge_only:
        print(f"  refined: {meta.get('refined_total', 0)} / {meta.get('vision_needed_total', 0)}")
        print(f"  errors : {meta.get('errors', 0)}")
        if meta.get("subdivide_unknown_enabled"):
            print(
                "  subdivide unknown: "
                f"{meta.get('unknown_inputs_subdivided', 0)} / "
                f"{meta.get('unknown_inputs_total', 0)} inputs split into 2+ events, "
                f"{meta.get('unknown_sub_segments_emitted', 0)} sub-segments emitted "
                f"({meta.get('segments_in', 0)} -> "
                f"{meta.get('segments_out_after_refine', 0)} segments)"
            )
        if meta.get("label_changes"):
            print("  label changes (parent -> refined: count):")
            for parent, sub in sorted(meta["label_changes"].items()):
                for refined_label, count in sorted(sub.items(), key=lambda item: -item[1]):
                    print(f"    {parent:24s} -> {refined_label:24s}  x {count}")

    merge_meta = meta.get("merge_unknown_meta")
    if merge_meta is not None:
        print(
            f"  merged: {merge_meta['merges_performed']} merges, "
            f"{merge_meta['segments_before']} -> {merge_meta['segments_after']} segments"
        )
        if merge_meta["merges_by_label"]:
            print("  merges by label:")
            for label, count in merge_meta["merges_by_label"].items():
                print(f"    {label:24s}  x {count}")


if __name__ == "__main__":
    main()
