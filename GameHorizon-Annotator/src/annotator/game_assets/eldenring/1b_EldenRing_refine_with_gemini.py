#!/usr/bin/env python3
"""Stage 1b: OpenAI-compatible vision refinement for Elden Ring segments."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from EldenRing_taxonomy import (
    LEAF_BY_LABEL,
    LEAVES,
    format_taxonomy_for_prompt,
    normalize_label,
    subtype_choices,
    vision_needed as taxonomy_vision_needed,
)
from EldenRing_video_io import (
    JsonCache,
    NS_PER_S,
    build_image_content,
    extract_frame_jpeg,
    extract_json_object,
    gemini_client,
    resolve_video_start_ns,
    sample_segment_timestamps,
    with_retry,
)


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_S = 300.0
PROMPT_VERSION = 1
KEYBINDS_PATH = Path(__file__).resolve().parent / "EldenRing_keybinds.json"


def _load_keybinds() -> Dict[str, str]:
    try:
        value = json.loads(KEYBINDS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {
        str(key): item.strip()
        for key, item in value.items()
        if not str(key).startswith("_") and isinstance(item, str)
    }


KEYBINDS = _load_keybinds()
UNKNOWN_CANDIDATES = [leaf.label for leaf in LEAVES if not leaf.label.startswith("idle")]
if "unknown" not in UNKNOWN_CANDIDATES:
    UNKNOWN_CANDIDATES.append("unknown")
GUI_CANDIDATES = [leaf.label for leaf in LEAVES if leaf.tier1 == "gui"]


def _keybind_text() -> str:
    lines = ["Elden Ring keyboard/mouse bindings (raw name -> action):"]
    lines.extend(f"  - {key}: {value}" for key, value in KEYBINDS.items())
    return "\n".join(lines)


def build_system_prompt(*, subdivide: bool = False) -> str:
    task = (
        "Split the ordered frames into contiguous event ranges."
        if subdivide
        else "Choose the single most accurate event label."
    )
    output = (
        '{"sub_events":[{"start_frame_idx":0,"end_frame_idx":0,'
        '"label":"candidate","confidence":0.0,"reason":"short clause"}]}'
        if subdivide
        else '{"label":"candidate","confidence":0.0,"reason":"short clause"}'
    )
    return (
        "You classify short clips from ELDEN RING (the base game, not "
        "ELDEN RING NIGHTREIGN). Frames are ordered from first to last. "
        f"{task} Use only labels supplied in the user candidate list. "
        "Do not invent an idle label; return unknown when evidence is insufficient. "
        "Raw inputs are evidence but overloaded actions must be resolved visually.\n\n"
        f"{_keybind_text()}\n\n"
        "Elden Ring taxonomy:\n"
        f"{format_taxonomy_for_prompt(include_descriptions=True)}\n\n"
        f"Return strictly one JSON object shaped like: {output}"
    )


SYSTEM_PROMPT = build_system_prompt()
SUBDIVIDE_SYSTEM_PROMPT = build_system_prompt(subdivide=True)


def _candidates(parent: str) -> List[str]:
    if parent == "unknown":
        return list(UNKNOWN_CANDIDATES)
    if parent == "menu_other":
        return list(GUI_CANDIDATES)
    return list(subtype_choices(parent))


def build_user_prompt(
    segment: Dict[str, Any], candidates: Sequence[str], *, subdivide: bool = False, k: int = 0
) -> str:
    evidence = {
        "parent_label": segment.get("label", "unknown"),
        "candidates": list(candidates),
        "duration_s": segment.get("duration_s"),
        "tier1": segment.get("tier1"),
        "tier2": segment.get("tier2"),
        "keys_pressed": segment.get("keys_pressed", []),
        "mouse_buttons_pressed": segment.get("mouse_buttons_pressed", []),
        "wheel_events": segment.get("wheel_events", 0),
        "mouse_motion_total": segment.get("mouse_motion_total", 0),
    }
    if subdivide:
        evidence["K"] = k
        instruction = (
            f"Cover every sampled frame index 0..{k - 1} exactly once with contiguous, "
            "non-overlapping sub_events. Merge adjacent frames with the same activity."
        )
    else:
        instruction = "Choose one candidate; keep parent_label if no subtype is clearly visible."
    return f"{instruction}\n\nSegment evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"


@with_retry(max_attempts=4, initial_delay_s=1.5, max_delay_s=20.0)
def _call_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[bytes],
    image_detail: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    content = [{"type": "text", "text": system_prompt}]
    content.extend(build_image_content(user_prompt, images, detail=image_detail))
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


def _validated_label(raw: Any, candidates: Sequence[str], fallback: str) -> str:
    label = str(raw or "").strip()
    if label in candidates:
        return label
    normalized = normalize_label(label)
    return normalized if normalized in candidates else fallback


def _confidence(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def _sample_images(
    segment: Dict[str, Any],
    *,
    video_path: Path,
    k: int,
    video_start_ns: int,
) -> Tuple[List[float], List[float], List[bytes]]:
    absolute = sample_segment_timestamps(
        int(segment["start_timestamp_ns"]),
        int(segment["end_timestamp_ns"]),
        k,
    )
    relative = [max(0.0, value - video_start_ns / NS_PER_S) for value in absolute]
    images = [extract_frame_jpeg(video_path, value, max_width=1024) for value in relative]
    return absolute, relative, images


def refine_one(
    segment: Dict[str, Any],
    *,
    video_path: Path,
    client: Any,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    video_start_ns: int,
) -> Dict[str, Any]:
    parent = str(segment.get("label") or "unknown")
    candidates = _candidates(parent)
    key = {
        "scene": scene_name,
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "parent": parent,
        "model": model,
        "k": frames_per_segment,
        "detail": image_detail,
        "stage": "eldenring_refine",
        "video_start_ns": video_start_ns,
        "prompt_version": PROMPT_VERSION,
    }
    cached = cache.get(key)
    if isinstance(cached, dict):
        return cached
    _, relative, images = _sample_images(
        segment,
        video_path=video_path,
        k=frames_per_segment,
        video_start_ns=video_start_ns,
    )
    parsed = _call_model(
        client,
        model,
        SYSTEM_PROMPT,
        build_user_prompt(segment, candidates),
        images,
        image_detail,
    )
    label = _validated_label(parsed.get("label"), candidates, parent)
    leaf = LEAF_BY_LABEL.get(label)
    result = {
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else segment.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else segment.get("tier2"),
        "refined_confidence": _confidence(parsed.get("confidence")),
        "refined_reason": str(parsed.get("reason") or "")[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [round(value, 3) for value in relative],
    }
    cache.set(key, result)
    return result


def _normalize_sub_events(raw: Any, candidates: Sequence[str], k: int) -> List[Dict[str, Any]]:
    fallback = [
        {
            "start_frame_idx": 0,
            "end_frame_idx": k - 1,
            "label": "unknown",
            "confidence": None,
            "reason": "invalid subdivision response",
        }
    ]
    if not isinstance(raw, list) or not raw:
        return fallback
    parsed: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start_frame_idx"))
            end = int(item.get("end_frame_idx"))
        except (TypeError, ValueError):
            continue
        start = max(0, min(k - 1, start))
        end = max(start, min(k - 1, end))
        parsed.append(
            {
                "start_frame_idx": start,
                "end_frame_idx": end,
                "label": _validated_label(item.get("label"), candidates, "unknown"),
                "confidence": _confidence(item.get("confidence")),
                "reason": str(item.get("reason") or "")[:300],
            }
        )
    if not parsed:
        return fallback
    parsed.sort(key=lambda item: (item["start_frame_idx"], item["end_frame_idx"]))
    fixed: List[Dict[str, Any]] = []
    cursor = 0
    for item in parsed:
        if cursor >= k:
            break
        start = cursor
        end = max(start, item["end_frame_idx"])
        fixed.append({**item, "start_frame_idx": start, "end_frame_idx": min(k - 1, end)})
        cursor = end + 1
    fixed[-1]["end_frame_idx"] = k - 1
    merged: List[Dict[str, Any]] = []
    for item in fixed:
        if merged and merged[-1]["label"] == item["label"]:
            merged[-1]["end_frame_idx"] = item["end_frame_idx"]
            if (item.get("confidence") or 0) > (merged[-1].get("confidence") or 0):
                merged[-1]["confidence"] = item.get("confidence")
                merged[-1]["reason"] = item.get("reason")
        else:
            merged.append(item)
    return merged


def _interpolate_frame(segment: Dict[str, Any], timestamp_ns: int) -> int:
    start_ns = int(segment["start_timestamp_ns"])
    end_ns = int(segment["end_timestamp_ns"])
    start_frame = int(segment["start_frame"])
    end_frame = int(segment["end_frame"])
    if end_ns <= start_ns or end_frame <= start_frame:
        return start_frame
    fraction = (max(start_ns, min(end_ns, timestamp_ns)) - start_ns) / (end_ns - start_ns)
    return start_frame + int(round(fraction * (end_frame - start_frame)))


def subdivide_unknown(
    segment: Dict[str, Any],
    *,
    video_path: Path,
    client: Any,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    video_start_ns: int,
) -> List[Dict[str, Any]]:
    k = max(2, int(frames_per_segment))
    candidates = list(UNKNOWN_CANDIDATES)
    key = {
        "scene": scene_name,
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "parent": "unknown",
        "model": model,
        "k": k,
        "detail": image_detail,
        "stage": "eldenring_subdivide_unknown",
        "video_start_ns": video_start_ns,
        "prompt_version": PROMPT_VERSION,
    }
    cached = cache.get(key)
    absolute = sample_segment_timestamps(
        int(segment["start_timestamp_ns"]), int(segment["end_timestamp_ns"]), k
    )
    relative = [max(0.0, value - video_start_ns / NS_PER_S) for value in absolute]
    if isinstance(cached, list):
        events = cached
    else:
        images = [extract_frame_jpeg(video_path, value, max_width=1024) for value in relative]
        parsed = _call_model(
            client,
            model,
            SUBDIVIDE_SYSTEM_PROMPT,
            build_user_prompt(segment, candidates, subdivide=True, k=k),
            images,
            image_detail,
        )
        events = _normalize_sub_events(parsed.get("sub_events"), candidates, k)
        cache.set(key, events)

    parent_start = int(segment["start_timestamp_ns"])
    parent_end = int(segment["end_timestamp_ns"])
    parent_duration = max(1, parent_end - parent_start)
    result: List[Dict[str, Any]] = []
    previous_end = parent_start
    previous_end_frame = int(segment["start_frame"]) - 1
    for index, event in enumerate(events):
        if index + 1 == len(events):
            end_ns = parent_end
        else:
            end_idx = int(event["end_frame_idx"])
            next_idx = min(k - 1, end_idx + 1)
            midpoint_s = (absolute[end_idx] + absolute[next_idx]) / 2.0
            end_ns = max(previous_end + 1, min(parent_end - 1, int(round(midpoint_s * NS_PER_S))))
        label = str(event["label"])
        leaf = LEAF_BY_LABEL.get(label)
        child = dict(segment)
        start_frame = max(previous_end_frame + 1, _interpolate_frame(segment, previous_end))
        if index + 1 == len(events):
            end_frame = int(segment["end_frame"])
        else:
            end_frame = max(start_frame, _interpolate_frame(segment, end_ns) - 1)
        share = (end_ns - previous_end) / parent_duration
        si = int(event["start_frame_idx"])
        ei = int(event["end_frame_idx"])
        child.update(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": max(1, end_frame - start_frame + 1),
                "start_timestamp_ns": previous_end,
                "end_timestamp_ns": end_ns,
                "duration_s": round((end_ns - previous_end) / NS_PER_S, 3),
                "label": "unknown",
                "refined_label": label,
                "refined_tier1": leaf.tier1 if leaf else segment.get("tier1"),
                "refined_tier2": leaf.tier2 if leaf else segment.get("tier2"),
                "refined_confidence": event.get("confidence"),
                "refined_reason": event.get("reason", ""),
                "refined_via": model,
                "refined_candidates": candidates,
                "refined_sampled_seconds": [round(relative[pos], 3) for pos in range(si, ei + 1)],
                "subdivided_from_unknown": True,
                "sub_event_index": index,
                "sub_event_total": len(events),
                "sub_event_frame_range": [si, ei],
                "mouse_motion_total": round(float(segment.get("mouse_motion_total") or 0) * share, 1),
                "wheel_events": int(round(float(segment.get("wheel_events") or 0) * share)),
            }
        )
        child.pop("sub_segments", None)
        child.pop("merged", None)
        result.append(child)
        previous_end = end_ns
        previous_end_frame = end_frame
    return result


def refine_segments(
    payload: Dict[str, Any],
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
    subdivide_unknown_enabled: bool = True,
    subdivide_frames_per_segment: int = 10,
    video_start_ns: Optional[int] = None,
) -> Dict[str, Any]:
    source = list(payload.get("segments") or [])
    scene_name = str(payload.get("scene_name") or video_path.stem)
    origin = resolve_video_start_ns(video_path, source, explicit=video_start_ns)
    all_targets = [index for index, segment in enumerate(source) if segment.get("vision_needed")]
    targets = all_targets[:limit] if limit is not None else all_targets
    print(
        f"refining {len(targets)} / {len(all_targets)} vision targets via {model} "
        f"(video_start_ns={origin}, concurrency={concurrency})"
    )
    client = gemini_client(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir)
    outputs: Dict[int, List[Dict[str, Any]]] = {}
    errors: Dict[int, str] = {}
    lock = threading.Lock()

    def worker(index: int) -> None:
        segment = source[index]
        try:
            if subdivide_unknown_enabled and segment.get("label") == "unknown":
                value = subdivide_unknown(
                    segment,
                    video_path=video_path,
                    client=client,
                    model=model,
                    frames_per_segment=subdivide_frames_per_segment,
                    image_detail=image_detail,
                    cache=cache,
                    scene_name=scene_name,
                    video_start_ns=origin,
                )
            else:
                refined = dict(segment)
                refined.update(
                    refine_one(
                        segment,
                        video_path=video_path,
                        client=client,
                        model=model,
                        frames_per_segment=frames_per_segment,
                        image_detail=image_detail,
                        cache=cache,
                        scene_name=scene_name,
                        video_start_ns=origin,
                    )
                )
                value = [refined]
            with lock:
                outputs[index] = value
        except Exception as exc:
            failed = dict(segment)
            failed["refine_error"] = f"{type(exc).__name__}: {exc}"[:800]
            with lock:
                outputs[index] = [failed]
                errors[index] = failed["refine_error"]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(worker, index): index for index in targets}
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count == len(futures) or count % 10 == 0:
                print(f"  completed {count}/{len(futures)}")

    combined: List[Dict[str, Any]] = []
    for index, segment in enumerate(source):
        combined.extend(outputs.get(index, [dict(segment)]))
    changes: Dict[str, Dict[str, int]] = {}
    refined_total = 0
    subdivided_inputs = 0
    for index in targets:
        values = outputs.get(index, [])
        if values and "refined_label" in values[0]:
            refined_total += 1
        if len(values) > 1:
            subdivided_inputs += 1
        parent = str(source[index].get("label") or "unknown")
        for value in values:
            label = value.get("refined_label")
            if label and label != parent:
                changes.setdefault(parent, {})[str(label)] = changes.setdefault(parent, {}).get(str(label), 0) + 1
    result = dict(payload)
    result["segments"] = combined
    result["video_start_ns"] = origin
    result["refine_meta"] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "refined_at": datetime.now(timezone.utc).isoformat(),
        "video_start_ns": origin,
        "segments_in": len(source),
        "segments_out_after_refine": len(combined),
        "vision_needed_total": len(all_targets),
        "attempted_total": len(targets),
        "refined_total": refined_total,
        "errors": len(errors),
        "error_segments": errors,
        "limit": limit,
        "subdivide_unknown_enabled": subdivide_unknown_enabled,
        "unknown_inputs_subdivided": subdivided_inputs,
        "label_changes": changes,
    }
    return result


def _effective_label(segment: Dict[str, Any]) -> str:
    return str(segment.get("refined_label") or segment.get("label") or "unknown")


def _components(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = segment.get("sub_segments")
    if isinstance(existing, list) and existing:
        return [dict(item) for item in existing]
    item = {
        "label": segment.get("label"),
        "start_frame": segment.get("start_frame"),
        "end_frame": segment.get("end_frame"),
        "start_timestamp_ns": segment.get("start_timestamp_ns"),
        "end_timestamp_ns": segment.get("end_timestamp_ns"),
        "duration_s": segment.get("duration_s"),
    }
    if segment.get("refined_label"):
        item["refined_label"] = segment["refined_label"]
    return [item]


def _merge_pair(left: Dict[str, Any], right: Dict[str, Any], label: str) -> Dict[str, Any]:
    leaf = LEAF_BY_LABEL.get(label)
    duration_ns = int(right["end_timestamp_ns"]) - int(left["start_timestamp_ns"])
    result = {
        "start_frame": left["start_frame"],
        "end_frame": right["end_frame"],
        "num_frames": int(right["end_frame"]) - int(left["start_frame"]) + 1,
        "start_timestamp_ns": left["start_timestamp_ns"],
        "end_timestamp_ns": right["end_timestamp_ns"],
        "duration_s": round(duration_ns / NS_PER_S, 3),
        "label": label,
        "tier1": leaf.tier1 if leaf else left.get("tier1"),
        "tier2": leaf.tier2 if leaf else left.get("tier2"),
        "action_distinguishable": leaf.action_distinguishable if leaf else "n/a",
        "vision_needed": taxonomy_vision_needed(label),
        "vision_subtypes": list(leaf.vision_subtypes) if leaf else [],
        "keys_pressed": sorted(set(left.get("keys_pressed", [])) | set(right.get("keys_pressed", []))),
        "mouse_buttons_pressed": sorted(
            set(left.get("mouse_buttons_pressed", [])) | set(right.get("mouse_buttons_pressed", []))
        ),
        "mouse_motion_total": round(
            float(left.get("mouse_motion_total") or 0) + float(right.get("mouse_motion_total") or 0), 1
        ),
        "wheel_events": int(left.get("wheel_events") or 0) + int(right.get("wheel_events") or 0),
        "frame_file_start": left.get("frame_file_start"),
        "frame_file_end": right.get("frame_file_end"),
        "sub_segments": _components(left) + _components(right),
        "merged": True,
    }
    if left.get("label") == "unknown" or right.get("label") == "unknown":
        result["merged_from_unknown"] = True
    source = left if left.get("refined_label") == label else right if right.get("refined_label") == label else None
    if source:
        for field in (
            "refined_label", "refined_tier1", "refined_tier2", "refined_confidence",
            "refined_reason", "refined_via", "refined_candidates", "refined_sampled_seconds",
        ):
            if field in source:
                result[field] = source[field]
    return result


def merge_refined_unknowns(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = list(payload.get("segments") or [])
    out: List[Dict[str, Any]] = []
    merge_count = 0
    by_kind: Dict[str, int] = {}
    for segment in source:
        candidate = dict(segment)
        if not out:
            out.append(candidate)
            continue
        left_label = _effective_label(out[-1])
        right_label = _effective_label(candidate)
        kind: Optional[str] = None
        label: Optional[str] = None
        if left_label == right_label and left_label != "unknown":
            kind, label = "same_label", left_label
        elif left_label in {"walking", "sprinting"} and right_label in {"walking", "sprinting"}:
            kind = "traversal_group"
            label = left_label if float(out[-1].get("duration_s") or 0) >= float(candidate.get("duration_s") or 0) else right_label
        if label:
            out[-1] = _merge_pair(out[-1], candidate, label)
            merge_count += 1
            by_kind[kind or "unknown"] = by_kind.get(kind or "unknown", 0) + 1
        else:
            out.append(candidate)
    counts: Dict[str, Dict[str, float]] = {}
    for segment in out:
        label = str(segment.get("label") or "unknown")
        item = counts.setdefault(label, {"segments": 0, "total_s": 0.0})
        item["segments"] += 1
        item["total_s"] += float(segment.get("duration_s") or 0)
    result = dict(payload)
    result["segments"] = out
    result["label_counts"] = {
        key: {"segments": int(value["segments"]), "total_s": round(value["total_s"], 2)}
        for key, value in counts.items()
    }
    meta = dict(result.get("refine_meta") or {})
    meta["merge_unknown_meta"] = {
        "merges_performed": merge_count,
        "segments_before": len(source),
        "segments_after": len(out),
        "merges_by_kind": by_kind,
        "traversal_group": ["sprinting", "walking"],
    }
    result["refine_meta"] = meta
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments_json", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--merge-unknowns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--subdivide-unknown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subdivide-frames-per-segment", type=int, default=10)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache_eldenring"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--video-start-ns", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.segments_json.exists():
        sys.exit(f"segments JSON not found: {args.segments_json}")
    payload = json.loads(args.segments_json.read_text(encoding="utf-8"))
    if args.merge_only:
        refined = payload
    else:
        if args.video is None or not args.video.exists():
            sys.exit("--video is required and must exist unless --merge-only is used")
        refined = refine_segments(
            payload,
            video_path=args.video,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            frames_per_segment=max(1, args.frames_per_segment),
            image_detail=args.image_detail,
            concurrency=max(1, args.concurrency),
            cache_dir=args.cache_dir if str(args.cache_dir) else None,
            limit=args.limit,
            subdivide_unknown_enabled=args.subdivide_unknown,
            subdivide_frames_per_segment=max(2, args.subdivide_frames_per_segment),
            video_start_ns=args.video_start_ns,
        )
    if args.merge_unknowns:
        refined = merge_refined_unknowns(refined)
    suffix = "_merged.json" if args.merge_only else "_refined.json"
    out = args.out or args.segments_json.with_name(args.segments_json.stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = refined.get("refine_meta") or {}
    print(f"wrote {out}")
    print(
        f"  refined={meta.get('refined_total', 0)}/{meta.get('attempted_total', 0)} "
        f"vision_total={meta.get('vision_needed_total', 0)} errors={meta.get('errors', 0)}"
    )


if __name__ == "__main__":
    main()
