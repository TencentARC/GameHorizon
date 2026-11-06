"""Stage 1b: Gemini-vision refinement of action-segmented Apex clips.

Reads a segments JSON produced by ``1a_Apex_Legends_segment_from_actions.py``, picks
out the segments where ``vision_needed=True`` (the rule classifier marked
them as ``unknown`` or as a vision-refineable parent like ``interact`` /
``using_health_item`` / ``pinging``), samples K frames per segment from the
source video via ``ffmpeg``, and asks Gemini to choose the best leaf label
from the parent's ``vision_subtypes`` (or to keep the parent label if
nothing fits).

The result is written back into the segments JSON with new fields:

* ``refined_label`` - the chosen leaf label
* ``refined_tier1`` / ``refined_tier2`` - re-derived from taxonomy
* ``refined_confidence`` - 0..1 from the model
* ``refined_reason`` - <=20 word explanation
* ``refined_via`` - which model produced the refinement

Segments where ``vision_needed=False`` are passed through unchanged.

CLI::

    python 1b_Apex_Legends_refine_with_gemini.py 0805_01_segments.json \
        --video 0805_01.mkv --out 0805_01_segments_refined.json \
        --frames-per-segment 5 --model gemini-2.5-flash --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import re

from Apex_Legends_taxonomy import (
    LEAF_BY_LABEL,
    LEAVES,
    format_taxonomy_for_prompt,
    normalize_label,
    subtype_choices,
    vision_needed as taxonomy_vision_needed,
)
from Apex_Legends_video_io import (
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

# Bumping ``PROMPT_VERSION`` invalidates on-disk caches whose prompts were
# generated under an older revision (e.g. before keybind context or the
# expanded unknown candidate set were introduced).
PROMPT_VERSION = 2

# NOTE: the Gemini client is constructed lazily inside
# ``refine_segments_with_gemini`` via ``Apex_Legends_video_io.gemini_client(api_key,
# base_url)`` which honours --api-key / --base-url and falls back to
# ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` env vars. Earlier revisions had a
# module-level ``client = OpenAI(...)`` here, but ``OpenAI`` was never
# imported at this scope -- it lived in ``Apex_Legends_video_io`` -- so importing
# the file would crash. The dead lines were removed in favour of the
# function-scoped client.

# ---------------------------------------------------------------------------
# Keybind reference (injected into the system prompt so the model can map
# raw RawInput key names like ``KEY_W`` / ``RI_MOUSE_LEFT_BUTTON_DOWN`` to
# their in-game function without relying purely on pretraining priors).
# ---------------------------------------------------------------------------

_KEYBINDS_JSON_PATH = Path(__file__).resolve().parent / "Apex_Legends_keybinds.json"
_PAREN_EN_RE = re.compile(r"\(([^)]+)\)")


def _load_keybinds_english() -> Dict[str, str]:
    """Read ``Apex_Legends_keybinds.json`` and keep only the English portion of each
    value (the text in parentheses; falls back to the raw value if no
    parenthesised English description is present)."""
    try:
        raw = json.loads(_KEYBINDS_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            continue
        m = _PAREN_EN_RE.search(v)
        out[k] = (m.group(1) if m else v).strip()
    return out


KEYBINDS_EN: Dict[str, str] = _load_keybinds_english()


def format_keybinds_for_prompt() -> str:
    """Render the keybind dictionary as a compact bullet list."""
    if not KEYBINDS_EN:
        return ""
    lines = ["Apex default key bindings (raw event name -> in-game action):"]
    for k, v in KEYBINDS_EN.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


KEYBINDS_REFERENCE_TEXT = format_keybinds_for_prompt()


# ---------------------------------------------------------------------------
# Atomic-leaf candidate set used when the rule classifier said ``unknown``.
# ---------------------------------------------------------------------------


def all_non_idle_atomic_labels() -> List[str]:
    """All taxonomy leaf labels except the rule-only ``idle_*`` family.

    Idle is decided strictly upstream by the action classifier (>=1s of
    completely zero input) and cannot be verified from a handful of still
    frames, so we never want Gemini to pick it.
    """
    seen: set = set()
    out: List[str] = []
    for leaf in LEAVES:
        lbl = leaf.label
        if lbl.startswith("idle"):
            continue
        if lbl in seen:
            continue
        seen.add(lbl)
        out.append(lbl)
    # Make sure "unknown" is always present as the explicit escape valve
    # even if some future taxonomy revision drops it from LEAVES.
    if "unknown" not in seen:
        out.append("unknown")
    return out


UNKNOWN_FULL_CANDIDATES: List[str] = all_non_idle_atomic_labels()


def _probe_video_start_ns(video_path: Path) -> Optional[int]:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format_tags=creation_time",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
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


def _resolve_video_start_ns(video_path: Path, segments: List[Dict[str, Any]]) -> int:
    origin_ns = _probe_video_start_ns(video_path) or 0
    if origin_ns and segments:
        max_end_ns = max(int(segment.get("end_timestamp_ns") or 0) for segment in segments)
        if max_end_ns < origin_ns * 0.5:
            return 0
    return origin_ns


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """Static system prompt with the full 3-tier taxonomy + keybind reference."""
    keybind_block = (
        f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    )
    return (
        "You are a fine-grained event classifier for short Apex Legends "
        "gameplay clips.\n"
        "You will see 1-10 still frames sampled from a clip plus a short "
        "JSON description of what the rule-based action segmenter inferred "
        "from keyboard / mouse alone (the `keys_pressed` / "
        "`mouse_buttons_pressed` fields use raw RawInput event names; map "
        "them to in-game actions using the keybind reference below).\n"
        "Your job is to pick the SINGLE most accurate leaf label from a "
        "constrained candidate list. If none of the candidates fit, return "
        "the parent label given in `parent_label`.\n\n"
        f"{keybind_block}"
        "Apex 3-tier taxonomy (for reference, descriptions included):\n"
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
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    extra_constraint = ""
    if parent == "unknown":
        extra_constraint = (
            "\nIMPORTANT: do NOT return any `idle_*` label "
            "(e.g. `idle_camera_only`, `idle_stationary`). Idle is handled by "
            "the rule classifier upstream. If the clip truly shows nothing "
            "happening and no candidate fits, return `unknown`."
        )
    return (
        "Pick the most appropriate label for this clip from `candidates`. "
        "If none clearly fit, return `parent_label`."
        f"{extra_constraint}\n\n"
        f"Segment metadata:\n```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        "Frames are attached below in temporal order (first -> last)."
    )


# ---------------------------------------------------------------------------
# Subdivision prompt construction (used only for parent == "unknown")
# ---------------------------------------------------------------------------
#
# When the rule classifier could not assign any label ("unknown") we don't
# trust a single global label for the whole segment -- the segment is often
# a stretch of dead-time stitched between two real events that share its
# parentage (e.g. "running to a loot box -> looting -> running on").
#
# Instead we ask Gemini to look at K frames in temporal order and tell us
# which event each *frame range* belongs to. The result is a list of
# contiguous sub-events covering frames 0..K-1, which the driver then
# rewrites as multiple smaller segments. The downstream merge step
# (:func:`merge_refined_unknowns`) naturally folds adjacent same-label
# sub-events back together so we don't over-fragment uniform clips.


def build_subdivide_system_prompt() -> str:
    """System prompt for the unknown-segment subdivision pass."""
    keybind_block = (
        f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    )
    return (
        "You are a fine-grained event SEGMENTER for short Apex Legends "
        "gameplay clips.\n"
        "You will see K still frames sampled in temporal order from a single "
        "clip that the upstream rule classifier could NOT categorize "
        "(parent label = 'unknown'). You will also receive a JSON description "
        "of the keyboard / mouse activity that occurred over the whole clip "
        "(`keys_pressed` / `mouse_buttons_pressed` use raw RawInput event "
        "names; consult the keybind reference below to map them to in-game "
        "actions).\n\n"
        "Task: walk through the K frames in order and decide which "
        "candidate event each frame belongs to. Group consecutive frames "
        "of the same event into a single `sub_event`. The clip may contain "
        "ONE event (output a single sub_event spanning [0, K-1]) or "
        "MULTIPLE events in sequence (e.g. running -> looting -> running).\n\n"
        "STRICT rules:\n"
        "1. Sub-events must be CONTIGUOUS and TOGETHER COVER ALL frames 0..K-1 "
        "(no gaps, no overlap). `start_frame_idx` is inclusive, "
        "`end_frame_idx` is inclusive.\n"
        "2. Labels MUST come from the `candidates` list in the user message. "
        "If a sub-range fits no candidate, label it `unknown` (the explicit "
        "escape valve that is always present in `candidates`).\n"
        "3. Do NOT return any `idle_*` label (e.g. `idle_camera_only`, "
        "`idle_stationary`). Idle is handled by the rule classifier upstream.\n"
        "4. Frames showing essentially the same activity / scene should be "
        "ONE sub_event, not several. Only split when the event class "
        "actually changes.\n\n"
        f"{keybind_block}"
        "Apex 3-tier taxonomy (for reference, descriptions included):\n"
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
    seg: Dict[str, Any], candidates: List[str], K: int
) -> str:
    """User prompt for the unknown-segment subdivision pass."""
    fields = {
        "K": K,
        "parent_label": "unknown",
        "candidates": candidates,
        "duration_s": seg.get("duration_s"),
        "tier1": seg.get("tier1"),
        "tier2": seg.get("tier2"),
        "keys_pressed": seg.get("keys_pressed", []),
        "mouse_buttons_pressed": seg.get("mouse_buttons_pressed", []),
        "wheel_events": seg.get("wheel_events", 0),
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    return (
        f"This is an 'unknown' clip. Walk through the {K} attached frames "
        f"(indices 0..{K - 1}) and output `sub_events` covering all of them.\n"
        "Do NOT return any `idle_*` label. If a sub-range fits no candidate, "
        "label it `unknown`. Merge runs of the same event into a single "
        "sub_event; only split when the event actually changes.\n\n"
        "Segment metadata (aggregated over the WHOLE clip):\n"
        f"```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        f"{K} frames attached below in temporal order "
        f"(frame 0 -> frame {K - 1})."
    )


def _normalize_sub_events(
    raw: Any, candidates: List[str], K: int
) -> List[Dict[str, Any]]:
    """Best-effort sanitisation of Gemini's `sub_events` array.

    Guarantees on the returned list:

    * non-empty
    * sorted by ``start_frame_idx``
    * each entry has integer ``start_frame_idx``, ``end_frame_idx`` in
      ``[0, K-1]`` with ``start <= end``
    * the entries together CONTIGUOUSLY cover ``[0, K-1]`` (gaps are
      absorbed into the previous entry, overlaps are clipped, the first
      entry is forced to start at 0 and the last to end at ``K-1``)
    * each ``label`` is either in ``candidates`` or ``"unknown"``
    * adjacent entries sharing the same label are merged

    On any irrecoverable problem, returns ``[{0..K-1: "unknown"}]``.
    """
    fallback = [{
        "start_frame_idx": 0, "end_frame_idx": K - 1,
        "label": "unknown", "confidence": None,
        "reason": "could not parse sub_events response",
    }]
    if not isinstance(raw, list) or not raw:
        return fallback

    candidates_set = set(candidates)
    parsed: List[Dict[str, Any]] = []
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        try:
            si = int(ev.get("start_frame_idx"))
            ei = int(ev.get("end_frame_idx"))
        except (TypeError, ValueError):
            continue
        si = max(0, min(si, K - 1))
        ei = max(si, min(ei, K - 1))
        raw_label = str(ev.get("label", "")).strip()
        label = raw_label if raw_label in candidates_set else normalize_label(raw_label)
        if label not in candidates_set:
            label = "unknown"
        try:
            conf = float(ev["confidence"]) if "confidence" in ev else None
        except (TypeError, ValueError, KeyError):
            conf = None
        parsed.append({
            "start_frame_idx": si,
            "end_frame_idx": ei,
            "label": label,
            "confidence": conf,
            "reason": str(ev.get("reason", ""))[:300],
        })

    if not parsed:
        return fallback

    parsed.sort(key=lambda e: (e["start_frame_idx"], e["end_frame_idx"]))

    # Resolve overlaps / gaps: each event takes [cur_start, min(its_end,
    # next_start - 1)]. Then force the first event to start at 0 and the
    # last to end at K-1 so coverage is exact.
    fixed: List[Dict[str, Any]] = []
    cur_start = 0
    for i, ev in enumerate(parsed):
        si = max(cur_start, ev["start_frame_idx"])
        if i + 1 < len(parsed):
            next_start = max(parsed[i + 1]["start_frame_idx"], si + 1)
            ei = min(ev["end_frame_idx"], next_start - 1)
        else:
            ei = K - 1
        ei = max(ei, si)
        fixed.append({**ev, "start_frame_idx": si, "end_frame_idx": ei})
        cur_start = ei + 1
        if cur_start > K - 1:
            break
    if not fixed:
        return fallback
    fixed[0]["start_frame_idx"] = 0
    fixed[-1]["end_frame_idx"] = K - 1

    # Merge adjacent same-label runs so we don't return [{sprint 0-2},
    # {sprint 3-5}] when the model emits two split events of the same kind.
    merged: List[Dict[str, Any]] = [fixed[0]]
    for ev in fixed[1:]:
        last = merged[-1]
        if ev["label"] == last["label"]:
            last["end_frame_idx"] = ev["end_frame_idx"]
            if (ev.get("confidence") or 0.0) > (last.get("confidence") or 0.0):
                last["confidence"] = ev["confidence"]
                last["reason"] = ev["reason"]
        else:
            merged.append(ev)
    return merged


# ---------------------------------------------------------------------------
# Per-segment Gemini call
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
    """Single Gemini call. Mirrors prompt_gemini_ref.py: no response_format,
    no temperature, single user message with text + images, parse JSON
    out of the response best-effort."""
    content = build_image_content(user_prompt, jpeg_blobs, detail=image_detail)
    # Prepend the system instruction directly into the user message so the
    # OpenAI-compatible providers that ignore `system` still
    # see it. The system role is kept too for the public Google endpoint.
    content = [{"type": "text", "text": system_prompt}] + content
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        timeout=timeout_s,
    )
    raw = resp.choices[0].message.content or ""
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
    # Candidate-set policy:
    # * non-unknown parents stay restricted to their taxonomy children
    #   (parent-bucket evidence from rule classifier is real and we don't
    #   want Gemini hallucinating a wildly different label).
    # * 'unknown' parents get the FULL non-idle leaf vocabulary -- because
    #   the rule pass had no opinion, we let the model + frames + keybind
    #   context speak. ``unknown`` itself stays in the list as an explicit
    #   escape valve for genuinely unclassifiable clips.
    strict_candidates = False
    if parent == "unknown":
        candidates = list(UNKNOWN_FULL_CANDIDATES)
        strict_candidates = True
    else:
        candidates = list(subtype_choices(parent))

    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": parent,
        "model": model,
        "k": frames_per_segment,
        "detail": image_detail,
        "stage": "refine",
        "video_start_ns": video_start_ns,
        "no_idle_for_unknown": strict_candidates,
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)
    if cached:
        return cached

    timestamps = sample_segment_timestamps(
        seg["start_timestamp_ns"], seg["end_timestamp_ns"], frames_per_segment
    )
    video_timestamps = [max(0.0, value - video_start_ns / NS_PER_S) for value in timestamps]
    jpeg_blobs: List[bytes] = []
    for value in video_timestamps:
        jpeg_blobs.append(extract_frame_jpeg(video_path, value, max_width=1024))

    user_prompt = build_user_prompt(seg, candidates)
    parsed = _call_gemini(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        jpeg_blobs=jpeg_blobs,
        image_detail=image_detail,
    )

    raw_label = str(parsed.get("label", "")).strip()
    label = raw_label if raw_label in candidates else normalize_label(raw_label)
    if strict_candidates:
        # Hard-enforce candidates for 'unknown' so the model can't escape
        # to an idle_* (or any other out-of-list) leaf by hallucinating
        # from the full taxonomy shown in the system prompt.
        if label not in candidates:
            label = parent
    elif label not in candidates and label not in LEAF_BY_LABEL:
        label = parent  # fall back to parent
    leaf = LEAF_BY_LABEL.get(label)
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    result = {
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else seg.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else seg.get("tier2"),
        "refined_confidence": confidence,
        "refined_reason": str(parsed.get("reason", ""))[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [round(value, 3) for value in video_timestamps],
    }
    cache.set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Unknown segment -> 1+ sub-segments via Gemini "subdivide" pass
# ---------------------------------------------------------------------------


def _interp_frame_index(seg: Dict[str, Any], t_ns: int) -> int:
    """Map an absolute mcap timestamp (ns) inside ``seg`` to the
    corresponding per-frame annotation index, linearly interpolated against
    the parent segment's ``[start_frame, end_frame]`` interval."""
    s_ns = int(seg["start_timestamp_ns"])
    e_ns = int(seg["end_timestamp_ns"])
    s_f = int(seg["start_frame"])
    e_f = int(seg["end_frame"])
    if e_ns <= s_ns or e_f <= s_f:
        return s_f
    t_ns = max(s_ns, min(t_ns, e_ns))
    frac = (t_ns - s_ns) / (e_ns - s_ns)
    return s_f + int(round(frac * (e_f - s_f)))


def _build_sub_segment(
    *,
    parent: Dict[str, Any],
    sub_event: Dict[str, Any],
    start_ns: int,
    end_ns: int,
    sample_timestamps_s: List[float],
    sub_index: int,
    sub_total: int,
    candidates: List[str],
    model: str,
) -> Dict[str, Any]:
    """Materialise one sub-segment dict from a Gemini ``sub_event`` record.

    ``start_ns`` / ``end_ns`` are computed by :func:`subdivide_unknown_segment`,
    which tiles sub-segments contiguously across the parent interval:

    * The first sub-segment starts at ``parent.start_timestamp_ns``.
    * The last sub-segment ends at ``parent.end_timestamp_ns``.
    * Internal cut points land at the midpoint between adjacent sampled
      frames; non-first sub-segments start at the previous sub-segment's
      ``end_ns`` so there are no temporal gaps.

    Aggregated action fields (``mouse_motion_total``, ``wheel_events``)
    are split proportionally to duration so that they sum back to the
    parent's value when the merge step folds adjacent same-label
    sub-segments together.
    """
    si = int(sub_event["start_frame_idx"])
    ei = int(sub_event["end_frame_idx"])
    label = str(sub_event.get("label") or "unknown")

    parent_start_ns = int(parent["start_timestamp_ns"])
    parent_end_ns = int(parent["end_timestamp_ns"])
    parent_dur_ns = max(parent_end_ns - parent_start_ns, 1)
    end_ns = max(start_ns + 1, end_ns)

    start_frame = _interp_frame_index(parent, start_ns)
    end_frame = max(start_frame, _interp_frame_index(parent, end_ns))
    duration_ns = end_ns - start_ns
    duration_s = round(duration_ns / NS_PER_S, 3)
    share = duration_ns / parent_dur_ns

    leaf = LEAF_BY_LABEL.get(label)
    new_seg: Dict[str, Any] = dict(parent)
    new_seg.update({
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": max(end_frame - start_frame + 1, 1),
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "duration_s": duration_s,
        # The rule label stays "unknown" so Apex_Legends_split_to_clips.py's
        # effective_label() still produces the `unknown_<refined>` clip
        # naming for sub-segments that came out of a subdivided unknown.
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
            round(sample_timestamps_s[k], 3) for k in range(si, ei + 1)
        ],
        "subdivided_from_unknown": True,
        "sub_event_index": sub_index,
        "sub_event_total": sub_total,
        "sub_event_frame_range": [si, ei],
        # Aggregated input fields: split proportionally so that re-merging
        # adjacent sub-segments back together recovers the parent's totals.
        "mouse_motion_total": round(
            float(parent.get("mouse_motion_total", 0.0) or 0.0) * share, 1
        ),
        "wheel_events": int(
            round(float(parent.get("wheel_events", 0) or 0) * share)
        ),
    })
    # Drop fields that no longer make sense on a child segment.
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
    """Ask Gemini to identify the distinct events inside an ``unknown``
    segment and return one segment dict per sub-event.

    If Gemini sees a single uniform event (or for any reason returns one
    sub-event spanning all K frames), this returns a single-element list
    whose dict is the original segment with ``refined_*`` fields filled
    in -- semantically identical to the single-label refine path.

    All returned dicts carry ``label == "unknown"`` so the downstream
    clip splitter still prefixes their filenames with ``unknown_``.
    """
    # Subdivision uses the full non-idle leaf vocabulary so the model can
    # spot any kind of event hidden inside an unknown stretch (combat
    # micro-bursts, traversal flavours, GUI flashes, ...). ``unknown``
    # stays as the explicit escape valve for genuinely unclassifiable
    # sub-ranges.
    candidates = list(UNKNOWN_FULL_CANDIDATES)
    K = max(int(frames_per_segment), 2)

    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": "unknown",
        "model": model,
        "k": K,
        "detail": image_detail,
        "stage": "subdivide_unknown_v2",
        "video_start_ns": video_start_ns,
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)

    timestamps = sample_segment_timestamps(
        seg["start_timestamp_ns"], seg["end_timestamp_ns"], K
    )
    video_timestamps = [max(0.0, value - video_start_ns / NS_PER_S) for value in timestamps]

    if cached is not None:
        sub_events = cached
    else:
        jpeg_blobs: List[bytes] = []
        for value in video_timestamps:
            jpeg_blobs.append(extract_frame_jpeg(video_path, value, max_width=1024))
        user_prompt = build_subdivide_user_prompt(seg, candidates, K)
        parsed = _call_gemini(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            jpeg_blobs=jpeg_blobs,
            image_detail=image_detail,
        )
        sub_events = _normalize_sub_events(parsed.get("sub_events"), candidates, K)
        cache.set(cache_key, sub_events)

    parent_start_ns = int(seg["start_timestamp_ns"])
    parent_end_ns = int(seg["end_timestamp_ns"])
    n = len(sub_events)
    out: List[Dict[str, Any]] = []
    prev_end_ns = parent_start_ns
    for i, ev in enumerate(sub_events):
        ei = int(ev["end_frame_idx"])
        if i == 0:
            start_ns = parent_start_ns
        else:
            start_ns = prev_end_ns
        if i == n - 1:
            end_ns = parent_end_ns
        else:
            next_si = int(sub_events[i + 1]["start_frame_idx"])
            mid_s = (timestamps[ei] + timestamps[next_si]) / 2.0
            end_ns = max(
                start_ns + 1,
                min(parent_end_ns - 1, int(round(mid_s * NS_PER_S))),
            )
        prev_end_ns = end_ns
        out.append(_build_sub_segment(
            parent=seg,
            sub_event=ev,
            start_ns=start_ns,
            end_ns=end_ns,
            sample_timestamps_s=video_timestamps,
            sub_index=i,
            sub_total=n,
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
    """Run the Stage 2 Gemini pass.

    Two routes inside the same parallel pool:

    * **single-label refine** (existing behaviour) for all
      ``vision_needed`` segments whose rule label is NOT ``"unknown"``
      (i.e. ``interact`` / ``pinging`` / ``using_health_item`` /
      ``throwing_grenade`` ...). Each such segment gets one
      ``refined_label`` written back in place.
    * **subdivision** (new, on by default when ``subdivide_unknown=True``)
      for segments whose rule label IS ``"unknown"``. Each such segment
      is replaced by 1+ sub-segments, one per Gemini-identified event
      inside it. The downstream ``merge_refined_unknowns`` step then
      folds adjacent same-label sub-events back together.

    Set ``subdivide_unknown=False`` to fall back to the old single-label
    behaviour for unknowns too. ``subdivide_frames_per_segment`` lets the
    subdivision route use a larger K than the single-label route
    (defaults to ``max(frames_per_segment, 10)`` so we have enough frames
    to actually pick out boundaries).
    """
    segments: List[Dict[str, Any]] = segments_payload["segments"]
    scene_name = segments_payload.get("scene_name") or video_path.stem
    if video_start_ns is None:
        video_start_ns = _resolve_video_start_ns(video_path, segments)
    if video_start_ns:
        print(f"video start (creation_time) = {video_start_ns} ns")
    else:
        print("video timestamps appear relative; using origin 0")

    targets: List[int] = [i for i, s in enumerate(segments) if s.get("vision_needed")]
    if limit is not None:
        targets = targets[:limit]

    unknown_targets = sum(
        1 for i in targets if segments[i].get("label") == "unknown"
    )
    other_targets = len(targets) - unknown_targets
    sub_K = (
        int(subdivide_frames_per_segment)
        if subdivide_frames_per_segment is not None
        else max(int(frames_per_segment), 10)
    )
    print(
        f"refining {len(targets)} / {len(segments)} segments via {model} "
        f"(single-label: {other_targets}, "
        f"{'subdivide' if subdivide_unknown else 'single-label'} unknown: "
        f"{unknown_targets}, K={frames_per_segment}"
        f"{f' / sub_K={sub_K}' if subdivide_unknown else ''})"
    )

    client = gemini_client(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir)
    system_prompt = build_system_prompt()
    subdivide_system_prompt = build_subdivide_system_prompt()

    # Each target -> the list of segment dicts it should expand to in the
    # output. For non-unknown / non-subdivided refines this list has length
    # 1; for subdivided unknowns it can be longer.
    target_outputs: Dict[int, List[Dict[str, Any]]] = {}
    target_lock = threading.Lock()
    error_count = 0
    subdivided_count = 0
    sub_segments_emitted = 0
    error_lock = threading.Lock()

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
                    frames_per_segment=sub_K,
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
        except Exception as e:  # noqa: BLE001
            with error_lock:
                error_count += 1
            sys.stderr.write(f"[seg {idx}] error: {e}\n")

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, idx): idx for idx in targets}
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="refine", unit="seg")
        for fut in iterator:
            fut.result()  # surface any unexpected non-handled exceptions

    # Rebuild the segment list, expanding subdivided unknowns into their
    # children while leaving everything else in place.
    out_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if i in target_outputs:
            out_segments.extend(target_outputs[i])
        else:
            out_segments.append(dict(seg))

    refined_count = sum(1 for s in out_segments if "refined_label" in s)
    label_changes: Dict[str, Dict[str, int]] = {}
    for s in out_segments:
        rl = s.get("refined_label")
        if rl and rl != s.get("label"):
            label_changes.setdefault(s["label"], {}).setdefault(rl, 0)
            label_changes[s["label"]][rl] += 1

    payload = dict(segments_payload)
    payload["segments"] = out_segments
    payload["refine_meta"] = {
        "model": model,
        "frames_per_segment": frames_per_segment,
        "subdivide_frames_per_segment": sub_K if subdivide_unknown else None,
        "image_detail": image_detail,
        "video_start_ns": video_start_ns,
        "vision_needed_total": sum(1 for s in segments if s.get("vision_needed")),
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
# Stage 3: Merge adjacent segments after Gemini vision refine.
# ---------------------------------------------------------------------------
#
# Walks the segment list left-to-right and collapses adjacent pairs ``(L, R)``
# under either of these rules:
#
# 1. **Same-label merge** -- ``effective_label(L) == effective_label(R)`` and
#    that label is not ``"unknown"`` (we intentionally keep two genuinely
#    unknown blocks separate so they remain visible / debuggable).
# 2. **Movement-group merge** -- both effective labels are in
#    ``MOVEMENT_GROUP = {walking, sprinting, crouch_walking}``. The merged
#    block adopts whichever movement label has the longer total duration
#    across the two sides.
#
# ``effective_label(s) = s["refined_label"] or s["label"]``. Sub-segment
# history is preserved one level deep so a merged super-segment carries the
# breakdown of what got coalesced together.


# Fields copied through from the segment that carries the refinement we keep.
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

# Locomotion labels considered interchangeable for the purpose of merging
# adjacent movement runs (sprint-to-walk transitions, brief crouch dips, etc.)
MOVEMENT_GROUP = frozenset({"walking", "sprinting", "crouch_walking"})

# Combat-engagement labels collapse into one continuous engagement block so a
# typical "ADS -> open fire -> hip-fire to finish" sequence reads as a single
# combat segment instead of three jittery sub-labels.
COMBAT_GROUP = frozenset({"aiming_down_sights", "firing_hipfire", "firing_ads"})

# The dropping-in sequence (still in the dropship -> skydive -> glide to land)
# is essentially one continuous "deploying into the match" action.
DROPPING_IN_GROUP = frozenset(
    {"dropship_transition", "skydiving", "gliding"}
)

# Adjacent segments whose effective labels both fall into the same group are
# merged. Each entry pairs the *kind* tag used in merge bookkeeping with the
# label set. ``_resolve_merge`` walks this list in order and stops at the
# first matching pair (the groups must stay disjoint -- they currently are).
MERGE_GROUPS: Tuple[Tuple[str, frozenset], ...] = (
    ("movement_group", MOVEMENT_GROUP),
    ("combat_group", COMBAT_GROUP),
    ("dropping_in_group", DROPPING_IN_GROUP),
)


def _effective_label(seg: Dict[str, Any]) -> str:
    """``refined_label`` if Gemini set one, otherwise the rule label."""
    rl = seg.get("refined_label")
    if isinstance(rl, str) and rl:
        return rl
    return seg.get("label", "") or ""


def _seg_components(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the constituent sub-segments of ``seg``, flattening any
    pre-existing ``sub_segments`` so the merge history stays one level deep."""
    existing = seg.get("sub_segments")
    if existing:
        return [dict(c) for c in existing]
    comp: Dict[str, Any] = {
        "label": seg.get("label"),
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "start_timestamp_ns": seg["start_timestamp_ns"],
        "end_timestamp_ns": seg["end_timestamp_ns"],
        "duration_s": seg.get("duration_s"),
    }
    rl = seg.get("refined_label")
    if rl:
        comp["refined_label"] = rl
    return [comp]


def _merge_pair(
    left: Dict[str, Any], right: Dict[str, Any], label: str
) -> Dict[str, Any]:
    """Build the merged segment dict for two adjacent segments collapsing
    onto ``label``."""
    leaf = LEAF_BY_LABEL.get(label)
    duration_ns = right["end_timestamp_ns"] - left["start_timestamp_ns"]
    keys = sorted(
        set(left.get("keys_pressed", []) or []) | set(right.get("keys_pressed", []) or [])
    )
    mbtns = sorted(
        set(left.get("mouse_buttons_pressed", []) or [])
        | set(right.get("mouse_buttons_pressed", []) or [])
    )

    # Only carry refined_* provenance through when one side's refined_label
    # actually equals the merged label. If neither side justifies the merged
    # label (e.g. movement-group merge where walking+sprinting -> sprinting),
    # we drop refined_* so the merged segment doesn't carry a contradictory
    # provenance.
    if left.get("refined_label") == label:
        ref_src: Optional[Dict[str, Any]] = left
    elif right.get("refined_label") == label:
        ref_src = right
    else:
        ref_src = None

    left_unknown_driven = left.get("label") == "unknown" or bool(
        left.get("merged_from_unknown")
    )
    right_unknown_driven = right.get("label") == "unknown" or bool(
        right.get("merged_from_unknown")
    )

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
        "mouse_buttons_pressed": mbtns,
        "mouse_motion_total": round(
            float(left.get("mouse_motion_total", 0.0) or 0.0)
            + float(right.get("mouse_motion_total", 0.0) or 0.0),
            1,
        ),
        "wheel_events": int(left.get("wheel_events", 0) or 0)
        + int(right.get("wheel_events", 0) or 0),
        "frame_file_start": left.get("frame_file_start"),
        "frame_file_end": right.get("frame_file_end"),
        "sub_segments": _seg_components(left) + _seg_components(right),
        "merged": True,
    }
    if left_unknown_driven or right_unknown_driven:
        merged["merged_from_unknown"] = True
    if ref_src is not None:
        for key in _REFINED_FIELDS:
            if key in ref_src:
                merged[key] = ref_src[key]
    return merged


def _recount_labels(
    segments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        label = seg.get("label", "unknown")
        c = counts.setdefault(label, {"segments": 0, "total_s": 0.0})
        c["segments"] += 1
        c["total_s"] += float(seg.get("duration_s") or 0.0)
    return {
        k: {"segments": int(v["segments"]), "total_s": round(v["total_s"], 2)}
        for k, v in counts.items()
    }


def _resolve_merge(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Decide whether ``left`` and ``right`` should be merged and, if so,
    return ``{"label": merged_label, "kind": <kind>}`` where ``kind`` is
    ``"same_label"`` or one of the group tags from :data:`MERGE_GROUPS`.
    Returns ``None`` when they must stay separate."""
    le = _effective_label(left)
    re = _effective_label(right)
    # Never merge two segments that are both still "unknown" -- keep them
    # visible so they can be inspected / further-refined later.
    if le == "unknown" and re == "unknown":
        return None
    if le and le == re:
        return {"label": le, "kind": "same_label"}
    for kind, group in MERGE_GROUPS:
        if le in group and re in group:
            # Pick whichever side's label has accumulated the longer top-level
            # duration so far (a tie keeps the left side's label).
            ld = float(left.get("duration_s") or 0.0)
            rd = float(right.get("duration_s") or 0.0)
            dominant = le if ld >= rd else re
            return {"label": dominant, "kind": kind}
    return None


def merge_refined_unknowns(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse adjacent segments after the Gemini refine pass.

    Two adjacent segments ``L``, ``R`` merge when *either*:

    1. **Same effective label** -- ``effective_label(L) == effective_label(R)``
       and that label is not ``"unknown"``. Two segments both still labelled
       ``"unknown"`` are *kept separate* (so the remaining unknowns stay
       visible for inspection).
    2. **Same merge group** -- both effective labels fall into the same
       :data:`MERGE_GROUPS` entry, e.g.

       - ``MOVEMENT_GROUP``    = {walking, sprinting, crouch_walking}
       - ``COMBAT_GROUP``      = {aiming_down_sights, firing_hipfire,
         firing_ads}
       - ``DROPPING_IN_GROUP`` = {dropship_transition, skydiving, gliding}

       The merged block adopts whichever side's label has the longer
       top-level duration (ties keep the left).

    ``effective_label(s)`` is ``s["refined_label"]`` if Gemini set one,
    otherwise ``s["label"]``.

    The pass is iterative via the running ``out[-1]``: a chain like
    ``walking | unknown->walking | sprinting | crouch_walking`` collapses
    into one block whose final label is the longest-duration movement member.
    The same chaining applies to the combat and dropping-in groups.

    The input ``payload`` is *not* mutated. The returned payload has
    recomputed ``label_counts`` and a ``merge_unknown_meta`` block under
    ``refine_meta`` describing what was merged.
    """
    segments: List[Dict[str, Any]] = list(payload.get("segments", []))
    if not segments:
        return dict(payload)

    out: List[Dict[str, Any]] = []
    merges_performed = 0
    merges_by_label: Dict[str, int] = {}
    merges_by_kind: Dict[str, int] = {}

    for seg in segments:
        cand = dict(seg)
        if not out:
            out.append(cand)
            continue
        prev = out[-1]
        decision = _resolve_merge(prev, cand)
        if decision is not None:
            label = decision["label"]
            kind = decision["kind"]
            out[-1] = _merge_pair(prev, cand, label)
            merges_performed += 1
            merges_by_label[label] = merges_by_label.get(label, 0) + 1
            merges_by_kind[kind] = merges_by_kind.get(kind, 0) + 1
        else:
            out.append(cand)

    new_payload = dict(payload)
    new_payload["segments"] = out
    new_payload["label_counts"] = _recount_labels(out)

    refine_meta = dict(new_payload.get("refine_meta") or {})
    refine_meta["merge_unknown_meta"] = {
        "merges_performed": merges_performed,
        "segments_before": len(segments),
        "segments_after": len(out),
        "merges_by_label": dict(
            sorted(merges_by_label.items(), key=lambda kv: -kv[1])
        ),
        "merges_by_kind": merges_by_kind,
        "merge_groups": {
            kind: sorted(group) for kind, group in MERGE_GROUPS
        },
        # Kept for backwards compat with consumers that read this key.
        "movement_group": sorted(MOVEMENT_GROUP),
    }
    new_payload["refine_meta"] = refine_meta
    return new_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("segments_json", type=Path,
                   help="Input segments JSON from 1a_Apex_Legends_segment_from_actions.py "
                        "(or an already-refined JSON when used with --merge-only).")
    p.add_argument("--video", type=Path, default=None,
                   help="Source MKV / MP4 video to grab frames from. "
                        "Required unless --merge-only is set.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write the refined segments JSON.")
    p.add_argument("--merge-unknowns", dest="merge_unknowns", action="store_true",
                   default=True,
                   help="After refine, merge each refined unknown into an "
                        "adjacent segment that shares the same effective "
                        "label (default: on).")
    p.add_argument("--no-merge-unknowns", dest="merge_unknowns",
                   action="store_false",
                   help="Disable the post-refine unknown-merge step.")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip the Gemini refine call entirely; read the input "
                        "as an already-refined JSON and only run the "
                        "unknown-merge step.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model id (default {DEFAULT_MODEL}).")
    p.add_argument("--frames-per-segment", type=int, default=5,
                   help="K frames sampled per non-unknown vision_needed "
                        "segment (default 5). Unknown segments use "
                        "--subdivide-frames-per-segment instead.")
    p.add_argument("--subdivide-unknown", dest="subdivide_unknown",
                   action="store_true", default=True,
                   help="For segments where the rule label is 'unknown', "
                        "ask Gemini to walk through K frames and emit a "
                        "list of contiguous sub-events; the original "
                        "segment is replaced by one segment per sub-event "
                        "(default: on). The downstream merge step folds "
                        "adjacent same-label sub-events back together.")
    p.add_argument("--no-subdivide-unknown", dest="subdivide_unknown",
                   action="store_false",
                   help="Disable per-frame subdivision of unknown segments; "
                        "fall back to single-label refinement for them too.")
    p.add_argument("--subdivide-frames-per-segment", type=int, default=10,
                   help="K frames sampled for the subdivision pass on "
                        "unknown segments (default 10). Should be >= 3 to "
                        "have any chance of catching event boundaries.")
    p.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Parallel Gemini calls (default 8).")
    p.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache"),
                   help="JSON cache directory (default .gemini_cache). Use empty string to disable.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N vision_needed segments (for smoke tests).")
    p.add_argument("--api-key", default=None,
                   help="Override GEMINI_API_KEY for this run.")
    p.add_argument("--base-url", default=None,
                   help="Override Gemini OpenAI-compat base URL.")
    p.add_argument(
        "--video-start-ns",
        type=int,
        default=None,
        help="Epoch-ns timestamp of the video's first frame; auto-detected when omitted. Pass 0 for relative timestamps.",
    )
    return p.parse_args()


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
        out_path = args.segments_json.with_name(
            args.segments_json.stem + "_merged.json"
        )
    else:
        out_path = args.segments_json.with_name(
            args.segments_json.stem + "_refined.json"
        )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    meta = refined.get("refine_meta") or {}
    if not args.merge_only:
        print(f"  refined: {meta.get('refined_total', 0)} / {meta.get('vision_needed_total', 0)}")
        print(f"  errors : {meta.get('errors', 0)}")
        if meta.get("subdivide_unknown_enabled"):
            print(
                f"  subdivide unknown: "
                f"{meta.get('unknown_inputs_subdivided', 0)} / "
                f"{meta.get('unknown_inputs_total', 0)} inputs split into 2+ events, "
                f"{meta.get('unknown_sub_segments_emitted', 0)} sub-segments emitted "
                f"({meta.get('segments_in', 0)} -> "
                f"{meta.get('segments_out_after_refine', 0)} segments)"
            )
        if meta.get("label_changes"):
            print("  label changes (parent -> refined: count):")
            for parent, sub in sorted(meta["label_changes"].items()):
                for rl, cnt in sorted(sub.items(), key=lambda kv: -kv[1]):
                    print(f"    {parent:22s} -> {rl:22s}  x {cnt}")

    mm = meta.get("merge_unknown_meta")
    if mm is not None:
        print(
            f"  merged unknowns: {mm['merges_performed']} merges, "
            f"{mm['segments_before']} -> {mm['segments_after']} segments"
        )
        if mm["merges_by_label"]:
            print("  merges by label (label: count):")
            for label, cnt in mm["merges_by_label"].items():
                print(f"    {label:22s}  x {cnt}")


if __name__ == "__main__":
    main()
