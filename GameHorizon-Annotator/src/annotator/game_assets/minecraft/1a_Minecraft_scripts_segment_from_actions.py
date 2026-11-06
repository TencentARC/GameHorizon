"""Action-only event segmentation for a Minecraft recording.

Reads a per-frame annotation JSON (``*_annotations_per_frame.json``) and emits
event segments together with frame indices where the action label changes.

The classifier emits labels from :mod:`Minecraft_taxonomy` that are either
directly distinguishable from input alone (movement, hotbar selection, menu
toggles) or medium-confidence parent buckets that a later vision pass can
refine (``primary_action`` -> mining vs attacking,
``using_or_placing`` -> placing vs interacting vs item use).

Pipeline
--------
1. Load the annotation JSON and Minecraft taxonomy.
2. Replay keyboard / mouse events in chronological frame order to derive held
   keys, held mouse buttons, discrete key presses, wheel ticks, and mouse
   motion per frame.
3. Convert single-frame keypresses (hotbar keys, Q, E, T, ESC, F3, F5, middle
   click, wheel) into short lock-window "burst" segments.
4. Classify every frame with a priority cascade:
   GUI/menu toggles > discrete bursts > held block/item actions > movement >
   idle/unknown.
5. Collapse consecutive identical labels, smooth short non-discrete segments,
   and optionally merge consecutive segments that share a configured taxonomy
   tier2 group.

Run::

    python 1a_Minecraft_scripts_segment_from_actions.py annotations_per_frame.json \\
        --video recording.mkv --out recording_segments.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from Minecraft_taxonomy import LEAF_BY_LABEL, EventLeaf, vision_needed as taxonomy_vision_needed


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

NS_PER_S = 1_000_000_000

BURST_LABELS: Tuple[str, ...] = (
    "dropping_item",
    "select_hotbar_slot_1",
    "select_hotbar_slot_2",
    "select_hotbar_slot_3",
    "select_hotbar_slot_4",
    "select_hotbar_slot_5",
    "select_hotbar_slot_6",
    "select_hotbar_slot_7",
    "cycling_hotbar_slot",
    "pick_block",
    "inventory_screen",
    "chat_screen",
    "close_gui_or_pause",
    "showing_player_list",
    "toggling_debug_overlay",
    "changing_perspective",
)


@dataclass
class SegmenterConfig:
    # Idle thresholds
    idle_motion_pixels: float = 8.0
    idle_min_s: float = 1.0

    # Continuous action thresholds
    primary_action_min_s: float = 0.05
    use_action_min_s: float = 0.05

    # Burst lock windows
    hotbar_select_lock_s: float = 0.35
    hotbar_wheel_lock_s: float = 0.25
    drop_item_lock_s: float = 0.35
    pick_block_lock_s: float = 0.35
    inventory_lock_s: float = 0.8
    chat_lock_s: float = 0.8
    close_gui_lock_s: float = 0.8
    player_list_lock_s: float = 0.35
    debug_toggle_lock_s: float = 0.5
    perspective_lock_s: float = 0.5

    # Smoothing
    min_segment_s: float = 1.0
    unsmoothable_labels: Tuple[str, ...] = BURST_LABELS
    held_without_release_timeout_s: float = 0.8
    absorb_labels: Tuple[str, ...] = ()
    absorb_fallback_label: str = "idle_stationary"
    merge_tier2_groups: Tuple[str, ...] = ("movement",)


# ---------------------------------------------------------------------------
# Per-frame state replay
# ---------------------------------------------------------------------------

WASD = {"KEY_W", "KEY_A", "KEY_S", "KEY_D"}
SPRINT_KEYS = {"LCONTROL", "RCONTROL"}
SNEAK_KEYS = {"LSHIFT", "RSHIFT"}
JUMP_KEYS = {"SPACE"}

LMB = "RI_MOUSE_LEFT_BUTTON"
RMB = "RI_MOUSE_RIGHT_BUTTON"
MMB = "RI_MOUSE_MIDDLE_BUTTON"

BUTTON_FLAG_NAMES: Dict[str, Tuple[str, ...]] = {
    "1": ("RI_MOUSE_LEFT_BUTTON_DOWN",),
    "2": ("RI_MOUSE_LEFT_BUTTON_UP",),
    "4": ("RI_MOUSE_RIGHT_BUTTON_DOWN",),
    "8": ("RI_MOUSE_RIGHT_BUTTON_UP",),
    "16": ("RI_MOUSE_MIDDLE_BUTTON_DOWN",),
    "32": ("RI_MOUSE_MIDDLE_BUTTON_UP",),
    "1024": ("RI_MOUSE_WHEEL",),
}


@dataclass
class FrameState:
    frame_index: int
    timestamp_ns: int
    next_timestamp_ns: int
    active_keys: Set[str] = field(default_factory=set)
    active_mouse: Set[str] = field(default_factory=set)
    key_held_s: Dict[str, float] = field(default_factory=dict)
    mouse_held_s: Dict[str, float] = field(default_factory=dict)
    mouse_motion: float = 0.0
    discrete_keys: Set[str] = field(default_factory=set)
    mouse_downs: Set[str] = field(default_factory=set)
    wheel_events: int = 0


def _mouse_flag_names(ms: dict) -> Tuple[str, ...]:
    names = ms.get("button_flag_names")
    if isinstance(names, list) and names:
        return tuple(str(n) for n in names)
    flag = ms.get("button_flags")
    if flag is None:
        return ()
    return BUTTON_FLAG_NAMES.get(str(flag), ())


def _replay_state(frames: List[dict], cfg: SegmenterConfig) -> List[FrameState]:
    """Walk frames in order and produce a per-frame snapshot of held inputs."""
    held_since: Dict[str, int] = {}
    last_press_ts: Dict[str, int] = {}
    held_mouse: Dict[str, int] = {}
    timeout_ns = int(cfg.held_without_release_timeout_s * NS_PER_S)

    out: List[FrameState] = []
    for idx, f in enumerate(frames):
        ts = f["timestamp_ns"]
        nxt = f.get("next_frame_timestamp_ns")
        if nxt is None:
            if idx > 0:
                prev_ts = frames[idx - 1]["timestamp_ns"]
                nxt = ts + (ts - prev_ts)
            else:
                nxt = ts

        st = FrameState(
            frame_index=f.get("frame_index", idx),
            timestamp_ns=ts,
            next_timestamp_ns=nxt,
        )

        for kb in f.get("keyboard_events", []):
            kn = kb.get("key_name")
            if not kn:
                continue
            ev_type = kb.get("event_type")
            ets = kb.get("timestamp_ns", ts)
            if ev_type == "press":
                if kn not in held_since:
                    held_since[kn] = ets
                    st.discrete_keys.add(kn)
                last_press_ts[kn] = ets
            elif ev_type == "release":
                held_since.pop(kn, None)
                last_press_ts.pop(kn, None)

        for ms in f.get("mouse_events", []):
            st.mouse_motion += abs(ms.get("dx", 0) or 0) + abs(ms.get("dy", 0) or 0)
            for fl in _mouse_flag_names(ms):
                if fl.endswith("_BUTTON_DOWN"):
                    base = fl[: -len("_DOWN")]
                    held_mouse[base] = ms.get("timestamp_ns", ts)
                    st.mouse_downs.add(base)
                elif fl.endswith("_BUTTON_UP"):
                    base = fl[: -len("_UP")]
                    held_mouse.pop(base, None)
                elif fl == "RI_MOUSE_WHEEL":
                    st.wheel_events += 1

        active_keys: Set[str] = set()
        key_held_s: Dict[str, float] = {}
        for kn, start_ts in list(held_since.items()):
            lp = last_press_ts.get(kn, 0)
            if ts - lp <= timeout_ns:
                active_keys.add(kn)
                key_held_s[kn] = max(0.0, (ts - start_ts) / NS_PER_S)
            else:
                held_since.pop(kn, None)
                last_press_ts.pop(kn, None)

        st.active_keys = active_keys
        st.key_held_s = key_held_s
        st.active_mouse = set(held_mouse.keys())
        st.mouse_held_s = {
            button: max(0.0, (ts - start_ts) / NS_PER_S)
            for button, start_ts in held_mouse.items()
        }
        out.append(st)

    return out


# ---------------------------------------------------------------------------
# Burst / lock-window detection
# ---------------------------------------------------------------------------

KEY_TO_BURST: Dict[str, Tuple[str, str]] = {
    "KEY_Q": ("dropping_item", "drop_item_lock_s"),
    "KEY_1": ("select_hotbar_slot_1", "hotbar_select_lock_s"),
    "KEY_2": ("select_hotbar_slot_2", "hotbar_select_lock_s"),
    "KEY_3": ("select_hotbar_slot_3", "hotbar_select_lock_s"),
    "KEY_4": ("select_hotbar_slot_4", "hotbar_select_lock_s"),
    "KEY_5": ("select_hotbar_slot_5", "hotbar_select_lock_s"),
    "KEY_6": ("select_hotbar_slot_6", "hotbar_select_lock_s"),
    "KEY_7": ("select_hotbar_slot_7", "hotbar_select_lock_s"),
    "KEY_E": ("inventory_screen", "inventory_lock_s"),
    "KEY_T": ("chat_screen", "chat_lock_s"),
    "ESCAPE": ("close_gui_or_pause", "close_gui_lock_s"),
    "ESC": ("close_gui_or_pause", "close_gui_lock_s"),
    "F3": ("toggling_debug_overlay", "debug_toggle_lock_s"),
    "F5": ("changing_perspective", "perspective_lock_s"),
}


def _burst_priority(label: str) -> int:
    """Higher = wins when multiple bursts overlap on the same frame."""
    return {
        "close_gui_or_pause": 100,
        "chat_screen": 98,
        "inventory_screen": 96,
        "showing_player_list": 92,
        "toggling_debug_overlay": 90,
        "changing_perspective": 88,
        "pick_block": 84,
        "dropping_item": 82,
        "cycling_hotbar_slot": 80,
        "select_hotbar_slot_1": 78,
        "select_hotbar_slot_2": 78,
        "select_hotbar_slot_3": 78,
        "select_hotbar_slot_4": 78,
        "select_hotbar_slot_5": 78,
        "select_hotbar_slot_6": 78,
        "select_hotbar_slot_7": 78,
    }.get(label, 50)


def _build_burst_intervals(
    frame_states: List[FrameState],
    cfg: SegmenterConfig,
) -> List[List[Tuple[str, int]]]:
    """For each frame return active ``(label, priority_rank)`` bursts."""
    n = len(frame_states)
    active: List[List[Tuple[str, int]]] = [[] for _ in range(n)]

    def add_interval(start_idx: int, label: str, lock_s: float) -> None:
        end_ts = frame_states[start_idx].timestamp_ns + int(lock_s * NS_PER_S)
        i = start_idx
        priority = _burst_priority(label)
        while i < n and frame_states[i].timestamp_ns <= end_ts:
            active[i].append((label, priority))
            i += 1

    for idx, st in enumerate(frame_states):
        for kn in st.discrete_keys:
            if kn in KEY_TO_BURST:
                label, attr = KEY_TO_BURST[kn]
                add_interval(idx, label, getattr(cfg, attr))

        if st.wheel_events:
            add_interval(idx, "cycling_hotbar_slot", cfg.hotbar_wheel_lock_s)
        if MMB in st.mouse_downs:
            add_interval(idx, "pick_block", cfg.pick_block_lock_s)
        if "TAB" in st.discrete_keys:
            add_interval(idx, "showing_player_list", cfg.player_list_lock_s)

    return active


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

UNKNOWN_LABEL = "unknown"


def _is_held(state: FrameState, keys: Iterable[str]) -> bool:
    return any(k in state.active_keys for k in keys)


def _top_burst(bursts: List[Tuple[str, int]]) -> Optional[str]:
    if not bursts:
        return None
    return sorted(bursts, key=lambda lp: -lp[1])[0][0]


def _classify_frame(
    st: FrameState,
    bursts: List[Tuple[str, int]],
    rolling_no_input_s: float,
    cfg: SegmenterConfig,
) -> str:
    top_burst = _top_burst(bursts)
    if top_burst is not None:
        return top_burst

    if "TAB" in st.active_keys:
        return "showing_player_list"

    lmb_held = LMB in st.active_mouse
    rmb_held = RMB in st.active_mouse
    lmb_long_enough = st.mouse_held_s.get(LMB, 0.0) >= cfg.primary_action_min_s or LMB in st.mouse_downs
    rmb_long_enough = st.mouse_held_s.get(RMB, 0.0) >= cfg.use_action_min_s or RMB in st.mouse_downs

    if lmb_held and lmb_long_enough:
        return "primary_action"
    if rmb_held and rmb_long_enough:
        return "using_or_placing"

    moving = _is_held(st, WASD)
    sprinting = _is_held(st, SPRINT_KEYS)
    sneaking = _is_held(st, SNEAK_KEYS)
    jumping = _is_held(st, JUMP_KEYS)

    if jumping and moving:
        return "jumping"
    if sprinting and moving:
        return "sprinting"
    if sneaking:
        return "sneaking"
    if moving:
        return "walking"
    if jumping:
        return "jumping"

    if rolling_no_input_s >= cfg.idle_min_s:
        return "idle_stationary"

    return UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Segment building & smoothing
# ---------------------------------------------------------------------------


@dataclass
class SubSegment:
    label: str
    start_frame: int
    end_frame: int
    start_ts: int
    end_ts: int

    @property
    def duration_s(self) -> float:
        return (self.end_ts - self.start_ts) / NS_PER_S


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    start_ts: int
    end_ts: int
    label: str
    components: List[SubSegment] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_s(self) -> float:
        return (self.end_ts - self.start_ts) / NS_PER_S


def _frames_to_segments(
    frame_states: List[FrameState],
    labels: List[str],
) -> List[Segment]:
    def make(start: int, end: int, label: str) -> Segment:
        start_ts = frame_states[start].timestamp_ns
        end_ts = frame_states[end].next_timestamp_ns
        sub = SubSegment(
            label=label,
            start_frame=frame_states[start].frame_index,
            end_frame=frame_states[end].frame_index,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return Segment(
            start_frame=start,
            end_frame=end,
            start_ts=start_ts,
            end_ts=end_ts,
            label=label,
            components=[sub],
        )

    if not labels:
        return []

    segments: List[Segment] = []
    cur_label = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            segments.append(make(cur_start, i - 1, cur_label))
            cur_start = i
            cur_label = labels[i]
    segments.append(make(cur_start, len(labels) - 1, cur_label))
    return segments


def _absorb_into_prev(prev_seg: Segment, seg: Segment) -> None:
    prev_seg.end_frame = seg.end_frame
    prev_seg.end_ts = seg.end_ts
    prev_seg.components.extend(seg.components)


def _absorb_into_next(next_seg: Segment, seg: Segment) -> None:
    next_seg.start_frame = seg.start_frame
    next_seg.start_ts = seg.start_ts
    next_seg.components = list(seg.components) + next_seg.components


def _merge_both(prev_seg: Segment, seg: Segment, next_seg: Segment) -> None:
    prev_seg.end_frame = next_seg.end_frame
    prev_seg.end_ts = next_seg.end_ts
    prev_seg.components.extend(seg.components)
    prev_seg.components.extend(next_seg.components)


def _smooth_segments(
    segments: List[Segment],
    min_duration_s: float,
    unsmoothable_labels: Iterable[str] = (),
) -> List[Segment]:
    """Merge short segments into a neighbour unless the label is discrete."""
    if not segments:
        return segments
    unsmoothable_set = set(unsmoothable_labels)
    out: List[Segment] = list(segments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            seg = out[i]
            if seg.duration_s >= min_duration_s or seg.label in unsmoothable_set:
                i += 1
                continue

            prev_seg = out[i - 1] if i > 0 else None
            next_seg = out[i + 1] if i + 1 < len(out) else None
            target = None
            if prev_seg and next_seg:
                if prev_seg.label == next_seg.label:
                    target = "merge_both"
                elif prev_seg.duration_s >= next_seg.duration_s:
                    target = "prev"
                else:
                    target = "next"
            elif prev_seg:
                target = "prev"
            elif next_seg:
                target = "next"

            if target == "prev":
                _absorb_into_prev(prev_seg, seg)  # type: ignore[arg-type]
                out.pop(i)
                changed = True
            elif target == "next":
                _absorb_into_next(next_seg, seg)  # type: ignore[arg-type]
                out.pop(i)
                changed = True
            elif target == "merge_both":
                _merge_both(prev_seg, seg, next_seg)  # type: ignore[arg-type]
                del out[i : i + 2]
                changed = True
            else:
                i += 1
    return out


def _absorb_segments(
    segments: List[Segment],
    absorb_labels: Iterable[str],
    fallback_label: str,
) -> List[Segment]:
    """Force-merge any configured non-event labels into a neighbour."""
    absorb_set = set(absorb_labels)
    if not segments or not absorb_set:
        return list(segments)

    out: List[Segment] = list(segments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            seg = out[i]
            if seg.label not in absorb_set:
                i += 1
                continue

            prev_seg = out[i - 1] if i > 0 else None
            next_seg = out[i + 1] if i + 1 < len(out) else None
            target = None
            if prev_seg and next_seg:
                if prev_seg.label == next_seg.label:
                    target = "merge_both"
                elif prev_seg.duration_s >= next_seg.duration_s:
                    target = "prev"
                else:
                    target = "next"
            elif prev_seg:
                target = "prev"
            elif next_seg:
                target = "next"

            if target == "prev":
                _absorb_into_prev(prev_seg, seg)  # type: ignore[arg-type]
                out.pop(i)
                changed = True
            elif target == "next":
                _absorb_into_next(next_seg, seg)  # type: ignore[arg-type]
                out.pop(i)
                changed = True
            elif target == "merge_both":
                _merge_both(prev_seg, seg, next_seg)  # type: ignore[arg-type]
                del out[i : i + 2]
                changed = True
            else:
                seg.label = fallback_label
                for comp in seg.components:
                    comp.label = fallback_label
                i += 1
    return out


def _merge_segments_by_tier2(
    segments: List[Segment],
    merge_tier2_groups: Iterable[str],
) -> List[Segment]:
    """Collapse contiguous segments sharing selected taxonomy tier2 groups."""
    merge_set = set(merge_tier2_groups)
    if not segments or not merge_set:
        return list(segments)

    def tier2_of(label: str) -> Optional[str]:
        leaf = LEAF_BY_LABEL.get(label)
        return leaf.tier2 if leaf else None

    out: List[Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        t2 = tier2_of(seg.label)
        if t2 not in merge_set:
            out.append(seg)
            i += 1
            continue

        j = i + 1
        while j < n and tier2_of(segments[j].label) == t2:
            j += 1
        run = segments[i:j]
        if len(run) == 1:
            out.append(run[0])
        else:
            dur_by_label: Dict[str, float] = defaultdict(float)
            first_seen: Dict[str, int] = {}
            for k, s in enumerate(run):
                dur_by_label[s.label] += s.duration_s
                first_seen.setdefault(s.label, k)
            dominant = max(
                dur_by_label.items(),
                key=lambda kv: (kv[1], -first_seen[kv[0]]),
            )[0]
            merged_components: List[SubSegment] = []
            for s in run:
                merged_components.extend(s.components)
            out.append(
                Segment(
                    start_frame=run[0].start_frame,
                    end_frame=run[-1].end_frame,
                    start_ts=run[0].start_ts,
                    end_ts=run[-1].end_ts,
                    label=dominant,
                    components=merged_components,
                )
            )
        i = j
    return out


# ---------------------------------------------------------------------------
# Idle helpers
# ---------------------------------------------------------------------------


def _rolling_no_input(
    frame_states: List[FrameState], cfg: SegmenterConfig
) -> List[float]:
    """Seconds of continuous no-input ending at each frame."""
    out = [0.0] * len(frame_states)
    no_input_start_ts: Optional[int] = None
    for i, st in enumerate(frame_states):
        has_input = (
            bool(st.active_keys)
            or bool(st.active_mouse)
            or st.wheel_events > 0
            or st.mouse_motion >= cfg.idle_motion_pixels
        )
        if has_input:
            no_input_start_ts = None
            out[i] = 0.0
        else:
            if no_input_start_ts is None:
                no_input_start_ts = st.timestamp_ns
            out[i] = (st.timestamp_ns - no_input_start_ts) / NS_PER_S
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def segment_video(
    annotations_path: Path,
    cfg: SegmenterConfig,
    *,
    video_name: Optional[str] = None,
) -> Dict[str, object]:
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{annotations_path} does not contain a top-level frames list")
    if not frames:
        raise ValueError(f"{annotations_path} contains no frames")

    frame_states = _replay_state(frames, cfg)
    bursts_per_frame = _build_burst_intervals(frame_states, cfg)
    no_input_s = _rolling_no_input(frame_states, cfg)

    raw_labels = [
        _classify_frame(
            st,
            bursts_per_frame[i],
            rolling_no_input_s=no_input_s[i],
            cfg=cfg,
        )
        for i, st in enumerate(frame_states)
    ]

    raw_segments = _frames_to_segments(frame_states, raw_labels)
    absorbed = _absorb_segments(raw_segments, cfg.absorb_labels, cfg.absorb_fallback_label)
    smoothed = _smooth_segments(absorbed, cfg.min_segment_s, cfg.unsmoothable_labels)
    final_segments = _merge_segments_by_tier2(smoothed, cfg.merge_tier2_groups)

    transitions = []
    for i in range(1, len(final_segments)):
        prev, cur = final_segments[i - 1], final_segments[i]
        transitions.append(
            {
                "frame_index": cur.start_frame,
                "timestamp_ns": cur.start_ts,
                "from": prev.label,
                "to": cur.label,
            }
        )

    duration_s = (frames[-1]["timestamp_ns"] - frames[0]["timestamp_ns"]) / NS_PER_S
    fps = len(frames) / duration_s if duration_s > 0 else 0.0

    return {
        "video": video_name or data.get("source_mcap"),
        "scene_name": data.get("scene_name"),
        "num_frames": len(frames),
        "fps_estimate": round(fps, 3),
        "interval_start_ns": data.get("interval_start_ns"),
        "interval_end_ns": data.get("interval_end_ns"),
        "config": cfg.__dict__,
        "label_counts": _label_counts(final_segments),
        "segments": [_segment_to_dict(seg, frame_states, frames) for seg in final_segments],
        "transitions": transitions,
    }


def _label_counts(segments: List[Segment]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"segments": 0, "total_s": 0.0})
    for seg in segments:
        c = counts[seg.label]
        c["segments"] += 1
        c["total_s"] += seg.duration_s
    return {
        k: {"segments": int(v["segments"]), "total_s": round(v["total_s"], 2)}
        for k, v in counts.items()
    }


def _segment_to_dict(
    seg: Segment,
    frame_states: List[FrameState],
    frames: List[dict],
) -> Dict[str, object]:
    leaf: Optional[EventLeaf] = LEAF_BY_LABEL.get(seg.label)

    keys_pressed: Set[str] = set()
    mouse_buttons: Set[str] = set()
    wheel_events_total = 0
    motion_total = 0.0
    for st in frame_states[seg.start_frame : seg.end_frame + 1]:
        keys_pressed.update(st.active_keys)
        keys_pressed.update(st.discrete_keys)
        mouse_buttons.update(st.active_mouse)
        mouse_buttons.update(f"{m}_DOWN" for m in st.mouse_downs)
        wheel_events_total += st.wheel_events
        motion_total += st.mouse_motion
    if wheel_events_total:
        mouse_buttons.add("RI_MOUSE_WHEEL")

    frame_file_start = frames[seg.start_frame].get("frame_file") if seg.start_frame < len(frames) else None
    frame_file_end = frames[seg.end_frame].get("frame_file") if seg.end_frame < len(frames) else None

    out: Dict[str, object] = {
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "num_frames": seg.num_frames,
        "start_timestamp_ns": seg.start_ts,
        "end_timestamp_ns": seg.end_ts,
        "duration_s": round(seg.duration_s, 3),
        "label": seg.label,
        "tier1": leaf.tier1 if leaf else None,
        "tier2": leaf.tier2 if leaf else None,
        "action_distinguishable": leaf.action_distinguishable if leaf else "n/a",
        "vision_needed": taxonomy_vision_needed(seg.label),
        "vision_subtypes": list(leaf.vision_subtypes) if leaf else [],
        "keys_pressed": sorted(keys_pressed),
        "mouse_buttons_pressed": sorted(mouse_buttons),
        "mouse_motion_total": round(motion_total, 1),
        "wheel_events": wheel_events_total,
        "frame_file_start": frame_file_start,
        "frame_file_end": frame_file_end,
    }

    distinct_sub_labels = {c.label for c in seg.components}
    if len(seg.components) > 1 and (
        len(distinct_sub_labels) > 1 or distinct_sub_labels != {seg.label}
    ):
        out["sub_segments"] = [
            {
                "label": c.label,
                "start_frame": c.start_frame,
                "end_frame": c.end_frame,
                "start_timestamp_ns": c.start_ts,
                "end_timestamp_ns": c.end_ts,
                "duration_s": round(c.duration_s, 3),
            }
            for c in seg.components
        ]
    return out


def _print_summary(payload: Dict[str, object], top_n_segments: int = 30) -> None:
    print(f"video         : {payload['video']}")
    print(f"scene         : {payload['scene_name']}")
    print(f"frames        : {payload['num_frames']} @ ~{payload['fps_estimate']} fps")
    print(f"#segments     : {len(payload['segments'])}")
    print(f"#transitions  : {len(payload['transitions'])}")
    n_vision = sum(1 for s in payload["segments"] if s.get("vision_needed"))
    print(f"#vision_needed: {n_vision}  ({n_vision * 100.0 / max(len(payload['segments']), 1):.1f}%)")
    print()
    print("label distribution (segments / total_s):")
    rows = sorted(
        payload["label_counts"].items(),
        key=lambda kv: -kv[1]["total_s"],
    )
    for label, c in rows:
        print(f"  {label:24s}  {c['segments']:5d}    {c['total_s']:8.2f}s")
    print()
    print(f"first {top_n_segments} segments:")
    for seg in payload["segments"][:top_n_segments]:
        flag = "*" if seg.get("vision_needed") else " "
        print(
            f" {flag}[{seg['start_frame']:6d}->{seg['end_frame']:6d}] "
            f"{seg['duration_s']:6.2f}s  {seg['label']:24s} "
            f"keys={','.join(seg.get('keys_pressed', [])[:4]):<24s} "
            f"mouse={','.join(seg.get('mouse_buttons_pressed', []))}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("annotations", type=Path, help="Path to *_annotations_per_frame.json")
    p.add_argument("--video", type=str, default=None, help="Original video filename (for metadata).")
    p.add_argument("--out", type=Path, default=None, help="Where to write segments JSON.")
    p.add_argument("--min-duration-s", type=float, default=None, help="Override smoothing threshold (seconds).")
    p.add_argument(
        "--no-default-tier2-merge",
        action="store_true",
        help="Disable every built-in Tier2 merge group; --merge-tier2 may add groups back.",
    )
    p.add_argument(
        "--no-merge-movement",
        action="store_true",
        help="Do not collapse consecutive walking / sprinting / sneaking / jumping segments.",
    )
    p.add_argument(
        "--merge-tier2",
        action="append",
        default=None,
        metavar="TIER2",
        help="Additional taxonomy tier2 groups to collapse (repeatable).",
    )
    p.add_argument("--summary-rows", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegmenterConfig()
    if args.min_duration_s is not None:
        cfg.min_segment_s = args.min_duration_s

    merge_groups: List[str] = [] if args.no_default_tier2_merge else list(cfg.merge_tier2_groups)
    if args.no_merge_movement and "movement" in merge_groups:
        merge_groups.remove("movement")
    if args.merge_tier2:
        for group in args.merge_tier2:
            if group not in merge_groups:
                merge_groups.append(group)
    cfg.merge_tier2_groups = tuple(merge_groups)

    payload = segment_video(args.annotations, cfg, video_name=args.video)

    out_path = args.out or args.annotations.with_name(
        f"{(args.video or args.annotations.stem)}_segments.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")
    print()
    _print_summary(payload, top_n_segments=args.summary_rows)


if __name__ == "__main__":
    main()
