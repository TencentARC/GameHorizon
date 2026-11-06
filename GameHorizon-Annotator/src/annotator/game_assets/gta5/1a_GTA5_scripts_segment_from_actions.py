"""Action-only event segmentation for a Grand Theft Auto V recording.

Reads a per-frame annotation JSON (the ``*_annotations_per_frame.json`` format)
and emits a list of event segments together with frame indices where the event
label changes.

The classifier emits labels from :mod:`GTA5_taxonomy`. Labels that are reliable
from keyboard / mouse alone are emitted directly; labels that depend on visual
context (for example on-foot movement vs. vehicle driving vs. aircraft control)
are intentionally kept as medium-confidence parent buckets for a later vision
pass to refine.

Pipeline
--------
1. Load the annotation JSON and the GTA5 taxonomy.
2. Replay all keyboard / mouse events in chronological order to derive per-frame
   held state for each key / mouse button. OS auto-repeat presses are handled,
   and unmatched releases are tolerated.
3. Detect discrete "burst" events (reload, enter vehicle, character switch,
   interaction, grenade, camera, menu navigation, etc.) and lock their label for
   a short window so one keypress yields a real segment.
4. Classify every frame with a priority cascade:
   GUI/menu > combat > combat-locking bursts > other bursts > movement > idle.
5. Collapse consecutive identical labels into segments, smooth tiny non-atomic
   segments, and optionally merge consecutive taxonomy Tier2 groups.

Run::

    python 1a_GTA5_scripts_segment_from_actions.py annotations_per_frame.json \\
        --video gta5_scene.mkv --out gta5_scene_segments.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from GTA5_taxonomy import LEAF_BY_LABEL, EventLeaf, vision_needed as taxonomy_vision_needed


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

NS_PER_S = 1_000_000_000

BURST_LABELS: Tuple[str, ...] = (
    "reloading",
    "context_interaction",
    "enter_exit_vehicle",
    "character_switch",
    "weapon_wheel",
    "quick_switch_weapon",
    "sniper_or_camera_zoom",
    "throwing_explosive",
    "taking_cover",
    "vehicle_headlights",
    "vehicle_ducking",
    "change_camera_view",
    "radar_zoom",
    "cinematic_camera",
    "push_to_talk",
    "point_at",
    "menu_pause",
    "menu_interaction",
    "menu_phone",
    "menu_weapon_wheel",
    "menu_navigation",
)


@dataclass
class SegmenterConfig:
    # Idle thresholds
    idle_motion_pixels: float = 8.0
    idle_min_s: float = 1.0
    # Combat thresholds
    lmb_min_s: float = 0.05
    ads_min_s: float = 0.10
    # Burst lock windows
    reload_lock_s: float = 1.2
    interact_lock_s: float = 0.7
    enter_exit_vehicle_lock_s: float = 1.2
    character_switch_lock_s: float = 1.0
    weapon_wheel_lock_s: float = 0.5
    quick_switch_lock_s: float = 0.4
    explosive_lock_s: float = 0.8
    cover_lock_s: float = 0.7
    camera_lock_s: float = 0.5
    vehicle_control_lock_s: float = 0.5
    communication_lock_s: float = 0.5
    menu_lock_s: float = 0.7
    menu_navigation_lock_s: float = 0.35
    # Smoothing
    min_segment_s: float = 1.0
    unsmoothable_labels: Tuple[str, ...] = BURST_LABELS
    held_without_release_timeout_s: float = 0.8
    absorb_labels: Tuple[str, ...] = ()
    absorb_fallback_label: str = "idle_stationary"
    # Consecutive segments in these Tier2 groups are collapsed into one segment.
    # Most groups carry the dominant sub-label by duration; combat prefers
    # firing labels so a shootout does not get named as plain aiming.
    merge_tier2_groups: Tuple[str, ...] = (
        "on_foot_movement",
        "combat",
        "vehicle_driving",
        "aircraft",
        "camera_hud",
    )


# ---------------------------------------------------------------------------
# Per-frame state replay
# ---------------------------------------------------------------------------

WASD = {"KEY_W", "KEY_A", "KEY_S", "KEY_D"}
SPRINT_KEYS = {"LSHIFT", "RSHIFT"}
STEALTH_KEYS = {"LCONTROL", "RCONTROL"}
JUMP_KEYS = {"SPACE"}
AIRCRAFT_PITCH_KEYS = {"NUMPAD8", "NUMPAD5"}
AIRCRAFT_ROLL_KEYS = {"NUMPAD4", "NUMPAD6"}
MENU_NAV_KEYS = {"UP", "DOWN", "LEFT", "RIGHT", "RETURN", "BACK"}

LMB = "RI_MOUSE_LEFT_BUTTON"
RMB = "RI_MOUSE_RIGHT_BUTTON"
MMB = "RI_MOUSE_MIDDLE_BUTTON"


@dataclass
class FrameState:
    frame_index: int
    timestamp_ns: int
    next_timestamp_ns: int
    active_keys: Set[str] = field(default_factory=set)
    active_mouse: Set[str] = field(default_factory=set)
    mouse_motion: float = 0.0
    discrete_keys: Set[str] = field(default_factory=set)
    discrete_mouse: Set[str] = field(default_factory=set)
    wheel_events: int = 0


def _mouse_base_from_flag(flag_name: str) -> Optional[str]:
    if flag_name.endswith("_DOWN"):
        return flag_name[: -len("_DOWN")]
    if flag_name.endswith("_UP"):
        return flag_name[: -len("_UP")]
    return None


def _replay_state(frames: List[dict], cfg: SegmenterConfig) -> List[FrameState]:
    """Walk frames in order and produce a per-frame snapshot of held inputs."""
    held_since: Dict[str, int] = {}
    last_press_ts: Dict[str, int] = {}
    held_mouse: Dict[str, int] = {}
    timeout_ns = int(cfg.held_without_release_timeout_s * NS_PER_S)

    out: List[FrameState] = []
    for idx, frame in enumerate(frames):
        ts = frame["timestamp_ns"]
        nxt = frame.get("next_frame_timestamp_ns")
        if nxt is None:
            if idx > 0:
                prev_ts = frames[idx - 1]["timestamp_ns"]
                nxt = ts + (ts - prev_ts)
            else:
                nxt = ts
        st = FrameState(frame_index=frame["frame_index"], timestamp_ns=ts, next_timestamp_ns=nxt)

        for kb in frame.get("keyboard_events", []):
            key_name = kb.get("key_name")
            if not key_name:
                continue
            event_type = kb.get("event_type")
            event_ts = kb.get("timestamp_ns", ts)
            if event_type == "press":
                if key_name not in held_since:
                    held_since[key_name] = event_ts
                    st.discrete_keys.add(key_name)
                last_press_ts[key_name] = event_ts
            elif event_type == "release":
                held_since.pop(key_name, None)
                last_press_ts.pop(key_name, None)

        # Mouse button state and relative motion come from the raw input stream.
        # ``mouse_events`` normally contains only processed absolute cursor
        # coordinates, so keep it only as a compatibility fallback.
        raw_mouse = frame.get("raw_mouse_events")
        if not raw_mouse:
            raw_mouse = frame.get("mouse_events", [])
        for ms in raw_mouse:
            st.mouse_motion += abs(ms.get("dx", 0) or 0) + abs(ms.get("dy", 0) or 0)
            for flag_name in ms.get("button_flag_names", []) or []:
                if flag_name == "RI_MOUSE_WHEEL":
                    st.wheel_events += 1
                    continue
                base = _mouse_base_from_flag(flag_name)
                if base is None:
                    continue
                if flag_name.endswith("_DOWN"):
                    if base not in held_mouse:
                        st.discrete_mouse.add(base)
                    held_mouse[base] = ms.get("timestamp_ns", ts)
                elif flag_name.endswith("_UP"):
                    held_mouse.pop(base, None)

        active_keys: Set[str] = set()
        for key_name in list(held_since):
            last_ts = last_press_ts.get(key_name, 0)
            if ts - last_ts <= timeout_ns:
                active_keys.add(key_name)
            else:
                held_since.pop(key_name, None)
                last_press_ts.pop(key_name, None)
        st.active_keys = active_keys
        st.active_mouse = set(held_mouse.keys())
        out.append(st)
    return out


# ---------------------------------------------------------------------------
# Burst / lock-window detection
# ---------------------------------------------------------------------------

KEY_TO_BURST: Dict[str, Tuple[str, str]] = {
    "KEY_R": ("reloading", "reload_lock_s"),
    "KEY_E": ("context_interaction", "interact_lock_s"),
    "KEY_F": ("enter_exit_vehicle", "enter_exit_vehicle_lock_s"),
    "LMENU": ("character_switch", "character_switch_lock_s"),
    "KEY_G": ("throwing_explosive", "explosive_lock_s"),
    "KEY_Q": ("taking_cover", "cover_lock_s"),
    "KEY_H": ("vehicle_headlights", "vehicle_control_lock_s"),
    "KEY_X": ("vehicle_ducking", "vehicle_control_lock_s"),
    "KEY_V": ("change_camera_view", "camera_lock_s"),
    "KEY_Z": ("radar_zoom", "camera_lock_s"),
    "KEY_N": ("push_to_talk", "communication_lock_s"),
    "KEY_B": ("point_at", "communication_lock_s"),
    "KEY_P": ("menu_pause", "menu_lock_s"),
    "ESCAPE": ("menu_pause", "menu_lock_s"),
    "KEY_M": ("menu_interaction", "menu_lock_s"),
    "TAB": ("menu_weapon_wheel", "weapon_wheel_lock_s"),
    "UP": ("menu_navigation", "menu_navigation_lock_s"),
    "DOWN": ("menu_navigation", "menu_navigation_lock_s"),
    "LEFT": ("menu_navigation", "menu_navigation_lock_s"),
    "RIGHT": ("menu_navigation", "menu_navigation_lock_s"),
    "RETURN": ("menu_navigation", "menu_navigation_lock_s"),
    "BACK": ("menu_navigation", "menu_navigation_lock_s"),
}


def _burst_priority(label: str) -> int:
    """Higher priority wins when multiple bursts overlap on a frame."""
    return {
        "menu_pause":            120,
        "menu_interaction":      118,
        "menu_phone":            116,
        "menu_weapon_wheel":     114,
        "menu_navigation":       110,
        "throwing_explosive":     95,
        "enter_exit_vehicle":     92,
        "character_switch":       90,
        "reloading":              88,
        "context_interaction":     84,
        "taking_cover":            82,
        "vehicle_headlights":      78,
        "vehicle_ducking":         76,
        "change_camera_view":      74,
        "radar_zoom":              72,
        "cinematic_camera":        70,
        "quick_switch_weapon":     68,
        "sniper_or_camera_zoom":   68,
        "push_to_talk":            64,
        "point_at":                62,
    }.get(label, 50)


def _build_burst_intervals(
    frame_states: List[FrameState],
    cfg: SegmenterConfig,
) -> List[List[Tuple[str, int]]]:
    """For each frame return active ``(label, priority_rank)`` bursts."""
    n = len(frame_states)
    active: List[List[Tuple[str, int]]] = [[] for _ in range(n)]

    def add_interval(start_idx: int, label: str, lock_s: float, priority: int) -> None:
        end_ts = frame_states[start_idx].timestamp_ns + int(lock_s * NS_PER_S)
        i = start_idx
        while i < n and frame_states[i].timestamp_ns <= end_ts:
            active[i].append((label, priority))
            i += 1

    for idx, st in enumerate(frame_states):
        for key_name in st.discrete_keys:
            if key_name in KEY_TO_BURST:
                label, attr = KEY_TO_BURST[key_name]
                add_interval(idx, label, getattr(cfg, attr), _burst_priority(label))
        if MMB in st.discrete_mouse:
            add_interval(idx, "menu_phone", cfg.menu_lock_s, _burst_priority("menu_phone"))
        if st.wheel_events:
            label = "sniper_or_camera_zoom" if RMB in st.active_mouse else "quick_switch_weapon"
            add_interval(idx, label, cfg.quick_switch_lock_s, _burst_priority(label))
    return active


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

UNKNOWN_LABEL = "unknown"

# These burst labels can be interrupted by aim / fire input. Menu labels are not
# listed because GUI has higher priority and should suppress combat.
COMBAT_OVERRIDES_BURST: frozenset = frozenset({
    "reloading",
    "context_interaction",
    "enter_exit_vehicle",
    "character_switch",
    "taking_cover",
    "vehicle_headlights",
    "vehicle_ducking",
    "change_camera_view",
    "radar_zoom",
    "cinematic_camera",
    "quick_switch_weapon",
    "sniper_or_camera_zoom",
    "push_to_talk",
    "point_at",
})


def _is_held(state: FrameState, keys: Iterable[str]) -> bool:
    return any(key in state.active_keys for key in keys)


def _top_burst(bursts: List[Tuple[str, int]]) -> Optional[str]:
    if not bursts:
        return None
    return sorted(bursts, key=lambda label_priority: -label_priority[1])[0][0]


def _classify_frame(
    st: FrameState,
    bursts: List[Tuple[str, int]],
    rolling_no_input_s: float,
    cfg: SegmenterConfig,
) -> str:
    top_burst = _top_burst(bursts)

    # 1) GUI / menu overlays suppress everything else.
    if top_burst in {
        "menu_pause",
        "menu_interaction",
        "menu_phone",
        "menu_weapon_wheel",
        "menu_navigation",
    }:
        return top_burst
    if "KEY_P" in st.active_keys or "ESCAPE" in st.active_keys:
        return "menu_pause"
    if "KEY_M" in st.active_keys:
        return "menu_interaction"
    if "TAB" in st.active_keys:
        return "menu_weapon_wheel"
    if _is_held(st, MENU_NAV_KEYS):
        return "menu_navigation"
    if MMB in st.active_mouse:
        return "menu_phone"

    # 2) Combat. If a combat-locking burst is active, let the burst win.
    lmb_held = LMB in st.active_mouse
    rmb_held = RMB in st.active_mouse
    combat_input = lmb_held or rmb_held
    if combat_input and (top_burst is None or top_burst in COMBAT_OVERRIDES_BURST):
        if lmb_held and rmb_held:
            return "firing_ads"
        if lmb_held:
            return "firing_weapon"
        return "aiming"

    # 3) Burst event.
    if top_burst is not None:
        return top_burst

    # 4) Aircraft-specific controls are distinguishable from keyboard alone,
    # though the actual vehicle type still benefits from vision confirmation.
    if _is_held(st, AIRCRAFT_PITCH_KEYS):
        return "aircraft_pitching"
    if _is_held(st, AIRCRAFT_ROLL_KEYS):
        return "aircraft_rolling"

    # 5) Movement and general controls. Plain WASD is marked as on-foot movement
    # in the taxonomy, but it remains vision_needed because it may be swimming or
    # visually similar vehicle / aircraft movement in GTA5.
    moving = _is_held(st, WASD)
    sprinting = _is_held(st, SPRINT_KEYS)
    stealth = _is_held(st, STEALTH_KEYS)
    jumping = _is_held(st, JUMP_KEYS)

    if "KEY_C" in st.active_keys:
        return "looking_behind"
    if jumping and not moving:
        return "jumping_climbing"
    if jumping and moving:
        return "jumping_climbing"
    if stealth and moving:
        return "stealth_movement"
    if sprinting and moving:
        return "sprinting"
    if moving:
        return "walking"
    if stealth:
        return "stealth_movement"

    # 6) Idle only after a continuous no-input window.
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


def _frames_to_segments(frame_states: List[FrameState], labels: List[str]) -> List[Segment]:
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
    """Merge short non-atomic segments into a neighbour."""
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
    """Force-merge labels that should never own a standalone segment."""
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
    """Collapse contiguous segments sharing one configured taxonomy Tier2."""
    merge_set = set(merge_tier2_groups)
    if not segments or not merge_set:
        return list(segments)

    def tier2_of(label: str) -> Optional[str]:
        leaf = LEAF_BY_LABEL.get(label)
        return leaf.tier2 if leaf else None

    def dominant_label_for_run(run: List[Segment], tier2: str) -> str:
        dur_by_label: Dict[str, float] = defaultdict(float)
        first_seen: Dict[str, int] = {}
        seen_idx = 0
        for segment in run:
            parts = segment.components or [
                SubSegment(
                    label=segment.label,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                )
            ]
            for component in parts:
                dur_by_label[component.label] += component.duration_s
                first_seen.setdefault(component.label, seen_idx)
                seen_idx += 1

        if tier2 == "combat":
            for label in ("firing_ads", "firing_weapon", "aiming"):
                if dur_by_label.get(label, 0.0) > 0:
                    return label

        return max(
            dur_by_label.items(),
            key=lambda item: (item[1], -first_seen[item[0]]),
        )[0]

    out: List[Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        tier2 = tier2_of(seg.label)
        if tier2 not in merge_set:
            out.append(seg)
            i += 1
            continue
        j = i + 1
        while j < n and tier2_of(segments[j].label) == tier2:
            j += 1
        run = segments[i:j]
        if len(run) == 1:
            only = run[0]
            preferred = dominant_label_for_run(run, tier2)
            if preferred == only.label:
                out.append(only)
            else:
                out.append(
                    Segment(
                        start_frame=only.start_frame,
                        end_frame=only.end_frame,
                        start_ts=only.start_ts,
                        end_ts=only.end_ts,
                        label=preferred,
                        components=list(only.components),
                    )
                )
        else:
            dominant = dominant_label_for_run(run, tier2)
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


def _rolling_no_input(frame_states: List[FrameState], cfg: SegmenterConfig) -> List[float]:
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
# Main segmentation
# ---------------------------------------------------------------------------


def segment_video(
    annotations_path: Path,
    cfg: SegmenterConfig,
    *,
    video_name: Optional[str] = None,
) -> Dict[str, object]:
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
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
                "frame_index": frame_states[cur.start_frame].frame_index,
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
        counts[seg.label]["segments"] += 1
        counts[seg.label]["total_s"] += seg.duration_s
    return {
        label: {"segments": int(values["segments"]), "total_s": round(values["total_s"], 2)}
        for label, values in counts.items()
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
        mouse_buttons.update(st.discrete_mouse)
        wheel_events_total += st.wheel_events
        motion_total += st.mouse_motion
    if wheel_events_total:
        mouse_buttons.add("RI_MOUSE_WHEEL")

    frame_file_start = frames[seg.start_frame].get("frame_file") if seg.start_frame < len(frames) else None
    frame_file_end = frames[seg.end_frame].get("frame_file") if seg.end_frame < len(frames) else None

    out: Dict[str, object] = {
        "start_frame": frame_states[seg.start_frame].frame_index,
        "end_frame": frame_states[seg.end_frame].frame_index,
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

    distinct_sub_labels = {component.label for component in seg.components}
    if len(seg.components) > 1 and (
        len(distinct_sub_labels) > 1 or distinct_sub_labels != {seg.label}
    ):
        out["sub_segments"] = [
            {
                "label": component.label,
                "start_frame": component.start_frame,
                "end_frame": component.end_frame,
                "start_timestamp_ns": component.start_ts,
                "end_timestamp_ns": component.end_ts,
                "duration_s": round(component.duration_s, 3),
            }
            for component in seg.components
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
    rows = sorted(payload["label_counts"].items(), key=lambda item: -item[1]["total_s"])
    for label, counts in rows:
        print(f"  {label:24s}  {counts['segments']:5d}    {counts['total_s']:8.2f}s")
    print()
    print(f"first {top_n_segments} segments:")
    for seg in payload["segments"][:top_n_segments]:
        flag = "*" if seg.get("vision_needed") else " "
        print(
            f" {flag}[{seg['start_frame']:6d}->{seg['end_frame']:6d}] "
            f"{seg['duration_s']:6.2f}s  {seg['label']:24s} "
            f"keys={','.join(seg.get('keys_pressed', [])[:4]):<28s} "
            f"mouse={','.join(seg.get('mouse_buttons_pressed', []))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", type=Path, help="Path to *_annotations_per_frame.json")
    parser.add_argument("--video", type=str, default=None, help="Original video filename (for metadata).")
    parser.add_argument("--out", type=Path, default=None, help="Where to write segments JSON.")
    parser.add_argument("--min-duration-s", type=float, default=None, help="Override smoothing threshold.")
    parser.add_argument(
        "--no-default-tier2-merge",
        action="store_true",
        help="Disable every built-in Tier2 merge group; --merge-tier2 may add groups back.",
    )
    parser.add_argument(
        "--no-merge-movement",
        action="store_true",
        help="Do not collapse consecutive on-foot movement labels into one segment.",
    )
    parser.add_argument(
        "--no-merge-combat",
        action="store_true",
        help="Do not collapse consecutive aiming / firing labels into one combat segment.",
    )
    parser.add_argument(
        "--merge-tier2",
        action="append",
        default=None,
        metavar="TIER2",
        help="Additional taxonomy Tier2 groups to collapse (repeatable).",
    )
    parser.add_argument("--summary-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegmenterConfig()
    if args.min_duration_s is not None:
        cfg.min_segment_s = args.min_duration_s

    merge_groups: List[str] = [] if args.no_default_tier2_merge else list(cfg.merge_tier2_groups)
    if args.no_merge_movement and "on_foot_movement" in merge_groups:
        merge_groups.remove("on_foot_movement")
    if args.no_merge_combat and "combat" in merge_groups:
        merge_groups.remove("combat")
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
