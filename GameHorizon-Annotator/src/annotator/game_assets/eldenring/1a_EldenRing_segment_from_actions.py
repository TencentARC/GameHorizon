#!/usr/bin/env python3
"""Action-grounded event segmentation for ELDEN RING (base game).

The input is an ``annotations_per_frame.json`` file containing absolute frame
timestamps plus keyboard, processed mouse, and raw mouse events.  The output is
the Stage-1 segments JSON consumed by ``1b_EldenRing_refine_with_gemini.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from EldenRing_taxonomy import (
    LEAF_BY_LABEL,
    EventLeaf,
    vision_needed as taxonomy_vision_needed,
)


NS_PER_S = 1_000_000_000
WASD = frozenset({"KEY_W", "KEY_A", "KEY_S", "KEY_D"})
SPRINT_KEYS = frozenset({"TAB"})
LMB = "RI_MOUSE_LEFT_BUTTON"
RMB = "RI_MOUSE_RIGHT_BUTTON"

ATOMIC_LABELS: Tuple[str, ...] = (
    "light_attack",
    "dodge_roll",
    "jumping",
    "skill_ash_of_war",
    "use_item",
    "interact",
    "switch_armament",
    "switch_right_armament",
    "switch_left_armament",
    "switch_sorcery",
    "switch_quick_item",
)


@dataclass
class SegmenterConfig:
    idle_motion_pixels: float = 8.0
    idle_min_s: float = 1.0
    held_without_release_timeout_s: float = 0.8
    min_segment_s: float = 0.8
    light_attack_lock_s: float = 0.70
    attack_release_tail_s: float = 0.35
    dodge_lock_s: float = 0.55
    jump_lock_s: float = 0.70
    skill_lock_s: float = 0.75
    use_item_lock_s: float = 1.10
    interact_lock_s: float = 0.75
    switch_lock_s: float = 0.40
    unsmoothable_labels: Tuple[str, ...] = ATOMIC_LABELS
    absorb_labels: Tuple[str, ...] = ()
    absorb_fallback_label: str = "idle_stationary"
    merge_tier2_groups: Tuple[str, ...] = ("traversal", "combat")


@dataclass
class FrameState:
    pos: int
    frame_index: int
    timestamp_ns: int
    next_timestamp_ns: int
    active_keys: Set[str] = field(default_factory=set)
    discrete_keys: Set[str] = field(default_factory=set)
    active_mouse: Set[str] = field(default_factory=set)
    mouse_downs: Set[str] = field(default_factory=set)
    mouse_ups: Set[str] = field(default_factory=set)
    wheel_events: int = 0
    mouse_motion: float = 0.0
    crouched: bool = False


@dataclass
class Component:
    label: str
    start_pos: int
    end_pos: int
    start_ts: int
    end_ts: int

    @property
    def duration_s(self) -> float:
        return max(0, self.end_ts - self.start_ts) / NS_PER_S


@dataclass
class Segment:
    label: str
    start_pos: int
    end_pos: int
    start_ts: int
    end_ts: int
    components: List[Component] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0, self.end_ts - self.start_ts) / NS_PER_S

    @property
    def num_frames(self) -> int:
        return self.end_pos - self.start_pos + 1


def _replay_state(frames: Sequence[dict], cfg: SegmenterConfig) -> List[FrameState]:
    held_since: Dict[str, int] = {}
    last_press: Dict[str, int] = {}
    held_mouse: Dict[str, int] = {}
    timeout_ns = int(cfg.held_without_release_timeout_s * NS_PER_S)
    crouched = False
    out: List[FrameState] = []

    for pos, frame in enumerate(frames):
        ts = int(frame["timestamp_ns"])
        nxt = frame.get("next_frame_timestamp_ns")
        if nxt is None:
            step = ts - int(frames[pos - 1]["timestamp_ns"]) if pos else 0
            nxt = ts + max(step, 1)
        state = FrameState(
            pos=pos,
            frame_index=int(frame.get("frame_index", pos)),
            timestamp_ns=ts,
            next_timestamp_ns=int(nxt),
        )

        for event in frame.get("keyboard_events", []) or []:
            key = event.get("key_name")
            if not key:
                continue
            kind = event.get("event_type")
            event_ts = int(event.get("timestamp_ns", ts))
            if kind == "press":
                if key not in held_since:
                    held_since[key] = event_ts
                    state.discrete_keys.add(key)
                last_press[key] = event_ts
            elif kind == "release":
                held_since.pop(key, None)
                last_press.pop(key, None)

        raw_mouse = frame.get("raw_mouse_events") or frame.get("mouse_events") or []
        for event in raw_mouse:
            state.mouse_motion += abs(float(event.get("dx", 0) or 0))
            state.mouse_motion += abs(float(event.get("dy", 0) or 0))
            for flag in event.get("button_flag_names", []) or []:
                if flag.endswith("_BUTTON_DOWN"):
                    base = flag[: -len("_DOWN")]
                    held_mouse[base] = int(event.get("timestamp_ns", ts))
                    state.mouse_downs.add(base)
                elif flag.endswith("_BUTTON_UP"):
                    base = flag[: -len("_UP")]
                    held_mouse.pop(base, None)
                    state.mouse_ups.add(base)
                elif flag == "RI_MOUSE_WHEEL":
                    state.wheel_events += 1

        active: Set[str] = set()
        for key in list(held_since):
            if ts - last_press.get(key, 0) <= timeout_ns:
                active.add(key)
            else:
                held_since.pop(key, None)
                last_press.pop(key, None)
        if "KEY_C" in state.discrete_keys:
            crouched = not crouched
        if (
            active & SPRINT_KEYS
            or {"LSHIFT", "RSHIFT", "SPACE"} & state.discrete_keys
        ):
            crouched = False
        state.active_keys = active
        state.active_mouse = set(held_mouse)
        state.crouched = crouched
        out.append(state)
    return out


_KEY_BURSTS: Dict[str, Tuple[str, str]] = {
    "LSHIFT": ("dodge_roll", "dodge_lock_s"),
    "RSHIFT": ("dodge_roll", "dodge_lock_s"),
    "SPACE": ("jumping", "jump_lock_s"),
    "KEY_Q": ("skill_ash_of_war", "skill_lock_s"),
    "KEY_E": ("interact", "interact_lock_s"),
    "KEY_R": ("use_item", "use_item_lock_s"),
    # KEY_F is only tentatively known to be two-hand; keep the wider parent.
    "KEY_F": ("switch_armament", "switch_lock_s"),
    "RIGHT": ("switch_right_armament", "switch_lock_s"),
    "LEFT": ("switch_left_armament", "switch_lock_s"),
    "UP": ("switch_sorcery", "switch_lock_s"),
    "DOWN": ("switch_quick_item", "switch_lock_s"),
}


def _priority(label: str) -> int:
    return {
        "use_item": 96,
        "skill_ash_of_war": 92,
        "light_attack": 88,
        "dodge_roll": 82,
        "jumping": 78,
        "interact": 70,
        "switch_armament": 64,
        "switch_right_armament": 64,
        "switch_left_armament": 64,
        "switch_sorcery": 64,
        "switch_quick_item": 64,
    }.get(label, 50)


def _build_bursts(
    states: Sequence[FrameState], cfg: SegmenterConfig
) -> List[List[Tuple[str, int]]]:
    active: List[List[Tuple[str, int]]] = [[] for _ in states]

    def add(start: int, label: str, duration_s: float) -> None:
        end_ts = states[start].timestamp_ns + int(duration_s * NS_PER_S)
        pos = start
        while pos < len(states) and states[pos].timestamp_ns <= end_ts:
            active[pos].append((label, _priority(label)))
            pos += 1

    for pos, state in enumerate(states):
        for key in state.discrete_keys:
            item = _KEY_BURSTS.get(key)
            if item:
                label, attr = item
                add(pos, label, float(getattr(cfg, attr)))
        if state.wheel_events:
            add(pos, "switch_armament", cfg.switch_lock_s)
        if LMB in state.mouse_downs:
            # LMB can be melee, heavy/charged, sorcery, incantation, bow, or
            # mounted combat.  Preserve it as the vision-refineable parent.
            add(pos, "light_attack", cfg.light_attack_lock_s)
        if LMB in state.mouse_ups:
            add(pos, "light_attack", cfg.attack_release_tail_s)
    return active


_GAMEPLAY_MENU_CLOSERS = WASD | SPRINT_KEYS | frozenset(
    {"SPACE", "LSHIFT", "RSHIFT", "KEY_Q", "KEY_E", "KEY_R", "KEY_C"}
)


def _menu_states(states: Sequence[FrameState]) -> List[Optional[str]]:
    current: Optional[str] = None
    out: List[Optional[str]] = []
    for state in states:
        if "KEY_M" in state.discrete_keys:
            current = None if current == "map" else "map"
        if {"ESCAPE", "ESC"} & state.discrete_keys:
            current = None if current is not None else "equipment"
        if current and (state.active_keys | state.discrete_keys) & _GAMEPLAY_MENU_CLOSERS:
            current = None
        out.append(current)
    return out


def _rolling_no_input(states: Sequence[FrameState], cfg: SegmenterConfig) -> List[float]:
    start: Optional[int] = None
    out: List[float] = []
    for state in states:
        has_input = bool(
            state.active_keys
            or state.active_mouse
            or state.discrete_keys
            or state.mouse_downs
            or state.mouse_ups
            or state.wheel_events
            or state.mouse_motion >= cfg.idle_motion_pixels
        )
        if has_input:
            start = None
            out.append(0.0)
        else:
            if start is None:
                start = state.timestamp_ns
            out.append((state.timestamp_ns - start) / NS_PER_S)
    return out


def _classify(
    state: FrameState,
    bursts: Sequence[Tuple[str, int]],
    menu: Optional[str],
    no_input_s: float,
    cfg: SegmenterConfig,
) -> str:
    if menu == "map":
        return "menu_map"
    if menu == "equipment":
        return "menu_equipment"
    if bursts:
        return max(bursts, key=lambda item: item[1])[0]
    if LMB in state.active_mouse:
        return "light_attack"
    if RMB in state.active_mouse:
        return "guarding"
    moving = bool(state.active_keys & WASD)
    if state.crouched and moving:
        return "crouch_sneaking"
    if moving and state.active_keys & SPRINT_KEYS:
        return "sprinting"
    if moving:
        return "walking"
    if no_input_s >= cfg.idle_min_s:
        return "idle_stationary"
    return "unknown"


def _frames_to_segments(states: Sequence[FrameState], labels: Sequence[str]) -> List[Segment]:
    if not states:
        return []
    out: List[Segment] = []
    start = 0
    for pos in range(1, len(states) + 1):
        if pos == len(states) or labels[pos] != labels[start]:
            component = Component(
                label=labels[start],
                start_pos=start,
                end_pos=pos - 1,
                start_ts=states[start].timestamp_ns,
                end_ts=states[pos - 1].next_timestamp_ns,
            )
            out.append(
                Segment(
                    label=labels[start],
                    start_pos=start,
                    end_pos=pos - 1,
                    start_ts=component.start_ts,
                    end_ts=component.end_ts,
                    components=[component],
                )
            )
            start = pos
    return out


def _merge_pair(left: Segment, right: Segment, label: str) -> Segment:
    return Segment(
        label=label,
        start_pos=left.start_pos,
        end_pos=right.end_pos,
        start_ts=left.start_ts,
        end_ts=right.end_ts,
        components=left.components + right.components,
    )


def _coalesce_same(segments: Sequence[Segment]) -> List[Segment]:
    out: List[Segment] = []
    for segment in segments:
        if out and out[-1].label == segment.label:
            out[-1] = _merge_pair(out[-1], segment, segment.label)
        else:
            out.append(segment)
    return out


def _absorb(segments: Sequence[Segment], labels: Iterable[str], fallback: str) -> List[Segment]:
    absorb = set(labels)
    if not absorb:
        return list(segments)
    current = list(segments)
    for index, segment in enumerate(current):
        if segment.label not in absorb:
            continue
        if index:
            segment.label = current[index - 1].label
        elif index + 1 < len(current):
            segment.label = current[index + 1].label
        else:
            segment.label = fallback
    return _coalesce_same(current)


def _smooth(segments: Sequence[Segment], cfg: SegmenterConfig) -> List[Segment]:
    current = list(segments)
    protected = set(cfg.unsmoothable_labels)
    while True:
        index = next(
            (
                i
                for i, segment in enumerate(current)
                if segment.duration_s < cfg.min_segment_s
                and segment.label not in protected
                and len(current) > 1
            ),
            None,
        )
        if index is None:
            break
        if 0 < index < len(current) - 1 and current[index - 1].label == current[index + 1].label:
            merged = _merge_pair(current[index - 1], current[index], current[index - 1].label)
            merged = _merge_pair(merged, current[index + 1], merged.label)
            current[index - 1 : index + 2] = [merged]
        elif index == 0:
            current[0:2] = [_merge_pair(current[0], current[1], current[1].label)]
        elif index == len(current) - 1:
            current[-2:] = [_merge_pair(current[-2], current[-1], current[-2].label)]
        elif current[index - 1].duration_s >= current[index + 1].duration_s:
            current[index - 1 : index + 1] = [
                _merge_pair(current[index - 1], current[index], current[index - 1].label)
            ]
        else:
            current[index : index + 2] = [
                _merge_pair(current[index], current[index + 1], current[index + 1].label)
            ]
        current = _coalesce_same(current)
    return current


def _merge_tier2(segments: Sequence[Segment], groups: Iterable[str], protected: Iterable[str]) -> List[Segment]:
    requested = set(groups)
    protected_set = set(protected)

    def group_for(label: str) -> Optional[str]:
        if label in protected_set:
            return None
        if "traversal" in requested and label in {"walking", "sprinting"}:
            return "traversal"
        leaf = LEAF_BY_LABEL.get(label)
        if leaf and leaf.tier2 in requested and leaf.tier2 != "traversal":
            return leaf.tier2
        return None

    out: List[Segment] = []
    for segment in segments:
        if out and group_for(out[-1].label) and group_for(out[-1].label) == group_for(segment.label):
            dominant = out[-1].label if out[-1].duration_s >= segment.duration_s else segment.label
            out[-1] = _merge_pair(out[-1], segment, dominant)
        else:
            out.append(segment)
    return _coalesce_same(out)


def _segment_dict(segment: Segment, states: Sequence[FrameState], frames: Sequence[dict]) -> Dict[str, object]:
    leaf: Optional[EventLeaf] = LEAF_BY_LABEL.get(segment.label)
    keys: Set[str] = set()
    buttons: Set[str] = set()
    motion = 0.0
    wheel = 0
    for state in states[segment.start_pos : segment.end_pos + 1]:
        keys.update(state.active_keys)
        keys.update(state.discrete_keys)
        buttons.update(state.active_mouse)
        buttons.update(state.mouse_downs)
        wheel += state.wheel_events
        motion += state.mouse_motion
    if wheel:
        buttons.add("RI_MOUSE_WHEEL")
    start_state = states[segment.start_pos]
    end_state = states[segment.end_pos]
    result: Dict[str, object] = {
        "start_frame": start_state.frame_index,
        "end_frame": end_state.frame_index,
        "num_frames": segment.num_frames,
        "start_timestamp_ns": segment.start_ts,
        "end_timestamp_ns": segment.end_ts,
        "duration_s": round(segment.duration_s, 3),
        "label": segment.label,
        "tier1": leaf.tier1 if leaf else None,
        "tier2": leaf.tier2 if leaf else None,
        "action_distinguishable": leaf.action_distinguishable if leaf else "n/a",
        "vision_needed": taxonomy_vision_needed(segment.label),
        "vision_subtypes": list(leaf.vision_subtypes) if leaf else [],
        "keys_pressed": sorted(keys),
        "mouse_buttons_pressed": sorted(buttons),
        "mouse_motion_total": round(motion, 1),
        "wheel_events": wheel,
        "frame_file_start": frames[segment.start_pos].get("frame_file"),
        "frame_file_end": frames[segment.end_pos].get("frame_file"),
    }
    distinct = {component.label for component in segment.components}
    if len(segment.components) > 1 and (len(distinct) > 1 or distinct != {segment.label}):
        result["sub_segments"] = [
            {
                "label": component.label,
                "start_frame": states[component.start_pos].frame_index,
                "end_frame": states[component.end_pos].frame_index,
                "start_timestamp_ns": component.start_ts,
                "end_timestamp_ns": component.end_ts,
                "duration_s": round(component.duration_s, 3),
            }
            for component in segment.components
        ]
    return result


def segment_video(
    annotations_path: Path,
    cfg: SegmenterConfig,
    *,
    video_name: Optional[str] = None,
    merge_tier2_groups: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    if not frames:
        raise ValueError(f"no frames in {annotations_path}")
    states = _replay_state(frames, cfg)
    bursts = _build_bursts(states, cfg)
    menus = _menu_states(states)
    idle = _rolling_no_input(states, cfg)
    labels = [
        _classify(state, bursts[pos], menus[pos], idle[pos], cfg)
        for pos, state in enumerate(states)
    ]
    raw = _frames_to_segments(states, labels)
    absorbed = _absorb(raw, cfg.absorb_labels, cfg.absorb_fallback_label)
    smoothed = _smooth(absorbed, cfg)
    groups = tuple(merge_tier2_groups) if merge_tier2_groups is not None else cfg.merge_tier2_groups
    final = _merge_tier2(smoothed, groups, cfg.unsmoothable_labels)
    segment_dicts = [_segment_dict(segment, states, frames) for segment in final]
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"segments": 0, "total_s": 0.0})
    for segment in segment_dicts:
        item = counts[str(segment["label"])]
        item["segments"] += 1
        item["total_s"] += float(segment["duration_s"])
    duration = (states[-1].timestamp_ns - states[0].timestamp_ns) / NS_PER_S
    transitions = [
        {
            "frame_index": segment_dicts[pos]["start_frame"],
            "timestamp_ns": segment_dicts[pos]["start_timestamp_ns"],
            "from": segment_dicts[pos - 1]["label"],
            "to": segment_dicts[pos]["label"],
        }
        for pos in range(1, len(segment_dicts))
    ]
    return {
        "game": "EldenRing",
        "video": video_name or data.get("source_mcap"),
        "scene_name": data.get("scene_name"),
        "source_mcap": data.get("source_mcap"),
        "num_frames": len(frames),
        "fps_estimate": round(len(frames) / duration, 3) if duration > 0 else 0.0,
        "interval_start_ns": data.get("interval_start_ns"),
        "interval_end_ns": data.get("interval_end_ns"),
        "config": {**asdict(cfg), "merge_tier2_groups": list(groups)},
        "label_counts": {
            label: {"segments": int(value["segments"]), "total_s": round(value["total_s"], 2)}
            for label, value in counts.items()
        },
        "segments": segment_dicts,
        "transitions": transitions,
    }


def _print_summary(payload: Dict[str, object], rows: int) -> None:
    segments = payload["segments"]
    print(f"video         : {payload['video']}")
    print(f"scene         : {payload['scene_name']}")
    print(f"frames        : {payload['num_frames']} @ ~{payload['fps_estimate']} fps")
    print(f"#segments     : {len(segments)}")
    vision_count = sum(bool(item.get("vision_needed")) for item in segments)
    print(f"#vision_needed: {vision_count}")
    print("label distribution (segments / total_s):")
    for label, value in sorted(payload["label_counts"].items(), key=lambda item: -item[1]["total_s"]):
        print(f"  {label:24s} {value['segments']:5d} {value['total_s']:9.2f}s")
    print(f"first {rows} segments:")
    for item in segments[:rows]:
        marker = "*" if item.get("vision_needed") else " "
        print(
            f" {marker}[{item['start_frame']:6d}->{item['end_frame']:6d}] "
            f"{item['duration_s']:7.2f}s {item['label']:24s} "
            f"keys={','.join(item.get('keys_pressed', [])[:5])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--video")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-duration-s", type=float)
    parser.add_argument("--no-default-tier2-merge", action="store_true")
    parser.add_argument("--no-merge-traversal", action="store_true")
    parser.add_argument("--merge-tier2", action="append", default=[])
    parser.add_argument("--summary-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegmenterConfig()
    if args.min_duration_s is not None:
        cfg.min_segment_s = args.min_duration_s
    groups: List[str] = [] if args.no_default_tier2_merge else list(cfg.merge_tier2_groups)
    if args.no_merge_traversal and "traversal" in groups:
        groups.remove("traversal")
    for group in args.merge_tier2:
        if group not in groups:
            groups.append(group)
    payload = segment_video(args.annotations, cfg, video_name=args.video, merge_tier2_groups=groups)
    out = args.out or args.annotations.with_name(args.annotations.stem + "_segments.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    _print_summary(payload, args.summary_rows)


if __name__ == "__main__":
    main()
