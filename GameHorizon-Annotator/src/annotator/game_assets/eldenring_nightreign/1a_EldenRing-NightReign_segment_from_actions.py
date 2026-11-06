"""Action-only event segmentation for an ELDEN RING NIGHTREIGN recording.

Nightreign twin of ``1a_Apex_Legends_segment_from_actions.py``. Reads a per-frame
annotation JSON (same format as the Apex pipeline) and emits a list of event
segments with the frame indices where the event label changes.

The classifier emits labels that are ``action_distinguishable == "high"`` in
:mod:`EldenRing_NightReign_taxonomy` plus a few ``medium`` parent buckets
(``interact``, ``use_item``, ``switch_armament``) that a later vision pass can
refine. Anything that cannot be decided from keyboard / mouse alone falls back
to ``unknown``.

Keybind grounding (verified against the recording, see the taxonomy module):
this capture uses a stock Nightreign PC layout with two twists --
**Surge Sprint is Left Alt (``LMENU``)** (``LSHIFT`` is Dodge Roll, a tap), and
**lock-on is rebound to mouse side-button 5 (``RI_MOUSE_BUTTON_5``)**.

Pipeline
--------
1. Load the annotation JSON and the taxonomy.
2. Replay all keyboard / mouse events in chronological order to derive per-frame
   "held" state for each key / mouse button. OS auto-repeat presses are handled,
   and unmatched releases are tolerated.
3. Detect "burst" events (dodge, skill / ash of war, character skill, ultimate,
   flask, use-item, switch-item, switch-armament, interact, lock-on) and lock
   their label for a small window of frames after the trigger so that one
   keypress yields a real segment instead of a single-frame blip.
4. Classify every frame with a priority cascade:

   menu screens > committed-animation bursts (flask / ultimate / character-skill
   / skill / use-item) > combat (LMB / RMB, incl. Ctrl+LMB heavy) >
   interruptible bursts (dodge / interact / switch-*) > traversal > lock-on >
   idle.

   Combat detection: ``LCONTROL + LMB`` = ``heavy_attack``; ``LMB`` alone =
   ``light_attack``; ``RMB`` held = ``guarding``. The "committed" bursts win over
   combat because their animations lock out attacks in Nightreign; the
   interruptible bursts lose to combat (e.g. roll-cancel into an attack shows the
   attack). ``lock_on_toggle`` is the lowest-priority discrete action -- it only
   surfaces when nothing else (combat / burst / movement) is happening, so it
   never steals a combat or traversal segment.
5. Collapse consecutive identical labels into segments, then apply three
   smoothing passes:

   a. **Absorb** segments whose label is in ``cfg.absorb_labels`` (empty by
      default).
   b. **Duration smoothing**: merge any segment shorter than
      ``--min-duration-s`` into its neighbour, except segments whose label is in
      ``cfg.unsmoothable_labels`` (all discrete bursts by default -- dodge,
      skill, flask, etc.). This protects short atomic actions from being
      absorbed into surrounding traversal.
   c. **Tier2 grouping**: collapse consecutive segments that share the same
      taxonomy tier2 (default: ``traversal`` -- walking / surge_sprinting /
      jumping -- and ``combat`` -- light_attack / heavy_attack / guarding) and
      label the merged segment with the dominant sub-label by total duration.
      Burst labels are excluded from this merge so a dodge between two attacks,
      or a skill inside a combo, survives as its own atomic segment.

   Sub-segment details survive in each segment's ``sub_segments`` field so the
   downstream vision pass can still see the original breakdown.

Idle detection
--------------
"Idle" is only emitted when the player is **completely still** for at least
``cfg.idle_min_s``: no keys held, no mouse buttons held, no wheel ticks, and
per-frame mouse motion below ``cfg.idle_motion_pixels``. Any of those resets the
no-input counter.

Run::

    python 1a_EldenRing-NightReign_segment_from_actions.py annotations_per_frame.json \\
        --video recording.mkv --out recording_segments.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from EldenRing_NightReign_taxonomy import (
    LEAF_BY_LABEL,
    EventLeaf,
    vision_needed as taxonomy_vision_needed,
)


# ---------------------------------------------------------------------------
# Tunables (all times in seconds; converted to ns where needed).
# ---------------------------------------------------------------------------

NS_PER_S = 1_000_000_000

# All labels produced by the discrete-keypress / discrete-click burst pipeline.
# These represent intentional, atomic player actions and must NOT be merged away
# by the short-segment smoothing pass, nor collapsed by the tier2 grouping pass,
# even when their lock-window puts them below ``min_segment_s``.
#
# ``lock_on_toggle`` is intentionally NOT in this tuple: it is a low-priority
# camera modifier that should quietly merge into whatever it overlaps rather than
# owning a standalone segment. ``guarding`` is also excluded (it is a held state,
# not a discrete trigger, and is allowed to merge into the combat tier2 group).
BURST_LABELS: Tuple[str, ...] = (
    "light_attack",
    "heavy_attack",
    "dodge_roll",
    "skill_ash_of_war",
    "character_skill",
    "ultimate_art",
    "healing_flask",
    "use_item",
    "switch_item",
    "switch_armament",
    "interact",
)


@dataclass
class SegmenterConfig:
    # Idle thresholds
    idle_motion_pixels: float = 8.0          # |dx|+|dy| per frame at/above this = "moving"
    idle_min_s: float = 1.0                  # need this much continuous no-input to call it idle
    # Burst lock windows (how long a single trigger "owns" the label). Each
    # discrete action holds its label for this long so one click / tap yields a
    # real segment and rapid repeats cluster into a continuous run.
    light_attack_lock_s: float = 0.70        # minimum visible light-attack animation
    heavy_attack_lock_s: float = 1.00        # minimum visible heavy-attack animation
    attack_release_tail_s: float = 0.35      # animation tail after LMB release
    dodge_lock_s: float = 0.45               # LSHIFT (roll / backstep / dash)
    skill_lock_s: float = 0.6                # KEY_Q (ash of war / weapon skill)
    char_skill_lock_s: float = 0.8           # KEY_E (character skill)
    ultimate_lock_s: float = 1.4             # KEY_R (ultimate art)
    flask_delay_s: float = 0.35              # KEY_1 press -> visible drink animation
    flask_duration_s: float = 1.70           # visible drink / recovery animation
    use_item_lock_s: float = 0.9             # KEY_G (use selected item)
    switch_item_lock_s: float = 0.35         # KEY_2
    switch_armament_lock_s: float = 0.35     # mouse wheel
    interact_lock_s: float = 0.6             # KEY_F (event action)
    lock_on_lock_s: float = 0.25             # RI_MOUSE_BUTTON_5 (rebound lock-on)
    # Short post-burst carries for committed combat / ability animations. These
    # tails prevent held movement from immediately relabelling visible recovery
    # frames as walking after the input-owned burst window expires.
    combat_animation_tail_s: Dict[str, float] = field(default_factory=lambda: {
        "light_attack": 0.35,
        "heavy_attack": 0.50,
        "skill_ash_of_war": 0.60,
        "character_skill": 0.80,
        "ultimate_art": 1.00,
    })
    # Smoothing
    min_segment_s: float = 0.8               # segments shorter than this are merged into neighbours
    # Labels immune to the short-segment smoothing pass (all discrete bursts).
    unsmoothable_labels: Tuple[str, ...] = BURST_LABELS
    # If both press and release counts disagree (auto-repeat), how long after the
    # last press without seeing a release do we still consider the key held?
    held_without_release_timeout_s: float = 0.8
    # Labels that should never own a standalone segment (force-merged). Empty by
    # default; kept available for callers.
    absorb_labels: Tuple[str, ...] = ()
    absorb_fallback_label: str = "idle_stationary"
    # Tier2 groups whose consecutive sub-labels are collapsed into a single
    # segment (dominant sub-label by cumulative duration):
    #   - ``traversal`` : walking / surge_sprinting / jumping
    #   - ``combat``    : light_attack / heavy_attack / guarding
    # Burst labels (dodge_roll, skill_ash_of_war, ...) are excluded from the
    # merge so they survive as their own atomic segments.
    merge_tier2_groups: Tuple[str, ...] = ("traversal", "combat")


# ---------------------------------------------------------------------------
# Per-frame state replay
# ---------------------------------------------------------------------------

WASD = {"KEY_W", "KEY_A", "KEY_S", "KEY_D"}
SPRINT_KEYS = {"LMENU", "RMENU"}   # Surge Sprint = Left Alt (LMENU is its VK name)
JUMP_KEYS = {"SPACE"}
HEAVY_MOD = "LCONTROL"             # Ctrl+LMB = heavy/strong attack; Ctrl+Wheel = two-hand

LMB = "RI_MOUSE_LEFT_BUTTON"
RMB = "RI_MOUSE_RIGHT_BUTTON"
BUTTON5 = "RI_MOUSE_BUTTON_5"      # rebound lock-on / reset camera


@dataclass
class FrameState:
    frame_index: int
    timestamp_ns: int
    next_timestamp_ns: int
    active_keys: Set[str] = field(default_factory=set)
    active_mouse: Set[str] = field(default_factory=set)
    mouse_motion: float = 0.0                # sum |dx|+|dy| over events in this frame
    discrete_keys: Set[str] = field(default_factory=set)   # discrete press events seen this frame
    mouse_downs: Set[str] = field(default_factory=set)
    mouse_ups: Set[str] = field(default_factory=set)
    wheel_events: int = 0


def _replay_state(frames: List[dict], cfg: SegmenterConfig) -> List[FrameState]:
    """Walk frames in order and produce a per-frame snapshot of held inputs."""
    held_since: Dict[str, int] = {}             # key_name -> last press ts (cleared on release)
    last_press_ts: Dict[str, int] = {}
    held_mouse: Dict[str, int] = {}             # button base name -> last DOWN ts
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
        st = FrameState(frame_index=f["frame_index"], timestamp_ns=ts, next_timestamp_ns=nxt)

        # 1) keyboard
        for kb in f.get("keyboard_events", []):
            kn = kb.get("key_name")
            if not kn:
                continue
            ev_type = kb.get("event_type")
            ets = kb.get("timestamp_ns", ts)
            if ev_type == "press":
                if kn not in held_since:
                    held_since[kn] = ets
                    st.discrete_keys.add(kn)   # first press of a hold = discrete trigger
                last_press_ts[kn] = ets
            elif ev_type == "release":
                held_since.pop(kn, None)
                last_press_ts.pop(kn, None)

        # 2) mouse — button state + motion deltas come from the RAW input stream
        # (``raw_mouse_events``). That array carries per-event dx/dy deltas and
        # the ``RI_*`` ``button_flag_names`` we rely on for attack / guard / wheel
        # / lock-on detection. The processed ``mouse_events`` stream only has
        # absolute cursor x/y (no deltas / button flags), so we fall back to it
        # only if a frame has no raw stream.
        raw_mouse = f.get("raw_mouse_events")
        if not raw_mouse:
            raw_mouse = f.get("mouse_events", [])
        for ms in raw_mouse:
            st.mouse_motion += abs(ms.get("dx", 0) or 0) + abs(ms.get("dy", 0) or 0)
            for fl in ms.get("button_flag_names", []) or []:
                if fl.endswith("_BUTTON_DOWN"):
                    base = fl[: -len("_DOWN")]
                    held_mouse[base] = ms.get("timestamp_ns", ts)
                    st.mouse_downs.add(base)
                elif fl.endswith("_BUTTON_UP"):
                    base = fl[: -len("_UP")]
                    held_mouse.pop(base, None)
                    st.mouse_ups.add(base)
                elif fl == "RI_MOUSE_WHEEL":
                    st.wheel_events += 1

        # 3) snapshot keys still held; drop ones whose last press was too long ago
        active_keys: Set[str] = set()
        for kn, _start_ts in list(held_since.items()):
            lp = last_press_ts.get(kn, 0)
            if ts - lp <= timeout_ns:
                active_keys.add(kn)
            else:
                held_since.pop(kn, None)
                last_press_ts.pop(kn, None)
        st.active_keys = active_keys
        st.active_mouse = set(held_mouse.keys())
        out.append(st)
    return out


# ---------------------------------------------------------------------------
# Burst / lock-window detection
# ---------------------------------------------------------------------------

# Map (key) -> (label, lock_seconds_attr) for keys that fire a discrete event.
KEY_TO_BURST: Dict[str, Tuple[str, str]] = {
    "LSHIFT": ("dodge_roll",        "dodge_lock_s"),
    "RSHIFT": ("dodge_roll",        "dodge_lock_s"),
    "KEY_Q":  ("skill_ash_of_war",  "skill_lock_s"),
    "KEY_E":  ("character_skill",   "char_skill_lock_s"),
    "KEY_R":  ("ultimate_art",      "ultimate_lock_s"),
    "KEY_1":  ("healing_flask",     "flask_duration_s"),
    "KEY_G":  ("use_item",          "use_item_lock_s"),
    "KEY_2":  ("switch_item",       "switch_item_lock_s"),
    "KEY_F":  ("interact",          "interact_lock_s"),
}


def _burst_priority(label: str) -> int:
    """Higher = wins when multiple bursts overlap on the same frame."""
    return {
        "healing_flask":     95,
        "ultimate_art":      92,
        "character_skill":   90,
        "use_item":          88,
        "skill_ash_of_war":  85,
        "heavy_attack":      82,
        "light_attack":      80,
        "dodge_roll":        72,
        "interact":          60,
        "switch_item":       55,
        "switch_armament":   52,
        "lock_on_toggle":    10,
    }.get(label, 50)


def _build_burst_intervals(
    frame_states: List[FrameState],
    cfg: SegmenterConfig,
) -> List[List[Tuple[str, int]]]:
    """For each frame return a list of ``(label, priority_rank)`` active bursts.

    A burst is created when a key transitions from "not held" to "held" (i.e.
    appears in ``discrete_keys``), on every wheel tick (switch_armament), and on
    each mouse button-5 DOWN transition (lock_on_toggle). The lock window extends
    ``cfg.<lock_attr>`` seconds forward in wall time.
    """
    n = len(frame_states)
    active: List[List[Tuple[str, int]]] = [[] for _ in range(n)]

    def add_interval(
        start_idx: int,
        label: str,
        duration_s: float,
        priority: int,
        delay_s: float = 0.0,
    ) -> None:
        trigger_ts = frame_states[start_idx].timestamp_ns
        start_ts = trigger_ts + int(delay_s * NS_PER_S)
        end_ts = start_ts + int(duration_s * NS_PER_S)
        i = start_idx
        while i < n and frame_states[i].timestamp_ns < start_ts:
            i += 1
        while i < n and frame_states[i].timestamp_ns <= end_ts:
            active[i].append((label, priority))
            i += 1

    for idx, st in enumerate(frame_states):
        # Discrete key triggers
        for kn in st.discrete_keys:
            if kn in KEY_TO_BURST:
                label, attr = KEY_TO_BURST[kn]
                delay_s = cfg.flask_delay_s if label == "healing_flask" else 0.0
                add_interval(
                    idx,
                    label,
                    getattr(cfg, attr),
                    priority=_burst_priority(label),
                    delay_s=delay_s,
                )
        # Mouse wheel = switch armament (up = right hand, down = left hand; the
        # vision pass refines which, and two-hand toggles). We only know a tick
        # happened, so emit the medium ``switch_armament`` parent bucket.
        if st.wheel_events > 0:
            add_interval(idx, "switch_armament", cfg.switch_armament_lock_s,
                         priority=_burst_priority("switch_armament"))

    # Attack detection uses raw DOWN / UP edges so a click is not lost when both
    # edges land in one annotated frame. Each press owns a minimum animation
    # window, and release adds a short tail for visible recovery frames.
    last_attack_label: Optional[str] = None
    prev_b5 = False
    for idx, st in enumerate(frame_states):
        if LMB in st.mouse_downs:
            label = "heavy_attack" if HEAVY_MOD in st.active_keys else "light_attack"
            duration_s = (
                cfg.heavy_attack_lock_s
                if label == "heavy_attack"
                else cfg.light_attack_lock_s
            )
            add_interval(idx, label, duration_s, priority=_burst_priority(label))
            last_attack_label = label
        if LMB in st.mouse_ups and last_attack_label is not None:
            add_interval(
                idx,
                last_attack_label,
                cfg.attack_release_tail_s,
                priority=_burst_priority(last_attack_label),
            )
            last_attack_label = None
        # Lock-on detection (fires on mouse button-5 DOWN transition).
        cur_b5 = BUTTON5 in st.active_mouse
        if cur_b5 and not prev_b5:
            add_interval(idx, "lock_on_toggle", cfg.lock_on_lock_s,
                         priority=_burst_priority("lock_on_toggle"))
        prev_b5 = cur_b5

    return active


# ---------------------------------------------------------------------------
# Committed combat / ability animation tails
# ---------------------------------------------------------------------------

def _recent_animation_labels(
    frame_states: List[FrameState],
    bursts_per_frame: List[List[Tuple[str, int]]],
    cfg: SegmenterConfig,
) -> List[Optional[str]]:
    """Carry the latest committed action briefly beyond its burst interval.

    The action-only pipeline cannot observe the exact visual recovery frame, so
    each supported combat / ability label gets a conservative, label-specific
    tail. Active bursts still win normally; this array is only consulted after
    active / held combat and before traversal classification.
    """
    tails = cfg.combat_animation_tail_s
    out: List[Optional[str]] = []
    last_label: Optional[str] = None
    carry_until_ts = 0

    for st, bursts in zip(frame_states, bursts_per_frame):
        committed = [item for item in bursts if item[0] in tails]
        if committed:
            label, _priority = max(committed, key=lambda item: item[1])
            last_label = label
            carry_until_ts = (
                st.next_timestamp_ns + int(tails[label] * NS_PER_S)
            )
            out.append(None)
            continue

        # A different explicit burst supersedes any older combat carry so it
        # cannot reappear after an interaction / item action finishes.
        if bursts:
            last_label = None
            carry_until_ts = 0
            out.append(None)
            continue

        if last_label is not None and st.timestamp_ns <= carry_until_ts:
            out.append(last_label)
        else:
            last_label = None
            out.append(None)

    return out


# ---------------------------------------------------------------------------
# Menu / overlay toggle tracking
# ---------------------------------------------------------------------------

# Nightreign's Map (``KEY_M``) and Menu (``ESCAPE``) are *toggles*: a single
# press opens the overlay and it stays open (no key held) until it is toggled
# closed. So we cannot detect them from held state -- we track a toggle machine.
# Keys / buttons that mean the player is clearly controlling the character and
# therefore is NOT sitting in a menu; seeing any of them force-closes the overlay
# so a missed close event cannot leave us "stuck" in a menu label. Mouse motion
# and the wheel are excluded (they navigate / zoom menus), as is ``KEY_Q`` (menu
# Back / Cancel).
MENU_CLOSING_KEYS = (
    WASD | JUMP_KEYS | SPRINT_KEYS | {"KEY_E", "KEY_R", "KEY_1", "KEY_G", "LSHIFT", "RSHIFT"}
)


def _compute_menu_states(frame_states: List[FrameState]) -> List[Optional[str]]:
    """Per-frame overlay state: ``None`` / ``"map"`` / ``"other"``.

    ``ESCAPE`` alone cannot identify the visible menu type. It therefore opens
    the vision-only ``menu_other`` candidate; a later visual pass can refine it
    to equipment, level-up, merchant, results, or another concrete GUI label.
    """
    states: List[Optional[str]] = []
    cur: Optional[str] = None
    for st in frame_states:
        if "KEY_M" in st.discrete_keys:
            cur = None if cur == "map" else "map"
        if "ESCAPE" in st.discrete_keys or "ESC" in st.discrete_keys:
            cur = None if cur is not None else "other"
        if cur is not None:
            # Mouse clicks are normal menu navigation and must not force-close
            # the overlay state. Only unmistakable gameplay controls retain the
            # fallback that recovers from a missed explicit close event.
            if st.active_keys & MENU_CLOSING_KEYS:
                cur = None
        states.append(cur)
    return states


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

UNKNOWN_LABEL = "unknown"


def _is_held(state: FrameState, keys: Iterable[str]) -> bool:
    return any(k in state.active_keys for k in keys)


def _classify_frame(
    st: FrameState,
    bursts: List[Tuple[str, int]],
    menu_state: Optional[str],
    rolling_no_input_s: float,
    recent_animation_label: Optional[str],
    cfg: SegmenterConfig,
) -> str:
    # 1) Menu / overlay screens (highest priority - they suppress everything).
    # Detected from the toggle machine, since Map / Menu are not held keys.
    if menu_state == "map":
        return "menu_map"
    if menu_state == "other":
        return "menu_other"

    top_burst: Optional[str] = None
    if bursts:
        top_burst = sorted(bursts, key=lambda lp: -lp[1])[0][0]

    # 2) Discrete action bursts win (attacks, skills, abilities, flask, dodge,
    # interact, switch-*). Lock-on is deferred to step 5 so it never overrides
    # movement. Committed animations (flask / ultimate / skill / ...) naturally
    # outrank attacks via _burst_priority, so a flask during a combo still reads
    # as the flask.
    if top_burst is not None and top_burst != "lock_on_toggle":
        return top_burst

    # 3) Held-combat fallback: sustained LMB (e.g. a charged heavy) or a held
    # guard whose down-transition burst has already expired.
    lmb_held = LMB in st.active_mouse
    rmb_held = RMB in st.active_mouse
    if lmb_held:
        return "heavy_attack" if HEAVY_MOD in st.active_keys else "light_attack"
    if rmb_held:
        return "guarding"

    # 4) Recently committed animation recovery. Movement keys may already be
    # held for the next action, but walking must not steal visible combat /
    # ability tail frames.
    if recent_animation_label is not None:
        return recent_animation_label

    # 5) Movement
    moving = _is_held(st, WASD)
    sprinting = _is_held(st, SPRINT_KEYS)
    jumping = _is_held(st, JUMP_KEYS)

    if jumping and moving:
        return "jumping"
    if sprinting and moving:
        return "surge_sprinting"
    if moving:
        return "walking"
    if jumping:
        return "jumping"

    # 6) Lock-on: only when there is no combat, no other burst, and no movement.
    if top_burst == "lock_on_toggle":
        return "lock_on_toggle"

    # 7) Idle
    if rolling_no_input_s >= cfg.idle_min_s:
        return "idle_stationary"

    # 8) Truly nothing actionable (short gap between actions, camera-only pan).
    return UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Segment building & smoothing
# ---------------------------------------------------------------------------


@dataclass
class SubSegment:
    """One contiguous run of frames with a single rule-classifier label."""
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
    end_frame: int                      # inclusive
    start_ts: int
    end_ts: int                         # uses next_frame_timestamp of end_frame
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
    """Merge segments shorter than ``min_duration_s`` into a neighbour.

    Segments whose label is in ``unsmoothable_labels`` are exempt (they are
    preserved regardless of duration), protecting short discrete bursts (dodge /
    skill / flask / ...) from being swallowed by an adjacent traversal segment.
    """
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
    """Force-merge any segment whose label is in ``absorb_labels`` into a neighbour."""
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
    protected_labels: Iterable[str] = (),
) -> List[Segment]:
    """Collapse contiguous segments sharing a tier2 group into one segment.

    For each run of consecutive *mergeable* segments (taxonomy ``tier2`` in
    ``merge_tier2_groups`` and label not in ``protected_labels``), the run is
    merged into a single segment whose label is the sub-label with the longest
    cumulative duration (ties broken by first occurrence). Protected labels
    (discrete bursts) break runs and are preserved, so a dodge between attacks or
    a skill inside a combo keeps its own atomic segment.
    """
    merge_set = set(merge_tier2_groups)
    protected = set(protected_labels)
    if not segments or not merge_set:
        return list(segments)

    def tier2_of(label: str) -> Optional[str]:
        leaf = LEAF_BY_LABEL.get(label)
        return leaf.tier2 if leaf else None

    def mergeable(seg: Segment) -> bool:
        return tier2_of(seg.label) in merge_set and seg.label not in protected

    out: List[Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        if not mergeable(seg):
            out.append(seg)
            i += 1
            continue
        t2 = tier2_of(seg.label)
        j = i + 1
        while j < n and mergeable(segments[j]) and tier2_of(segments[j].label) == t2:
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
# Idle helpers (rolling window over short look-back)
# ---------------------------------------------------------------------------


def _rolling_no_input(
    frame_states: List[FrameState], cfg: SegmenterConfig
) -> List[float]:
    """For each frame, return seconds of continuous no-input ending at that frame."""
    out = [0.0] * len(frame_states)
    no_input_start_ts: Optional[int] = None
    motion_threshold = cfg.idle_motion_pixels
    for i, st in enumerate(frame_states):
        has_input = (
            bool(st.active_keys)
            or bool(st.active_mouse)
            or st.wheel_events > 0
            or st.mouse_motion >= motion_threshold
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
    frames = data["frames"]
    frame_states = _replay_state(frames, cfg)

    bursts_per_frame = _build_burst_intervals(frame_states, cfg)
    recent_animation_labels = _recent_animation_labels(
        frame_states, bursts_per_frame, cfg
    )
    menu_states = _compute_menu_states(frame_states)
    no_input_s = _rolling_no_input(frame_states, cfg)

    raw_labels = [
        _classify_frame(
            st,
            bursts_per_frame[i],
            menu_state=menu_states[i],
            rolling_no_input_s=no_input_s[i],
            recent_animation_label=recent_animation_labels[i],
            cfg=cfg,
        )
        for i, st in enumerate(frame_states)
    ]

    raw_segments = _frames_to_segments(frame_states, raw_labels)
    absorbed = _absorb_segments(
        raw_segments, cfg.absorb_labels, cfg.absorb_fallback_label
    )
    smoothed = _smooth_segments(
        absorbed, cfg.min_segment_s, cfg.unsmoothable_labels
    )
    final_segments = _merge_segments_by_tier2(
        smoothed, cfg.merge_tier2_groups, protected_labels=cfg.unsmoothable_labels
    )

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

    payload: Dict[str, object] = {
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
    return payload


def _label_counts(segments: List[Segment]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"segments": 0, "total_s": 0.0})
    for seg in segments:
        c = counts[seg.label]
        c["segments"] += 1
        c["total_s"] += seg.duration_s
    return {k: {"segments": int(v["segments"]), "total_s": round(v["total_s"], 2)} for k, v in counts.items()}


def _segment_to_dict(
    seg: Segment,
    frame_states: List[FrameState],
    frames: List[dict],
) -> Dict[str, object]:
    leaf: Optional[EventLeaf] = LEAF_BY_LABEL.get(seg.label)

    # Aggregate evidence over all frames in this segment.
    keys_pressed: Set[str] = set()
    mouse_buttons: Set[str] = set()
    wheel_events_total = 0
    motion_total = 0.0
    for st in frame_states[seg.start_frame : seg.end_frame + 1]:
        keys_pressed.update(st.active_keys)
        keys_pressed.update(st.discrete_keys)
        mouse_buttons.update(st.active_mouse)
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
        print(f"  {label:22s}  {c['segments']:5d}    {c['total_s']:8.2f}s")
    print()
    print(f"first {top_n_segments} segments:")
    for seg in payload["segments"][:top_n_segments]:
        flag = "*" if seg.get("vision_needed") else " "
        print(
            f" {flag}[{seg['start_frame']:6d}->{seg['end_frame']:6d}] "
            f"{seg['duration_s']:6.2f}s  {seg['label']:22s} "
            f"keys={','.join(seg.get('keys_pressed', [])[:4]):<24s} "
            f"mouse={','.join(seg.get('mouse_buttons_pressed', []))}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("annotations", type=Path, help="Path to annotations_per_frame.json")
    p.add_argument("--video", type=str, default=None, help="Original video filename (for metadata).")
    p.add_argument("--out", type=Path, default=None, help="Where to write segments JSON.")
    p.add_argument("--min-duration-s", type=float, default=None, help="Override smoothing threshold (seconds).")
    p.add_argument(
        "--no-default-tier2-merge",
        action="store_true",
        help="Disable every built-in Tier2 merge group; --merge-tier2 may add groups back.",
    )
    p.add_argument(
        "--no-merge-traversal",
        action="store_true",
        help="Do not collapse consecutive walking / surge_sprinting / jumping "
        "segments into one traversal segment. The combat merge group is "
        "unaffected.",
    )
    p.add_argument(
        "--merge-tier2",
        action="append",
        default=None,
        metavar="TIER2",
        help="Additional taxonomy tier2 groups to collapse (repeatable, e.g. "
        "--merge-tier2 abilities). Defaults from SegmenterConfig.merge_tier2_groups "
        "(currently traversal, combat) are always included unless "
        "--no-merge-traversal is set.",
    )
    p.add_argument("--summary-rows", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegmenterConfig()
    if args.min_duration_s is not None:
        cfg.min_segment_s = args.min_duration_s

    merge_groups: List[str] = [] if args.no_default_tier2_merge else list(cfg.merge_tier2_groups)
    if args.no_merge_traversal and "traversal" in merge_groups:
        merge_groups.remove("traversal")
    if args.merge_tier2:
        for g in args.merge_tier2:
            if g not in merge_groups:
                merge_groups.append(g)
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
