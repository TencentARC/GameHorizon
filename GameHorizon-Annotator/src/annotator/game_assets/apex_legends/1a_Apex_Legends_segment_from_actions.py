"""Action-only event segmentation for an Apex Legends recording.

Reads a per-frame annotation JSON (the format produced upstream and exemplified
by ``annotations_per_frame.json``) and emits a list of event segments together
with the frame indices where the event label changes.

The classifier emits labels that are ``action_distinguishable == "high"`` in
:mod:`Apex_Legends_taxonomy` plus a small number of ``medium`` parent buckets
(``interact``, ``throwing_grenade``, ``sliding``, ``menu_loadout``) that a
later vision pass can refine. Anything that cannot be decided from keyboard /
mouse alone falls back to ``unknown``.

Pipeline
--------
1. Load the annotation JSON and the taxonomy.
2. Replay all keyboard / mouse events in chronological order to derive per-frame
   "held" state for each key / mouse button.  OS auto-repeat presses are
   handled, and unmatched releases are tolerated.
3. Detect "burst" events (reload, weapon-swap, tactical, ultimate, melee,
   grenade, heal/shield, ping, interact) and lock their label for a small
   window of frames after the trigger so that one keypress yields a real
   segment instead of a single-frame blip.
4. Classify every frame with a priority cascade:
   menu screens > combat (LMB/RMB) > "fire-locking" bursts > other bursts >
   traversal > idle.

   Note: combat (firing / ADS) is now checked **before** bursts, with one
   exception -- bursts whose Apex animation truly locks out firing
   (``throwing_grenade``, ``melee``, ``using_health_item``,
   ``tactical_ability``, ``ultimate_ability``) still win over combat. This
   means cases like "reload-cancel-fire" or "equip-then-shoot" now correctly
   show up as ``firing_hipfire`` instead of a stale reload / equip label.
   See ``COMBAT_OVERRIDES_BURST`` in the source for the partition.
5. Collapse consecutive identical labels into segments, then apply three
   smoothing passes:

   a. **Absorb** segments whose label is in ``cfg.absorb_labels`` (empty by
      default; available for callers that want to treat custom labels as
      no-op transitions).
   b. **Duration smoothing**: merge any segment shorter than
      ``--min-duration-s`` (default 0.5s) into its neighbour, except segments
      whose label is in ``cfg.unsmoothable_labels`` (all burst events by
      default -- reload, melee, ping, equip_*, etc.). This protects short
      discrete actions from being absorbed into surrounding traversal.
   c. **Tier2 grouping**: collapse consecutive segments that share the same
      taxonomy tier2 (default: ``traversal`` -- walking / sprinting /
      crouch_walking / sliding / jumping_mantling all become one segment) and
      label the merged segment with the dominant sub-label by total duration.

   Sub-segment details survive in each segment's ``sub_segments`` field so the
   downstream vision pass can still see the original walking / sprinting
   breakdown.

Idle detection
--------------
"Idle" is only emitted when the player is **completely still** for at least
``cfg.idle_min_s`` (default 1.0s): no keys held, no mouse buttons held, no
wheel ticks, and per-frame mouse motion below ``cfg.idle_motion_pixels``.
Any of those resets the no-input counter. Because mouse motion now resets
the counter, the previous ``idle_camera_only`` label is no longer emitted --
"looking around" is treated as ongoing input.

Run::

    python 1a_Apex_Legends_segment_from_actions.py annotations_per_frame.json \\
        --video 0805_01.mkv --out 0805_01_segments.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from Apex_Legends_taxonomy import LEAF_BY_LABEL, EventLeaf, vision_needed as taxonomy_vision_needed


# ---------------------------------------------------------------------------
# Tunables (all times in seconds; converted to ns where needed).
# ---------------------------------------------------------------------------

NS_PER_S = 1_000_000_000

# All labels produced by the discrete-keypress burst pipeline
# (KEY_TO_BURST values + middle-mouse "pinging"). These represent intentional,
# atomic player actions and must NOT be merged away by the short-segment
# smoothing pass even when their lock-window puts them below ``min_segment_s``.
#
# Note: the player has rebound the mouse wheel away from weapon cycling, so
# wheel ticks no longer produce a ``cycle_weapon`` burst (the leaf has been
# removed from the taxonomy too).
BURST_LABELS: Tuple[str, ...] = (
    "reloading",
    "equip_primary",
    "equip_secondary",
    "holster_weapon",
    "weapon_inspect",
    "tactical_ability",
    "ultimate_ability",
    "melee",
    "throwing_grenade",
    "using_health_item",
    "interact",
    "pinging",
)


@dataclass
class SegmenterConfig:
    # Idle thresholds
    idle_motion_pixels: float = 8.0          # |dx|+|dy| per frame at/above this = "moving"
    idle_min_s: float = 1.0                  # need this much continuous no-input (no keys, no mouse buttons, no wheel, mouse motion below idle_motion_pixels) to call it idle
    # Combat thresholds
    lmb_min_s: float = 0.05                  # treat any LMB hold >= this as engagement
    ads_min_s: float = 0.10                  # RMB hold >= this counts as ADS
    # Burst lock windows (how long a single keypress "owns" the label)
    reload_lock_s: float = 1.8
    weapon_swap_lock_s: float = 0.4          # KEY_1/2/3 and mouse wheel
    weapon_inspect_lock_s: float = 1.0       # KEY_N
    tactical_lock_s: float = 0.8
    ultimate_lock_s: float = 1.0
    melee_lock_s: float = 0.5
    grenade_lock_s: float = 0.8
    heal_lock_s: float = 4.0                 # KEY_4 -> use selected health item (heal or shield)
    ping_lock_s: float = 0.4
    interact_lock_s: float = 0.7             # F / E
    # Smoothing
    min_segment_s: float = 1.0               # segments shorter than this are merged into neighbours
    # Labels that are immune to the short-segment smoothing pass: they always
    # survive even when their duration is below ``min_segment_s`` (e.g. a 0.5s
    # melee should never get absorbed into an adjacent walking segment).
    unsmoothable_labels: Tuple[str, ...] = BURST_LABELS
    # If both press and release counts disagree (auto-repeat), how long after the last
    # press without seeing a release do we still consider the key held?
    held_without_release_timeout_s: float = 0.8
    # Labels that should never own a standalone segment; they are always
    # merged into a neighbour. With ``idle_camera_only`` no longer emitted by
    # the classifier (mouse motion now resets the idle counter), this is empty
    # by default, but kept available so callers can absorb other transient
    # labels if they want.
    absorb_labels: Tuple[str, ...] = ()
    # If an absorb-label segment has no neighbour to merge into, fall back
    # to this label rather than emitting the absorb-label itself.
    absorb_fallback_label: str = "idle_stationary"
    # Tier2 groups whose consecutive sub-labels are collapsed into a single
    # segment. Each group below is collapsed so the listed sub-labels flow
    # into one continuous segment carrying the dominant sub-label
    # (longest cumulative duration; ties broken by first occurrence):
    #   - ``traversal``    : walking / sprinting / crouch_walking /
    #                        sliding / jumping_mantling
    #   - ``combat``       : aiming_down_sights / firing_hipfire / firing_ads
    #                        (i.e. one continuous combat-engagement block)
    #   - ``dropping_in``  : dropship_transition / skydiving / gliding
    #                        (the full dive-in sequence as one segment)
    merge_tier2_groups: Tuple[str, ...] = ("traversal", "combat", "dropping_in")


# ---------------------------------------------------------------------------
# Per-frame state replay
# ---------------------------------------------------------------------------

WASD = {"KEY_W", "KEY_A", "KEY_S", "KEY_D"}
SPRINT_KEYS = {"LSHIFT", "RSHIFT"}
CROUCH_KEYS = {"LCONTROL", "RCONTROL", "KEY_C"}
JUMP_KEYS = {"SPACE"}

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
    mouse_motion: float = 0.0                # sum |dx|+|dy| over events in this frame
    discrete_keys: Set[str] = field(default_factory=set)   # discrete press events seen this frame
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
            # last frame (or any frame missing the field): estimate using previous step
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

        # 2) mouse — button state + motion deltas come from the RAW input
        # stream (``raw_mouse_events``). That array carries per-event dx/dy
        # deltas and the ``RI_*`` ``button_flag_names`` we rely on for firing /
        # ADS / wheel detection. The processed ``mouse_events`` stream only has
        # absolute cursor x/y (no deltas) and no button flags, so reading it
        # here silently drops ALL combat detection and leaves mouse_motion ~0.
        # Fall back to ``mouse_events`` only if a frame has no raw stream.
        raw_mouse = f.get("raw_mouse_events")
        if not raw_mouse:
            raw_mouse = f.get("mouse_events", [])
        for ms in raw_mouse:
            st.mouse_motion += abs(ms.get("dx", 0) or 0) + abs(ms.get("dy", 0) or 0)
            for fl in ms.get("button_flag_names", []) or []:
                if fl.endswith("_BUTTON_DOWN"):
                    base = fl[: -len("_DOWN")]
                    held_mouse[base] = ms.get("timestamp_ns", ts)
                elif fl.endswith("_BUTTON_UP"):
                    base = fl[: -len("_UP")]
                    held_mouse.pop(base, None)
                elif fl == "RI_MOUSE_WHEEL":
                    st.wheel_events += 1

        # 3) snapshot keys that are still held; drop ones whose last press was too long ago
        active_keys: Set[str] = set()
        for kn, _start_ts in list(held_since.items()):
            lp = last_press_ts.get(kn, 0)
            if ts - lp <= timeout_ns:
                active_keys.add(kn)
            else:
                # treat as released (lost release event)
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
    "KEY_R": ("reloading",          "reload_lock_s"),
    "KEY_1": ("equip_primary",      "weapon_swap_lock_s"),
    "KEY_2": ("equip_secondary",    "weapon_swap_lock_s"),
    "KEY_3": ("holster_weapon",     "weapon_swap_lock_s"),
    "KEY_N": ("weapon_inspect",     "weapon_inspect_lock_s"),
    "KEY_Q": ("tactical_ability",   "tactical_lock_s"),
    "KEY_Z": ("ultimate_ability",   "ultimate_lock_s"),
    "KEY_V": ("melee",              "melee_lock_s"),
    "KEY_G": ("throwing_grenade",   "grenade_lock_s"),
    "KEY_4": ("using_health_item",  "heal_lock_s"),  # heal or shield, indistinguishable from input alone
    # Interact (looting / picking up / reviving / respawning - all medium-confidence aggregate)
    "KEY_F": ("interact",           "interact_lock_s"),
    "KEY_E": ("interact",           "interact_lock_s"),
}


def _build_burst_intervals(
    frame_states: List[FrameState],
    cfg: SegmenterConfig,
) -> List[List[Tuple[str, int]]]:
    """For each frame return a list of ``(label, priority_rank)`` active bursts.

    A burst is created when a key transitions from "not held" to "held" (i.e.
    appears in ``discrete_keys``) or, for the wheel, on every wheel tick. The
    lock window extends ``cfg.<lock_attr>`` seconds forward in wall time.
    """
    n = len(frame_states)
    active: List[List[Tuple[str, int]]] = [[] for _ in range(n)]

    def add_interval(start_idx: int, label: str, lock_s: float, priority: int) -> None:
        end_ts = frame_states[start_idx].timestamp_ns + int(lock_s * NS_PER_S)
        i = start_idx
        while i < n and frame_states[i].timestamp_ns <= end_ts:
            active[i].append((label, priority))
            i += 1

    for idx, st in enumerate(frame_states):
        # Discrete key triggers
        for kn in st.discrete_keys:
            if kn in KEY_TO_BURST:
                label, attr = KEY_TO_BURST[kn]
                add_interval(idx, label, getattr(cfg, attr), priority=_burst_priority(label))
        # Mouse wheel: the player has rebound it away from weapon cycling, so
        # we no longer emit a burst here. Wheel ticks still count as "input"
        # for idle detection (see ``_rolling_no_input``) and are still
        # aggregated into the segment evidence (``wheel_events`` field).
        # Middle-mouse press = ping (treat the first frame the button transitions to held)
        if MMB + "_DOWN" in {f"{m}_DOWN" for m in st.active_mouse}:  # fallback: handled below
            pass
    # Ping detection (fires on MMB DOWN): need to look at button flags directly.
    # We re-scan the raw frame list via timestamps using the "active_mouse" diff.
    prev_mmb = False
    for idx, st in enumerate(frame_states):
        cur_mmb = MMB in st.active_mouse
        if cur_mmb and not prev_mmb:
            add_interval(idx, "pinging", cfg.ping_lock_s, priority=_burst_priority("pinging"))
        prev_mmb = cur_mmb

    return active


def _burst_priority(label: str) -> int:
    """Higher = wins when multiple bursts overlap on the same frame."""
    return {
        "using_health_item": 95,
        "reloading":         90,
        "throwing_grenade":  88,
        "ultimate_ability":  86,
        "tactical_ability":  84,
        "melee":             82,
        "equip_primary":     80,
        "equip_secondary":   80,
        "holster_weapon":    78,
        "weapon_inspect":    72,
        "pinging":           70,
        "interact":          60,
    }.get(label, 50)


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

# Per-frame fallback when no rule matches.
UNKNOWN_LABEL = "unknown"

# Bursts that combat input (LMB / RMB held) is allowed to override. These are
# the discrete actions whose Apex animations CAN be cancelled / interrupted by
# firing or ADS, so if the player presses LMB during the lock window we want
# the segment to reflect what is actually happening on screen (combat) rather
# than the stale lock label.
#
# The complement set inside ``BURST_LABELS`` -- throwing_grenade, melee,
# using_health_item, tactical_ability, ultimate_ability -- are NOT in here
# because those animations genuinely lock out firing in Apex (LMB / RMB
# input during them is ignored or repurposed), so the burst label is the
# correct ground-truth even when LMB happens to be held.
COMBAT_OVERRIDES_BURST: frozenset = frozenset({
    "reloading",
    "equip_primary",
    "equip_secondary",
    "holster_weapon",
    "weapon_inspect",
    "interact",
    "pinging",
})


def _is_held(state: FrameState, keys: Iterable[str]) -> bool:
    return any(k in state.active_keys for k in keys)


def _classify_frame(
    st: FrameState,
    bursts: List[Tuple[str, int]],
    rolling_no_input_s: float,
    cfg: SegmenterConfig,
) -> str:
    # 1) Menu screens (highest priority - they suppress everything)
    if "TAB" in st.active_keys:
        return "menu_inventory"
    if "KEY_M" in st.active_keys:
        return "menu_map"
    if "KEY_B" in st.active_keys:
        return "menu_loadout"
    if "ESCAPE" in st.active_keys or "ESC" in st.active_keys:
        return "menu_pause"

    # 2) Combat is checked before bursts -- but only "overrides" the bursts
    # whose animations can be interrupted by firing (reload / equip_* /
    # holster / cycle / weapon_inspect / interact / pinging). For the bursts
    # that truly lock out firing in Apex (throwing_grenade / melee /
    # using_health_item / tactical_ability / ultimate_ability) we still let
    # the burst label win, because the LMB press during that window is
    # either ignored by the engine or repurposed (e.g. LMB during a grenade
    # cook releases the throw rather than firing the weapon).
    lmb_held = LMB in st.active_mouse
    rmb_held = RMB in st.active_mouse
    combat_input = lmb_held or rmb_held

    top_burst: Optional[str] = None
    if bursts:
        top_burst = sorted(bursts, key=lambda lp: -lp[1])[0][0]

    if combat_input and (top_burst is None or top_burst in COMBAT_OVERRIDES_BURST):
        if lmb_held and rmb_held:
            return "firing_ads"
        if lmb_held:
            return "firing_hipfire"
        return "aiming_down_sights"

    # 3) Burst event won (either no combat input, or the active burst is one
    # that locks out firing).
    if top_burst is not None:
        return top_burst

    # 4) Movement
    moving = _is_held(st, WASD)
    sprinting = _is_held(st, SPRINT_KEYS)
    crouch = _is_held(st, CROUCH_KEYS)
    jumping = _is_held(st, JUMP_KEYS)

    if jumping and moving:
        return "jumping_mantling"
    if sprinting and crouch and moving:
        return "sliding"            # slide = sprint then crouch while moving
    if crouch and moving:
        return "crouch_walking"
    if sprinting and moving:
        return "sprinting"
    if moving:
        return "walking"
    if jumping:
        return "jumping_mantling"

    # 5) Idle: the rolling-no-input counter already factors in mouse motion
    # (any frame whose |dx|+|dy| >= cfg.idle_motion_pixels resets the counter),
    # so we no longer need a separate idle_camera_only branch -- if we reach
    # cfg.idle_min_s of no-input, the player is genuinely standing still.
    if rolling_no_input_s >= cfg.idle_min_s:
        return "idle_stationary"

    # 6) Truly nothing actionable (short gap between actions etc.)
    return UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Segment building & smoothing
# ---------------------------------------------------------------------------


@dataclass
class SubSegment:
    """One contiguous run of frames with a single rule-classifier label.

    A ``Segment`` keeps a list of these so that callers can still see the
    walking / sprinting / sliding breakdown after the tier2 grouping pass
    collapses them into a single dominant-label segment.
    """
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

    Strategy: walk left to right; when a short segment is found, absorb it into
    the longer neighbour (prev / next). If both neighbours are absent (start /
    end), just keep it.

    Segments whose label is in ``unsmoothable_labels`` are exempt from
    smoothing: they are preserved as-is regardless of duration. This protects
    burst events (reload / melee / ping / equip_*) whose lock-window can
    intentionally be shorter than the smoothing threshold from being
    swallowed by a neighbouring walking / sprinting segment.
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
    """Force-merge any segment whose label is in ``absorb_labels`` into a neighbour.

    Unlike :func:`_smooth_segments`, this ignores duration entirely -- the
    labels in ``absorb_labels`` are treated as "non-events" that should never
    own their own segment. If an absorb-label segment has no neighbour at all
    (single segment in the timeline), its label is replaced with
    ``fallback_label``.
    """
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
                # No neighbour at all: relabel rather than emit the absorb label.
                seg.label = fallback_label
                for comp in seg.components:
                    comp.label = fallback_label
                i += 1
    return out


def _merge_segments_by_tier2(
    segments: List[Segment],
    merge_tier2_groups: Iterable[str],
) -> List[Segment]:
    """Collapse contiguous segments sharing a tier2 group into one segment.

    For each run of consecutive segments whose taxonomy ``tier2`` is in
    ``merge_tier2_groups``, the run is merged into a single segment whose
    label is the sub-label with the longest *cumulative* duration in the run
    (ties broken by first occurrence). Sub-segment details are kept in
    ``components`` so the per-sub-label breakdown is not lost.
    """
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
            # Pick the dominant sub-label by total duration, then by earliest
            # appearance (stable for deterministic output).
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
    """For each frame, return seconds of continuous no-input ending at that frame.

    "No input" means: no keys held, no mouse buttons held, no wheel ticks, AND
    the frame's mouse motion (|dx|+|dy|) is below ``cfg.idle_motion_pixels``.
    Any input above threshold resets the counter to 0, so the resulting value
    only crosses ``cfg.idle_min_s`` when the player has truly been still --
    including the camera -- for that long.
    """
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


def _rolling_motion(frame_states: List[FrameState], window_s: float = 0.5) -> List[float]:
    """Sum of |dx|+|dy| over a backward window of ``window_s`` seconds."""
    n = len(frame_states)
    out = [0.0] * n
    win_ns = int(window_s * NS_PER_S)
    j = 0
    running = 0.0
    for i in range(n):
        running += frame_states[i].mouse_motion
        # shrink window from the left
        while j <= i and frame_states[i].timestamp_ns - frame_states[j].timestamp_ns > win_ns:
            running -= frame_states[j].mouse_motion
            j += 1
        out[i] = running
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
    absorbed = _absorb_segments(
        raw_segments, cfg.absorb_labels, cfg.absorb_fallback_label
    )
    smoothed = _smooth_segments(
        absorbed, cfg.min_segment_s, cfg.unsmoothable_labels
    )
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

    # Only emit sub-segment detail when the segment is a merger of distinct
    # sub-labels (i.e. tier2 grouping or absorb actually combined things).
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
        help="Do not collapse consecutive walking / sprinting / sliding / "
        "jumping_mantling segments into one traversal segment. Other "
        "default tier2 merge groups (combat, dropping_in) are unaffected.",
    )
    p.add_argument(
        "--keep-camera-idle",
        action="store_true",
        help="Keep `idle_camera_only` as its own segment instead of merging "
        "it into the surrounding event.",
    )
    p.add_argument(
        "--merge-tier2",
        action="append",
        default=None,
        metavar="TIER2",
        help="Additional taxonomy tier2 groups to collapse "
        "(repeatable, e.g. --merge-tier2 abilities). Defaults from "
        "SegmenterConfig.merge_tier2_groups (currently traversal, combat, "
        "dropping_in) are always included unless --no-merge-traversal is set.",
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

    if args.keep_camera_idle:
        cfg.absorb_labels = tuple(l for l in cfg.absorb_labels if l != "idle_camera_only")

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
