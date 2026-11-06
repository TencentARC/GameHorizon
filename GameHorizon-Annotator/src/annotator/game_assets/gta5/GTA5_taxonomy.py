"""Unified 3-tier event taxonomy for Grand Theft Auto V gameplay segmentation.

This module mirrors the interface used by ``Apex_Legends_taxonomy.py`` while
using GTA5-specific gameplay concepts and key signatures from
``GTA5_keybinds.json``.

The hierarchy is:

* **Tier 1 - Scene** (``in_game`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``on_foot_movement``, ``combat``,
  ``vehicle_driving`` ...)
* **Tier 3 - Atomic event leaf** (``sprinting``, ``vehicle_handbrake`` ...)

Each leaf carries:

* ``action_distinguishable`` - ``"high"`` (rule classifier can emit it),
  ``"medium"`` (rule classifier can emit a parent but vision should refine it),
  ``"vision_only"`` (only the vision pass can produce it).
* ``vision_subtypes`` - if non-empty, this leaf is a parent bucket that a vision
  refinement stage may split into one of the listed sub-leaves.
* ``instruction_phrasing`` - a short imperative fallback for instruction text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

TIER1_IN_GAME = "in_game"
TIER1_GUI = "gui"
TIER1_META = "meta"

TIER1_LABELS: Dict[str, str] = {
    TIER1_IN_GAME: "In-Game Open World / Mission Gameplay",
    TIER1_GUI:     "GUI / Menu Screens",
    TIER1_META:    "Loading, Cutscenes & Flow",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_GAME: {
        "idle":             "Idle",
        "on_foot_movement": "On-Foot Movement",
        "interaction":      "Interaction / Character Actions",
        "weapon_handling":  "Weapon Handling",
        "combat":           "Combat",
        "vehicle_driving":  "Vehicle Driving",
        "aircraft":         "Aircraft Control",
        "camera_hud":       "Camera / HUD",
        "communication":    "Communication",
        "unknown":          "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_pause":        "Pause / Map / Settings",
        "gui_interaction":  "Interaction Menu",
        "gui_phone":        "Phone",
        "gui_weapon_wheel": "Weapon Wheel",
        "gui_navigation":   "Menu Navigation",
        "gui_other":        "Other Menu",
    },
    TIER1_META: {
        "meta_flow":        "Loading / Cutscene / Mission Flow",
    },
}


# ---------------------------------------------------------------------------
# Leaf data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventLeaf:
    label: str
    tier1: str
    tier2: str
    action_distinguishable: str           # "high" | "medium" | "vision_only"
    key_signature: str
    description: str
    vision_subtypes: Tuple[str, ...] = ()
    instruction_phrasing: str = ""


# ---------------------------------------------------------------------------
# All leaves
# ---------------------------------------------------------------------------

LEAVES: List[EventLeaf] = [
    # ---- in_game.idle -------------------------------------------------------
    EventLeaf("idle_stationary", TIER1_IN_GAME, "idle", "high",
              "no input >= idle_min_s, tiny mouse delta",
              "Player is in control but standing or sitting still.",
              instruction_phrasing="Stay still and wait."),
    EventLeaf("idle_camera_only", TIER1_IN_GAME, "idle", "high",
              "no keys held, non-trivial mouse delta",
              "Player is holding position while looking around.",
              instruction_phrasing="Hold position and look around."),

    # ---- in_game.on_foot_movement -----------------------------------------
    EventLeaf("walking", TIER1_IN_GAME, "on_foot_movement", "medium",
              "WASD held, no vehicle / aircraft visible",
              "Normal on-foot movement.",
              vision_subtypes=("walking", "swimming"),
              instruction_phrasing="Walk forward."),
    EventLeaf("sprinting", TIER1_IN_GAME, "on_foot_movement", "medium",
              "LSHIFT + WASD",
              "Fast on-foot movement; may also be swim faster in water.",
              vision_subtypes=("sprinting", "swimming_fast"),
              instruction_phrasing="Sprint forward."),
    EventLeaf("stealth_movement", TIER1_IN_GAME, "on_foot_movement", "medium",
              "LCONTROL + movement",
              "Sneaking or moving in stealth mode.",
              vision_subtypes=("sneaking", "vehicle_ducking"),
              instruction_phrasing="Move in stealth mode."),
    EventLeaf("sneaking", TIER1_IN_GAME, "on_foot_movement", "vision_only",
              "LCONTROL while on foot",
              "On-foot stealth movement.",
              instruction_phrasing="Sneak forward carefully."),
    EventLeaf("jumping_climbing", TIER1_IN_GAME, "on_foot_movement", "high",
              "SPACE burst while on foot",
              "Jumping, vaulting, or climbing over an obstacle.",
              instruction_phrasing="Jump and climb over the obstacle."),
    EventLeaf("swimming", TIER1_IN_GAME, "on_foot_movement", "vision_only",
              "WASD in water",
              "Swimming at normal speed.",
              instruction_phrasing="Swim through the water."),
    EventLeaf("swimming_fast", TIER1_IN_GAME, "on_foot_movement", "vision_only",
              "LSHIFT + WASD in water",
              "Swimming faster.",
              instruction_phrasing="Swim faster."),
    EventLeaf("looking_behind", TIER1_IN_GAME, "on_foot_movement", "high",
              "KEY_C press / hold",
              "Looking behind the player or vehicle.",
              instruction_phrasing="Look behind you."),
    EventLeaf("taking_cover", TIER1_IN_GAME, "on_foot_movement", "high",
              "KEY_Q near cover",
              "Entering, staying in, or leaving cover.",
              instruction_phrasing="Take cover behind the object."),
    EventLeaf("falling_or_ragdoll", TIER1_IN_GAME, "on_foot_movement", "vision_only",
              "no reliable key signature",
              "Falling, stumbling, ragdolling, or recovering control.",
              instruction_phrasing="Regain control after falling."),

    # ---- in_game.interaction -----------------------------------------------
    EventLeaf("context_interaction", TIER1_IN_GAME, "interaction", "medium",
              "KEY_E context action",
              "Generic context action; vision decides the target.",
              vision_subtypes=(
                  "pickup_collectible", "open_close_door",
                  "activate_prompt", "vehicle_horn", "toggle_weapon_accessory",
              ),
              instruction_phrasing="Interact with the prompt."),
    EventLeaf("enter_exit_vehicle", TIER1_IN_GAME, "interaction", "high",
              "KEY_F press",
              "Entering or exiting a nearby vehicle.",
              instruction_phrasing="Enter or exit the vehicle."),
    EventLeaf("character_switch", TIER1_IN_GAME, "interaction", "high",
              "LMENU press / hold",
              "Opening or using the GTA5 character switch wheel.",
              instruction_phrasing="Switch to another character."),
    EventLeaf("pickup_collectible", TIER1_IN_GAME, "interaction", "vision_only",
              "KEY_E on item / pickup prompt",
              "Picking up an item, cash, weapon, or collectible.",
              instruction_phrasing="Pick up the item."),
    EventLeaf("open_close_door", TIER1_IN_GAME, "interaction", "vision_only",
              "KEY_E / context prompt near a door",
              "Opening or closing a door / gate / interactable entrance.",
              instruction_phrasing="Open the door."),
    EventLeaf("activate_prompt", TIER1_IN_GAME, "interaction", "vision_only",
              "KEY_E on mission / world prompt",
              "Activating a mission prompt or world interaction.",
              instruction_phrasing="Activate the prompt."),

    # ---- in_game.weapon_handling -------------------------------------------
    EventLeaf("weapon_wheel", TIER1_IN_GAME, "weapon_handling", "high",
              "TAB held / toggled",
              "Opening the radial weapon wheel.",
              instruction_phrasing="Open the weapon wheel."),
    EventLeaf("quick_switch_weapon", TIER1_IN_GAME, "weapon_handling", "high",
              "RI_MOUSE_WHEEL",
              "Quickly switching weapons or zooming a scope / camera.",
              vision_subtypes=("quick_switch_weapon", "sniper_or_camera_zoom"),
              instruction_phrasing="Quick-switch to another weapon."),
    EventLeaf("reloading", TIER1_IN_GAME, "weapon_handling", "high",
              "KEY_R press",
              "Reloading the equipped weapon.",
              instruction_phrasing="Reload your weapon."),
    EventLeaf("toggle_weapon_accessory", TIER1_IN_GAME, "weapon_handling", "medium",
              "KEY_E while aiming / weapon equipped",
              "Toggling a weapon flashlight, suppressor mode, or accessory.",
              instruction_phrasing="Toggle the weapon accessory."),
    EventLeaf("throwing_explosive", TIER1_IN_GAME, "weapon_handling", "high",
              "KEY_G press",
              "Throwing a grenade or detonating a sticky bomb.",
              instruction_phrasing="Throw or detonate the explosive."),
    EventLeaf("melee_attack", TIER1_IN_GAME, "weapon_handling", "medium",
              "KEY_R / LMB in melee range",
              "Light melee attack or close-range strike.",
              instruction_phrasing="Perform a melee attack."),

    # ---- in_game.combat -----------------------------------------------------
    EventLeaf("aiming", TIER1_IN_GAME, "combat", "high",
              "RI_MOUSE_RIGHT_BUTTON_DOWN held",
              "Aiming, zooming, or locking on without sustained fire.",
              instruction_phrasing="Aim at the target."),
    EventLeaf("firing_weapon", TIER1_IN_GAME, "combat", "high",
              "RI_MOUSE_LEFT_BUTTON_DOWN without RMB",
              "Firing or attacking without aiming down sights.",
              instruction_phrasing="Fire at the target."),
    EventLeaf("firing_ads", TIER1_IN_GAME, "combat", "high",
              "RI_MOUSE_LEFT_BUTTON_DOWN while RMB held",
              "Firing while aiming or locked on.",
              instruction_phrasing="Aim and fire at the target."),
    EventLeaf("combat_in_cover", TIER1_IN_GAME, "combat", "medium",
              "KEY_Q near cover plus aim / fire",
              "Fighting from cover.",
              vision_subtypes=("combat_in_cover", "taking_cover"),
              instruction_phrasing="Fight from cover."),
    EventLeaf("wanted_escape_or_chase", TIER1_IN_GAME, "combat", "vision_only",
              "movement / driving with wanted-level HUD",
              "Escaping police, chasing a target, or being pursued.",
              instruction_phrasing="Evade the pursuit."),

    # ---- in_game.vehicle_driving -------------------------------------------
    EventLeaf("vehicle_driving", TIER1_IN_GAME, "vehicle_driving", "medium",
              "WASD while inside a land vehicle",
              "Generic land vehicle driving.",
              vision_subtypes=(
                  "vehicle_accelerating", "vehicle_braking_reversing",
                  "vehicle_steering", "vehicle_cruising",
              ),
              instruction_phrasing="Drive the vehicle."),
    EventLeaf("vehicle_accelerating", TIER1_IN_GAME, "vehicle_driving", "medium",
              "KEY_W in vehicle",
              "Accelerating forward in a car, bike, boat, or similar vehicle.",
              instruction_phrasing="Accelerate forward."),
    EventLeaf("vehicle_braking_reversing", TIER1_IN_GAME, "vehicle_driving", "medium",
              "KEY_S in vehicle",
              "Braking or reversing a vehicle.",
              instruction_phrasing="Brake or reverse the vehicle."),
    EventLeaf("vehicle_steering", TIER1_IN_GAME, "vehicle_driving", "medium",
              "KEY_A / KEY_D in vehicle",
              "Steering a vehicle left or right.",
              instruction_phrasing="Steer the vehicle."),
    EventLeaf("vehicle_cruising", TIER1_IN_GAME, "vehicle_driving", "vision_only",
              "vehicle visible, little active input",
              "Coasting, cruising, or rolling with minimal input.",
              instruction_phrasing="Keep driving along the road."),
    EventLeaf("vehicle_handbrake", TIER1_IN_GAME, "vehicle_driving", "high",
              "SPACE in vehicle",
              "Using the handbrake, often to drift or stop sharply.",
              instruction_phrasing="Use the handbrake."),
    EventLeaf("vehicle_horn", TIER1_IN_GAME, "vehicle_driving", "medium",
              "KEY_E in vehicle",
              "Honking the vehicle horn.",
              instruction_phrasing="Honk the horn."),
    EventLeaf("vehicle_headlights", TIER1_IN_GAME, "vehicle_driving", "high",
              "KEY_H in vehicle",
              "Toggling vehicle headlights.",
              instruction_phrasing="Toggle the headlights."),
    EventLeaf("vehicle_ducking", TIER1_IN_GAME, "vehicle_driving", "high",
              "KEY_X or LCONTROL in vehicle",
              "Ducking down inside a vehicle.",
              instruction_phrasing="Duck down in the vehicle."),
    EventLeaf("vehicle_weapon_fire", TIER1_IN_GAME, "vehicle_driving", "high",
              "RI_MOUSE_LEFT_BUTTON_DOWN while in vehicle",
              "Firing a vehicle-mounted or drive-by weapon.",
              instruction_phrasing="Fire the vehicle weapon."),
    EventLeaf("radio_wheel", TIER1_IN_GAME, "vehicle_driving", "high",
              "KEY_Q in vehicle",
              "Opening or using the radio station wheel.",
              instruction_phrasing="Open the radio wheel."),

    # ---- in_game.aircraft ---------------------------------------------------
    EventLeaf("aircraft_flying", TIER1_IN_GAME, "aircraft", "medium",
              "WASD / NUMPAD controls while in aircraft",
              "Generic aircraft or helicopter flight.",
              vision_subtypes=(
                  "aircraft_throttle_up", "aircraft_throttle_down",
                  "aircraft_pitching", "aircraft_rolling", "aircraft_landing_gear",
              ),
              instruction_phrasing="Fly the aircraft."),
    EventLeaf("aircraft_throttle_up", TIER1_IN_GAME, "aircraft", "medium",
              "KEY_W in aircraft",
              "Increasing aircraft throttle.",
              instruction_phrasing="Increase throttle."),
    EventLeaf("aircraft_throttle_down", TIER1_IN_GAME, "aircraft", "medium",
              "KEY_S in aircraft",
              "Reducing aircraft throttle.",
              instruction_phrasing="Reduce throttle."),
    EventLeaf("aircraft_pitching", TIER1_IN_GAME, "aircraft", "high",
              "NUMPAD8 / NUMPAD5",
              "Pitching the aircraft nose forward or back.",
              instruction_phrasing="Pitch the aircraft."),
    EventLeaf("aircraft_rolling", TIER1_IN_GAME, "aircraft", "high",
              "NUMPAD4 / NUMPAD6",
              "Rolling the aircraft left or right.",
              instruction_phrasing="Roll the aircraft."),
    EventLeaf("aircraft_landing_gear", TIER1_IN_GAME, "aircraft", "high",
              "KEY_G in aircraft",
              "Toggling aircraft landing gear.",
              instruction_phrasing="Toggle the landing gear."),

    # ---- in_game.camera_hud -------------------------------------------------
    EventLeaf("change_camera_view", TIER1_IN_GAME, "camera_hud", "high",
              "KEY_V press",
              "Changing camera view, including first-person / third-person.",
              instruction_phrasing="Change the camera view."),
    EventLeaf("cinematic_camera", TIER1_IN_GAME, "camera_hud", "medium",
              "KEY_R in vehicle / non-combat context",
              "Toggling or using the cinematic camera.",
              instruction_phrasing="Switch to the cinematic camera."),
    EventLeaf("radar_zoom", TIER1_IN_GAME, "camera_hud", "high",
              "KEY_Z press",
              "Zooming or expanding the radar / minimap.",
              instruction_phrasing="Zoom the radar."),
    EventLeaf("sniper_or_camera_zoom", TIER1_IN_GAME, "camera_hud", "medium",
              "RI_MOUSE_WHEEL while scoped / camera active",
              "Zooming a sniper scope or camera view.",
              instruction_phrasing="Adjust the zoom level."),

    # ---- in_game.communication ---------------------------------------------
    EventLeaf("push_to_talk", TIER1_IN_GAME, "communication", "high",
              "KEY_N held",
              "Using push-to-talk voice chat.",
              instruction_phrasing="Hold push-to-talk and speak."),
    EventLeaf("point_at", TIER1_IN_GAME, "communication", "high",
              "KEY_B press",
              "Pointing at something in multiplayer / online mode.",
              instruction_phrasing="Point at the target."),

    # ---- in_game.unknown ----------------------------------------------------
    EventLeaf("unknown", TIER1_IN_GAME, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  "walking", "sprinting", "sneaking", "jumping_climbing",
                  "swimming", "taking_cover",
                  "context_interaction", "enter_exit_vehicle",
                  "weapon_wheel", "reloading", "throwing_explosive",
                  "aiming", "firing_weapon", "firing_ads",
                  "vehicle_driving", "vehicle_accelerating",
                  "vehicle_braking_reversing", "vehicle_steering",
                  "vehicle_handbrake", "vehicle_weapon_fire",
                  "aircraft_flying", "aircraft_pitching", "aircraft_rolling",
                  "change_camera_view", "radar_zoom",
                  "menu_pause", "menu_interaction", "menu_phone",
                  "loading_screen", "cutscene", "mission_flow", "unknown",
              ),
              instruction_phrasing="Continue the current activity."),

    # ---- gui ----------------------------------------------------------------
    EventLeaf("menu_pause", TIER1_GUI, "gui_pause", "high",
              "ESCAPE or KEY_P",
              "Pause menu, map, or settings screen.",
              instruction_phrasing="Open the pause menu."),
    EventLeaf("menu_interaction", TIER1_GUI, "gui_interaction", "high",
              "KEY_M",
              "GTA Online interaction menu.",
              instruction_phrasing="Open the interaction menu."),
    EventLeaf("menu_phone", TIER1_GUI, "gui_phone", "high",
              "RI_MOUSE_MIDDLE_BUTTON_DOWN",
              "Phone UI open.",
              instruction_phrasing="Open the phone."),
    EventLeaf("menu_weapon_wheel", TIER1_GUI, "gui_weapon_wheel", "high",
              "TAB held / toggled",
              "Weapon wheel overlay open.",
              instruction_phrasing="Choose a weapon from the wheel."),
    EventLeaf("menu_navigation", TIER1_GUI, "gui_navigation", "high",
              "UP / DOWN / LEFT / RIGHT / RETURN / BACK / ESCAPE",
              "Navigating, confirming, or backing out of a menu.",
              instruction_phrasing="Navigate the menu."),
    EventLeaf("menu_settings", TIER1_GUI, "gui_pause", "vision_only",
              "pause/settings screen visible",
              "Changing settings or browsing pause-menu tabs.",
              instruction_phrasing="Adjust the settings."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no unique key signature",
              "Other overlay, shop, mission, dialog, or online menu.",
              instruction_phrasing="Use the menu."),

    # ---- meta ---------------------------------------------------------------
    EventLeaf("loading_screen", TIER1_META, "meta_flow", "vision_only",
              "no input, loading artwork / black screen",
              "Loading screen or transition.",
              instruction_phrasing="Wait through the loading screen."),
    EventLeaf("cutscene", TIER1_META, "meta_flow", "vision_only",
              "cinematic framing, player not in control",
              "Story cutscene or scripted cinematic.",
              instruction_phrasing="Watch the cutscene."),
    EventLeaf("mission_flow", TIER1_META, "meta_flow", "vision_only",
              "mission title / objective / result screen",
              "Mission start, objective update, passed / failed screen, or respawn flow.",
              instruction_phrasing="Proceed through the mission flow."),
]


# ---------------------------------------------------------------------------
# Indexes & convenience views
# ---------------------------------------------------------------------------


LEAF_BY_LABEL: Dict[str, EventLeaf] = {leaf.label: leaf for leaf in LEAVES}

ALL_LABELS: List[str] = [leaf.label for leaf in LEAVES]

ACTION_DISTINGUISHABLE_LABELS: List[str] = [
    leaf.label for leaf in LEAVES if leaf.action_distinguishable == "high"
]
VISION_ONLY_LABELS: List[str] = [
    leaf.label for leaf in LEAVES if leaf.action_distinguishable == "vision_only"
]
VISION_PARENT_LABELS: List[str] = [
    leaf.label for leaf in LEAVES if leaf.vision_subtypes
]


# ---------------------------------------------------------------------------
# Backwards-compatible aliases / common label normalizations
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    "idle": "idle_stationary",
    "move_forward": "walking",
    "move_back": "walking",
    "move_left": "walking",
    "move_right": "walking",
    "running": "sprinting",
    "sprint": "sprinting",
    "stealth": "stealth_movement",
    "jump": "jumping_climbing",
    "climb": "jumping_climbing",
    "cover": "taking_cover",
    "interact": "context_interaction",
    "enter_vehicle": "enter_exit_vehicle",
    "exit_vehicle": "enter_exit_vehicle",
    "reload": "reloading",
    "fire_weapon": "firing_weapon",
    "attack": "firing_weapon",
    "aim": "aiming",
    "zoom": "sniper_or_camera_zoom",
    "throw_grenade": "throwing_explosive",
    "detonate_sticky_bomb": "throwing_explosive",
    "drive": "vehicle_driving",
    "vehicle_accelerate": "vehicle_accelerating",
    "vehicle_brake_reverse": "vehicle_braking_reversing",
    "vehicle_steer": "vehicle_steering",
    "handbrake": "vehicle_handbrake",
    "horn": "vehicle_horn",
    "headlights": "vehicle_headlights",
    "aircraft_throttle": "aircraft_flying",
    "landing_gear": "aircraft_landing_gear",
    "pause_menu": "menu_pause",
    "interaction_menu": "menu_interaction",
    "phone": "menu_phone",
    "weapon_wheel": "weapon_wheel",
    "menu_ui": "menu_other",
    "loading": "loading_screen",
    "cinematic": "cutscene",
}


def normalize_label(label: str) -> str:
    """Map legacy / model-suggested labels into the canonical leaf names."""
    if label in LEAF_BY_LABEL:
        return label
    return LEGACY_LABEL_ALIASES.get(label, "unknown")


def get_leaf(label: str) -> EventLeaf:
    return LEAF_BY_LABEL[normalize_label(label)]


# ---------------------------------------------------------------------------
# Hierarchical views
# ---------------------------------------------------------------------------


def vision_needed(label: str) -> bool:
    """True iff a vision refinement pass should inspect this segment."""
    if label == "unknown":
        return True
    leaf = LEAF_BY_LABEL.get(label)
    if leaf is None:
        return True
    return bool(leaf.vision_subtypes) or leaf.action_distinguishable in ("medium", "vision_only")


def subtype_choices(parent_label: str) -> Tuple[str, ...]:
    """Return candidate sub-leaf labels for ``parent_label``.

    Always includes the parent label itself as an "keep parent" option.
    """
    leaf = LEAF_BY_LABEL.get(parent_label)
    if leaf is None or not leaf.vision_subtypes:
        return (parent_label,)
    seen: Set[str] = set()
    out: List[str] = []
    for candidate in leaf.vision_subtypes + (parent_label,):
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return tuple(out)


def labels_by_tier() -> Dict[str, Dict[str, List[str]]]:
    """``tier1 -> tier2 -> [leaf_labels]`` (preserves declaration order)."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for leaf in LEAVES:
        out.setdefault(leaf.tier1, {}).setdefault(leaf.tier2, []).append(leaf.label)
    return out


def format_taxonomy_for_prompt(
    only_subtypes_of: Optional[str] = None,
    include_descriptions: bool = False,
) -> str:
    """Format the taxonomy as a compact bullet list for LLM prompts."""
    if only_subtypes_of is not None:
        candidates = subtype_choices(only_subtypes_of)
        lines: List[str] = [f"Candidates for parent '{only_subtypes_of}':"]
        for label in candidates:
            leaf = LEAF_BY_LABEL.get(label)
            if leaf is None:
                lines.append(f"  - {label}")
                continue
            tag = leaf.action_distinguishable
            if include_descriptions:
                lines.append(f"  - {label} [{tag}] -- {leaf.description}")
            else:
                lines.append(f"  - {label} [{tag}]")
        return "\n".join(lines)

    lines = []
    for tier1, tier2_groups in labels_by_tier().items():
        lines.append(f"- {TIER1_LABELS[tier1]} ({tier1})")
        for tier2, leaves in tier2_groups.items():
            tier2_human = TIER2_LABELS.get(tier1, {}).get(tier2, tier2)
            lines.append(f"  - {tier2_human} ({tier2})")
            for label in leaves:
                leaf = LEAF_BY_LABEL[label]
                tag = leaf.action_distinguishable
                if include_descriptions:
                    lines.append(f"    * {label} [{tag}] -- {leaf.description}")
                else:
                    lines.append(f"    * {label} [{tag}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atomic-label vocabulary used by instruction prompts
# ---------------------------------------------------------------------------

ATOMIC_LABELS_FOR_INSTRUCTION_PROMPT: str = "\n".join(
    f"- {leaf.label}: {leaf.description}"
    for leaf in LEAVES
    if leaf.label != "unknown"
)


# ---------------------------------------------------------------------------
# CLI / debug
# ---------------------------------------------------------------------------


def to_dict() -> Dict[str, object]:
    tree: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for leaf in LEAVES:
        tree.setdefault(leaf.tier1, {}).setdefault(leaf.tier2, []).append(asdict(leaf))
    return {
        "tier1_labels": TIER1_LABELS,
        "tier2_labels": TIER2_LABELS,
        "tree": tree,
        "leaf_count": len(LEAVES),
        "action_distinguishable": ACTION_DISTINGUISHABLE_LABELS,
        "vision_only": VISION_ONLY_LABELS,
        "vision_parents": {p: list(LEAF_BY_LABEL[p].vision_subtypes) for p in VISION_PARENT_LABELS},
        "legacy_aliases": LEGACY_LABEL_ALIASES,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        out = sys.argv[2] if len(sys.argv) > 2 else "GTA5_taxonomy.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(to_dict(), f, ensure_ascii=False, indent=2)
        print(f"wrote taxonomy -> {out} ({len(LEAVES)} leaves)")
    else:
        print(format_taxonomy_for_prompt(include_descriptions=True))
        print()
        print(f"# {len(LEAVES)} leaves total")
        print(f"# {len(ACTION_DISTINGUISHABLE_LABELS)} high-confidence (rule-emitted)")
        print(f"# {len(VISION_ONLY_LABELS)} vision-only")
        print(f"# {len(VISION_PARENT_LABELS)} parents that vision can refine")
