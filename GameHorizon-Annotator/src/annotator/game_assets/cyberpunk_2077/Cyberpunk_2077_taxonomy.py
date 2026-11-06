"""Unified 3-tier event taxonomy for Cyberpunk 2077 gameplay segmentation.

This module mirrors the structure used by ``Apex_Legends_taxonomy`` while
adapting the leaf labels and key signatures to Cyberpunk 2077's keyboard and
mouse controls.

The hierarchy is:

* **Tier 1 - Scene** (``in_world`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``traversal``, ``combat``, ``vehicle`` ...)
* **Tier 3 - Atomic event leaf** (``sprinting``, ``firing_ads`` ...)

Each leaf carries:

* ``action_distinguishable`` - ``"high"`` (rule classifier outputs it),
  ``"medium"`` (rule classifier outputs but a vision pass should refine it),
  ``"vision_only"`` (only the vision pass can produce it).
* ``vision_subtypes`` - if non-empty, this leaf is a *parent bucket* that the
  Gemini refinement stage may split into one of the listed sub-leaves.
* ``instruction_phrasing`` - a short imperative fallback used when Stage 3
  decides not to call the captioning model (very short segments, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

TIER1_IN_WORLD = "in_world"
TIER1_GUI = "gui"
TIER1_META = "meta"

TIER1_LABELS: Dict[str, str] = {
    TIER1_IN_WORLD: "In-World Gameplay",
    TIER1_GUI:      "GUI / Menu Screens",
    TIER1_META:     "Meta / Cinematic Flow",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_WORLD: {
        "idle":             "Idle",
        "traversal":        "Traversal / Parkour",
        "stealth":          "Stealth / Crouch",
        "interaction":      "Interaction / Looting",
        "dialogue":         "Dialogue",
        "weapon_handling":  "Weapon Handling",
        "combat":           "Combat",
        "quickhacking":     "Scanning / Quickhacking",
        "support":          "Consumables / Recovery",
        "vehicle":          "Vehicle Gameplay",
        "braindance":       "Braindance Playback",
        "unknown":          "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_inventory":     "Inventory / Character Screen",
        "gui_map":           "Map Screen",
        "gui_journal":       "Journal / Quests",
        "gui_crafting":      "Crafting Screen",
        "gui_phone":         "Phone / Messages",
        "gui_notifications": "Notifications / Database",
        "gui_pause":         "Pause / Settings",
        "gui_radial":        "Radial / Selection Wheels",
        "gui_other":         "Other Menu",
    },
    TIER1_META: {
        "meta_menu":         "Title / Main Menu",
        "meta_flow":         "Loading / Cinematic / Cutscene",
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
    # ---- in_world.idle ------------------------------------------------------
    EventLeaf("idle_stationary", TIER1_IN_WORLD, "idle", "high",
              "no input >= idle_min_s, tiny mouse delta",
              "V is standing still or waiting without meaningful camera motion.",
              instruction_phrasing="Stand still and wait."),
    EventLeaf("idle_camera_only", TIER1_IN_WORLD, "idle", "high",
              "no keys held, non-trivial mouse delta",
              "Looking around in first person without moving.",
              instruction_phrasing="Hold position and look around."),

    # ---- in_world.traversal -------------------------------------------------
    EventLeaf("walking", TIER1_IN_WORLD, "traversal", "high",
              "KEY_W / KEY_A / KEY_S / KEY_D held, no sprint or crouch",
              "Normal on-foot traversal through the world.",
              instruction_phrasing="Walk through the area."),
    EventLeaf("sprinting", TIER1_IN_WORLD, "traversal", "high",
              "LSHIFT or RSHIFT + WASD",
              "Sprinting on foot.",
              instruction_phrasing="Sprint forward."),
    EventLeaf("jumping", TIER1_IN_WORLD, "traversal", "high",
              "SPACE press while on foot",
              "Jumping or vaulting a low obstacle.",
              instruction_phrasing="Jump over the obstacle."),
    EventLeaf("parkour_climbing", TIER1_IN_WORLD, "traversal", "vision_only",
              "SPACE + forward motion near a ledge",
              "Climbing, vaulting, or mantling geometry.",
              instruction_phrasing="Climb up onto the ledge."),
    EventLeaf("swimming", TIER1_IN_WORLD, "traversal", "vision_only",
              "WASD + mouse while underwater",
              "Swimming or diving in water.",
              instruction_phrasing="Swim through the water."),

    # ---- in_world.stealth ---------------------------------------------------
    EventLeaf("crouching", TIER1_IN_WORLD, "stealth", "high",
              "KEY_C toggle or LCONTROL hold, no WASD",
              "Entering or holding a crouched stance.",
              instruction_phrasing="Crouch down."),
    EventLeaf("crouch_walking", TIER1_IN_WORLD, "stealth", "high",
              "KEY_C / LCONTROL + WASD",
              "Moving while crouched for stealth.",
              instruction_phrasing="Crouch-walk quietly."),
    EventLeaf("stealth_takedown", TIER1_IN_WORLD, "stealth", "vision_only",
              "F / LMB prompt behind an enemy",
              "Performing a stealth grab or takedown.",
              instruction_phrasing="Silently neutralize the enemy."),
    EventLeaf("hiding_body", TIER1_IN_WORLD, "stealth", "vision_only",
              "F interaction while carrying or near a body container",
              "Hiding or disposing of a body.",
              instruction_phrasing="Hide the body out of sight."),

    # ---- in_world.interaction ----------------------------------------------
    EventLeaf("interact", TIER1_IN_WORLD, "interaction", "medium",
              "KEY_F single press / hold",
              "Generic world interaction; vision splits into pickup, doors, terminals, vendors, and vehicle entry.",
              vision_subtypes=(
                  "pickup_item", "open_close_door", "use_terminal",
                  "talk_to_npc", "vendor_interaction", "enter_exit_vehicle",
              ),
              instruction_phrasing="Interact with the highlighted object."),
    EventLeaf("pickup_item", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_F on loot prompt",
              "Picking up loot, ammo, shards, or mission items.",
              instruction_phrasing="Pick up the item."),
    EventLeaf("open_close_door", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_F on door prompt",
              "Opening, closing, forcing, or unlocking a door.",
              instruction_phrasing="Open the door."),
    EventLeaf("use_terminal", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_F on computer / access point / elevator",
              "Using a terminal, elevator panel, access point, or other device.",
              instruction_phrasing="Use the terminal."),
    EventLeaf("talk_to_npc", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_F near an NPC prompt",
              "Starting or continuing a conversation with an NPC.",
              instruction_phrasing="Talk to the NPC."),
    EventLeaf("vendor_interaction", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_F on shop / ripperdoc / vendor",
              "Opening a shop, ripperdoc, stash, or service interface.",
              instruction_phrasing="Open the vendor interface."),

    # ---- in_world.dialogue --------------------------------------------------
    EventLeaf("dialogue_confirm", TIER1_IN_WORLD, "dialogue", "high",
              "KEY_Y press",
              "Confirming or selecting a dialogue option.",
              instruction_phrasing="Confirm the dialogue choice."),
    EventLeaf("skip_dialogue", TIER1_IN_WORLD, "dialogue", "high",
              "KEY_C or KEY_Y during dialogue",
              "Skipping a spoken line or fast-forwarding dialogue.",
              instruction_phrasing="Skip the current dialogue line."),
    EventLeaf("dialogue_choice", TIER1_IN_WORLD, "dialogue", "vision_only",
              "mouse / keyboard selection on dialogue wheel",
              "Choosing among multiple dialogue responses.",
              instruction_phrasing="Choose a dialogue response."),

    # ---- in_world.weapon_handling ------------------------------------------
    EventLeaf("draw_or_holster_weapon", TIER1_IN_WORLD, "weapon_handling", "high",
              "KEY_B press",
              "Drawing or withdrawing the currently equipped weapon.",
              instruction_phrasing="Draw or holster your weapon."),
    EventLeaf("equip_weapon_slot_1", TIER1_IN_WORLD, "weapon_handling", "high",
              "KEY_1 tap",
              "Selecting the first weapon slot.",
              instruction_phrasing="Equip the first weapon."),
    EventLeaf("equip_weapon_slot_2", TIER1_IN_WORLD, "weapon_handling", "high",
              "KEY_2 tap",
              "Selecting the second weapon slot.",
              instruction_phrasing="Equip the second weapon."),
    EventLeaf("equip_weapon_slot_3", TIER1_IN_WORLD, "weapon_handling", "high",
              "KEY_3 tap",
              "Selecting the third weapon slot.",
              instruction_phrasing="Equip the third weapon."),
    EventLeaf("cycle_weapon", TIER1_IN_WORLD, "weapon_handling", "high",
              "RI_MOUSE_WHEEL or LMENU tap",
              "Cycling weapons or opening the weapon wheel.",
              instruction_phrasing="Cycle to another weapon."),
    EventLeaf("reload_weapon", TIER1_IN_WORLD, "weapon_handling", "high",
              "KEY_R press",
              "Reloading the equipped weapon.",
              instruction_phrasing="Reload your weapon."),

    # ---- in_world.combat ----------------------------------------------------
    EventLeaf("aiming_down_sights", TIER1_IN_WORLD, "combat", "high",
              "RI_MOUSE_RIGHT_BUTTON_DOWN held without sustained LMB",
              "Aiming or blocking without attacking.",
              instruction_phrasing="Aim down sights."),
    EventLeaf("firing_hipfire", TIER1_IN_WORLD, "combat", "high",
              "RI_MOUSE_LEFT_BUTTON_DOWN without RMB",
              "Firing a ranged weapon without aiming down sights.",
              instruction_phrasing="Hip-fire at the enemy."),
    EventLeaf("firing_ads", TIER1_IN_WORLD, "combat", "high",
              "RI_MOUSE_LEFT_BUTTON_DOWN while RMB is held",
              "Firing while aiming down sights.",
              instruction_phrasing="Aim and fire at the enemy."),
    EventLeaf("melee_fast_attack", TIER1_IN_WORLD, "combat", "medium",
              "RI_MOUSE_LEFT_BUTTON_DOWN with melee weapon / fists",
              "Performing a fast melee attack.",
              instruction_phrasing="Strike the enemy with a fast attack."),
    EventLeaf("blocking_or_heavy_attack", TIER1_IN_WORLD, "combat", "medium",
              "RI_MOUSE_RIGHT_BUTTON_DOWN with melee weapon",
              "Blocking or preparing a heavy melee action.",
              instruction_phrasing="Block the incoming attack."),
    EventLeaf("combat_gadget", TIER1_IN_WORLD, "combat", "high",
              "RI_MOUSE_MIDDLE_BUTTON_DOWN",
              "Using a combat gadget such as a grenade or equipped cyberware gadget.",
              instruction_phrasing="Use your combat gadget."),

    # ---- in_world.quickhacking ---------------------------------------------
    EventLeaf("scanner_hold", TIER1_IN_WORLD, "quickhacking", "high",
              "TAB held",
              "Holding the scanner overlay to inspect enemies, objects, and devices.",
              instruction_phrasing="Scan the area."),
    EventLeaf("quick_attack", TIER1_IN_WORLD, "quickhacking", "high",
              "KEY_Q press",
              "Triggering a quick attack or contextual quickhack action.",
              instruction_phrasing="Use a quick attack."),
    EventLeaf("cyberware_systems", TIER1_IN_WORLD, "quickhacking", "high",
              "KEY_E press",
              "Opening or activating cyberware systems.",
              instruction_phrasing="Use your cyberware system."),
    EventLeaf("tag_target", TIER1_IN_WORLD, "quickhacking", "medium",
              "RI_MOUSE_MIDDLE_BUTTON_DOWN while scanner is active",
              "Tagging a target or device through scanner vision.",
              instruction_phrasing="Tag the target."),
    EventLeaf("quickhack_details", TIER1_IN_WORLD, "quickhacking", "high",
              "KEY_Z press",
              "Opening quickhack details or related notification information.",
              instruction_phrasing="Open the quickhack details."),

    # ---- in_world.support ---------------------------------------------------
    EventLeaf("use_consumable", TIER1_IN_WORLD, "support", "high",
              "KEY_X press",
              "Using a healing item, inhaler, food, or selected consumable.",
              vision_subtypes=("heal_with_consumable", "buff_with_consumable"),
              instruction_phrasing="Use a consumable."),
    EventLeaf("heal_with_consumable", TIER1_IN_WORLD, "support", "vision_only",
              "KEY_X + health recovery on HUD",
              "Using a consumable to recover health.",
              instruction_phrasing="Heal with a consumable."),
    EventLeaf("buff_with_consumable", TIER1_IN_WORLD, "support", "vision_only",
              "KEY_X + buff / food / drink feedback",
              "Using food, drink, or another buff consumable.",
              instruction_phrasing="Use the consumable for a buff."),

    # ---- in_world.vehicle ---------------------------------------------------
    EventLeaf("driving_forward", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_W held while in vehicle",
              "Accelerating or driving forward in a vehicle.",
              instruction_phrasing="Drive forward."),
    EventLeaf("vehicle_brake_reverse", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_S held while in vehicle",
              "Braking or reversing a vehicle.",
              instruction_phrasing="Brake or reverse the vehicle."),
    EventLeaf("vehicle_turning", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_A / KEY_D held while in vehicle",
              "Steering a vehicle left or right.",
              instruction_phrasing="Steer the vehicle."),
    EventLeaf("vehicle_handbrake", TIER1_IN_WORLD, "vehicle", "high",
              "SPACE while in vehicle",
              "Using the handbrake.",
              instruction_phrasing="Use the handbrake."),
    EventLeaf("enter_exit_vehicle", TIER1_IN_WORLD, "vehicle", "medium",
              "KEY_F near / inside vehicle",
              "Entering or exiting a vehicle.",
              instruction_phrasing="Enter or exit the vehicle."),
    EventLeaf("call_vehicle", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_V press",
              "Calling the personal vehicle.",
              instruction_phrasing="Call your vehicle."),
    EventLeaf("vehicle_horn", TIER1_IN_WORLD, "vehicle", "high",
              "LCONTROL while in vehicle",
              "Using the vehicle horn or a vehicle special action.",
              instruction_phrasing="Use the vehicle horn."),
    EventLeaf("vehicle_camera", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_Q press while in vehicle",
              "Cycling the vehicle camera.",
              instruction_phrasing="Cycle the vehicle camera."),
    EventLeaf("vehicle_radio", TIER1_IN_WORLD, "vehicle", "high",
              "KEY_R press while in vehicle",
              "Opening or changing vehicle radio.",
              instruction_phrasing="Open the vehicle radio."),
    EventLeaf("vehicle_lights", TIER1_IN_WORLD, "vehicle", "high",
              "LMENU press while in vehicle",
              "Cycling vehicle lights.",
              instruction_phrasing="Cycle the vehicle lights."),
    EventLeaf("vehicle_combat", TIER1_IN_WORLD, "vehicle", "medium",
              "LMB / RMB while in vehicle or Panzer",
              "Shooting from a vehicle or using Panzer weapons.",
              vision_subtypes=("vehicle_primary_fire", "vehicle_secondary_fire"),
              instruction_phrasing="Fire the vehicle weapon."),
    EventLeaf("vehicle_primary_fire", TIER1_IN_WORLD, "vehicle", "vision_only",
              "RI_MOUSE_LEFT_BUTTON_DOWN while in vehicle / Panzer",
              "Using the vehicle or Panzer primary weapon.",
              instruction_phrasing="Fire the primary vehicle weapon."),
    EventLeaf("vehicle_secondary_fire", TIER1_IN_WORLD, "vehicle", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN while in vehicle / Panzer",
              "Using the vehicle or Panzer secondary weapon.",
              instruction_phrasing="Fire the secondary vehicle weapon."),

    # ---- in_world.braindance ------------------------------------------------
    EventLeaf("braindance_pause", TIER1_IN_WORLD, "braindance", "high",
              "SPACE while in braindance",
              "Pausing or resuming braindance playback.",
              instruction_phrasing="Pause the braindance."),
    EventLeaf("braindance_rewind", TIER1_IN_WORLD, "braindance", "high",
              "KEY_Q while in braindance",
              "Rewinding braindance playback.",
              instruction_phrasing="Rewind the braindance."),
    EventLeaf("braindance_fast_forward", TIER1_IN_WORLD, "braindance", "high",
              "KEY_E while in braindance",
              "Fast-forwarding braindance playback.",
              instruction_phrasing="Fast-forward the braindance."),
    EventLeaf("braindance_restart", TIER1_IN_WORLD, "braindance", "high",
              "KEY_R while in braindance",
              "Restarting braindance playback.",
              instruction_phrasing="Restart the braindance."),
    EventLeaf("braindance_layer_change", TIER1_IN_WORLD, "braindance", "high",
              "LSHIFT while in braindance",
              "Changing braindance analysis layer.",
              instruction_phrasing="Switch braindance layers."),
    EventLeaf("braindance_analysis_mode", TIER1_IN_WORLD, "braindance", "high",
              "TAB while in braindance",
              "Entering or leaving braindance analysis mode.",
              instruction_phrasing="Enter braindance analysis mode."),
    EventLeaf("braindance_zoom", TIER1_IN_WORLD, "braindance", "high",
              "RI_MOUSE_WHEEL or RI_MOUSE_MIDDLE_BUTTON_DOWN while in braindance",
              "Zooming in or out in braindance.",
              instruction_phrasing="Zoom the braindance view."),
    EventLeaf("braindance_exit", TIER1_IN_WORLD, "braindance", "high",
              "KEY_X while in braindance",
              "Exiting the braindance editor or sequence.",
              instruction_phrasing="Exit the braindance."),

    # ---- in_world.unknown (catch-all parent) -------------------------------
    EventLeaf("unknown", TIER1_IN_WORLD, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  "walking", "sprinting", "jumping", "parkour_climbing",
                  "crouching", "crouch_walking", "pickup_item",
                  "open_close_door", "use_terminal", "talk_to_npc",
                  "dialogue_choice", "firing_hipfire", "firing_ads",
                  "melee_fast_attack", "scanner_hold", "quick_attack",
                  "driving_forward", "vehicle_combat", "braindance_pause",
                  "menu_inventory", "menu_map", "menu_pause",
                  "cinematic_cutscene", "loading_screen", "unknown",
              ),
              instruction_phrasing="Continue with the current activity."),

    # ---- gui ----------------------------------------------------------------
    EventLeaf("menu_inventory", TIER1_GUI, "gui_inventory", "high",
              "KEY_I press",
              "Inventory or character equipment screen.",
              instruction_phrasing="Open the inventory."),
    EventLeaf("menu_map", TIER1_GUI, "gui_map", "high",
              "KEY_M press",
              "World map screen.",
              instruction_phrasing="Open the map."),
    EventLeaf("menu_journal", TIER1_GUI, "gui_journal", "high",
              "KEY_J press",
              "Quest journal screen.",
              instruction_phrasing="Open the journal."),
    EventLeaf("menu_crafting", TIER1_GUI, "gui_crafting", "high",
              "KEY_K press",
              "Crafting screen.",
              instruction_phrasing="Open the crafting screen."),
    EventLeaf("menu_phone", TIER1_GUI, "gui_phone", "high",
              "KEY_T press",
              "Phone calls, messages, or contacts screen.",
              instruction_phrasing="Open the phone."),
    EventLeaf("menu_notifications", TIER1_GUI, "gui_notifications", "high",
              "KEY_Z press",
              "Notifications, shards, database, or quickhack details screen.",
              instruction_phrasing="Open notifications."),
    EventLeaf("menu_pause", TIER1_GUI, "gui_pause", "high",
              "ESCAPE press",
              "Pause, settings, or close-menu action.",
              instruction_phrasing="Open the pause menu."),
    EventLeaf("menu_weapon_wheel", TIER1_GUI, "gui_radial", "high",
              "LMENU held",
              "Weapon wheel or radial selection menu.",
              instruction_phrasing="Open the weapon wheel."),
    EventLeaf("menu_scanner_overlay", TIER1_GUI, "gui_radial", "medium",
              "TAB held with overlay UI",
              "Scanner or quickhack overlay shown as a menu-like radial interface.",
              instruction_phrasing="Use the scanner overlay."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no fingerprint key",
              "Other menu, vendor UI, stash, ripperdoc, or minigame screen.",
              instruction_phrasing="Navigate the menu."),

    # ---- meta ---------------------------------------------------------------
    EventLeaf("main_menu", TIER1_META, "meta_menu", "vision_only",
              "title / main menu UI",
              "Title screen or main menu before loading gameplay.",
              instruction_phrasing="Wait at the main menu."),
    EventLeaf("loading_screen", TIER1_META, "meta_flow", "vision_only",
              "no input, loading screen",
              "Loading screen or transition between scenes.",
              instruction_phrasing="Wait through the loading screen."),
    EventLeaf("cinematic_cutscene", TIER1_META, "meta_flow", "vision_only",
              "little input, cinematic framing",
              "Non-interactive cinematic or scripted cutscene.",
              instruction_phrasing="Watch the cinematic."),
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
# Backwards-compatible aliases for likely model / legacy labels
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    "idle":                       "idle_stationary",
    "look_around":                "idle_camera_only",
    "camera_only":                "idle_camera_only",
    "moving":                     "walking",
    "walk":                       "walking",
    "run":                        "sprinting",
    "running":                    "sprinting",
    "sprint":                     "sprinting",
    "jump":                       "jumping",
    "mantle":                     "parkour_climbing",
    "climb":                      "parkour_climbing",
    "crouch":                     "crouching",
    "crouch_walk":                "crouch_walking",
    "stealth":                    "crouch_walking",
    "interacting":                "interact",
    "interaction":                "interact",
    "pickup":                     "pickup_item",
    "looting":                    "pickup_item",
    "door_interaction":           "open_close_door",
    "talking":                    "talk_to_npc",
    "dialogue":                   "dialogue_choice",
    "weapon_swap":                "cycle_weapon",
    "weapon_cycle":               "cycle_weapon",
    "reload":                     "reload_weapon",
    "aiming":                     "aiming_down_sights",
    "aiming_ads":                 "aiming_down_sights",
    "shooting":                   "firing_hipfire",
    "firing_weapon":              "firing_hipfire",
    "melee":                      "melee_fast_attack",
    "grenade":                    "combat_gadget",
    "scanner":                    "scanner_hold",
    "scanning":                   "scanner_hold",
    "quickhack":                  "quick_attack",
    "quickhacking":               "quick_attack",
    "healing":                    "heal_with_consumable",
    "using_consumable":           "use_consumable",
    "driving":                    "driving_forward",
    "vehicle":                    "driving_forward",
    "enter_vehicle":              "enter_exit_vehicle",
    "exit_vehicle":               "enter_exit_vehicle",
    "braindance":                 "braindance_analysis_mode",
    "inventory_screen":           "menu_inventory",
    "map_screen":                 "menu_map",
    "journal_screen":             "menu_journal",
    "crafting_screen":            "menu_crafting",
    "phone_screen":               "menu_phone",
    "pause_settings_screen":      "menu_pause",
    "menu_ui":                    "menu_other",
    "cinematic":                  "cinematic_cutscene",
    "cutscene":                   "cinematic_cutscene",
    "loading":                    "loading_screen",
}


def normalize_label(label: str) -> str:
    """Map legacy / Gemini-suggested labels into the canonical leaf names."""
    if label in LEAF_BY_LABEL:
        return label
    return LEGACY_LABEL_ALIASES.get(label, "unknown")


def get_leaf(label: str) -> EventLeaf:
    return LEAF_BY_LABEL[normalize_label(label)]


# ---------------------------------------------------------------------------
# Hierarchical views
# ---------------------------------------------------------------------------


def vision_needed(label: str) -> bool:
    """True iff the Stage 2 (Gemini-vision) refinement should look at this segment."""
    if label == "unknown":
        return True
    leaf = LEAF_BY_LABEL.get(label)
    if leaf is None:
        return True
    return bool(leaf.vision_subtypes) or leaf.action_distinguishable in ("medium", "vision_only")


def subtype_choices(parent_label: str) -> Tuple[str, ...]:
    """Return the candidate sub-leaf labels Gemini may pick from for ``parent_label``.

    Always includes the parent label itself as an "I don't know, keep parent"
    option so the model is not forced to commit to a wrong sub-label.
    """
    leaf = LEAF_BY_LABEL.get(parent_label)
    if leaf is None or not leaf.vision_subtypes:
        return (parent_label,)
    seen: Set[str] = set()
    out: List[str] = []
    for c in leaf.vision_subtypes + (parent_label,):
        if c not in seen:
            seen.add(c)
            out.append(c)
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
    """Format the taxonomy as a compact bullet list for LLM prompts.

    If ``only_subtypes_of`` is given, only the candidate sub-leaves of that
    parent (plus the parent itself) are shown; otherwise the full hierarchy is
    rendered.
    """
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
# Atomic-label vocabulary used by the captioning prompt (Stage 3)
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
        out = sys.argv[2] if len(sys.argv) > 2 else "Cyberpunk_2077_taxonomy.json"
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
