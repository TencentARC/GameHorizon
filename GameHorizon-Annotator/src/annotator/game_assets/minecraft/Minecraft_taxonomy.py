"""Unified 3-tier event taxonomy for Minecraft gameplay segmentation.

The hierarchy mirrors :mod:`Apex_Legends_scripts.Apex_Legends_taxonomy` so
Minecraft processing code can share the same helper API:

* **Tier 1 - Scene** (``in_world`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``movement``, ``block_interaction`` ...)
* **Tier 3 - Atomic event leaf** (``sprinting``, ``mining_block`` ...)

Each leaf carries:

* ``action_distinguishable`` - ``"high"`` (rule classifier outputs it),
  ``"medium"`` (rule classifier outputs but a vision pass should refine it),
  ``"vision_only"`` (only the vision pass can produce it).
* ``vision_subtypes`` - if non-empty, this leaf is a parent bucket that the
  vision refinement stage may split into one of the listed sub-leaves.
* ``instruction_phrasing`` - a short imperative fallback used when captioning
  is skipped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

TIER1_IN_WORLD = "in_world"
TIER1_GUI = "gui"
TIER1_META = "meta"

TIER1_LABELS: Dict[str, str] = {
    TIER1_IN_WORLD: "In-World Gameplay",
    TIER1_GUI:      "GUI / Overlay Screens",
    TIER1_META:     "World & Session Flow",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_WORLD: {
        "idle":                 "Idle / Looking",
        "movement":             "Movement / Traversal",
        "block_interaction":    "Mining / Building / Interacting",
        "combat":               "Combat / Defense",
        "item_use":             "Item Use",
        "inventory_hotbar":     "Hotbar / Carried Items",
        "communication":        "Communication",
        "camera_debug":         "Camera / Debug Controls",
        "unknown":              "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_inventory":        "Inventory / Container Screen",
        "gui_crafting":         "Crafting / Smelting / Trading",
        "gui_chat":             "Chat / Command Screen",
        "gui_pause":            "Close GUI / Pause",
        "gui_debug":            "Debug Overlay",
        "gui_other":            "Other Menu",
    },
    TIER1_META: {
        "meta_title":           "Title / Server Menu",
        "meta_world_flow":      "Loading / World Transition",
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
              "Player is in the world and standing still.",
              instruction_phrasing="Stand still and wait."),
    EventLeaf("idle_camera_only", TIER1_IN_WORLD, "idle", "high",
              "no keys held, non-trivial mouse delta",
              "Player is looking around without moving or using items.",
              instruction_phrasing="Hold position and look around."),

    # ---- in_world.movement --------------------------------------------------
    EventLeaf("walking", TIER1_IN_WORLD, "movement", "high",
              "KEY_W / KEY_A / KEY_S / KEY_D held, no LCONTROL / LSHIFT",
              "Normal-speed ground movement.",
              instruction_phrasing="Walk through the world."),
    EventLeaf("sprinting", TIER1_IN_WORLD, "movement", "high",
              "LCONTROL + KEY_W / WASD",
              "Sprinting on foot.",
              instruction_phrasing="Sprint forward."),
    EventLeaf("sneaking", TIER1_IN_WORLD, "movement", "high",
              "LSHIFT held, optionally with WASD",
              "Sneaking or crouch-walking to avoid falling or stay hidden.",
              instruction_phrasing="Sneak carefully."),
    EventLeaf("jumping", TIER1_IN_WORLD, "movement", "high",
              "SPACE press / burst",
              "Jumping while standing or moving.",
              instruction_phrasing="Jump over the obstacle."),
    EventLeaf("flying_up", TIER1_IN_WORLD, "movement", "medium",
              "SPACE held while in creative / spectator flight",
              "Ascending during creative or spectator flight.",
              instruction_phrasing="Fly upward."),
    EventLeaf("flying_down", TIER1_IN_WORLD, "movement", "medium",
              "LSHIFT held while in creative / spectator flight",
              "Descending during creative or spectator flight.",
              instruction_phrasing="Fly downward."),
    EventLeaf("swimming_or_boating", TIER1_IN_WORLD, "movement", "vision_only",
              "WASD + SPACE / mouse while in water or boat",
              "Swimming, rowing, or moving through water.",
              instruction_phrasing="Move through the water."),
    EventLeaf("climbing", TIER1_IN_WORLD, "movement", "vision_only",
              "WASD / SPACE while on ladder, vine, or climbable block",
              "Climbing vertically on a ladder, vine, or similar block.",
              instruction_phrasing="Climb upward."),

    # ---- in_world.block_interaction ----------------------------------------
    EventLeaf("primary_action", TIER1_IN_WORLD, "block_interaction", "medium",
              "RI_MOUSE_LEFT_BUTTON_DOWN",
              "Left-click action; vision splits mining from attacking.",
              vision_subtypes=("mining_block", "attacking_entity"),
              instruction_phrasing="Use the primary action."),
    EventLeaf("mining_block", TIER1_IN_WORLD, "block_interaction", "vision_only",
              "RI_MOUSE_LEFT_BUTTON_DOWN held on a block",
              "Breaking or mining a block.",
              instruction_phrasing="Mine the block."),
    EventLeaf("using_or_placing", TIER1_IN_WORLD, "block_interaction", "medium",
              "RI_MOUSE_RIGHT_BUTTON_DOWN",
              "Right-click action; vision splits block placement, interaction, and item use.",
              vision_subtypes=(
                  "placing_block", "interacting_block", "opening_container",
                  "eating_drinking", "using_tool_or_item", "blocking_with_shield",
              ),
              instruction_phrasing="Use or place the selected item."),
    EventLeaf("placing_block", TIER1_IN_WORLD, "block_interaction", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN while holding a placeable block",
              "Placing a block into the world.",
              instruction_phrasing="Place the selected block."),
    EventLeaf("interacting_block", TIER1_IN_WORLD, "block_interaction", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN on a door, lever, button, bed, or entity",
              "Interacting with a world object or entity.",
              instruction_phrasing="Interact with the object."),
    EventLeaf("opening_container", TIER1_IN_WORLD, "block_interaction", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN on chest, barrel, furnace, villager, etc.",
              "Opening a container or interaction screen from the world.",
              instruction_phrasing="Open the container."),
    EventLeaf("pick_block", TIER1_IN_WORLD, "block_interaction", "high",
              "RI_MOUSE_MIDDLE_BUTTON_DOWN",
              "Picking the targeted block into the hotbar in creative mode.",
              instruction_phrasing="Pick the targeted block."),

    # ---- in_world.combat ----------------------------------------------------
    EventLeaf("attacking_entity", TIER1_IN_WORLD, "combat", "vision_only",
              "RI_MOUSE_LEFT_BUTTON_DOWN on mob / player",
              "Melee attacking a mob or another player.",
              instruction_phrasing="Attack the enemy."),
    EventLeaf("charging_ranged_weapon", TIER1_IN_WORLD, "combat", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN held with bow / crossbow / trident",
              "Charging or aiming a ranged weapon.",
              instruction_phrasing="Draw the ranged weapon."),
    EventLeaf("blocking_with_shield", TIER1_IN_WORLD, "combat", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN held with shield",
              "Raising a shield to block incoming attacks.",
              instruction_phrasing="Raise the shield to block."),

    # ---- in_world.item_use --------------------------------------------------
    EventLeaf("using_tool_or_item", TIER1_IN_WORLD, "item_use", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN with tool or utility item",
              "Using an equipped item such as a bucket, flint and steel, or potion.",
              instruction_phrasing="Use the selected item."),
    EventLeaf("eating_drinking", TIER1_IN_WORLD, "item_use", "vision_only",
              "RI_MOUSE_RIGHT_BUTTON_DOWN held with food or potion",
              "Eating food or drinking a potion.",
              instruction_phrasing="Eat or drink the selected item."),
    EventLeaf("dropping_item", TIER1_IN_WORLD, "item_use", "high",
              "KEY_Q press",
              "Dropping the selected item from the hotbar or inventory.",
              instruction_phrasing="Drop the selected item."),

    # ---- in_world.inventory_hotbar -----------------------------------------
    EventLeaf("selecting_hotbar_slot", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_1 through KEY_7",
              "Selecting a specific hotbar slot.",
              vision_subtypes=(
                  "select_hotbar_slot_1", "select_hotbar_slot_2",
                  "select_hotbar_slot_3", "select_hotbar_slot_4",
                  "select_hotbar_slot_5", "select_hotbar_slot_6",
                  "select_hotbar_slot_7",
              ),
              instruction_phrasing="Select a hotbar slot."),
    EventLeaf("select_hotbar_slot_1", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_1 press",
              "Selecting hotbar slot 1.",
              instruction_phrasing="Select hotbar slot 1."),
    EventLeaf("select_hotbar_slot_2", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_2 press",
              "Selecting hotbar slot 2.",
              instruction_phrasing="Select hotbar slot 2."),
    EventLeaf("select_hotbar_slot_3", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_3 press",
              "Selecting hotbar slot 3.",
              instruction_phrasing="Select hotbar slot 3."),
    EventLeaf("select_hotbar_slot_4", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_4 press",
              "Selecting hotbar slot 4.",
              instruction_phrasing="Select hotbar slot 4."),
    EventLeaf("select_hotbar_slot_5", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_5 press",
              "Selecting hotbar slot 5.",
              instruction_phrasing="Select hotbar slot 5."),
    EventLeaf("select_hotbar_slot_6", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_6 press",
              "Selecting hotbar slot 6.",
              instruction_phrasing="Select hotbar slot 6."),
    EventLeaf("select_hotbar_slot_7", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "KEY_7 press",
              "Selecting hotbar slot 7.",
              instruction_phrasing="Select hotbar slot 7."),
    EventLeaf("cycling_hotbar_slot", TIER1_IN_WORLD, "inventory_hotbar", "high",
              "RI_MOUSE_WHEEL",
              "Cycling through hotbar slots with the mouse wheel.",
              instruction_phrasing="Scroll to another hotbar slot."),

    # ---- in_world.communication --------------------------------------------
    EventLeaf("opening_chat", TIER1_IN_WORLD, "communication", "high",
              "KEY_T press",
              "Opening the chat box.",
              instruction_phrasing="Open chat."),
    EventLeaf("showing_player_list", TIER1_IN_WORLD, "communication", "high",
              "TAB held",
              "Showing the player list or autocompleting a command.",
              instruction_phrasing="Show the player list."),

    # ---- in_world.camera_debug ---------------------------------------------
    EventLeaf("toggling_debug_overlay", TIER1_IN_WORLD, "camera_debug", "high",
              "F3 press",
              "Toggling the debug overlay.",
              instruction_phrasing="Toggle the debug overlay."),
    EventLeaf("changing_perspective", TIER1_IN_WORLD, "camera_debug", "high",
              "F5 press",
              "Switching between first-person and third-person perspectives.",
              instruction_phrasing="Change camera perspective."),

    # ---- in_world.unknown ---------------------------------------------------
    EventLeaf("unknown", TIER1_IN_WORLD, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  "walking", "sprinting", "sneaking", "jumping",
                  "swimming_or_boating", "climbing",
                  "mining_block", "placing_block", "interacting_block",
                  "opening_container", "attacking_entity", "charging_ranged_weapon",
                  "blocking_with_shield", "using_tool_or_item", "eating_drinking",
                  "inventory_screen", "crafting_screen", "chat_screen",
                  "close_gui_or_pause", "debug_overlay", "title_menu", "loading_world",
                  "unknown",
              ),
              instruction_phrasing="Continue with the current activity."),

    # ---- gui ---------------------------------------------------------------
    EventLeaf("inventory_screen", TIER1_GUI, "gui_inventory", "high",
              "KEY_E toggled",
              "Player inventory screen is open.",
              vision_subtypes=("inventory_management", "crafting_screen"),
              instruction_phrasing="Open the inventory and manage items."),
    EventLeaf("inventory_management", TIER1_GUI, "gui_inventory", "vision_only",
              "inventory GUI with mouse clicks / drags",
              "Moving, sorting, equipping, or dropping items in an inventory.",
              instruction_phrasing="Manage items in the inventory."),
    EventLeaf("crafting_screen", TIER1_GUI, "gui_crafting", "vision_only",
              "inventory / crafting table UI visible",
              "Crafting items in the 2x2 or 3x3 crafting grid.",
              instruction_phrasing="Craft the needed item."),
    EventLeaf("smelting_or_brewing_screen", TIER1_GUI, "gui_crafting", "vision_only",
              "furnace / smoker / blast furnace / brewing stand UI visible",
              "Managing smelting, cooking, or brewing interfaces.",
              instruction_phrasing="Use the station to process items."),
    EventLeaf("trading_or_anvil_screen", TIER1_GUI, "gui_crafting", "vision_only",
              "villager trade, anvil, grindstone, loom, or similar UI visible",
              "Using a specialized workstation or trading screen.",
              instruction_phrasing="Use the workstation interface."),
    EventLeaf("chat_screen", TIER1_GUI, "gui_chat", "high",
              "KEY_T opened chat; text entry visible",
              "Chat or command input is open.",
              instruction_phrasing="Type in chat."),
    EventLeaf("close_gui_or_pause", TIER1_GUI, "gui_pause", "high",
              "ESCAPE press",
              "Closing the current GUI, cancelling an overlay, or opening / closing the pause menu.",
              instruction_phrasing="Press Escape to close the current GUI or pause menu."),
    EventLeaf("debug_overlay", TIER1_GUI, "gui_debug", "high",
              "F3 toggled on",
              "Debug overlay is visible on top of gameplay.",
              instruction_phrasing="View the debug overlay."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no unique key fingerprint",
              "Other Minecraft GUI or overlay.",
              instruction_phrasing="Navigate the menu."),

    # ---- meta --------------------------------------------------------------
    EventLeaf("title_menu", TIER1_META, "meta_title", "vision_only",
              "title / multiplayer / singleplayer menu visible",
              "Main menu, world list, server list, or title screen.",
              instruction_phrasing="Navigate the Minecraft menu."),
    EventLeaf("loading_world", TIER1_META, "meta_world_flow", "vision_only",
              "loading terrain / joining world screen, little or no input",
              "Loading, joining, respawning, or transitioning between worlds.",
              instruction_phrasing="Wait for the world to load."),
    EventLeaf("death_or_respawn_screen", TIER1_META, "meta_world_flow", "vision_only",
              "death screen with respawn / title buttons",
              "Death screen or respawn choice after the player dies.",
              instruction_phrasing="Respawn after dying."),
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
# Backwards-compatible aliases / common Gemini label variants
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    "standing":                   "idle_stationary",
    "looking_around":             "idle_camera_only",
    "moving":                     "walking",
    "running":                    "sprinting",
    "crouching":                  "sneaking",
    "crouch_walking":             "sneaking",
    "jump":                       "jumping",
    "swimming":                   "swimming_or_boating",
    "boating":                    "swimming_or_boating",
    "breaking_block":             "mining_block",
    "digging":                    "mining_block",
    "building":                   "placing_block",
    "place_block":                "placing_block",
    "interact":                   "interacting_block",
    "opening_chest":              "opening_container",
    "container_screen":           "inventory_screen",
    "fighting":                   "attacking_entity",
    "melee_attack":               "attacking_entity",
    "shooting_bow":               "charging_ranged_weapon",
    "using_shield":               "blocking_with_shield",
    "using_item":                 "using_tool_or_item",
    "eating":                     "eating_drinking",
    "drinking":                   "eating_drinking",
    "drop_item":                  "dropping_item",
    "hotbar_select":              "selecting_hotbar_slot",
    "hotbar_scroll":              "cycling_hotbar_slot",
    "inventory":                  "inventory_screen",
    "crafting":                   "crafting_screen",
    "chat":                       "chat_screen",
    "close_gui":                  "close_gui_or_pause",
    "pause":                      "close_gui_or_pause",
    "pause_menu":                 "close_gui_or_pause",
    "settings_menu":              "close_gui_or_pause",
    "debug":                      "debug_overlay",
    "third_person_toggle":        "changing_perspective",
    "menu_ui":                    "menu_other",
    "main_menu":                  "title_menu",
    "world_loading":              "loading_world",
    "respawning":                 "death_or_respawn_screen",
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
    """True iff the vision refinement stage should look at this segment."""
    if label == "unknown":
        return True
    leaf = LEAF_BY_LABEL.get(label)
    if leaf is None:
        return True
    return bool(leaf.vision_subtypes) or leaf.action_distinguishable in ("medium", "vision_only")


def subtype_choices(parent_label: str) -> Tuple[str, ...]:
    """Return candidate sub-leaf labels for ``parent_label``.

    Always includes the parent label itself as an "I don't know, keep parent"
    option so the model is not forced to commit to a wrong sub-label.
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
# Atomic-label vocabulary used by the captioning prompt
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
        out = sys.argv[2] if len(sys.argv) > 2 else "Minecraft_taxonomy.json"
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
