"""Unified 3-tier event taxonomy for Apex Legends gameplay segmentation.

This module supersedes :mod:`Apex_Legends_taxonomy2` (action-distinguishable focus)
and :mod:`Apex_Legends_event_taxonomy3` (semantic / prompt focus). The two legacy
modules are kept for backwards compatibility with anything that still imports
them; new code should import from here.

The hierarchy is:

* **Tier 1 - Scene** (``in_match`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``traversal``, ``combat``, ``looting`` ...)
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

TIER1_IN_MATCH = "in_match"
TIER1_GUI = "gui"
TIER1_META = "meta"

TIER1_LABELS: Dict[str, str] = {
    TIER1_IN_MATCH: "In-Match Gameplay",
    TIER1_GUI:      "GUI / Menu Screens",
    TIER1_META:     "Lobby & Match Flow",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_MATCH: {
        "idle":             "Idle",
        "dropping_in":      "Dropping In",
        "traversal":        "Traversal / Movement",
        "looting":          "Looting / Interaction",
        "weapon_handling":  "Weapon Handling",
        "combat":           "Combat",
        "abilities":        "Legend Abilities",
        "support":          "Support / Survival",
        "communication":    "Communication",
        "down_spectate":    "Knocked / Spectate",
        "unknown":          "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_inventory":    "Inventory Screen",
        "gui_map":          "Map Screen",
        "gui_loadout":      "Loadout / Crafting",
        "gui_pause":        "Pause / Settings",
        "gui_other":        "Other Menu",
    },
    TIER1_META: {
        "meta_lobby":       "Lobby / Pre-match",
        "meta_match_flow":  "Match Flow / Cinematic",
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
    # ---- in_match.idle ------------------------------------------------------
    EventLeaf("idle_stationary", TIER1_IN_MATCH, "idle", "high",
              "no input >= idle_min_s, tiny mouse delta",
              "Player alive but standing still, no camera motion.",
              instruction_phrasing="Stand still and wait."),
    EventLeaf("idle_camera_only", TIER1_IN_MATCH, "idle", "high",
              "no keys held, non-trivial mouse delta",
              "Player looking around without moving.",
              instruction_phrasing="Hold position and look around."),

    # ---- in_match.dropping_in ----------------------------------------------
    EventLeaf("skydiving", TIER1_IN_MATCH, "dropping_in", "vision_only",
              "WASD + mouse, indistinguishable from walking",
              "Free-fall after jumping from the dropship.",
              instruction_phrasing="Dive toward the landing zone."),
    EventLeaf("gliding", TIER1_IN_MATCH, "dropping_in", "vision_only",
              "WASD + mouse, indistinguishable from walking",
              "Wingsuit phase before touching down.",
              instruction_phrasing="Glide in to land."),
    EventLeaf("dropship_transition", TIER1_IN_MATCH, "dropping_in", "vision_only",
              "no inputs, dropship cutscene",
              "Riding the dropship before jumping.",
              instruction_phrasing="Wait on the dropship for the jump."),

    # ---- in_match.traversal -------------------------------------------------
    EventLeaf("walking", TIER1_IN_MATCH, "traversal", "high",
              "WASD held, no LSHIFT / LCTRL",
              "Normal-speed ground traversal.",
              instruction_phrasing="Move forward at walking pace."),
    EventLeaf("sprinting", TIER1_IN_MATCH, "traversal", "high",
              "LSHIFT + WASD",
              "Sprinting on foot.",
              instruction_phrasing="Sprint forward."),
    EventLeaf("crouch_walking", TIER1_IN_MATCH, "traversal", "high",
              "LCTRL (or C) + WASD",
              "Crouch-moving for stealth / cover.",
              instruction_phrasing="Crouch-walk to stay quiet."),
    EventLeaf("sliding", TIER1_IN_MATCH, "traversal", "medium",
              "LSHIFT then LCTRL while moving forward",
              "Tactical slide chained from a sprint.",
              instruction_phrasing="Slide forward to gain speed."),
    EventLeaf("jumping_mantling", TIER1_IN_MATCH, "traversal", "high",
              "SPACE bursts (may overlay walking / sprinting)",
              "Jumping, vaulting, or mantling ledges.",
              instruction_phrasing="Jump and mantle the obstacle."),
    EventLeaf("using_zipline_balloon", TIER1_IN_MATCH, "traversal", "vision_only",
              "F or no keys while gaining altitude",
              "Riding a zipline or balloon to redeploy.",
              instruction_phrasing="Take the zipline up."),

    # ---- in_match.looting ---------------------------------------------------
    EventLeaf("interact", TIER1_IN_MATCH, "looting", "medium",
              "F or E single press / burst (~0.7s lock)",
              "Generic interact event; vision splits into looting / pickup / "
              "door / revive / respawn / zipline / crafting.",
              vision_subtypes=(
                  "looting_box", "picking_up_item", "opening_door",
                  "reviving_teammate", "respawning",
                  "using_zipline_balloon", "crafting_replicator",
              ),
              instruction_phrasing="Interact with the object."),
    EventLeaf("looting_box", TIER1_IN_MATCH, "looting", "vision_only",
              "F twice within ~2s, low movement",
              "Looting a death box / supply bin.",
              instruction_phrasing="Loot the death box for items."),
    EventLeaf("picking_up_item", TIER1_IN_MATCH, "looting", "vision_only",
              "single F tap, low movement",
              "Picking up a single floor item.",
              instruction_phrasing="Pick up the item on the ground."),
    EventLeaf("opening_door", TIER1_IN_MATCH, "looting", "vision_only",
              "single F tap on a door",
              "Opening or closing a door.",
              instruction_phrasing="Open the door."),
    EventLeaf("crafting_replicator", TIER1_IN_MATCH, "looting", "vision_only",
              "F + UI menu, not separable from looting",
              "Crafting at a replicator station.",
              instruction_phrasing="Craft items at the replicator."),

    # ---- in_match.weapon_handling ------------------------------------------
    EventLeaf("reloading", TIER1_IN_MATCH, "weapon_handling", "high",
              "R press, lock state for reload_lock_s",
              "Reloading the equipped weapon.",
              instruction_phrasing="Reload your weapon."),
    EventLeaf("equip_primary", TIER1_IN_MATCH, "weapon_handling", "high",
              "KEY_1 tap (~0.4s lock)",
              "Equip primary weapon (slot 1).",
              instruction_phrasing="Swap to your primary weapon."),
    EventLeaf("equip_secondary", TIER1_IN_MATCH, "weapon_handling", "high",
              "KEY_2 tap (~0.4s lock)",
              "Equip secondary weapon (slot 2).",
              instruction_phrasing="Swap to your secondary weapon."),
    EventLeaf("holster_weapon", TIER1_IN_MATCH, "weapon_handling", "high",
              "KEY_3 tap (~0.4s lock)",
              "Holster weapons / empty hands for faster sprint.",
              instruction_phrasing="Holster your weapons."),
    EventLeaf("weapon_inspect", TIER1_IN_MATCH, "weapon_handling", "high",
              "KEY_N tap (~1s lock)",
              "Cosmetic weapon inspect animation.",
              instruction_phrasing="Inspect your weapon."),
    EventLeaf("melee", TIER1_IN_MATCH, "weapon_handling", "high",
              "V press",
              "Melee strike.",
              instruction_phrasing="Punch the enemy with melee."),
    EventLeaf("throwing_grenade", TIER1_IN_MATCH, "weapon_handling", "medium",
              "G + LMB, or grenade slot then LMB",
              "Cooking and throwing an ordnance.",
              instruction_phrasing="Throw a grenade at the target."),

    # ---- in_match.combat ----------------------------------------------------
    # Grouped together so adjacent aim/fire segments can collapse into a single
    # combat-engagement block (see SegmenterConfig.merge_tier2_groups and the
    # COMBAT_GROUP merge rule in 1b_Apex_Legends_refine_with_gemini.py).
    EventLeaf("aiming_down_sights", TIER1_IN_MATCH, "combat", "high",
              "RMB held >= ads_min_s without sustained LMB",
              "Aiming through the weapon scope without firing.",
              instruction_phrasing="Aim down sights to line up the shot."),
    EventLeaf("firing_hipfire", TIER1_IN_MATCH, "combat", "high",
              "LMB held without RMB",
              "Hip-firing the equipped weapon.",
              instruction_phrasing="Hip-fire at the enemy."),
    EventLeaf("firing_ads", TIER1_IN_MATCH, "combat", "high",
              "LMB held while RMB held",
              "Firing while aiming down sights.",
              instruction_phrasing="ADS and fire at the enemy."),

    # ---- in_match.abilities -------------------------------------------------
    EventLeaf("tactical_ability", TIER1_IN_MATCH, "abilities", "high",
              "Q press (~1s lock)",
              "Casting the legend tactical.",
              instruction_phrasing="Use your tactical ability."),
    EventLeaf("ultimate_ability", TIER1_IN_MATCH, "abilities", "high",
              "Z press (~1s lock)",
              "Casting the legend ultimate.",
              instruction_phrasing="Activate your ultimate."),
    EventLeaf("passive_scan", TIER1_IN_MATCH, "abilities", "vision_only",
              "passive (Bloodhound / Seer), no dedicated key",
              "Triggering a passive scan ability.",
              instruction_phrasing="Use your passive ability."),

    # ---- in_match.support ---------------------------------------------------
    EventLeaf("using_health_item", TIER1_IN_MATCH, "support", "high",
              "KEY_4 press (~4s lock)",
              "Using selected health item; vision splits into heal vs shield.",
              vision_subtypes=("healing", "shielding"),
              instruction_phrasing="Use a health item."),
    EventLeaf("healing", TIER1_IN_MATCH, "support", "vision_only",
              "KEY_4 + the health bar fills on HUD",
              "Using a syringe or med kit.",
              instruction_phrasing="Heal up with a syringe."),
    EventLeaf("shielding", TIER1_IN_MATCH, "support", "vision_only",
              "KEY_4 + the shield bar fills on HUD",
              "Using a shield cell or battery.",
              instruction_phrasing="Pop a shield battery."),
    EventLeaf("reviving_teammate", TIER1_IN_MATCH, "support", "vision_only",
              "E or F held while looking at a downed teammate",
              "Reviving a knocked teammate.",
              instruction_phrasing="Revive your teammate."),
    EventLeaf("respawning", TIER1_IN_MATCH, "support", "vision_only",
              "F + respawn beacon prompt",
              "Using a respawn beacon to bring a dead teammate back.",
              instruction_phrasing="Respawn your teammate at the beacon."),

    # ---- in_match.communication --------------------------------------------
    EventLeaf("pinging", TIER1_IN_MATCH, "communication", "high",
              "MMB click (single or burst)",
              "Generic ping; vision splits into location / enemy / loot.",
              vision_subtypes=("ping_location", "ping_enemy", "ping_loot"),
              instruction_phrasing="Ping the target."),
    EventLeaf("ping_location", TIER1_IN_MATCH, "communication", "vision_only",
              "MMB click on terrain / map cursor",
              "Pinging a position to mark a destination.",
              instruction_phrasing="Ping the location to mark it."),
    EventLeaf("ping_enemy", TIER1_IN_MATCH, "communication", "vision_only",
              "MMB click on an enemy in view",
              "Calling out a spotted enemy.",
              instruction_phrasing="Ping the enemy you spotted."),
    EventLeaf("ping_loot", TIER1_IN_MATCH, "communication", "vision_only",
              "MMB click on a loot item",
              "Pinging a piece of loot for the squad.",
              instruction_phrasing="Ping the loot for your squad."),

    # ---- in_match.down_spectate --------------------------------------------
    EventLeaf("knocked_crawling", TIER1_IN_MATCH, "down_spectate", "vision_only",
              "slow WASD with no fire, vision-required",
              "Knocked down, crawling for cover.",
              instruction_phrasing="Crawl to safety while knocked."),
    EventLeaf("spectating", TIER1_IN_MATCH, "down_spectate", "vision_only",
              "mostly mouse, occasional space / WASD",
              "Spectating after death.",
              instruction_phrasing="Spectate your teammate."),

    # ---- in_match.unknown (catch-all parent) -------------------------------
    EventLeaf("unknown", TIER1_IN_MATCH, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  # the most common things hidden inside an unknown block
                  "skydiving", "gliding", "dropship_transition",
                  "using_zipline_balloon",
                  "looting_box", "picking_up_item", "opening_door",
                  "crafting_replicator",
                  "passive_scan",
                  "reviving_teammate", "respawning",
                  "ping_location", "ping_enemy", "ping_loot",
                  "knocked_crawling", "spectating",
                  "lobby", "legend_select", "firing_range",
                  "cinematic_loading", "menu_other",
                  "walking", "sprinting",
                  "unknown",
              ),
              instruction_phrasing="Continue with the current activity."),

    # ---- gui ----------------------------------------------------------------
    EventLeaf("menu_inventory", TIER1_GUI, "gui_inventory", "high",
              "TAB held / toggled",
              "Inventory screen open.",
              instruction_phrasing="Open the inventory and manage your loadout."),
    EventLeaf("menu_map", TIER1_GUI, "gui_map", "high",
              "M held / toggled",
              "Full map screen open.",
              instruction_phrasing="Open the map to plan your rotation."),
    EventLeaf("menu_pause", TIER1_GUI, "gui_pause", "high",
              "ESC pressed",
              "In-game pause / settings.",
              instruction_phrasing="Open the pause menu."),
    EventLeaf("menu_loadout", TIER1_GUI, "gui_loadout", "medium",
              "B pressed (loadout / crafting menu)",
              "Loadout or crafting menu.",
              instruction_phrasing="Open the loadout menu."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no fingerprint key",
              "Other menu / overlay; resolve with vision.",
              instruction_phrasing="Navigate the menu."),

    # ---- meta ---------------------------------------------------------------
    EventLeaf("lobby", TIER1_META, "meta_lobby", "vision_only",
              "WASD + mouse in lobby, not separable from walking",
              "Main lobby / squad screen.",
              instruction_phrasing="Wait in the lobby."),
    EventLeaf("legend_select", TIER1_META, "meta_lobby", "vision_only",
              "mouse clicks on legend grid",
              "Choosing a legend before drop.",
              instruction_phrasing="Select your legend."),
    EventLeaf("firing_range", TIER1_META, "meta_lobby", "vision_only",
              "full gameplay inputs inside firing range",
              "Firing-range practice.",
              instruction_phrasing="Practice in the firing range."),
    EventLeaf("cinematic_loading", TIER1_META, "meta_match_flow", "vision_only",
              "no input, static / black frames",
              "Cinematic intro or loading screen.",
              instruction_phrasing="Wait through the loading screen."),
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
# Backwards-compatible aliases for old labels (taxonomy2 / taxonomy3)
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    # Apex_Legends_taxonomy2 -> new
    # cycle_weapon has been removed from the taxonomy (the player rebound the
    # mouse wheel away from weapon cycling), so the legacy "weapon_swap*"
    # aliases now fall back to `unknown` and let the vision pass decide.
    "weapon_swap":                 "unknown",
    "engaging_enemy":              "firing_hipfire",
    "reviving":                    "reviving_teammate",
    # Apex_Legends_event_taxonomy3 -> new
    "walking_running_sprinting":   "walking",
    "crouching_sliding":           "crouch_walking",
    "jumping_climbing":            "jumping_mantling",
    "zipline_travel":              "using_zipline_balloon",
    "gliding_skydiving":           "gliding",
    "rotating_to_ring":            "walking",
    "scouting_scanning":           "idle_camera_only",
    "opening_supply_bin":          "looting_box",
    "ground_item_looting":         "picking_up_item",
    "deathbox_looting":            "looting_box",
    "inventory_sorting":           "menu_inventory",
    "door_object_interaction":     "opening_door",
    "aiming_ads":                  "aiming_down_sights",
    "firing_weapon":               "firing_hipfire",
    "weapon_swapping":             "unknown",
    "grenade_throwing":            "throwing_grenade",
    "taking_cover_strafing":       "walking",
    "pushing_chasing_enemy":       "sprinting",
    "retreating_resetting":        "walking",
    "knocked_or_downed":           "knocked_crawling",
    "shield_recharging":           "shielding",
    "health_healing":              "healing",
    "pinging_communication":       "pinging",
    "inventory_screen":            "menu_inventory",
    "map_screen":                  "menu_map",
    "pause_settings_screen":       "menu_pause",
    "loot_box_screen":             "menu_inventory",
    "ping_wheel_screen":           "pinging",
    "lobby_menu":                  "lobby",
    "spectating_post_match":       "spectating",
    "watching_cinematic":          "cinematic_loading",
    "dropping_in":                 "skydiving",
    "menu_ui":                     "menu_inventory",
    "menu_inventory":              "menu_inventory",
    "scanning":                    "idle_camera_only",
    "swapping_weapon":             "unknown",
    "using_ability":               "tactical_ability",
    "using_ultimate":              "ultimate_ability",
    "downed":                      "knocked_crawling",
    "knocked":                     "knocked_crawling",
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
        out = sys.argv[2] if len(sys.argv) > 2 else "Apex_Legends_taxonomy.json"
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
