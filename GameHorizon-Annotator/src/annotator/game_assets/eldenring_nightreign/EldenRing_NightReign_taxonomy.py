"""Unified 3-tier event taxonomy for ELDEN RING NIGHTREIGN gameplay segmentation.

This is the Nightreign twin of :mod:`Apex_Legends_taxonomy`: same ``EventLeaf``
dataclass and same public API, so the downstream segmentation / refinement /
instruction scripts can import it identically (just swap the module name).

The hierarchy is:

* **Tier 1 - Scene** (``in_field`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``traversal``, ``combat``, ``interaction`` ...)
* **Tier 3 - Atomic event leaf** (``surge_sprinting``, ``dodge_roll`` ...)

Each leaf carries:

* ``action_distinguishable`` - ``"high"`` (rule classifier outputs it),
  ``"medium"`` (rule classifier outputs but a vision pass should refine it),
  ``"vision_only"`` (only the vision pass can produce it).
* ``vision_subtypes`` - if non-empty, this leaf is a *parent bucket* that the
  Gemini refinement stage may split into one of the listed sub-leaves.
* ``instruction_phrasing`` - a short imperative fallback used when Stage 3
  decides not to call the captioning model (very short segments, etc.).

Keybind grounding
-----------------
The ``key_signature`` strings use the exact ``KEY_*`` / ``RI_MOUSE_*`` names that
appear in the per-frame annotation stream (see
``EldenRing_NightReign_keybinds.json``). This capture uses a stock Nightreign PC
layout with two things worth remembering:

* **Surge Sprint is ``LMENU``** (the virtual-key name for Left Alt), *not*
  ``LSHIFT`` -- ``LSHIFT`` is Backstep / Dodge Roll.
* **Lock-on is rebound to mouse side-button 5** (``RI_MOUSE_BUTTON_5``), freeing
  the middle mouse button for pins. This was verified against the recording with
  gemini-2.5-flash (~0.9 confidence: the lock-on reticle toggles on/off and the
  camera snaps to the target across each button-5 tap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

TIER1_IN_FIELD = "in_field"
TIER1_GUI = "gui"
TIER1_META = "meta"

TIER1_LABELS: Dict[str, str] = {
    TIER1_IN_FIELD: "In-Expedition Gameplay",
    TIER1_GUI:      "GUI / Menu Screens",
    TIER1_META:     "Hub & Expedition Flow",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_FIELD: {
        "idle":             "Idle",
        "traversal":        "Traversal / Movement",
        "combat":           "Combat",
        "abilities":        "Nightfarer Abilities",
        "items":            "Items / Flask",
        "weapon_handling":  "Weapon / Armament Handling",
        "interaction":      "Interaction / Looting",
        "survival":         "Support / Survival",
        "communication":    "Communication / Targeting",
        "down_spectate":    "Downed / Spectate",
        "unknown":          "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_map":          "Map Screen",
        "gui_equipment":    "Equipment / Status",
        "gui_level_up":     "Level Up (Site of Grace)",
        "gui_merchant":     "Merchant / Shop",
        "gui_other":        "Other Menu",
    },
    TIER1_META: {
        "meta_hub":         "Roundtable Hold",
        "meta_select":      "Character / Expedition Select",
        "meta_flow":        "Match Flow / Cinematic",
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
    # ---- in_field.idle ------------------------------------------------------
    EventLeaf("idle_stationary", TIER1_IN_FIELD, "idle", "high",
              "no input >= idle_min_s, tiny mouse delta",
              "Player alive but standing still, no camera motion.",
              instruction_phrasing="Stand still and wait."),

    # ---- in_field.traversal -------------------------------------------------
    EventLeaf("walking", TIER1_IN_FIELD, "traversal", "high",
              "WASD held, no LMENU",
              "Normal-speed running on foot (Nightreign's default jog).",
              instruction_phrasing="Run forward."),
    EventLeaf("surge_sprinting", TIER1_IN_FIELD, "traversal", "high",
              "LMENU (Left Alt) + WASD",
              "Surge sprint -- the faster stamina-free dash used to cross Limveld.",
              instruction_phrasing="Sprint forward."),
    EventLeaf("dodge_roll", TIER1_IN_FIELD, "traversal", "high",
              "LSHIFT tap (roll while moving, backstep while still)",
              "Dodge roll / backstep / quickstep to evade.",
              instruction_phrasing="Dodge roll to evade."),
    EventLeaf("jumping", TIER1_IN_FIELD, "traversal", "high",
              "SPACE (may overlay walking / sprinting)",
              "Jumping or hopping over obstacles and gaps.",
              instruction_phrasing="Jump across the gap."),
    EventLeaf("climbing", TIER1_IN_FIELD, "traversal", "vision_only",
              "no dedicated key; ladder auto-climb",
              "Climbing a ladder.",
              instruction_phrasing="Climb the ladder."),
    EventLeaf("spectral_hawk_flight", TIER1_IN_FIELD, "traversal", "vision_only",
              "F on a Spectral Hawk then auto-glide, few inputs",
              "Fast-travel gliding on a Spectral Hawk across the map.",
              instruction_phrasing="Ride the Spectral Hawk across the map."),
    EventLeaf("spiritspring_launch", TIER1_IN_FIELD, "traversal", "vision_only",
              "no dedicated key; geyser launch",
              "Launched upward by a Spiritspring geyser.",
              instruction_phrasing="Use the Spiritspring to launch up."),

    # ---- in_field.combat ----------------------------------------------------
    # Grouped so adjacent attack/guard segments can collapse into one
    # combat-engagement block (see merge_tier2_groups in the segmenter).
    EventLeaf("light_attack", TIER1_IN_FIELD, "combat", "high",
              "LMB without LCONTROL",
              "Standard light attack with the equipped armament.",
              instruction_phrasing="Attack the enemy."),
    EventLeaf("heavy_attack", TIER1_IN_FIELD, "combat", "high",
              "LCONTROL + LMB (hold to charge)",
              "Strong / heavy or charged attack.",
              instruction_phrasing="Land a heavy attack."),
    EventLeaf("guarding", TIER1_IN_FIELD, "combat", "high",
              "RMB held",
              "Blocking with a shield or left-hand armament.",
              instruction_phrasing="Raise your guard to block."),
    EventLeaf("skill_ash_of_war", TIER1_IN_FIELD, "combat", "high",
              "KEY_Q (tap to cast, hold to charge)",
              "Weapon skill / Ash of War.",
              instruction_phrasing="Use your weapon skill."),

    # ---- in_field.abilities -------------------------------------------------
    EventLeaf("character_skill", TIER1_IN_FIELD, "abilities", "high",
              "KEY_E (~cooldown gated)",
              "The Nightfarer's unique Character Skill.",
              instruction_phrasing="Use your character skill."),
    EventLeaf("ultimate_art", TIER1_IN_FIELD, "abilities", "high",
              "KEY_R (~cooldown gated)",
              "The Nightfarer's Ultimate Art.",
              instruction_phrasing="Unleash your Ultimate Art."),

    # ---- in_field.items -----------------------------------------------------
    EventLeaf("healing_flask", TIER1_IN_FIELD, "items", "high",
              "KEY_1",
              "Drinking the flask to restore HP.",
              instruction_phrasing="Heal with your flask."),
    EventLeaf("use_item", TIER1_IN_FIELD, "items", "medium",
              "KEY_G (uses the selected quick item)",
              "Using the currently selected quick item; vision splits by item type.",
              vision_subtypes=("throwing_item", "buff_item", "using_consumable"),
              instruction_phrasing="Use your selected item."),
    EventLeaf("throwing_item", TIER1_IN_FIELD, "items", "vision_only",
              "KEY_G + a thrown pot / knife on screen",
              "Throwing an offensive item (pot, knife, etc.).",
              instruction_phrasing="Throw an item at the enemy."),
    EventLeaf("buff_item", TIER1_IN_FIELD, "items", "vision_only",
              "KEY_G + a buff effect / grease animation",
              "Applying a temporary buff item (grease, boluses, etc.).",
              instruction_phrasing="Apply a buff item."),
    EventLeaf("using_consumable", TIER1_IN_FIELD, "items", "vision_only",
              "KEY_G + a misc consumable animation",
              "Using a miscellaneous consumable item.",
              instruction_phrasing="Use a consumable item."),

    # ---- in_field.weapon_handling ------------------------------------------
    EventLeaf("switch_armament", TIER1_IN_FIELD, "weapon_handling", "medium",
              "RI_MOUSE_WHEEL (up = right hand, down = left hand)",
              "Cycling the equipped armament; vision / wheel-direction refines "
              "which hand and whether it is a two-hand toggle.",
              vision_subtypes=(
                  "switch_right_armament", "switch_left_armament", "two_hand_toggle",
              ),
              instruction_phrasing="Switch your armament."),
    EventLeaf("switch_right_armament", TIER1_IN_FIELD, "weapon_handling", "vision_only",
              "wheel up",
              "Switching the right-hand armament.",
              instruction_phrasing="Switch your right-hand weapon."),
    EventLeaf("switch_left_armament", TIER1_IN_FIELD, "weapon_handling", "vision_only",
              "wheel down",
              "Switching the left-hand armament.",
              instruction_phrasing="Switch your left-hand weapon."),
    EventLeaf("two_hand_toggle", TIER1_IN_FIELD, "weapon_handling", "vision_only",
              "LCONTROL + wheel",
              "Toggling two-handing of the current armament.",
              instruction_phrasing="Two-hand your weapon."),
    EventLeaf("switch_item", TIER1_IN_FIELD, "weapon_handling", "high",
              "KEY_2",
              "Cycling the active quick-item slot.",
              instruction_phrasing="Cycle your quick item."),

    # ---- in_field.interaction ----------------------------------------------
    EventLeaf("interact", TIER1_IN_FIELD, "interaction", "medium",
              "KEY_F single press (~0.7s lock)",
              "Generic Event Action; vision splits into chest / pickup / door / "
              "lever / grace / NPC / hawk / spiritspring / revive.",
              vision_subtypes=(
                  "opening_chest", "picking_up_item", "opening_door", "pulling_lever",
                  "resting_at_grace", "talking_to_npc", "activating_spectral_hawk",
                  "using_spiritspring", "reviving_teammate",
              ),
              instruction_phrasing="Interact with it."),
    EventLeaf("opening_chest", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on a chest",
              "Opening a treasure chest.",
              instruction_phrasing="Open the chest."),
    EventLeaf("picking_up_item", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on a ground item",
              "Picking up a dropped item / rune pickup.",
              instruction_phrasing="Pick up the item."),
    EventLeaf("opening_door", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on a door / gate",
              "Opening a door or gate.",
              instruction_phrasing="Open the door."),
    EventLeaf("pulling_lever", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on a lever / mechanism",
              "Operating a lever or mechanism.",
              instruction_phrasing="Pull the lever."),
    EventLeaf("resting_at_grace", TIER1_IN_FIELD, "interaction", "vision_only",
              "F at a Site of Grace",
              "Resting at a Site of Grace.",
              instruction_phrasing="Rest at the Site of Grace."),
    EventLeaf("talking_to_npc", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on an NPC / merchant",
              "Talking to an NPC or merchant.",
              instruction_phrasing="Talk to the NPC."),
    EventLeaf("activating_spectral_hawk", TIER1_IN_FIELD, "interaction", "vision_only",
              "F on a Spectral Hawk marker",
              "Activating a Spectral Hawk fast-travel point.",
              instruction_phrasing="Take the Spectral Hawk."),
    EventLeaf("using_spiritspring", TIER1_IN_FIELD, "interaction", "vision_only",
              "F / walking into a Spiritspring",
              "Using a Spiritspring geyser.",
              instruction_phrasing="Use the Spiritspring."),

    # ---- in_field.survival --------------------------------------------------
    EventLeaf("reviving_teammate", TIER1_IN_FIELD, "survival", "vision_only",
              "attacking (LMB) a downed ally to fill the revive gauge",
              "Reviving a downed teammate by attacking them (mechanically an "
              "attack, so indistinguishable from combat by input alone).",
              instruction_phrasing="Revive your teammate."),

    # ---- in_field.communication --------------------------------------------
    EventLeaf("lock_on_toggle", TIER1_IN_FIELD, "communication", "high",
              "RI_MOUSE_BUTTON_5 tap (rebound lock-on / reset camera)",
              "Toggling camera lock-on onto a target (or resetting the camera).",
              instruction_phrasing="Lock on to the target."),
    EventLeaf("ping_location", TIER1_IN_FIELD, "communication", "vision_only",
              "MMB / KEY_X (rare)",
              "Placing a map pin / marking a location for the squad.",
              instruction_phrasing="Ping the location."),

    # ---- in_field.down_spectate --------------------------------------------
    EventLeaf("knocked_crawling", TIER1_IN_FIELD, "down_spectate", "vision_only",
              "slow WASD, no attacks; downed state",
              "Downed and crawling before death or revive.",
              instruction_phrasing="Crawl to safety while downed."),
    EventLeaf("spectating", TIER1_IN_FIELD, "down_spectate", "vision_only",
              "mostly camera, occasional inputs",
              "Spectating a teammate after dying.",
              instruction_phrasing="Spectate your teammate."),

    # ---- in_field.unknown (catch-all parent) -------------------------------
    EventLeaf("unknown", TIER1_IN_FIELD, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  # the most common things hidden inside an unknown block
                  "climbing", "spectral_hawk_flight", "spiritspring_launch",
                  "opening_chest", "picking_up_item", "opening_door", "pulling_lever",
                  "resting_at_grace", "talking_to_npc", "activating_spectral_hawk",
                  "using_spiritspring", "reviving_teammate",
                  "ping_location", "knocked_crawling", "spectating",
                  "roundtable_hub", "character_select", "expedition_select",
                  "relic_loadout", "matchmaking_lobby", "cinematic_loading",
                  "results_screen",
                  "walking", "surge_sprinting",
                  "unknown",
              ),
              instruction_phrasing="Continue with the current activity."),

    # ---- gui ----------------------------------------------------------------
    EventLeaf("menu_map", TIER1_GUI, "gui_map", "high",
              "KEY_M held / toggled",
              "Full map screen open.",
              instruction_phrasing="Open the map to plan your route."),
    EventLeaf("menu_equipment", TIER1_GUI, "gui_equipment", "high",
              "ESCAPE pressed",
              "Equipment / status / inventory menu.",
              instruction_phrasing="Open the equipment menu."),
    EventLeaf("menu_level_up", TIER1_GUI, "gui_level_up", "vision_only",
              "menu at a Site of Grace, spending runes",
              "Leveling up the Nightfarer at a Site of Grace.",
              instruction_phrasing="Level up your Nightfarer."),
    EventLeaf("menu_merchant", TIER1_GUI, "gui_merchant", "vision_only",
              "shop / merchant screen",
              "Buying or selling at a merchant.",
              instruction_phrasing="Browse the merchant's wares."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no fingerprint key",
              "Other menu / overlay; resolve with vision.",
              instruction_phrasing="Navigate the menu."),

    # ---- meta ---------------------------------------------------------------
    EventLeaf("roundtable_hub", TIER1_META, "meta_hub", "vision_only",
              "movement / UI in the hub, not separable from field",
              "The Roundtable Hold hub between expeditions.",
              instruction_phrasing="Prepare in the Roundtable Hold."),
    EventLeaf("character_select", TIER1_META, "meta_select", "vision_only",
              "clicks on the Nightfarer grid",
              "Choosing a Nightfarer before an expedition.",
              instruction_phrasing="Select your Nightfarer."),
    EventLeaf("expedition_select", TIER1_META, "meta_select", "vision_only",
              "clicks on the expedition / Nightlord map",
              "Choosing the expedition and target Nightlord.",
              instruction_phrasing="Choose your expedition."),
    EventLeaf("relic_loadout", TIER1_META, "meta_select", "vision_only",
              "relic slotting menu",
              "Equipping relics before an expedition.",
              instruction_phrasing="Set up your relics."),
    EventLeaf("matchmaking_lobby", TIER1_META, "meta_flow", "vision_only",
              "waiting / matchmaking screen",
              "Matchmaking or squad lobby before drop-in.",
              instruction_phrasing="Wait for matchmaking."),
    EventLeaf("cinematic_loading", TIER1_META, "meta_flow", "vision_only",
              "no input, static / black frames",
              "Cinematic intro or loading screen.",
              instruction_phrasing="Wait through the loading screen."),
    EventLeaf("results_screen", TIER1_META, "meta_flow", "vision_only",
              "post-expedition results / death / victory",
              "Expedition results, death, or victory screen.",
              instruction_phrasing="Review the expedition results."),
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
# Backwards-compatible / Gemini-suggested label aliases
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    # movement
    "running":            "walking",
    "jogging":            "walking",
    "sprinting":          "surge_sprinting",
    "surge_sprint":       "surge_sprinting",
    "dodging":            "dodge_roll",
    "rolling":            "dodge_roll",
    "roll":               "dodge_roll",
    "backstep":           "dodge_roll",
    "jump":               "jumping",
    # combat
    "attacking":          "light_attack",
    "attack":             "light_attack",
    "light_attacking":    "light_attack",
    "heavy_attacking":    "heavy_attack",
    "strong_attack":      "heavy_attack",
    "charged_attack":     "heavy_attack",
    "blocking":           "guarding",
    "guard":              "guarding",
    "block":              "guarding",
    "weapon_skill":       "skill_ash_of_war",
    "ash_of_war":         "skill_ash_of_war",
    "skill":              "skill_ash_of_war",
    # abilities
    "ability":            "character_skill",
    "using_skill":        "character_skill",
    "ultimate":           "ultimate_art",
    "using_ultimate":     "ultimate_art",
    # items
    "healing":            "healing_flask",
    "heal":               "healing_flask",
    "drinking_flask":     "healing_flask",
    "flask":              "healing_flask",
    "using_item":         "use_item",
    "throwing":           "throwing_item",
    "throw_pot":          "throwing_item",
    "buff":               "buff_item",
    "applying_grease":    "buff_item",
    # weapon handling
    "swap_weapon":        "switch_armament",
    "switch_weapon":      "switch_armament",
    "weapon_swap":        "switch_armament",
    "two_handing":        "two_hand_toggle",
    # interaction
    "looting":            "interact",
    "interacting":        "interact",
    "open_chest":         "opening_chest",
    "looting_chest":      "opening_chest",
    "pickup_item":        "picking_up_item",
    "pick_up_item":       "picking_up_item",
    "open_door":          "opening_door",
    "resting":            "resting_at_grace",
    "site_of_grace":      "resting_at_grace",
    "talking":            "talking_to_npc",
    "npc":                "talking_to_npc",
    "merchant":           "menu_merchant",
    "spectral_hawk":      "activating_spectral_hawk",
    "spiritspring":       "using_spiritspring",
    # survival
    "reviving":           "reviving_teammate",
    "revive":             "reviving_teammate",
    # targeting / comms
    "lock_on":            "lock_on_toggle",
    "locking_on":         "lock_on_toggle",
    "targeting":          "lock_on_toggle",
    "ping":               "ping_location",
    "pinging":            "ping_location",
    # downed / spectate
    "downed":             "knocked_crawling",
    "knocked":            "knocked_crawling",
    "crawling":           "knocked_crawling",
    "spectate":           "spectating",
    # gui
    "map":                "menu_map",
    "map_screen":         "menu_map",
    "inventory":          "menu_equipment",
    "equipment":          "menu_equipment",
    "status_screen":      "menu_equipment",
    "menu":               "menu_equipment",
    "level_up":           "menu_level_up",
    "leveling_up":        "menu_level_up",
    "shop":               "menu_merchant",
    # meta
    "hub":                "roundtable_hub",
    "roundtable":         "roundtable_hub",
    "lobby":              "matchmaking_lobby",
    "matchmaking":        "matchmaking_lobby",
    "loading":            "cinematic_loading",
    "cinematic":          "cinematic_loading",
    "watching_cinematic": "cinematic_loading",
    "results":            "results_screen",
    "death_screen":       "results_screen",
    "victory_screen":     "results_screen",
    "select_character":   "character_select",
    "select_expedition":  "expedition_select",
    "relics":             "relic_loadout",
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
        out = sys.argv[2] if len(sys.argv) > 2 else "EldenRing_NightReign_taxonomy.json"
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
