"""3-tier event taxonomy for ELDEN RING (base game) gameplay segmentation.

This is the base-game twin of :mod:`EldenRing_NightReign_taxonomy` /
:mod:`RedDeadRedemption_taxonomy`: same ``EventLeaf`` dataclass and same public
API, so the downstream segmentation / refinement / instruction scripts can
import it identically (just swap the module name).

Where Nightreign is a 3-player co-op roguelite, the **base game is a solo,
third-person open-world action-RPG** set in the Lands Between. That changes the
taxonomy in a few important ways versus the Nightreign twin:

* **Torrent (the Spectral Steed)** is a first-class traversal pillar -- mounted
  riding / galloping / double-jump / Spiritspring launches.
* **Magic** matters: sorceries (staff) and incantations (sacred seal) are cast
  with the *attack* button, so they are input-indistinguishable from a melee
  swing and can only be split by vision.
* **Two flasks + Wondrous Physick + Spirit Ashes**: the single "use item" key is
  heavily overloaded (HP flask / FP flask / throwables / greases / physick /
  summon bell), so it is a ``medium`` parent bucket a vision pass refines.
* **No teammates**: there is no revive / spectate / downed-ally state.

The hierarchy is:

* **Tier 1 - Scene** (``in_world`` / ``gui`` / ``meta``)
* **Tier 2 - Gameplay content** (``traversal``, ``combat``, ``interaction`` ...)
* **Tier 3 - Atomic event leaf** (``sprinting``, ``dodge_roll`` ...)

Each leaf carries:

* ``action_distinguishable`` - ``"high"`` (rule classifier outputs it directly
  from keyboard / mouse), ``"medium"`` (rule classifier outputs a parent bucket
  that a vision pass should refine), ``"vision_only"`` (only the vision pass can
  produce it).
* ``vision_subtypes`` - if non-empty, this leaf is a *parent bucket* that the
  Gemini refinement stage may split into one of the listed sub-leaves.
* ``instruction_phrasing`` - a short imperative fallback used when a later stage
  decides not to call the captioning model (very short segments, etc.).

Keybind grounding
-----------------
The ``key_signature`` strings use the exact ``KEY_*`` / ``RI_MOUSE_*`` names that
appear in the per-frame annotation stream (see ``EldenRing_keybinds.json``).
They were grounded in the base Elden Ring default layout plus the observed input
vocabulary of ``video_annotations_EldenRing/`` (scene 0001 of
``85882349-EldenRing-paotu-084748``). This capture uses a "modernised" custom
layout that -- like the Nightreign layout -- splits jump / sprint / dodge onto
separate keys:

* **Sprint / Dash is ``TAB``** (held; 428 presses vs 25 releases in scene 0001 =
  a long auto-repeating hold, the signature of a locomotion modifier held while
  running the map), *not* the vanilla Space-hold.
* **Jump is ``SPACE``** (pure taps) and **Dodge Roll / Backstep is ``LSHIFT``**
  (pure taps).
* **Attack is ``LMB``**, **Guard is ``RMB``**, **Ash of War / weapon skill is
  ``KEY_Q``**, **Event Action / Interact is ``KEY_E``**, **Use Item is
  ``KEY_R``**, **Crouch is ``KEY_C``**.
* **Armament / spell / item switching** uses the d-pad arrows (``RIGHT`` = right
  hand, ``LEFT`` = left hand, ``UP`` = sorcery/incantation, ``DOWN`` = quick
  item) and the mouse wheel.

Two honest caveats (documented rather than guessed away):

* **Camera lock-on** has no fingerprint in scene 0001's input stream (no middle
  mouse button, no side button), so ``lock_on_toggle`` is ``vision_only`` here.
* **Two-handing** is tentatively mapped to the low-frequency ``KEY_F`` (13
  presses); it is a ``medium`` bucket the vision pass confirms.

Anything context-overloaded (``KEY_E`` interact, ``KEY_R`` use-item, the wheel
switch, ``LMB`` attack-vs-cast) is emitted as a ``high``/``medium`` parent bucket
whose real leaf the vision pass resolves -- the same trick the RDR2 twin uses.
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
    TIER1_IN_WORLD: "In-World Gameplay (The Lands Between)",
    TIER1_GUI:      "GUI / Menu Screens",
    TIER1_META:     "Cinematic / Loading / Outcome",
}


# ---------------------------------------------------------------------------
# Tier 2 (grouped, ordered)
# ---------------------------------------------------------------------------

TIER2_LABELS: Dict[str, Dict[str, str]] = {
    TIER1_IN_WORLD: {
        "idle":             "Idle",
        "traversal":        "Traversal / Movement",
        "mount":            "Torrent / Spectral Steed",
        "combat":           "Combat",
        "magic":            "Sorcery / Incantation",
        "weapon_handling":  "Armament / Spell Handling",
        "items":            "Items / Flasks",
        "interaction":      "Interaction / Looting",
        "communication":    "Camera / Communication",
        "unknown":          "Unknown / Catch-all",
    },
    TIER1_GUI: {
        "gui_map":          "Map Screen",
        "gui_menu":         "Equipment / Inventory / Status",
        "gui_level_up":     "Level Up (Site of Grace)",
        "gui_grace":        "Site of Grace Menu",
        "gui_crafting":     "Crafting Menu",
        "gui_merchant":     "Merchant / Shop",
        "gui_other":        "Other Menu",
    },
    TIER1_META: {
        "meta_cinematic":   "Cutscene / Cinematic",
        "meta_loading":     "Loading Screen",
        "meta_outcome":     "Death / Boss Defeated",
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
              "no movement / combat keys or mouse buttons held for >= idle_min_s, tiny mouse delta",
              "Tarnished alive but standing still (player may pan the camera), no "
              "locomotion, combat, or interaction input.",
              instruction_phrasing="Stand still and look around."),

    # ---- in_world.traversal -------------------------------------------------
    # WASD movement cannot tell on-foot from Torrent riding, so the locomotion
    # leaves carry a ``torrent_riding`` vision_subtype the refine pass can promote.
    EventLeaf("walking", TIER1_IN_WORLD, "traversal", "high",
              "WASD held, no TAB (sprint) and no KEY_C (crouch)",
              "Normal-speed running on foot across the Lands Between.",
              vision_subtypes=("torrent_riding",),
              instruction_phrasing="Run forward."),
    EventLeaf("sprinting", TIER1_IN_WORLD, "traversal", "high",
              "TAB held + WASD (TAB is this layout's Sprint / Dash key)",
              "Sprinting / dashing on foot to cross ground quickly.",
              vision_subtypes=("torrent_riding",),
              instruction_phrasing="Sprint forward."),
    EventLeaf("crouch_sneaking", TIER1_IN_WORLD, "traversal", "high",
              "KEY_C toggled (crouch) + WASD",
              "Crouch-walking / sneaking to stay quiet or set up a stealth attack.",
              instruction_phrasing="Crouch and sneak forward."),
    EventLeaf("dodge_roll", TIER1_IN_WORLD, "traversal", "high",
              "LSHIFT tap (roll while moving, backstep while still, quickstep)",
              "Dodge roll / backstep / quickstep to evade.",
              instruction_phrasing="Dodge roll to evade."),
    EventLeaf("jumping", TIER1_IN_WORLD, "traversal", "high",
              "SPACE (may overlay walking / sprinting)",
              "Jumping or hopping over obstacles and gaps.",
              vision_subtypes=("jump_attack", "torrent_jump"),
              instruction_phrasing="Jump across the gap."),
    EventLeaf("climbing", TIER1_IN_WORLD, "traversal", "vision_only",
              "no dedicated key; ladder auto-climb",
              "Climbing a ladder.",
              instruction_phrasing="Climb the ladder."),
    EventLeaf("fast_travel", TIER1_IN_WORLD, "traversal", "vision_only",
              "selected from the map / a Site of Grace, then a loading screen",
              "Fast-travelling to a discovered Site of Grace.",
              instruction_phrasing="Fast travel to the Site of Grace."),

    # ---- in_world.mount (Torrent) ------------------------------------------
    EventLeaf("summon_torrent", TIER1_IN_WORLD, "mount", "vision_only",
              "Spectral Steed Whistle; the bind is not present in scene 0001's "
              "input vocabulary, so only vision can confirm the summon / dismiss",
              "Summoning or dismissing Torrent, the Spectral Steed.",
              instruction_phrasing="Summon Torrent."),
    EventLeaf("torrent_riding", TIER1_IN_WORLD, "mount", "vision_only",
              "WASD (+ TAB gallop) while mounted -- identical keys to on foot",
              "Riding / galloping on Torrent across the world.",
              instruction_phrasing="Ride Torrent across the field."),
    EventLeaf("torrent_jump", TIER1_IN_WORLD, "mount", "vision_only",
              "SPACE while mounted (Torrent can double-jump)",
              "Jumping / double-jumping while mounted on Torrent.",
              instruction_phrasing="Jump with Torrent."),
    EventLeaf("mounted_attack", TIER1_IN_WORLD, "mount", "vision_only",
              "LMB / RMB while mounted on Torrent",
              "Attacking from horseback (left- or right-side swing).",
              instruction_phrasing="Attack from Torrent."),
    EventLeaf("spiritspring_launch", TIER1_IN_WORLD, "mount", "vision_only",
              "no dedicated key; geyser launch (usually on Torrent)",
              "Launched upward by a Spiritspring geyser.",
              instruction_phrasing="Use the Spiritspring to launch up."),
    EventLeaf("dismount", TIER1_IN_WORLD, "mount", "vision_only",
              "leaving Torrent (whistle / auto-dismount)",
              "Dismounting from Torrent.",
              instruction_phrasing="Dismount from Torrent."),

    # ---- in_world.combat ----------------------------------------------------
    # Grouped so adjacent attack / guard segments can collapse into one
    # combat-engagement block. ``light_attack`` is the LMB parent: with a staff
    # or sacred seal equipped the same button casts a spell, so it carries
    # vision_subtypes the refine pass can promote.
    EventLeaf("light_attack", TIER1_IN_WORLD, "combat", "high",
              "LMB (right-hand armament, no modifier)",
              "Standard light attack (R1) with the equipped armament; vision may "
              "promote it to a heavy / charged / jump attack or a spell cast.",
              vision_subtypes=(
                  "heavy_attack", "charged_attack", "jump_attack",
                  "cast_sorcery", "cast_incantation", "firing_bow",
              ),
              instruction_phrasing="Attack the enemy."),
    EventLeaf("heavy_attack", TIER1_IN_WORLD, "combat", "vision_only",
              "LMB hold / Ctrl+LMB (strong attack, R2); no heavy modifier key was "
              "observed in scene 0001, so vision confirms it",
              "Strong / heavy attack (R2).",
              instruction_phrasing="Land a heavy attack."),
    EventLeaf("charged_attack", TIER1_IN_WORLD, "combat", "vision_only",
              "held strong attack until fully charged",
              "Fully charged strong attack.",
              instruction_phrasing="Charge and release a heavy attack."),
    EventLeaf("jump_attack", TIER1_IN_WORLD, "combat", "vision_only",
              "SPACE then LMB in the air",
              "Jumping attack that comes down on the target.",
              instruction_phrasing="Perform a jumping attack."),
    EventLeaf("guarding", TIER1_IN_WORLD, "combat", "high",
              "RMB held (block with shield / left-hand armament)",
              "Blocking with a shield or left-hand armament (RMB may also aim a "
              "bow / catalyst -- vision refines).",
              vision_subtypes=("guard_counter", "aiming_ranged"),
              instruction_phrasing="Raise your guard to block."),
    EventLeaf("guard_counter", TIER1_IN_WORLD, "combat", "vision_only",
              "RMB (guard) immediately followed by LMB",
              "Guard counter -- an attack launched right after blocking.",
              instruction_phrasing="Land a guard counter."),
    EventLeaf("riposte_critical", TIER1_IN_WORLD, "combat", "vision_only",
              "critical-hit prompt after a parry or from stealth",
              "Critical hit -- riposte after a parry, or a backstab / stealth "
              "critical.",
              instruction_phrasing="Land a critical hit."),
    EventLeaf("skill_ash_of_war", TIER1_IN_WORLD, "combat", "high",
              "KEY_Q (Ash of War / weapon skill; tap to cast, hold to charge)",
              "Weapon skill / Ash of War.",
              instruction_phrasing="Use your weapon skill."),

    # ---- in_world.magic -----------------------------------------------------
    # Casting uses the attack button with a catalyst equipped, so these are
    # input-indistinguishable from a melee swing -> vision_only. They are also
    # listed as vision_subtypes of ``light_attack``.
    EventLeaf("cast_sorcery", TIER1_IN_WORLD, "magic", "vision_only",
              "LMB with a glintstone staff equipped -- input-identical to a melee attack",
              "Casting a sorcery (staff spell), consuming FP.",
              instruction_phrasing="Cast a sorcery at the enemy."),
    EventLeaf("cast_incantation", TIER1_IN_WORLD, "magic", "vision_only",
              "LMB with a sacred seal equipped -- input-identical to a melee attack",
              "Casting an incantation (faith spell), consuming FP.",
              instruction_phrasing="Cast an incantation."),
    EventLeaf("firing_bow", TIER1_IN_WORLD, "magic", "vision_only",
              "LMB / RMB with a bow or crossbow equipped",
              "Firing a bow or crossbow at range.",
              instruction_phrasing="Fire an arrow at the target."),
    EventLeaf("aiming_ranged", TIER1_IN_WORLD, "magic", "vision_only",
              "RMB held to aim a bow / catalyst (first-person aim)",
              "Aiming a bow or ranged spell without releasing.",
              instruction_phrasing="Aim at the target."),

    # ---- in_world.weapon_handling ------------------------------------------
    EventLeaf("switch_right_armament", TIER1_IN_WORLD, "weapon_handling", "high",
              "RIGHT arrow (cycle right-hand armament)",
              "Switching the right-hand armament.",
              instruction_phrasing="Switch your right-hand weapon."),
    EventLeaf("switch_left_armament", TIER1_IN_WORLD, "weapon_handling", "high",
              "LEFT arrow (cycle left-hand armament)",
              "Switching the left-hand armament / shield.",
              instruction_phrasing="Switch your left-hand weapon."),
    EventLeaf("switch_sorcery", TIER1_IN_WORLD, "weapon_handling", "high",
              "UP arrow (cycle the equipped sorcery / incantation)",
              "Switching the selected sorcery or incantation.",
              instruction_phrasing="Switch your spell."),
    EventLeaf("switch_quick_item", TIER1_IN_WORLD, "weapon_handling", "high",
              "DOWN arrow (cycle the active quick item)",
              "Cycling the active quick-item slot.",
              instruction_phrasing="Cycle your quick item."),
    EventLeaf("switch_armament", TIER1_IN_WORLD, "weapon_handling", "medium",
              "RI_MOUSE_WHEEL (this layout also cycles armaments on the wheel)",
              "Cycling the equipped armament on the wheel; vision / wheel "
              "direction refines which hand and whether it is a two-hand toggle.",
              vision_subtypes=(
                  "switch_right_armament", "switch_left_armament", "two_hand_toggle",
              ),
              instruction_phrasing="Switch your armament."),
    EventLeaf("two_hand_toggle", TIER1_IN_WORLD, "weapon_handling", "medium",
              "KEY_F (toggle two-handing; bind inferred, low confidence)",
              "Toggling two-handing of the current armament.",
              instruction_phrasing="Two-hand your weapon."),

    # ---- in_world.items -----------------------------------------------------
    EventLeaf("use_item", TIER1_IN_WORLD, "items", "medium",
              "KEY_R (use the currently selected quick item)",
              "Using the selected quick item; the slot is heavily overloaded so "
              "vision splits it into flask / throwable / buff / physick / summon.",
              vision_subtypes=(
                  "healing_flask", "fp_flask", "throwing_item", "buff_item",
                  "wondrous_physick", "summon_spirit_ashes", "using_consumable",
              ),
              instruction_phrasing="Use your selected item."),
    EventLeaf("healing_flask", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + Flask of Crimson Tears drink animation",
              "Drinking the Flask of Crimson Tears to restore HP.",
              instruction_phrasing="Heal with your Crimson flask."),
    EventLeaf("fp_flask", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + Flask of Cerulean Tears drink animation",
              "Drinking the Flask of Cerulean Tears to restore FP.",
              instruction_phrasing="Restore FP with your Cerulean flask."),
    EventLeaf("throwing_item", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + a thrown pot / knife / bolt on screen",
              "Throwing an offensive item (pot, knife, kukri, etc.).",
              instruction_phrasing="Throw an item at the enemy."),
    EventLeaf("buff_item", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + a grease / boluses / buff animation",
              "Applying a temporary buff item (grease, boluses, perfume, etc.).",
              instruction_phrasing="Apply a buff item."),
    EventLeaf("wondrous_physick", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + the Flask of Wondrous Physick",
              "Drinking the Flask of Wondrous Physick for its mixed effect.",
              instruction_phrasing="Drink the Wondrous Physick."),
    EventLeaf("summon_spirit_ashes", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R with the Spirit Calling Bell (near a rebirth monument)",
              "Summoning Spirit Ashes to fight alongside you.",
              instruction_phrasing="Summon your Spirit Ashes."),
    EventLeaf("using_consumable", TIER1_IN_WORLD, "items", "vision_only",
              "KEY_R + a misc consumable animation",
              "Using a miscellaneous consumable item.",
              instruction_phrasing="Use a consumable item."),

    # ---- in_world.interaction ----------------------------------------------
    EventLeaf("interact", TIER1_IN_WORLD, "interaction", "medium",
              "KEY_E single press (Event Action)",
              "Generic Event Action; vision splits into chest / pickup / door / "
              "lever / grace / NPC / message / examine / mechanism.",
              vision_subtypes=(
                  "opening_chest", "picking_up_item", "opening_door",
                  "pulling_lever", "resting_at_grace", "talking_to_npc",
                  "reading_message", "examining", "operating_mechanism",
              ),
              instruction_phrasing="Interact with it."),
    EventLeaf("opening_chest", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a chest",
              "Opening a treasure chest.",
              instruction_phrasing="Open the chest."),
    EventLeaf("picking_up_item", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a ground item / glowing pickup",
              "Picking up a dropped item, remembrance, or rune pickup.",
              instruction_phrasing="Pick up the item."),
    EventLeaf("opening_door", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a door / fog gate / gate",
              "Opening a door, gate, or entering a fog wall.",
              instruction_phrasing="Open the door."),
    EventLeaf("pulling_lever", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a lever / pull-chain",
              "Operating a lever or pull-chain.",
              instruction_phrasing="Pull the lever."),
    EventLeaf("resting_at_grace", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E at a Site of Grace",
              "Resting at a Site of Grace (opens the grace menu, restores flasks).",
              instruction_phrasing="Rest at the Site of Grace."),
    EventLeaf("talking_to_npc", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on an NPC / merchant",
              "Talking to an NPC or merchant.",
              instruction_phrasing="Talk to the NPC."),
    EventLeaf("reading_message", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a co-op message on the ground",
              "Reading or rating a player-left message.",
              instruction_phrasing="Read the message."),
    EventLeaf("examining", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E to examine an object / read an in-world description",
              "Examining an object or reading an in-world notice.",
              instruction_phrasing="Examine the object."),
    EventLeaf("operating_mechanism", TIER1_IN_WORLD, "interaction", "vision_only",
              "KEY_E on a lift / rise-and-fall platform / mechanism",
              "Operating a lift or environmental mechanism.",
              instruction_phrasing="Operate the mechanism."),

    # ---- in_world.communication --------------------------------------------
    EventLeaf("lock_on_toggle", TIER1_IN_WORLD, "communication", "vision_only",
              "camera lock-on (default MMB / a rebind) -- absent from scene 0001's "
              "input vocabulary, so only vision can confirm it here",
              "Toggling camera lock-on onto a target.",
              instruction_phrasing="Lock on to the target."),
    EventLeaf("reset_camera", TIER1_IN_WORLD, "communication", "vision_only",
              "recenter camera behind the player (same control as lock-on)",
              "Recentering the camera behind the player.",
              instruction_phrasing="Reset the camera."),
    EventLeaf("gesture", TIER1_IN_WORLD, "communication", "vision_only",
              "gesture / emote selected from a menu",
              "Performing a gesture / emote.",
              instruction_phrasing="Perform a gesture."),
    EventLeaf("place_summon_sign", TIER1_IN_WORLD, "communication", "vision_only",
              "using a summon sign / finger remedy for co-op",
              "Placing or answering a co-op summon sign.",
              instruction_phrasing="Place a summon sign."),

    # ---- in_world.unknown (catch-all parent) -------------------------------
    EventLeaf("unknown", TIER1_IN_WORLD, "unknown", "medium",
              "fallback when no rule matches",
              "Catch-all bucket; resolved by the vision pass.",
              vision_subtypes=(
                  # the most common things hidden inside an unknown block
                  "climbing", "fast_travel",
                  "summon_torrent", "torrent_riding", "torrent_jump",
                  "mounted_attack", "spiritspring_launch", "dismount",
                  "heavy_attack", "charged_attack", "jump_attack",
                  "guard_counter", "riposte_critical",
                  "cast_sorcery", "cast_incantation", "firing_bow", "aiming_ranged",
                  "opening_chest", "picking_up_item", "opening_door",
                  "pulling_lever", "resting_at_grace", "talking_to_npc",
                  "reading_message", "examining", "operating_mechanism",
                  "lock_on_toggle", "reset_camera", "gesture", "place_summon_sign",
                  "menu_level_up", "menu_grace", "menu_crafting", "menu_merchant",
                  "cinematic", "loading_screen", "death_screen", "boss_defeated",
                  "walking", "sprinting",
                  "unknown",
              ),
              instruction_phrasing="Continue with the current activity."),

    # ---- gui ----------------------------------------------------------------
    EventLeaf("menu_map", TIER1_GUI, "gui_map", "high",
              "KEY_M held / toggled",
              "Full map screen open (place markers, plan a route, fast travel).",
              instruction_phrasing="Open the map to plan your route."),
    EventLeaf("menu_equipment", TIER1_GUI, "gui_menu", "high",
              "ESCAPE pressed (opens the game menu)",
              "Game menu: Equipment / Inventory / Status / Equip Spells; vision "
              "splits which tab is open.",
              vision_subtypes=(
                  "menu_status", "menu_level_up", "menu_inventory",
                  "menu_equip_spells", "menu_crafting",
              ),
              instruction_phrasing="Open the equipment menu."),
    EventLeaf("menu_status", TIER1_GUI, "gui_menu", "vision_only",
              "Status tab of the game menu",
              "Viewing the Status / attributes screen.",
              instruction_phrasing="Check your status screen."),
    EventLeaf("menu_inventory", TIER1_GUI, "gui_menu", "vision_only",
              "Inventory tab of the game menu",
              "Browsing the inventory.",
              instruction_phrasing="Open your inventory."),
    EventLeaf("menu_equip_spells", TIER1_GUI, "gui_menu", "vision_only",
              "Memorize Spells screen",
              "Memorising / equipping sorceries and incantations.",
              instruction_phrasing="Equip your spells."),
    EventLeaf("menu_level_up", TIER1_GUI, "gui_level_up", "vision_only",
              "Level Up screen at a Site of Grace, spending runes",
              "Levelling up the Tarnished at a Site of Grace.",
              instruction_phrasing="Level up your character."),
    EventLeaf("menu_grace", TIER1_GUI, "gui_grace", "vision_only",
              "Site of Grace options menu",
              "The Site of Grace options list (level up / travel / memorize / ...).",
              instruction_phrasing="Choose an option at the Site of Grace."),
    EventLeaf("menu_crafting", TIER1_GUI, "gui_crafting", "vision_only",
              "Item-crafting screen",
              "Crafting items at a Site of Grace.",
              instruction_phrasing="Craft an item."),
    EventLeaf("menu_merchant", TIER1_GUI, "gui_merchant", "vision_only",
              "shop / merchant screen",
              "Buying or selling at a merchant.",
              instruction_phrasing="Browse the merchant's wares."),
    EventLeaf("menu_other", TIER1_GUI, "gui_other", "vision_only",
              "menu screen with no fingerprint key",
              "Other menu / overlay; resolve with vision.",
              instruction_phrasing="Navigate the menu."),

    # ---- meta ---------------------------------------------------------------
    EventLeaf("cinematic", TIER1_META, "meta_cinematic", "vision_only",
              "no / few inputs, cinematic frames",
              "Story cutscene, boss intro, or scripted cinematic.",
              instruction_phrasing="Watch the cutscene."),
    EventLeaf("loading_screen", TIER1_META, "meta_loading", "vision_only",
              "no input, static / black / loading frames",
              "Loading screen (fast travel, death respawn, area transition).",
              instruction_phrasing="Wait through the loading screen."),
    EventLeaf("death_screen", TIER1_META, "meta_outcome", "vision_only",
              "no input, 'YOU DIED' overlay",
              "The 'YOU DIED' death screen before respawn.",
              instruction_phrasing="Wait through the death screen."),
    EventLeaf("boss_defeated", TIER1_META, "meta_outcome", "vision_only",
              "no input, 'ENEMY FELLED' / 'GREAT ENEMY FELLED' overlay",
              "Boss / great-enemy defeated banner.",
              instruction_phrasing="Watch the enemy-felled banner."),
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
# Backwards-compatible / model-suggested label aliases -> canonical leaf names
# ---------------------------------------------------------------------------

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    # idle
    "idle":                   "idle_stationary",
    "standing":               "idle_stationary",
    "standing_still":         "idle_stationary",
    # movement
    "running":                "walking",
    "jogging":                "walking",
    "walk":                   "walking",
    "run":                    "walking",
    "sprint":                 "sprinting",
    "surge_sprinting":        "sprinting",
    "surge_sprint":           "sprinting",
    "dashing":                "sprinting",
    "dash":                   "sprinting",
    "crouching":              "crouch_sneaking",
    "sneaking":               "crouch_sneaking",
    "stealth":                "crouch_sneaking",
    "crouch":                 "crouch_sneaking",
    "dodging":                "dodge_roll",
    "rolling":                "dodge_roll",
    "roll":                   "dodge_roll",
    "backstep":               "dodge_roll",
    "quickstep":              "dodge_roll",
    "jump":                   "jumping",
    "hopping":                "jumping",
    "ladder":                 "climbing",
    "climb":                  "climbing",
    "warp":                   "fast_travel",
    "fast_traveling":         "fast_travel",
    "fast_travelling":        "fast_travel",
    "teleport":               "fast_travel",
    # mount / Torrent
    "torrent":                "torrent_riding",
    "riding":                 "torrent_riding",
    "riding_torrent":         "torrent_riding",
    "on_horseback":           "torrent_riding",
    "horseback_riding":       "torrent_riding",
    "galloping":              "torrent_riding",
    "mounted":                "torrent_riding",
    "mounted_movement":       "torrent_riding",
    "summon_steed":           "summon_torrent",
    "whistle":                "summon_torrent",
    "call_torrent":           "summon_torrent",
    "spectral_steed":         "summon_torrent",
    "spiritspring":           "spiritspring_launch",
    "geyser_launch":          "spiritspring_launch",
    "mount":                  "summon_torrent",
    "dismounting":            "dismount",
    # combat
    "attacking":              "light_attack",
    "attack":                 "light_attack",
    "light_attacking":        "light_attack",
    "r1":                     "light_attack",
    "heavy_attacking":        "heavy_attack",
    "strong_attack":          "heavy_attack",
    "r2":                     "heavy_attack",
    "charged":                "charged_attack",
    "charging_attack":        "charged_attack",
    "jumping_attack":         "jump_attack",
    "plunge_attack":          "jump_attack",
    "blocking":               "guarding",
    "guard":                  "guarding",
    "block":                  "guarding",
    "shielding":              "guarding",
    "parry":                  "riposte_critical",
    "parrying":               "riposte_critical",
    "riposte":                "riposte_critical",
    "critical_hit":           "riposte_critical",
    "backstab":               "riposte_critical",
    "visceral":               "riposte_critical",
    "guardcounter":           "guard_counter",
    "weapon_skill":           "skill_ash_of_war",
    "ash_of_war":             "skill_ash_of_war",
    "skill":                  "skill_ash_of_war",
    "using_skill":            "skill_ash_of_war",
    # magic
    "casting":                "cast_sorcery",
    "cast_spell":             "cast_sorcery",
    "sorcery":                "cast_sorcery",
    "casting_sorcery":        "cast_sorcery",
    "magic":                  "cast_sorcery",
    "spell":                  "cast_sorcery",
    "incantation":            "cast_incantation",
    "casting_incantation":    "cast_incantation",
    "prayer":                 "cast_incantation",
    "bow":                    "firing_bow",
    "shooting_bow":           "firing_bow",
    "firing_arrow":           "firing_bow",
    "archery":                "firing_bow",
    "aiming":                 "aiming_ranged",
    "aim":                    "aiming_ranged",
    "ads":                    "aiming_ranged",
    # weapon / spell handling
    "switch_weapon":          "switch_armament",
    "swap_weapon":            "switch_armament",
    "weapon_swap":            "switch_armament",
    "switch_right_hand":      "switch_right_armament",
    "switch_left_hand":       "switch_left_armament",
    "switch_spell":           "switch_sorcery",
    "cycle_spell":            "switch_sorcery",
    "switch_item":            "switch_quick_item",
    "cycle_item":             "switch_quick_item",
    "two_handing":            "two_hand_toggle",
    "two_hand":               "two_hand_toggle",
    # items / flasks
    "using_item":             "use_item",
    "use_flask":              "healing_flask",
    "healing":                "healing_flask",
    "heal":                   "healing_flask",
    "drinking_flask":         "healing_flask",
    "flask":                  "healing_flask",
    "crimson_flask":          "healing_flask",
    "hp_flask":               "healing_flask",
    "cerulean_flask":         "fp_flask",
    "restore_fp":             "fp_flask",
    "throwing":               "throwing_item",
    "throw":                  "throwing_item",
    "throw_pot":              "throwing_item",
    "buff":                   "buff_item",
    "applying_grease":        "buff_item",
    "grease":                 "buff_item",
    "physick":                "wondrous_physick",
    "wondrous_flask":         "wondrous_physick",
    "summon":                 "summon_spirit_ashes",
    "summon_ashes":           "summon_spirit_ashes",
    "spirit_ashes":           "summon_spirit_ashes",
    "summoning_spirits":      "summon_spirit_ashes",
    # interaction
    "interacting":            "interact",
    "looting":                "interact",
    "open_chest":             "opening_chest",
    "looting_chest":          "opening_chest",
    "chest":                  "opening_chest",
    "pickup_item":            "picking_up_item",
    "pick_up_item":           "picking_up_item",
    "pickup":                 "picking_up_item",
    "pick_up":                "picking_up_item",
    "open_door":              "opening_door",
    "door":                   "opening_door",
    "fog_gate":               "opening_door",
    "lever":                  "pulling_lever",
    "pull_lever":             "pulling_lever",
    "resting":                "resting_at_grace",
    "rest":                   "resting_at_grace",
    "site_of_grace":          "resting_at_grace",
    "bonfire":                "resting_at_grace",
    "talking":                "talking_to_npc",
    "talk":                   "talking_to_npc",
    "npc":                    "talking_to_npc",
    "dialogue":               "talking_to_npc",
    "message":                "reading_message",
    "reading":                "reading_message",
    "examine":                "examining",
    "inspect":                "examining",
    "lift":                   "operating_mechanism",
    "elevator":               "operating_mechanism",
    "mechanism":              "operating_mechanism",
    # camera / comms
    "lock_on":                "lock_on_toggle",
    "locking_on":             "lock_on_toggle",
    "targeting":              "lock_on_toggle",
    "target":                 "lock_on_toggle",
    "recenter_camera":        "reset_camera",
    "center_camera":          "reset_camera",
    "emote":                  "gesture",
    "gesturing":              "gesture",
    "summon_sign":            "place_summon_sign",
    # gui
    "map":                    "menu_map",
    "map_screen":             "menu_map",
    "menu":                   "menu_equipment",
    "equipment":              "menu_equipment",
    "inventory":              "menu_inventory",
    "status":                 "menu_status",
    "status_screen":          "menu_status",
    "equip_spells":           "menu_equip_spells",
    "memorize_spells":        "menu_equip_spells",
    "level_up":               "menu_level_up",
    "leveling_up":            "menu_level_up",
    "levelling_up":           "menu_level_up",
    "grace_menu":             "menu_grace",
    "crafting":               "menu_crafting",
    "craft":                  "menu_crafting",
    "shop":                   "menu_merchant",
    "merchant":               "menu_merchant",
    "store":                  "menu_merchant",
    # meta
    "cutscene":               "cinematic",
    "watching_cutscene":      "cinematic",
    "loading":                "loading_screen",
    "load_screen":            "loading_screen",
    "you_died":               "death_screen",
    "death":                  "death_screen",
    "died":                   "death_screen",
    "enemy_felled":           "boss_defeated",
    "great_enemy_felled":     "boss_defeated",
    "boss_killed":            "boss_defeated",
    "victory":                "boss_defeated",
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
    """True iff the Stage 1b (Gemini-vision) refinement should look at this segment."""
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
# Atomic-label vocabulary used by a downstream captioning prompt
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
        out = sys.argv[2] if len(sys.argv) > 2 else "EldenRing_taxonomy.json"
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
