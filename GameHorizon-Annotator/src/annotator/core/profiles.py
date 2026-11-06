from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = PACKAGE_ROOT / "game_profiles"


@dataclass(frozen=True)
class GameProfile:
    game_id: str
    display_name: str
    stage_1a_script: Path
    stage_1b_script: Path
    taxonomy_path: Path
    keybinds_path: Path
    fallback_fps: float
    stage_1a_defaults: Dict[str, Any]
    stage_1b_defaults: Dict[str, Any]
    prompt_focus: Dict[str, str]
    continuity_hint: str
    level_3_include_actions: bool = False
    level_3_include_keybinds: bool = False
    stage_1_backend: str = "legacy_cli"
    segment_disable_default_flag: str | None = None
    refine_supports_video_start_ns: bool = False
    taxonomy_format: str = "leaf_by_label"
    keybinds_format: str = "json"
    raw: Dict[str, Any] | None = None


def supported_games() -> List[str]:
    return sorted(path.stem for path in PROFILE_DIR.glob("*.json") if not path.name.startswith("._"))


def _package_path(value: str, *, field: str) -> Path:
    path = (PACKAGE_ROOT / value).resolve()
    try:
        path.relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"profile field {field!r} escapes the installed package: {value}") from exc
    return path


def load_profile(game_id: str) -> GameProfile:
    profile_path = PROFILE_DIR / f"{game_id}.json"
    if not profile_path.exists():
        choices = ", ".join(supported_games())
        raise ValueError(f"unknown game {game_id!r}; expected one of: {choices}")
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported game profile schema: {profile_path}")
    if data.get("game_id") != game_id:
        raise ValueError(f"profile game_id mismatch in {profile_path}")

    scripts = data.get("scripts") or {}
    assets = data.get("assets") or {}
    prompts = data.get("prompts") or {}
    required_prompt_keys = {"level_1", "level_2", "level_3"}
    if not required_prompt_keys.issubset(prompts):
        missing = sorted(required_prompt_keys - set(prompts))
        raise ValueError(f"profile {game_id} is missing prompt fields: {missing}")
    level_3_context = prompts.get("level_3_context") or {}
    if not isinstance(level_3_context, dict):
        raise ValueError(f"profile {game_id} prompts.level_3_context must be an object")

    defaults = data.get("defaults") or {}
    stage_1a_defaults = defaults.get("stage_1a") or {}
    stage_1b_defaults = defaults.get("stage_1b") or {}
    if not isinstance(stage_1a_defaults, dict) or not isinstance(stage_1b_defaults, dict):
        raise ValueError(f"profile {game_id} stage defaults must be objects")
    stage_1_backend = str(scripts.get("stage_1_backend", "legacy_cli"))
    if stage_1_backend != "legacy_cli":
        raise ValueError(
            f"profile {game_id} has unsupported Stage 1 backend {stage_1_backend!r}"
        )
    taxonomy_format = str(assets.get("taxonomy_format", "leaf_by_label"))
    if taxonomy_format != "leaf_by_label":
        raise ValueError(
            f"profile {game_id} has unsupported taxonomy format {taxonomy_format!r}"
        )
    keybinds_format = str(assets.get("keybinds_format", "json"))
    if keybinds_format != "json":
        raise ValueError(
            f"profile {game_id} has unsupported keybinds format {keybinds_format!r}"
        )

    return GameProfile(
        game_id=game_id,
        display_name=str(data["display_name"]),
        stage_1a_script=_package_path(str(scripts["stage_1a"]), field="stage_1a"),
        stage_1b_script=_package_path(str(scripts["stage_1b"]), field="stage_1b"),
        taxonomy_path=_package_path(str(assets["taxonomy"]), field="taxonomy"),
        keybinds_path=_package_path(str(assets["keybinds"]), field="keybinds"),
        fallback_fps=float(defaults.get("fallback_fps", 20.0)),
        stage_1a_defaults=dict(stage_1a_defaults),
        stage_1b_defaults=dict(stage_1b_defaults),
        prompt_focus={key: str(prompts[key]) for key in required_prompt_keys},
        continuity_hint=str(prompts.get("continuity", "")),
        level_3_include_actions=bool(level_3_context.get("include_actions", False)),
        level_3_include_keybinds=bool(level_3_context.get("include_keybinds", False)),
        stage_1_backend=stage_1_backend,
        segment_disable_default_flag=(
            str(scripts["segment_disable_default_flag"])
            if scripts.get("segment_disable_default_flag")
            else None
        ),
        refine_supports_video_start_ns=bool(scripts.get("refine_supports_video_start_ns")),
        taxonomy_format=taxonomy_format,
        keybinds_format=keybinds_format,
        raw=data,
    )
