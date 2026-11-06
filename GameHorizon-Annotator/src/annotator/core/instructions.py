from __future__ import annotations

import base64
import importlib.util
import json
import mimetypes
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .frames import compact_actions, load_annotation_frames, uniform_sample
from .profiles import GameProfile
from .schema import (
    atomic_write_json,
    instruction_map,
    instruction_path,
    instruction_payload,
    load_json,
    load_instruction_items,
    load_manifest,
    relative_ref,
    resolve_frame_refs,
)


RequestFn = Callable[
    [str, Sequence[Path], str, str | None, str | None, float, float | None], str
]
_TAXONOMY_LOCK = threading.Lock()
_TAXONOMY_CONTEXT_CACHE: Dict[tuple[str, str], str] = {}
_CLIENT_LOCAL = threading.local()
# Bumped from "legacy-detailed-v1" when the prompt started carrying mouse
# motion and buttons: the reader had been pointed at the semantic pointer
# layer, which holds neither, so earlier runs saw frames and keyboard only.
INSTRUCTION_PROMPT_VERSION = "detailed-v2-mouse"

# Without this the model receives bare delta pairs and has to guess the sign
# convention, the units, and whether the pairs are ordered.
#
# The vertical sign follows Windows RawInput, which the captures are taken
# with - the button flags are its literal ``RI_MOUSE_*`` constants - and whose
# ``RAWMOUSE.lLastY`` grows downward. Blind frame-pair probing agreed: across
# the six samples the model called the pitch "down" for positive dy in 36 of
# 57 decidable pairs.
ACTION_ANNOTATION_LEGEND = (
    "For mouse movement `[dx, dy]`, dx and dy represent the movement of the mouse in the x "
    "and y directions. dx > 0 means movement horizontally to the right. dx < 0 means movement "
    "horizontally to the left. dy > 0 means movement vertically downward, i.e. the view pitches "
    "toward the ground. dy < 0 means movement vertically upward. If one frame contains multiple "
    "actions, e.g. `mouse_movement: [[dx1,dy1],[dx2,dy2]]`, they are listed in the order of "
    "occurrence.\n"
    "Mouse-button and keyboard entries name the specific buttons and keys the player pressed "
    "or released; map them to in-game actions through the key bindings above."
)
PROHIBITED_CONTENT_MARKER = "PROHIBITED_CONTENT"


def _is_prohibited_content_error(exc: BaseException) -> bool:
    return PROHIBITED_CONTENT_MARKER in str(exc).upper()


def _load_python_asset(path: Path, module_name: str) -> Any:
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import Python asset: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_keybinds(path: Path, asset_format: str = "json") -> Dict[str, str]:
    if asset_format != "json":
        raise ValueError(f"unsupported keybind asset format: {asset_format}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    bindings = raw.get("bindings") if isinstance(raw.get("bindings"), dict) else raw
    result: Dict[str, str] = {}
    for key, value in bindings.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, str):
            result[str(key)] = value
            continue
        if not isinstance(value, dict):
            continue
        meanings = value.get("candidate_meanings")
        parts: List[str] = []
        if isinstance(meanings, list):
            parts.append("candidate meanings: " + ", ".join(str(item) for item in meanings))
        verification = value.get("verification")
        if verification:
            parts.append(f"verification: {verification}")
        if parts:
            result[str(key)] = "; ".join(parts)
    return result


def taxonomy_context(profile: GameProfile, label: str) -> str:
    cache_key = (profile.game_id, label)
    with _TAXONOMY_LOCK:
        if cache_key in _TAXONOMY_CONTEXT_CACHE:
            return _TAXONOMY_CONTEXT_CACHE[cache_key]
        module_name = f"unified_taxonomy_{profile.game_id}"
        try:
            module = sys.modules.get(module_name)
            if module is None:
                module = _load_python_asset(profile.taxonomy_path, module_name)
            if profile.taxonomy_format == "leaf_by_label":
                leaf = getattr(module, "LEAF_BY_LABEL", {}).get(label)
            else:
                raise ValueError(
                    f"unsupported taxonomy asset format: {profile.taxonomy_format}"
                )
        except Exception:
            leaf = None
        if leaf is None:
            context = ""
        elif isinstance(leaf, Mapping):
            names = " / ".join(
                str(leaf[key]) for key in ("name_en", "name_zh") if leaf.get(key)
            )
            tier = "/".join(
                str(leaf[key]) for key in ("tier1", "tier2") if leaf.get(key)
            )
            phrasing = " / ".join(
                str(leaf[key])
                for key in ("instruction_en", "instruction_zh")
                if leaf.get(key)
            )
            context = (
                f"Taxonomy description: {names} (tier={tier})\n"
                f"Suggested phrasing: {phrasing}"
            ).strip()
        else:
            description = str(getattr(leaf, "description", "") or "")
            phrasing = str(getattr(leaf, "instruction_phrasing", "") or "")
            context = f"Taxonomy description: {description}\nSuggested phrasing: {phrasing}".strip()
        _TAXONOMY_CONTEXT_CACHE[cache_key] = context
        return context


def _source_context(item: Mapping[str, Any], prior: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = []
    embedded = item.get("source_instructions")
    if isinstance(embedded, list):
        context.extend(dict(value) for value in embedded if isinstance(value, dict))
    seen = {str(value.get("scene")) for value in context}
    for scene in item.get("source_scenes", []) or []:
        key = str(scene)
        source = prior.get(key)
        if source and key not in seen:
            context.append({
                "scene": key,
                "instruction": source.get("instruction", ""),
                "duration_s": source.get("duration_s"),
            })
            seen.add(key)
    return context


def build_prompt(
    *,
    level: int,
    profile: GameProfile,
    item: Mapping[str, Any],
    sampled_refs: Sequence[str],
    actions: Sequence[Mapping[str, Any]],
    keybinds: Mapping[str, str],
    source_context: Sequence[Mapping[str, Any]],
    duration_window_s: tuple[float, float] | None = None,
) -> str:
    if level not in (1, 2, 3):
        raise ValueError(f"unsupported instruction level: {level}")
    label = str(item.get("effective_label") or item.get("label") or "unknown")
    segment_meta = {
        key: item.get(key)
        for key in (
            "scene", "duration_s", "start_frame", "end_frame", "effective_label",
            "tier1", "tier2", "refined_confidence", "refined_reason",
            "refined_reason_en", "refined_reason_zh", "refined_label_en",
            "refined_label_zh", "name_en", "name_zh", "instruction_en",
            "instruction_zh", "context_label", "context_at_start",
            "context_at_end", "control_mode", "control_mode_at_start",
            "control_mode_at_end", "nethack_visible", "context_transition_class",
            "context_confidence", "world_context", "locomotion_context",
            "combat_context", "concurrent_actions", "refinement_fallback",
            "quality_flags", "model_evidence_flags",
            "refined_sampled_video_pts_ns",
            "source_scenes", "num_source_scenes", "label_counts", "tier1_counts",
            "tier2_counts",
        )
        if item.get(key) is not None
    }
    task_by_level = {
        1: (
            "Analyze this short gameplay segment's visual context, UI state, important entities, "
            "spatial relationships, segment label, and the corresponding mouse/keyboard actions. "
            "Describe what the player is trying to do and the observable behavior used to do it."
        ),
        2: (
            "Analyze the visual context, important entities, spatial relationships, player actions, "
            "merge metadata, and the ordered Level-1 instructions. Describe the player's broader "
            "behavioral objective across the complete merged segment rather than narrating isolated frames."
        ),
        3: (
            "Analyze the visual context across the whole merged gameplay segment. Identify the major "
            "objective, important entities or targets, locations or route, combat/navigation/menu states, "
            "and meaningful transitions. Use the ordered Level-2 instructions as supporting evidence and "
            "describe the overall long-term behavioral goal."
        ),
    }
    goal_by_level = {
        1: (
            "The instruction must retain critical visible details—such as the target object or entity, "
            "direction, vehicle, cover, door, road, enemy, menu/UI state, or spatial relation—when they "
            "define the behavior. Do not reduce a visually specific behavior to a bare verb-object command."
        ),
        2: (
            "The instruction must be higher-level than the Level-1 inputs and may describe a short sequence "
            "of related behaviors when that sequence explains the sub-goal. Avoid excessive frame-by-frame "
            "detail, but preserve essential targets, NPCs, routes, vehicles, enemies, destinations, and UI states."
        ),
        3: (
            "State the overall objective of the entire segment rather than a local action. Preserve defining "
            "quest or mission goals, names, destinations, vehicles, combat targets, progression steps, or menu "
            "actions, and include the related behavior sequence when it is needed to make the objective clear."
        ),
    }
    duration_lines = [f"Actual segment duration: {float(item.get('duration_s', 0.0) or 0.0):.3f} seconds."]
    if duration_window_s is not None:
        minimum_s, maximum_s = duration_window_s
        duration_lines.append(
            f"Nominal Level-{level} duration window: {minimum_s / 60:.2f}–{maximum_s / 60:.2f} minutes "
            "when enough footage is available."
        )

    sections = [
        f"I am providing a sequence of sampled {profile.display_name} gameplay frames in temporal order, "
        "together with the available action annotations and segment context. Your task is to specify the "
        "player's behavioral goal as a natural-language instruction for a game agent.",
        f"### Task\n{task_by_level[level]}",
        f"### Game-Specific Focus\n{profile.prompt_focus[f'level_{level}']}",
        f"### Goal\n{goal_by_level[level]}",
        (
            "### Output Rules\n"
            "Only output the instruction text. Do not add a prefix such as \"The instruction is\" and do not "
            "output analysis, bullets, or a numbered list. Write one natural sentence that is focused but "
            "detailed enough to describe the observed behavior and its goal; it does not have to be a terse "
            "imperative command."
        ),
        "### Duration\n" + "\n".join(duration_lines),
        f"### Segment Metadata\n{json.dumps(segment_meta, ensure_ascii=False, indent=2)}",
    ]
    taxonomy = taxonomy_context(profile, label) if level == 1 else ""
    if taxonomy:
        sections.append(f"### Event Taxonomy Context\n{taxonomy}")
    if level >= 2:
        sections.append(
            "### Lower-Level Instructions Merged Into This Segment\n"
            + json.dumps(list(source_context), ensure_ascii=False, indent=2)
        )
    include_actions = level < 3 or profile.level_3_include_actions
    include_keybinds = level < 3 or profile.level_3_include_keybinds
    if include_keybinds:
        sections.append(f"### Key Bindings\n{json.dumps(dict(keybinds), ensure_ascii=False, indent=2)}")
    if include_actions:
        sections.append(
            "### Action Annotations For The Sampled Frames\n"
            + ACTION_ANNOTATION_LEGEND
            + "\n"
            + json.dumps(list(actions), ensure_ascii=False, indent=2)
        )
    sections.append(
        "### Sampled Frame References\n"
        + json.dumps(list(sampled_refs), ensure_ascii=False, indent=2)
        + "\nThe attached game frames follow this same temporal order."
    )
    return "\n\n".join(sections)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_instruction(
    prompt: str,
    images: Sequence[Path],
    model: str,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
    temperature: float | None = None,
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("no API key; use --api-key or OPENAI_API_KEY/GEMINI_API_KEY")
    from openai import OpenAI

    resolved_url = base_url or os.getenv("OPENAI_BASE_URL")
    clients = getattr(_CLIENT_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _CLIENT_LOCAL.clients = clients
    client_key = (key, resolved_url)
    client = clients.get(client_key)
    if client is None:
        client = OpenAI(api_key=key, base_url=resolved_url)
        clients[client_key] = client
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": _image_data_url(path)}} for path in images)
    # Left unset by default so the endpoint's own default applies: some
    # reasoning models reject an explicit temperature outright.
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        timeout=timeout,
        **options,
    )
    text = response.choices[0].message.content if response.choices else None
    if not text or not str(text).strip():
        raise RuntimeError("model returned no instruction")
    return str(text).strip()


def _frame_action_lookup(annotations_json: Path | None) -> Dict[int, Dict[str, Any]]:
    if annotations_json is None or not annotations_json.exists():
        return {}
    _, _, by_index = load_annotation_frames(annotations_json)
    return {index: compact_actions(frame) for index, frame in by_index.items()}


def _frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def generate_instructions(
    *,
    game_id: str,
    profile: GameProfile,
    workdir: Path,
    level: int,
    model: str,
    annotations_json: Path | None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 300.0,
    temperature: float | None = None,
    concurrency: int = 8,
    api_max_frames: int = 30,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    limit: int | None = None,
    skip_existing: bool = False,
    request_fn: RequestFn = request_instruction,
) -> Dict[str, Any]:
    manifest = load_manifest(workdir, level, fps=profile.fallback_fps)
    items = list(manifest["items"])
    if limit is not None:
        items = items[: max(0, limit)]
    output_path = instruction_path(workdir, level)
    existing_payload = load_json(output_path, {}) if skip_existing else {}
    existing_version_matches = (
        isinstance(existing_payload, dict)
        and existing_payload.get("prompt_version") == INSTRUCTION_PROMPT_VERSION
    )
    existing = {
        str(item.get("scene")): item
        for item in load_instruction_items(output_path)
        if item.get("scene") and item.get("status", "ok") == "ok" and item.get("instruction")
    } if skip_existing and existing_version_matches else {}
    results: Dict[str, Dict[str, Any]] = dict(existing)
    keybinds = load_keybinds(profile.keybinds_path, profile.keybinds_format)
    action_lookup = _frame_action_lookup(annotations_json)
    prior = instruction_map(instruction_path(workdir, level - 1)) if level > 1 else {}
    pending = [item for item in items if str(item["scene"]) not in existing]
    write_lock = threading.Lock()
    source = manifest.get("source") or {}
    try:
        minimum_s = float(source.get("minimum_s"))
        maximum_s = float(source.get("maximum_s"))
        duration_window_s = (minimum_s, maximum_s) if maximum_s >= minimum_s >= 0 else None
    except (TypeError, ValueError):
        duration_window_s = None

    def checkpoint() -> None:
        atomic_write_json(
            output_path,
            instruction_payload(
                game_id=game_id,
                level=level,
                model=model,
                items=results.values(),
                prompt_version=INSTRUCTION_PROMPT_VERSION,
            ),
        )

    def worker(item: Mapping[str, Any]) -> Dict[str, Any]:
        scene = str(item["scene"])
        all_paths = [path for path in resolve_frame_refs(workdir, item.get("frame_refs", [])) if path.exists()]
        sampled_paths = uniform_sample(all_paths, api_max_frames)
        context = _source_context(item, prior)

        def build_evidence(paths: Sequence[Path]) -> tuple[list[str], str]:
            refs = [relative_ref(path, workdir) for path in paths]
            actions = []
            for path in paths:
                index = _frame_index(path)
                actions.append(
                    action_lookup.get(index, {"frame_index": index})
                    if index is not None
                    else {}
                )
            return refs, build_prompt(
                level=level,
                profile=profile,
                item=item,
                sampled_refs=refs,
                actions=actions,
                keybinds=keybinds,
                source_context=context,
                duration_window_s=duration_window_s,
            )

        sampled_refs, prompt = build_evidence(sampled_paths)
        error = ""
        content_filter_fallback: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                instruction = request_fn(
                    prompt, sampled_paths, model, api_key, base_url, timeout, temperature
                )
                result = {
                    "scene": scene,
                    "order": item.get("order"),
                    "level": level,
                    "duration_s": item.get("duration_s"),
                    "instruction": instruction,
                    "status": "ok",
                    "attempts": attempt,
                    "sampled_frame_refs": sampled_refs,
                    "source_scenes": list(item.get("source_scenes", []) or []),
                    "prompt_version": INSTRUCTION_PROMPT_VERSION,
                }
                if content_filter_fallback:
                    result["content_filter_fallback"] = content_filter_fallback
                return result
            except Exception as exc:
                error = str(exc)[:1000]
                if attempt < max_retries:
                    if _is_prohibited_content_error(exc):
                        if len(sampled_paths) > 4:
                            sampled_paths = uniform_sample(sampled_paths, 4)
                            content_filter_fallback = "reduced_visual_evidence"
                        elif sampled_paths:
                            sampled_paths = []
                            content_filter_fallback = "metadata_only"
                        sampled_refs, prompt = build_evidence(sampled_paths)
                    time.sleep(max(0.0, retry_delay) * attempt)
        result = {
            "scene": scene,
            "order": item.get("order"),
            "level": level,
            "duration_s": item.get("duration_s"),
            "instruction": "",
            "status": "error",
            "error": error,
            "attempts": max_retries,
            "sampled_frame_refs": sampled_refs,
            "source_scenes": list(item.get("source_scenes", []) or []),
            "prompt_version": INSTRUCTION_PROMPT_VERSION,
        }
        if content_filter_fallback:
            result["content_filter_fallback"] = content_filter_fallback
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(worker, item): str(item["scene"]) for item in pending}
        for future in as_completed(futures):
            result = future.result()
            with write_lock:
                results[str(result["scene"])] = result
                checkpoint()
    checkpoint()
    return instruction_payload(
        game_id=game_id,
        level=level,
        model=model,
        items=results.values(),
        prompt_version=INSTRUCTION_PROMPT_VERSION,
    )
