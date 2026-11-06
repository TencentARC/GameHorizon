from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .frames import uniform_sample
from .instructions import RequestFn, request_instruction
from .profiles import GameProfile
from .schema import (
    atomic_write_json,
    instruction_map,
    make_manifest,
    manifest_path,
    relative_ref,
    resolve_frame_refs,
)


DecisionFn = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Dict[str, Any]]
MERGE_PROMPT_VERSION = "legacy-adjacent-detailed-v1"


def group_duration(items: Sequence[Mapping[str, Any]]) -> float:
    if not items:
        return 0.0
    try:
        start = int(items[0]["start_timestamp_ns"])
        end = int(items[-1]["end_timestamp_ns"])
        if end > start:
            return (end - start) / 1_000_000_000
    except (KeyError, TypeError, ValueError):
        pass
    return sum(float(item.get("duration_s", 0.0) or 0.0) for item in items)


def deterministic_duration_groups(
    items: Sequence[Mapping[str, Any]],
    *,
    minimum_s: float,
    maximum_s: float,
    seed: int,
) -> List[List[Mapping[str, Any]]]:
    if minimum_s < 0 or maximum_s <= 0 or maximum_s < minimum_s:
        raise ValueError("invalid duration window")
    if not items:
        return []
    rng = random.Random(seed)
    groups: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    target = rng.uniform(minimum_s, maximum_s)

    def close() -> None:
        nonlocal current, target
        if current:
            groups.append(current)
        current = []
        target = rng.uniform(minimum_s, maximum_s)

    for item in items:
        if not current:
            current = [item]
            if group_duration(current) > maximum_s:
                close()
            continue
        candidate_duration = group_duration([*current, item])
        current_duration = group_duration(current)
        if candidate_duration > maximum_s:
            close()
            current = [item]
            if group_duration(current) > maximum_s:
                close()
        elif current_duration >= minimum_s and candidate_duration > target:
            close()
            current = [item]
        else:
            current.append(item)
    close()

    if len(groups) >= 2 and group_duration(groups[-1]) < minimum_s:
        combined = [*groups[-2], *groups[-1]]
        if group_duration(combined) <= maximum_s:
            groups[-2:] = [combined]
    return groups


def _instruction_context(
    source_items: Sequence[Mapping[str, Any]],
    instructions: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    for item in source_items:
        scene = str(item.get("scene", ""))
        instruction = instructions.get(scene, {})
        result.append({
            "scene": scene,
            "duration_s": item.get("duration_s"),
            "instruction": instruction.get("instruction", ""),
            "status": instruction.get("status"),
        })
    return result


def merge_items(
    source_items: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    order: int,
    instructions: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    if not source_items:
        raise ValueError("cannot merge an empty source group")
    first, last = source_items[0], source_items[-1]
    refs: List[str] = []
    seen_refs: set[str] = set()
    for item in source_items:
        for ref in item.get("frame_refs", []) or []:
            value = str(ref)
            if value not in seen_refs:
                refs.append(value)
                seen_refs.add(value)
    labels = {str(item.get("effective_label", "")) for item in source_items}
    label_counts: Counter[str] = Counter()
    tier1_counts: Counter[str] = Counter()
    tier2_counts: Counter[str] = Counter()
    for item in source_items:
        nested_labels = item.get("label_counts")
        if isinstance(nested_labels, Mapping):
            label_counts.update({str(key): int(value) for key, value in nested_labels.items()})
        else:
            label = str(item.get("effective_label") or item.get("label") or "").strip()
            if label:
                label_counts[label] += 1
        for field, nested_field, counter in (
            ("tier1", "tier1_counts", tier1_counts),
            ("tier2", "tier2_counts", tier2_counts),
        ):
            nested = item.get(nested_field)
            if isinstance(nested, Mapping):
                counter.update({str(key): int(value) for key, value in nested.items()})
            else:
                value = str(item.get(field) or "").strip()
                if value:
                    counter[value] += 1
    source_scenes = [str(item.get("scene")) for item in source_items]
    return {
        "scene": scene,
        "order": order,
        "start_frame": first.get("start_frame"),
        "end_frame": last.get("end_frame"),
        "start_timestamp_ns": first.get("start_timestamp_ns"),
        "end_timestamp_ns": last.get("end_timestamp_ns"),
        "duration_s": round(group_duration(source_items), 9),
        "effective_label": next(iter(labels)) if len(labels) == 1 else "mixed",
        "frame_refs": refs,
        "source_scenes": source_scenes,
        "merged_source_scenes": source_scenes,
        "num_source_scenes": len(source_scenes),
        "num_sampled_frames": len(refs),
        "label_counts": dict(sorted(label_counts.items())),
        "tier1_counts": dict(sorted(tier1_counts.items())),
        "tier2_counts": dict(sorted(tier2_counts.items())),
        "source_instructions": _instruction_context(source_items, instructions),
        "merge_decisions": [dict(value) for value in decisions or []],
    }


def materialize_manifest(
    *,
    workdir: Path,
    level: int,
    manifest: Dict[str, Any],
    mode: str,
) -> None:
    if mode not in {"none", "symlink", "hardlink", "copy"}:
        raise ValueError(f"unsupported materialization mode: {mode}")
    level_dir = workdir / f"frames_level_{level}"
    level_dir.mkdir(parents=True, exist_ok=True)
    for path in level_dir.iterdir():
        if path.is_dir() and path.name.startswith("scene_"):
            shutil.rmtree(path)
    if mode == "none":
        for item in manifest["items"]:
            item.pop("materialized_frame_refs", None)
        return

    for item in manifest["items"]:
        scene_dir = level_dir / str(item["scene"])
        scene_dir.mkdir(parents=True, exist_ok=True)
        materialized: List[str] = []
        for index, source in enumerate(resolve_frame_refs(workdir, item.get("frame_refs", []))):
            if not source.exists():
                continue
            destination = scene_dir / f"{index:06d}__{source.name}"
            if mode == "symlink":
                relative_target = os.path.relpath(source.resolve(), destination.parent.resolve())
                destination.symlink_to(relative_target)
            elif mode == "hardlink":
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            else:
                shutil.copy2(source, destination)
            materialized.append(destination.absolute().relative_to(workdir.absolute()).as_posix())
        item["materialized_frame_refs"] = materialized


def build_level2_manifest(
    *,
    game_id: str,
    workdir: Path,
    level1_manifest: Mapping[str, Any],
    minimum_s: float,
    maximum_s: float,
    seed: int,
    materialize: str,
) -> Dict[str, Any]:
    instructions = instruction_map(workdir / "level_1_instructions.json")
    groups = deterministic_duration_groups(
        list(level1_manifest.get("items", [])),
        minimum_s=minimum_s,
        maximum_s=maximum_s,
        seed=seed,
    )
    items = [
        merge_items(
            group,
            scene=f"scene_{index:06d}",
            order=index,
            instructions=instructions,
        )
        for index, group in enumerate(groups)
    ]
    manifest = make_manifest(
        game_id=game_id,
        level=2,
        source={
            "level_1_manifest": "frames_level_1/manifest.json",
            "minimum_s": minimum_s,
            "maximum_s": maximum_s,
            "seed": seed,
            "materialize": materialize,
            "grouping_strategy": "legacy_seeded_real_duration_window",
        },
        items=items,
    )
    materialize_manifest(workdir=workdir, level=2, manifest=manifest, mode=materialize)
    atomic_write_json(manifest_path(workdir, 2), manifest)
    return manifest


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("model response contains no JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def semantic_decision_prompt(
    *,
    profile: GameProfile,
    current: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    minimum_s: float,
    maximum_s: float,
    current_frame_refs: Sequence[str] = (),
    candidate_frame_refs: Sequence[str] = (),
) -> str:
    def level2_context(item: Mapping[str, Any]) -> Dict[str, Any]:
        instruction = item.get("level_2_instruction")
        instruction = instruction if isinstance(instruction, Mapping) else {}
        return {
            "scene": item.get("scene"),
            "duration_s": item.get("duration_s"),
            "instruction": instruction.get("instruction", ""),
            "status": instruction.get("status"),
            "lower_level_instructions": list(item.get("source_instructions", []) or []),
        }

    if not current:
        raise ValueError("current Level-2 context cannot be empty")
    current_context = level2_context(current[-1])
    candidate_context = level2_context(candidate)
    return (
        "You are an expert game AI researcher. I am providing sampled video frames and lower-level "
        f"behavioral sub-goals for two consecutive {profile.display_name} Level-2 gameplay segments.\n\n"
        "### Goal\n"
        "Decide whether Segment B is a direct continuation, logical next step, or part of the same broad "
        "quest, mission, encounter, navigation, construction, preparation, progression, or menu-management "
        "objective as Segment A. Return merge=false when there is a clear high-level objective shift.\n\n"
        "### Duration Guidance\n"
        f"Level-3 segments normally target {minimum_s / 60:.2f}–{maximum_s / 60:.2f} minutes. "
        "The maximum is a hard boundary enforced outside this prompt; semantic continuity determines this "
        "adjacent decision and is more important than merely reaching the nominal minimum.\n\n"
        f"### Game-Specific Continuity Criteria\n{profile.continuity_hint}\n\n"
        "### General Decision Criteria\n"
        "- MERGE: Segment B continues the same broader objective, encounter, route, sustained fight, recovery "
        "sequence, interaction flow, or logical task phase as Segment A.\n"
        "- SPLIT: Segment B starts an unrelated objective or destination, a new encounter or task, a major "
        "reset/death/results/lobby transition, a cutscene or character/session change, or another clear break "
        "in behavioral intent.\n\n"
        f"### Segment A Information\n{json.dumps(current_context, ensure_ascii=False, indent=2)}\n"
        f"Sampled frame references for Segment A:\n{json.dumps(list(current_frame_refs), ensure_ascii=False, indent=2)}\n\n"
        f"### Segment B Information\n{json.dumps(candidate_context, ensure_ascii=False, indent=2)}\n"
        f"Sampled frame references for Segment B:\n{json.dumps(list(candidate_frame_refs), ensure_ascii=False, indent=2)}\n\n"
        "### Output Requirement\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        '{"merge": true|false, "reason": "brief explanation"}. Do not use markdown.'
    )


def make_api_decision_fn(
    *,
    profile: GameProfile,
    workdir: Path,
    model: str,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
    minimum_s: float,
    maximum_s: float,
    api_max_frames: int,
    temperature: float | None = None,
    request_fn: RequestFn = request_instruction,
) -> DecisionFn:
    level2_instructions = instruction_map(workdir / "level_2_instructions.json")

    def with_level2_instruction(item: Mapping[str, Any]) -> Dict[str, Any]:
        enriched = dict(item)
        enriched["level_2_instruction"] = level2_instructions.get(str(item.get("scene")), {})
        return enriched

    def decide(current: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> Dict[str, Any]:
        if not current:
            raise ValueError("current Level-2 group cannot be empty")
        boundary = current[-1]
        sample_each = max(1, api_max_frames // 2)
        current_paths = [
            path
            for path in resolve_frame_refs(workdir, [str(value) for value in boundary.get("frame_refs", []) or []])
            if path.exists()
        ]
        candidate_paths = [
            path
            for path in resolve_frame_refs(workdir, [str(value) for value in candidate.get("frame_refs", []) or []])
            if path.exists()
        ]
        sampled_current = uniform_sample(current_paths, sample_each)
        sampled_candidate = uniform_sample(candidate_paths, sample_each)
        sampled = [*sampled_current, *sampled_candidate]
        current_refs = [relative_ref(path, workdir) for path in sampled_current]
        candidate_refs = [relative_ref(path, workdir) for path in sampled_candidate]
        prompt = semantic_decision_prompt(
            profile=profile,
            current=[with_level2_instruction(boundary)],
            candidate=with_level2_instruction(candidate),
            minimum_s=minimum_s,
            maximum_s=maximum_s,
            current_frame_refs=current_refs,
            candidate_frame_refs=candidate_refs,
        )
        raw = request_fn(prompt, sampled, model, api_key, base_url, timeout, temperature)
        parsed = extract_json_object(raw)
        if not isinstance(parsed.get("merge"), bool):
            raise ValueError("merge decision must contain a boolean 'merge'")
        return {
            "merge": parsed["merge"],
            "reason": str(parsed.get("reason", ""))[:500],
            "prompt_version": MERGE_PROMPT_VERSION,
        }

    return decide


def build_level3_manifest(
    *,
    game_id: str,
    workdir: Path,
    level2_manifest: Mapping[str, Any],
    minimum_s: float,
    maximum_s: float,
    decision_fn: DecisionFn,
    materialize: str,
) -> Dict[str, Any]:
    source_items = list(level2_manifest.get("items", []))
    instructions = instruction_map(workdir / "level_2_instructions.json")
    groups: List[tuple[List[Mapping[str, Any]], List[Dict[str, Any]]]] = []
    current: List[Mapping[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    def close() -> None:
        nonlocal current, decisions
        if current:
            groups.append((current, decisions))
        current = []
        decisions = []

    for item in source_items:
        if not current:
            current = [item]
            continue
        candidate_duration = group_duration([*current, item])
        if candidate_duration > maximum_s:
            decisions.append({
                "candidate": item.get("scene"),
                "merge": False,
                "reason": "hard maximum duration would be exceeded",
                "source": "duration_guard",
            })
            close()
            current = [item]
            continue
        try:
            decision = dict(decision_fn(current, item))
            decision["source"] = "model"
        except Exception as exc:
            decision = {"merge": False, "reason": str(exc)[:500], "source": "fail_safe"}
        decision["candidate"] = item.get("scene")
        decisions.append(decision)
        if decision.get("merge") is True:
            current.append(item)
        else:
            close()
            current = [item]
    close()

    items = [
        merge_items(
            group,
            scene=f"scene_{index:06d}",
            order=index,
            instructions=instructions,
            decisions=group_decisions,
        )
        for index, (group, group_decisions) in enumerate(groups)
    ]
    manifest = make_manifest(
        game_id=game_id,
        level=3,
        source={
            "level_2_manifest": "frames_level_2/manifest.json",
            "minimum_s": minimum_s,
            "maximum_s": maximum_s,
            "materialize": materialize,
            "failure_policy": "split",
            "decision_scope": "adjacent_level2_boundary",
            "decision_prompt_version": MERGE_PROMPT_VERSION,
        },
        items=items,
    )
    materialize_manifest(workdir=workdir, level=3, manifest=manifest, mode=materialize)
    atomic_write_json(manifest_path(workdir, 3), manifest)
    return manifest
