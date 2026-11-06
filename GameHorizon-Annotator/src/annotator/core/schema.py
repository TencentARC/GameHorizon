from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = 2
# On Windows a virus scanner or the search indexer can hold a handle to a file
# for a few hundred milliseconds after it is written, which makes os.replace
# fail with "access denied" (WinError 5) or "sharing violation" (WinError 32).
# The instruction stages checkpoint after every batch, so over a long run this
# transient collision is close to certain; retrying briefly is enough.
_REPLACE_ATTEMPTS = 8
_REPLACE_BACKOFF_S = 0.25
NS_PER_S = 1_000_000_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_with_retry(tmp_name: str, path: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp_name, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short. Repeating a long target basename here can
    # push otherwise valid Windows paths over MAX_PATH before the atomic replace.
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def effective_label(item: Mapping[str, Any]) -> str:
    return str(item.get("refined_label") or item.get("effective_label") or item.get("label") or "unknown")


def duration_seconds(item: Mapping[str, Any], fps: float | None = None) -> float:
    start_ts = _as_int(item.get("start_timestamp_ns"))
    end_ts = _as_int(item.get("end_timestamp_ns"))
    if start_ts is not None and end_ts is not None and end_ts > start_ts:
        return (end_ts - start_ts) / NS_PER_S
    supplied = _as_float(item.get("duration_s"))
    if supplied is not None and supplied >= 0:
        return supplied
    start_frame = _as_int(item.get("start_frame"))
    end_frame = _as_int(item.get("end_frame"))
    if fps and fps > 0 and start_frame is not None and end_frame is not None and end_frame >= start_frame:
        return (end_frame - start_frame + 1) / fps
    raise ValueError(
        f"cannot determine real duration for {item.get('scene') or item.get('segment_index')}; "
        "provide timestamps, duration_s, or a frame span with FPS"
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _scene_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    scene = str(item.get("scene", ""))
    order = _as_int(item.get("order"))
    if order is not None:
        return order, scene
    try:
        return int(scene.rsplit("_", 1)[-1]), scene
    except ValueError:
        return 10**12, scene


def normalize_segments(payload: Any, *, fps: float | None = None) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_items = payload.get("segments") or payload.get("items") or payload.get("scenes") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("segments JSON must be an object or list")
    if not isinstance(raw_items, list):
        raise ValueError("segments/items/scenes must be a list")

    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["segment_index"] = int(item.get("segment_index", index))
        item["order"] = index
        item["effective_label"] = effective_label(item)
        item["duration_s"] = round(duration_seconds(item, fps), 9)
        normalized.append(item)
    normalized.sort(key=lambda value: (
        _as_int(value.get("start_frame")) if _as_int(value.get("start_frame")) is not None else 10**18,
        _as_int(value.get("start_timestamp_ns")) if _as_int(value.get("start_timestamp_ns")) is not None else 10**30,
        int(value["segment_index"]),
    ))
    for order, item in enumerate(normalized):
        item["order"] = order
    return normalized


def manifest_path(workdir: Path, level: int) -> Path:
    return workdir / f"frames_level_{level}" / "manifest.json"


def legacy_manifest_candidates(workdir: Path, level: int) -> List[Path]:
    base = workdir / f"frames_level_{level}"
    names = ["_segments_map.json"] if level == 1 else ["_merge_map.json", f"_level{level}_map.json"]
    return [base / name for name in names]


def load_manifest(workdir: Path, level: int, *, fps: float | None = None) -> Dict[str, Any]:
    path = manifest_path(workdir, level)
    if not path.exists():
        path = next((candidate for candidate in legacy_manifest_candidates(workdir, level) if candidate.exists()), path)
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise FileNotFoundError(f"manifest not found or invalid for level {level}: {path}")
    raw_items = raw.get("items") if isinstance(raw.get("items"), list) else raw.get("scenes", [])
    items: List[Dict[str, Any]] = []
    for index, value in enumerate(raw_items or []):
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item["scene"] = str(item.get("scene") or f"scene_{index:06d}")
        item["order"] = int(item.get("order", index))
        item["duration_s"] = round(duration_seconds(item, fps), 9)
        item["effective_label"] = effective_label(item)
        item.setdefault("source_scenes", list(item.get("source_scene_names") or []))
        item.setdefault("source_instructions", list(item.get("source_instructions") or []))
        refs = item.get("frame_refs")
        if not isinstance(refs, list):
            refs = _legacy_frame_refs(workdir, level, item["scene"])
        item["frame_refs"] = [str(ref) for ref in refs]
        items.append(item)
    items.sort(key=_scene_sort_key)
    for order, item in enumerate(items):
        item["order"] = order
    return {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "game_id": raw.get("game_id"),
        "source": dict(raw.get("source") or {}),
        "items": items,
        "legacy_source": str(path) if path.name != "manifest.json" else None,
    }


def _legacy_frame_refs(workdir: Path, level: int, scene: str) -> List[str]:
    scene_dir = workdir / f"frames_level_{level}" / scene
    if not scene_dir.is_dir():
        return []
    paths = sorted(
        (path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else 10**12, path.name),
    )
    return [relative_ref(path, workdir) for path in paths]


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _configured_frame_cache_root() -> Path | None:
    value = os.environ.get("OWA_LOCAL_FRAME_CACHE_DIR", "").strip()
    return _lexical_absolute(Path(value)) if value else None


def _require_within(path: Path, root: Path, ref: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"frame reference escapes allowed roots: {ref}") from exc


def relative_ref(path: Path, workdir: Path) -> str:
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(workdir)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        cache_root = _configured_frame_cache_root()
        if cache_root is None:
            raise ValueError(f"frame reference escapes workdir: {path}")
        _require_within(candidate, cache_root, str(path))
        return str(candidate)


def resolve_frame_refs(workdir: Path, refs: Sequence[str]) -> List[Path]:
    root = _lexical_absolute(workdir)
    cache_root = _configured_frame_cache_root()
    resolved: List[Path] = []
    for ref in refs:
        raw = Path(ref)
        if raw.is_absolute():
            path = _lexical_absolute(raw)
            if cache_root is None:
                raise ValueError(f"absolute frame reference requires configured cache: {ref}")
            _require_within(path, cache_root, ref)
        else:
            path = _lexical_absolute(root / raw)
            _require_within(path, root, ref)
        resolved.append(path)
    return resolved


def instruction_path(workdir: Path, level: int) -> Path:
    return workdir / f"level_{level}_instructions.json"


def load_instruction_items(path: Path) -> List[Dict[str, Any]]:
    raw = load_json(path, [])
    if isinstance(raw, dict):
        items = raw.get("items", [])
    else:
        items = raw
    return [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def instruction_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("scene")): item for item in load_instruction_items(path) if item.get("scene")}


def instruction_payload(
    *,
    game_id: str,
    level: int,
    model: str,
    items: Iterable[Dict[str, Any]],
    prompt_version: str | None = None,
) -> Dict[str, Any]:
    ordered = sorted((dict(item) for item in items), key=_scene_sort_key)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "level": level,
        "model": model,
        "items": ordered,
    }
    if prompt_version:
        payload["prompt_version"] = prompt_version
    return payload


def make_manifest(*, game_id: str, level: int, source: Mapping[str, Any], items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted((dict(item) for item in items), key=_scene_sort_key)
    for index, item in enumerate(ordered):
        item["order"] = index
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "level": level,
        "source": dict(source),
        "items": ordered,
    }
