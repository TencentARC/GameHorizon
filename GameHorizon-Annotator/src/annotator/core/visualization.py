from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .profiles import load_profile
from .schema import (
    NS_PER_S,
    atomic_write_json,
    instruction_path,
    load_instruction_items,
    load_manifest,
    manifest_path,
)


DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL")
DEFAULT_TRANSLATION_MODEL = "gemini-3.5-flash"
TRANSLATION_PROMPT_VERSION = "game-instruction-zh-v1"
GAME_ORDER = (
    "minecraft",
    "eldenring",
    "eldenring_nightreign",
    "apex_legends",
    "gta5",
    "cyberpunk_2077",
)
_WINDOWS_FONT_DIR = (
    Path(os.environ["WINDIR"]) / "Fonts" if os.environ.get("WINDIR") else None
)
DEFAULT_FONT_CANDIDATES = tuple(
    str(_WINDOWS_FONT_DIR / name)
    for name in ("msyh.ttc", "msyhl.ttc", "simsun.ttc")
    if _WINDOWS_FONT_DIR is not None
) + (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

_CLIENT_LOCAL = threading.local()
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class VisualizationError(RuntimeError):
    pass


@dataclass
class CaptionInterval:
    scene: str
    order: int
    start_s: float
    end_s: float
    instruction_en: str
    instruction_zh: str = ""


@dataclass
class VisualizationJob:
    workdir: Path
    game_id: str
    display_name: str
    level: int
    video_path: Path
    video_start_ns: int
    timeline_origin_ns: int
    timeline_end_ns: int
    seek_s: float
    duration_s: float
    captions: list[CaptionInterval]

    def limited(self, seconds: float | None) -> "VisualizationJob":
        if seconds is None or seconds <= 0 or seconds >= self.duration_s:
            return self
        captions: list[CaptionInterval] = []
        for caption in self.captions:
            start_s = max(0.0, caption.start_s)
            end_s = min(float(seconds), caption.end_s)
            if end_s > start_s:
                captions.append(replace(caption, start_s=start_s, end_s=end_s))
        return replace(
            self,
            timeline_end_ns=self.timeline_origin_ns + round(seconds * NS_PER_S),
            duration_s=float(seconds),
            captions=captions,
        )


TranslationRequestFn = Callable[
    [str, Mapping[str, str], str, str | None, float], Mapping[str, str]
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_game_workdirs(input_root: Path, selected: Sequence[str] | None = None) -> list[Path]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise VisualizationError(f"input root not found: {root}")
    if selected:
        names = list(dict.fromkeys(str(value) for value in selected))
    else:
        found = {
            child.name
            for child in root.iterdir()
            if child.is_dir() and manifest_path(child, 1).is_file()
        }
        names = [name for name in GAME_ORDER if name in found]
        names.extend(sorted(found.difference(names)))
    missing = [name for name in names if not manifest_path(root / name, 1).is_file()]
    if missing:
        raise VisualizationError(f"game workdir(s) missing Level 1 manifest: {', '.join(missing)}")
    return [root / name for name in names]


def _integer(value: Any, *, field: str, scene: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise VisualizationError(f"{scene}: missing or invalid {field}") from exc


def _source_video(workdir: Path, level1_manifest: Mapping[str, Any]) -> Path:
    source = level1_manifest.get("source") or {}
    raw = str(source.get("video") or "").strip()
    if not raw:
        raise VisualizationError(f"Level 1 manifest has no source.video: {workdir}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = workdir / path
    path = path.resolve()
    if not path.is_file():
        raise VisualizationError(f"source video not found: {path}")
    return path


def load_visualization_job(workdir: Path, level: int) -> VisualizationJob:
    if level not in (1, 2):
        raise VisualizationError(f"only Level 1 and Level 2 are supported, got {level}")
    workdir = workdir.expanduser().resolve()
    level1_manifest = load_manifest(workdir, 1)
    manifest = level1_manifest if level == 1 else load_manifest(workdir, level)
    game_id = str(manifest.get("game_id") or level1_manifest.get("game_id") or workdir.name)
    try:
        display_name = load_profile(game_id).display_name
    except Exception:
        display_name = game_id.replace("_", " ").title()
    video_path = _source_video(workdir, level1_manifest)
    video_start_ns = int((level1_manifest.get("source") or {}).get("video_start_ns") or 0)

    manifest_items = list(manifest.get("items") or [])
    instruction_items = load_instruction_items(instruction_path(workdir, level))
    if not manifest_items:
        raise VisualizationError(f"{game_id} Level {level}: manifest has no items")
    if not instruction_items:
        raise VisualizationError(f"{game_id} Level {level}: instruction file has no items")

    manifest_by_scene: dict[str, Mapping[str, Any]] = {}
    for item in manifest_items:
        scene = str(item.get("scene") or "")
        if not scene:
            raise VisualizationError(f"{game_id} Level {level}: manifest item has no scene")
        if scene in manifest_by_scene:
            raise VisualizationError(f"{game_id} Level {level}: duplicate manifest scene {scene}")
        manifest_by_scene[scene] = item

    instruction_by_scene: dict[str, Mapping[str, Any]] = {}
    for item in instruction_items:
        scene = str(item.get("scene") or "")
        if not scene:
            raise VisualizationError(f"{game_id} Level {level}: instruction item has no scene")
        if scene in instruction_by_scene:
            raise VisualizationError(f"{game_id} Level {level}: duplicate instruction scene {scene}")
        instruction_by_scene[scene] = item

    manifest_scenes = set(manifest_by_scene)
    instruction_scenes = set(instruction_by_scene)
    if manifest_scenes != instruction_scenes:
        missing = sorted(manifest_scenes - instruction_scenes)
        extra = sorted(instruction_scenes - manifest_scenes)
        raise VisualizationError(
            f"{game_id} Level {level}: manifest/instruction scene mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    raw_rows: list[tuple[str, int, int, int, str, str]] = []
    ordered_items = sorted(
        manifest_items,
        key=lambda item: (int(item.get("order", 10**12)), str(item.get("scene", ""))),
    )
    for fallback_order, manifest_item in enumerate(ordered_items):
        scene = str(manifest_item["scene"])
        instruction_item = instruction_by_scene[scene]
        if str(instruction_item.get("status", "ok")) != "ok":
            raise VisualizationError(f"{game_id} Level {level} {scene}: instruction status is not ok")
        instruction_en = str(instruction_item.get("instruction") or "").strip()
        if not instruction_en:
            raise VisualizationError(f"{game_id} Level {level} {scene}: instruction is empty")
        instruction_zh = str(
            instruction_item.get("instruction_zh")
            or instruction_item.get("instruction_cn")
            or instruction_item.get("instruction_chinese")
            or ""
        ).strip()
        start_ns = _integer(manifest_item.get("start_timestamp_ns"), field="start_timestamp_ns", scene=scene)
        end_ns = _integer(manifest_item.get("end_timestamp_ns"), field="end_timestamp_ns", scene=scene)
        if end_ns <= start_ns:
            raise VisualizationError(f"{game_id} Level {level} {scene}: end timestamp is not after start")
        raw_rows.append(
            (
                scene,
                int(manifest_item.get("order", fallback_order)),
                start_ns,
                end_ns,
                instruction_en,
                instruction_zh,
            )
        )

    raw_rows.sort(key=lambda value: (value[2], value[3], value[1], value[0]))
    first_start_ns = min(value[2] for value in raw_rows)
    final_end_ns = max(value[3] for value in raw_rows)
    if first_start_ns >= 100_000_000_000_000_000 and video_start_ns == 0:
        raise VisualizationError(
            f"{game_id}: manifest uses epoch timestamps but source.video_start_ns is zero"
        )
    timeline_origin_ns = max(first_start_ns, video_start_ns)
    if final_end_ns <= timeline_origin_ns:
        raise VisualizationError(f"{game_id} Level {level}: scene ends before the source video starts")
    seek_s = max(0.0, (timeline_origin_ns - video_start_ns) / NS_PER_S)
    duration_s = (final_end_ns - timeline_origin_ns) / NS_PER_S

    captions: list[CaptionInterval] = []
    previous_end = 0.0
    for scene, order, start_ns, end_ns, instruction_en, instruction_zh in raw_rows:
        start_s = max(0.0, (start_ns - timeline_origin_ns) / NS_PER_S)
        end_s = min(duration_s, (end_ns - timeline_origin_ns) / NS_PER_S)
        if end_s <= start_s:
            continue
        if start_s < previous_end - 1e-6:
            raise VisualizationError(
                f"{game_id} Level {level} {scene}: overlapping instruction intervals "
                f"({start_s:.6f}s < {previous_end:.6f}s)"
            )
        captions.append(
            CaptionInterval(
                scene=scene,
                order=order,
                start_s=start_s,
                end_s=end_s,
                instruction_en=instruction_en,
                instruction_zh=instruction_zh,
            )
        )
        previous_end = end_s

    if not captions:
        raise VisualizationError(f"{game_id} Level {level}: no instruction overlaps the source video")
    return VisualizationJob(
        workdir=workdir,
        game_id=game_id,
        display_name=display_name,
        level=level,
        video_path=video_path,
        video_start_ns=video_start_ns,
        timeline_origin_ns=timeline_origin_ns,
        timeline_end_ns=final_end_ns,
        seek_s=seek_s,
        duration_s=duration_s,
        captions=captions,
    )


def resolve_font_path(font_path: Path | None = None) -> Path:
    if font_path is not None:
        path = font_path.expanduser().resolve()
        if not path.is_file():
            raise VisualizationError(f"font not found: {path}")
        return path
    for candidate in DEFAULT_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    raise VisualizationError("no usable CJK font found; pass --font")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    text = " ".join(str(text).strip().split())
    if not text:
        return []
    tokens = text.split(" ") if " " in text else list(text)
    separator = " " if " " in text else ""
    lines: list[str] = []
    current = ""
    for token in tokens:
        if _text_width(draw, token, font) > max_width:
            if current:
                lines.append(current)
                current = ""
            piece = ""
            for character in token:
                trial = piece + character
                if piece and _text_width(draw, trial, font) > max_width:
                    lines.append(piece)
                    piece = character
                else:
                    piece = trial
            current = piece
            continue
        trial = f"{current}{separator}{token}".strip()
        if current and _text_width(draw, trial, font) > max_width:
            lines.append(current)
            current = token
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _line_step(font: ImageFont.FreeTypeFont) -> int:
    box = font.getbbox("Ag国")
    return max(1, box[3] - box[1]) + 7


def render_caption_image(
    caption: CaptionInterval | None,
    destination: Path,
    *,
    game_name: str,
    level: int,
    width: int,
    height: int,
    font_path: Path,
) -> None:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if caption is None:
        image.save(destination)
        return

    base_body_size = max(17, round(height * 0.038))
    header_size = max(15, round(base_body_size * 0.82))
    minimum_body_size = max(14, round(height * 0.025))
    margin = max(10, round(width * 0.018))
    max_text_width = width - 2 * margin
    max_panel_height = round(height * 0.48)
    scratch = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    while True:
        header_font = ImageFont.truetype(str(font_path), header_size)
        body_font = ImageFont.truetype(str(font_path), base_body_size)
        en_lines = wrap_text(f"EN  {caption.instruction_en}", scratch, body_font, max_text_width)
        zh_lines = wrap_text(f"中文  {caption.instruction_zh}", scratch, body_font, max_text_width)
        panel_height = (
            margin
            + _line_step(header_font)
            + 5
            + len(en_lines) * _line_step(body_font)
            + len(zh_lines) * _line_step(body_font)
            + margin
        )
        if panel_height <= max_panel_height or base_body_size <= minimum_body_size:
            break
        base_body_size -= 1
        header_size = max(13, round(base_body_size * 0.82))

    if panel_height > height:
        raise VisualizationError(f"caption is too tall to render: {caption.scene}")
    draw = ImageDraw.Draw(image, "RGBA")
    y0 = height - panel_height
    draw.rectangle((0, y0, width, height), fill=(0, 0, 0, 175))
    accent = (76, 224, 118, 255) if level == 1 else (76, 205, 235, 255)
    draw.rectangle((0, y0, width, y0 + max(3, height // 240)), fill=accent)
    header = (
        f"{game_name}  ·  Level {level}  ·  {caption.scene}  ·  "
        f"{caption.start_s:.1f}s–{caption.end_s:.1f}s"
    )
    y = y0 + margin
    draw.text((margin, y), header, font=header_font, fill=accent)
    y += _line_step(header_font) + 5
    for line in en_lines:
        draw.text((margin, y), line, font=body_font, fill=(255, 255, 255, 255))
        y += _line_step(body_font)
    for line in zh_lines:
        draw.text((margin, y), line, font=body_font, fill=(255, 244, 178, 255))
        y += _line_step(body_font)
    image.save(destination)


def _cache_key(game_id: str, source_text: str, model: str) -> str:
    value = json.dumps(
        {
            "game_id": game_id,
            "source_text": source_text,
            "model": model,
            "prompt_version": TRANSLATION_PROMPT_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TranslationCache:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._lock = threading.Lock()
        self.items: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise VisualizationError(f"invalid translation cache: {self.path}: {exc}") from exc
            values = raw.get("items") if isinstance(raw, dict) else None
            if not isinstance(values, dict):
                raise VisualizationError(f"invalid translation cache items: {self.path}")
            self.items = {str(key): dict(value) for key, value in values.items() if isinstance(value, dict)}

    def get(self, game_id: str, source_text: str, model: str) -> str:
        key = _cache_key(game_id, source_text, model)
        entry = self.items.get(key) or {}
        if (
            entry.get("game_id") == game_id
            and entry.get("source_text") == source_text
            and entry.get("model") == model
            and entry.get("prompt_version") == TRANSLATION_PROMPT_VERSION
        ):
            return str(entry.get("translation_zh") or "").strip()
        return ""

    def put(self, game_id: str, source_text: str, translation_zh: str, model: str) -> None:
        self.put_many(game_id, {source_text: translation_zh}, model)

    def put_many(self, game_id: str, translations: Mapping[str, str], model: str) -> None:
        with self._lock:
            for source_text, raw_translation in translations.items():
                translation_zh = str(raw_translation).strip()
                if not translation_zh:
                    raise VisualizationError("refusing to cache an empty translation")
                key = _cache_key(game_id, source_text, model)
                self.items[key] = {
                    "game_id": game_id,
                    "source_text": source_text,
                    "translation_zh": translation_zh,
                    "model": model,
                    "prompt_version": TRANSLATION_PROMPT_VERSION,
                    "updated_at": utc_now(),
                }
            self.save_locked()

    def save_locked(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "prompt_version": TRANSLATION_PROMPT_VERSION,
                "items": self.items,
            },
        )


def _extract_json_object(text: str) -> Mapping[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise VisualizationError("translation model did not return a JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VisualizationError("translation model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise VisualizationError("translation model response is not a JSON object")
    return value


def parse_translation_response(text: str, expected_ids: Iterable[str]) -> dict[str, str]:
    expected = set(expected_ids)
    payload = _extract_json_object(text)
    raw = payload.get("translations", payload)
    translations: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            identifier = str(item.get("id") or "")
            value = str(item.get("zh") or item.get("translation") or "").strip()
            if identifier:
                if identifier in translations:
                    raise VisualizationError(f"duplicate translation id: {identifier}")
                translations[identifier] = value
    elif isinstance(raw, dict):
        translations = {str(key): str(value).strip() for key, value in raw.items()}
    else:
        raise VisualizationError("translation response has no translations collection")
    if set(translations) != expected:
        raise VisualizationError(
            f"translation response ids mismatch: missing={sorted(expected - set(translations))}, "
            f"extra={sorted(set(translations) - expected)}"
        )
    for identifier, value in translations.items():
        if not value:
            raise VisualizationError(f"empty translation for {identifier}")
        if not _CJK_RE.search(value):
            raise VisualizationError(f"translation for {identifier} contains no Chinese characters")
    return translations


def request_translation_batch(
    game_name: str,
    entries: Mapping[str, str],
    model: str,
    base_url: str | None,
    timeout: float,
) -> Mapping[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise VisualizationError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    clients = getattr(_CLIENT_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _CLIENT_LOCAL.clients = clients
    client_key = (base_url, api_key)
    client = clients.get(client_key)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        clients[client_key] = client
    prompt = (
        f"Translate the following imperative gameplay instructions for {game_name} into concise, natural "
        "Simplified Chinese. Preserve game-specific proper nouns, button names, directions, counts, and the "
        "imperative meaning. Do not add explanations. Return only one JSON object in this exact shape: "
        '{"translations":[{"id":"t0000","zh":"中文"}]}. Every input id must appear exactly once.\n\n'
        f"Inputs:\n{json.dumps([{'id': key, 'text': value} for key, value in entries.items()], ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    content = response.choices[0].message.content if response.choices else ""
    return parse_translation_response(str(content or ""), entries.keys())


def redact_secret(text: str) -> str:
    value = str(text)
    secret = os.getenv("OPENAI_API_KEY")
    return value.replace(secret, "[REDACTED]") if secret else value


def translate_jobs(
    jobs: Sequence[VisualizationJob],
    cache: TranslationCache,
    *,
    model: str = DEFAULT_TRANSLATION_MODEL,
    base_url: str | None = DEFAULT_BASE_URL,
    batch_size: int = 20,
    concurrency: int = 4,
    timeout: float = 180.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    request_fn: TranslationRequestFn = request_translation_batch,
) -> dict[str, int]:
    if batch_size <= 0 or concurrency <= 0 or max_retries <= 0:
        raise VisualizationError("batch size, concurrency, and max retries must be positive")
    unique: dict[tuple[str, str], tuple[str, str]] = {}
    existing_count = 0
    cache_hits = 0
    for job in jobs:
        for caption in job.captions:
            if caption.instruction_zh:
                existing_count += 1
                continue
            cached = cache.get(job.game_id, caption.instruction_en, model)
            if cached:
                caption.instruction_zh = cached
                cache_hits += 1
            else:
                unique[(job.game_id, caption.instruction_en)] = (job.display_name, caption.instruction_en)

    grouped: dict[str, list[tuple[str, str]]] = {}
    display_names: dict[str, str] = {}
    for (game_id, source_text), (display_name, _) in unique.items():
        grouped.setdefault(game_id, []).append((source_text, display_name))
        display_names[game_id] = display_name

    batches: list[tuple[str, str, dict[str, str]]] = []
    for game_id, values in grouped.items():
        values.sort(key=lambda value: value[0])
        source_texts = [value[0] for value in values]
        for offset in range(0, len(source_texts), batch_size):
            chunk = source_texts[offset : offset + batch_size]
            entries = {f"t{index:04d}": text for index, text in enumerate(chunk)}
            batches.append((game_id, display_names[game_id], entries))

    errors: list[str] = []
    attempts = 0
    attempts_lock = threading.Lock()

    def worker(game_id: str, display_name: str, entries: Mapping[str, str]) -> tuple[str, Mapping[str, str], Mapping[str, str]]:
        nonlocal attempts
        error = ""
        for attempt in range(1, max_retries + 1):
            with attempts_lock:
                attempts += 1
            try:
                translated = request_fn(display_name, entries, model, base_url, timeout)
                if set(translated) != set(entries):
                    raise VisualizationError("translation request returned an incomplete id set")
                for value in translated.values():
                    if not str(value).strip() or not _CJK_RE.search(str(value)):
                        raise VisualizationError("translation request returned invalid Chinese text")
                return game_id, entries, translated
            except Exception as exc:
                error = redact_secret(str(exc))[:1000]
                if attempt < max_retries:
                    time.sleep(max(0.0, retry_delay) * attempt)
        raise VisualizationError(f"{game_id} translation batch failed: {error}")

    if batches:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(worker, game_id, display_name, entries): (game_id, entries)
                for game_id, display_name, entries in batches
            }
            for future in as_completed(future_map):
                game_id, entries = future_map[future]
                try:
                    resolved_game, original, translated = future.result()
                    cache.put_many(
                        resolved_game,
                        {source_text: str(translated[identifier]) for identifier, source_text in original.items()},
                        model,
                    )
                except Exception as exc:
                    errors.append(redact_secret(str(exc)))
    if errors:
        raise VisualizationError("; ".join(errors))

    translated_count = 0
    for job in jobs:
        for caption in job.captions:
            if not caption.instruction_zh:
                caption.instruction_zh = cache.get(job.game_id, caption.instruction_en, model)
                if caption.instruction_zh:
                    translated_count += 1
            if not caption.instruction_zh:
                raise VisualizationError(
                    f"missing Chinese translation after translation pass: {job.game_id} Level {job.level} {caption.scene}"
                )
    return {
        "existing": existing_count,
        "cache_hits": cache_hits,
        "unique_requested": len(unique),
        "batches": len(batches),
        "attempts": attempts,
        "captions_filled": translated_count,
    }


def apply_cached_translations(
    jobs: Sequence[VisualizationJob],
    cache: TranslationCache,
    *,
    model: str = DEFAULT_TRANSLATION_MODEL,
) -> dict[str, int]:
    hits = 0
    missing = 0
    for job in jobs:
        for caption in job.captions:
            if caption.instruction_zh:
                continue
            caption.instruction_zh = cache.get(job.game_id, caption.instruction_en, model)
            if caption.instruction_zh:
                hits += 1
            else:
                missing += 1
    return {"cache_hits": hits, "missing": missing}


def _caption_slices(job: VisualizationJob) -> list[tuple[float, CaptionInterval | None]]:
    slices: list[tuple[float, CaptionInterval | None]] = []
    cursor = 0.0
    for caption in sorted(job.captions, key=lambda value: (value.start_s, value.end_s, value.order)):
        if caption.start_s > cursor + 1e-6:
            slices.append((caption.start_s - cursor, None))
        if caption.start_s < cursor - 1e-6:
            raise VisualizationError(f"overlapping captions while rendering {caption.scene}")
        duration = caption.end_s - caption.start_s
        if duration > 0:
            slices.append((duration, caption))
        cursor = caption.end_s
    if cursor < job.duration_s - 1e-6:
        slices.append((job.duration_s - cursor, None))
    if not slices:
        raise VisualizationError("caption timeline is empty")
    return slices


def _run_command(command: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VisualizationError(f"required executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VisualizationError(f"command timed out: {command[0]}") from exc
    if completed.returncode != 0:
        detail = redact_secret((completed.stderr or completed.stdout or "").strip())[-4000:]
        raise VisualizationError(f"{command[0]} failed with exit {completed.returncode}: {detail}")
    return completed


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    completed = _run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VisualizationError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = list(payload.get("streams") or [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise VisualizationError(f"media has no video stream: {path}")
    raw_duration = (payload.get("format") or {}).get("duration")
    if raw_duration in (None, "", "N/A"):
        stream_durations = []
        for stream in streams:
            try:
                stream_durations.append(float(stream.get("duration")))
            except (TypeError, ValueError):
                continue
        duration_s = max(stream_durations) if stream_durations else None
    else:
        try:
            duration_s = float(raw_duration)
        except (TypeError, ValueError):
            duration_s = None
    return {
        "duration_s": duration_s,
        "video_codec": str(video_stream.get("codec_name") or ""),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def validate_rendered_video(
    output_path: Path,
    *,
    source_path: Path,
    expected_duration_s: float,
    width: int,
    height: int,
    ffprobe: str = "ffprobe",
    tolerance_s: float = 0.25,
) -> dict[str, Any]:
    output = probe_media(output_path, ffprobe=ffprobe)
    source = probe_media(source_path, ffprobe=ffprobe)
    if output["duration_s"] is None:
        raise VisualizationError(f"rendered output has no duration: {output_path}")
    if output["video_codec"] != "h264":
        raise VisualizationError(f"expected H.264 output, got {output['video_codec']}")
    if output["width"] != width or output["height"] != height:
        raise VisualizationError(
            f"unexpected output size {output['width']}x{output['height']}; expected {width}x{height}"
        )
    duration_error = abs(float(output["duration_s"]) - expected_duration_s)
    if duration_error > tolerance_s:
        raise VisualizationError(
            f"output duration differs by {duration_error:.3f}s; expected {expected_duration_s:.3f}s, "
            f"got {output['duration_s']:.3f}s"
        )
    if source["has_audio"] and not output["has_audio"]:
        raise VisualizationError("source has audio but rendered output does not")
    return {**output, "duration_error_s": duration_error, "source_has_audio": source["has_audio"]}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_fingerprint(
    job: VisualizationJob,
    *,
    width: int,
    height: int,
    font_path: Path,
    crf: int,
    preset: str,
) -> str:
    source_stat = job.video_path.stat()
    payload = {
        "game_id": job.game_id,
        "level": job.level,
        "video": str(job.video_path),
        "video_size": source_stat.st_size,
        "video_mtime_ns": source_stat.st_mtime_ns,
        "manifest_sha256": _file_hash(manifest_path(job.workdir, job.level)),
        "instruction_sha256": _file_hash(instruction_path(job.workdir, job.level)),
        "timeline_origin_ns": job.timeline_origin_ns,
        "timeline_end_ns": job.timeline_end_ns,
        "captions": [
            {
                "scene": caption.scene,
                "start_s": round(caption.start_s, 9),
                "end_s": round(caption.end_s, 9),
                "en": caption.instruction_en,
                "zh": caption.instruction_zh,
            }
            for caption in job.captions
        ],
        "width": width,
        "height": height,
        "font": str(font_path),
        "font_mtime_ns": font_path.stat().st_mtime_ns,
        "crf": crf,
        "preset": preset,
        "renderer_version": 2,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_concat_file(
    path: Path,
    image_paths: Sequence[Path],
    durations: Sequence[float],
    *,
    terminal_image: Path | None = None,
) -> None:
    if len(image_paths) != len(durations) or not image_paths:
        raise VisualizationError("caption concat inputs are inconsistent")
    lines = ["ffconcat version 1.0"]
    for image_path, duration in zip(image_paths, durations):
        if duration <= 0:
            continue
        lines.append(f"file '{image_path.as_posix()}'")
        lines.append(f"duration {duration:.9f}")
    # With VFR output, ffmpeg may de-duplicate a repeated final caption image.
    # A distinct transparent terminal frame gives the last caption packet a
    # real end timestamp instead of letting the overlay disappear early.
    lines.append(f"file '{(terminal_image or image_paths[-1]).as_posix()}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_job(
    job: VisualizationJob,
    output_path: Path,
    *,
    font_path: Path,
    width: int = 1280,
    height: int = 720,
    crf: int = 20,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
    resume: bool = False,
    keep_temp: bool = False,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise VisualizationError("output width and height must be positive even numbers")
    for caption in job.captions:
        if not caption.instruction_zh:
            raise VisualizationError(f"missing Chinese translation: {job.game_id} Level {job.level} {caption.scene}")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".meta.json")
    fingerprint = job_fingerprint(
        job,
        width=width,
        height=height,
        font_path=font_path,
        crf=crf,
        preset=preset,
    )
    if output_path.exists():
        if resume and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if metadata.get("fingerprint") == fingerprint:
                validation = validate_rendered_video(
                    output_path,
                    source_path=job.video_path,
                    expected_duration_s=job.duration_s,
                    width=width,
                    height=height,
                    ffprobe=ffprobe,
                )
                return {
                    "status": "skipped",
                    "output": str(output_path),
                    "metadata": str(metadata_path),
                    "caption_count": len(job.captions),
                    "duration_s": job.duration_s,
                    "validation": validation,
                }
        if not overwrite:
            raise VisualizationError(f"output already exists; use --resume or --overwrite: {output_path}")

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}.captions.", dir=output_path.parent))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.captions.", dir=output_path.parent)
        temp_root = Path(temp_dir_obj.name)
    started = time.monotonic()
    try:
        slices = _caption_slices(job)
        images: list[Path] = []
        durations: list[float] = []
        blank_path: Path | None = None
        for index, (duration, caption) in enumerate(slices):
            if caption is None and blank_path is not None:
                image_path = blank_path
            else:
                image_path = temp_root / f"caption_{index:06d}.png"
                render_caption_image(
                    caption,
                    image_path,
                    game_name=job.display_name,
                    level=job.level,
                    width=width,
                    height=height,
                    font_path=font_path,
                )
                if caption is None:
                    blank_path = image_path
            images.append(image_path)
            durations.append(duration)
        terminal_path = temp_root / "caption_terminal.png"
        render_caption_image(
            None,
            terminal_path,
            game_name=job.display_name,
            level=job.level,
            width=width,
            height=height,
            font_path=font_path,
        )
        concat_path = temp_root / "captions.ffconcat"
        _write_concat_file(concat_path, images, durations, terminal_image=terminal_path)
        caption_track = temp_root / "caption_track.mov"
        _run_command(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-fps_mode",
                "vfr",
                "-c:v",
                "qtrle",
                "-pix_fmt",
                "argb",
                str(caption_track),
            ],
            timeout=max(120.0, job.duration_s * 2.0),
        )
        temp_output = output_path.parent / f".{output_path.stem}.{os.getpid()}.tmp.mp4"
        temp_output.unlink(missing_ok=True)
        scale_filter = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            "setpts=PTS-STARTPTS[base];"
            "[1:v]format=argb[caption];"
            # Some captured sources finish their filtered timeline slightly
            # before the annotation timeline because of timestamp
            # discontinuities.  Add a bounded cloned tail before the explicit
            # -t cutoff so the output still reaches the manifest end time.
            "[base][caption]overlay=0:0:eof_action=pass:repeatlast=1:shortest=0,"
            "tpad=stop_mode=clone:stop_duration=2,format=yuv420p[v]"
        )
        try:
            _run_command(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{job.seek_s:.9f}",
                    "-i",
                    str(job.video_path),
                    "-i",
                    str(caption_track),
                    "-filter_complex",
                    scale_filter,
                    "-map",
                    "[v]",
                    "-map",
                    "0:a?",
                    "-t",
                    f"{job.duration_s:.9f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-movflags",
                    "+faststart",
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                    str(temp_output),
                ],
                timeout=max(600.0, job.duration_s * 12.0),
            )
            validation = validate_rendered_video(
                temp_output,
                source_path=job.video_path,
                expected_duration_s=job.duration_s,
                width=width,
                height=height,
                ffprobe=ffprobe,
            )
            os.replace(temp_output, output_path)
        finally:
            temp_output.unlink(missing_ok=True)
        elapsed_s = time.monotonic() - started
        metadata = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "game_id": job.game_id,
            "level": job.level,
            "caption_count": len(job.captions),
            "timeline_origin_ns": job.timeline_origin_ns,
            "timeline_end_ns": job.timeline_end_ns,
            "duration_s": job.duration_s,
            "output": str(output_path),
            "width": width,
            "height": height,
            "video_codec": "h264",
            "audio_codec": "aac" if validation["has_audio"] else None,
            "validation": validation,
            "elapsed_s": elapsed_s,
            "created_at": utc_now(),
        }
        atomic_write_json(metadata_path, metadata)
        return {
            "status": "ok",
            "output": str(output_path),
            "metadata": str(metadata_path),
            "caption_count": len(job.captions),
            "duration_s": job.duration_s,
            "elapsed_s": elapsed_s,
            "validation": validation,
            "temporary_assets": str(temp_root) if keep_temp else None,
        }
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def output_filename(level: int, preview_seconds: float | None = None) -> str:
    if preview_seconds is None:
        return f"level_{level}_bilingual.mp4"
    compact = f"{preview_seconds:g}".replace(".", "p")
    return f"level_{level}_bilingual_preview_{compact}s.mp4"


def ensure_tools(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
    for executable in (ffmpeg, ffprobe):
        if shutil.which(executable) is None:
            raise VisualizationError(f"required executable not found: {executable}")
