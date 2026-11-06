#!/usr/bin/env python3
"""Render schema-v2 Level 1/2 instructions as bilingual gameplay videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotator.core.schema import atomic_write_json
from annotator.core.visualization import (
    DEFAULT_TRANSLATION_MODEL,
    GAME_ORDER,
    DEFAULT_BASE_URL,
    TranslationCache,
    VisualizationError,
    apply_cached_translations,
    discover_game_workdirs,
    ensure_tools,
    load_visualization_job,
    output_filename,
    redact_secret,
    render_job,
    resolve_font_path,
    translate_jobs,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "test_outputs" / "unified_eight_games"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render all schema-v2 Level 1/2 instructions over the original gameplay video, "
            "with English and Simplified Chinese captions."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--game",
        action="append",
        choices=GAME_ORDER,
        help="Game id to render; repeat for multiple games. Defaults to every discovered game.",
    )
    parser.add_argument("--levels", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--output-subdir", default="instruction_videos")
    parser.add_argument("--translate", action="store_true", help="Translate missing Chinese text via an OpenAI-compatible API.")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--translation-batch-size", type=int, default=20)
    parser.add_argument("--translation-concurrency", type=int, default=4)
    parser.add_argument("--translation-timeout", type=float, default=180.0)
    parser.add_argument("--translation-max-retries", type=int, default=3)
    parser.add_argument("--translation-retry-delay", type=float, default=2.0)
    parser.add_argument("--translation-cache", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--preview-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def _summary_skeleton(args: argparse.Namespace, input_root: Path, font_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "input_root": str(input_root),
        "settings": {
            "levels": list(dict.fromkeys(args.levels)),
            "translate": bool(args.translate),
            "translation_model": args.translation_model,
            "base_url": args.base_url,
            "translation_batch_size": args.translation_batch_size,
            "translation_concurrency": args.translation_concurrency,
            "width": args.width,
            "height": args.height,
            "font": str(font_path),
            "crf": args.crf,
            "preset": args.preset,
            "preview_seconds": args.preview_seconds,
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
            "dry_run": bool(args.dry_run),
        },
        "translation": {},
        "games": {},
        "errors": [],
    }


def _record_error(summary: dict[str, Any], message: str) -> None:
    summary["errors"].append(redact_secret(message)[:2000])


def _finish_summary(summary: dict[str, Any]) -> None:
    summary["completed_at"] = utc_now()
    summary["status"] = "error" if summary["errors"] else "ok"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.preview_seconds is not None and args.preview_seconds <= 0:
        parser.error("--preview-seconds must be positive")
    if args.crf < 0 or args.crf > 51:
        parser.error("--crf must be between 0 and 51")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even numbers")

    input_root = args.input_root.expanduser().resolve()
    try:
        ensure_tools(args.ffmpeg, args.ffprobe)
        font_path = resolve_font_path(args.font)
        workdirs = discover_game_workdirs(input_root, args.game)
    except Exception as exc:
        print(f"ERROR: {redact_secret(str(exc))}", file=sys.stderr)
        return 1

    summary = _summary_skeleton(args, input_root, font_path)
    jobs = []
    levels = list(dict.fromkeys(args.levels))
    for workdir in workdirs:
        game_entry = summary["games"].setdefault(workdir.name, {"workdir": str(workdir), "levels": {}})
        for level in levels:
            try:
                job = load_visualization_job(workdir, level).limited(args.preview_seconds)
                jobs.append(job)
                game_entry["levels"][str(level)] = {
                    "status": "planned",
                    "caption_count": len(job.captions),
                    "duration_s": job.duration_s,
                    "source_video": str(job.video_path),
                    "seek_s": job.seek_s,
                }
            except Exception as exc:
                message = f"{workdir.name} Level {level}: {exc}"
                game_entry["levels"][str(level)] = {"status": "error", "error": redact_secret(str(exc))}
                _record_error(summary, message)

    cache_path = (
        args.translation_cache.expanduser().resolve()
        if args.translation_cache
        else input_root / "instruction_translation_cache.json"
    )
    try:
        cache = TranslationCache(cache_path)
    except Exception as exc:
        _record_error(summary, str(exc))
        cache = None

    translation_ready = cache is not None
    if cache is not None:
        if args.dry_run:
            cache_result = apply_cached_translations(jobs, cache, model=args.translation_model)
            summary["translation"] = {
                **cache_result,
                "cache": str(cache_path),
                "would_call_api": bool(args.translate and cache_result["missing"]),
            }
            if not args.translate and cache_result["missing"]:
                _record_error(
                    summary,
                    f"{cache_result['missing']} captions have no Chinese translation; rerun with --translate",
                )
            _finish_summary(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if summary["errors"] else 0

        try:
            if args.translate:
                if not os.getenv("OPENAI_API_KEY"):
                    raise VisualizationError("OPENAI_API_KEY is not set; translation cannot start")
                summary["translation"] = {
                    **translate_jobs(
                        jobs,
                        cache,
                        model=args.translation_model,
                        base_url=args.base_url,
                        batch_size=args.translation_batch_size,
                        concurrency=args.translation_concurrency,
                        timeout=args.translation_timeout,
                        max_retries=args.translation_max_retries,
                        retry_delay=args.translation_retry_delay,
                    ),
                    "cache": str(cache_path),
                }
            else:
                cache_result = apply_cached_translations(jobs, cache, model=args.translation_model)
                summary["translation"] = {**cache_result, "cache": str(cache_path)}
                if cache_result["missing"]:
                    raise VisualizationError(
                        f"{cache_result['missing']} captions have no Chinese translation; rerun with --translate"
                    )
        except Exception as exc:
            _record_error(summary, f"translation: {exc}")
            translation_ready = False

    if translation_ready:
        for job in jobs:
            level_entry = summary["games"][job.workdir.name]["levels"][str(job.level)]
            output_path = (
                job.workdir
                / args.output_subdir
                / output_filename(job.level, args.preview_seconds)
            )
            started = time.monotonic()
            try:
                result = render_job(
                    job,
                    output_path,
                    font_path=font_path,
                    width=args.width,
                    height=args.height,
                    crf=args.crf,
                    preset=args.preset,
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    keep_temp=args.keep_temp,
                )
                level_entry.update(result)
            except Exception as exc:
                level_entry.update(
                    {
                        "status": "error",
                        "error": redact_secret(str(exc))[:2000],
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                _record_error(summary, f"{job.game_id} Level {job.level}: {exc}")

    _finish_summary(summary)
    summary_path = input_root / "visualization_summary.json"
    try:
        atomic_write_json(summary_path, summary)
    except Exception as exc:
        print(f"ERROR: could not write summary: {redact_secret(str(exc))}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}, ensure_ascii=False))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
