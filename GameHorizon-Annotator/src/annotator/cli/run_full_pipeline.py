#!/usr/bin/env python3
"""Run the unified end-to-end annotation pipeline for one game scene."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from annotator.core.profiles import load_profile, supported_games
from annotator.core.frames import load_annotation_frames, partition_segment_timeline
from annotator.core.local_cache import ensure_local_file_copy
from annotator.core.schema import atomic_write_json
from annotator.core.scenedetect_segment import DEFAULT_THRESHOLD as SCENEDETECT_THRESHOLD
from annotator.core.stage1 import build_stage1_commands


STAGES = ["segment", "refine", "level1_split", "level1", "merge12", "level2", "merge23", "level3"]


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--video-name")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--video-start-ns", type=int)
    parser.add_argument(
        "--scene-start-offset-ns",
        type=int,
        help="video-relative position of the first annotation frame",
    )
    parser.add_argument("--fps", type=float)

    parser.add_argument("--segment-min-duration-s", type=float)
    parser.add_argument("--merge-tier2", action="append", default=[])
    parser.add_argument("--no-default-tier2-merge", action="store_true")

    parser.add_argument(
        "--segmenter",
        choices=("action", "scenedetect"),
        default="action",
        help=(
            "Stage 1a backend. 'action' (default) cuts on this game's keybind "
            "taxonomy and labels every span; 'scenedetect' cuts on visual shot "
            "changes and leaves labelling to Stage 1b."
        ),
    )
    parser.add_argument("--scenedetect-threshold", type=float, default=SCENEDETECT_THRESHOLD)
    parser.add_argument("--scenedetect-min-scene-frames", type=int)
    parser.add_argument("--scenedetect-frame-skip", type=int, default=0)

    parser.add_argument("--refine-model")
    parser.add_argument("--refine-frames-per-segment", type=int)
    parser.add_argument("--refine-subdivide-frames-per-segment", type=int)
    parser.add_argument("--refine-concurrency", type=int)
    parser.add_argument("--refine-cache-dir", type=Path)
    parser.add_argument("--refine-image-detail", choices=("low", "high", "auto"))
    parser.add_argument("--refine-limit", type=int)

    parser.add_argument(
        "--level1-sample-max-frames",
        type=int,
        default=0,
        help="Frames retained by 1c per segment; 0 preserves the complete legacy frame span.",
    )
    parser.add_argument(
        "--level1-sparse-batch-size",
        type=int,
        default=100,
        help="Sampled frames decoded per exact-PTS FFmpeg batch (default: 100).",
    )
    parser.add_argument("--level1-sparse-jobs", type=int, default=_env_int("OWA_LEVEL1_SPARSE_JOBS", 1))
    parser.add_argument("--level1-ffmpeg-threads", type=int, default=_env_int("OWA_LEVEL1_FFMPEG_THREADS", 0))
    parser.add_argument("--local-video-cache-dir", type=Path, default=_env_path("OWA_LOCAL_VIDEO_CACHE_DIR"))
    parser.add_argument("--local-frame-cache-dir", type=Path, default=_env_path("OWA_LOCAL_FRAME_CACHE_DIR"))
    parser.add_argument("--video-pts-cache-dir", type=Path, default=_env_path("OWA_VIDEO_PTS_CACHE_DIR"))
    parser.add_argument("--level1-skip-label", action="append", default=[])
    parser.add_argument("--level1-min-frames", type=int, default=1)
    parser.add_argument("--image-ext", choices=(".jpg", ".jpeg", ".png"), default=".jpg")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--frame-max-width", type=int, default=1280)

    parser.add_argument("--instruction-model", default="gemini-3.5-flash")
    parser.add_argument("--instruction-concurrency", type=int, default=8)
    parser.add_argument("--api-max-frames", type=int, default=30)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Sampling temperature for the instruction stages; unset leaves the "
            "endpoint default in place, which means wording varies between runs. "
            "Pass 0 for repeatable output."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--level1-limit", type=int)
    parser.add_argument("--level2-limit", type=int)
    parser.add_argument("--level3-limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--level2-min-seconds", type=float, default=60.0)
    parser.add_argument("--level2-max-seconds", type=float, default=120.0)
    parser.add_argument("--merge-seed", type=int, default=42)
    parser.add_argument("--level3-min-seconds", type=float, default=240.0)
    parser.add_argument("--level3-max-seconds", type=float, default=480.0)
    parser.add_argument("--materialize", choices=("none", "symlink", "hardlink", "copy"), default="none")

    parser.add_argument(
        "--cleanup",
        choices=("none", "cache", "frames", "all"),
        default="none",
        help=(
            "Discard intermediates once the run finishes. 'cache' drops the "
            "Stage 1b response cache; 'frames' also drops the decoded frames "
            "(the bulk of the output); 'all' additionally drops the Stage 1 "
            "segment files, leaving the instruction files and manifests. "
            "Default keeps everything."
        ),
    )

    parser.add_argument("--skip", action="append", default=[], choices=STAGES)
    parser.add_argument("--only", action="append", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: Sequence[str], *, env: Dict[str, str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        raise SystemExit(f"stage failed with exit code {completed.returncode}: {command[1]}")


def _should_run(stage: str, args: argparse.Namespace) -> bool:
    return stage in args.only if args.only else stage not in args.skip


def _normalize_refined_coverage(args: argparse.Namespace) -> None:
    """Make the persisted 1b timeline a gap-free, non-overlapping partition."""

    video_name = args.video_name or args.video.stem
    refined_path = args.workdir / f"{video_name}_segments_refined.json"
    if not refined_path.is_file():
        raise SystemExit(f"refined output not found after Stage 1b: {refined_path}")
    _, annotation_frames, frame_by_index = load_annotation_frames(args.annotations)
    payload = json.loads(refined_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        raise SystemExit(f"refined output has no segments: {refined_path}")

    gaps_fixed = overlaps_fixed = 0
    for left, right in zip(segments, segments[1:]):
        try:
            delta = int(right["start_frame"]) - int(left["end_frame"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if delta > 0:
            gaps_fixed += delta
        elif delta < 0:
            overlaps_fixed += -delta
    normalized = partition_segment_timeline(segments, frame_by_index)
    if normalized:
        expected_start = min(frame_by_index) if frame_by_index else int(normalized[0]["start_frame"])
        expected_end = max(frame_by_index) if frame_by_index else int(normalized[-1]["end_frame"])
        if int(normalized[0]["start_frame"]) != expected_start:
            raise SystemExit(
                f"Stage 1b starts at frame {normalized[0].get('start_frame')}, expected {expected_start}"
            )
        if int(normalized[-1]["end_frame"]) != expected_end:
            raise SystemExit(
                f"Stage 1b ends at frame {normalized[-1].get('end_frame')}, expected {expected_end}"
            )
        for left, right in zip(normalized, normalized[1:]):
            if int(left["end_frame"]) + 1 != int(right["start_frame"]):
                raise SystemExit(
                    "Stage 1b coverage normalization could not create a contiguous partition"
                )
    payload["segments"] = normalized
    payload["coverage_normalization"] = {
        "version": 1,
        "annotations": str(args.annotations.resolve()),
        "frame_count": len(annotation_frames),
        "gaps_fixed_frames": gaps_fixed,
        "overlaps_fixed_frames": overlaps_fixed,
        "partitioned": True,
    }
    atomic_write_json(refined_path, payload)
    print(
        f"[stage1 coverage] segments={len(normalized)} gaps_fixed={gaps_fixed} "
        f"overlaps_fixed={overlaps_fixed}",
        flush=True,
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cleanup_workdir(args: argparse.Namespace) -> None:
    """Delete intermediates the finished run no longer needs.

    Deliberately conservative: the Level-2 and Level-3 manifests address their
    frames by path inside ``frames_level_1``, so dropping those frames keeps
    the instruction files valid but makes re-running an instruction stage
    require Stage 1c again.
    """

    mode = args.cleanup
    if mode == "none" or args.dry_run:
        return
    if not _should_run(STAGES[-1], args):
        print(
            f"\n=== cleanup: skipped (--cleanup {mode} only applies to a run that "
            f"reaches '{STAGES[-1]}') ===",
            flush=True,
        )
        return

    video_name = args.video_name or args.video.stem
    targets: List[Path] = [args.refine_cache_dir or (args.workdir / ".gemini_cache")]
    if mode in ("frames", "all"):
        targets.extend(
            sorted(
                scene
                for level in (1, 2, 3)
                for scene in (args.workdir / f"frames_level_{level}").glob("scene_*")
            )
        )
    if mode == "all":
        targets.extend(
            [
                args.workdir / f"{video_name}_segments.json",
                args.workdir / f"{video_name}_segments_refined.json",
            ]
        )

    print(f"\n=== cleanup ({mode}) ===", flush=True)
    freed = 0
    removed = 0
    for target in targets:
        if target.is_dir():
            freed += _directory_bytes(target)
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            freed += target.stat().st_size
            target.unlink(missing_ok=True)
        else:
            continue
        removed += 1
    print(
        f"removed {removed} item(s), freed {freed / 1024 / 1024:.1f} MiB; "
        f"kept {_directory_bytes(args.workdir) / 1024 / 1024:.1f} MiB in {args.workdir}",
        flush=True,
    )


def _instruction_command(
    script: str,
    level: int,
    args: argparse.Namespace,
    *,
    base_url: str | None,
) -> List[str]:
    command = [
        sys.executable, "-m", f"annotator.cli.{Path(script).stem}",
        "--game", args.game,
        "--workdir", str(args.workdir),
        "--annotations-json", str(args.annotations),
        "--model", args.instruction_model,
        "--timeout", str(args.timeout),
        "--concurrency", str(args.instruction_concurrency),
        "--api-max-frames", str(args.api_max_frames),
        "--max-retries", str(args.max_retries),
        "--retry-delay", str(args.retry_delay),
    ]
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if base_url:
        command.extend(["--base-url", base_url])
    if args.skip_existing:
        command.append("--skip-existing")
    limit = {1: args.level1_limit, 2: args.level2_limit, 3: args.level3_limit}[level]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def build_commands(args: argparse.Namespace) -> Dict[str, List[str]]:
    profile = load_profile(args.game)
    video_name = args.video_name or args.video.stem
    segments = args.workdir / f"{video_name}_segments.json"
    refined = args.workdir / f"{video_name}_segments_refined.json"
    cache = args.refine_cache_dir or (args.workdir / ".gemini_cache")
    base_url = args.base_url or profile.stage_1b_defaults.get("base_url")
    stage1 = build_stage1_commands(
        profile=profile,
        args=args,
        video_name=video_name,
        segments=segments,
        refined=refined,
        cache=cache,
    )

    level1_split = [
        sys.executable, "-m", "annotator.cli.1c_build_frames_level1",
        "--game", args.game,
        "--segments-json", str(refined),
        "--annotations-json", str(args.annotations),
        "--workdir", str(args.workdir),
        "--video", str(args.video),
        "--sample-max-frames", str(args.level1_sample_max_frames),
        "--sparse-batch-size", str(args.level1_sparse_batch_size),
        "--sparse-jobs", str(getattr(args, "level1_sparse_jobs", 1)),
        "--ffmpeg-threads", str(getattr(args, "level1_ffmpeg_threads", 0)),
        "--min-frames", str(args.level1_min_frames),
        "--image-ext", args.image_ext,
        "--jpeg-quality", str(args.jpeg_quality),
        "--max-width", str(args.frame_max_width),
    ]
    source_video = getattr(args, "source_video", None)
    if source_video is not None and Path(source_video) != Path(args.video):
        level1_split.extend(["--source-video", str(source_video)])
    local_frame_cache_dir = getattr(args, "local_frame_cache_dir", None)
    if local_frame_cache_dir is not None:
        level1_split.extend(["--sparse-cache-dir", str(local_frame_cache_dir)])
    video_pts_cache_dir = getattr(args, "video_pts_cache_dir", None)
    if video_pts_cache_dir is not None:
        level1_split.extend(["--pts-cache-dir", str(video_pts_cache_dir)])

    if args.frames_dir:
        level1_split.extend(["--frames-dir", str(args.frames_dir)])
    if args.clips_dir:
        level1_split.extend(["--clips-dir", str(args.clips_dir)])
    if args.video_start_ns is not None:
        level1_split.extend(["--video-start-ns", str(args.video_start_ns)])
    if args.scene_start_offset_ns is not None:
        level1_split.extend(
            ["--scene-start-offset-ns", str(args.scene_start_offset_ns)]
        )
    if args.fps is not None:
        level1_split.extend(["--fps", str(args.fps)])
    for label in args.level1_skip_label:
        level1_split.extend(["--skip-label", label])

    return {
        "segment": stage1["segment"],
        "refine": stage1["refine"],
        "level1_split": level1_split,
        "level1": _instruction_command(
            "2a_instruct_level1_with_action.py", 1, args, base_url=base_url
        ),
        "merge12": [
            sys.executable, "-m", "annotator.cli.2b_merge_level1to2",
            "--game", args.game, "--workdir", str(args.workdir),
            "--min-seconds", str(args.level2_min_seconds),
            "--max-seconds", str(args.level2_max_seconds),
            "--seed", str(args.merge_seed), "--materialize", args.materialize,
        ],
        "level2": _instruction_command(
            "3a_instruct_level2_with_action.py", 2, args, base_url=base_url
        ),
        "merge23": [
            sys.executable, "-m", "annotator.cli.3b_merge_level2to3",
            "--game", args.game, "--workdir", str(args.workdir),
            "--min-seconds", str(args.level3_min_seconds),
            "--max-seconds", str(args.level3_max_seconds),
            "--model", args.instruction_model,
            "--timeout", str(args.timeout),
            "--api-max-frames", str(args.api_max_frames),
            "--materialize", args.materialize,
            *(["--temperature", str(args.temperature)] if args.temperature is not None else []),
            *(["--base-url", str(base_url)] if base_url else []),
        ],
        "level3": _instruction_command(
            "4a_instruct_level3.py", 3, args, base_url=base_url
        ),
    }


def main() -> None:
    args = parse_args()
    for path, label in ((args.video, "video"), (args.annotations, "annotations")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    if args.level1_sparse_jobs < 1:
        raise SystemExit("--level1-sparse-jobs must be >= 1")
    if args.level1_ffmpeg_threads < 0:
        raise SystemExit("--level1-ffmpeg-threads must be >= 0")
    if not args.dry_run:
        args.workdir.mkdir(parents=True, exist_ok=True)

    args.source_video = args.video
    if args.video_name is None:
        args.video_name = args.source_video.stem
    needs_local_video = any(
        _should_run(stage, args) for stage in ("segment", "refine", "level1_split")
    )
    if not args.dry_run and args.local_video_cache_dir is not None and needs_local_video:
        try:
            args.video = ensure_local_file_copy(args.source_video, args.local_video_cache_dir)
        except (OSError, TimeoutError) as exc:
            print(
                f"[local-video-cache] fallback_to_source type={type(exc).__name__}",
                flush=True,
            )
            args.video = args.source_video

    env = os.environ.copy()
    if args.api_key:
        env["OPENAI_API_KEY"] = args.api_key
    try:
        commands = build_commands(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    profile = load_profile(args.game)
    resolved_base_url = args.base_url or profile.stage_1b_defaults.get("base_url")
    if resolved_base_url:
        env["OPENAI_BASE_URL"] = str(resolved_base_url)
    for stage in STAGES:
        if _should_run(stage, args):
            print(f"\n=== {stage} ===", flush=True)
            _run(commands[stage], env=env, dry_run=args.dry_run)
            if stage == "refine" and not args.dry_run:
                _normalize_refined_coverage(args)
        else:
            print(f"\n=== {stage}: skipped ===", flush=True)
    _cleanup_workdir(args)


if __name__ == "__main__":
    main()
