"""Build commands for the supported legacy Stage 1 backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from .profiles import GameProfile


def _resolved_defaults(profile: GameProfile, args: Any) -> Dict[str, Any]:
    defaults = profile.stage_1b_defaults
    return {
        "model": args.refine_model or str(defaults.get("model", "gemini-2.5-flash")),
        "frames": (
            args.refine_frames_per_segment
            if args.refine_frames_per_segment is not None
            else int(defaults.get("frames_per_segment", 5))
        ),
        "concurrency": (
            args.refine_concurrency
            if args.refine_concurrency is not None
            else int(defaults.get("concurrency", 8))
        ),
        "image_detail": args.refine_image_detail or str(defaults.get("image_detail", "low")),
        "base_url": args.base_url or defaults.get("base_url"),
    }


def _segment_minimum(profile: GameProfile, args: Any) -> float | None:
    if args.segment_min_duration_s is not None:
        return args.segment_min_duration_s
    value = profile.stage_1a_defaults.get("min_duration_s")
    return float(value) if value is not None else None


def _scenedetect_segment_command(
    *,
    profile: GameProfile,
    args: Any,
    video_name: str,
    segments: Path,
    minimum: float | None,
) -> List[str]:
    """Stage 1a via visual shot changes, shared by every game.

    The taxonomy-driven flags (``--merge-tier2`` and friends) have no meaning
    here because this backend emits unlabelled spans for Stage 1b to resolve.
    """

    command = [
        sys.executable,
        "-m",
        "annotator.cli.1a_scenedetect_segment",
        "--game",
        profile.game_id,
        "--annotations",
        str(args.annotations),
        "--video",
        str(args.video),
        "--out",
        str(segments),
        "--video-name",
        video_name,
        "--threshold",
        str(args.scenedetect_threshold),
    ]
    if minimum is not None:
        command.extend(["--min-duration-s", str(minimum)])
    if args.scenedetect_min_scene_frames is not None:
        command.extend(["--min-scene-frames", str(args.scenedetect_min_scene_frames)])
    if args.scenedetect_frame_skip:
        command.extend(["--frame-skip", str(args.scenedetect_frame_skip)])
    if args.video_start_ns is not None:
        command.extend(["--video-start-ns", str(args.video_start_ns)])
    return command


def build_stage1_commands(
    *,
    profile: GameProfile,
    args: Any,
    video_name: str,
    segments: Path,
    refined: Path,
    cache: Path,
) -> Dict[str, List[str]]:
    if profile.stage_1_backend != "legacy_cli":
        raise ValueError(f"unsupported Stage 1 backend: {profile.stage_1_backend}")

    resolved = _resolved_defaults(profile, args)
    minimum = _segment_minimum(profile, args)
    if getattr(args, "segmenter", "action") == "scenedetect":
        segment = _scenedetect_segment_command(
            profile=profile,
            args=args,
            video_name=video_name,
            segments=segments,
            minimum=minimum,
        )
    else:
        segment = [
            sys.executable,
            str(profile.stage_1a_script),
            str(args.annotations),
            "--video",
            video_name,
            "--out",
            str(segments),
        ]
        if minimum is not None:
            segment.extend(["--min-duration-s", str(minimum)])
        default_groups = list(profile.stage_1a_defaults.get("merge_tier2_groups") or [])
        groups = [] if args.no_default_tier2_merge else default_groups
        groups.extend(args.merge_tier2)
        for group in dict.fromkeys(str(value) for value in groups):
            segment.extend(["--merge-tier2", group])
        if args.no_default_tier2_merge:
            segment.append(profile.segment_disable_default_flag or "--no-default-tier2-merge")

    subdivide = (
        args.refine_subdivide_frames_per_segment
        if args.refine_subdivide_frames_per_segment is not None
        else int(profile.stage_1b_defaults.get("subdivide_frames_per_segment", 10))
    )
    refine = [
        sys.executable,
        str(profile.stage_1b_script),
        str(segments),
        "--video",
        str(args.video),
        "--out",
        str(refined),
        "--model",
        str(resolved["model"]),
        "--frames-per-segment",
        str(resolved["frames"]),
        "--subdivide-frames-per-segment",
        str(subdivide),
        "--image-detail",
        str(resolved["image_detail"]),
        "--concurrency",
        str(resolved["concurrency"]),
        "--cache-dir",
        str(cache),
    ]
    if args.refine_limit is not None:
        refine.extend(["--limit", str(args.refine_limit)])
    if resolved["base_url"]:
        refine.extend(["--base-url", str(resolved["base_url"])])
    if args.video_start_ns is not None and profile.refine_supports_video_start_ns:
        refine.extend(["--video-start-ns", str(args.video_start_ns)])
    return {"segment": segment, "refine": refine}
