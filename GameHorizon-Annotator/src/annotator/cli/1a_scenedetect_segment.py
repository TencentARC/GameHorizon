#!/usr/bin/env python3
"""Cut the Stage 1a timeline on visual shot changes instead of keybind rules."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotator.core.profiles import load_profile, supported_games
from annotator.core.scenedetect_segment import (
    DEFAULT_MIN_DURATION_S,
    DEFAULT_THRESHOLD,
    SceneDetectConfig,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"ContentDetector threshold; lower cuts more (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=DEFAULT_MIN_DURATION_S,
        help=f"Shortest segment to keep, in seconds (default: {DEFAULT_MIN_DURATION_S})",
    )
    parser.add_argument(
        "--min-scene-frames",
        type=int,
        help="Detector-side floor in video frames (default: one second at the annotation fps)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Frames skipped between detector samples; trades accuracy for speed",
    )
    parser.add_argument("--video-backend", default="opencv")
    parser.add_argument(
        "--video-start-ns",
        type=int,
        help="Override the video start timestamp instead of probing creation_time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.video, "video"), (args.annotations, "annotations")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    run(
        profile=load_profile(args.game),
        annotations=args.annotations,
        video=args.video,
        out=args.out,
        video_name=args.video_name,
        config=SceneDetectConfig(
            threshold=args.threshold,
            min_duration_s=args.min_duration_s,
            min_scene_frames=args.min_scene_frames,
            frame_skip=args.frame_skip,
            backend=args.video_backend,
        ),
        video_start_ns=args.video_start_ns,
    )


if __name__ == "__main__":
    main()
