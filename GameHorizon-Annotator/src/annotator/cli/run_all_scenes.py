#!/usr/bin/env python3
"""Run the unified pipeline sequentially over per-scene annotation JSON files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from annotator.core.profiles import supported_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--annotations-glob", default="*_scene_*_annotations_per_frame.json")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--video-start-ns", type=int)
    parser.add_argument("--scene-start-offset-ns", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--min-seconds", type=float, default=0.0)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scene_metadata(path: Path) -> tuple[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", []) if isinstance(data, dict) else []
    name = str(data.get("scene_name") or path.stem.replace("_annotations_per_frame", "")) if isinstance(data, dict) else path.stem
    duration = 0.0
    if frames:
        try:
            start = int(frames[0]["timestamp_ns"])
            end = int(frames[-1].get("next_frame_timestamp_ns", frames[-1]["timestamp_ns"]))
            duration = max(0.0, (end - start) / 1_000_000_000)
        except (KeyError, TypeError, ValueError):
            pass
    return name, duration


def main() -> None:
    args = parse_args()
    if args.scene_start_offset_ns is not None and len(args.scene) != 1:
        raise SystemExit(
            "--scene-start-offset-ns may only be used with exactly one explicit --scene"
        )
    files = sorted(args.annotations_dir.glob(args.annotations_glob))
    if not files:
        raise SystemExit(f"no annotations matched {args.annotations_glob!r}")
    failures = []
    for path in files:
        scene, duration = scene_metadata(path)
        if args.scene and scene not in args.scene:
            continue
        if duration < args.min_seconds:
            print(f"[skip] {scene}: {duration:.2f}s < {args.min_seconds:.2f}s")
            continue
        command = [
            sys.executable, "-m", "annotator.cli.run_full_pipeline",
            "--game", args.game,
            "--video", str(args.video),
            "--annotations", str(path),
            "--workdir", str(args.work_root / scene),
            "--video-name", scene,
        ]
        if args.clips_dir:
            command.extend(["--clips-dir", str(args.clips_dir / scene)])
        if args.video_start_ns is not None:
            command.extend(["--video-start-ns", str(args.video_start_ns)])
        if args.scene_start_offset_ns is not None:
            command.extend(
                ["--scene-start-offset-ns", str(args.scene_start_offset_ns)]
            )
        if args.fps is not None:
            command.extend(["--fps", str(args.fps)])
        for stage in args.skip:
            command.extend(["--skip", stage])
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        if args.dry_run:
            command.append("--dry-run")
        print(f"\n### {scene} ({duration:.2f}s) ###")
        env = os.environ.copy()
        if args.api_key:
            env["OPENAI_API_KEY"] = args.api_key
        if args.base_url:
            env["OPENAI_BASE_URL"] = args.base_url
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode:
            failures.append(scene)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
    if failures:
        raise SystemExit(f"failed scenes: {', '.join(failures)}")


if __name__ == "__main__":
    main()
