#!/usr/bin/env python3
"""Merge Level-1 scenes into a canonical real-duration Level-2 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotator.core.merging import build_level2_manifest
from annotator.core.profiles import load_profile, supported_games
from annotator.core.schema import load_manifest, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", choices=("none", "symlink", "hardlink", "copy"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.game)
    source = load_manifest(args.workdir, 1, fps=profile.fallback_fps)
    payload = build_level2_manifest(
        game_id=args.game,
        workdir=args.workdir,
        level1_manifest=source,
        minimum_s=args.min_seconds,
        maximum_s=args.max_seconds,
        seed=args.seed,
        materialize=args.materialize,
    )
    print(f"[2b] scenes={len(payload['items'])} -> {manifest_path(args.workdir, 2)}")


if __name__ == "__main__":
    main()

