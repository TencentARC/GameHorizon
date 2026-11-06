#!/usr/bin/env python3
"""Semantically merge Level-2 scenes into a canonical Level-3 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotator.core.merging import build_level3_manifest, make_api_decision_fn
from annotator.core.profiles import load_profile, supported_games
from annotator.core.schema import load_manifest, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=240.0)
    parser.add_argument("--max-seconds", type=float, default=480.0)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature for the merge decision; unset leaves the endpoint default.",
    )
    parser.add_argument("--api-max-frames", type=int, default=30)
    parser.add_argument("--materialize", choices=("none", "symlink", "hardlink", "copy"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.game)
    source = load_manifest(args.workdir, 2, fps=profile.fallback_fps)
    decision_fn = make_api_decision_fn(
        profile=profile,
        workdir=args.workdir,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        minimum_s=args.min_seconds,
        maximum_s=args.max_seconds,
        api_max_frames=args.api_max_frames,
        temperature=args.temperature,
    )
    payload = build_level3_manifest(
        game_id=args.game,
        workdir=args.workdir,
        level2_manifest=source,
        minimum_s=args.min_seconds,
        maximum_s=args.max_seconds,
        decision_fn=decision_fn,
        materialize=args.materialize,
    )
    print(f"[3b] scenes={len(payload['items'])} -> {manifest_path(args.workdir, 3)}")


if __name__ == "__main__":
    main()

