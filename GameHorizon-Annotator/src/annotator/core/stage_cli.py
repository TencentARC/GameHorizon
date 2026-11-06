from __future__ import annotations

import argparse
from pathlib import Path

from .instructions import generate_instructions
from .profiles import load_profile, supported_games
from .schema import instruction_path


def instruction_parser(level: int, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature; unset leaves the endpoint default in place. "
        "Pass 0 for repeatable wording.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--api-max-frames", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.set_defaults(level=level)
    return parser


def run_instruction_stage(args: argparse.Namespace) -> None:
    profile = load_profile(args.game)
    payload = generate_instructions(
        game_id=args.game,
        profile=profile,
        workdir=args.workdir,
        level=args.level,
        model=args.model,
        annotations_json=args.annotations_json,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        temperature=args.temperature,
        concurrency=args.concurrency,
        api_max_frames=args.api_max_frames,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
    ok = sum(1 for item in payload["items"] if item.get("status") == "ok")
    errors = sum(1 for item in payload["items"] if item.get("status") == "error")
    print(f"[instruction L{args.level}] ok={ok} errors={errors} -> {instruction_path(args.workdir, args.level)}")

