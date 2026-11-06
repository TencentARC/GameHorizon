#!/usr/bin/env python3
"""Generate canonical Level-3 instructions from the Level-3 manifest."""

from annotator.core.stage_cli import instruction_parser, run_instruction_stage


def main() -> None:
    run_instruction_stage(instruction_parser(3, __doc__).parse_args())


if __name__ == "__main__":
    main()

