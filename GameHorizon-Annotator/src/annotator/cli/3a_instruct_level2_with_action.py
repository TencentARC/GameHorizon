#!/usr/bin/env python3
"""Generate canonical Level-2 instructions from the Level-2 manifest."""

from annotator.core.stage_cli import instruction_parser, run_instruction_stage


def main() -> None:
    run_instruction_stage(instruction_parser(2, __doc__).parse_args())


if __name__ == "__main__":
    main()

