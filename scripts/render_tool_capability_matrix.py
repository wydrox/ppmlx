#!/usr/bin/env python3
"""Render the public local tool capability matrix from reviewed evidence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ppmlx.local_runtime.profile_publication import (
    ProfilePublicationError,
    load_reports,
    render_capability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "docs" / "capabilities" / "tool-profile-evidence"
DEFAULT_OUTPUT = ROOT / "docs" / "capabilities" / "tool-profiles.md"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render reviewed exact-model evidence into the tool capability matrix."
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the checked-in matrix differs from generated output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rendered = render_capability_matrix(load_reports(args.evidence))
    except ProfilePublicationError as error:
        print(f"Capability matrix generation failed: {error.code}", file=sys.stderr)
        return 2

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError:
            print("Capability matrix is missing", file=sys.stderr)
            return 1
        if existing != rendered:
            print("Capability matrix is out of date", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote capability matrix to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
