#!/usr/bin/env python3
"""Materialize the pinned SmolVLA base policy for fully offline training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lerobot_robot_hexbase.smolvla_base_bundle import (  # noqa: E402
    SmolVLABaseBundleError,
    materialize_smolvla_base,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
        help="Existing Hugging Face snapshot for the pinned lerobot/smolvla_base revision.",
    )
    parser.add_argument(
        "--smolvlm-bundle",
        type=Path,
        required=True,
        help="Previously materialized and validated Firefly SmolVLM asset bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New local policy directory; an existing path is never overwritten.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base, provenance = materialize_smolvla_base(
            args.source_snapshot, args.smolvlm_bundle, args.output
        )
    except (SmolVLABaseBundleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"base_path": str(base), "provenance": provenance},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
