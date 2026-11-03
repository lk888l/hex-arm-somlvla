#!/usr/bin/env python3
"""Materialize the pinned Firefly SmolVLM assets from a Hugging Face snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lerobot_robot_hexbase.smolvlm_bundle import (  # noqa: E402
    SmolVLMBundleError,
    materialize_smolvlm_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        required=True,
        help="Existing Hugging Face snapshot directory for the pinned SmolVLM revision.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; an existing path is never overwritten.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bundle, manifest = materialize_smolvlm_bundle(args.source_snapshot, args.output)
    except (SmolVLMBundleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"bundle_path": str(bundle), "manifest": manifest},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
