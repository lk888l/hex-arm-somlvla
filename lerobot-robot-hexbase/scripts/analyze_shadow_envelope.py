#!/usr/bin/env python
"""Counterfactually expand j2/j3 shadow bounds without touching hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot_robot_hexbase.shadow_envelope import EnvelopeContract, analyze_shadow_envelope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument(
        "--controller-contract",
        type=Path,
        required=True,
        help="Saved live controller bounds and effective 30 Hz step limits.",
    )
    parser.add_argument("--joint-2-min", type=float, required=True)
    parser.add_argument("--joint-3-max", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = EnvelopeContract.from_mapping(
        json.loads(args.controller_contract.read_text(encoding="utf-8"))
    )
    events = [
        json.loads(line)
        for line in args.telemetry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = analyze_shadow_envelope(
        events,
        contract,
        joint_2_min=args.joint_2_min,
        joint_3_max=args.joint_3_max,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
