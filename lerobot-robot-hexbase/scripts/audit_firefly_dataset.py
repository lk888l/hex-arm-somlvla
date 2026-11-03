#!/usr/bin/env python3
"""Read-only safety audit of a Firefly LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_dataset_audit import (  # noqa: E402
    DatasetAuditInputError,
    audit_firefly_dataset,
    load_controller_contract,
    resolve_dataset_root,
    strict_json_dumps,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory, or a LeRobot cache directory containing repo_id.",
    )
    parser.add_argument(
        "--repo-id", required=True, help="LeRobot repository ID used as a label/path."
    )
    parser.add_argument(
        "--controller-contract",
        type=Path,
        help=(
            "Saved live controller JSON. Without it, structural checks run but the "
            "safety verdict fails closed."
        ),
    )
    parser.add_argument(
        "--expected-task",
        help="Require every used task to exactly equal this SmolVLA instruction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_root = resolve_dataset_root(args.root, args.repo_id)
        contract = (
            load_controller_contract(args.controller_contract)
            if args.controller_contract is not None
            else None
        )
        report = audit_firefly_dataset(
            dataset_root,
            args.repo_id,
            controller_contract=contract,
            expected_task=args.expected_task,
        )
    except DatasetAuditInputError as exc:
        report = {
            "schema_version": 1,
            "audit": "firefly_lerobot_dataset",
            "read_only": True,
            "summary": {
                "verdict": "error",
                "safe_for_training": False,
                "error_count": 1,
                "warning_count": 0,
                "errors": [{"code": "audit_input_error", "message": str(exc)}],
                "warnings": [],
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(strict_json_dumps(report), end="")
    return 0 if report["summary"]["safe_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
