#!/usr/bin/env python3
"""Run formal local Firefly SmolVLA training with dataset attestation."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# The training environment deliberately has no Zenoh/RealSense dependencies.
# Import only the offline attestation submodules without executing the plugin
# package __init__, which registers hardware classes and imports Zenoh.
_PACKAGE_NAME = "lerobot_robot_hexbase"
if _PACKAGE_NAME not in sys.modules:
    package_dir = _REPOSITORY_ROOT / _PACKAGE_NAME
    package = types.ModuleType(_PACKAGE_NAME)
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = _PACKAGE_NAME
    package.__path__ = [str(package_dir)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        _PACKAGE_NAME, loader=None, is_package=True
    )
    sys.modules[_PACKAGE_NAME] = package

from lerobot_robot_hexbase.firefly_training_attestation import (  # noqa: E402
    TrainingAttestationError,
    run_attested_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--controller-contract", type=Path, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smolvla-base",
        type=Path,
        required=True,
        help="Pinned, materialized local SmolVLA base policy directory.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python containing LeRobot training dependencies (default: this Python).",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments for LeRobot after '--'; do not include the lerobot-train executable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        written = run_attested_training(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            controller_contract=args.controller_contract,
            expected_task=args.expected_task,
            output_dir=args.output_dir,
            smolvla_base=args.smolvla_base,
            training_args=args.training_args,
            python_executable=args.python_executable,
        )
    except (TrainingAttestationError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"attested training refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "attested",
                "checkpoint_count": len(written),
                "attestations": [str(path) for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
