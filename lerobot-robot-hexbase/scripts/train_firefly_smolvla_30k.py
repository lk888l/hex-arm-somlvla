#!/usr/bin/env python3
"""Launch the fixed formal Firefly SmolVLA 30k attested training job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
OUTPUT_DIR = Path(
    "/home/kk/kk_data/lerobot/outputs/train/"
    "firefly_y6_smolvla_validated_v2_attested_run1"
)
SMOLVLA_BASE = Path(
    "/home/kk/kk_data/lerobot/assets/firefly/"
    "smolvla_base-c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)
EXPECTED_TASK = "Pick up the red block and place it in the blue box."


def build_command() -> list[str]:
    wrapper = REPOSITORY_ROOT / "scripts/train_firefly_smolvla_attested.py"
    config = REPOSITORY_ROOT / "configs/train_smolvla_firefly_y6.example.yaml"
    return [
        sys.executable,
        str(wrapper),
        f"--dataset-root={DATASET_ROOT}",
        "--repo-id=local/firefly_y6_manual_validated_v2",
        f"--controller-contract={CONTROLLER_CONTRACT}",
        f"--expected-task={EXPECTED_TASK}",
        f"--output-dir={OUTPUT_DIR}",
        f"--smolvla-base={SMOLVLA_BASE}",
        "--",
        f"--config_path={config}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command()
    if args.dry_run:
        print("\n".join(command))
        return 0

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
