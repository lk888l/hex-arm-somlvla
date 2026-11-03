#!/usr/bin/env python3
"""Launch the fixed 50-episode Firefly full-cycle SmolVLA 100k job."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# The training environment deliberately omits Zenoh. Import the offline
# protocol module without executing the hardware-registering package __init__.
_PACKAGE_NAME = "lerobot_robot_hexbase"
if _PACKAGE_NAME not in sys.modules:
    package_dir = REPOSITORY_ROOT / _PACKAGE_NAME
    package = types.ModuleType(_PACKAGE_NAME)
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = _PACKAGE_NAME
    package.__path__ = [str(package_dir)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        _PACKAGE_NAME,
        loader=None,
        is_package=True,
    )
    sys.modules[_PACKAGE_NAME] = package

from lerobot_robot_hexbase.firefly_fullcycle_v4 import (  # noqa: E402
    FullcycleV4ProtocolError,
    validate_start_motion_windows,
)
REPO_ID = "local/firefly_y6_fullcycle_v4"
EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
EXPECTED_EPISODES = 50
FPS = 30
MIN_EPISODE_SECONDS = 35
MAX_EPISODE_SECONDS = 120
TRAINING_STEPS = 100_000
DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
OUTPUT_DIR = Path(
    "/home/kk/kk_data/lerobot/outputs/train/"
    "firefly_y6_smolvla_fullcycle_v4_100k_attested_run1"
)
SMOLVLA_BASE = Path(
    "/home/kk/kk_data/lerobot/assets/firefly/"
    "smolvla_base-c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)
CONFIG = REPOSITORY_ROOT / "configs/train_smolvla_firefly_y6_fullcycle_v4.yaml"


EpisodeLoader = Callable[[Path], Sequence[Mapping[str, Any]]]


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _read_yaml_object(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular, non-symlink file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must contain a YAML mapping: {path}")
    return value


def _mapping(parent: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"training config {label}.{key} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    elif isinstance(expected, (int, float)):
        matches = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) == float(expected)
        )
    else:
        matches = value == expected
    if not matches:
        raise RuntimeError(
            f"training config {label} must be exactly {expected!r}, got {value!r}"
        )


def validate_training_config(
    config_path: Path = CONFIG,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> None:
    """Fail closed if the human-readable YAML drifts from the v4 baseline."""

    raw = _read_yaml_object(config_path, label="training config")
    dataset = _mapping(raw, "dataset", label="root")
    transforms = _mapping(dataset, "image_transforms", label="dataset")
    wandb = _mapping(raw, "wandb", label="root")

    _exact(dataset.get("repo_id"), REPO_ID, "dataset.repo_id")
    configured_root = dataset.get("root")
    if not isinstance(configured_root, str):
        raise RuntimeError("training config dataset.root must be an absolute path")
    configured_path = Path(configured_root).expanduser()
    if not configured_path.is_absolute() or configured_path.resolve(strict=False) != dataset_root.resolve(strict=False):
        raise RuntimeError(
            f"training config dataset.root must be exactly {dataset_root}, got {configured_root!r}"
        )
    _exact(dataset.get("return_uint8"), True, "dataset.return_uint8")
    _exact(transforms.get("enable"), False, "dataset.image_transforms.enable")

    fixed_fields = {
        "batch_size": 1,
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "steps": TRAINING_STEPS,
        "env_eval_freq": 0,
        "eval_steps": 0,
        "max_eval_samples": 0,
        "log_freq": 20,
        "save_checkpoint": True,
        "save_freq": 10_000,
        "use_policy_training_preset": True,
    }
    for field, expected in fixed_fields.items():
        _exact(raw.get(field), expected, field)
    _exact(wandb.get("enable"), False, "wandb.enable")


def _load_episode_rows(dataset_root: Path) -> Sequence[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to validate episode durations") from exc

    files = sorted((dataset_root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise RuntimeError("dataset contains no episode metadata parquet files")
    rows: list[Mapping[str, Any]] = []
    for path in files:
        try:
            table = parquet.read_table(path, columns=["episode_index", "tasks", "length"])
        except Exception as exc:
            raise RuntimeError(f"cannot read episode metadata {path}: {exc}") from exc
        rows.extend(table.to_pylist())
    return rows


def validate_dataset_metadata(
    dataset_root: Path = DATASET_ROOT,
    *,
    episode_loader: EpisodeLoader = _load_episode_rows,
) -> None:
    """Require 50 complete, consistently sized, single-task episodes."""

    info = _read_json_object(dataset_root / "meta/info.json", label="dataset metadata")
    if type(info.get("total_episodes")) is not int or info["total_episodes"] != EXPECTED_EPISODES:
        raise RuntimeError(
            f"full-cycle dataset must contain exactly {EXPECTED_EPISODES} episodes, "
            f"got {info.get('total_episodes')!r}"
        )
    _exact(info.get("fps"), FPS, "dataset metadata fps")
    _exact(info.get("total_tasks"), 1, "dataset metadata total_tasks")
    expected_split = f"0:{EXPECTED_EPISODES}"
    splits = info.get("splits")
    if not isinstance(splits, Mapping) or splits.get("train") != expected_split:
        raise RuntimeError(
            f"full-cycle dataset train split must be {expected_split!r}, got {splits!r}"
        )
    total_frames = info.get("total_frames")
    if type(total_frames) is not int or total_frames <= 0:
        raise RuntimeError(f"dataset total_frames must be a positive integer, got {total_frames!r}")

    rows = list(episode_loader(dataset_root))
    if len(rows) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"episode metadata must contain exactly {EXPECTED_EPISODES} rows, got {len(rows)}"
        )
    rows.sort(key=lambda row: row.get("episode_index", -1))
    expected_min = MIN_EPISODE_SECONDS * FPS
    expected_max = MAX_EPISODE_SECONDS * FPS
    length_sum = 0
    for expected_index, row in enumerate(rows):
        if row.get("episode_index") != expected_index:
            raise RuntimeError("episode metadata indexes must be contiguous from 0 to 49")
        tasks = row.get("tasks")
        if tasks != [EXPECTED_TASK]:
            raise RuntimeError(
                f"episode {expected_index} task must be exactly {EXPECTED_TASK!r}, got {tasks!r}"
            )
        length = row.get("length")
        if type(length) is not int or not expected_min <= length <= expected_max:
            raise RuntimeError(
                f"episode {expected_index} length must be in [{expected_min}, {expected_max}] "
                f"frames ({MIN_EPISODE_SECONDS}-{MAX_EPISODE_SECONDS}s), got {length!r}"
            )
        length_sum += length
    if length_sum != total_frames:
        raise RuntimeError(
            f"episode lengths sum to {length_sum} frames but info.json reports {total_frames}"
        )


def build_command() -> list[str]:
    wrapper = REPOSITORY_ROOT / "scripts/train_firefly_smolvla_attested.py"
    return [
        sys.executable,
        str(wrapper),
        f"--dataset-root={DATASET_ROOT}",
        f"--repo-id={REPO_ID}",
        f"--controller-contract={CONTROLLER_CONTRACT}",
        f"--expected-task={EXPECTED_TASK}",
        f"--output-dir={OUTPUT_DIR}",
        f"--smolvla-base={SMOLVLA_BASE}",
        "--",
        f"--config_path={CONFIG}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.optimizer_lr=0.0001",
        "--policy.scheduler_warmup_steps=2000",
        "--policy.scheduler_decay_steps=100000",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--batch_size=1",
        "--num_workers=4",
        "--prefetch_factor=2",
        "--persistent_workers=true",
        "--steps=100000",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--max_eval_samples=0",
        "--log_freq=20",
        "--save_freq=10000",
        "--use_policy_training_preset=true",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pinned command without touching the dataset or starting training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command()
    if args.dry_run:
        print("\n".join(command))
        return 0

    try:
        validate_training_config()
        validate_dataset_metadata()
        validate_start_motion_windows(
            DATASET_ROOT,
            expected_episodes=EXPECTED_EPISODES,
        )
    except (RuntimeError, FullcycleV4ProtocolError) as exc:
        print(f"Full-cycle v4 training refused: {exc}", file=sys.stderr)
        return 2

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
