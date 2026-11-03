#!/usr/bin/env python3
"""Record one audited Firefly Y6 full-cycle demonstration per invocation."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import record_firefly_y6_validated_v2 as validated_v2  # noqa: E402
from lerobot_robot_hexbase.firefly_fullcycle_v4 import (  # noqa: E402
    FullcycleV4ProtocolError,
    MAX_START_MOTION_S,
    PREFERRED_START_MOTION_S,
    validate_start_motion_windows,
)


REPO_ID = "local/firefly_y6_fullcycle_v4"
EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
TARGET_TOTAL = 50
EPISODES_PER_INVOCATION = 1
EPISODE_TIME_S = 600
RESET_TIME_S = 600
DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/record_firefly_y6_fullcycle_v4.local.yaml"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)


RecordingPreflightError = validated_v2.RecordingPreflightError
AuditFunction = validated_v2.AuditFunction
ContractLoader = validated_v2.ContractLoader
ExecFunction = Callable[[str, Sequence[str]], Any]
ProtocolValidator = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class RecordingTarget:
    dataset_root: Path = DATASET_ROOT
    repo_id: str = REPO_ID
    config_path: Path = CONFIG_PATH
    controller_contract: Path = CONTROLLER_CONTRACT
    expected_task: str = EXPECTED_TASK
    target_total: int = TARGET_TOTAL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RecordingPreflightError(f"record config could not be parsed: {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RecordingPreflightError("record config must contain a YAML mapping")
    return raw


def _mapping(parent: Mapping[str, Any], key: str, *, label: str = "") -> Mapping[str, Any]:
    value = parent.get(key)
    name = f"{label}.{key}" if label else key
    if not isinstance(value, Mapping):
        raise RecordingPreflightError(f"record config {name} must be a mapping")
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
        raise RecordingPreflightError(
            f"record config {label} must be exactly {expected!r}, got {value!r}"
        )


def _validate_camera(camera: Mapping[str, Any], *, label: str, rotation: int | None) -> None:
    expected = {
        "type": "intelrealsense",
        "fps": 30,
        "width": 640,
        "height": 480,
        "use_rgb": True,
        "use_depth": False,
    }
    for field, value in expected.items():
        _exact(camera.get(field), value, f"robot.cameras.{label}.{field}")
    if rotation is None:
        if "rotation" in camera:
            _exact(camera.get("rotation"), 0, f"robot.cameras.{label}.rotation")
    else:
        _exact(camera.get("rotation"), rotation, f"robot.cameras.{label}.rotation")
    serial = camera.get("serial_number_or_name")
    if not isinstance(serial, str) or not serial or serial.startswith("CHANGE_ME"):
        raise RecordingPreflightError(
            f"record config robot.cameras.{label}.serial_number_or_name is not resolved"
        )


def validate_record_config(target: RecordingTarget) -> None:
    """Pin the long/manual episode contract before any hardware connection."""

    # Reuse the v2 fail-closed destination, task, rate, and controller-step gates.
    validated_v2.validate_record_config(target)
    raw = _load_config(target.config_path.expanduser())
    robot = _mapping(raw, "robot")
    teleop = _mapping(raw, "teleop")
    dataset = _mapping(raw, "dataset")
    cameras = _mapping(robot, "cameras", label="robot")
    scene = _mapping(cameras, "scene", label="robot.cameras")
    wrist = _mapping(cameras, "wrist", label="robot.cameras")

    _exact(robot.get("type"), "hex_zenoh_observer", "robot.type")
    _exact(teleop.get("type"), "hex_zenoh_command_tap", "teleop.type")
    _validate_camera(scene, label="scene", rotation=None)
    _validate_camera(wrist, label="wrist", rotation=180)

    fixed_dataset_fields = {
        "episode_time_s": EPISODE_TIME_S,
        "reset_time_s": RESET_TIME_S,
        "num_episodes": EPISODES_PER_INVOCATION,
        "video": True,
        "push_to_hub": False,
        "streaming_encoding": False,
        "num_image_writer_processes": 0,
        "num_image_writer_threads_per_camera": 4,
    }
    for field, value in fixed_dataset_fields.items():
        _exact(dataset.get(field), value, f"dataset.{field}")
    _exact(raw.get("display_data"), False, "display_data")
    _exact(raw.get("play_sounds"), False, "play_sounds")
    _exact(raw.get("resume"), False, "resume")


def build_record_command(
    target: RecordingTarget,
    plan: validated_v2.RecordingPlan,
    *,
    record_executable: Path,
) -> list[str]:
    if plan.remaining_episodes <= 0:
        raise RecordingPreflightError(
            "cannot build a recording command with no episodes remaining"
        )
    return [
        str(record_executable),
        f"--config_path={target.config_path}",
        f"--resume={'true' if plan.resume else 'false'}",
        f"--dataset.num_episodes={EPISODES_PER_INVOCATION}",
    ]


def run_recording(
    target: RecordingTarget,
    *,
    audit: AuditFunction = validated_v2.audit_firefly_dataset,
    contract_loader: ContractLoader = validated_v2.load_controller_contract,
    protocol_validator: ProtocolValidator = validate_start_motion_windows,
    exec_function: ExecFunction = os.execv,
    record_executable: Path | None = None,
) -> int:
    """Audit the accumulated dataset, then record exactly one more episode."""

    validate_record_config(target)
    plan = validated_v2.make_recording_plan(
        target,
        audit=audit,
        contract_loader=contract_loader,
    )
    if plan.existing_episodes:
        try:
            protocol_report = protocol_validator(
                target.dataset_root,
                expected_episodes=plan.existing_episodes,
            )
        except FullcycleV4ProtocolError as exc:
            raise RecordingPreflightError(
                f"existing dataset failed the full-cycle v4 start-motion audit: {exc}"
            ) from exc
        warning_count = len(protocol_report.get("warnings", []))
        if warning_count:
            print(
                f"Start-motion audit accepted {warning_count} episode(s) after the "
                f"preferred {PREFERRED_START_MOTION_S:.1f}s window but within the "
                f"hard {MAX_START_MOTION_S:.1f}s limit.",
                flush=True,
            )
    if plan.remaining_episodes == 0:
        print(
            f"Full-cycle recording complete: {plan.existing_episodes}/"
            f"{plan.target_total} audited episodes."
        )
        return 0

    executable = (
        validated_v2.sibling_lerobot_record()
        if record_executable is None
        else Path(record_executable)
    )
    command = build_record_command(target, plan, record_executable=executable)
    mode = "resume" if plan.resume else "fresh"
    print(
        f"Full-cycle recording preflight passed: mode={mode}, "
        f"existing={plan.existing_episodes}, this_run=1, "
        f"target={plan.target_total}. Complete the full task, return home, "
        f"and begin purposeful arm motion within {PREFERRED_START_MOTION_S:.1f}s "
        f"(hard limit {MAX_START_MOTION_S:.1f}s). Then press n/right-arrow to save "
        "and exit; press r/left-arrow to discard a failed attempt.",
        flush=True,
    )
    exec_function(str(executable), command)
    return 0


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    try:
        return run_recording(RecordingTarget())
    except (RecordingPreflightError, OSError) as exc:
        print(f"Full-cycle recording refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
