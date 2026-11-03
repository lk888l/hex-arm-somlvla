#!/usr/bin/env python3
"""Fail-closed launcher for a fresh or safely resumable Firefly recording."""

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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_dataset_audit import (  # noqa: E402
    DatasetAuditInputError,
    FireflyAuditContract,
    audit_firefly_dataset,
    load_controller_contract,
)


REPO_ID = "local/firefly_y6_manual_validated_v2"
EXPECTED_TASK = "Pick up the red block and place it in the blue box."
DEFAULT_TARGET_TOTAL = 20
DEFAULT_DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2"
)
DEFAULT_CONFIG_PATH = (
    _REPOSITORY_ROOT / "configs" / "record_firefly_y6_validated_v2.local.yaml"
)
DEFAULT_CONTROLLER_CONTRACT = (
    _REPOSITORY_ROOT
    / "outputs"
    / "firefly_diagnostics"
    / "controller_contract_validated_v2_arm0435_ee0165.json"
)


class RecordingPreflightError(RuntimeError):
    """The existing dataset cannot be safely resumed."""


@dataclass(frozen=True)
class RecordingTarget:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    repo_id: str = REPO_ID
    config_path: Path = DEFAULT_CONFIG_PATH
    controller_contract: Path = DEFAULT_CONTROLLER_CONTRACT
    expected_task: str = EXPECTED_TASK
    target_total: int = DEFAULT_TARGET_TOTAL


@dataclass(frozen=True)
class RecordingPlan:
    resume: bool
    existing_episodes: int
    remaining_episodes: int
    target_total: int


AuditFunction = Callable[..., Mapping[str, Any]]
ContractLoader = Callable[[Path], FireflyAuditContract]
ExecFunction = Callable[[str, Sequence[str]], Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _validated_episode_count(report: Mapping[str, Any]) -> int:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise RecordingPreflightError("full audit returned no summary")
    if summary.get("safe_for_training") is not True:
        errors = summary.get("errors")
        details: list[str] = []
        if isinstance(errors, list):
            for finding in errors[:3]:
                if isinstance(finding, Mapping):
                    code = finding.get("code", "audit_error")
                    message = finding.get("message", "")
                    details.append(f"{code}: {message}".rstrip())
        suffix = f" ({'; '.join(details)})" if details else ""
        raise RecordingPreflightError(
            "existing dataset failed the full safety audit and will not be resumed"
            f"{suffix}"
        )

    checks = report.get("checks")
    metadata = checks.get("metadata_consistency") if isinstance(checks, Mapping) else None
    totals = metadata.get("total_episodes") if isinstance(metadata, Mapping) else None
    actual = totals.get("actual") if isinstance(totals, Mapping) else None
    recorded = totals.get("metadata") if isinstance(totals, Mapping) else None
    if (
        isinstance(actual, bool)
        or not isinstance(actual, int)
        or actual < 0
        or recorded != actual
    ):
        raise RecordingPreflightError(
            "full audit did not return a consistent non-negative episode count"
        )
    return actual


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RecordingPreflightError(
            f"record config field {key!r} must be a mapping"
        )
    return value


def _require_exact_number(value: Any, expected: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise RecordingPreflightError(
            f"record config {label} must be exactly {expected}, got {value!r}"
        )


def validate_record_config(target: RecordingTarget) -> None:
    """Pin the YAML destination, task, rate, and recording step contract."""

    config_path = target.config_path.expanduser()
    if config_path.is_symlink() or not config_path.is_file():
        raise RecordingPreflightError(
            f"record config must be a regular, non-symlink file: {config_path}"
        )
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RecordingPreflightError(
            f"record config could not be parsed: {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RecordingPreflightError("record config must contain a YAML mapping")

    dataset = _required_mapping(raw, "dataset")
    teleop = _required_mapping(raw, "teleop")
    if dataset.get("repo_id") != target.repo_id:
        raise RecordingPreflightError(
            f"record config dataset.repo_id must be exactly {target.repo_id!r}"
        )
    configured_root_value = dataset.get("root")
    if not isinstance(configured_root_value, str):
        raise RecordingPreflightError(
            "record config dataset.root must be an absolute path"
        )
    configured_root = Path(configured_root_value).expanduser()
    if not configured_root.is_absolute():
        raise RecordingPreflightError(
            "record config dataset.root must be an absolute path"
        )
    configured_canonical = configured_root.resolve(strict=False)
    target_canonical = target.dataset_root.expanduser().resolve(strict=False)
    if configured_canonical != target_canonical:
        raise RecordingPreflightError(
            "record config dataset.root does not match the audited target: "
            f"{configured_canonical} != {target_canonical}"
        )
    if dataset.get("single_task") != target.expected_task:
        raise RecordingPreflightError(
            f"record config dataset.single_task must be exactly {target.expected_task!r}"
        )
    _require_exact_number(dataset.get("fps"), 30.0, "dataset.fps")
    _require_exact_number(teleop.get("record_hz"), 30.0, "teleop.record_hz")
    _require_exact_number(
        teleop.get("arm_step_limit_rad"), 0.0435, "teleop.arm_step_limit_rad"
    )
    _require_exact_number(
        teleop.get("ee_step_limit_rad"), 0.0165, "teleop.ee_step_limit_rad"
    )


def make_recording_plan(
    target: RecordingTarget,
    *,
    audit: AuditFunction = audit_firefly_dataset,
    contract_loader: ContractLoader = load_controller_contract,
) -> RecordingPlan:
    """Audit an existing dataset and decide whether/how many episodes to record."""

    if (
        isinstance(target.target_total, bool)
        or not isinstance(target.target_total, int)
        or target.target_total <= 0
    ):
        raise RecordingPreflightError("target_total must be a positive integer")

    validate_record_config(target)
    dataset_root = target.dataset_root.expanduser()
    try:
        contract = contract_loader(target.controller_contract)
    except DatasetAuditInputError as exc:
        raise RecordingPreflightError(
            f"controller contract validation could not complete: {exc}"
        ) from exc
    if not dataset_root.exists():
        return RecordingPlan(
            resume=False,
            existing_episodes=0,
            remaining_episodes=target.target_total,
            target_total=target.target_total,
        )
    if not dataset_root.is_dir():
        raise RecordingPreflightError(
            f"dataset root exists but is not a directory: {dataset_root}"
        )
    try:
        report = audit(
            dataset_root,
            target.repo_id,
            controller_contract=contract,
            expected_task=target.expected_task,
        )
    except DatasetAuditInputError as exc:
        raise RecordingPreflightError(
            f"existing dataset audit could not complete: {exc}"
        ) from exc

    existing = _validated_episode_count(report)
    if existing > target.target_total:
        raise RecordingPreflightError(
            f"existing dataset has {existing} episodes, above target {target.target_total}"
        )
    return RecordingPlan(
        resume=True,
        existing_episodes=existing,
        remaining_episodes=target.target_total - existing,
        target_total=target.target_total,
    )


def sibling_lerobot_record(python_executable: str | Path = sys.executable) -> Path:
    """Return lerobot-record from the same environment as this Python process."""

    return Path(python_executable).with_name("lerobot-record")


def build_record_command(
    target: RecordingTarget,
    plan: RecordingPlan,
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
        f"--dataset.num_episodes={plan.remaining_episodes}",
    ]


def run_recording(
    target: RecordingTarget,
    *,
    audit: AuditFunction = audit_firefly_dataset,
    contract_loader: ContractLoader = load_controller_contract,
    exec_function: ExecFunction = os.execv,
    record_executable: Path | None = None,
) -> int:
    """Run preflight, then replace this process with the environment's recorder."""

    plan = make_recording_plan(target, audit=audit, contract_loader=contract_loader)
    if plan.remaining_episodes == 0:
        print(
            f"Recording already complete: {plan.existing_episodes}/{plan.target_total} "
            "audited episodes."
        )
        return 0

    executable = (
        sibling_lerobot_record() if record_executable is None else Path(record_executable)
    )
    command = build_record_command(target, plan, record_executable=executable)
    mode = "resume" if plan.resume else "fresh"
    print(
        f"Recording preflight passed: mode={mode}, "
        f"existing={plan.existing_episodes}, remaining={plan.remaining_episodes}, "
        f"target={plan.target_total}.",
        flush=True,
    )
    exec_function(str(executable), command)
    return 0  # A test double may return; os.execv never does.


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    target = RecordingTarget()
    try:
        return run_recording(target)
    except (RecordingPreflightError, OSError) as exc:
        print(f"Recording refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
