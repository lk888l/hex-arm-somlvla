#!/usr/bin/env python
"""Commission and inspect a staged Firefly Y6 recording recovery profile.

The CLI is intentionally plan-only.  ``--execute`` is recognized so an
operator cannot mistake an unknown option for protection, but it fails before
loading a profile, importing a hardware follower, opening Zenoh, or acquiring
a controller session.  See ``ACTIVE_EXECUTION_BLOCKER`` below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

def _load_offline_module(module_name: str, filename: str):
    """Load a pure package module without executing the hardware-heavy __init__."""

    package_name = "lerobot_robot_hexbase"
    package_root = Path(__file__).resolve().parents[1] / package_name
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    qualified_name = f"{package_name}.{module_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified_name, package_root / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load offline module {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_load_offline_module("firefly_recording_recovery", "firefly_recording_recovery.py")
_load_offline_module(
    "firefly_recording_recovery_executor", "firefly_recording_recovery_executor.py"
)

from lerobot_robot_hexbase.firefly_recording_recovery import (  # noqa: E402
    AXIS_NAMES,
    RecoveryPlanningError,
    RecoveryProfile,
    audit_recovery_plan,
    plan_recording_recovery,
    stage_contract,
)
from lerobot_robot_hexbase.firefly_recording_recovery_executor import (  # noqa: E402
    RecoveryControlContract,
)


SCHEMA_VERSION = 1
ACTIVE_EXECUTION_BLOCKER = (
    "active recovery is intentionally unavailable: HexZenohFollower currently enters ACTIVE "
    "and publishes a hold before an application can enforce the commissioned recovery "
    "envelope. Add and test a follower pre-ACTIVE startup-position gate, then commission the "
    "local home/envelopes and staged motion before enabling execution"
)
DATASET_CONTRACT_NOTE = (
    "the recorded demonstration must include the operator return-home motion before the "
    "episode is saved; a post-save recovery is outside the episode and cannot repair missing "
    "return-home training labels"
)
COMMISSIONING_GATES = (
    "home_validated",
    "recovery_envelope_validated",
    "j4_clearance_validated",
    "j2_stage_validated",
    "j3_stage_validated",
    "remaining_ee_stage_validated",
    "independent_estop_procedure_validated",
    "teleop_handoff_validated",
)
PLANNING_FIELDS = (
    "home",
    "home_tolerance",
    "position_min",
    "position_max",
    "entry_min",
    "entry_max",
    "recovery_min",
    "recovery_max",
    "step_cap",
    "j4_clearance",
    "max_frames",
    "max_total_travel_rad",
    "allow_positive_j3_motion",
)
CONTROL_FIELDS = (
    "candidate_lead_budget",
    "command_hz",
    "stall_frames",
    "max_movement_control_frames",
    "settle_frames",
    "required_home_frames",
    "minimum_progress_rad",
)


class RecoveryProfileFileError(ValueError):
    """The local commissioning document is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class LoadedProfile:
    path: Path
    sha256: str
    profile_id: str
    profile: RecoveryProfile | None
    control: RecoveryControlContract | None
    commissioning: dict[str, bool]
    incomplete_fields: tuple[str, ...]

    @property
    def profile_complete(self) -> bool:
        return self.profile is not None and self.control is not None

    @property
    def commissioning_complete(self) -> bool:
        return all(self.commissioning.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="validate and report; no I/O")
    action.add_argument("--plan", action="store_true", help="plan from --start; no I/O")
    action.add_argument(
        "--execute",
        action="store_true",
        help="reserved, but fail-closed until the pre-ACTIVE gate is implemented",
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "EE"),
        help="offline measured seven-axis start pose in radians",
    )
    parser.add_argument(
        "--expected-profile-sha256",
        help="optional lowercase SHA-256 binding for check/plan",
    )
    parser.add_argument(
        "--operator-authorized-staged-recovery",
        action="store_true",
        help="reserved one-shot execution assertion; cannot unlock the current CLI",
    )
    parser.add_argument(
        "--independent-estop-verified",
        action="store_true",
        help="reserved one-shot execution assertion; cannot unlock the current CLI",
    )
    args = parser.parse_args(argv)
    if args.plan and args.start is None:
        parser.error("--plan requires --start J1 J2 J3 J4 J5 J6 EE")
    if not args.plan and args.start is not None:
        parser.error("--start is valid only with --plan")
    if not args.execute and (
        args.operator_authorized_staged_recovery or args.independent_estop_verified
    ):
        parser.error("operator execution assertions are valid only with --execute")
    return args


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryProfileFileError(f"{label} must be a mapping")
    return value


def _bool_mapping(value: Any, label: str, keys: Sequence[str]) -> dict[str, bool]:
    raw = _mapping(value, label)
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise RecoveryProfileFileError(f"{label} has unknown fields: {unknown}")
    result: dict[str, bool] = {}
    for key in keys:
        item = raw.get(key, False)
        if type(item) is not bool:
            raise RecoveryProfileFileError(f"{label}.{key} must be boolean")
        result[key] = item
    return result


def _optional_dataclass(
    raw: Any,
    *,
    label: str,
    fields: Sequence[str],
    constructor,
) -> tuple[Any | None, tuple[str, ...]]:
    mapping = _mapping(raw, label)
    unknown = sorted(set(mapping) - set(fields))
    if unknown:
        raise RecoveryProfileFileError(f"{label} has unknown fields: {unknown}")
    missing = tuple(
        f"{label}.{field}"
        for field in fields
        if field not in mapping or mapping[field] is None
    )
    if missing:
        return None, missing
    try:
        return constructor(**{field: mapping[field] for field in fields}), ()
    except (RecoveryPlanningError, TypeError, ValueError) as exc:
        raise RecoveryProfileFileError(f"invalid {label}: {exc}") from exc


def load_profile(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedProfile:
    path = path.expanduser()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha256
        ):
            raise RecoveryProfileFileError(
                "--expected-profile-sha256 must be a lowercase SHA-256 digest"
            )
        if digest != expected_sha256:
            raise RecoveryProfileFileError(
                f"profile SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
    try:
        document = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise RecoveryProfileFileError(f"invalid YAML: {exc}") from exc
    root = _mapping(document, "profile")
    allowed = {"schema_version", "profile_id", "commissioning", "planning", "control"}
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise RecoveryProfileFileError(f"profile has unknown fields: {unknown}")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise RecoveryProfileFileError(
            f"schema_version must be exactly {SCHEMA_VERSION}"
        )
    profile_id = root.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise RecoveryProfileFileError("profile_id must be a non-empty string")
    commissioning = _bool_mapping(
        root.get("commissioning"), "commissioning", COMMISSIONING_GATES
    )
    profile, planning_missing = _optional_dataclass(
        root.get("planning"),
        label="planning",
        fields=PLANNING_FIELDS,
        constructor=RecoveryProfile,
    )
    control, control_missing = _optional_dataclass(
        root.get("control"),
        label="control",
        fields=CONTROL_FIELDS,
        constructor=RecoveryControlContract,
    )
    return LoadedProfile(
        path=path,
        sha256=digest,
        profile_id=profile_id,
        profile=profile,
        control=control,
        commissioning=commissioning,
        incomplete_fields=planning_missing + control_missing,
    )


def check_report(loaded: LoadedProfile) -> dict[str, Any]:
    blockers = [
        *(f"missing {field}" for field in loaded.incomplete_fields),
        *(
            f"commissioning.{name} is false"
            for name, passed in loaded.commissioning.items()
            if not passed
        ),
        ACTIVE_EXECUTION_BLOCKER,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": loaded.profile_id,
        "profile_path": str(loaded.path.absolute()),
        "profile_sha256": loaded.sha256,
        "profile_complete": loaded.profile_complete,
        "commissioning_complete": loaded.commissioning_complete,
        "active_executor_available": False,
        "execution_ready": False,
        "axis_order": AXIS_NAMES,
        "stage_contract": stage_contract(),
        "dataset_contract": DATASET_CONTRACT_NOTE,
        "blockers": blockers,
        "hardware_io_performed": False,
    }


def plan_report(loaded: LoadedProfile, start: Sequence[float]) -> dict[str, Any]:
    if loaded.profile is None:
        raise RecoveryProfileFileError(
            "cannot plan until every planning field is populated and validated"
        )
    if loaded.control is None:
        raise RecoveryProfileFileError(
            "cannot plan until every control field is populated and validated"
        )
    plan = plan_recording_recovery(start, loaded.profile)
    audit = audit_recovery_plan(plan, loaded.profile)
    return {
        **check_report(loaded),
        "start": plan.start,
        "home": loaded.profile.home,
        "required_frames": plan.required_frames,
        "nominal_duration_s": plan.required_frames / loaded.control.command_hz,
        "total_travel_rad": plan.total_travel_rad,
        "stages": plan.stages,
        "audit": audit,
        "waypoints": [asdict(step) for step in plan.steps],
    }


def _print_json(payload: Mapping[str, Any], *, stream=None) -> None:
    print(
        json.dumps(payload, indent=2, sort_keys=True),
        file=sys.stdout if stream is None else stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # This branch must remain before profile loading and before any potential
    # future hardware import.  Neither authorization flag can weaken it.
    if args.execute:
        _print_json(
            {
                "execution_ready": False,
                "hardware_io_performed": False,
                "reason": ACTIVE_EXECUTION_BLOCKER,
            },
            stream=sys.stderr,
        )
        return 2

    try:
        loaded = load_profile(
            args.profile,
            expected_sha256=args.expected_profile_sha256,
        )
        if args.check:
            report = check_report(loaded)
            _print_json(report)
            return 0 if loaded.profile_complete else 2
        report = plan_report(loaded, args.start)
        _print_json(report)
        return 0
    except (OSError, RecoveryPlanningError, RecoveryProfileFileError, ValueError) as exc:
        _print_json(
            {
                "execution_ready": False,
                "hardware_io_performed": False,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
