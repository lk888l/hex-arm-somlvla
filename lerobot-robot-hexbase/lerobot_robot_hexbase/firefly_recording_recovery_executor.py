"""Closed-loop core for a *previously safely acquired* recording recovery session.

This module does not open Zenoh, CAN, or controller sessions.  In particular,
it must not be mistaken for an acquisition wrapper: the current
``HexZenohFollower`` enters ACTIVE and publishes a measured-position hold while
connecting, before an application can apply a recording recovery envelope.
Until that pre-ACTIVE gate exists, the public recovery CLI remains plan-only.

The core is kept here so the staged feedback contract can be tested offline
and reused after commissioning.  A caller must provide an already connected
robot whose observation and send methods enforce freshness, ownership,
tracking, live controller limits, and watchdog checks.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .firefly_recording_recovery import (
    AXIS_NAMES,
    STAGE_ORDER,
    RecoveryPlan,
    RecoveryProfile,
    audit_recovery_plan,
    plan_recording_recovery,
)


_EPSILON = 1e-12
Vector7 = tuple[float, float, float, float, float, float, float]


class RecoveryExecutionError(RuntimeError):
    """A connected recovery run violated a hard execution gate."""


class ConnectedRecoveryRobot(Protocol):
    """Narrow interface required by :func:`execute_connected_recovery`."""

    config: Any
    action_features: Mapping[str, type]
    command_target: Mapping[str, float] | None
    is_connected: bool
    stopped_reason: str | None

    def get_observation(self) -> Mapping[str, Any]: ...

    def preflight_actions(self, actions: Sequence[Sequence[float]]) -> None: ...

    def send_action(self, action: Mapping[str, float]) -> Mapping[str, float]: ...

    def record_event(self, event: str, **fields: Any) -> None: ...

    def stop(self, reason: str) -> None: ...


def _vector7(raw: Sequence[float], label: str) -> Vector7:
    if isinstance(raw, (str, bytes)) or len(raw) != 7:
        raise ValueError(f"{label} must contain exactly seven values")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains NaN or infinity")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RecoveryControlContract:
    """Fixed feedback limits for one commissioned recovery profile."""

    candidate_lead_budget: Vector7
    command_hz: float = 30.0
    stall_frames: int = 90
    max_movement_control_frames: int = 1200
    settle_frames: int = 60
    required_home_frames: int = 5
    minimum_progress_rad: float = 0.0002

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_lead_budget",
            _vector7(self.candidate_lead_budget, "candidate_lead_budget"),
        )
        if any(value <= 0.0 for value in self.candidate_lead_budget):
            raise ValueError("candidate_lead_budget values must be positive")
        if not math.isfinite(self.command_hz) or not math.isclose(
            self.command_hz, 30.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("command_hz is fixed at 30 Hz")
        for name in (
            "stall_frames",
            "max_movement_control_frames",
            "settle_frames",
            "required_home_frames",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.required_home_frames > self.settle_frames:
            raise ValueError("required_home_frames must not exceed settle_frames")
        if (
            not math.isfinite(self.minimum_progress_rad)
            or self.minimum_progress_rad <= 0.0
        ):
            raise ValueError("minimum_progress_rad must be finite and positive")


@dataclass(frozen=True, slots=True)
class RecoveryExecutionResult:
    plan: RecoveryPlan
    control_frames: int
    hold_frames: int
    settle_frames: int
    consecutive_home_frames: int


def _positions_from_mapping(values: Mapping[str, Any], label: str) -> Vector7:
    missing = [name for name in AXIS_NAMES if name not in values]
    if missing:
        raise RecoveryExecutionError(f"{label} is missing axes: {missing}")
    return _vector7(tuple(values[name] for name in AXIS_NAMES), label)


def _inside_recovery_envelope(values: Vector7, profile: RecoveryProfile) -> bool:
    return all(
        lower - _EPSILON <= value <= upper + _EPSILON
        for value, lower, upper in zip(
            values,
            profile.recovery_min,
            profile.recovery_max,
            strict=True,
        )
    )


def _home_passes(values: Vector7, profile: RecoveryProfile) -> bool:
    return all(
        abs(value - target) <= tolerance + _EPSILON
        for value, target, tolerance in zip(
            values,
            profile.home,
            profile.home_tolerance,
            strict=True,
        )
    )


def _action(values: Vector7) -> dict[str, float]:
    return dict(zip(AXIS_NAMES, values, strict=True))


def _stop(robot: ConnectedRecoveryRobot, reason: str) -> None:
    try:
        robot.stop(reason)
    finally:
        stopped_reason = getattr(robot, "stopped_reason", None)
        raise RecoveryExecutionError(stopped_reason or reason)


def _require_connected_contract(
    robot: ConnectedRecoveryRobot,
    control: RecoveryControlContract,
) -> None:
    if not bool(getattr(robot, "is_connected", False)):
        raise RecoveryExecutionError("recovery robot is not connected")
    if getattr(getattr(robot, "config", None), "mode", None) != "execute":
        raise RecoveryExecutionError("recovery requires an execute-mode follower")
    if tuple(robot.action_features) != AXIS_NAMES:
        raise RecoveryExecutionError(
            "recovery requires the exact ordered seven-axis follower schema"
        )

    config = robot.config
    arm_limit = float(getattr(config, "arm_tracking_error_rad", float("nan")))
    ee_limit = float(getattr(config, "ee_tracking_error_rad", float("nan")))
    tracking_limits = (arm_limit,) * 6 + (ee_limit,)
    if not all(math.isfinite(value) and value > 0.0 for value in tracking_limits):
        raise RecoveryExecutionError("follower tracking limits are unavailable")
    if any(
        budget >= limit
        for budget, limit in zip(
            control.candidate_lead_budget,
            tracking_limits,
            strict=True,
        )
    ):
        raise RecoveryExecutionError(
            "candidate lead budgets must be strictly below follower tracking limits"
        )
    if not math.isclose(
        float(getattr(config, "command_hz", float("nan"))),
        control.command_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RecoveryExecutionError("recovery and follower command rates do not match")


def execute_connected_recovery(
    robot: ConnectedRecoveryRobot,
    profile: RecoveryProfile,
    control: RecoveryControlContract,
    *,
    stop_requested: Callable[[], bool] = lambda: False,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RecoveryExecutionResult:
    """Run a feedback-paced J4 -> J2 -> J3 -> remaining recovery.

    This function deliberately does not connect or disconnect ``robot``.  Its
    caller must have applied a commissioned recovery-envelope gate *before*
    ACTIVE.  Every observation is checked again against that envelope.  The
    supplied follower remains responsible for state/status freshness, session
    ownership, tracking-error, live-limit, and watchdog enforcement on every
    ``get_observation``/``send_action`` cycle.
    """

    _require_connected_contract(robot, control)
    reference = robot.command_target
    if reference is None:
        raise RecoveryExecutionError("connected follower has no measured startup hold target")
    start = _positions_from_mapping(reference, "startup hold target")
    if not _inside_recovery_envelope(start, profile):
        _stop(robot, "startup hold target is outside the commissioned recovery envelope")

    plan = plan_recording_recovery(start, profile)
    audit = audit_recovery_plan(plan, profile)
    if audit["safe"] is not True:
        _stop(robot, f"offline recovery plan failed audit: {audit['errors']}")
    robot.preflight_actions([plan.start, *(step.positions for step in plan.steps)])
    robot.record_event(
        "recording_recovery_begin",
        start=plan.start,
        home=profile.home,
        stage_order=STAGE_ORDER,
        required_frames=plan.required_frames,
        total_travel_rad=plan.total_travel_rad,
        candidate_lead_budget=control.candidate_lead_budget,
        stall_frames=control.stall_frames,
        max_movement_control_frames=control.max_movement_control_frames,
    )

    period_s = 1.0 / control.command_hz
    waypoint_index = 0
    control_frames = 0
    hold_frames = 0
    stall_count = 0
    best_error = math.inf
    last_target = plan.start
    current_stage: str | None = None

    while waypoint_index < len(plan.steps):
        if stop_requested():
            _stop(robot, "operator interrupted staged recording recovery")
        if control_frames >= control.max_movement_control_frames:
            _stop(
                robot,
                "staged recording recovery exceeded max_movement_control_frames="
                f"{control.max_movement_control_frames}",
            )

        cycle_start = monotonic()
        measured = _positions_from_mapping(robot.get_observation(), "measured position")
        if not _inside_recovery_envelope(measured, profile):
            _stop(robot, "measured position left the commissioned recovery envelope")

        errors = tuple(
            abs(actual - target)
            for actual, target in zip(measured, last_target, strict=True)
        )
        ready = all(
            error <= budget + _EPSILON
            for error, budget in zip(
                errors,
                control.candidate_lead_budget,
                strict=True,
            )
        )
        if ready:
            step = plan.steps[waypoint_index]
            if step.stage != current_stage:
                current_stage = step.stage
                robot.record_event(
                    "recording_recovery_stage_begin",
                    stage=current_stage,
                    waypoint_index=waypoint_index,
                    allowed_axis_names=step.allowed_axis_names,
                )
            command = step.positions
            waypoint_index += 1
            last_target = command
            stall_count = 0
            best_error = math.inf
        else:
            command = last_target
            hold_frames += 1
            worst_error = max(errors)
            if worst_error < best_error - control.minimum_progress_rad:
                best_error = worst_error
                stall_count = 0
            else:
                stall_count += 1
            if stall_count >= control.stall_frames:
                _stop(
                    robot,
                    "staged recording recovery stalled for "
                    f"{control.stall_frames} frames at waypoint "
                    f"{waypoint_index}/{len(plan.steps)}; errors_rad={errors}",
                )

        # HexZenohFollower.send_action performs the live state/status freshness,
        # session-holder, tracking, live-limit, step, and watchdog gates.
        robot.send_action(_action(command))
        control_frames += 1
        remaining = period_s - (monotonic() - cycle_start)
        if remaining > 0.0:
            sleep(remaining)

    home_pass_streak = 0
    settle_used = 0
    for settle_used in range(1, control.settle_frames + 1):
        if stop_requested():
            _stop(robot, "operator interrupted staged recording recovery settle")
        cycle_start = monotonic()
        measured = _positions_from_mapping(robot.get_observation(), "measured position")
        if not _inside_recovery_envelope(measured, profile):
            _stop(robot, "measured position left the recovery envelope while settling")
        home_pass_streak = home_pass_streak + 1 if _home_passes(measured, profile) else 0
        robot.send_action(_action(profile.home))
        control_frames += 1
        if home_pass_streak >= control.required_home_frames:
            robot.record_event(
                "recording_recovery_complete",
                home=profile.home,
                measured=measured,
                movement_waypoints=plan.required_frames,
                control_frames=control_frames,
                hold_frames=hold_frames,
                settle_frames=settle_used,
                consecutive_home_frames=home_pass_streak,
            )
            return RecoveryExecutionResult(
                plan=plan,
                control_frames=control_frames,
                hold_frames=hold_frames,
                settle_frames=settle_used,
                consecutive_home_frames=home_pass_streak,
            )
        remaining = period_s - (monotonic() - cycle_start)
        if remaining > 0.0:
            sleep(remaining)

    _stop(
        robot,
        "staged recording recovery did not settle at home within "
        f"{control.settle_frames} frames",
    )
    raise AssertionError("unreachable")
