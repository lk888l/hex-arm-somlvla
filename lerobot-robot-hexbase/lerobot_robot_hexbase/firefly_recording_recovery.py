"""Offline-only recovery planning for Firefly recording boundaries.

This module deliberately has no robot, Zenoh, CAN, camera, or controller
imports.  It turns an already-observed seven-axis position into an auditable
sequence of waypoints; publishing those waypoints is outside its scope.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


AXIS_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "jaw_joint1.pos",
)

STAGE_ORDER = (
    "j4_clearance",
    "j2_home",
    "j3_home",
    "remaining_ee_home",
)

_STAGE_AXIS_INDEXES: dict[str, tuple[int, ...]] = {
    "j4_clearance": (3,),
    "j2_home": (1,),
    "j3_home": (2,),
    "remaining_ee_home": (0, 4, 5, 6),
}

_EPSILON = 1e-12
Vector7 = tuple[float, float, float, float, float, float, float]


class RecoveryPlanningError(ValueError):
    """A recovery request is unsafe or violates the fixed plan contract."""


def _vector7(raw: Any, label: str) -> Vector7:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RecoveryPlanningError(f"{label} must be a seven-value sequence")
    if len(raw) != 7:
        raise RecoveryPlanningError(f"{label} must contain exactly seven values")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryPlanningError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise RecoveryPlanningError(f"{label} contains NaN or infinity")
    return values  # type: ignore[return-value]


def _inside(
    values: Vector7,
    lower: Vector7,
    upper: Vector7,
) -> bool:
    return all(
        lo - _EPSILON <= value <= hi + _EPSILON
        for value, lo, hi in zip(values, lower, upper, strict=True)
    )


def _within_home(values: Vector7, profile: "RecoveryProfile") -> bool:
    return all(
        abs(value - target) <= tolerance + _EPSILON
        for value, target, tolerance in zip(
            values,
            profile.home,
            profile.home_tolerance,
            strict=True,
        )
    )


def stage_contract() -> tuple[dict[str, Any], ...]:
    """Return a fresh, JSON-friendly copy of the immutable stage contract."""

    return tuple(
        {
            "stage": stage,
            "axis_indexes": _STAGE_AXIS_INDEXES[stage],
            "axis_names": tuple(AXIS_NAMES[index] for index in _STAGE_AXIS_INDEXES[stage]),
        }
        for stage in STAGE_ORDER
    )


@dataclass(frozen=True, slots=True)
class RecoveryProfile:
    """All authority needed by the offline planner, expressed as fixed data.

    ``position_min`` and ``position_max`` are expected to be controller limits
    that have already been inset by the deployment's safety margin.  The
    planner never widens or discovers them from hardware.
    """

    home: Vector7
    home_tolerance: Vector7
    position_min: Vector7
    position_max: Vector7
    entry_min: Vector7
    entry_max: Vector7
    recovery_min: Vector7
    recovery_max: Vector7
    step_cap: Vector7
    j4_clearance: float
    max_frames: int
    max_total_travel_rad: float
    allow_positive_j3_motion: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "home",
            "home_tolerance",
            "position_min",
            "position_max",
            "entry_min",
            "entry_max",
            "recovery_min",
            "recovery_max",
            "step_cap",
        ):
            object.__setattr__(self, field_name, _vector7(getattr(self, field_name), field_name))

        try:
            clearance = float(self.j4_clearance)
            travel = float(self.max_total_travel_rad)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RecoveryPlanningError("scalar profile fields must be numeric") from exc
        if not math.isfinite(clearance):
            raise RecoveryPlanningError("j4_clearance must be finite")
        if not math.isfinite(travel) or travel <= 0.0:
            raise RecoveryPlanningError("max_total_travel_rad must be finite and positive")
        if type(self.max_frames) is not int or self.max_frames <= 0:
            raise RecoveryPlanningError("max_frames must be a positive integer")
        if type(self.allow_positive_j3_motion) is not bool:
            raise RecoveryPlanningError("allow_positive_j3_motion must be boolean")
        object.__setattr__(self, "j4_clearance", clearance)
        object.__setattr__(self, "max_total_travel_rad", travel)

        if any(value <= 0.0 for value in self.home_tolerance):
            raise RecoveryPlanningError("home_tolerance values must be positive")
        if any(value <= 0.0 for value in self.step_cap):
            raise RecoveryPlanningError("step_cap values must be positive")

        for index, name in enumerate(AXIS_NAMES):
            pos_lo, pos_hi = self.position_min[index], self.position_max[index]
            rec_lo, rec_hi = self.recovery_min[index], self.recovery_max[index]
            entry_lo, entry_hi = self.entry_min[index], self.entry_max[index]
            if pos_lo >= pos_hi:
                raise RecoveryPlanningError(f"invalid position bounds for {name}")
            if rec_lo >= rec_hi:
                raise RecoveryPlanningError(f"invalid recovery envelope for {name}")
            if entry_lo >= entry_hi:
                raise RecoveryPlanningError(f"invalid entry envelope for {name}")
            if rec_lo < pos_lo - _EPSILON or rec_hi > pos_hi + _EPSILON:
                raise RecoveryPlanningError(
                    f"recovery envelope exceeds inset position bounds for {name}"
                )
            if entry_lo < rec_lo - _EPSILON or entry_hi > rec_hi + _EPSILON:
                raise RecoveryPlanningError(
                    f"entry envelope exceeds recovery envelope for {name}"
                )
            target = self.home[index]
            tolerance = self.home_tolerance[index]
            if target - tolerance < entry_lo - _EPSILON or target + tolerance > entry_hi + _EPSILON:
                raise RecoveryPlanningError(
                    f"home tolerance does not fit inside entry envelope for {name}"
                )

        if not math.isclose(clearance, self.home[3], rel_tol=0.0, abs_tol=_EPSILON):
            raise RecoveryPlanningError(
                "j4_clearance must equal home joint_4 so recovery never lowers J4 later"
            )


@dataclass(frozen=True, slots=True)
class RecoveryStep:
    """One offline waypoint and the fixed axis authority for its stage."""

    stage: str
    allowed_axis_indexes: tuple[int, ...]
    allowed_axis_names: tuple[str, ...]
    positions: Vector7


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """An offline plan.  ``steps`` excludes the measured start sample."""

    start: Vector7
    steps: tuple[RecoveryStep, ...]
    total_travel_rad: float

    @property
    def required_frames(self) -> int:
        return len(self.steps)

    @property
    def final(self) -> Vector7:
        return self.start if not self.steps else self.steps[-1].positions

    @property
    def stages(self) -> tuple[str, ...]:
        result: list[str] = []
        for step in self.steps:
            if not result or result[-1] != step.stage:
                result.append(step.stage)
        return tuple(result)


def _stage_target(
    current: Vector7,
    profile: RecoveryProfile,
    stage: str,
) -> Vector7:
    target = list(current)
    if stage == "j4_clearance":
        target[3] = profile.j4_clearance
    elif stage == "j2_home":
        target[1] = profile.home[1]
    elif stage == "j3_home":
        target[2] = profile.home[2]
    elif stage == "remaining_ee_home":
        for index in _STAGE_AXIS_INDEXES[stage]:
            target[index] = profile.home[index]
    else:  # pragma: no cover - protected by the module-owned stage list
        raise RecoveryPlanningError(f"unknown recovery stage: {stage}")
    return tuple(target)  # type: ignore[return-value]


def _interpolate_stage(
    current: Vector7,
    target: Vector7,
    profile: RecoveryProfile,
    stage: str,
) -> list[RecoveryStep]:
    axes = _STAGE_AXIS_INDEXES[stage]
    ratios = [abs(target[index] - current[index]) / profile.step_cap[index] for index in axes]
    frames = max((math.ceil(ratio - _EPSILON) for ratio in ratios), default=0)
    if frames <= 0:
        return []
    names = tuple(AXIS_NAMES[index] for index in axes)
    steps: list[RecoveryStep] = []
    for frame in range(1, frames + 1):
        fraction = frame / frames
        values = list(current)
        for index in axes:
            values[index] = current[index] + (target[index] - current[index]) * fraction
        if frame == frames:
            values = list(target)
        steps.append(
            RecoveryStep(
                stage=stage,
                allowed_axis_indexes=axes,
                allowed_axis_names=names,
                positions=tuple(values),  # type: ignore[arg-type]
            )
        )
    return steps


def plan_recording_recovery(
    start: Sequence[float],
    profile: RecoveryProfile,
) -> RecoveryPlan:
    """Plan J4 -> J2 -> J3 -> remaining/EE without performing I/O.

    A start outside the wider recovery envelope is rejected rather than
    clipped.  Positive J3 travel is also rejected unless the profile carries
    explicit authorization.
    """

    start_vector = _vector7(start, "start")
    if not _inside(start_vector, profile.recovery_min, profile.recovery_max):
        raise RecoveryPlanningError(
            "start is outside the automatic recovery envelope; fail closed"
        )
    if (
        profile.home[2] > start_vector[2] + _EPSILON
        and not profile.allow_positive_j3_motion
    ):
        raise RecoveryPlanningError(
            "positive J3 recovery requires explicit profile authorization"
        )

    current = start_vector
    steps: list[RecoveryStep] = []
    for stage in STAGE_ORDER:
        target = _stage_target(current, profile, stage)
        stage_steps = _interpolate_stage(current, target, profile, stage)
        steps.extend(stage_steps)
        current = target

    previous = start_vector
    travel = 0.0
    for step in steps:
        travel += sum(
            abs(after - before)
            for before, after in zip(previous, step.positions, strict=True)
        )
        previous = step.positions

    if len(steps) > profile.max_frames:
        raise RecoveryPlanningError(
            f"recovery requires {len(steps)} frames, exceeding max_frames={profile.max_frames}"
        )
    if travel > profile.max_total_travel_rad + _EPSILON:
        raise RecoveryPlanningError(
            "recovery travel {:.9f} exceeds max_total_travel_rad={:.9f}".format(
                travel,
                profile.max_total_travel_rad,
            )
        )

    plan = RecoveryPlan(
        start=start_vector,
        steps=tuple(steps),
        total_travel_rad=travel,
    )
    audit = audit_recovery_plan(plan, profile)
    if audit["safe"] is not True:  # defensive check against planner regressions
        raise RecoveryPlanningError("generated recovery plan failed audit: " + audit["errors"][0])
    return plan


def audit_recovery_plan(
    plan: RecoveryPlan,
    profile: RecoveryProfile,
) -> dict[str, Any]:
    """Audit stage ordering, axis authority, monotonicity, and all hard caps."""

    errors: list[str] = []
    try:
        previous = _vector7(plan.start, "plan.start")
    except RecoveryPlanningError as exc:
        return {
            "safe": False,
            "errors": (str(exc),),
            "stage_contract": stage_contract(),
        }
    if not _inside(previous, profile.recovery_min, profile.recovery_max):
        errors.append("plan start is outside the recovery envelope")

    last_stage_index = -1
    observed_stages: list[str] = []
    travel = 0.0
    for frame_index, step in enumerate(plan.steps):
        if step.stage not in _STAGE_AXIS_INDEXES:
            errors.append(f"frame {frame_index} has an unknown stage")
            continue
        stage_index = STAGE_ORDER.index(step.stage)
        if stage_index < last_stage_index:
            errors.append(f"frame {frame_index} reverses stage order")
        if stage_index > last_stage_index:
            for prerequisite in STAGE_ORDER[:stage_index]:
                required = _stage_target(previous, profile, prerequisite)
                if any(
                    not math.isclose(a, b, rel_tol=0.0, abs_tol=_EPSILON)
                    for a, b in zip(previous, required, strict=True)
                ):
                    errors.append(
                        f"frame {frame_index} enters {step.stage} before {prerequisite} completes"
                    )
                    break
            observed_stages.append(step.stage)
            last_stage_index = max(last_stage_index, stage_index)

        expected_axes = _STAGE_AXIS_INDEXES[step.stage]
        expected_names = tuple(AXIS_NAMES[index] for index in expected_axes)
        if step.allowed_axis_indexes != expected_axes or step.allowed_axis_names != expected_names:
            errors.append(f"frame {frame_index} stage axis contract was modified")
        try:
            positions = _vector7(step.positions, f"frame {frame_index} positions")
        except RecoveryPlanningError as exc:
            errors.append(str(exc))
            continue
        if not _inside(positions, profile.recovery_min, profile.recovery_max):
            errors.append(f"frame {frame_index} leaves the recovery envelope")

        target = _stage_target(previous, profile, step.stage)
        for axis_index, (before, after) in enumerate(
            zip(previous, positions, strict=True)
        ):
            delta = after - before
            if abs(delta) > profile.step_cap[axis_index] + _EPSILON:
                errors.append(f"frame {frame_index} exceeds step cap on {AXIS_NAMES[axis_index]}")
            if abs(delta) > _EPSILON and axis_index not in expected_axes:
                errors.append(f"frame {frame_index} changes forbidden axis {AXIS_NAMES[axis_index]}")
            if axis_index in expected_axes:
                before_error = target[axis_index] - before
                after_error = target[axis_index] - after
                if before_error > _EPSILON and delta < -_EPSILON:
                    errors.append(f"frame {frame_index} is non-monotonic on {AXIS_NAMES[axis_index]}")
                elif before_error < -_EPSILON and delta > _EPSILON:
                    errors.append(f"frame {frame_index} is non-monotonic on {AXIS_NAMES[axis_index]}")
                if abs(after_error) > abs(before_error) + _EPSILON:
                    errors.append(f"frame {frame_index} moves away from its stage target")
                if before_error * after_error < -_EPSILON:
                    errors.append(f"frame {frame_index} overshoots its stage target")
            if axis_index == 2 and delta > _EPSILON and not profile.allow_positive_j3_motion:
                errors.append(f"frame {frame_index} has unauthorized positive J3 motion")
            travel += abs(delta)
        previous = positions

    if len(plan.steps) > profile.max_frames:
        errors.append("plan exceeds max_frames")
    if travel > profile.max_total_travel_rad + _EPSILON:
        errors.append("plan exceeds max_total_travel_rad")
    if not math.isclose(
        travel,
        plan.total_travel_rad,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append("reported total travel does not match the waypoints")
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=_EPSILON)
        for a, b in zip(previous, profile.home, strict=True)
    ):
        errors.append("plan does not finish at home")

    return {
        "safe": not errors,
        "errors": tuple(errors),
        "required_frames": len(plan.steps),
        "total_travel_rad": travel,
        "observed_stages": tuple(observed_stages),
        "stage_contract": stage_contract(),
    }


@dataclass(frozen=True, slots=True)
class RecordingBoundaryAssessment:
    """Continuous boundary gates computed from an offline position stream."""

    sample_count: int
    head_entry_consecutive: int
    tail_entry_consecutive: int
    final_home_consecutive: int
    head_entry_passed: bool
    tail_entry_passed: bool
    final_home_passed: bool
    first_outside_recovery_index: int | None
    fail_closed: bool
    ready_for_next_recording: bool
    episode_boundaries_passed: bool


def _prefix_count(flags: Sequence[bool]) -> int:
    count = 0
    for flag in flags:
        if not flag:
            break
        count += 1
    return count


def _suffix_count(flags: Sequence[bool]) -> int:
    return _prefix_count(tuple(reversed(flags)))


def assess_recording_boundaries(
    samples: Sequence[Sequence[float]],
    profile: RecoveryProfile,
    *,
    head_entry_frames: int,
    tail_entry_frames: int,
    final_home_frames: int,
) -> RecordingBoundaryAssessment:
    """Assess continuous start, tail-entry, and final-home gates.

    The function does not trim samples or infer that a single final home frame
    is stable.  Each requested window must be present and consecutive.
    """

    for label, value in (
        ("head_entry_frames", head_entry_frames),
        ("tail_entry_frames", tail_entry_frames),
        ("final_home_frames", final_home_frames),
    ):
        if type(value) is not int or value <= 0:
            raise RecoveryPlanningError(f"{label} must be a positive integer")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        raise RecoveryPlanningError("samples must be a non-empty sequence")

    vectors = tuple(_vector7(sample, f"samples[{index}]") for index, sample in enumerate(samples))
    in_entry = tuple(
        _inside(sample, profile.entry_min, profile.entry_max) for sample in vectors
    )
    in_home = tuple(_within_home(sample, profile) for sample in vectors)
    in_recovery = tuple(
        _inside(sample, profile.recovery_min, profile.recovery_max) for sample in vectors
    )

    head_count = _prefix_count(in_entry)
    tail_count = _suffix_count(in_entry)
    final_count = _suffix_count(in_home)
    outside_index = next(
        (index for index, inside in enumerate(in_recovery) if not inside),
        None,
    )
    fail_closed = outside_index is not None
    head_passed = head_count >= head_entry_frames
    tail_passed = tail_count >= tail_entry_frames
    final_passed = final_count >= final_home_frames
    ready = not fail_closed and tail_passed and final_passed
    boundaries = ready and head_passed
    return RecordingBoundaryAssessment(
        sample_count=len(vectors),
        head_entry_consecutive=head_count,
        tail_entry_consecutive=tail_count,
        final_home_consecutive=final_count,
        head_entry_passed=head_passed,
        tail_entry_passed=tail_passed,
        final_home_passed=final_passed,
        first_outside_recovery_index=outside_index,
        fail_closed=fail_closed,
        ready_for_next_recording=ready,
        episode_boundaries_passed=boundaries,
    )
