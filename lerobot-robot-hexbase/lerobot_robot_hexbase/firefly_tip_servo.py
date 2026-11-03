"""Offline-only visual takeover and bounded tip-servo planning for Firefly Y6.

This module deliberately imports no robot, camera, Zenoh, CAN, or controller
code.  It consumes already-evaluated visual evidence and a caller-supplied FK
callable, then returns immutable decisions.  Publishing a decision is outside
its scope.

The planner owns no grasp semantics.  ``VISUAL_CLOSE_READY`` means only that
the arm must remain held while the existing grasp latch independently decides
whether and when to close the jaw.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
import math
from typing import Any

import numpy as np


Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
ForwardKinematics = Callable[[Sequence[float]], tuple[Any, Any]]

HARD_MAX_DESCENT_STEP_M = 0.002
HARD_MAX_TOTAL_DESCENT_M = 0.070
HARD_MAX_JOINT_STEP_RAD = 0.0435
HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD = 0.035
HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS = 20_000.0
HARD_MAX_OPEN_ONLY_JAW_TARGET_RAD = 0.41
HARD_OPEN_ONLY_CAPTURE_JAW_RAD = (0.285, 0.325)
_EPSILON = 1e-12


class TipServoError(ValueError):
    """A configuration or immutable input violates the servo contract."""


class TipServoMode(StrEnum):
    DISABLED = "disabled"
    ADVISORY = "advisory"
    EXECUTE = "execute"


class TipServoProbeProfile(StrEnum):
    """Fingerprintable motion authority after a visual takeover."""

    BOUNDED_DESCENT = "bounded_descent"
    OPEN_ONLY = "open_only"


class TipServoFrameStatus(StrEnum):
    FRESH_UNIQUE = "fresh_unique"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    STALE = "stale"
    FUTURE = "future"
    INVALID = "invalid"


class TipServoPhase(StrEnum):
    SEEKING = "seeking"
    OPENING = "opening"
    DESCENDING = "descending"
    PROBE_COMPLETE = "probe_complete"
    CLOSE_READY = "close_ready"
    FAULT_HOLD = "fault_hold"


class TipServoIntent(StrEnum):
    OBSERVE = "observe"
    HOLD = "hold"
    OPEN = "open"
    DESCEND = "descend"
    VISUAL_CLOSE_READY = "visual_close_ready"
    FAULT_HOLD = "fault_hold"


def _finite_tuple(raw: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TipServoError(f"{label} must be a {length}-value sequence")
    if len(raw) != length:
        raise TipServoError(f"{label} must contain exactly {length} values")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TipServoError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise TipServoError(f"{label} contains NaN or infinity")
    return values


def _vector3(raw: Any, label: str) -> Vector3:
    return _finite_tuple(raw, 3, label)  # type: ignore[return-value]


def _vector6(raw: Any, label: str) -> Vector6:
    return _finite_tuple(raw, 6, label)  # type: ignore[return-value]


def _matrix3(raw: Any, label: str, *, validate_rotation: bool = True) -> Matrix3:
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise TipServoError(f"{label} must be a 3x3 sequence")
    rows = tuple(_vector3(row, f"{label}[{index}]") for index, row in enumerate(raw))
    matrix = np.asarray(rows, dtype=np.float64)
    if validate_rotation:
        orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(matrix))
        if orthogonality_error > 1e-5 or not math.isclose(
            determinant, 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise TipServoError(
                f"{label} must be a finite proper rotation matrix; "
                f"orthogonality_error={orthogonality_error:.3e}, det={determinant:.9f}"
            )
    return rows  # type: ignore[return-value]


def _interval(raw: Any, label: str) -> tuple[float, float]:
    values = _finite_tuple(raw, 2, label)
    if not 0.0 <= values[0] < values[1] <= 1.0:
        raise TipServoError(f"{label} must be an increasing interval inside [0, 1]")
    return values  # type: ignore[return-value]


def _positive_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TipServoError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TipServoError(f"{label} must be finite and positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TipServoError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise TipServoError(f"{label} must be finite and nonnegative")
    return result


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TipServoError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TipServoConfig:
    """Reviewed image gates and bounded descent authority.

    The default coarse gate is derived from the successful fullcycle-v4
    pregrasp corridor.  The close-ready gate is intentionally narrower.  A
    deployment may reduce ``max_total_descent_m`` freely; increasing it above
    the default requires separate review but never beyond the module-owned
    70-mm hard ceiling.
    """

    mode: TipServoMode = TipServoMode.DISABLED
    probe_profile: TipServoProbeProfile = TipServoProbeProfile.BOUNDED_DESCENT
    coarse_x_norm: tuple[float, float] = (0.42, 0.50)
    coarse_y_norm: tuple[float, float] = (0.42, 0.56)
    # The coarse-area upper edge is exclusive.  This prevents the takeover
    # profile from silently overlapping the final approach gate.
    coarse_area_fraction: tuple[float, float] = (0.017, 0.029)
    tracking_x_norm: tuple[float, float] = (0.40, 0.52)
    tracking_y_norm: tuple[float, float] = (0.40, 0.62)
    tracking_area_fraction: tuple[float, float] = (0.014, 0.041)
    close_ready_x_norm: tuple[float, float] = (0.43, 0.48)
    close_ready_y_norm: tuple[float, float] = (0.51, 0.60)
    close_ready_area_fraction: tuple[float, float] = (0.0295, 0.040)

    takeover_confirm_frames: int = 6
    open_confirm_frames: int = 3
    close_ready_confirm_frames: int = 9
    probe_success_confirm_frames: int = 6
    area_drop_fault_frames: int = 3

    pregrasp_open_target_rad: float = 0.41
    pregrasp_open_ready_rad: float = 0.38
    # ``None`` preserves the original open-capture gate
    # (jaw >= pregrasp_open_ready_rad).  An explicit interval permits a
    # reviewed closed-jaw capture without changing that legacy default.
    coarse_capture_jaw_rad: tuple[float, float] | None = None
    max_arm_speed_rad_s: float = 0.10
    max_jaw_speed_rad_s: float = 0.05
    max_takeover_reference_error_rad: float = (
        HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD
    )
    arm_target_settle_rad: Vector6 = (0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
    opening_timeout_ms: float = 1500.0
    # OPEN_ONLY may leave the policy outside its demonstrated jaw timing while
    # SEEKING. Bound that exposure independently of the outer ACT deadline.
    seeking_timeout_ms: float = HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS

    descent_step_m: float = HARD_MAX_DESCENT_STEP_M
    max_total_descent_m: float = 0.025
    probe_success_area_fraction: float = 0.020
    area_drop_fault_fraction: float = 0.002
    vision_loss_timeout_ms: float = 250.0
    allow_close: bool = False

    def __post_init__(self) -> None:
        try:
            mode = TipServoMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise TipServoError("mode is unsupported") from exc
        object.__setattr__(self, "mode", mode)
        try:
            probe_profile = TipServoProbeProfile(self.probe_profile)
        except (TypeError, ValueError) as exc:
            raise TipServoError("probe_profile is unsupported") from exc
        object.__setattr__(self, "probe_profile", probe_profile)
        for name in (
            "coarse_x_norm",
            "coarse_y_norm",
            "coarse_area_fraction",
            "tracking_x_norm",
            "tracking_y_norm",
            "tracking_area_fraction",
            "close_ready_x_norm",
            "close_ready_y_norm",
            "close_ready_area_fraction",
        ):
            object.__setattr__(self, name, _interval(getattr(self, name), name))
        for name in (
            "takeover_confirm_frames",
            "open_confirm_frames",
            "close_ready_confirm_frames",
            "probe_success_confirm_frames",
            "area_drop_fault_frames",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

        for name in (
            "pregrasp_open_target_rad",
            "pregrasp_open_ready_rad",
            "max_arm_speed_rad_s",
            "max_jaw_speed_rad_s",
            "probe_success_area_fraction",
            "area_drop_fault_fraction",
            "vision_loss_timeout_ms",
            "opening_timeout_ms",
            "seeking_timeout_ms",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "descent_step_m",
            _nonnegative_finite(self.descent_step_m, "descent_step_m"),
        )
        object.__setattr__(
            self,
            "max_total_descent_m",
            _nonnegative_finite(self.max_total_descent_m, "max_total_descent_m"),
        )
        object.__setattr__(
            self,
            "max_takeover_reference_error_rad",
            _nonnegative_finite(
                self.max_takeover_reference_error_rad,
                "max_takeover_reference_error_rad",
            ),
        )
        if self.coarse_capture_jaw_rad is not None:
            object.__setattr__(
                self,
                "coarse_capture_jaw_rad",
                _interval(self.coarse_capture_jaw_rad, "coarse_capture_jaw_rad"),
            )
        object.__setattr__(
            self,
            "arm_target_settle_rad",
            _vector6(self.arm_target_settle_rad, "arm_target_settle_rad"),
        )
        if any(value <= 0.0 for value in self.arm_target_settle_rad):
            raise TipServoError("arm_target_settle_rad values must be positive")
        if self.pregrasp_open_ready_rad > self.pregrasp_open_target_rad:
            raise TipServoError("pregrasp_open_ready_rad must not exceed the open target")
        if self.descent_step_m > HARD_MAX_DESCENT_STEP_M + _EPSILON:
            raise TipServoError(
                f"descent_step_m exceeds the {HARD_MAX_DESCENT_STEP_M:.3f}-m hard limit"
            )
        if self.max_total_descent_m > HARD_MAX_TOTAL_DESCENT_M + _EPSILON:
            raise TipServoError(
                "max_total_descent_m exceeds the 0.070-m hard limit"
            )
        if (
            self.max_takeover_reference_error_rad
            > HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD + _EPSILON
        ):
            raise TipServoError(
                "max_takeover_reference_error_rad exceeds the 0.035-rad hard limit"
            )
        if self.area_drop_fault_fraction >= self.tracking_area_fraction[1]:
            raise TipServoError("area_drop_fault_fraction is too large for the tracking gate")
        if not _inside(self.probe_success_area_fraction, self.tracking_area_fraction):
            raise TipServoError(
                "probe_success_area_fraction must fit inside tracking_area_fraction"
            )
        if type(self.allow_close) is not bool:
            raise TipServoError("allow_close must be boolean")
        if self.probe_profile is TipServoProbeProfile.OPEN_ONLY:
            if not math.isclose(
                self.seeking_timeout_ms,
                HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS,
                rel_tol=0.0,
                abs_tol=_EPSILON,
            ):
                raise TipServoError(
                    "open_only seeking_timeout_ms must be exactly 20000 ms"
                )
            if (
                self.pregrasp_open_target_rad
                > HARD_MAX_OPEN_ONLY_JAW_TARGET_RAD + _EPSILON
            ):
                raise TipServoError(
                    "open_only pregrasp_open_target_rad exceeds the 0.41-rad hard limit"
                )
            if self.descent_step_m > _EPSILON:
                raise TipServoError(
                    "open_only probe_profile requires descent_step_m=0"
                )
            if self.max_total_descent_m > _EPSILON:
                raise TipServoError(
                    "open_only probe_profile requires max_total_descent_m=0"
                )
            if self.allow_close:
                raise TipServoError("open_only probe_profile cannot allow_close")
            if self.coarse_capture_jaw_rad is None:
                raise TipServoError(
                    "open_only probe_profile requires coarse_capture_jaw_rad"
                )
            if (
                self.coarse_capture_jaw_rad[1]
                >= self.pregrasp_open_ready_rad - _EPSILON
            ):
                raise TipServoError(
                    "open_only coarse_capture_jaw_rad must remain below "
                    "pregrasp_open_ready_rad"
                )
        else:
            if self.descent_step_m <= _EPSILON:
                raise TipServoError(
                    "bounded_descent probe_profile requires positive descent_step_m"
                )
            if self.max_total_descent_m <= _EPSILON:
                raise TipServoError(
                    "bounded_descent probe_profile requires positive "
                    "max_total_descent_m"
                )
            if self.descent_step_m > self.max_total_descent_m + _EPSILON:
                raise TipServoError(
                    "descent_step_m must not exceed max_total_descent_m"
                )

        for axis in ("x", "y"):
            coarse = getattr(self, f"coarse_{axis}_norm")
            tracking = getattr(self, f"tracking_{axis}_norm")
            close = getattr(self, f"close_ready_{axis}_norm")
            if coarse[0] < tracking[0] or coarse[1] > tracking[1]:
                raise TipServoError(f"coarse_{axis}_norm must fit inside tracking_{axis}_norm")
            if close[0] < tracking[0] or close[1] > tracking[1]:
                raise TipServoError(
                    f"close_ready_{axis}_norm must fit inside tracking_{axis}_norm"
                )
        for name, interval in (
            ("coarse_area_fraction", self.coarse_area_fraction),
            ("close_ready_area_fraction", self.close_ready_area_fraction),
        ):
            if (
                interval[0] < self.tracking_area_fraction[0]
                or interval[1] > self.tracking_area_fraction[1]
            ):
                raise TipServoError(f"{name} must fit inside tracking_area_fraction")
        if self.coarse_area_fraction[1] > self.close_ready_area_fraction[0] + _EPSILON:
            raise TipServoError(
                "coarse_area_fraction must end no later than close-ready area begins"
            )


@dataclass(frozen=True, slots=True)
class DlsIkConfig:
    """Numerical full-pose DLS limits for one feedback-paced arm step."""

    joint_lower_rad: Vector6 = (-2.86, -1.57, 0.0, -1.57, -1.54, -2.79)
    joint_upper_rad: Vector6 = (2.86, 2.09, 3.14, 1.57, 1.54, 2.79)
    joint_step_limit_rad: Vector6 = (0.012, 0.012, 0.010, 0.012, 0.010, 0.012)
    joint_limit_margin_rad: float = 0.005
    # The deployed FK intentionally rounds every operation to float32.  A
    # 1e-3-rad stencil remains small relative to every default joint step but
    # avoids amplifying float32 quantization in the rotational Jacobian.
    finite_difference_rad: float = 1e-3
    damping: float = 1e-3
    orientation_weight_m_per_rad: float = 0.10
    condition_max: float = 5000.0
    residual_position_m: float = 0.00025
    residual_orientation_rad: float = 0.003
    max_iterations: int = 16

    def __post_init__(self) -> None:
        for name in ("joint_lower_rad", "joint_upper_rad", "joint_step_limit_rad"):
            object.__setattr__(self, name, _vector6(getattr(self, name), name))
        for name in (
            "finite_difference_rad",
            "damping",
            "orientation_weight_m_per_rad",
            "condition_max",
            "residual_position_m",
            "residual_orientation_rad",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "joint_limit_margin_rad",
            _nonnegative_finite(self.joint_limit_margin_rad, "joint_limit_margin_rad"),
        )
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations"),
        )
        if self.condition_max <= 1.0:
            raise TipServoError("condition_max must be greater than one")
        for index, (lower, upper, step) in enumerate(
            zip(
                self.joint_lower_rad,
                self.joint_upper_rad,
                self.joint_step_limit_rad,
                strict=True,
            )
        ):
            if lower >= upper:
                raise TipServoError(f"invalid joint limits at axis {index}")
            if step <= 0.0 or step > HARD_MAX_JOINT_STEP_RAD + _EPSILON:
                raise TipServoError(
                    f"joint_step_limit_rad[{index}] must be in (0, {HARD_MAX_JOINT_STEP_RAD}]"
                )
            if 2.0 * self.joint_limit_margin_rad >= upper - lower:
                raise TipServoError(f"joint limit margin consumes axis {index} range")
        if self.finite_difference_rad >= min(self.joint_step_limit_rad):
            raise TipServoError("finite_difference_rad must be below every joint step limit")


@dataclass(frozen=True, slots=True)
class TipServoObservation:
    """One already-oriented and already-timestamped control observation."""

    frame_id: int
    evaluated_at_ns: int
    frame_status: TipServoFrameStatus
    candidate_count: int
    red_centroid_norm: tuple[float, float] | None
    red_area_fraction: float | None
    arm_q_rad: Vector6
    # Last arm target accepted by the controller.  It is intentionally
    # separate from measured arm_q_rad: takeover geometry is evaluated at the
    # measurement while every post-takeover arm command holds this reference.
    arm_reference_q_rad: Vector6
    # Last jaw target accepted by the controller. OPEN_ONLY binds its
    # immutable capture target from this value on the first evaluated cycle;
    # measured jaw feedback is never allowed to redefine command authority.
    jaw_reference_rad: float
    jaw_rad: float
    arm_speed_rad_s: float
    jaw_speed_rad_s: float

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise TipServoError("frame_id must be a nonnegative integer")
        if type(self.evaluated_at_ns) is not int or self.evaluated_at_ns < 0:
            raise TipServoError("evaluated_at_ns must be a nonnegative integer")
        try:
            status = TipServoFrameStatus(self.frame_status)
        except (TypeError, ValueError) as exc:
            raise TipServoError("frame_status is unsupported") from exc
        object.__setattr__(self, "frame_status", status)
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise TipServoError("candidate_count must be a nonnegative integer")
        if self.red_centroid_norm is None:
            centroid = None
        else:
            centroid = _finite_tuple(self.red_centroid_norm, 2, "red_centroid_norm")
            if not all(0.0 <= value <= 1.0 for value in centroid):
                raise TipServoError("red_centroid_norm values must be inside [0, 1]")
        object.__setattr__(self, "red_centroid_norm", centroid)
        if self.red_area_fraction is None:
            area = None
        else:
            area = _nonnegative_finite(self.red_area_fraction, "red_area_fraction")
            if area > 1.0:
                raise TipServoError("red_area_fraction must not exceed one")
        object.__setattr__(self, "red_area_fraction", area)
        if self.candidate_count > 0 and (centroid is None or area is None):
            raise TipServoError("positive candidate_count requires red centroid and area")
        object.__setattr__(self, "arm_q_rad", _vector6(self.arm_q_rad, "arm_q_rad"))
        object.__setattr__(
            self,
            "arm_reference_q_rad",
            _vector6(self.arm_reference_q_rad, "arm_reference_q_rad"),
        )
        object.__setattr__(
            self,
            "jaw_reference_rad",
            _nonnegative_finite(self.jaw_reference_rad, "jaw_reference_rad"),
        )
        object.__setattr__(self, "jaw_rad", _nonnegative_finite(self.jaw_rad, "jaw_rad"))
        object.__setattr__(
            self,
            "arm_speed_rad_s",
            _nonnegative_finite(self.arm_speed_rad_s, "arm_speed_rad_s"),
        )
        object.__setattr__(
            self,
            "jaw_speed_rad_s",
            _nonnegative_finite(self.jaw_speed_rad_s, "jaw_speed_rad_s"),
        )


@dataclass(frozen=True, slots=True)
class TipServoState:
    """All state required to advance the pure servo transition function."""

    phase: TipServoPhase = TipServoPhase.SEEKING
    capture_jaw_target_rad: float | None = None
    seeking_started_at_ns: int | None = None
    takeover_q_rad: Vector6 | None = None
    takeover_position_m: Vector3 | None = None
    takeover_rotation: Matrix3 | None = None
    last_arm_target_q_rad: Vector6 | None = None
    takeover_jaw_target_rad: float | None = None
    takeover_started_at_ns: int | None = None
    open_commanded: bool = False
    commanded_descent_m: float = 0.0
    coarse_streak: int = 0
    open_streak: int = 0
    close_ready_streak: int = 0
    probe_success_streak: int = 0
    area_drop_streak: int = 0
    peak_red_area_fraction: float | None = None
    last_fresh_frame_id: int | None = None
    last_evaluated_at_ns: int | None = None
    vision_loss_since_ns: int | None = None
    fault_reason: str | None = None

    def __post_init__(self) -> None:
        try:
            phase = TipServoPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise TipServoError("phase is unsupported") from exc
        object.__setattr__(self, "phase", phase)
        if self.capture_jaw_target_rad is not None:
            object.__setattr__(
                self,
                "capture_jaw_target_rad",
                _nonnegative_finite(
                    self.capture_jaw_target_rad,
                    "capture_jaw_target_rad",
                ),
            )
        if (self.capture_jaw_target_rad is None) != (
            self.seeking_started_at_ns is None
        ):
            raise TipServoError(
                "capture_jaw_target_rad and seeking_started_at_ns must be both present or both absent"
            )
        for name in ("takeover_q_rad", "last_arm_target_q_rad"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector6(value, name))
        if self.takeover_position_m is not None:
            object.__setattr__(
                self,
                "takeover_position_m",
                _vector3(self.takeover_position_m, "takeover_position_m"),
            )
        if self.takeover_rotation is not None:
            object.__setattr__(
                self,
                "takeover_rotation",
                _matrix3(self.takeover_rotation, "takeover_rotation"),
            )
        if self.takeover_jaw_target_rad is not None:
            object.__setattr__(
                self,
                "takeover_jaw_target_rad",
                _nonnegative_finite(
                    self.takeover_jaw_target_rad,
                    "takeover_jaw_target_rad",
                ),
            )
        if type(self.open_commanded) is not bool:
            raise TipServoError("open_commanded must be boolean")
        geometry = (
            self.takeover_q_rad,
            self.takeover_position_m,
            self.takeover_rotation,
            self.last_arm_target_q_rad,
            self.takeover_jaw_target_rad,
            self.takeover_started_at_ns,
        )
        if any(value is None for value in geometry) != all(
            value is None for value in geometry
        ):
            raise TipServoError(
                "takeover geometry, targets, and start time must be all present "
                "or all absent"
            )
        descent = _nonnegative_finite(self.commanded_descent_m, "commanded_descent_m")
        if descent > HARD_MAX_TOTAL_DESCENT_M + _EPSILON:
            raise TipServoError("commanded_descent_m exceeds the module hard limit")
        object.__setattr__(self, "commanded_descent_m", descent)
        for name in (
            "coarse_streak",
            "open_streak",
            "close_ready_streak",
            "probe_success_streak",
            "area_drop_streak",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TipServoError(f"{name} must be a nonnegative integer")
        if self.peak_red_area_fraction is not None:
            peak = _nonnegative_finite(
                self.peak_red_area_fraction, "peak_red_area_fraction"
            )
            if peak > 1.0:
                raise TipServoError("peak_red_area_fraction must not exceed one")
            object.__setattr__(self, "peak_red_area_fraction", peak)
        for name in (
            "last_fresh_frame_id",
            "last_evaluated_at_ns",
            "vision_loss_since_ns",
            "seeking_started_at_ns",
            "takeover_started_at_ns",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise TipServoError(f"{name} must be a nonnegative integer or None")
        if phase is TipServoPhase.SEEKING and any(value is not None for value in geometry):
            raise TipServoError("SEEKING state must not contain takeover geometry")
        if phase is TipServoPhase.SEEKING and self.open_commanded:
            raise TipServoError("SEEKING state cannot have open_commanded")
        if phase not in (TipServoPhase.SEEKING, TipServoPhase.FAULT_HOLD) and any(
            value is None for value in geometry
        ):
            raise TipServoError("active servo state requires takeover geometry")
        if self.open_commanded and any(value is None for value in geometry):
            raise TipServoError("open_commanded requires takeover geometry")
        if phase in (
            TipServoPhase.DESCENDING,
            TipServoPhase.PROBE_COMPLETE,
            TipServoPhase.CLOSE_READY,
        ) and not self.open_commanded:
            raise TipServoError(f"{phase.value} state requires open_commanded")
        if phase is TipServoPhase.FAULT_HOLD and not self.fault_reason:
            raise TipServoError("FAULT_HOLD state requires fault_reason")
        if phase is not TipServoPhase.FAULT_HOLD and self.fault_reason is not None:
            raise TipServoError("non-fault state must not contain fault_reason")


@dataclass(frozen=True, slots=True)
class DlsIkResult:
    ok: bool
    q_rad: Vector6 | None
    reason: str | None
    iterations: int
    condition_number: float | None
    position_residual_m: float | None
    orientation_residual_rad: float | None

    def telemetry(self) -> dict[str, Any]:
        def finite_or_none(value: float | None) -> float | None:
            return (
                None
                if value is None or not math.isfinite(float(value))
                else float(value)
            )

        return {
            "ok": bool(self.ok),
            "q_rad": None if self.q_rad is None else [float(value) for value in self.q_rad],
            "reason": self.reason,
            "iterations": int(self.iterations),
            "condition_number": finite_or_none(self.condition_number),
            "position_residual_m": finite_or_none(self.position_residual_m),
            "orientation_residual_rad": finite_or_none(
                self.orientation_residual_rad
            ),
        }


@dataclass(frozen=True, slots=True)
class TipServoDecision:
    mode: TipServoMode
    phase_before: TipServoPhase
    phase_after: TipServoPhase
    intent: TipServoIntent
    event: str | None
    fault_reason: str | None
    proposed_arm_target_q_rad: Vector6 | None
    commanded_arm_target_q_rad: Vector6 | None
    proposed_jaw_target_rad: float | None
    commanded_jaw_target_rad: float | None
    proposed_descent_step_m: float
    commanded_descent_m: float
    coarse_streak: int
    open_streak: int
    close_ready_streak: int
    probe_success_streak: int
    area_drop_streak: int
    vision_loss_elapsed_ms: float
    would_close: bool
    close_authorized: bool
    ik_result: DlsIkResult | None

    def telemetry(self) -> dict[str, Any]:
        """Return only JSON primitive containers and scalar values."""

        return {
            "mode": self.mode.value,
            "phase_before": self.phase_before.value,
            "phase_after": self.phase_after.value,
            "intent": self.intent.value,
            "event": self.event,
            "fault_reason": self.fault_reason,
            "proposed_arm_target_q_rad": (
                None
                if self.proposed_arm_target_q_rad is None
                else [float(value) for value in self.proposed_arm_target_q_rad]
            ),
            "commanded_arm_target_q_rad": (
                None
                if self.commanded_arm_target_q_rad is None
                else [float(value) for value in self.commanded_arm_target_q_rad]
            ),
            "proposed_jaw_target_rad": self.proposed_jaw_target_rad,
            "commanded_jaw_target_rad": self.commanded_jaw_target_rad,
            "proposed_descent_step_m": float(self.proposed_descent_step_m),
            "commanded_descent_m": float(self.commanded_descent_m),
            "coarse_streak": int(self.coarse_streak),
            "open_streak": int(self.open_streak),
            "close_ready_streak": int(self.close_ready_streak),
            "probe_success_streak": int(self.probe_success_streak),
            "area_drop_streak": int(self.area_drop_streak),
            "vision_loss_elapsed_ms": float(self.vision_loss_elapsed_ms),
            "would_close": bool(self.would_close),
            "close_authorized": bool(self.close_authorized),
            "ik": None if self.ik_result is None else self.ik_result.telemetry(),
        }


class _FkEvaluationError(RuntimeError):
    pass


def _evaluate_fk(fk: ForwardKinematics, q_rad: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = fk(tuple(float(value) for value in q_rad))
    except Exception as exc:  # caller-owned FK must fail closed
        raise _FkEvaluationError(f"fk_exception_{type(exc).__name__}") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise _FkEvaluationError("fk_result_shape_invalid")
    try:
        position = np.asarray(_vector3(raw[0], "fk_position"), dtype=np.float64)
        rotation = np.asarray(_matrix3(raw[1], "fk_rotation"), dtype=np.float64)
    except TipServoError as exc:
        raise _FkEvaluationError("fk_result_invalid") from exc
    return position, rotation


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-6:
        symmetric = 0.5 * (rotation + np.eye(3))
        values, vectors = np.linalg.eigh(symmetric)
        axis = vectors[:, int(np.argmax(values))]
        norm = float(np.linalg.norm(axis))
        if norm <= _EPSILON:
            raise _FkEvaluationError("rotation_log_singular")
        axis = axis / norm
        if float(np.dot(axis, skew)) < 0.0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * skew


def _pose_error(
    position: np.ndarray,
    rotation: np.ndarray,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return desired_position - position, _rotation_vector(desired_rotation @ rotation.T)


def _failure_result(
    reason: str,
    *,
    q_rad: Vector6 | None,
    iterations: int = 0,
    condition_number: float | None = None,
    position_residual_m: float | None = None,
    orientation_residual_rad: float | None = None,
) -> DlsIkResult:
    return DlsIkResult(
        ok=False,
        q_rad=q_rad,
        reason=reason,
        iterations=iterations,
        condition_number=condition_number,
        position_residual_m=position_residual_m,
        orientation_residual_rad=orientation_residual_rad,
    )


def _numeric_jacobian(
    q: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    fk: ForwardKinematics,
    lower: np.ndarray,
    upper: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for index in range(6):
        forward_room = float(upper[index] - q[index])
        backward_room = float(q[index] - lower[index])
        if forward_room >= epsilon and backward_room >= epsilon:
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[index] += epsilon
            q_minus[index] -= epsilon
            p_plus, r_plus = _evaluate_fk(fk, q_plus)
            p_minus, r_minus = _evaluate_fk(fk, q_minus)
            jacobian[:3, index] = (p_plus - p_minus) / (2.0 * epsilon)
            jacobian[3:, index] = _rotation_vector(r_plus @ r_minus.T) / (
                2.0 * epsilon
            )
        elif forward_room >= epsilon:
            q_plus = q.copy()
            q_plus[index] += epsilon
            p_plus, r_plus = _evaluate_fk(fk, q_plus)
            jacobian[:3, index] = (p_plus - position) / epsilon
            jacobian[3:, index] = _rotation_vector(r_plus @ rotation.T) / epsilon
        elif backward_room >= epsilon:
            q_minus = q.copy()
            q_minus[index] -= epsilon
            p_minus, r_minus = _evaluate_fk(fk, q_minus)
            jacobian[:3, index] = (position - p_minus) / epsilon
            jacobian[3:, index] = _rotation_vector(rotation @ r_minus.T) / epsilon
        else:
            raise _FkEvaluationError(f"joint_{index + 1}_has_no_jacobian_room")
    if not np.isfinite(jacobian).all():
        raise _FkEvaluationError("jacobian_nonfinite")
    return jacobian


def solve_full_pose_dls(
    q_rad: Sequence[float],
    desired_position_m: Sequence[float],
    desired_rotation: Sequence[Sequence[float]],
    *,
    fk: ForwardKinematics,
    config: DlsIkConfig = DlsIkConfig(),
) -> DlsIkResult:
    """Solve one bounded, full-pose numerical DLS update.

    The returned joint vector is always inside both the inset joint limits and
    the six per-call step limits around ``q_rad``.  A singular Jacobian,
    caller-owned FK error, or residual outside the configured gates returns a
    failed result instead of a best-effort target.
    """

    try:
        q0_tuple = _vector6(q_rad, "q_rad")
    except TipServoError:
        return _failure_result("input_q_invalid", q_rad=None)
    try:
        desired_position = np.asarray(
            _vector3(desired_position_m, "desired_position_m"), dtype=np.float64
        )
        desired_rotation_array = np.asarray(
            _matrix3(desired_rotation, "desired_rotation"), dtype=np.float64
        )
    except TipServoError:
        return _failure_result("desired_pose_invalid", q_rad=q0_tuple)

    q0 = np.asarray(q0_tuple, dtype=np.float64)
    lower = np.asarray(config.joint_lower_rad, dtype=np.float64) + config.joint_limit_margin_rad
    upper = np.asarray(config.joint_upper_rad, dtype=np.float64) - config.joint_limit_margin_rad
    if np.any(q0 < lower - _EPSILON) or np.any(q0 > upper + _EPSILON):
        return _failure_result("input_q_outside_inset_joint_limits", q_rad=q0_tuple)
    step = np.asarray(config.joint_step_limit_rad, dtype=np.float64)
    cycle_lower = np.maximum(lower, q0 - step)
    cycle_upper = np.minimum(upper, q0 + step)
    q = q0.copy()
    last_condition: float | None = None
    clipped = False

    for iteration in range(config.max_iterations + 1):
        try:
            position, rotation = _evaluate_fk(fk, q)
            position_error, orientation_error = _pose_error(
                position, rotation, desired_position, desired_rotation_array
            )
        except _FkEvaluationError as exc:
            return _failure_result(
                str(exc),
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=last_condition,
            )
        position_residual = float(np.linalg.norm(position_error))
        orientation_residual = float(np.linalg.norm(orientation_error))
        if (
            position_residual <= config.residual_position_m + _EPSILON
            and orientation_residual <= config.residual_orientation_rad + _EPSILON
        ):
            return DlsIkResult(
                ok=True,
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                reason=None,
                iterations=iteration,
                condition_number=last_condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        if iteration == config.max_iterations:
            reason = "joint_or_step_limit_residual" if clipped else "residual_gate_failed"
            return _failure_result(
                reason,
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=last_condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        try:
            jacobian = _numeric_jacobian(
                q,
                position,
                rotation,
                fk=fk,
                lower=lower,
                upper=upper,
                epsilon=config.finite_difference_rad,
            )
        except _FkEvaluationError as exc:
            return _failure_result(
                str(exc),
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=last_condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:, :] *= config.orientation_weight_m_per_rad
        singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
        smallest = float(singular_values[-1])
        condition = math.inf if smallest <= _EPSILON else float(singular_values[0] / smallest)
        last_condition = condition
        if not math.isfinite(condition) or condition > config.condition_max:
            return _failure_result(
                "jacobian_condition_gate",
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        weighted_error = np.concatenate(
            (position_error, config.orientation_weight_m_per_rad * orientation_error)
        )
        system = (
            weighted_jacobian @ weighted_jacobian.T
            + (config.damping**2) * np.eye(6)
        )
        try:
            delta = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
        except np.linalg.LinAlgError:
            return _failure_result(
                "dls_linear_solve_failed",
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        if not np.isfinite(delta).all():
            return _failure_result(
                "dls_delta_nonfinite",
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        unconstrained = q + delta
        candidate = np.clip(unconstrained, cycle_lower, cycle_upper)
        clipped = clipped or not np.allclose(candidate, unconstrained, rtol=0.0, atol=1e-15)
        if float(np.linalg.norm(candidate - q)) <= _EPSILON:
            return _failure_result(
                "joint_or_step_limit_residual" if clipped else "dls_no_progress",
                q_rad=tuple(float(value) for value in q),  # type: ignore[arg-type]
                iterations=iteration,
                condition_number=condition,
                position_residual_m=position_residual,
                orientation_residual_rad=orientation_residual,
            )
        q = candidate

    raise AssertionError("unreachable DLS loop exit")


def _inside(value: float, interval: tuple[float, float]) -> bool:
    return interval[0] - _EPSILON <= value <= interval[1] + _EPSILON


def _validated_open_only_capture_jaw_target(
    value: Any,
    *,
    config: TipServoConfig,
) -> float:
    if config.probe_profile is not TipServoProbeProfile.OPEN_ONLY:
        raise TipServoError("capture jaw validation requires OPEN_ONLY")
    capture = _nonnegative_finite(value, "capture_jaw_target_rad")
    configured = config.coarse_capture_jaw_rad
    if configured is None:  # pragma: no cover - TipServoConfig invariant
        raise TipServoError("OPEN_ONLY has no configured capture jaw interval")
    if not _inside(capture, configured) or not _inside(
        capture, HARD_OPEN_ONLY_CAPTURE_JAW_RAD
    ):
        raise TipServoError(
            "OPEN_ONLY capture jaw target must be inside the hard [0.285, 0.325] rad interval"
        )
    return capture


def resolve_open_only_jaw_target(
    *,
    capture_jaw_target_rad: float,
    previous_accepted_jaw_target_rad: float,
    open_commanded: bool,
    config: TipServoConfig,
    ee_step_limit_rad: float,
) -> float:
    """Resolve the only jaw target OPEN_ONLY may publish this cycle.

    Before ``open_commanded`` the accepted target must remain exactly at the
    immutable capture value. Afterwards this function permits only a positive,
    step-limited move toward the hard-bounded open target. Any already-accepted
    early-open, close, or out-of-envelope target is rejected instead of being
    corrected with a closing command.
    """

    capture = _validated_open_only_capture_jaw_target(
        capture_jaw_target_rad,
        config=config,
    )
    previous = _nonnegative_finite(
        previous_accepted_jaw_target_rad,
        "previous_accepted_jaw_target_rad",
    )
    step_limit = _positive_finite(ee_step_limit_rad, "ee_step_limit_rad")
    if type(open_commanded) is not bool:
        raise TipServoError("open_commanded must be boolean")
    open_target = float(config.pregrasp_open_target_rad)
    if open_target > HARD_MAX_OPEN_ONLY_JAW_TARGET_RAD + _EPSILON:
        raise TipServoError("OPEN_ONLY open target exceeds the 0.41-rad hard limit")
    if previous < capture - _EPSILON or previous > open_target + _EPSILON:
        raise TipServoError(
            "previous accepted jaw target is outside the OPEN_ONLY no-close envelope"
        )
    if not open_commanded:
        if not math.isclose(previous, capture, rel_tol=0.0, abs_tol=_EPSILON):
            raise TipServoError(
                "OPEN_ONLY jaw left capture before visual open authorization"
            )
        return capture
    return min(open_target, previous + step_limit)


def _open_only_planner_jaw_target(
    state: TipServoState,
    config: TipServoConfig,
    *,
    fallback_reference_rad: float | None = None,
) -> float:
    if state.open_commanded:
        return float(config.pregrasp_open_target_rad)
    if state.capture_jaw_target_rad is not None:
        return float(state.capture_jaw_target_rad)
    if fallback_reference_rad is not None:
        # A malformed/unbound state may only hold the already accepted target
        # while its caller transitions to FAULT_HOLD.
        return _nonnegative_finite(fallback_reference_rad, "jaw_reference_rad")
    raise TipServoError("OPEN_ONLY state has no immutable capture jaw target")


def _single_red(observation: TipServoObservation) -> bool:
    return (
        observation.candidate_count == 1
        and observation.red_centroid_norm is not None
        and observation.red_area_fraction is not None
    )


def _coarse_gate(observation: TipServoObservation, config: TipServoConfig) -> bool:
    if not _single_red(observation):
        return False
    assert observation.red_centroid_norm is not None
    assert observation.red_area_fraction is not None
    jaw_capture_ready = (
        observation.jaw_rad >= config.pregrasp_open_ready_rad - _EPSILON
        if config.coarse_capture_jaw_rad is None
        else _inside(observation.jaw_rad, config.coarse_capture_jaw_rad)
    )
    reference_error_rad = max(
        abs(measured - reference)
        for measured, reference in zip(
            observation.arm_q_rad,
            observation.arm_reference_q_rad,
            strict=True,
        )
    )
    return (
        _inside(observation.red_centroid_norm[0], config.coarse_x_norm)
        and _inside(observation.red_centroid_norm[1], config.coarse_y_norm)
        and observation.red_area_fraction >= config.coarse_area_fraction[0] - _EPSILON
        and observation.red_area_fraction < config.coarse_area_fraction[1]
        and jaw_capture_ready
        and observation.jaw_speed_rad_s <= config.max_jaw_speed_rad_s + _EPSILON
        and observation.arm_speed_rad_s <= config.max_arm_speed_rad_s + _EPSILON
        and reference_error_rad
        <= config.max_takeover_reference_error_rad + _EPSILON
    )


def _tracking_gate(observation: TipServoObservation, config: TipServoConfig) -> bool:
    if not _single_red(observation):
        return False
    assert observation.red_centroid_norm is not None
    assert observation.red_area_fraction is not None
    return (
        _inside(observation.red_centroid_norm[0], config.tracking_x_norm)
        and _inside(observation.red_centroid_norm[1], config.tracking_y_norm)
        and _inside(observation.red_area_fraction, config.tracking_area_fraction)
    )


def _close_ready_gate(observation: TipServoObservation, config: TipServoConfig) -> bool:
    if not _single_red(observation):
        return False
    assert observation.red_centroid_norm is not None
    assert observation.red_area_fraction is not None
    return (
        _inside(observation.red_centroid_norm[0], config.close_ready_x_norm)
        and _inside(observation.red_centroid_norm[1], config.close_ready_y_norm)
        and _inside(observation.red_area_fraction, config.close_ready_area_fraction)
    )


def _arm_settled(
    observation: TipServoObservation, state: TipServoState, config: TipServoConfig
) -> bool:
    target = state.last_arm_target_q_rad
    if target is None:
        return False
    return (
        observation.arm_speed_rad_s <= config.max_arm_speed_rad_s + _EPSILON
        and all(
            abs(measured - commanded) <= tolerance + _EPSILON
            for measured, commanded, tolerance in zip(
                observation.arm_q_rad,
                target,
                config.arm_target_settle_rad,
                strict=True,
            )
        )
    )


def _arm_slow(observation: TipServoObservation, config: TipServoConfig) -> bool:
    return observation.arm_speed_rad_s <= config.max_arm_speed_rad_s + _EPSILON


def _mode_command(mode: TipServoMode, proposed: Any) -> Any:
    return proposed if mode is TipServoMode.EXECUTE else None


def _decision(
    *,
    mode: TipServoMode,
    before: TipServoPhase,
    state: TipServoState,
    intent: TipServoIntent,
    event: str | None = None,
    proposed_arm: Vector6 | None = None,
    proposed_jaw: float | None = None,
    proposed_step_m: float = 0.0,
    vision_loss_elapsed_ms: float = 0.0,
    would_close: bool = False,
    close_authorized: bool = False,
    ik_result: DlsIkResult | None = None,
) -> TipServoDecision:
    return TipServoDecision(
        mode=mode,
        phase_before=before,
        phase_after=state.phase,
        intent=intent,
        event=event,
        fault_reason=state.fault_reason,
        proposed_arm_target_q_rad=proposed_arm,
        commanded_arm_target_q_rad=_mode_command(mode, proposed_arm),
        proposed_jaw_target_rad=proposed_jaw,
        commanded_jaw_target_rad=_mode_command(mode, proposed_jaw),
        proposed_descent_step_m=proposed_step_m,
        commanded_descent_m=state.commanded_descent_m,
        coarse_streak=state.coarse_streak,
        open_streak=state.open_streak,
        close_ready_streak=state.close_ready_streak,
        probe_success_streak=state.probe_success_streak,
        area_drop_streak=state.area_drop_streak,
        vision_loss_elapsed_ms=vision_loss_elapsed_ms,
        would_close=would_close,
        close_authorized=close_authorized,
        ik_result=ik_result,
    )


def _fault(
    state: TipServoState,
    observation: TipServoObservation,
    *,
    mode: TipServoMode,
    config: TipServoConfig,
    reason: str,
    ik_result: DlsIkResult | None = None,
) -> tuple[TipServoState, TipServoDecision]:
    before = state.phase
    hold_q = state.last_arm_target_q_rad or observation.arm_q_rad
    stored_hold_q = hold_q if state.takeover_q_rad is not None else None
    fault_state = replace(
        state,
        phase=TipServoPhase.FAULT_HOLD,
        last_arm_target_q_rad=stored_hold_q,
        last_evaluated_at_ns=observation.evaluated_at_ns,
        fault_reason=reason,
    )
    return fault_state, _decision(
        mode=mode,
        before=before,
        state=fault_state,
        intent=TipServoIntent.FAULT_HOLD,
        event="tip_servo_fault",
        proposed_arm=hold_q,
        proposed_jaw=(
            _open_only_planner_jaw_target(
                fault_state,
                config,
                fallback_reference_rad=observation.jaw_reference_rad,
            )
            if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
            else observation.jaw_rad
        ),
        ik_result=ik_result,
    )


def _active_hold(
    state: TipServoState,
    *,
    mode: TipServoMode,
    before: TipServoPhase,
    config: TipServoConfig,
    intent: TipServoIntent = TipServoIntent.HOLD,
    event: str | None = None,
    vision_loss_elapsed_ms: float = 0.0,
    would_close: bool = False,
    close_authorized: bool = False,
    proposed_jaw: float | None = None,
) -> TipServoDecision:
    assert state.last_arm_target_q_rad is not None
    if proposed_jaw is None:
        proposed_jaw = (
            _open_only_planner_jaw_target(state, config)
            if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
            else (
                config.pregrasp_open_target_rad
                if state.open_commanded
                else state.takeover_jaw_target_rad
            )
        )
    assert proposed_jaw is not None
    return _decision(
        mode=mode,
        before=before,
        state=state,
        intent=intent,
        event=event,
        proposed_arm=state.last_arm_target_q_rad,
        proposed_jaw=proposed_jaw,
        vision_loss_elapsed_ms=vision_loss_elapsed_ms,
        would_close=would_close,
        close_authorized=close_authorized,
    )


def _is_new_fresh_frame(state: TipServoState, observation: TipServoObservation) -> bool:
    return (
        observation.frame_status is TipServoFrameStatus.FRESH_UNIQUE
        and (
            state.last_fresh_frame_id is None
            or observation.frame_id > state.last_fresh_frame_id
        )
    )


def _active_vision_gap(
    state: TipServoState,
    observation: TipServoObservation,
    *,
    mode: TipServoMode,
    config: TipServoConfig,
) -> tuple[TipServoState, TipServoDecision]:
    before = state.phase
    since = (
        observation.evaluated_at_ns
        if state.vision_loss_since_ns is None
        else state.vision_loss_since_ns
    )
    elapsed_ms = max(0.0, (observation.evaluated_at_ns - since) / 1_000_000.0)
    if elapsed_ms >= config.vision_loss_timeout_ms - _EPSILON:
        return _fault(
            state,
            observation,
            mode=mode,
            config=config,
            reason="fresh_visual_timeout",
        )
    held = replace(
        state,
        last_evaluated_at_ns=observation.evaluated_at_ns,
        vision_loss_since_ns=since,
    )
    return held, _active_hold(
        held,
        mode=mode,
        before=before,
        config=config,
        event="visual_frame_hold",
        vision_loss_elapsed_ms=elapsed_ms,
    )


def _update_active_visual_state(
    state: TipServoState,
    observation: TipServoObservation,
    config: TipServoConfig,
) -> TipServoState:
    assert observation.red_area_fraction is not None
    previous_peak = state.peak_red_area_fraction
    peak = observation.red_area_fraction if previous_peak is None else max(
        previous_peak, observation.red_area_fraction
    )
    dropped = (
        previous_peak is not None
        and observation.red_area_fraction
        < previous_peak - config.area_drop_fault_fraction - _EPSILON
    )
    return replace(
        state,
        peak_red_area_fraction=peak,
        area_drop_streak=state.area_drop_streak + 1 if dropped else 0,
        last_fresh_frame_id=observation.frame_id,
        last_evaluated_at_ns=observation.evaluated_at_ns,
        vision_loss_since_ns=None,
    )


def _advance_tip_servo_enabled(
    state: TipServoState,
    observation: TipServoObservation,
    *,
    mode: TipServoMode,
    fk: ForwardKinematics,
    config: TipServoConfig = TipServoConfig(),
    ik_config: DlsIkConfig = DlsIkConfig(),
    takeover_authorized: bool = True,
) -> tuple[TipServoState, TipServoDecision]:
    """Advance one pure visual-servo transition.

    This internal transition is also used to form one-step advisory proposals.
    ``EXECUTE`` decisions are still only data: the caller retains all authority
    and must independently enforce controller tracking and safety.
    """

    try:
        mode = TipServoMode(mode)
    except (TypeError, ValueError) as exc:
        raise TipServoError("mode is unsupported") from exc
    before = state.phase
    if config.probe_profile is TipServoProbeProfile.OPEN_ONLY:
        if state.capture_jaw_target_rad is None:
            if state.phase is not TipServoPhase.SEEKING:
                raise TipServoError(
                    "active OPEN_ONLY state has no immutable capture jaw target"
                )
            capture = _validated_open_only_capture_jaw_target(
                observation.jaw_reference_rad,
                config=config,
            )
            state = replace(
                state,
                capture_jaw_target_rad=capture,
                seeking_started_at_ns=observation.evaluated_at_ns,
            )
        else:
            _validated_open_only_capture_jaw_target(
                state.capture_jaw_target_rad,
                config=config,
            )
            if state.seeking_started_at_ns is None:  # dataclass invariant
                raise TipServoError("OPEN_ONLY state has no SEEKING start time")
    if (
        config.probe_profile is TipServoProbeProfile.OPEN_ONLY
        and state.phase is TipServoPhase.PROBE_COMPLETE
    ):
        # OPEN_ONLY completion is terminal.  Later visual loss, geometry
        # changes, or policy close intent cannot reacquire motion authority.
        # The bounded-descent profile retains its pre-existing post-probe
        # visual checks.
        return state, _active_hold(
            state,
            mode=mode,
            before=before,
            config=config,
        )
    if state.last_evaluated_at_ns is not None and (
        observation.evaluated_at_ns < state.last_evaluated_at_ns
    ):
        if state.phase is TipServoPhase.SEEKING:
            reset = replace(state, coarse_streak=0)
            return reset, _decision(
                mode=mode,
                before=before,
                state=reset,
                intent=TipServoIntent.OBSERVE,
                event="evaluation_time_hold",
                proposed_jaw=(
                    _open_only_planner_jaw_target(reset, config)
                    if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
                    else None
                ),
            )
        return _fault(
            state,
            observation,
            mode=mode,
            config=config,
            reason="evaluation_time_out_of_order",
        )

    if state.phase is TipServoPhase.FAULT_HOLD:
        hold_q = state.last_arm_target_q_rad or observation.arm_q_rad
        held = replace(state, last_evaluated_at_ns=observation.evaluated_at_ns)
        return held, _decision(
            mode=mode,
            before=before,
            state=held,
            intent=TipServoIntent.FAULT_HOLD,
            proposed_arm=hold_q,
            proposed_jaw=(
                _open_only_planner_jaw_target(
                    held,
                    config,
                    fallback_reference_rad=observation.jaw_reference_rad,
                )
                if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
                else observation.jaw_rad
            ),
        )

    if (
        config.probe_profile is TipServoProbeProfile.OPEN_ONLY
        and state.phase not in (TipServoPhase.SEEKING, TipServoPhase.OPENING)
    ):
        return _fault(
            state,
            observation,
            mode=mode,
            config=config,
            reason="open_only_invalid_phase",
        )

    new_fresh = _is_new_fresh_frame(state, observation)
    if state.phase is TipServoPhase.SEEKING:
        if config.probe_profile is TipServoProbeProfile.OPEN_ONLY:
            assert state.seeking_started_at_ns is not None
            seeking_elapsed_ms = max(
                0.0,
                (
                    observation.evaluated_at_ns - state.seeking_started_at_ns
                )
                / 1_000_000.0,
            )
            if seeking_elapsed_ms >= config.seeking_timeout_ms - _EPSILON:
                return _fault(
                    state,
                    observation,
                    mode=mode,
                    config=config,
                    reason="seeking_timeout",
                )
        seeking_jaw = (
            _open_only_planner_jaw_target(state, config)
            if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
            else None
        )
        if not new_fresh:
            reset = replace(
                state,
                coarse_streak=0,
                last_evaluated_at_ns=observation.evaluated_at_ns,
            )
            return reset, _decision(
                mode=mode,
                before=before,
                state=reset,
                intent=TipServoIntent.OBSERVE,
                event="seeking_visual_hold",
                proposed_jaw=seeking_jaw,
            )
        valid = _coarse_gate(observation, config)
        streak = (
            min(state.coarse_streak + 1, config.takeover_confirm_frames)
            if valid
            else 0
        )
        seeking = replace(
            state,
            coarse_streak=streak,
            last_fresh_frame_id=observation.frame_id,
            last_evaluated_at_ns=observation.evaluated_at_ns,
        )
        if streak < config.takeover_confirm_frames:
            return seeking, _decision(
                mode=mode,
                before=before,
                state=seeking,
                intent=TipServoIntent.OBSERVE,
                event="coarse_candidate" if valid else None,
                proposed_jaw=seeking_jaw,
            )
        if not takeover_authorized:
            return seeking, _decision(
                mode=mode,
                before=before,
                state=seeking,
                intent=TipServoIntent.OBSERVE,
                event="coarse_ready_waiting_authorization",
                proposed_jaw=seeking_jaw,
            )
        try:
            position, rotation = _evaluate_fk(fk, observation.arm_q_rad)
        except _FkEvaluationError as exc:
            return _fault(
                seeking,
                observation,
                mode=mode,
                config=config,
                reason=str(exc),
            )
        active = replace(
            seeking,
            phase=TipServoPhase.OPENING,
            takeover_q_rad=observation.arm_q_rad,
            takeover_position_m=tuple(float(value) for value in position),
            takeover_rotation=tuple(
                tuple(float(value) for value in row) for row in rotation
            ),
            last_arm_target_q_rad=observation.arm_reference_q_rad,
            takeover_jaw_target_rad=(
                state.capture_jaw_target_rad
                if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
                else observation.jaw_rad
            ),
            takeover_started_at_ns=observation.evaluated_at_ns,
            open_commanded=False,
            peak_red_area_fraction=observation.red_area_fraction,
            open_streak=0,
        )
        return active, _active_hold(
            active,
            mode=mode,
            before=before,
            config=config,
            intent=TipServoIntent.HOLD,
            event="coarse_takeover",
        )

    if state.phase is TipServoPhase.OPENING:
        assert state.takeover_started_at_ns is not None
        opening_elapsed_ms = max(
            0.0,
            (
                observation.evaluated_at_ns - state.takeover_started_at_ns
            )
            / 1_000_000.0,
        )
        if opening_elapsed_ms >= config.opening_timeout_ms - _EPSILON:
            return _fault(
                state,
                observation,
                mode=mode,
                config=config,
                reason="opening_timeout",
            )

    if not new_fresh:
        return _active_vision_gap(
            state,
            observation,
            mode=mode,
            config=config,
        )
    if not _tracking_gate(observation, config):
        return _fault(
            state,
            observation,
            mode=mode,
            config=config,
            reason="red_outside_tracking_corridor",
        )
    visual_state = _update_active_visual_state(state, observation, config)
    if visual_state.area_drop_streak >= config.area_drop_fault_frames:
        return _fault(
            visual_state,
            observation,
            mode=mode,
            config=config,
            reason="red_area_significant_drop",
        )
    if visual_state.area_drop_streak > 0:
        return visual_state, _active_hold(
            visual_state,
            mode=mode,
            before=before,
            config=config,
            event="red_area_drawdown_hold",
        )

    if visual_state.phase is TipServoPhase.OPENING:
        if not visual_state.open_commanded:
            if not _arm_settled(observation, visual_state, config):
                return visual_state, _active_hold(
                    visual_state,
                    mode=mode,
                    before=before,
                    config=config,
                    event="takeover_settle_hold",
                )
            visual_state = replace(
                visual_state,
                open_commanded=True,
                open_streak=0,
            )
        open_ready = (
            observation.jaw_rad >= config.pregrasp_open_ready_rad - _EPSILON
            and observation.jaw_speed_rad_s <= config.max_jaw_speed_rad_s + _EPSILON
            and (
                _arm_slow(observation, config)
                if config.probe_profile is TipServoProbeProfile.OPEN_ONLY
                else _arm_settled(observation, visual_state, config)
            )
        )
        opening = replace(
            visual_state,
            open_streak=visual_state.open_streak + 1 if open_ready else 0,
        )
        if opening.open_streak >= config.open_confirm_frames:
            if config.probe_profile is TipServoProbeProfile.OPEN_ONLY:
                complete = replace(
                    opening,
                    phase=TipServoPhase.PROBE_COMPLETE,
                    close_ready_streak=0,
                    probe_success_streak=0,
                )
                return complete, _active_hold(
                    complete,
                    mode=mode,
                    before=before,
                    config=config,
                    intent=TipServoIntent.OPEN,
                    event="open_only_probe_complete",
                )
            descending = replace(
                opening,
                phase=TipServoPhase.DESCENDING,
                close_ready_streak=0,
                probe_success_streak=0,
            )
            return descending, _active_hold(
                descending,
                mode=mode,
                before=before,
                config=config,
                intent=TipServoIntent.OPEN,
                event="open_stable",
            )
        return opening, _active_hold(
            opening,
            mode=mode,
            before=before,
            config=config,
            intent=TipServoIntent.OPEN,
            event="opening_hold",
        )

    if visual_state.phase is TipServoPhase.PROBE_COMPLETE:
        return visual_state, _active_hold(
            visual_state,
            mode=mode,
            before=before,
            config=config,
        )

    if visual_state.phase is TipServoPhase.CLOSE_READY:
        return visual_state, _active_hold(
            visual_state,
            mode=mode,
            before=before,
            config=config,
            intent=TipServoIntent.VISUAL_CLOSE_READY,
            would_close=True,
            close_authorized=(
                config.allow_close and mode is TipServoMode.EXECUTE
            ),
        )

    assert visual_state.phase is TipServoPhase.DESCENDING
    settled = _arm_settled(observation, visual_state, config)
    close_candidate = settled and _close_ready_gate(observation, config)
    close_streak = visual_state.close_ready_streak + 1 if close_candidate else 0
    probe_candidate = (
        not config.allow_close
        and settled
        and not close_candidate
        and observation.red_area_fraction is not None
        and observation.red_area_fraction
        >= config.probe_success_area_fraction - _EPSILON
    )
    probe_streak = visual_state.probe_success_streak + 1 if probe_candidate else 0
    descending = replace(
        visual_state,
        close_ready_streak=close_streak,
        probe_success_streak=probe_streak,
    )
    if close_candidate:
        if close_streak >= config.close_ready_confirm_frames:
            if config.allow_close:
                ready = replace(descending, phase=TipServoPhase.CLOSE_READY)
                intent = TipServoIntent.VISUAL_CLOSE_READY
                event = "visual_close_ready"
            else:
                ready = replace(descending, phase=TipServoPhase.PROBE_COMPLETE)
                intent = TipServoIntent.HOLD
                event = "visual_would_close"
            return ready, _active_hold(
                ready,
                mode=mode,
                before=before,
                config=config,
                intent=intent,
                event=event,
                would_close=True,
                close_authorized=(
                    config.allow_close and mode is TipServoMode.EXECUTE
                ),
            )
        return descending, _active_hold(
            descending,
            mode=mode,
            before=before,
            config=config,
            event="close_ready_confirming",
        )
    if probe_candidate:
        if probe_streak >= config.probe_success_confirm_frames:
            complete = replace(descending, phase=TipServoPhase.PROBE_COMPLETE)
            return complete, _active_hold(
                complete,
                mode=mode,
                before=before,
                config=config,
                event="probe_progress_complete",
            )
        return descending, _active_hold(
            descending,
            mode=mode,
            before=before,
            config=config,
            event="probe_progress_confirming",
        )
    if not settled:
        return descending, _active_hold(
            descending,
            mode=mode,
            before=before,
            config=config,
            event="arm_target_settle_hold",
        )
    if descending.commanded_descent_m >= config.max_total_descent_m - _EPSILON:
        return _fault(
            descending,
            observation,
            mode=mode,
            config=config,
            reason=(
                "max_total_descent_without_close_ready"
                if config.allow_close
                else "probe_descent_without_minimum_area_progress"
            ),
        )

    step_m = min(
        config.descent_step_m,
        config.max_total_descent_m - descending.commanded_descent_m,
    )
    if step_m <= _EPSILON:
        return _fault(
            descending,
            observation,
            mode=mode,
            config=config,
            reason="descent_step_exhausted",
        )
    assert descending.takeover_position_m is not None
    assert descending.takeover_rotation is not None
    next_total = descending.commanded_descent_m + step_m
    desired_position = (
        descending.takeover_position_m[0],
        descending.takeover_position_m[1],
        descending.takeover_position_m[2] - next_total,
    )
    ik_result = solve_full_pose_dls(
        observation.arm_q_rad,
        desired_position,
        descending.takeover_rotation,
        fk=fk,
        config=ik_config,
    )
    if not ik_result.ok or ik_result.q_rad is None:
        return _fault(
            descending,
            observation,
            mode=mode,
            config=config,
            reason=f"ik_{ik_result.reason or 'unknown_failure'}",
            ik_result=ik_result,
        )
    advanced = replace(
        descending,
        last_arm_target_q_rad=ik_result.q_rad,
        commanded_descent_m=next_total,
    )
    if mode is TipServoMode.EXECUTE:
        event = "descent_step"
    else:
        event = "advisory_descent_proposed"
    return advanced, _decision(
        mode=mode,
        before=before,
        state=advanced,
        intent=TipServoIntent.DESCEND,
        event=event,
        proposed_arm=ik_result.q_rad,
        proposed_jaw=config.pregrasp_open_target_rad,
        proposed_step_m=step_m,
        ik_result=ik_result,
    )


def advance_tip_servo(
    state: TipServoState,
    observation: TipServoObservation,
    *,
    fk: ForwardKinematics,
    config: TipServoConfig = TipServoConfig(),
    ik_config: DlsIkConfig = DlsIkConfig(),
    mode: TipServoMode | None = None,
    takeover_authorized: bool = True,
) -> tuple[TipServoState, TipServoDecision]:
    """Advance according to the fingerprintable mode in ``config``.

    ``DISABLED`` returns the input state with no proposal.  ``ADVISORY``
    advances only the caller-owned tip-servo counterfactual state and exposes
    proposals; it emits no command and never authorizes close.
    ``EXECUTE`` is the only mode that advances state or fills command fields.
    If the optional explicit ``mode`` argument is supplied, it must exactly
    match ``config.mode``; this prevents a caller from silently widening the
    fingerprinted authority.

    ``takeover_authorized`` gates only the SEEKING -> OPENING edge.  Fresh,
    unique, gate-valid evidence still accumulates (and then holds) its coarse
    streak while authorization is false; a later authorized fresh frame must
    independently remain inside every coarse gate before takeover occurs.
    """

    if type(takeover_authorized) is not bool:
        raise TipServoError("takeover_authorized must be boolean")
    resolved_mode = config.mode
    if mode is not None:
        try:
            explicit_mode = TipServoMode(mode)
        except (TypeError, ValueError) as exc:
            raise TipServoError("mode is unsupported") from exc
        if explicit_mode is not resolved_mode:
            raise TipServoError("explicit mode must exactly match config.mode")
    if resolved_mode is TipServoMode.DISABLED:
        return state, _decision(
            mode=resolved_mode,
            before=state.phase,
            state=state,
            intent=TipServoIntent.OBSERVE,
        )
    proposed_state, proposed = _advance_tip_servo_enabled(
        state,
        observation,
        mode=resolved_mode,
        fk=fk,
        config=config,
        ik_config=ik_config,
        takeover_authorized=takeover_authorized,
    )
    if resolved_mode is TipServoMode.EXECUTE:
        return proposed_state, proposed
    event = proposed.event
    if event is not None and not event.startswith("advisory_"):
        event = f"advisory_{event}"
    return proposed_state, replace(
        proposed,
        event=event,
        commanded_arm_target_q_rad=None,
        commanded_jaw_target_rad=None,
        commanded_descent_m=state.commanded_descent_m,
        close_authorized=False,
    )


__all__ = [
    "DlsIkConfig",
    "DlsIkResult",
    "ForwardKinematics",
    "HARD_MAX_DESCENT_STEP_M",
    "HARD_MAX_JOINT_STEP_RAD",
    "HARD_MAX_OPEN_ONLY_JAW_TARGET_RAD",
    "HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS",
    "HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD",
    "HARD_MAX_TOTAL_DESCENT_M",
    "HARD_OPEN_ONLY_CAPTURE_JAW_RAD",
    "TipServoConfig",
    "TipServoDecision",
    "TipServoError",
    "TipServoFrameStatus",
    "TipServoIntent",
    "TipServoMode",
    "TipServoObservation",
    "TipServoPhase",
    "TipServoProbeProfile",
    "TipServoState",
    "advance_tip_servo",
    "resolve_open_only_jaw_target",
    "solve_full_pose_dls",
]
