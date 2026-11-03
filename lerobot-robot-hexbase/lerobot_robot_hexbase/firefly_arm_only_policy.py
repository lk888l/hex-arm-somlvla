"""Pure authority adapter for a Firefly arm-only SEEKING policy.

This module deliberately knows nothing about cameras, inference workers,
tip-servo phases, controllers, or command publication.  It exposes only the
model-facing six arm axes and expands a validated six-axis action chunk with a
capture jaw target that was already bound by an external safety layer.

The policy has no jaw input and no jaw output.  A value added as the seventh
candidate axis is therefore always named as synthesized authority, never as a
policy prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any


ARM6_OBSERVATION_FEATURES = tuple(
    f"joint_{index}.pos" for index in range(1, 7)
)
ARM6_SEEKING_ROLE = "arm6_seeking"

Vector6 = tuple[float, float, float, float, float, float]
Vector7 = tuple[float, float, float, float, float, float, float]
Arm6Chunk = tuple[Vector6, ...]
Candidate7Chunk = tuple[Vector7, ...]


class ArmOnlyPolicyError(ValueError):
    """An input violates the narrow arm-only policy authority contract."""


class ArmOnlyPolicyRole(StrEnum):
    """Explicit policy roles supported by this adapter."""

    ARM6_SEEKING = ARM6_SEEKING_ROLE


def _require_role(role: Any) -> ArmOnlyPolicyRole:
    try:
        resolved = ArmOnlyPolicyRole(role)
    except (TypeError, ValueError) as exc:
        raise ArmOnlyPolicyError(
            f"policy role must be exactly {ARM6_SEEKING_ROLE!r}"
        ) from exc
    if resolved is not ArmOnlyPolicyRole.ARM6_SEEKING:
        raise ArmOnlyPolicyError(
            f"policy role must be exactly {ARM6_SEEKING_ROLE!r}"
        )
    return resolved


def _as_sequence(raw: Any, *, label: str) -> Sequence[Any]:
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        raise ArmOnlyPolicyError(f"{label} must be a sequence")
    if not isinstance(raw, Sequence):
        tolist = getattr(raw, "tolist", None)
        if not callable(tolist):
            raise ArmOnlyPolicyError(f"{label} must be a sequence")
        raw = tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ArmOnlyPolicyError(f"{label} must be a sequence")
    return raw


def _finite_number(raw: Any, *, label: str) -> float:
    if isinstance(raw, bool):
        raise ArmOnlyPolicyError(f"{label} must be a finite number")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArmOnlyPolicyError(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise ArmOnlyPolicyError(f"{label} must be a finite number")
    return value


def _vector6(raw: Any, *, label: str) -> Vector6:
    values = _as_sequence(raw, label=label)
    if len(values) != 6:
        raise ArmOnlyPolicyError(f"{label} must contain exactly 6 arm values")
    return tuple(
        _finite_number(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )  # type: ignore[return-value]


def extract_arm6_observation(
    observation: Mapping[str, Any],
    *,
    role: ArmOnlyPolicyRole | str,
) -> Vector6:
    """Return measured J1--J6 without reading or exposing a jaw value."""

    _require_role(role)
    if not isinstance(observation, Mapping):
        raise ArmOnlyPolicyError("observation must be a mapping")
    values: list[float] = []
    for feature in ARM6_OBSERVATION_FEATURES:
        if feature not in observation:
            raise ArmOnlyPolicyError(
                f"arm6 observation is missing required feature {feature!r}"
            )
        values.append(
            _finite_number(observation[feature], label=f"observation[{feature!r}]")
        )
    return tuple(values)  # type: ignore[return-value]


def validate_arm6_action_chunk(
    raw_actions: Any,
    *,
    n_steps: int,
    role: ArmOnlyPolicyRole | str,
) -> Arm6Chunk:
    """Copy and strictly validate one finite ``(n_steps, 6)`` policy chunk."""

    _require_role(role)
    if type(n_steps) is not int or n_steps <= 0:
        raise ArmOnlyPolicyError("n_steps must be a positive integer")
    rows = _as_sequence(raw_actions, label="raw arm-only action chunk")
    if len(rows) != n_steps:
        raise ArmOnlyPolicyError(
            "raw arm-only action chunk must have shape "
            f"({n_steps}, 6), got {len(rows)} rows"
        )
    return tuple(
        _vector6(row, label=f"raw arm-only action chunk[{index}]")
        for index, row in enumerate(rows)
    )


def _capture_interval(raw: Any) -> tuple[float, float]:
    values = _as_sequence(raw, label="capture_jaw_range_rad")
    if len(values) != 2:
        raise ArmOnlyPolicyError("capture_jaw_range_rad must contain exactly 2 values")
    lower = _finite_number(values[0], label="capture_jaw_range_rad[0]")
    upper = _finite_number(values[1], label="capture_jaw_range_rad[1]")
    if lower >= upper:
        raise ArmOnlyPolicyError(
            "capture_jaw_range_rad must be a strictly increasing interval"
        )
    return lower, upper


@dataclass(frozen=True, slots=True)
class ArmOnlyExpandedChunk:
    """Immutable raw arm authority and its externally synthesized 7-D candidate."""

    policy_role: ArmOnlyPolicyRole
    raw_arm_action6: Arm6Chunk
    synthesized_capture_jaw: float
    candidate7: Candidate7Chunk

    def telemetry(self) -> dict[str, Any]:
        """Return JSON-friendly telemetry with no fictitious policy-jaw field."""

        return {
            "policy_role": self.policy_role.value,
            "raw_arm_action6": [list(row) for row in self.raw_arm_action6],
            "synthesized_capture_jaw": self.synthesized_capture_jaw,
            "candidate7": [list(row) for row in self.candidate7],
        }


def expand_arm6_action_chunk(
    raw_actions: Any,
    *,
    n_steps: int,
    role: ArmOnlyPolicyRole | str,
    capture_jaw_target_rad: Any,
    capture_jaw_range_rad: Any,
) -> ArmOnlyExpandedChunk:
    """Append an already-bound capture jaw to a validated arm-only chunk.

    The capture interval is caller-supplied so that this pure module does not
    become deployment authority.  Both interval edges are inclusive.  No
    clamping is performed: an invalid target is rejected rather than changed.
    """

    resolved_role = _require_role(role)
    chunk = validate_arm6_action_chunk(
        raw_actions,
        n_steps=n_steps,
        role=resolved_role,
    )
    lower, upper = _capture_interval(capture_jaw_range_rad)
    capture = _finite_number(
        capture_jaw_target_rad,
        label="capture_jaw_target_rad",
    )
    if not lower <= capture <= upper:
        raise ArmOnlyPolicyError(
            "capture_jaw_target_rad is outside the externally authorized "
            f"interval [{lower}, {upper}]"
        )
    candidate: Candidate7Chunk = tuple(
        (*arm, capture) for arm in chunk
    )
    return ArmOnlyExpandedChunk(
        policy_role=resolved_role,
        raw_arm_action6=chunk,
        synthesized_capture_jaw=capture,
        candidate7=candidate,
    )


@dataclass(frozen=True, slots=True)
class ArmOnlyTakeoverBoundary:
    """Caller-owned effects required at the externally detected takeover edge."""

    policy_role: ArmOnlyPolicyRole
    arm_policy_authorized: bool
    discard_policy_chunk: bool
    smoother_reset_required: bool
    replan_reason: str | None


def resolve_arm6_takeover_boundary(
    *,
    role: ArmOnlyPolicyRole | str,
    takeover_boundary: bool,
) -> ArmOnlyTakeoverBoundary:
    """Describe chunk ownership at a caller-detected takeover boundary.

    This function does not detect takeover and intentionally contains no tip
    state machine.  It only makes the required discard/reset edge explicit.
    """

    resolved_role = _require_role(role)
    if type(takeover_boundary) is not bool:
        raise ArmOnlyPolicyError("takeover_boundary must be boolean")
    return ArmOnlyTakeoverBoundary(
        policy_role=resolved_role,
        arm_policy_authorized=not takeover_boundary,
        discard_policy_chunk=takeover_boundary,
        smoother_reset_required=takeover_boundary,
        replan_reason="tip_servo_takeover" if takeover_boundary else None,
    )


__all__ = [
    "ARM6_OBSERVATION_FEATURES",
    "ARM6_SEEKING_ROLE",
    "ArmOnlyExpandedChunk",
    "ArmOnlyPolicyError",
    "ArmOnlyPolicyRole",
    "ArmOnlyTakeoverBoundary",
    "expand_arm6_action_chunk",
    "extract_arm6_observation",
    "resolve_arm6_takeover_boundary",
    "validate_arm6_action_chunk",
]
