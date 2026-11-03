"""Offline-only counterfactual analysis for Firefly shadow telemetry.

No robot, camera, Zenoh, CAN, or protobuf modules are imported here. Diagnostic
bounds never replace the live controller description used for execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


MAX_DIAGNOSTIC_EXPANSION_RAD = 0.020
TARGET_EVENTS = frozenset(("shadow_cycle", "safety_rejection"))


@dataclass(frozen=True)
class EnvelopeContract:
    joint_names: tuple[str, ...]
    pos_min: tuple[float, ...]
    pos_max: tuple[float, ...]
    step_limits: tuple[float, ...]
    source: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EnvelopeContract":
        try:
            names = tuple(str(value) for value in raw["joint_names"])
            lower = tuple(float(value) for value in raw["pos_min"])
            upper = tuple(float(value) for value in raw["pos_max"])
            steps = tuple(float(value) for value in raw["effective_step_limits"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid controller contract: {}".format(exc)) from exc
        size = len(names)
        if not size or len(set(names)) != size:
            raise ValueError("joint_names must be non-empty and unique")
        if not all(len(values) == size for values in (lower, upper, steps)):
            raise ValueError("contract arrays must match joint_names length")
        if not all(math.isfinite(value) for value in lower + upper + steps):
            raise ValueError("contract contains NaN or infinite values")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("contract contains an invalid position range")
        if any(value <= 0 for value in steps):
            raise ValueError("contract contains a non-positive step limit")
        return cls(names, lower, upper, steps, str(raw.get("source", "unspecified")))


def _vector(raw: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("{} must be a numeric sequence".format(label))
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} contains a non-numeric value".format(label)) from exc
    if len(values) != size:
        raise ValueError("{} has {} values, expected {}".format(label, len(values), size))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("{} contains NaN or infinite values".format(label))
    return values


def _diagnostic_bounds(
    contract: EnvelopeContract, joint_2_min: float, joint_3_max: float
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, float]]:
    try:
        j2 = contract.joint_names.index("joint_2.pos")
        j3 = contract.joint_names.index("joint_3.pos")
    except ValueError as exc:
        raise ValueError("contract must contain joint_2.pos and joint_3.pos") from exc
    requested = (float(joint_2_min), float(joint_3_max))
    if not all(math.isfinite(value) for value in requested):
        raise ValueError("hypothetical bounds must be finite")
    expansion = (contract.pos_min[j2] - requested[0], requested[1] - contract.pos_max[j3])
    if any(value < 0 for value in expansion):
        raise ValueError("diagnostic bounds may only expand outward")
    if any(value > MAX_DIAGNOSTIC_EXPANSION_RAD + 1e-12 for value in expansion):
        raise ValueError("diagnostic expansion is limited to 0.020 rad")
    lower, upper = list(contract.pos_min), list(contract.pos_max)
    lower[j2], upper[j3] = requested
    return tuple(lower), tuple(upper), {
        "joint_2_lower_expansion_rad": expansion[0],
        "joint_3_upper_expansion_rad": expansion[1],
    }


def analyze_shadow_envelope(
    events: Iterable[Mapping[str, Any]],
    contract: EnvelopeContract,
    *,
    joint_2_min: float,
    joint_3_max: float,
) -> dict[str, Any]:
    """Replay shadow targets while preserving fail-closed reference semantics."""

    diagnostic_min, diagnostic_max, expansion = _diagnostic_bounds(
        contract, joint_2_min, joint_3_max
    )
    materialized = list(events)
    if any(event.get("event") == "command_cycle" for event in materialized):
        raise ValueError("execute command telemetry is not accepted")
    rows = []
    chunks = 0
    size = len(contract.joint_names)
    for line_number, event in enumerate(materialized, 1):
        chunks += event.get("event") == "inference_chunk"
        if event.get("event") not in TARGET_EVENTS or event.get("target") is None:
            continue
        target = _vector(event.get("target"), size, "line {} target".format(line_number))
        measured = _vector(event.get("measured"), size, "line {} measured".format(line_number))
        if event.get("effective_step_limits") is not None:
            recorded = _vector(
                event["effective_step_limits"], size, "line {} step limits".format(line_number)
            )
            if any(
                not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                for a, b in zip(recorded, contract.step_limits, strict=True)
            ):
                raise ValueError("recorded step limits do not match the contract")
        rows.append((target, measured))
    if not rows:
        raise ValueError("telemetry contains no shadow targets")

    fields = (
        "controller_range_rejections",
        "hypothetical_range_rejections",
        "controller_adapter_step_rejections",
        "hypothetical_adapter_step_rejections",
        "open_loop_step_rejections",
    )
    per_joint = {
        name: {"target_min": math.inf, "target_max": -math.inf, **dict.fromkeys(fields, 0)}
        for name in contract.joint_names
    }
    real_last = None
    diagnostic_last = None
    open_last = None
    totals = dict.fromkeys(
        (
            "controller_position_rejections",
            "hypothetical_position_rejections",
            "controller_adapter_rejections",
            "hypothetical_adapter_rejections",
            "open_loop_step_rejections",
        ),
        0,
    )
    first_rejection = None
    for target_index, (target, measured) in enumerate(rows):
        real_ref = measured if real_last is None else real_last
        diagnostic_ref = measured if diagnostic_last is None else diagnostic_last
        open_ref = measured if open_last is None else open_last
        real_pos_bad, diag_pos_bad, real_step_bad, diag_step_bad, open_step_bad = (
            [], [], [], [], []
        )
        for index, name in enumerate(contract.joint_names):
            stats = per_joint[name]
            stats["target_min"] = min(stats["target_min"], target[index])
            stats["target_max"] = max(stats["target_max"], target[index])
            checks = (
                (
                    target[index] < contract.pos_min[index]
                    or target[index] > contract.pos_max[index],
                    real_pos_bad,
                    "controller_range_rejections",
                ),
                (
                    target[index] < diagnostic_min[index]
                    or target[index] > diagnostic_max[index],
                    diag_pos_bad,
                    "hypothetical_range_rejections",
                ),
                (
                    abs(target[index] - real_ref[index]) > contract.step_limits[index] + 1e-9,
                    real_step_bad,
                    "controller_adapter_step_rejections",
                ),
                (
                    abs(target[index] - diagnostic_ref[index])
                    > contract.step_limits[index] + 1e-9,
                    diag_step_bad,
                    "hypothetical_adapter_step_rejections",
                ),
                (
                    abs(target[index] - open_ref[index]) > contract.step_limits[index] + 1e-9,
                    open_step_bad,
                    "open_loop_step_rejections",
                ),
            )
            for failed, failures, field in checks:
                if failed:
                    failures.append(name)
                    stats[field] += 1
        real_rejected = bool(real_pos_bad or real_step_bad)
        diagnostic_rejected = bool(diag_pos_bad or diag_step_bad)
        totals["controller_position_rejections"] += bool(real_pos_bad)
        totals["hypothetical_position_rejections"] += bool(diag_pos_bad)
        totals["controller_adapter_rejections"] += real_rejected
        totals["hypothetical_adapter_rejections"] += diagnostic_rejected
        totals["open_loop_step_rejections"] += bool(open_step_bad)
        if not real_rejected:
            real_last = target
        if not diagnostic_rejected:
            diagnostic_last = target
        if first_rejection is None and diagnostic_rejected:
            first_rejection = {
                "target_index": target_index,
                "position_joints": diag_pos_bad,
                "step_joints": diag_step_bad,
            }
        open_last = target

    return {
        "hypothetical_only": True,
        "eligible_for_execution": False,
        "eligible_for_acceptance": False,
        "warning": "Diagnostic bounds never authorize command publication.",
        "contract_source": contract.source,
        "joint_names": list(contract.joint_names),
        "controller_pos_min": list(contract.pos_min),
        "controller_pos_max": list(contract.pos_max),
        "hypothetical_pos_min": list(diagnostic_min),
        "hypothetical_pos_max": list(diagnostic_max),
        "expansion": expansion,
        "inference_chunks": chunks,
        "targets": len(rows),
        **totals,
        "hypothetical_adapter_acceptances": len(rows)
        - totals["hypothetical_adapter_rejections"],
        "first_hypothetical_rejection": first_rejection,
        "per_joint": per_joint,
    }
