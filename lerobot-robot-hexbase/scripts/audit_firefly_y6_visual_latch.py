#!/usr/bin/env python3
"""Replay fullcycle-v4 D405 video through the deployed visual detector.

This is a read-only, pseudo-label audit.  The demonstrations do not contain
explicit phase annotations, so four alternating crossings of ``action[6]``
reconstruct pregrasp-open, pickup-close, release-open, and return-close.  The
script never rotates or rewrites the recorded MP4 files.

For the reviewed first tip probe it also measures a bounded, no-close
recorded-trace envelope over all demonstrations.  Schema 6 binds a historical
ACT trace containing real pre-fix jaw/TCP behavior as an adversarial
axis-authority replay.  Those checks do not claim hardware completion or
counterfactual closed-loop convergence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_visual_latch import (  # noqa: E402
    BlobEvidence,
    compose_visual_action,
    D405VisualDetector,
    FrameStatus,
    VisualEvidence,
    VisualGraspReleaseLatch,
    VisualLatchPhase,
    VisualPhaseLatchConfig,
)
from lerobot_robot_hexbase.arm_gravity import (  # noqa: E402
    SerialRevoluteGravityModel,
)
from lerobot_robot_hexbase.firefly_deployment import (  # noqa: E402
    visual_phase_latch_implementation_manifest,
    visual_phase_latch_provenance,
)
from lerobot_robot_hexbase.firefly_training_attestation import (  # noqa: E402
    TrainingAttestationError,
    capture_training_inputs,
)
from lerobot_robot_hexbase.firefly_tip_servo import (  # noqa: E402
    TipServoConfig,
    TipServoMode,
    TipServoProbeProfile,
    resolve_open_only_jaw_target,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
DEFAULT_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4"
DEFAULT_EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
DEFAULT_CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
DEFAULT_TIP_PROBE_NEGATIVE_CONTROL = (
    REPOSITORY_ROOT
    / "outputs/firefly_deployment_hold_fullcycle_v4_100k_visual_latch_v1/"
    "act_90s.jsonl"
)
EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_SHA256 = (
    "cf46da3d3ee50cea27ff3a2743fb9ce90a7d10e72e10c7508f30dab043273943"
)
EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_SIZE = 11_278_473
EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_RECORDS = 2_932
VISUAL_LATCH_AUDIT_SCHEMA_VERSION = 2
TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION = 3
REQUIRED_TIP_PROBE_DESCENT_STEP_M = 0.002
REQUIRED_TIP_PROBE_MAX_TOTAL_DESCENT_M = 0.025
DEFAULT_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL = (
    REPOSITORY_ROOT
    / "outputs/firefly_deployment_hold_fullcycle_v4_100k_tip_probe_v1/"
    "act_30s.jsonl"
)
EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SHA256 = (
    "e31d0834d29c6c06dd74224ebda525f926bf8d69ae5a6ae65b5abdff7a352a99"
)
EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE = 3_960_524
EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS = 976
EXPECTED_TIP_SERVO_OPEN_ONLY_COMMAND_CYCLES = 906
TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID = 487
TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID = 494
TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION = 6
WRIST_VIDEO_KEY = "observation.images.wrist"
DEFAULT_FPS = 30
DEFAULT_JAW_CROSSING_RAD = 0.370
DEFAULT_JAW_OPEN_CONFIRM_RAD = 0.380
DEFAULT_JAW_CLOSED_CONFIRM_RAD = 0.350
DEFAULT_POLICY_CLOSE_INTENT_MAX_RAD = 0.350
DEFAULT_POLICY_CLOSE_INTENT_TIMEOUT_MS = 500.0
DEFAULT_ARM_URDF_PATH = Path(
    "/home/kk/kk_data/git_clone/hex-descriptions/products/firefly_y6/urdf/"
    "xpkg_urdf_firefly_y6.urdf"
)
EXPECTED_ARM_URDF_SHA256 = (
    "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291"
)
ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
FULLCYCLE_V4_ACTION_STATE_NAMES = tuple(
    [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
)
TCP_TOOL_OFFSET_M = (0.0, 0.0, 0.13)
DEFAULT_CLOSE_WINDOW_BEFORE_FRAMES = 30
DEFAULT_CLOSE_WINDOW_AFTER_FRAMES = 30
DEFAULT_RELEASE_WINDOW_BEFORE_FRAMES = 90
DEFAULT_RELEASE_WINDOW_AFTER_FRAMES = 30
STATE_MACHINE_MILESTONES = (
    "grasp_latched",
    "grasp_confirmed",
    "lift_confirmed",
    "transport_confirmed",
    "box_aligned",
    "release_unlatched",
    "release_confirmed",
    "home_closing_started",
    "task_done",
)


class VisualLatchAuditInputError(RuntimeError):
    """The dataset cannot be audited without making an unsafe assumption."""


@dataclass(frozen=True)
class PhaseLabels:
    pregrasp_open_frame: int
    pickup_close_frame: int
    release_open_frame: int
    return_close_frame: int


@dataclass(frozen=True)
class AuditWindows:
    close_before_frames: int = DEFAULT_CLOSE_WINDOW_BEFORE_FRAMES
    close_after_frames: int = DEFAULT_CLOSE_WINDOW_AFTER_FRAMES
    release_before_frames: int = DEFAULT_RELEASE_WINDOW_BEFORE_FRAMES
    release_after_frames: int = DEFAULT_RELEASE_WINDOW_AFTER_FRAMES

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class TipProbeFrame:
    """Minimal detector record consumed by the trace-envelope audit.

    This is deliberately not a dynamics observation. In particular, replaying
    frames after a hypothetical takeover cannot prove what a real tip-servo
    command would have caused the camera to observe.
    """

    trace_index: int
    frame_id: int
    fresh_unique: bool
    candidate_count: int
    centroid_norm: tuple[float, float] | None
    area_fraction: float | None


def _finite_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 7
    ):
        raise VisualLatchAuditInputError(f"{label} must contain seven values")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VisualLatchAuditInputError(f"{label} is not numeric") from exc
    if not all(math.isfinite(item) for item in vector):
        raise VisualLatchAuditInputError(f"{label} contains NaN or infinity")
    return vector


def _strict_json_number(value: Any, *, label: str) -> float:
    """Accept a finite JSON number without coercing strings or booleans."""

    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise VisualLatchAuditInputError(f"{label} must be a finite JSON number")
    return float(value)


def _strict_json_vector(
    value: Any, *, length: int, label: str
) -> tuple[float, ...]:
    """Parse one fixed-length JSON numeric vector without lossy coercion."""

    if not isinstance(value, list) or len(value) != length:
        raise VisualLatchAuditInputError(
            f"{label} must be a {length}-value JSON array"
        )
    return tuple(
        _strict_json_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def reconstruct_phase_labels(
    actions: Sequence[Sequence[float]],
    *,
    jaw_crossing_rad: float = DEFAULT_JAW_CROSSING_RAD,
    jaw_open_confirm_rad: float = DEFAULT_JAW_OPEN_CONFIRM_RAD,
    jaw_closed_confirm_rad: float = DEFAULT_JAW_CLOSED_CONFIRM_RAD,
) -> PhaseLabels:
    """Reconstruct four canonical crossings with open/closed hysteresis.

    The returned labels remain the crossings of ``jaw_crossing_rad``.  The
    wider confirmation levels only reject threshold chatter and shallow
    post-release jaw corrections before choosing those crossings.
    """

    thresholds = (jaw_closed_confirm_rad, jaw_crossing_rad, jaw_open_confirm_rad)
    if not all(math.isfinite(value) for value in thresholds) or not (
        jaw_closed_confirm_rad < jaw_crossing_rad < jaw_open_confirm_rad
    ):
        raise VisualLatchAuditInputError(
            "jaw thresholds must be finite and ordered closed < crossing < open"
        )
    if len(actions) < 2:
        raise VisualLatchAuditInputError("episode must contain at least two actions")
    jaw = [
        _finite_vector(action, label=f"action[{index}]")[6]
        for index, action in enumerate(actions)
    ]
    above = [value >= jaw_crossing_rad for value in jaw]
    rising = [
        index
        for index in range(1, len(above))
        if above[index] and not above[index - 1]
    ]
    falling = [
        index
        for index in range(1, len(above))
        if not above[index] and above[index - 1]
    ]

    def first_confirmed(start: int, *, opening: bool) -> int:
        for index in range(start, len(jaw)):
            if (
                opening
                and jaw[index] >= jaw_open_confirm_rad
                or not opening
                and jaw[index] <= jaw_closed_confirm_rad
            ):
                return index
        raise VisualLatchAuditInputError(
            "action[6] does not contain a complete confirmed open/close cycle"
        )

    def latest_crossing(
        candidates: Sequence[int], *, after: int, through: int, name: str
    ) -> int:
        matches = [index for index in candidates if after < index <= through]
        if not matches:
            raise VisualLatchAuditInputError(
                f"action[6] has no {name} crossing before its confirmation"
            )
        return matches[-1]

    if above[0] or jaw[0] > jaw_closed_confirm_rad:
        raise VisualLatchAuditInputError(
            "action[6] canonical cycle must begin in the confirmed closed state"
        )
    first_open_confirm = first_confirmed(1, opening=True)
    pregrasp_open = latest_crossing(
        rising,
        after=0,
        through=first_open_confirm,
        name="pregrasp rising",
    )
    pickup_close_confirm = first_confirmed(first_open_confirm + 1, opening=False)
    pickup_close = latest_crossing(
        falling,
        after=first_open_confirm,
        through=pickup_close_confirm,
        name="pickup falling",
    )
    release_open_confirm = first_confirmed(pickup_close_confirm + 1, opening=True)
    release_open = latest_crossing(
        rising,
        after=pickup_close_confirm,
        through=release_open_confirm,
        name="release rising",
    )
    return_close_confirm = first_confirmed(release_open_confirm + 1, opening=False)
    return_close = latest_crossing(
        falling,
        after=release_open_confirm,
        through=return_close_confirm,
        name="return falling",
    )
    return PhaseLabels(
        pregrasp_open,
        pickup_close,
        release_open,
        return_close,
    )


def _first_confirmation(
    flags: Sequence[bool], *, start: int, stop: int, frames: int
) -> int | None:
    """Return the first end-frame of a streak wholly inside ``[start, stop)``."""

    if frames < 1:
        raise ValueError("confirmation frames must be positive")
    start = max(0, start)
    stop = min(len(flags), max(start, stop))
    streak = 0
    for index in range(start, stop):
        streak = streak + 1 if bool(flags[index]) else 0
        if streak >= frames:
            return index
    return None


def _confirmation_starts(
    flags: Sequence[bool], *, start: int, stop: int, frames: int
) -> list[int]:
    """Return one confirmation end-frame per true run in ``[start, stop)``."""

    start = max(0, start)
    stop = min(len(flags), max(start, stop))
    result: list[int] = []
    streak = 0
    for index in range(start, stop):
        if bool(flags[index]):
            streak += 1
            if streak == frames:
                result.append(index)
        else:
            streak = 0
    return result


def _max_false_run(flags: Sequence[bool], *, start: int, stop: int) -> int:
    return _max_false_run_details(flags, start=start, stop=stop)["frames"]


def _max_false_run_details(
    flags: Sequence[bool], *, start: int, stop: int
) -> dict[str, int | None]:
    maximum = 0
    current = 0
    maximum_start: int | None = None
    maximum_end: int | None = None
    current_start: int | None = None
    first = max(0, start)
    last = min(len(flags), max(start, stop))
    for index in range(first, last):
        value = flags[index]
        if bool(value):
            current = 0
            current_start = None
        else:
            if current_start is None:
                current_start = index
            current += 1
            if current > maximum:
                maximum = current
                maximum_start = current_start
                maximum_end = index
    return {
        "frames": maximum,
        "start_frame": maximum_start,
        "end_frame": maximum_end,
    }


def _reason_counts(
    reasons: Sequence[Sequence[str]], *, start: int, stop: int
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for frame_reasons in reasons[max(0, start) : min(len(reasons), max(start, stop))]:
        counter.update(str(reason) for reason in frame_reasons)
    return dict(sorted(counter.items()))


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return deterministic linearly-interpolated diagnostics quantiles."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "q05": None,
            "median": None,
            "q95": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "q05": at(0.05),
        "median": at(0.50),
        "q95": at(0.95),
        "max": ordered[-1],
    }


def _contour_rank_summary(
    diagnostics: Sequence[Mapping[str, Any]], rank: int
) -> dict[str, Any]:
    blobs = [
        frame["ambiguity_candidates"][rank]
        for frame in diagnostics
        if len(frame["ambiguity_candidates"]) > rank
    ]

    def component(field: str, index: int) -> dict[str, float | int | None]:
        return _quantiles([float(blob[field][index]) for blob in blobs])

    return {
        "frames_with_contour": len(blobs),
        "area_fraction": _quantiles(
            [float(blob["area_fraction"]) for blob in blobs]
        ),
        "centroid_norm": {
            "x": component("centroid_norm", 0),
            "y": component("centroid_norm", 1),
        },
        "bbox_norm": {
            "left": component("bbox_norm", 0),
            "top": component("bbox_norm", 1),
            "width": component("bbox_norm", 2),
            "height": component("bbox_norm", 3),
        },
    }


def _contour_window_summary(
    diagnostics: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    positive_counts = Counter(
        int(frame["positive_contour_count"]) for frame in diagnostics
    )
    candidate_counts = Counter(
        len(frame["ambiguity_candidates"]) for frame in diagnostics
    )
    return {
        "frames": len(diagnostics),
        "positive_contour_count_distribution": {
            str(key): value for key, value in sorted(positive_counts.items())
        },
        "ambiguity_candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "largest_ambiguity_candidate": _contour_rank_summary(diagnostics, 0),
        "second_largest_ambiguity_candidate": _contour_rank_summary(
            diagnostics, 1
        ),
        "third_largest_ambiguity_candidate": _contour_rank_summary(
            diagnostics, 2
        ),
    }


def audit_evidence_flags(
    labels: PhaseLabels,
    *,
    close_flags: Sequence[bool],
    held_flags: Sequence[bool],
    blue_flags: Sequence[bool],
    strict_release_flags: Sequence[bool] | None = None,
    close_runtime_gate_flags: Sequence[bool] | None = None,
    contact_confirm_frame: int | None = None,
    close_reasons: Sequence[Sequence[str]] | None = None,
    held_reasons: Sequence[Sequence[str]] | None = None,
    blue_reasons: Sequence[Sequence[str]] | None = None,
    config: VisualPhaseLatchConfig,
    fps: int = DEFAULT_FPS,
    windows: AuditWindows = AuditWindows(),
) -> dict[str, Any]:
    """Audit already-detected unique-frame evidence against pseudo-labels."""

    lengths = {len(close_flags), len(held_flags), len(blue_flags)}
    if len(lengths) != 1:
        raise VisualLatchAuditInputError("visual evidence arrays have unequal lengths")
    length = lengths.pop()
    if strict_release_flags is None:
        strict_release_flags = [
            bool(held and blue)
            for held, blue in zip(held_flags, blue_flags, strict=True)
        ]
    elif len(strict_release_flags) != length:
        raise VisualLatchAuditInputError(
            "strict release gate array has the wrong length"
        )
    if close_runtime_gate_flags is None:
        close_runtime_gate_flags = close_flags
    elif len(close_runtime_gate_flags) != length:
        raise VisualLatchAuditInputError(
            "close runtime gate array has the wrong length"
        )
    if type(fps) is not int or fps <= 0:
        raise VisualLatchAuditInputError("fps must be a positive integer")
    if not 0 < labels.pregrasp_open_frame < labels.pickup_close_frame:
        raise VisualLatchAuditInputError("invalid pregrasp/pickup pseudo-label order")
    if not labels.pickup_close_frame < labels.release_open_frame < labels.return_close_frame:
        raise VisualLatchAuditInputError("invalid pickup/release/return pseudo-label order")
    if labels.return_close_frame >= length:
        raise VisualLatchAuditInputError("pseudo-label lies outside visual evidence")

    empty_reasons: list[tuple[str, ...]] = [()] * length
    close_reasons = empty_reasons if close_reasons is None else close_reasons
    held_reasons = empty_reasons if held_reasons is None else held_reasons
    blue_reasons = empty_reasons if blue_reasons is None else blue_reasons
    if {len(close_reasons), len(held_reasons), len(blue_reasons)} != {length}:
        raise VisualLatchAuditInputError("visual rejection arrays have unequal lengths")

    close_start = max(0, labels.pickup_close_frame - windows.close_before_frames)
    close_stop = min(length, labels.pickup_close_frame + windows.close_after_frames + 1)
    close_confirmation = _first_confirmation(
        close_flags,
        start=close_start,
        stop=close_stop,
        frames=config.close_confirm_frames,
    )
    premature_close = _confirmation_starts(
        close_runtime_gate_flags,
        start=0,
        stop=close_start,
        frames=config.close_confirm_frames,
    )
    runtime_close_confirmations = _confirmation_starts(
        close_runtime_gate_flags,
        start=0,
        stop=close_stop,
        frames=config.close_confirm_frames,
    )
    runtime_close_confirmation = (
        runtime_close_confirmations[0] if runtime_close_confirmations else None
    )

    contact_confirmed = contact_confirm_frame is not None
    held_start = (
        labels.pickup_close_frame
        if contact_confirm_frame is None
        else int(contact_confirm_frame)
    )
    if not labels.pickup_close_frame <= held_start < labels.release_open_frame:
        raise VisualLatchAuditInputError(
            "contact-confirm proxy lies outside pickup-close/release interval"
        )
    held_stop = labels.release_open_frame
    held_total = held_stop - held_start
    held_count = sum(bool(value) for value in held_flags[held_start:held_stop])
    max_held_missing_details = _max_false_run_details(
        held_flags, start=held_start, stop=held_stop
    )
    max_held_missing = int(max_held_missing_details["frames"])
    # The latch faults when elapsed missing time is strictly greater than the
    # timeout.  A run of N samples spans (N-1)/fps seconds.
    held_fault_run_frames = (
        math.floor(config.held_red_loss_fault_ms * fps / 1000.0) + 2
    )

    # Before pickup-close the object cannot yet be in transport.  Blue streaks
    # here are therefore unambiguous pre-transport false triggers.  Blue seen
    # later can legitimately precede the demonstrator's release command.
    premature_blue = _confirmation_starts(
        blue_flags,
        start=0,
        stop=labels.pickup_close_frame,
        frames=config.box_confirm_frames,
    )
    release_start = max(held_start, labels.release_open_frame - windows.release_before_frames)
    release_stop = min(length, labels.release_open_frame + windows.release_after_frames + 1)
    blue_only_confirmation = _first_confirmation(
        blue_flags,
        start=release_start,
        stop=release_stop,
        frames=config.box_confirm_frames,
    )
    release_confirmation = _first_confirmation(
        strict_release_flags,
        start=release_start,
        stop=release_stop,
        frames=config.box_confirm_frames,
    )

    close_ok = close_confirmation is not None
    held_ok = contact_confirmed and max_held_missing < held_fault_run_frames
    release_ok = release_confirmation is not None
    preclose_ok = not premature_close
    pretransport_ok = not premature_blue
    common_safe = (
        close_ok
        and contact_confirmed
        and release_ok
        and preclose_ok
        and pretransport_ok
    )
    return {
        "safe": common_safe and held_ok,
        "close": {
            "window_frames": [close_start, close_stop - 1],
            "accepted_frames_before_pickup_close": int(
                sum(bool(value) for value in close_flags[: labels.pickup_close_frame])
            ),
            "runtime_gate_true_frames_before_pickup_close": int(
                sum(
                    bool(value)
                    for value in close_runtime_gate_flags[
                        : labels.pickup_close_frame
                    ]
                )
            ),
            "premature_confirmation_frames": premature_close,
            "confirmation_frame": close_confirmation,
            "confirmation_offset_frames": (
                None
                if close_confirmation is None
                else close_confirmation - labels.pickup_close_frame
            ),
            "detected_in_window": close_ok,
            "runtime_gate_confirmation_frame": runtime_close_confirmation,
            "runtime_gate_confirmation_offset_frames": (
                None
                if runtime_close_confirmation is None
                else runtime_close_confirmation - labels.pickup_close_frame
            ),
            "rejection_reason_counts_in_window": _reason_counts(
                close_reasons, start=close_start, stop=close_stop
            ),
        },
        "held_red": {
            "interval_frames": [held_start, held_stop - 1],
            "contact_confirm_frame": contact_confirm_frame,
            "contact_confirm_offset_from_pickup_close_frames": (
                None
                if contact_confirm_frame is None
                else contact_confirm_frame - labels.pickup_close_frame
            ),
            "contact_confirmed": contact_confirmed,
            "accepted_frames": held_count,
            "total_frames": held_total,
            "coverage": held_count / held_total,
            "max_consecutive_missing_frames": max_held_missing,
            "max_missing_run": max_held_missing_details,
            "red_missing_frames": held_total - held_count,
            "red_loss_semantics": (
                "Keep the jaw closed and continue the current arm command while "
                "ordinary held-red evidence is absent. Missing red prevents visual "
                "lift/transport/release confirmation and faults only after the "
                "configured held_red_loss_fault_ms; contact/FK loss is outside this "
                "video-only replay and remains an immediate arm-hold condition."
            ),
            "fault_run_frames": held_fault_run_frames,
            "survives_loss_timeout": held_ok,
            "rejection_reason_counts": _reason_counts(
                held_reasons, start=held_start, stop=held_stop
            ),
        },
        "blue": {
            "accepted_frames_before_transport_can_start": int(
                sum(bool(value) for value in blue_flags[: labels.pickup_close_frame])
            ),
            "premature_confirmation_frames": premature_blue,
            "release_window_frames": [release_start, release_stop - 1],
            "blue_only_confirmation_frame": blue_only_confirmation,
            "blue_only_detected_in_release_window": (
                blue_only_confirmation is not None
            ),
            "release_confirmation_frame": release_confirmation,
            "release_confirmation_offset_frames": (
                None
                if release_confirmation is None
                else release_confirmation - labels.release_open_frame
            ),
            "detected_in_release_window": release_ok,
            "release_gate": "fresh held-red AND blue",
            "rejection_reason_counts_in_release_window": _reason_counts(
                blue_reasons, start=release_start, stop=release_stop
            ),
        },
    }


def _runtime_gate_proxies(
    rows: Sequence[Mapping[str, Any]],
    *,
    close_visual_flags: Sequence[bool],
    config: VisualPhaseLatchConfig,
    fps: int,
) -> dict[str, Any]:
    """Replay measured-speed and state/command contact gates from a demo."""

    if len(rows) != len(close_visual_flags):
        raise VisualLatchAuditInputError(
            "runtime proxy rows and visual flags have unequal lengths"
        )
    states = [
        _finite_vector(row.get("observation.state"), label=f"state[{index}]")
        for index, row in enumerate(rows)
    ]
    commands = [
        _finite_vector(row.get("action"), label=f"action[{index}]")
        for index, row in enumerate(rows)
    ]
    arm_speeds: list[float | None] = [None]
    jaw_speeds: list[float | None] = [None]
    for previous, current in zip(states, states[1:]):
        arm_speeds.append(
            max(abs(current[joint] - previous[joint]) for joint in range(6))
            * fps
        )
        jaw_speeds.append(abs(current[6] - previous[6]) * fps)

    policy_close_intent_max_rad = config.pickup_policy_close_intent_max_rad
    policy_close_intent_timeout_ms = config.pickup_policy_close_intent_hold_ms
    if not math.isfinite(policy_close_intent_max_rad):
        raise VisualLatchAuditInputError("policy close-intent threshold must be finite")
    if (
        not math.isfinite(policy_close_intent_timeout_ms)
        or policy_close_intent_timeout_ms <= 0
    ):
        raise VisualLatchAuditInputError(
            "policy close-intent timeout must be finite and positive"
        )
    intent_timeout_frames = math.floor(
        policy_close_intent_timeout_ms * fps / 1000.0
    )
    pregrasp_open_seen = False
    last_policy_close_intent_frame: int | None = None
    first_policy_close_intent_frame: int | None = None
    policy_close_intent_flags: list[bool] = []
    close_gate_flags: list[bool] = []
    for index, (state, command) in enumerate(zip(states, commands, strict=True)):
        if (
            state[6] >= config.pregrasp_open_ready_rad
            or command[6] >= config.pregrasp_open_ready_rad
        ):
            pregrasp_open_seen = True
        # ``command`` is the recorded raw policy/teleop action and is used only
        # as close intent.  In deployment the accepted command remains clamped
        # at pregrasp_open_target_rad until the latch fires, so recorded jaw
        # closure must not make this counterfactual jaw-open gate false.
        jaw_is_open = pregrasp_open_seen
        if (
            pregrasp_open_seen
            and command[6] <= policy_close_intent_max_rad
        ):
            last_policy_close_intent_frame = index
            if first_policy_close_intent_frame is None:
                first_policy_close_intent_frame = index
        policy_close_intent_active = bool(
            last_policy_close_intent_frame is not None
            and index - last_policy_close_intent_frame < intent_timeout_frames
        )
        policy_close_intent_flags.append(policy_close_intent_active)
        speed = arm_speeds[index]
        close_gate_flags.append(
            bool(
                pregrasp_open_seen
                and jaw_is_open
                and policy_close_intent_active
                and close_visual_flags[index]
                and speed is not None
                and speed <= config.close_max_arm_speed_rad_s
            )
        )
    close_latch_frame = _first_confirmation(
        close_gate_flags,
        start=0,
        stop=len(rows),
        frames=config.close_confirm_frames,
    )

    contact_flags = [False] * len(rows)
    contact_gaps: list[float | None] = [None] * len(rows)
    simulated_commands: list[float | None] = [None] * len(rows)
    if close_latch_frame is not None:
        simulated_command = config.pregrasp_open_target_rad
        for index in range(close_latch_frame, len(rows)):
            error = config.close_target_rad - simulated_command
            step = max(
                -config.ee_step_limit_rad,
                min(config.ee_step_limit_rad, error),
            )
            simulated_command += step
            simulated_commands[index] = simulated_command
        for index in range(close_latch_frame + 1, len(rows)):
            simulated_command = simulated_commands[index]
            if simulated_command is None:
                continue
            gap = states[index][6] - simulated_command
            contact_gaps[index] = gap
            speed = jaw_speeds[index]
            contact_flags[index] = bool(
                close_visual_flags[index]
                and simulated_command <= config.close_command_ready_max_rad
                and config.contact_gap_min_rad <= gap <= config.contact_gap_max_rad
                and speed is not None
                and speed <= config.jaw_settled_speed_max_rad_s
            )
    contact_confirm_frame = _first_confirmation(
        contact_flags,
        start=0,
        stop=len(rows),
        frames=config.grasp_confirm_frames,
    )
    return {
        "close_gate_flags": close_gate_flags,
        "policy_close_intent_flags": policy_close_intent_flags,
        "first_policy_close_intent_frame": first_policy_close_intent_frame,
        "policy_close_intent_timeout_frames": intent_timeout_frames,
        "close_latch_frame": close_latch_frame,
        "contact_flags": contact_flags,
        "contact_confirm_frame": contact_confirm_frame,
        "contact_gap_rad_at_confirmation": (
            None
            if contact_confirm_frame is None
            else contact_gaps[contact_confirm_frame]
        ),
        "simulated_command_ee_rad_at_contact_confirmation": (
            None
            if contact_confirm_frame is None
            else simulated_commands[contact_confirm_frame]
        ),
        "arm_speed_rad_s_at_close_latch": (
            None
            if close_latch_frame is None
            else arm_speeds[close_latch_frame]
        ),
        "jaw_speed_rad_s_at_contact_confirmation": (
            None
            if contact_confirm_frame is None
            else jaw_speeds[contact_confirm_frame]
        ),
    }


def _load_fk_model(
    path: Path,
) -> tuple[SerialRevoluteGravityModel, dict[str, Any]]:
    """Load the hash-pinned Firefly URDF used by the deployed follower."""

    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise VisualLatchAuditInputError(
            f"arm URDF must be a regular, non-symlink file: {path}"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VisualLatchAuditInputError(f"cannot read arm URDF {path}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_ARM_URDF_SHA256:
        raise VisualLatchAuditInputError(
            "arm URDF SHA-256 does not match the deployed Firefly Y6 model: "
            f"{digest}"
        )
    try:
        xml = data.decode("utf-8")
        model = SerialRevoluteGravityModel.from_urdf_string(
            xml, ARM_JOINT_NAMES
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise VisualLatchAuditInputError(f"invalid arm URDF {path}: {exc}") from exc
    return model, {
        "path": str(path),
        "size": len(data),
        "sha256": digest,
        "joint_names": list(ARM_JOINT_NAMES),
        "tcp_tool_offset_m": list(TCP_TOOL_OFFSET_M),
    }


def _state_machine_replay(
    rows: Sequence[Mapping[str, Any]],
    evidences: Sequence[Any],
    labels: PhaseLabels,
    *,
    config: VisualPhaseLatchConfig,
    fk_model: SerialRevoluteGravityModel,
) -> dict[str, Any]:
    """Replay the exact visual latch against measured demo state and FK.

    Policy arm axes and measured states come from the demonstration. The EE
    command reference is counterfactually composed exactly as deployment does,
    including the 0.0165-rad slew. This proves the latch gates against all 50
    recorded trajectories without claiming a dynamics simulation.
    """

    if not config.enabled:
        raise VisualLatchAuditInputError(
            "exact state-machine replay requires VisualPhaseLatchConfig.enabled=true"
        )
    if len(rows) != len(evidences):
        raise VisualLatchAuditInputError(
            "state-machine rows and visual evidences have unequal lengths"
        )
    if not rows:
        raise VisualLatchAuditInputError("state-machine replay episode is empty")

    states = [
        _finite_vector(row.get("observation.state"), label=f"state[{index}]")
        for index, row in enumerate(rows)
    ]
    policies = [
        _finite_vector(row.get("action"), label=f"action[{index}]")
        for index, row in enumerate(rows)
    ]
    tcp_positions: list[tuple[float, float, float]] = []
    for index, state in enumerate(states):
        try:
            position, _rotation = fk_model.end_effector_pose(
                state[:6], tool_offset=TCP_TOOL_OFFSET_M
            )
        except ValueError as exc:
            raise VisualLatchAuditInputError(
                f"FK failed for measured state[{index}]: {exc}"
            ) from exc
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise VisualLatchAuditInputError(
                f"FK returned invalid TCP position for state[{index}]"
            )
        tcp_positions.append(tuple(float(value) for value in position))

    home_center = states[0][:6]
    latch = VisualGraspReleaseLatch(config)
    reference = list(states[0])
    previous_tcp_position: tuple[float, float, float] | None = None
    previous_tcp_ns: int | None = None
    previous_measured: tuple[float, ...] | None = None
    previous_measured_ns: int | None = None
    intent_until_ns: int | None = None
    last_arm_hold: bool | None = None
    milestone_frames: dict[str, int] = {}
    transition_events: list[dict[str, Any]] = []
    fault_frames: list[dict[str, Any]] = []
    command_ee_trace: list[float] = []
    grasp_contact_measured_ee_rad: float | None = None
    counterfactual_held_ee_frames = 0
    replan_events = frozenset((*STATE_MACHINE_MILESTONES, "fault_hold"))

    for frame_index, (state, policy, evidence, tcp_position) in enumerate(
        zip(states, policies, evidences, tcp_positions, strict=True)
    ):
        now_ns = int(evidence.evaluated_at_ns)
        measured_for_latch = list(state)
        if (
            grasp_contact_measured_ee_rad is not None
            and latch.phase
            in {
                VisualLatchPhase.GRASP_CONFIRMED,
                VisualLatchPhase.LIFT_CONFIRMED,
                VisualLatchPhase.TRANSPORT_CONFIRMED,
                VisualLatchPhase.BOX_ALIGNED,
            }
        ):
            # The demonstration opens its jaw on the human release command.
            # Deployment suppresses that policy action until the visual release
            # gate fires, so feeding the recorded opening into the counterfactual
            # held-command replay would manufacture contact loss. Preserve only
            # the measured grasp-contact aperture. This models the cube's hard
            # aperture constraint without inventing a jaw dynamics response.
            measured_for_latch[6] = grasp_contact_measured_ee_rad
            if not math.isclose(
                measured_for_latch[6], state[6], rel_tol=0.0, abs_tol=1e-12
            ):
                counterfactual_held_ee_frames += 1
        if policy[6] <= config.pickup_policy_close_intent_max_rad:
            intent_until_ns = now_ns + int(
                config.pickup_policy_close_intent_hold_ms * 1_000_000.0
            )
        policy_close_intent = bool(
            intent_until_ns is not None and now_ns < intent_until_ns
        )
        if not policy_close_intent:
            intent_until_ns = None

        tcp_speed_m_s: float | None = None
        if previous_tcp_position is not None and previous_tcp_ns is not None:
            elapsed_s = (now_ns - previous_tcp_ns) / 1_000_000_000.0
            if elapsed_s > 0:
                tcp_speed_m_s = math.sqrt(
                    sum(
                        (current - previous) ** 2
                        for current, previous in zip(
                            tcp_position, previous_tcp_position, strict=True
                        )
                    )
                ) / elapsed_s

        arm_speed_rad_s: float | None = None
        if previous_measured is not None and previous_measured_ns is not None:
            elapsed_s = (now_ns - previous_measured_ns) / 1_000_000_000.0
            if elapsed_s > 0:
                arm_speed_rad_s = max(
                    abs(state[joint] - previous_measured[joint])
                    for joint in range(6)
                ) / elapsed_s
        home_ready = bool(
            arm_speed_rad_s is not None
            and arm_speed_rad_s <= config.home_max_arm_speed_rad_s
            and all(
                abs(state[joint] - home_center[joint])
                <= config.home_arm_tolerance_rad[joint]
                for joint in range(6)
            )
            and policy[6] <= config.home_policy_close_intent_max_rad
        )

        decision = latch.step(
            evidence,
            measured_for_latch,
            reference,
            tcp_z_m=tcp_position[2],
            tcp_speed_m_s=tcp_speed_m_s,
            home_ready=home_ready,
            policy_close_intent=policy_close_intent,
            external_fault_reason=None,
        )
        arm_hold_edge = (
            last_arm_hold is not None and bool(decision.arm_hold) != last_arm_hold
        )
        requires_replan = bool(
            decision.event in replan_events or arm_hold_edge
        )
        composed = compose_visual_action(
            policy,
            reference,
            decision,
            ee_step_limit_rad=config.ee_step_limit_rad,
        )
        action = list(composed.action)
        if composed.arm_hold or requires_replan:
            action[:6] = reference[:6]
        reference = action
        command_ee_trace.append(float(reference[6]))
        last_arm_hold = bool(decision.arm_hold)

        if decision.event is not None:
            event_record = {
                "frame": frame_index,
                "event": decision.event,
                "phase_before": decision.phase_before.value,
                "phase_after": decision.phase_after.value,
                "recorded_measured_ee_rad": state[6],
                "effective_measured_ee_rad": measured_for_latch[6],
                "command_ee_rad": reference[6],
                "contact_gap_rad": decision.contact_gap_rad,
                "held_contact_ok": decision.held_contact_ok,
                "tcp_z_m": tcp_position[2],
                "tcp_speed_m_s": tcp_speed_m_s,
                "tcp_delta_z_m": decision.tcp_delta_z_m,
                "home_ready": home_ready,
                "policy_close_intent": policy_close_intent,
            }
            transition_events.append(event_record)
            if decision.event in STATE_MACHINE_MILESTONES:
                if decision.event in milestone_frames:
                    raise VisualLatchAuditInputError(
                        f"state machine repeated milestone {decision.event!r}"
                    )
                milestone_frames[decision.event] = frame_index
            if decision.event == "grasp_confirmed":
                grasp_contact_measured_ee_rad = float(measured_for_latch[6])
        if (
            decision.phase_after is VisualLatchPhase.FAULT_HOLD
            or decision.fault_reason is not None
        ) and not fault_frames:
            fault_frames.append(
                {
                    "frame": frame_index,
                    "event": decision.event,
                    "reason": decision.fault_reason,
                }
            )

        previous_tcp_position = tcp_position
        previous_tcp_ns = now_ns
        previous_measured = state
        previous_measured_ns = now_ns

    ordered_frames = [milestone_frames.get(name) for name in STATE_MACHINE_MILESTONES]
    ordered_complete = bool(
        all(frame is not None for frame in ordered_frames)
        and all(
            int(previous) < int(current)
            for previous, current in zip(
                ordered_frames, ordered_frames[1:]
            )
        )
        and latch.phase is VisualLatchPhase.DONE
    )
    fault_free = not fault_frames and latch.phase is not VisualLatchPhase.FAULT_HOLD

    def offset(event: str, anchor: int) -> int | None:
        frame = milestone_frames.get(event)
        return None if frame is None else frame - anchor

    return {
        "model": (
            "exact VisualGraspReleaseLatch + compose_visual_action replay; "
            "recorded measured arm trajectory and policy arm axes are not a dynamics "
            "simulation; while release remains latched, recorded human jaw opening is "
            "upper-clamped to the episode's measured grasp-contact aperture"
        ),
        "required_milestones": list(STATE_MACHINE_MILESTONES),
        "milestone_frames": milestone_frames,
        "ordered_complete": ordered_complete,
        "fault_free": fault_free,
        "final_phase": latch.phase.value,
        "fault_frames": fault_frames,
        "counterfactual_held_ee": {
            "grasp_contact_measured_ee_rad": grasp_contact_measured_ee_rad,
            "substituted_frames": counterfactual_held_ee_frames,
            "rule": (
                "effective measured EE = measured EE at grasp_confirmed in "
                "GRASP_CONFIRMED through BOX_ALIGNED; recorded measured EE resumes "
                "in RELEASING"
            ),
        },
        "events": transition_events,
        "offsets_frames": {
            "grasp_latched_from_pickup_close": offset(
                "grasp_latched", labels.pickup_close_frame
            ),
            "grasp_confirmed_from_pickup_close": offset(
                "grasp_confirmed", labels.pickup_close_frame
            ),
            "release_unlatched_from_release_open": offset(
                "release_unlatched", labels.release_open_frame
            ),
            "release_confirmed_from_release_open": offset(
                "release_confirmed", labels.release_open_frame
            ),
            "home_closing_started_from_return_close": offset(
                "home_closing_started", labels.return_close_frame
            ),
            "task_done_from_return_close": offset(
                "task_done", labels.return_close_frame
            ),
        },
        "command_ee_rad": {
            "min": min(command_ee_trace),
            "max": max(command_ee_trace),
            "final": command_ee_trace[-1],
        },
    }


def _capture_source_binding(
    dataset_root: Path,
    *,
    dataset_repo_id: str,
    controller_contract: Path,
    expected_task: str,
    config: VisualPhaseLatchConfig,
    checkpoint: Path | None = None,
    arm6_attestation_capture: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Bind the report to the formal training inputs and implementation."""

    try:
        capture = capture_training_inputs(
            dataset_root,
            dataset_repo_id,
            controller_contract=controller_contract,
            expected_task=expected_task,
        )
        profile = visual_phase_latch_provenance(config)
        manifest = visual_phase_latch_implementation_manifest()
    except (OSError, ValueError, PermissionError, TrainingAttestationError) as exc:
        raise VisualLatchAuditInputError(
            f"cannot capture audit source binding: {exc}"
        ) from exc
    audit_source = manifest.get("offline_visual_audit")
    if not isinstance(audit_source, Mapping):
        raise VisualLatchAuditInputError(
            "visual implementation manifest omits offline_visual_audit"
        )
    audit_digest = audit_source.get("sha256")
    if not isinstance(audit_digest, str) or len(audit_digest) != 64:
        raise VisualLatchAuditInputError(
            "visual implementation manifest has invalid audit script SHA-256"
        )
    binding = {
        "training_input_digest": capture.digest,
        "visual_profile": profile,
        "visual_implementation_manifest": manifest,
        "audit_script_sha256": audit_digest,
    }
    open_only = bool(
        config.tip_servo.mode is TipServoMode.EXECUTE
        and config.tip_servo.probe_profile is TipServoProbeProfile.OPEN_ONLY
    )
    if open_only:
        if checkpoint is None:
            raise VisualLatchAuditInputError(
                "schema-6 OPEN_ONLY source binding requires an attested checkpoint"
            )
        arm6_binding = _capture_arm6_source_binding(
            checkpoint,
            capture_factory=arm6_attestation_capture,
        )
        projection = arm6_binding["projection_binding"]
        projection_source = projection.get("source")
        if not isinstance(projection_source, dict):
            raise VisualLatchAuditInputError(
                "arm6 projection_binding.source must be an object"
            )
        expected_projection_source = {
            "repo_id": dataset_repo_id,
            "root": str(dataset_root),
            "upstream_training_input_digest": capture.digest,
        }
        for name, expected in expected_projection_source.items():
            actual = projection_source.get(name)
            if actual != expected or type(actual) is not type(expected):
                raise VisualLatchAuditInputError(
                    "arm6 projection source does not match the audited original "
                    f"7-D source: {name} expected {expected!r}, found {actual!r}"
                )
        binding.update(arm6_binding)
    elif checkpoint is not None or arm6_attestation_capture is not None:
        raise VisualLatchAuditInputError(
            "arm6 checkpoint binding is valid only for schema-6 OPEN_ONLY"
        )
    return binding


def _detector_config_from_source_binding(
    source_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the exact typed detector provenance bound into the report."""

    profile = source_binding.get("visual_profile")
    if not isinstance(profile, Mapping):
        raise VisualLatchAuditInputError(
            "source binding visual_profile must be an object"
        )
    fields = profile.get("fields")
    if not isinstance(fields, Mapping):
        raise VisualLatchAuditInputError(
            "source binding visual_profile.fields must be an object"
        )
    return fields


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualLatchAuditInputError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualLatchAuditInputError(f"{label} must contain a JSON object")
    return value


def _require_original_fullcycle_v4_7d_source(info: Mapping[str, Any]) -> None:
    """Reject the 6-D policy projection before reading ``action[6]``.

    Schema 6 still derives visual pseudo-labels from the immutable original
    demonstrations.  The arm-only projection is a policy-training input; it
    deliberately has no jaw channel and therefore cannot supply the four jaw
    crossings used by this observational audit.
    """

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise VisualLatchAuditInputError("dataset info.features must be an object")
    expected_names = list(FULLCYCLE_V4_ACTION_STATE_NAMES)
    for name in ("action", "observation.state"):
        feature = features.get(name)
        if not isinstance(feature, Mapping):
            raise VisualLatchAuditInputError(
                f"dataset info.features[{name!r}] must be an object"
            )
        shape = feature.get("shape")
        names = feature.get("names")
        if shape == [6]:
            raise VisualLatchAuditInputError(
                "schema-6 OPEN_ONLY visual audit requires the original 7-D "
                "fullcycle-v4 source; the 6-D arm-only projection is policy-only "
                f"and cannot provide {name}[6] jaw evidence"
            )
        if (
            feature.get("dtype") != "float32"
            or shape != [7]
            or names != expected_names
        ):
            raise VisualLatchAuditInputError(
                f"dataset {name} must be the exact original 7-D float32 J1-J6+jaw "
                "feature in canonical order"
            )


def _strict_json_contract_value(value: Any, *, label: str) -> Any:
    """Copy one consumer-owned binding without coercing non-JSON values."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise VisualLatchAuditInputError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _strict_json_contract_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise VisualLatchAuditInputError(
                    f"{label} object keys must be strings"
                )
            result[key] = _strict_json_contract_value(
                item, label=f"{label}.{key}"
            )
        return result
    raise VisualLatchAuditInputError(
        f"{label} contains non-JSON value of type {type(value).__name__}"
    )


def _capture_arm6_source_binding(
    checkpoint: Path,
    *,
    capture_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Delegate all attestation semantics to the frozen schema-6 consumer."""

    configured_checkpoint = Path(checkpoint).expanduser()
    if not configured_checkpoint.is_absolute():
        raise VisualLatchAuditInputError(
            "schema-6 OPEN_ONLY checkpoint must be an explicit absolute path"
        )
    if capture_factory is None:
        try:
            from lerobot_robot_hexbase.firefly_arm6_attestation import (
                capture_firefly_arm6_attestation,
            )
        except ImportError as exc:
            raise VisualLatchAuditInputError(
                "schema-6 arm6 attestation consumer is unavailable"
            ) from exc
        capture_factory = capture_firefly_arm6_attestation
    try:
        capture = capture_factory()
    except Exception as exc:
        raise VisualLatchAuditInputError(
            f"cannot capture schema-6 arm6 attestation: {type(exc).__name__}: {exc}"
        ) from exc

    selected_raw = getattr(capture, "selected_checkpoint", None)
    selected_step = getattr(capture, "selected_step", None)
    if not isinstance(selected_raw, (str, Path)):
        raise VisualLatchAuditInputError(
            "arm6 attestation capture selected_checkpoint must be a path"
        )
    selected_checkpoint = Path(selected_raw)
    if not selected_checkpoint.is_absolute():
        raise VisualLatchAuditInputError(
            "arm6 attestation selected_checkpoint must be absolute"
        )
    if configured_checkpoint != selected_checkpoint:
        raise VisualLatchAuditInputError(
            "schema-6 checkpoint must exactly equal the attested selected 020000 "
            f"checkpoint; configured={configured_checkpoint}, selected={selected_checkpoint}"
        )
    if type(selected_step) is not int or selected_step != 20_000:
        raise VisualLatchAuditInputError(
            "arm6 attestation selected_step must be exactly 20000"
        )
    source_binding_method = getattr(capture, "source_binding", None)
    if not callable(source_binding_method):
        raise VisualLatchAuditInputError(
            "arm6 attestation capture must provide source_binding()"
        )
    try:
        raw_binding = source_binding_method()
    except Exception as exc:
        raise VisualLatchAuditInputError(
            f"cannot serialize schema-6 arm6 binding: {type(exc).__name__}: {exc}"
        ) from exc
    binding = _strict_json_contract_value(
        raw_binding, label="arm6 attestation source binding"
    )
    if not isinstance(binding, dict) or set(binding) != {
        "policy_attestation",
        "projection_binding",
    }:
        raise VisualLatchAuditInputError(
            "arm6 attestation source_binding() must contain exactly "
            "policy_attestation and projection_binding"
        )
    if not isinstance(binding["policy_attestation"], dict) or not isinstance(
        binding["projection_binding"], dict
    ):
        raise VisualLatchAuditInputError(
            "arm6 policy_attestation and projection_binding must be objects"
        )
    return binding


def _read_parquet_rows(
    paths: Sequence[Path], *, columns: Sequence[str] | None = None, label: str
) -> list[dict[str, Any]]:
    if not paths:
        raise VisualLatchAuditInputError(f"dataset contains no {label} parquet files")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise VisualLatchAuditInputError("pyarrow is required for this audit") from exc
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            table = pq.read_table(path, columns=None if columns is None else list(columns))
        except Exception as exc:
            raise VisualLatchAuditInputError(f"cannot read {label} {path}: {exc}") from exc
        rows.extend(dict(row) for row in table.to_pylist())
    return rows


def _load_dataset(
    root: Path, *, expected_episodes: int | None
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    info = _read_json_object(root / "meta" / "info.json", label="dataset info")
    _require_original_fullcycle_v4_7d_source(info)
    total_episodes = info.get("total_episodes")
    if type(total_episodes) is not int or total_episodes <= 0:
        raise VisualLatchAuditInputError("info.total_episodes must be positive")
    if expected_episodes is not None and total_episodes != expected_episodes:
        raise VisualLatchAuditInputError(
            f"expected {expected_episodes} episodes, found {total_episodes}"
        )
    if info.get("fps") != DEFAULT_FPS:
        raise VisualLatchAuditInputError(
            f"fullcycle-v4 audit requires {DEFAULT_FPS} Hz, got {info.get('fps')!r}"
        )
    feature = info.get("features", {}).get(WRIST_VIDEO_KEY, {})
    if feature.get("shape") != [480, 640, 3] or feature.get("dtype") != "video":
        raise VisualLatchAuditInputError("wrist feature must be 640x480 RGB video")

    episode_rows = _read_parquet_rows(
        sorted((root / "meta" / "episodes").glob("**/*.parquet")),
        label="episode metadata",
    )
    episodes: dict[int, dict[str, Any]] = {}
    for row in episode_rows:
        try:
            index = int(row["episode_index"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VisualLatchAuditInputError("invalid episode metadata index") from exc
        if index in episodes:
            raise VisualLatchAuditInputError(f"duplicate episode metadata {index}")
        episodes[index] = row

    data_rows = _read_parquet_rows(
        sorted((root / "data").glob("**/*.parquet")),
        columns=("episode_index", "frame_index", "action", "observation.state"),
        label="data",
    )
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in data_rows:
        try:
            index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VisualLatchAuditInputError("invalid data row indexes") from exc
        _finite_vector(row.get("action"), label=f"episode {index} frame {frame_index} action")
        _finite_vector(
            row.get("observation.state"),
            label=f"episode {index} frame {frame_index} state",
        )
        frames[index].append(row)

    expected_indexes = set(range(total_episodes))
    if set(episodes) != expected_indexes or set(frames) != expected_indexes:
        raise VisualLatchAuditInputError("episode indexes are not contiguous from zero")
    for index in range(total_episodes):
        frames[index].sort(key=lambda row: int(row["frame_index"]))
        indexes = [int(row["frame_index"]) for row in frames[index]]
        if indexes != list(range(len(indexes))):
            raise VisualLatchAuditInputError(
                f"episode {index} frame indexes are not contiguous from zero"
            )
        if episodes[index].get("length") != len(indexes):
            raise VisualLatchAuditInputError(
                f"episode {index} metadata/data length mismatch"
            )
    return info, episodes, dict(frames)


def _wrist_video_path(
    root: Path,
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> Path:
    template = info.get("video_path")
    if not isinstance(template, str):
        raise VisualLatchAuditInputError("info.video_path is missing")
    try:
        relative = template.format(
            video_key=WRIST_VIDEO_KEY,
            chunk_index=int(episode[f"videos/{WRIST_VIDEO_KEY}/chunk_index"]),
            file_index=int(episode[f"videos/{WRIST_VIDEO_KEY}/file_index"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise VisualLatchAuditInputError("episode lacks wrist video mapping") from exc
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise VisualLatchAuditInputError(
            f"wrist video must be a regular, non-symlink file: {path}"
        )
    return path


def _decode_rgb_frames(path: Path) -> Iterator[Any]:
    try:
        import av
    except ImportError as exc:
        raise VisualLatchAuditInputError("PyAV is required for wrist replay") from exc
    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if len(streams) != 1:
                raise VisualLatchAuditInputError(
                    f"expected one video stream in {path}, found {len(streams)}"
                )
            for frame in container.decode(streams[0]):
                yield frame.to_ndarray(format="rgb24")
    except VisualLatchAuditInputError:
        raise
    except Exception as exc:
        raise VisualLatchAuditInputError(f"cannot decode wrist video {path}: {exc}") from exc


class _RedContourDiagnosticExtractor:
    """Mirror the detector's red-mask preprocessing for offline diagnostics."""

    def __init__(self, config: VisualPhaseLatchConfig):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise VisualLatchAuditInputError(
                "OpenCV and NumPy are required for contour diagnostics"
            ) from exc
        self.cv2 = cv2
        self.np = np
        self.config = config
        self.kernel = np.ones(
            (config.morphology_kernel_px, config.morphology_kernel_px),
            dtype=np.uint8,
        )
        self.ranges = [
            (
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
            for lower, upper in config.red_hsv_ranges
        ]

    def __call__(self, rgb: Any) -> dict[str, Any]:
        cv2 = self.cv2
        np = self.np
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        for left, top, right, bottom in self.config.excluded_rois_px:
            mask[top:bottom, left:right] = 0
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        image_area = float(mask.size)
        candidates: list[dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour)) / image_area
            if area < self.config.red_ambiguity_area_fraction_min:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            candidates.append(
                {
                    "area_fraction": area,
                    "centroid_norm": [
                        float(
                            moments["m10"]
                            / moments["m00"]
                            / self.config.expected_width
                        ),
                        float(
                            moments["m01"]
                            / moments["m00"]
                            / self.config.expected_height
                        ),
                    ],
                    "bbox_norm": [
                        x / self.config.expected_width,
                        y / self.config.expected_height,
                        width / self.config.expected_width,
                        height / self.config.expected_height,
                    ],
                }
            )
        candidates.sort(key=lambda item: item["area_fraction"], reverse=True)
        return {
            "positive_contour_count": len(contours),
            "ambiguity_candidates": candidates,
        }


def _detect_episode(
    path: Path,
    *,
    expected_frames: int,
    config: VisualPhaseLatchConfig,
) -> dict[str, Any]:
    detector = D405VisualDetector(config)
    contour_extractor = _RedContourDiagnosticExtractor(config)
    close_flags: list[bool] = []
    held_flags: list[bool] = []
    blue_flags: list[bool] = []
    close_reasons: list[tuple[str, ...]] = []
    held_reasons: list[tuple[str, ...]] = []
    blue_reasons: list[tuple[str, ...]] = []
    red_contours: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    evidences: list[Any] = []
    period_ns = round(1_000_000_000 / DEFAULT_FPS)
    decoded_frames = 0
    for frame_index, rgb in enumerate(_decode_rgb_frames(path)):
        decoded_frames += 1
        if frame_index >= expected_frames:
            continue
        timestamp_ns = (frame_index + 1) * period_ns
        evidence = detector.detect(
            rgb,
            frame_id=frame_index,
            captured_at_ns=timestamp_ns,
            now_ns=timestamp_ns,
        )
        statuses[evidence.frame_status.value] += 1
        evidences.append(evidence)
        close_flags.append(bool(evidence.red.accepted))
        held_flags.append(bool(evidence.red.held_accepted))
        blue_flags.append(bool(evidence.blue.accepted))
        close_reasons.append(tuple(evidence.red.rejection_reasons))
        held_reasons.append(tuple(evidence.red.held_rejection_reasons))
        blue_reasons.append(tuple(evidence.blue.rejection_reasons))
        red_contours.append(contour_extractor(rgb))
    if decoded_frames != expected_frames:
        raise VisualLatchAuditInputError(
            f"video {path} decoded {decoded_frames} frames, expected {expected_frames}"
        )
    if statuses != Counter({FrameStatus.FRESH_UNIQUE.value: expected_frames}):
        raise VisualLatchAuditInputError(
            f"video {path} did not replay as unique fresh frames: {dict(statuses)}"
        )
    return {
        "close_flags": close_flags,
        "held_flags": held_flags,
        "blue_flags": blue_flags,
        "close_reasons": close_reasons,
        "held_reasons": held_reasons,
        "blue_reasons": blue_reasons,
        "red_contours": red_contours,
        "frame_status_counts": dict(statuses),
        "evidences": evidences,
    }


def _number_stats(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _merge_reason_counts(
    target: Counter[str], source: Mapping[str, Any]
) -> None:
    for reason, count in source.items():
        target[str(reason)] += int(count)


def _closed_interval(value: float, interval: Sequence[float]) -> bool:
    return float(interval[0]) <= value <= float(interval[1])


def _tip_probe_frame_from_evidence(
    evidence: Any, *, trace_index: int
) -> TipProbeFrame:
    centroid = evidence.red.centroid_norm
    return TipProbeFrame(
        trace_index=trace_index,
        frame_id=int(evidence.frame_id),
        fresh_unique=evidence.frame_status is FrameStatus.FRESH_UNIQUE,
        candidate_count=int(evidence.red.candidate_count),
        centroid_norm=(
            None
            if centroid is None
            else (float(centroid[0]), float(centroid[1]))
        ),
        area_fraction=(
            None if centroid is None else float(evidence.red.contour_area_fraction)
        ),
    )


def _tip_probe_single_red(frame: TipProbeFrame) -> bool:
    return bool(
        frame.fresh_unique
        and frame.candidate_count == 1
        and frame.centroid_norm is not None
        and frame.area_fraction is not None
    )


def _tip_probe_coarse_gate(frame: TipProbeFrame, config: TipServoConfig) -> bool:
    if not _tip_probe_single_red(frame):
        return False
    assert frame.centroid_norm is not None
    assert frame.area_fraction is not None
    return bool(
        _closed_interval(frame.centroid_norm[0], config.coarse_x_norm)
        and _closed_interval(frame.centroid_norm[1], config.coarse_y_norm)
        and frame.area_fraction >= config.coarse_area_fraction[0]
        and frame.area_fraction < config.coarse_area_fraction[1]
    )


def _tip_probe_tracking_gate(frame: TipProbeFrame, config: TipServoConfig) -> bool:
    if not _tip_probe_single_red(frame):
        return False
    assert frame.centroid_norm is not None
    assert frame.area_fraction is not None
    return bool(
        _closed_interval(frame.centroid_norm[0], config.tracking_x_norm)
        and _closed_interval(frame.centroid_norm[1], config.tracking_y_norm)
        and _closed_interval(frame.area_fraction, config.tracking_area_fraction)
    )


def _tip_probe_close_ready_gate(
    frame: TipProbeFrame, config: TipServoConfig
) -> bool:
    if not _tip_probe_single_red(frame):
        return False
    assert frame.centroid_norm is not None
    assert frame.area_fraction is not None
    return bool(
        _closed_interval(frame.centroid_norm[0], config.close_ready_x_norm)
        and _closed_interval(frame.centroid_norm[1], config.close_ready_y_norm)
        and _closed_interval(
            frame.area_fraction, config.close_ready_area_fraction
        )
    )


def _tip_probe_progress_gate(frame: TipProbeFrame, config: TipServoConfig) -> bool:
    return bool(
        _tip_probe_tracking_gate(frame, config)
        and not _tip_probe_close_ready_gate(frame, config)
        and frame.area_fraction is not None
        and frame.area_fraction >= config.probe_success_area_fraction
    )


def _first_tip_probe_confirmation(
    frames: Sequence[TipProbeFrame],
    predicate: Any,
    *,
    required_frames: int,
    start: int = 0,
    stop: int | None = None,
) -> int | None:
    if required_frames < 1:
        raise ValueError("tip-probe confirmation length must be positive")
    stop = len(frames) if stop is None else min(len(frames), max(start, stop))
    streak = 0
    for index in range(max(0, start), stop):
        streak = streak + 1 if bool(predicate(frames[index])) else 0
        if streak >= required_frames:
            return index
    return None


def _tip_probe_fk_pose(
    fk_model: Any, state: Sequence[float], *, label: str
) -> tuple[Any, Any]:
    vector = _finite_vector(state, label=label)
    try:
        position, rotation = fk_model.end_effector_pose(
            vector[:6], tool_offset=TCP_TOOL_OFFSET_M
        )
    except ValueError as exc:
        raise VisualLatchAuditInputError(f"tip-probe FK failed for {label}: {exc}") from exc
    try:
        import numpy as np

        position_array = np.asarray(position, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VisualLatchAuditInputError(
            f"tip-probe FK returned non-numeric {label}"
        ) from exc
    if (
        position_array.shape != (3,)
        or rotation_array.shape != (3, 3)
        or not np.isfinite(position_array).all()
        or not np.isfinite(rotation_array).all()
    ):
        raise VisualLatchAuditInputError(
            f"tip-probe FK returned invalid pose for {label}"
        )
    return position_array, rotation_array


def _tip_probe_area_drawdown(
    frames: Sequence[TipProbeFrame],
    *,
    start: int,
    stop: int,
    config: TipServoConfig,
) -> dict[str, int | float]:
    peak: float | None = None
    max_drawdown = 0.0
    streak = 0
    max_streak = 0
    for frame in frames[start : stop + 1]:
        if frame.area_fraction is None:
            streak = 0
            continue
        area = float(frame.area_fraction)
        if peak is None:
            peak = area
        else:
            drawdown = max(0.0, peak - area)
            max_drawdown = max(max_drawdown, drawdown)
            streak = streak + 1 if drawdown > config.area_drop_fault_fraction else 0
            max_streak = max(max_streak, streak)
            peak = max(peak, area)
    return {
        "max_running_peak_drawdown_fraction": max_drawdown,
        "max_significant_drop_streak_frames": max_streak,
    }


def _tip_probe_demo_trace_envelope(
    rows: Sequence[Mapping[str, Any]],
    evidences: Sequence[Any],
    labels: PhaseLabels,
    *,
    config: TipServoConfig,
    fk_model: Any,
) -> dict[str, Any]:
    """Measure a recorded demonstration against the no-close probe envelope.

    The measured demonstration supplies a successful geometric path. It is not
    a simulation of the arm targets that the tip servo would publish, and the
    result must never be described as closed-loop convergence validation.
    """

    if config.allow_close:
        raise VisualLatchAuditInputError(
            "tip-probe trace audit requires tip_servo.allow_close=false"
        )
    if len(rows) != len(evidences):
        raise VisualLatchAuditInputError(
            "tip-probe rows and visual evidences have unequal lengths"
        )
    frames = [
        _tip_probe_frame_from_evidence(evidence, trace_index=index)
        for index, evidence in enumerate(evidences)
    ]
    stop = min(len(frames), labels.pickup_close_frame + 1)
    takeover = _first_tip_probe_confirmation(
        frames,
        lambda frame: _tip_probe_coarse_gate(frame, config),
        required_frames=config.takeover_confirm_frames,
        stop=stop,
    )
    progress = None
    close_ready = None
    if takeover is not None:
        # The planner opens after takeover. The pseudo-label is the earliest
        # recorded point at which a no-close progress envelope is meaningful;
        # target settling and command response remain deliberately unmodelled.
        progress_start = max(takeover + 1, labels.pregrasp_open_frame)
        progress = _first_tip_probe_confirmation(
            frames,
            lambda frame: _tip_probe_progress_gate(frame, config),
            required_frames=config.probe_success_confirm_frames,
            start=progress_start,
            stop=stop,
        )
        close_ready = _first_tip_probe_confirmation(
            frames,
            lambda frame: _tip_probe_close_ready_gate(frame, config),
            required_frames=config.close_ready_confirm_frames,
            start=progress_start,
            stop=stop,
        )
    candidates = [
        (frame, event)
        for frame, event in (
            (progress, "probe_progress_complete"),
            (close_ready, "visual_would_close"),
        )
        if frame is not None
    ]
    outcome_frame: int | None
    outcome: str | None
    if candidates:
        outcome_frame, outcome = min(candidates, key=lambda item: item[0])
    else:
        outcome_frame, outcome = None, None

    required_base_descent_m: float | None = None
    recorded_path_norm_m: float | None = None
    observed_delta_q_rad: list[float] | None = None
    required_steps: int | None = None
    tracking_continuous = False
    drawdown = {
        "max_running_peak_drawdown_fraction": 0.0,
        "max_significant_drop_streak_frames": 0,
    }
    if takeover is not None and outcome_frame is not None:
        takeover_state = _finite_vector(
            rows[takeover].get("observation.state"),
            label=f"tip-probe state[{takeover}]",
        )
        outcome_state = _finite_vector(
            rows[outcome_frame].get("observation.state"),
            label=f"tip-probe state[{outcome_frame}]",
        )
        takeover_position, _takeover_rotation = _tip_probe_fk_pose(
            fk_model, takeover_state, label=f"state[{takeover}]"
        )
        outcome_position, _outcome_rotation = _tip_probe_fk_pose(
            fk_model, outcome_state, label=f"state[{outcome_frame}]"
        )
        displacement = outcome_position - takeover_position
        required_base_descent_m = max(0.0, float(-displacement[2]))
        recorded_path_norm_m = float(
            math.sqrt(sum(float(value) ** 2 for value in displacement))
        )
        observed_delta_q_rad = [
            float(after - before)
            for before, after in zip(
                takeover_state[:6], outcome_state[:6], strict=True
            )
        ]
        required_steps = int(
            math.ceil(
                max(0.0, required_base_descent_m - 1e-12)
                / config.descent_step_m
            )
        )
        tracking_continuous = all(
            _tip_probe_tracking_gate(frame, config)
            for frame in frames[takeover : outcome_frame + 1]
        )
        drawdown = _tip_probe_area_drawdown(
            frames,
            start=takeover,
            stop=outcome_frame,
            config=config,
        )

    checks = {
        "takeover_confirmed": takeover is not None,
        "probe_outcome_confirmed": outcome_frame is not None,
        "within_recorded_base_z_budget": (
            required_base_descent_m is not None
            and required_base_descent_m <= config.max_total_descent_m + 1e-12
        ),
        "tracking_corridor_continuous": tracking_continuous,
        "area_drop_fault_free": (
            int(drawdown["max_significant_drop_streak_frames"])
            < config.area_drop_fault_frames
        ),
        "allow_close_false": config.allow_close is False,
        "close_not_authorized": True,
        "hardware_completion_not_claimed": True,
    }
    return {
        "semantics": "recorded_success_path_geometric_envelope_only",
        "allow_close": False,
        "close_authorized": False,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "settling_and_command_response_modeled": False,
        "takeover_confirmation_frame": takeover,
        "takeover_offset_from_pregrasp_open_frames": (
            None if takeover is None else takeover - labels.pregrasp_open_frame
        ),
        "probe_progress_confirmation_frame": progress,
        "visual_would_close_confirmation_frame": close_ready,
        "outcome": outcome,
        "outcome_frame": outcome_frame,
        "outcome_offset_from_pickup_close_frames": (
            None
            if outcome_frame is None
            else outcome_frame - labels.pickup_close_frame
        ),
        "required_recorded_base_z_descent_m": required_base_descent_m,
        "recorded_tcp_displacement_norm_m": recorded_path_norm_m,
        "required_descent_steps": required_steps,
        "observed_delta_q_rad": observed_delta_q_rad,
        **drawdown,
        "checks": checks,
        "trace_envelope_pass": all(checks.values()),
        "limitation": (
            "Frames after takeover are the human demonstration, not observations "
            "caused by counterfactual tip-servo commands. This proves only a "
            "recorded geometric/budget envelope and never hardware convergence."
        ),
    }


def _open_only_axis_authority() -> dict[str, Any]:
    """Describe OPEN_ONLY authority without making a whole-pipeline motion claim."""

    return {
        "scope": "axis_partitioned_open_only",
        "audit_schema_version": 6,
        "legacy_schema5_authorized": False,
        "policy_role": "arm6_seeking",
        "policy_numeric_observation_axes": [
            f"joint_{index}.pos" for index in range(1, 7)
        ],
        "policy_numeric_action_axes": [
            f"joint_{index}.pos" for index in range(1, 7)
        ],
        "policy_numeric_observation_dimension": 6,
        "policy_numeric_action_dimension": 6,
        "policy_numeric_jaw_input": False,
        "policy_numeric_jaw_output": False,
        "jaw_may_remain_visible_in_video": True,
        "seeking_jaw_candidate_source": "startup_accepted_reference",
        "pre_takeover_policy_arm_motion_authorized": True,
        "post_takeover_arm_motion_authorized": False,
        "whole_pipeline_arm_motion_prohibited": False,
        "whole_pipeline_tcp_descent_prohibited": False,
        "policy_jaw_authorized": False,
        "main_latch_jaw_authorized": False,
        "tip_capture_hold_authorized": True,
        "tip_open_after_takeover_and_settle_authorized": True,
        "tip_explicit_descent_authorized": False,
        "jaw_close_authorized": False,
        "seeking_jaw_owner": "tip_capture_hold",
        "post_takeover_arm_owner": "hold",
        "allow_close": False,
    }


def _open_only_runtime_authority_replay(
    *, config: TipServoConfig, capture_target: float
) -> dict[str, Any]:
    """Exercise the deployed OPEN_ONLY composer without cameras or hardware.

    This intentionally calls ``VisualLatchRuntime.compose_cycle`` instead of
    restating its ownership rules.  A deterministic detector and a minimal FK
    robot provide only immutable inputs; no follower method that publishes a
    command is present.
    """

    if config != _open_only_tip_servo_config():
        raise VisualLatchAuditInputError(
            "runtime authority replay requires the exact schema-v6 profile"
        )
    capture_gate = config.coarse_capture_jaw_rad
    assert capture_gate is not None
    if not _closed_interval(capture_target, capture_gate):
        raise VisualLatchAuditInputError(
            "runtime authority replay capture jaw is outside the reviewed gate"
        )

    try:
        from scripts import run_firefly_y6 as runtime_module
    except Exception as exc:
        raise VisualLatchAuditInputError(
            "cannot import the deployed OPEN_ONLY runtime composer"
        ) from exc

    replay_action_features = tuple(
        [f"joint_{index}.pos" for index in range(1, 7)]
        + ["jaw_joint1.pos"]
    )
    accepted_arm = (0.034, 0.0, 0.0, 0.0, 0.0, 0.0)
    measured_at_capture = (0.0,) * 6
    accepted = (*accepted_arm, float(capture_target))
    adversarial_arm_policy6 = (0.25, -0.25, 0.20, -0.20, 0.15, -0.15)
    visual_config = VisualPhaseLatchConfig(enabled=True, tip_servo=config)

    def arm6_infer(_observation: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
        return (adversarial_arm_policy6,) * 10

    arm6_infer.policy_runtime_contract = runtime_module.PolicyRuntimeContract(
        policy_type="smolvla",
        runtime_role="arm6_seeking",
        observation_state_dimension=6,
        policy_action_dimension=6,
    )

    class NumericJawReadTrap(dict[str, float]):
        def __getitem__(self, key: str) -> float:
            if key == "jaw_joint1.pos":
                raise VisualLatchAuditInputError(
                    "arm6 observation extractor attempted to read numeric jaw state"
                )
            return super().__getitem__(key)

    class ReplayDetector:
        evidence: VisualEvidence | None = None

        def detect(self, *_args: Any, **_kwargs: Any) -> VisualEvidence:
            if self.evidence is None:
                raise VisualLatchAuditInputError(
                    "runtime authority replay detector has no evidence"
                )
            return self.evidence

    class ReplayRobot:
        action_features = replay_action_features

        def __init__(self) -> None:
            self.sequence = 0
            self.captured_at_s = 0.0
            arm_limits = SimpleNamespace(
                pos_min=(-2.0,) * 6,
                pos_max=(2.0,) * 6,
                vel_max=(3.0,) * 6,
            )
            self.controller_limits = (arm_limits, SimpleNamespace())
            self.config = SimpleNamespace(
                command_hz=30.0,
                resolved_arm_step_limits_rad=lambda: (0.0435,) * 6,
            )

        def get_camera_frame_metadata(self) -> dict[str, Any]:
            return {
                "wrist": SimpleNamespace(
                    sequence=self.sequence,
                    captured_at_s=self.captured_at_s,
                )
            }

        @staticmethod
        def get_tcp_pose(arm_q: Sequence[float]) -> tuple[Any, Any]:
            q = tuple(float(value) for value in arm_q)
            return (
                (q[0], q[2], 0.10 + q[1]),
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )

    clock = {"now_ns": 10_000_000_000}
    original_time_module = runtime_module.time
    runtime_module.time = SimpleNamespace(
        perf_counter_ns=lambda: clock["now_ns"]
    )

    def new_context() -> dict[str, Any]:
        runtime = runtime_module.VisualLatchRuntime(
            visual_config,
            home_center_rad=accepted,
            home_tolerance_rad=(0.10,) * 7,
        )
        detector = ReplayDetector()
        runtime.detector = detector
        # Exercise the legacy/main jaw writer on every SEEKING/OPENING cycle.
        # Its 0.41-rad proposal must remain telemetry-only under OPEN_ONLY.
        runtime.latch.pregrasp_open_seen = True
        robot = ReplayRobot()
        arm6_runtime = runtime_module.bind_arm6_seeking_runtime(
            arm6_infer,
            accepted,
        )
        candidate_chunk, adapter_telemetry = (
            runtime_module.prepare_policy_action_chunk(
                SimpleNamespace(enable_action_slew_rate_adapter=False),
                robot,
                arm6_infer({}),
                accepted,
                arm6_runtime=arm6_runtime,
            )
        )
        return {
            "runtime": runtime,
            "detector": detector,
            "robot": robot,
            "arm6_runtime": arm6_runtime,
            "candidate_chunk": candidate_chunk,
            "adapter_telemetry": adapter_telemetry,
            "frame_id": 0,
        }

    def compose(
        context: Mapping[str, Any],
        *,
        valid_red: bool,
        measured_arm: Sequence[float],
        measured_jaw: float,
        reference: Sequence[float],
        action_source: Any,
        elapsed_ns: int = 33_000_000,
        policy_action: Sequence[float] | None = None,
    ) -> Any:
        clock["now_ns"] += elapsed_ns
        context["frame_id"] += 1
        frame_id = int(context["frame_id"])
        red = (
            BlobEvidence(
                accepted=False,
                held_accepted=False,
                contour_area_fraction=0.018,
                centroid_norm=(0.52, 0.45),
                candidate_count=1,
            )
            if valid_red
            else BlobEvidence()
        )
        evidence = VisualEvidence(
            frame_id=frame_id,
            captured_at_ns=clock["now_ns"] - 10_000_000,
            evaluated_at_ns=clock["now_ns"],
            frame_age_ms=10.0,
            frame_status=FrameStatus.FRESH_UNIQUE,
            red=red,
            blue=BlobEvidence(),
        )
        detector = context["detector"]
        detector.evidence = evidence
        robot = context["robot"]
        robot.sequence = frame_id
        robot.captured_at_s = evidence.captured_at_ns / 1_000_000_000.0
        observation = {
            name: float(value)
            for name, value in zip(
                replay_action_features,
                (*tuple(float(value) for value in measured_arm), float(measured_jaw)),
                strict=True,
            )
        }
        observation["wrist"] = None
        if policy_action is None:
            policy_action = (
                reference
                if action_source
                in {
                    runtime_module.VisualActionSource.INFERENCE_HOLD,
                    runtime_module.VisualActionSource.TIP_SERVO,
                }
                else context["candidate_chunk"][0]
            )
        return context["runtime"].compose_cycle(
            robot,
            observation,
            policy_action,
            reference,
            action_source=action_source,
            arm6_runtime=context["arm6_runtime"],
        )

    try:
        context = new_context()
        runtime = context["runtime"]
        extracted_observation6 = runtime_module.extract_arm6_observation(
            NumericJawReadTrap({
                name: float(index)
                for index, name in enumerate(replay_action_features)
            }),
            role=runtime_module.ArmOnlyPolicyRole.ARM6_SEEKING,
        )
        seven_dimensional_policy_output_rejected = False
        try:
            runtime_module.prepare_policy_action_chunk(
                SimpleNamespace(enable_action_slew_rate_adapter=False),
                context["robot"],
                tuple((*adversarial_arm_policy6, 0.0) for _ in range(10)),
                accepted,
                arm6_runtime=context["arm6_runtime"],
            )
        except ValueError:
            seven_dimensional_policy_output_rejected = True

        synthesized_jaw_tamper_rejected = False
        tamper_context = new_context()
        candidate7 = tamper_context["candidate_chunk"][0]
        try:
            compose(
                tamper_context,
                valid_red=False,
                measured_arm=measured_at_capture,
                measured_jaw=capture_target,
                reference=accepted,
                action_source=runtime_module.VisualActionSource.FRESH_POLICY,
                policy_action=(*candidate7[:6], float(capture_target) - 0.001),
            )
        except ValueError:
            synthesized_jaw_tamper_rejected = True

        compose(
            context,
            valid_red=False,
            measured_arm=measured_at_capture,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.FRESH_POLICY,
        )
        sources = (
            (runtime_module.VisualActionSource.FRESH_POLICY,) * 2
            + (runtime_module.VisualActionSource.INFERENCE_HOLD,) * 5
            + (runtime_module.VisualActionSource.FRESH_POLICY,)
        )
        boundary_cycles = [
            compose(
                context,
                valid_red=True,
                measured_arm=measured_at_capture,
                measured_jaw=capture_target,
                reference=accepted,
                action_source=source,
            )
            for source in sources
        ]
        takeover = boundary_cycles[-1]
        runtime.assert_open_only_publish_contract(
            takeover.action,
            accepted,
            takeover.telemetry,
        )
        unsettled = compose(
            context,
            valid_red=True,
            measured_arm=measured_at_capture,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.TIP_SERVO,
        )
        runtime.assert_open_only_publish_contract(
            unsettled.action,
            accepted,
            unsettled.telemetry,
        )
        catch_up = compose(
            context,
            valid_red=True,
            measured_arm=accepted_arm,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.TIP_SERVO,
        )
        runtime.assert_open_only_publish_contract(
            catch_up.action,
            accepted,
            catch_up.telemetry,
        )
        first_open = compose(
            context,
            valid_red=True,
            measured_arm=accepted_arm,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.TIP_SERVO,
        )
        runtime.assert_open_only_publish_contract(
            first_open.action,
            accepted,
            first_open.telemetry,
        )
        tampered_arm_rejected = False
        try:
            runtime.assert_open_only_publish_contract(
                (
                    float(first_open.action[0]) + 0.001,
                    *tuple(float(value) for value in first_open.action[1:]),
                ),
                accepted,
                first_open.telemetry,
            )
        except runtime_module.TipServoError:
            tampered_arm_rejected = True
        fault_reference = tuple(float(value) for value in first_open.action)
        active_source_fault = compose(
            context,
            valid_red=True,
            measured_arm=accepted_arm,
            measured_jaw=fault_reference[6],
            reference=fault_reference,
            action_source=runtime_module.VisualActionSource.FRESH_POLICY,
        )
        runtime.assert_open_only_publish_contract(
            active_source_fault.action,
            fault_reference,
            active_source_fault.telemetry,
        )

        timeout_context = new_context()
        compose(
            timeout_context,
            valid_red=False,
            measured_arm=accepted_arm,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.FRESH_POLICY,
        )
        seeking_timeout = compose(
            timeout_context,
            valid_red=False,
            measured_arm=accepted_arm,
            measured_jaw=capture_target,
            reference=accepted,
            action_source=runtime_module.VisualActionSource.FRESH_POLICY,
            elapsed_ns=int(config.seeking_timeout_ms * 1_000_000.0),
        )
        timeout_context["runtime"].assert_open_only_publish_contract(
            seeking_timeout.action,
            accepted,
            seeking_timeout.telemetry,
        )
    except Exception as exc:
        raise VisualLatchAuditInputError(
            f"OPEN_ONLY runtime authority replay failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        runtime_module.time = original_time_module

    active_cycles = [takeover, unsettled, catch_up, first_open, active_source_fault]
    active_tip = [cycle.telemetry["tip_servo"] for cycle in active_cycles]
    boundary_events = [
        cycle.telemetry["tip_servo"]["event"] for cycle in boundary_cycles
    ]
    post_takeover_arm_exact = all(
        all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(cycle.action[:6], accepted_arm, strict=True)
        )
        for cycle in active_cycles
    )
    post_takeover_owner_hold = all(
        cycle.arm_owner == "hold" for cycle in active_cycles
    )
    zero_descent = all(
        math.isclose(
            float(tip["commanded_descent_m"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for tip in active_tip
    )
    no_ik = all(
        all(
            key not in tip
            for key in (
                "ik_condition_number",
                "ik_position_residual_m",
                "ik_orientation_residual_rad",
                "ik_reason",
            )
        )
        for tip in active_tip
    )
    no_close = all(tip["probe_close_allowed"] is False for tip in active_tip)
    takeover_capture_hold = math.isclose(
        float(takeover.action[6]),
        capture_target,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    settle_before_open = bool(
        unsettled.telemetry["tip_servo"]["event"] == "takeover_settle_hold"
        and catch_up.telemetry["tip_servo"]["event"] == "takeover_settle_hold"
        and math.isclose(
            float(unsettled.action[6]),
            capture_target,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(catch_up.action[6]),
            capture_target,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and float(first_open.action[6]) > capture_target
    )
    adapter_telemetry = context["adapter_telemetry"]
    expected_arm6_authority = {
        "policy_role": "arm6_seeking",
        "observation_state_dimension": 6,
        "raw_policy_action_dimension": 6,
        "candidate_action_dimension": 7,
        "jaw_candidate_source": "startup_accepted_reference",
        "capture_jaw_target_rad": float(capture_target),
    }
    arm6_adapter_contract = bool(
        tuple(extracted_observation6) == tuple(float(index) for index in range(6))
        and adapter_telemetry.get("arm6_authority") == expected_arm6_authority
        and all(len(row) == 6 for row in adapter_telemetry["raw_actions"])
        and all(
            len(row) == 7
            and math.isclose(
                float(row[6]), capture_target, rel_tol=0.0, abs_tol=1e-12
            )
            for field in ("assisted_actions", "adapted_actions")
            for row in adapter_telemetry[field]
        )
    )
    all_replay_cycles = [
        *boundary_cycles,
        unsettled,
        catch_up,
        first_open,
        active_source_fault,
        seeking_timeout,
    ]
    raw_policy_jaw_telemetry_absent = all(
        "raw_policy_jaw_target_rad" not in cycle.telemetry["tip_servo"]
        for cycle in all_replay_cycles
    )
    policy_role_telemetry_exact = all(
        cycle.telemetry["tip_servo"].get("policy_role") == "arm6_seeking"
        for cycle in all_replay_cycles
    )
    jaw_candidate_source_telemetry_exact = bool(
        all(
            cycle.telemetry["tip_servo"].get("jaw_candidate_source")
            == "startup_accepted_reference"
            for cycle in boundary_cycles
        )
        and all(
            cycle.telemetry["tip_servo"].get("jaw_candidate_source")
            == "tip_servo_exclusive_reference"
            for cycle in (unsettled, catch_up, first_open)
        )
        and active_source_fault.telemetry["tip_servo"].get(
            "jaw_candidate_source"
        )
        == "startup_accepted_reference"
        and seeking_timeout.telemetry["tip_servo"].get("jaw_candidate_source")
        == "startup_accepted_reference"
    )
    main_latch_jaw_suppressed = all(
        math.isclose(
            float(cycle.telemetry["tip_servo"]["main_latch_jaw_target_rad_ignored"]),
            visual_config.pregrasp_open_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for cycle in (takeover, unsettled, catch_up, first_open)
    )
    fault_hold_non_decreasing = bool(
        active_source_fault.telemetry["tip_servo"]["fault_reason"]
        == "tip_servo_action_source_violation"
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                active_source_fault.action[:6],
                accepted_arm,
                strict=True,
            )
        )
        and float(active_source_fault.action[6]) >= fault_reference[6] - 1e-12
    )
    timeout_tip = seeking_timeout.telemetry["tip_servo"]
    seeking_timeout_holds_capture = bool(
        timeout_tip["fault_reason"] == "seeking_timeout"
        and seeking_timeout.arm_owner == "hold"
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                seeking_timeout.action,
                accepted,
                strict=True,
            )
        )
    )
    takeover_chunk_reset = bool(
        takeover.discard_policy_chunk is True
        and takeover.smoother_reset_required is True
        and takeover.replan_reason == "tip_servo_takeover"
        and takeover.arm_owner == "hold"
        and runtime.replan_epoch >= 1
    )
    boundary_sequence_exact = boundary_events == (
        ["coarse_candidate"] * 5
        + ["coarse_ready_waiting_authorization"] * 2
        + ["coarse_takeover"]
    )
    passed = bool(
        boundary_sequence_exact
        and takeover_chunk_reset
        and takeover_capture_hold
        and settle_before_open
        and post_takeover_arm_exact
        and post_takeover_owner_hold
        and tampered_arm_rejected
        and arm6_adapter_contract
        and raw_policy_jaw_telemetry_absent
        and policy_role_telemetry_exact
        and jaw_candidate_source_telemetry_exact
        and seven_dimensional_policy_output_rejected
        and synthesized_jaw_tamper_rejected
        and main_latch_jaw_suppressed
        and fault_hold_non_decreasing
        and seeking_timeout_holds_capture
        and zero_descent
        and no_ik
        and no_close
    )
    return {
        "implementation": (
            "scripts.run_firefly_y6.VisualLatchRuntime.compose_cycle"
        ),
        "software_only": True,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "boundary_events": boundary_events,
        "boundary_sequence_exact": boundary_sequence_exact,
        "takeover_discard_policy_chunk": takeover.discard_policy_chunk,
        "takeover_smoother_reset_required": takeover.smoother_reset_required,
        "takeover_replan_reason": takeover.replan_reason,
        "takeover_arm_owner": takeover.arm_owner,
        "post_takeover_arm_owner_values": sorted(
            {str(cycle.arm_owner) for cycle in active_cycles}
        ),
        "post_takeover_arm_exact_accepted_reference": post_takeover_arm_exact,
        "post_takeover_arm_owner_hold": post_takeover_owner_hold,
        "tampered_final_arm_rejected": tampered_arm_rejected,
        "takeover_boundary_still_capture_hold": takeover_capture_hold,
        "first_open_requires_takeover_and_settle": settle_before_open,
        "first_open_target_rad": float(first_open.action[6]),
        "policy_role": "arm6_seeking",
        "numeric_policy_observation_dimension": len(extracted_observation6),
        "raw_policy_action_dimension": len(adversarial_arm_policy6),
        "numeric_policy_jaw_input": False,
        "numeric_policy_jaw_output": False,
        "arm6_authority": expected_arm6_authority,
        "arm6_adapter_contract_pass": arm6_adapter_contract,
        "raw_policy_jaw_telemetry_absent": raw_policy_jaw_telemetry_absent,
        "policy_role_telemetry_exact": policy_role_telemetry_exact,
        "jaw_candidate_source_telemetry_exact": (
            jaw_candidate_source_telemetry_exact
        ),
        "seven_dimensional_policy_output_rejected": (
            seven_dimensional_policy_output_rejected
        ),
        "synthesized_jaw_tamper_rejected": synthesized_jaw_tamper_rejected,
        "main_latch_jaw_suppressed": main_latch_jaw_suppressed,
        "fault_hold_non_decreasing": fault_hold_non_decreasing,
        "fault_published_jaw_target_rad": float(active_source_fault.action[6]),
        "seeking_timeout_holds_capture": seeking_timeout_holds_capture,
        "zero_explicit_descent": zero_descent,
        "ik_never_invoked": no_ik,
        "close_never_authorized": no_close,
        "runtime_authority_replay_pass": passed,
        "limitation": (
            "Pure software composition replay; it verifies command ownership and "
            "state transitions, not controller or hardware response. Runtime "
            "arm_owner is required to remain hold and every post-takeover arm "
            "target must equal the accepted reference exactly."
        ),
    }


def _tip_probe_open_only_demo_envelope(
    rows: Sequence[Mapping[str, Any]],
    evidences: Sequence[Any],
    labels: PhaseLabels,
    *,
    config: TipServoConfig,
) -> dict[str, Any]:
    """Record visual coverage only; command authority is audited separately."""

    if config != _open_only_tip_servo_config():
        raise VisualLatchAuditInputError(
            "open-only demonstration audit requires the exact schema-v6 profile"
        )
    if len(rows) != len(evidences):
        raise VisualLatchAuditInputError(
            "open-only rows and visual evidences have unequal lengths"
        )
    frames = [
        _tip_probe_frame_from_evidence(evidence, trace_index=index)
        for index, evidence in enumerate(evidences)
    ]
    states = [
        _finite_vector(row.get("observation.state"), label=f"state[{index}]")
        for index, row in enumerate(rows)
    ]
    stop = min(len(frames), labels.pregrasp_open_frame + 1)
    if stop <= 0:
        raise VisualLatchAuditInputError(
            "open-only demonstration has no pregrasp observational window"
        )
    gate_frames: list[int] = []
    gate_flags: list[bool] = []
    observed_arm_speeds: list[float] = []
    observed_jaws: list[float] = []
    confirmation: int | None = None
    streak = 0
    capture_jaw = config.coarse_capture_jaw_rad
    assert capture_jaw is not None
    for index in range(stop):
        arm_speed = 0.0
        if index:
            arm_speed = max(
                abs(after - before) * DEFAULT_FPS
                for before, after in zip(
                    states[index - 1][:6], states[index][:6], strict=True
                )
            )
        matched = bool(
            _tip_probe_coarse_gate(frames[index], config)
            and _closed_interval(states[index][6], capture_jaw)
            and arm_speed <= config.max_arm_speed_rad_s + 1e-12
        )
        gate_flags.append(matched)
        observed_arm_speeds.append(float(arm_speed))
        observed_jaws.append(float(states[index][6]))
        if matched:
            gate_frames.append(index)
            streak += 1
            if confirmation is None and streak >= config.takeover_confirm_frames:
                confirmation = index
        else:
            streak = 0

    axis_authority = _open_only_axis_authority()
    checks = {
        "observational_trace_only": bool(
            len(frames) == len(states) == len(evidences)
        ),
        "axis_authority_namespaced": bool(
            axis_authority.get("scope") == "axis_partitioned_open_only"
        ),
        "tip_zero_arm_authority": bool(
            axis_authority.get("post_takeover_arm_motion_authorized") is False
            and axis_authority.get("post_takeover_arm_owner") == "hold"
        ),
        "tip_zero_explicit_descent_authority": bool(
            math.isclose(config.descent_step_m, 0.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                config.max_total_descent_m,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and axis_authority.get("tip_explicit_descent_authorized") is False
        ),
        "jaw_close_not_authorized": bool(
            config.allow_close is False
            and axis_authority.get("jaw_close_authorized") is False
        ),
        "hardware_completion_not_claimed": bool(
            stop <= len(frames) and len(gate_flags) == stop
        ),
        "observational_window_evaluated": bool(
            stop == labels.pregrasp_open_frame + 1
            and len(observed_arm_speeds) == stop
            and len(observed_jaws) == stop
        ),
    }
    trace_envelope_pass = all(checks.values())
    return {
        "semantics": (
            "recorded_demo_visual_envelope_axis_authority_not_hardware_response"
        ),
        "axis_authority": axis_authority,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "counterfactual_open_response_modeled": False,
        "takeover_confirmation_frame": None,
        "outcome_frame": None,
        "required_recorded_base_z_descent_m": 0.0,
        "required_descent_steps": 0,
        "observational_window": {
            "start_frame": 0,
            "end_frame": stop - 1,
            "evaluated_frames": stop,
            "coarse_gate_true_frames": sum(gate_flags),
            "maximum_arm_speed_rad_s": max(observed_arm_speeds),
            "observed_jaw_rad": {
                "min": min(observed_jaws),
                "max": max(observed_jaws),
            },
            "role": (
                "dataset_observation_only_not_deployment_positive_control"
            ),
        },
        "recorded_closed_capture_gate_frames": len(gate_frames),
        "recorded_closed_capture_observed": confirmation is not None,
        "recorded_closed_capture_confirmation_frame": confirmation,
        "checks": checks,
        "trace_envelope_pass": trace_envelope_pass,
        "limitation": (
            "The demonstrations prove only recorded closed-jaw visual coverage. "
            "They do not prohibit pre-takeover policy arm motion or TCP descent "
            "and contain no camera response caused by a tip open command. A zero "
            "closed-capture match count is reported as an observation, never as "
            "positive deployment evidence; the pinned v1 ACT supplies that window."
        ),
    }


def _tip_probe_frame_from_telemetry(
    visual: Mapping[str, Any], *, trace_index: int
) -> TipProbeFrame:
    frame = visual.get("frame")
    red = visual.get("red")
    if not isinstance(frame, Mapping) or not isinstance(red, Mapping):
        raise VisualLatchAuditInputError(
            "tip-probe negative control has malformed visual telemetry"
        )
    missing_frame_fields = [name for name in ("id", "status") if name not in frame]
    missing_red_fields = [
        name
        for name in (
            "centroid_norm",
            "contour_area_fraction",
            "candidate_count",
        )
        if name not in red
    ]
    if missing_frame_fields or missing_red_fields:
        raise VisualLatchAuditInputError(
            "tip-probe negative control omits required frame/red telemetry: "
            f"frame={missing_frame_fields!r}, red={missing_red_fields!r}"
        )
    centroid_raw = red.get("centroid_norm")
    centroid: tuple[float, float] | None
    if centroid_raw is None:
        centroid = None
    elif (
        isinstance(centroid_raw, list)
        and len(centroid_raw) == 2
    ):
        centroid = tuple(
            _strict_json_number(
                value,
                label=f"tip-probe negative control red centroid[{index}]",
            )
            for index, value in enumerate(centroid_raw)
        )
        if not all(0.0 <= value <= 1.0 for value in centroid):
            raise VisualLatchAuditInputError(
                "tip-probe negative control red centroid lies outside [0, 1]"
            )
    else:
        raise VisualLatchAuditInputError(
            "tip-probe negative control has invalid red centroid"
        )
    area_raw = red.get("contour_area_fraction")
    area: float | None = None
    if centroid is not None:
        area = _strict_json_number(
            area_raw,
            label="tip-probe negative control red area",
        )
        if not 0.0 <= area <= 1.0:
            raise VisualLatchAuditInputError(
                "tip-probe negative control red area lies outside [0, 1]"
            )
    frame_id = frame["id"]
    candidate_count = red["candidate_count"]
    if type(frame_id) is not int or frame_id < 0:
        raise VisualLatchAuditInputError(
            "tip-probe negative control frame id must be a nonnegative integer"
        )
    if type(candidate_count) is not int or candidate_count < 0:
        raise VisualLatchAuditInputError(
            "tip-probe negative control candidate count must be a nonnegative integer"
        )
    status = frame["status"]
    supported_statuses = {member.value for member in FrameStatus}
    if type(status) is not str or status not in supported_statuses:
        raise VisualLatchAuditInputError(
            "tip-probe negative control frame status must be a supported string"
        )
    return TipProbeFrame(
        trace_index=trace_index,
        frame_id=frame_id,
        fresh_unique=status == FrameStatus.FRESH_UNIQUE.value,
        candidate_count=candidate_count,
        centroid_norm=centroid,
        area_fraction=area,
    )


def _audit_tip_probe_negative_control_records(
    records: Sequence[Mapping[str, Any]], *, config: TipServoConfig
) -> dict[str, Any]:
    if config.allow_close:
        raise VisualLatchAuditInputError(
            "tip-probe negative control requires tip_servo.allow_close=false"
        )
    frames: list[TipProbeFrame] = []
    for record in records:
        if record.get("event") != "command_cycle":
            continue
        visual = record.get("visual_telemetry")
        if not isinstance(visual, Mapping):
            continue
        frames.append(
            _tip_probe_frame_from_telemetry(visual, trace_index=len(frames))
        )
    if not frames:
        raise VisualLatchAuditInputError(
            "tip-probe negative control contains no visual command cycles"
        )
    takeover = _first_tip_probe_confirmation(
        frames,
        lambda frame: _tip_probe_coarse_gate(frame, config),
        required_frames=config.takeover_confirm_frames,
    )
    progress = None
    close_ready = None
    tracking_exit = None
    if takeover is not None:
        start = takeover + 1
        progress = _first_tip_probe_confirmation(
            frames,
            lambda frame: _tip_probe_progress_gate(frame, config),
            required_frames=config.probe_success_confirm_frames,
            start=start,
        )
        close_ready = _first_tip_probe_confirmation(
            frames,
            lambda frame: _tip_probe_close_ready_gate(frame, config),
            required_frames=config.close_ready_confirm_frames,
            start=start,
        )
        tracking_exit = next(
            (
                index
                for index in range(start, len(frames))
                if not _tip_probe_tracking_gate(frames[index], config)
            ),
            None,
        )
    post_takeover_areas = [
        float(frame.area_fraction)
        for frame in frames[(takeover or 0) :]
        if frame.area_fraction is not None
    ]
    maximum_area = max(post_takeover_areas) if post_takeover_areas else None
    passed = bool(
        takeover is not None
        and progress is None
        and close_ready is None
        and config.allow_close is False
    )
    return {
        "semantics": "old_policy_trace_expected_safe_noncompletion",
        "allow_close": False,
        "close_authorized": False,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "visual_command_cycles": len(frames),
        "takeover_trace_index": takeover,
        "takeover_frame_id": None if takeover is None else frames[takeover].frame_id,
        "counterfactual_divergence_frame_id": (
            None if takeover is None else frames[takeover].frame_id
        ),
        "probe_progress_confirmation_frame_id": (
            None if progress is None else frames[progress].frame_id
        ),
        "visual_would_close_confirmation_frame_id": (
            None if close_ready is None else frames[close_ready].frame_id
        ),
        "first_tracking_exit_frame_id": (
            None if tracking_exit is None else frames[tracking_exit].frame_id
        ),
        "maximum_post_takeover_red_area_fraction": maximum_area,
        "expected_outcome": "captured_then_no_depth_progress",
        "negative_control_pass": passed,
        "post_takeover_recorded_frames_used_for_convergence": False,
        "limitation": (
            "The old ACT frames after takeover remain observations of the old "
            "policy. They are a fail-closed negative control, not a simulation "
            "of the proposed descent or proof of convergence."
        ),
    }


def _audit_tip_probe_open_only_negative_control_records(
    records: Sequence[Mapping[str, Any]], *, config: TipServoConfig
) -> dict[str, Any]:
    """Replay axis authority against the pinned historical ACT without hardware."""

    if config != _open_only_tip_servo_config():
        raise VisualLatchAuditInputError(
            "open-only negative control requires the exact schema-v6 profile"
        )
    command_records = [
        record for record in records if record.get("event") == "command_cycle"
    ]
    if not command_records:
        raise VisualLatchAuditInputError(
            "open-only negative control contains no command cycles"
        )
    acquired = [
        record for record in records if record.get("event") == "control_acquired"
    ]
    if len(acquired) != 1:
        raise VisualLatchAuditInputError(
            "open-only negative control requires exactly one control_acquired event"
        )
    capture_target = _strict_json_vector(
        acquired[0].get("hold_target"),
        length=7,
        label="open-only negative control acquired hold_target",
    )[6]
    capture_gate = config.coarse_capture_jaw_rad
    assert capture_gate is not None
    if not _closed_interval(capture_target, capture_gate):
        raise VisualLatchAuditInputError(
            "open-only negative control startup jaw is outside the capture gate"
        )

    parsed: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            TipProbeFrame,
            tuple[float, ...],
        ]
    ] = []
    arm_commands = 0
    descent_commands = 0
    jaw_commands = 0
    close_commands = 0
    final_jaw_targets: list[float] = []
    arm_targets: list[tuple[float, ...]] = []
    tcp_z_values: list[float] = []
    for trace_index, record in enumerate(command_records):
        if type(record.get("safety_ok")) is not bool:
            raise VisualLatchAuditInputError(
                "open-only negative control command cycle omits boolean safety_ok"
            )
        visual = record.get("visual_telemetry")
        if not isinstance(visual, Mapping):
            raise VisualLatchAuditInputError(
                "open-only negative control command cycle omits visual telemetry"
            )
        frame = _tip_probe_frame_from_telemetry(
            visual, trace_index=trace_index
        )
        tip = visual.get("tip_servo")
        if not isinstance(tip, Mapping):
            raise VisualLatchAuditInputError(
                "open-only negative control command cycle omits tip-servo telemetry"
            )
        target = _strict_json_vector(
            record.get("target"),
            length=7,
            label=f"open-only negative control target[{trace_index}]",
        )
        _strict_json_vector(
            record.get("measured"),
            length=7,
            label=f"open-only negative control measured[{trace_index}]",
        )
        required_tip_fields = (
            "commanded_arm_target_q_rad",
            "commanded_jaw_target_rad",
            "commanded_descent_m",
            "probe_close_allowed",
            "arm_owner",
            "discard_policy_chunk",
            "smoother_reset_required",
        )
        missing_tip_fields = [
            name for name in required_tip_fields if name not in tip
        ]
        if missing_tip_fields:
            raise VisualLatchAuditInputError(
                "open-only negative control tip-servo telemetry omits required "
                f"fields: {missing_tip_fields!r}"
            )
        commanded_arm = tip["commanded_arm_target_q_rad"]
        if commanded_arm is not None:
            _strict_json_vector(
                commanded_arm,
                length=6,
                label=(
                    "open-only negative control commanded arm target"
                    f"[{trace_index}]"
                ),
            )
            arm_commands += 1
        commanded_jaw = tip["commanded_jaw_target_rad"]
        if commanded_jaw is not None:
            _strict_json_number(
                commanded_jaw,
                label=(
                    "open-only negative control commanded jaw target"
                    f"[{trace_index}]"
                ),
            )
            jaw_commands += 1
        descent = _strict_json_number(
            tip["commanded_descent_m"],
            label=f"open-only negative control commanded descent[{trace_index}]",
        )
        probe_close_allowed = tip["probe_close_allowed"]
        if type(probe_close_allowed) is not bool:
            raise VisualLatchAuditInputError(
                "open-only negative control probe_close_allowed must be boolean"
            )
        descent_commands += abs(descent) > 1e-12
        close_commands += probe_close_allowed
        arm_owner = tip["arm_owner"]
        if type(arm_owner) is not str or arm_owner not in {
            "policy",
            "hold",
            "visual_tip_servo",
        }:
            raise VisualLatchAuditInputError(
                "open-only negative control has invalid arm_owner"
            )
        for name in ("discard_policy_chunk", "smoother_reset_required"):
            if type(tip[name]) is not bool:
                raise VisualLatchAuditInputError(
                    f"open-only negative control {name} must be boolean"
                )
        if "arm_speed_rad_s" not in visual:
            raise VisualLatchAuditInputError(
                "open-only negative control omits arm_speed_rad_s"
            )
        raw_arm_speed = visual["arm_speed_rad_s"]
        if raw_arm_speed is not None:
            parsed_arm_speed = _strict_json_number(
                raw_arm_speed,
                label=f"open-only negative control arm speed[{trace_index}]",
            )
            if parsed_arm_speed < 0.0:
                raise VisualLatchAuditInputError(
                    "open-only negative control arm speed must be nonnegative"
                )
        pickup = visual.get("pickup_policy_close_intent")
        if not isinstance(pickup, Mapping):
            raise VisualLatchAuditInputError(
                "open-only negative control omits historical policy action source"
            )
        action_source = pickup.get("action_source")
        if type(action_source) is not str or action_source not in {
            "fresh_policy",
            "inference_hold",
        }:
            raise VisualLatchAuditInputError(
                "open-only negative control has unsupported policy action source"
            )
        final_jaw_targets.append(target[6])
        arm_targets.append(target[:6])
        if "tcp_position_m" not in visual:
            raise VisualLatchAuditInputError(
                f"open-only negative control omits tcp[{trace_index}]"
            )
        tcp = _strict_json_vector(
            visual["tcp_position_m"],
            length=3,
            label=f"open-only negative control tcp[{trace_index}]",
        )
        tcp_z_values.append(tcp[2])
        parsed.append((record, visual, frame, target))

    window = [
        item
        for item in parsed
        if TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID
        <= item[2].frame_id
        <= TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
    ]
    expected_ids = list(
        range(
            TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID,
            TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID + 1,
        )
    )
    frame_ids = [item[2].frame_id for item in window]
    consecutive = frame_ids == expected_ids

    jaw_values: list[float] = []
    arm_speeds: list[float] = []
    action_sources: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    area_values: list[float] = []
    window_gate_flags: list[bool] = []
    all_fresh = True
    all_safety_ok = True
    for record, visual, frame, _target in window:
        measured = _strict_json_vector(
            record.get("measured"),
            length=7,
            label=f"open-only negative control measured[{frame.frame_id}]",
        )
        arm_speed = _strict_json_number(
            visual["arm_speed_rad_s"],
            label=f"open-only negative control arm speed[{frame.frame_id}]",
        )
        pickup = visual.get("pickup_policy_close_intent")
        if not isinstance(pickup, Mapping):
            raise VisualLatchAuditInputError(
                "open-only negative control omits historical policy action source"
            )
        action_source = pickup.get("action_source")
        if action_source not in {"fresh_policy", "inference_hold"}:
            raise VisualLatchAuditInputError(
                "open-only negative control has unsupported policy action source"
            )
        assert frame.centroid_norm is not None
        assert frame.area_fraction is not None
        jaw = measured[6]
        jaw_values.append(jaw)
        arm_speeds.append(arm_speed)
        action_sources.append(str(action_source))
        x_values.append(frame.centroid_norm[0])
        y_values.append(frame.centroid_norm[1])
        area_values.append(frame.area_fraction)
        all_fresh = all_fresh and frame.fresh_unique
        all_safety_ok = all_safety_ok and record.get("safety_ok") is True
        window_gate_flags.append(
            bool(
                _tip_probe_coarse_gate(frame, config)
                and _closed_interval(jaw, capture_gate)
                and arm_speed <= config.max_arm_speed_rad_s + 1e-12
            )
        )

    first_confirmed_index: int | None = None
    streak = 0
    for index, matched in enumerate(window_gate_flags):
        streak = streak + 1 if matched else 0
        if first_confirmed_index is None and streak >= config.takeover_confirm_frames:
            first_confirmed_index = index
    would_takeover_index = None
    if first_confirmed_index is not None:
        would_takeover_index = next(
            (
                index
                for index in range(first_confirmed_index, len(window))
                if action_sources[index] == "fresh_policy"
            ),
            None,
        )
    source_counts = dict(sorted(Counter(action_sources).items()))
    all_gate = bool(window) and all(window_gate_flags)

    boundary_parsed_index = next(
        (
            index
            for index, item in enumerate(parsed)
            if item[2].frame_id == TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
        ),
        None,
    )
    if boundary_parsed_index is None:
        raise VisualLatchAuditInputError(
            "open-only negative control omits the takeover boundary frame"
        )
    replay_records = parsed[: boundary_parsed_index + 1]
    replayed_targets: list[float] = []
    suppressed_lower = 0
    suppressed_early_open = 0
    previous = capture_target
    for _record, _visual, _frame, historical_target in replay_records:
        suppressed_lower += historical_target[6] < capture_target - 1e-12
        suppressed_early_open += historical_target[6] > capture_target + 1e-12
        previous = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture_target,
            previous_accepted_jaw_target_rad=previous,
            open_commanded=False,
            config=config,
            ee_step_limit_rad=0.0165,
        )
        replayed_targets.append(previous)

    takeover_target = resolve_open_only_jaw_target(
        capture_jaw_target_rad=capture_target,
        previous_accepted_jaw_target_rad=previous,
        open_commanded=False,
        config=config,
        ee_step_limit_rad=0.0165,
    )
    opened_targets = [takeover_target]
    previous = takeover_target
    for _ in range(32):
        next_target = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture_target,
            previous_accepted_jaw_target_rad=previous,
            open_commanded=True,
            config=config,
            ee_step_limit_rad=0.0165,
        )
        opened_targets.append(next_target)
        previous = next_target
        if math.isclose(
            previous,
            config.pregrasp_open_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            break
    open_steps = [
        after - before
        for before, after in zip(opened_targets, opened_targets[1:])
    ]
    reclose_probe_rad = min(final_jaw_targets)
    after_reclose = resolve_open_only_jaw_target(
        capture_jaw_target_rad=capture_target,
        previous_accepted_jaw_target_rad=opened_targets[1],
        open_commanded=True,
        config=config,
        ee_step_limit_rad=0.0165,
    )
    fault_hold = resolve_open_only_jaw_target(
        capture_jaw_target_rad=capture_target,
        previous_accepted_jaw_target_rad=opened_targets[-1],
        open_commanded=True,
        config=config,
        ee_step_limit_rad=0.0165,
    )
    timeout_hold = resolve_open_only_jaw_target(
        capture_jaw_target_rad=capture_target,
        previous_accepted_jaw_target_rad=capture_target,
        open_commanded=False,
        config=config,
        ee_step_limit_rad=0.0165,
    )
    runtime_replay = _open_only_runtime_authority_replay(
        config=config,
        capture_target=capture_target,
    )

    historical_lower = sum(
        value < capture_target - 1e-12 for value in final_jaw_targets
    )
    historical_higher = sum(
        value > capture_target + 1e-12 for value in final_jaw_targets
    )
    first_arm_target = arm_targets[0]
    policy_arm_motion_observed = any(
        max(
            abs(value - start)
            for value, start in zip(
                target,
                first_arm_target,
                strict=True,
            )
        )
        > 1e-6
        for target in arm_targets[1:]
    )
    minimum_tcp_delta = (
        min(value - tcp_z_values[0] for value in tcp_z_values)
        if tcp_z_values
        else None
    )
    seeking_hold_exact = bool(replayed_targets) and all(
        math.isclose(value, capture_target, rel_tol=0.0, abs_tol=1e-12)
        for value in replayed_targets
    )
    open_monotonic = all(step >= -1e-12 for step in open_steps)
    open_step_limited = all(step <= 0.0165 + 1e-12 for step in open_steps)
    open_reached = math.isclose(
        opened_targets[-1],
        config.pregrasp_open_target_rad,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    passed = bool(
        consecutive
        and all_fresh
        and all_safety_ok
        and all_gate
        and source_counts == {"fresh_policy": 3, "inference_hold": 5}
        and first_confirmed_index is not None
        and window[first_confirmed_index][2].frame_id == 492
        and action_sources[first_confirmed_index] == "inference_hold"
        and would_takeover_index is not None
        and window[would_takeover_index][2].frame_id
        == TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
        and arm_commands == 0
        and descent_commands == 0
        and jaw_commands == 0
        and close_commands == 0
        and historical_lower > 0
        and historical_higher > 0
        and policy_arm_motion_observed
        and minimum_tcp_delta is not None
        and minimum_tcp_delta < -0.020
        and suppressed_lower > 0
        and suppressed_early_open > 0
        and seeking_hold_exact
        and math.isclose(
            takeover_target, capture_target, rel_tol=0.0, abs_tol=1e-12
        )
        and opened_targets[1] > capture_target
        and open_monotonic
        and open_step_limited
        and open_reached
        and after_reclose >= opened_targets[1] - 1e-12
        and fault_hold >= opened_targets[-1] - 1e-12
        and math.isclose(
            timeout_hold, capture_target, rel_tol=0.0, abs_tol=1e-12
        )
        and runtime_replay["runtime_authority_replay_pass"] is True
    )
    return {
        "semantics": "historical_act_axis_authority_counterfactual_replay",
        "axis_authority": _open_only_axis_authority(),
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "visual_command_cycles": len(command_records),
        "calibrated_window_start_frame_id": (
            TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID
        ),
        "calibrated_window_end_frame_id": TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID,
        "calibrated_window_frames": len(window),
        "calibrated_window_consecutive": consecutive,
        "calibrated_window_all_fresh_unique": all_fresh,
        "calibrated_window_all_closed_capture_gate": all_gate,
        "calibrated_window_action_source_counts": source_counts,
        "first_confirmed_hold_frame_id": (
            None
            if first_confirmed_index is None
            else window[first_confirmed_index][2].frame_id
        ),
        "would_takeover_boundary_frame_id": (
            None
            if would_takeover_index is None
            else window[would_takeover_index][2].frame_id
        ),
        "would_takeover_boundary_action_source": (
            None
            if would_takeover_index is None
            else action_sources[would_takeover_index]
        ),
        "observed_x_norm": {
            "min": min(x_values) if x_values else None,
            "max": max(x_values) if x_values else None,
        },
        "observed_y_norm": {
            "min": min(y_values) if y_values else None,
            "max": max(y_values) if y_values else None,
        },
        "observed_area_fraction": {
            "min": min(area_values) if area_values else None,
            "max": max(area_values) if area_values else None,
        },
        "observed_jaw_rad": {
            "min": min(jaw_values) if jaw_values else None,
            "max": max(jaw_values) if jaw_values else None,
        },
        "observed_max_arm_speed_rad_s": (
            max(arm_speeds) if arm_speeds else None
        ),
        "actual_tip_servo_arm_command_cycles": arm_commands,
        "actual_tip_servo_descent_command_cycles": descent_commands,
        "actual_tip_servo_jaw_command_cycles": jaw_commands,
        "actual_tip_servo_close_command_cycles": close_commands,
        "historical_out_of_scope": {
            "policy_arm_motion_observed": policy_arm_motion_observed,
            "minimum_measured_tcp_delta_z_m": minimum_tcp_delta,
            "final_jaw_target_min_rad": min(final_jaw_targets),
            "final_jaw_target_max_rad": max(final_jaw_targets),
            "final_jaw_target_below_capture_cycles": historical_lower,
            "final_jaw_target_above_capture_cycles": historical_higher,
        },
        "axis_authority_replay": {
            "capture_jaw_target_rad": capture_target,
            "replayed_seeking_cycles": len(replayed_targets),
            "published_jaw_min_rad": min(replayed_targets),
            "published_jaw_max_rad": max(replayed_targets),
            "suppressed_lower_jaw_proposal_cycles": suppressed_lower,
            "suppressed_early_open_proposal_cycles": suppressed_early_open,
            "takeover_boundary_published_jaw_target_rad": capture_target,
            "takeover_boundary_still_capture_hold": math.isclose(
                takeover_target,
                capture_target,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "first_open_requires_takeover_and_settle": bool(
                opened_targets[0] == capture_target
                and opened_targets[1] > capture_target
            ),
            "opened_targets_rad": opened_targets,
            "opened_target_monotonic_nondecreasing": open_monotonic,
            "opened_target_step_limit_pass": open_step_limited,
            "open_target_reached": open_reached,
            "adversarial_reclose_proposal_rad": reclose_probe_rad,
            "adversarial_reclose_resolved_target_rad": after_reclose,
            "adversarial_reclose_suppressed": (
                after_reclose >= opened_targets[1] - 1e-12
            ),
            "fault_hold_non_decreasing": (
                fault_hold >= opened_targets[-1] - 1e-12
            ),
            "seeking_timeout_ms": config.seeking_timeout_ms,
            "seeking_timeout_holds_capture": math.isclose(
                timeout_hold,
                capture_target,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "counterfactual_open_response_modeled": False,
        },
        "runtime_authority_replay": runtime_replay,
        "expected_outcome": (
            "arm6_policy_j1_j6_search_with_external_capture_jaw_hold_then_"
            "arm_hold_and_tip_open"
        ),
        "negative_control_pass": passed,
        "limitation": (
            "The historical trace proves the visual boundary and supplies real "
            "adversarial pre-fix jaw/TCP behavior. The schema-v6 arm6 adapter and "
            "the deployed VisualLatchRuntime composer replay command authority; no "
            "counterfactual hardware response or closed-loop convergence is claimed."
        ),
    }


def _open_only_tip_servo_config() -> TipServoConfig:
    return TipServoConfig(
        mode=TipServoMode.EXECUTE,
        probe_profile=TipServoProbeProfile.OPEN_ONLY,
        coarse_x_norm=(0.50, 0.545),
        coarse_y_norm=(0.42, 0.56),
        coarse_area_fraction=(0.017, 0.029),
        coarse_capture_jaw_rad=(0.285, 0.325),
        tracking_x_norm=(0.40, 0.56),
        tracking_y_norm=(0.40, 0.62),
        tracking_area_fraction=(0.014, 0.041),
        max_arm_speed_rad_s=0.07,
        max_takeover_reference_error_rad=0.035,
        arm_target_settle_rad=(0.002,) * 6,
        opening_timeout_ms=1500.0,
        seeking_timeout_ms=20_000.0,
        descent_step_m=0.0,
        max_total_descent_m=0.0,
        allow_close=False,
    )


def _load_tip_probe_negative_control(
    path: Path, *, config: TipServoConfig
) -> dict[str, Any]:
    absolute = Path(path).expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise VisualLatchAuditInputError(
            "tip-probe negative control must be a regular, non-symlink JSONL file: "
            f"{absolute}"
        )
    try:
        data = absolute.read_bytes()
    except OSError as exc:
        raise VisualLatchAuditInputError(
            f"cannot read tip-probe negative control {absolute}: {exc}"
        ) from exc
    digest = hashlib.sha256(data).hexdigest()
    if config.probe_profile is TipServoProbeProfile.OPEN_ONLY:
        if (
            len(data) != EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE
            or digest != EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SHA256
        ):
            raise VisualLatchAuditInputError(
                "tip-probe negative control does not match the pinned failed "
                "ACT-30s artifact identity"
            )

        def reject_open_only_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value!r}")

        open_only_records: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(data.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=reject_open_only_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise VisualLatchAuditInputError(
                    "tip-probe negative control is not strict JSONL at line "
                    f"{line_number}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise VisualLatchAuditInputError(
                    "tip-probe negative control JSONL rows must be objects"
                )
            open_only_records.append(value)
        if (
            len(open_only_records)
            != EXPECTED_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS
        ):
            raise VisualLatchAuditInputError(
                "tip-probe negative control does not contain the pinned failed "
                "ACT-30s record count"
            )
        result = _audit_tip_probe_open_only_negative_control_records(
            open_only_records, config=config
        )
        result["source"] = {
            "path": str(absolute),
            "size": len(data),
            "sha256": digest,
            "jsonl_records": len(open_only_records),
        }
        return result
    if (
        len(data) != EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_SIZE
        or digest != EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_SHA256
    ):
        raise VisualLatchAuditInputError(
            "tip-probe negative control does not match the pinned old ACT "
            "artifact identity"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(
                raw_line.decode("utf-8"), parse_constant=reject_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise VisualLatchAuditInputError(
                "tip-probe negative control is not strict JSONL at line "
                f"{line_number}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise VisualLatchAuditInputError(
                "tip-probe negative control JSONL rows must be objects"
            )
        records.append(value)
    if len(records) != EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_RECORDS:
        raise VisualLatchAuditInputError(
            "tip-probe negative control does not contain the pinned old ACT "
            f"record count {EXPECTED_TIP_PROBE_NEGATIVE_CONTROL_RECORDS}"
        )
    result = _audit_tip_probe_negative_control_records(records, config=config)
    result["source"] = {
        "path": str(absolute),
        "size": len(data),
        "sha256": digest,
        "jsonl_records": len(records),
    }
    return result


def _validate_tip_probe_audit_config(config: TipServoConfig) -> None:
    """Pin schema-v3 reports to the reviewed first-probe authority."""

    if config.probe_profile is TipServoProbeProfile.OPEN_ONLY:
        if config != _open_only_tip_servo_config():
            raise VisualLatchAuditInputError(
                "schema-v6 tip-probe audit requires the exact execute/open-only "
                "zero-arm, zero-descent, allow_close=false profile"
            )
        return
    if config.allow_close:
        raise VisualLatchAuditInputError(
            "tip-probe audit profile requires tip_servo.allow_close=false"
        )
    if not math.isclose(
        config.descent_step_m,
        REQUIRED_TIP_PROBE_DESCENT_STEP_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise VisualLatchAuditInputError(
            "schema-v3 tip-probe audit requires "
            f"tip_servo.descent_step_m={REQUIRED_TIP_PROBE_DESCENT_STEP_M}"
        )
    if not math.isclose(
        config.max_total_descent_m,
        REQUIRED_TIP_PROBE_MAX_TOTAL_DESCENT_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise VisualLatchAuditInputError(
            "schema-v3 tip-probe audit requires "
            "tip_servo.max_total_descent_m="
            f"{REQUIRED_TIP_PROBE_MAX_TOTAL_DESCENT_M}"
        )


def _audit_schema_version(config: VisualPhaseLatchConfig) -> int:
    if (
        config.tip_servo.mode is TipServoMode.EXECUTE
        and config.tip_servo.probe_profile is TipServoProbeProfile.OPEN_ONLY
    ):
        return TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION
    if config.tip_servo.mode is TipServoMode.EXECUTE:
        return TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION
    return VISUAL_LATCH_AUDIT_SCHEMA_VERSION


def audit_firefly_y6_visual_latch(
    dataset_root: Path,
    *,
    expected_episodes: int | None = 50,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    controller_contract: Path = DEFAULT_CONTROLLER_CONTRACT,
    expected_task: str = DEFAULT_EXPECTED_TASK,
    arm_urdf_path: Path = DEFAULT_ARM_URDF_PATH,
    jaw_crossing_rad: float = DEFAULT_JAW_CROSSING_RAD,
    jaw_open_confirm_rad: float = DEFAULT_JAW_OPEN_CONFIRM_RAD,
    jaw_closed_confirm_rad: float = DEFAULT_JAW_CLOSED_CONFIRM_RAD,
    config: VisualPhaseLatchConfig | None = None,
    windows: AuditWindows = AuditWindows(),
    tip_probe_negative_control_path: Path = DEFAULT_TIP_PROBE_NEGATIVE_CONTROL,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run the complete read-only detector replay and return a strict JSON report."""

    root = Path(dataset_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise VisualLatchAuditInputError(
            f"dataset root must be a regular directory, not a symlink: {root}"
        )
    config = VisualPhaseLatchConfig(enabled=True) if config is None else config
    tip_probe_config = config.tip_servo
    _validate_tip_probe_audit_config(tip_probe_config)
    tip_probe_open_only = (
        tip_probe_config.probe_profile is TipServoProbeProfile.OPEN_ONLY
    )
    if tip_probe_open_only and checkpoint is None:
        raise VisualLatchAuditInputError(
            "schema-6 OPEN_ONLY audit requires --checkpoint for the attested "
            "arm6_seeking policy"
        )
    if not tip_probe_open_only and checkpoint is not None:
        raise VisualLatchAuditInputError(
            "checkpoint attestation binding is valid only for schema-6 OPEN_ONLY"
        )
    source_info = _read_json_object(
        root / "meta" / "info.json", label="dataset info"
    )
    _require_original_fullcycle_v4_7d_source(source_info)
    tip_probe_negative_control = _load_tip_probe_negative_control(
        tip_probe_negative_control_path,
        config=tip_probe_config,
    )
    source_binding_before = _capture_source_binding(
        root,
        dataset_repo_id=dataset_repo_id,
        controller_contract=Path(controller_contract),
        expected_task=expected_task,
        config=config,
        checkpoint=checkpoint,
    )
    info, metadata, frames = _load_dataset(
        root, expected_episodes=expected_episodes
    )
    fk_model, fk_binding = _load_fk_model(arm_urdf_path)
    episode_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    close_offsets: list[int] = []
    runtime_close_offsets: list[int] = []
    policy_close_intent_offsets: list[int] = []
    contact_offsets: list[int] = []
    release_offsets: list[int] = []
    held_coverages: list[float] = []
    close_contour_diagnostics: list[Mapping[str, Any]] = []
    held_contour_diagnostics: list[Mapping[str, Any]] = []
    close_rejections: Counter[str] = Counter()
    held_rejections: Counter[str] = Counter()
    blue_rejections: Counter[str] = Counter()
    state_machine_grasp_latch_offsets: list[int] = []
    state_machine_grasp_confirm_offsets: list[int] = []
    state_machine_release_unlatch_offsets: list[int] = []
    state_machine_release_confirm_offsets: list[int] = []
    state_machine_home_close_offsets: list[int] = []
    state_machine_done_offsets: list[int] = []
    tip_probe_episode_reports: list[dict[str, Any]] = []

    for episode_index in range(int(info["total_episodes"])):
        rows = frames[episode_index]
        actions = [row["action"] for row in rows]
        labels = reconstruct_phase_labels(
            actions,
            jaw_crossing_rad=jaw_crossing_rad,
            jaw_open_confirm_rad=jaw_open_confirm_rad,
            jaw_closed_confirm_rad=jaw_closed_confirm_rad,
        )
        path = _wrist_video_path(root, info, metadata[episode_index])
        detected = _detect_episode(
            path, expected_frames=len(rows), config=config
        )
        runtime = _runtime_gate_proxies(
            rows,
            close_visual_flags=detected["close_flags"],
            config=config,
            fps=DEFAULT_FPS,
        )
        state_machine = _state_machine_replay(
            rows,
            detected["evidences"],
            labels,
            config=config,
            fk_model=fk_model,
        )
        if tip_probe_open_only:
            tip_probe = _tip_probe_open_only_demo_envelope(
                rows,
                detected["evidences"],
                labels,
                config=tip_probe_config,
            )
        else:
            tip_probe = _tip_probe_demo_trace_envelope(
                rows,
                detected["evidences"],
                labels,
                config=tip_probe_config,
                fk_model=fk_model,
            )
        tip_probe_episode_reports.append(tip_probe)
        evidence = audit_evidence_flags(
            labels,
            close_flags=detected["close_flags"],
            held_flags=detected["held_flags"],
            blue_flags=detected["blue_flags"],
            close_runtime_gate_flags=runtime["close_gate_flags"],
            contact_confirm_frame=runtime["contact_confirm_frame"],
            close_reasons=detected["close_reasons"],
            held_reasons=detected["held_reasons"],
            blue_reasons=detected["blue_reasons"],
            config=config,
            fps=DEFAULT_FPS,
            windows=windows,
        )
        close = evidence["close"]
        held = evidence["held_red"]
        blue = evidence["blue"]
        # Use the unbounded runtime proxy result here. The windowed helper is
        # intentionally limited to visual pseudo-label diagnostics and can
        # otherwise hide a valid late policy-intent latch.
        close["runtime_gate_confirmation_frame"] = runtime[
            "close_latch_frame"
        ]
        close["runtime_gate_confirmation_offset_frames"] = (
            None
            if runtime["close_latch_frame"] is None
            else int(runtime["close_latch_frame"])
            - labels.pickup_close_frame
        )
        if close["confirmation_offset_frames"] is not None:
            close_offsets.append(int(close["confirmation_offset_frames"]))
        if close["runtime_gate_confirmation_offset_frames"] is not None:
            runtime_close_offsets.append(
                int(close["runtime_gate_confirmation_offset_frames"])
            )
        if runtime["first_policy_close_intent_frame"] is not None:
            policy_close_intent_offsets.append(
                int(runtime["first_policy_close_intent_frame"])
                - labels.pickup_close_frame
            )
        if held["contact_confirm_offset_from_pickup_close_frames"] is not None:
            contact_offsets.append(
                int(held["contact_confirm_offset_from_pickup_close_frames"])
            )
        if blue["release_confirmation_offset_frames"] is not None:
            release_offsets.append(int(blue["release_confirmation_offset_frames"]))
        held_coverages.append(float(held["coverage"]))
        state_offsets = state_machine["offsets_frames"]
        for key, target in (
            (
                "grasp_latched_from_pickup_close",
                state_machine_grasp_latch_offsets,
            ),
            (
                "grasp_confirmed_from_pickup_close",
                state_machine_grasp_confirm_offsets,
            ),
            (
                "release_unlatched_from_release_open",
                state_machine_release_unlatch_offsets,
            ),
            (
                "release_confirmed_from_release_open",
                state_machine_release_confirm_offsets,
            ),
            (
                "home_closing_started_from_return_close",
                state_machine_home_close_offsets,
            ),
            ("task_done_from_return_close", state_machine_done_offsets),
        ):
            if state_offsets[key] is not None:
                target.append(int(state_offsets[key]))
        _merge_reason_counts(close_rejections, close["rejection_reason_counts_in_window"])
        _merge_reason_counts(held_rejections, held["rejection_reason_counts"])
        _merge_reason_counts(blue_rejections, blue["rejection_reason_counts_in_release_window"])
        gap = held["max_missing_run"]
        if gap["start_frame"] is not None:
            gap_start = int(gap["start_frame"])
            gap_end = int(gap["end_frame"])
            gap.update(
                {
                    "start_offset_from_pickup_close_frames": (
                        gap_start - labels.pickup_close_frame
                    ),
                    "end_offset_from_pickup_close_frames": (
                        gap_end - labels.pickup_close_frame
                    ),
                    "start_offset_from_contact_confirm_frames": (
                        None
                        if runtime["contact_confirm_frame"] is None
                        else gap_start - int(runtime["contact_confirm_frame"])
                    ),
                    "end_offset_from_contact_confirm_frames": (
                        None
                        if runtime["contact_confirm_frame"] is None
                        else gap_end - int(runtime["contact_confirm_frame"])
                    ),
                    "start_offset_from_release_open_frames": (
                        gap_start - labels.release_open_frame
                    ),
                    "end_offset_from_release_open_frames": (
                        gap_end - labels.release_open_frame
                    ),
                    "reasons_by_frame": [
                        {
                            "frame": frame,
                            "reasons": list(detected["held_reasons"][frame]),
                        }
                        for frame in range(gap_start, gap_end + 1)
                    ],
                }
            )
        close_first, close_last = close["window_frames"]
        held_first, held_last = held["interval_frames"]
        episode_close_contours = detected["red_contours"][
            close_first : close_last + 1
        ]
        episode_held_contours = detected["red_contours"][
            held_first : held_last + 1
        ]
        close_contour_diagnostics.extend(episode_close_contours)
        held_contour_diagnostics.extend(episode_held_contours)

        checks = {
            "no_premature_close_confirmation": not close["premature_confirmation_frames"],
            "close_detected_in_window": close["detected_in_window"],
            "contact_confirmed": held["contact_confirmed"],
            "held_red_survives_loss_timeout": held["survives_loss_timeout"],
            "no_pretransport_blue_confirmation": not blue["premature_confirmation_frames"],
            "blue_detected_in_release_window": blue["detected_in_release_window"],
            "state_machine_ordered_complete": state_machine["ordered_complete"],
            "state_machine_fault_free": state_machine["fault_free"],
            "tip_probe_trace_envelope_pass": tip_probe["trace_envelope_pass"],
            "tip_probe_close_not_authorized": (
                (
                    tip_probe["axis_authority"]["jaw_close_authorized"] is False
                )
                if tip_probe_open_only
                else tip_probe["close_authorized"] is False
            ),
            "tip_probe_hardware_completion_not_claimed": (
                tip_probe["hardware_completion_validated"] is False
                and tip_probe["closed_loop_convergence_claimed"] is False
            ),
        }
        if tip_probe_open_only:
            axis_authority = tip_probe["axis_authority"]
            checks.update(
                {
                    "tip_probe_open_only_envelope_pass": tip_probe[
                        "trace_envelope_pass"
                    ],
                    "tip_probe_axis_authority_namespaced": (
                        axis_authority == _open_only_axis_authority()
                    ),
                    "tip_probe_policy_arm_only_before_takeover": (
                        axis_authority[
                            "pre_takeover_policy_arm_motion_authorized"
                        ]
                        is True
                        and axis_authority[
                            "post_takeover_arm_motion_authorized"
                        ]
                        is False
                    ),
                    "tip_probe_jaw_isolated_from_policy_and_main": (
                        axis_authority["policy_jaw_authorized"] is False
                        and axis_authority["main_latch_jaw_authorized"] is False
                    ),
                    "tip_probe_tip_zero_explicit_descent_authority": (
                        axis_authority[
                            "tip_explicit_descent_authorized"
                        ]
                        is False
                    ),
                    "tip_probe_close_not_authorized": (
                        axis_authority["jaw_close_authorized"] is False
                    ),
                    "tip_probe_hardware_completion_not_claimed": (
                        tip_probe["hardware_completion_validated"] is False
                        and tip_probe["closed_loop_convergence_claimed"] is False
                    ),
                }
            )
        for code, passed in checks.items():
            if not passed:
                errors.append({"episode_index": episode_index, "code": code})
        episode_reports.append(
            {
                "episode_index": episode_index,
                "frames": len(rows),
                "video": str(path.relative_to(root)),
                "frame_status_counts": detected["frame_status_counts"],
                "phase_labels": asdict(labels),
                "state_jaw_at_phase_labels_rad": {
                    name: float(rows[getattr(labels, name)]["observation.state"][6])
                    for name in asdict(labels)
                },
                "checks": checks,
                "runtime_proxy": {
                    "first_policy_close_intent_frame": runtime[
                        "first_policy_close_intent_frame"
                    ],
                    "first_policy_close_intent_offset_from_pickup_close_frames": (
                        None
                        if runtime["first_policy_close_intent_frame"] is None
                        else int(runtime["first_policy_close_intent_frame"])
                        - labels.pickup_close_frame
                    ),
                    "policy_close_intent_timeout_frames": runtime[
                        "policy_close_intent_timeout_frames"
                    ],
                    "close_latch_frame": runtime["close_latch_frame"],
                    "arm_speed_rad_s_at_close_latch": runtime[
                        "arm_speed_rad_s_at_close_latch"
                    ],
                    "contact_confirm_frame": runtime[
                        "contact_confirm_frame"
                    ],
                    "contact_gap_rad_at_confirmation": runtime[
                        "contact_gap_rad_at_confirmation"
                    ],
                    "simulated_command_ee_rad_at_contact_confirmation": runtime[
                        "simulated_command_ee_rad_at_contact_confirmation"
                    ],
                    "jaw_speed_rad_s_at_contact_confirmation": runtime[
                        "jaw_speed_rad_s_at_contact_confirmation"
                    ],
                },
                "red_contour_diagnostics": {
                    "close_window": _contour_window_summary(
                        episode_close_contours
                    ),
                    "held_interval": _contour_window_summary(
                        episode_held_contours
                    ),
                },
                **evidence,
                "safe": all(checks.values()),
                "state_machine_replay": state_machine,
                **(
                    {"tip_probe_open_only_envelope": tip_probe}
                    if tip_probe_open_only
                    else {"tip_probe_trace_envelope": tip_probe}
                ),
            }
        )

    total = len(episode_reports)
    close_detected = sum(
        report["close"]["detected_in_window"] for report in episode_reports
    )
    held_safe = sum(
        report["held_red"]["survives_loss_timeout"] for report in episode_reports
    )
    release_detected = sum(
        report["blue"]["detected_in_release_window"] for report in episode_reports
    )
    premature_close_episodes = sum(
        bool(report["close"]["premature_confirmation_frames"])
        for report in episode_reports
    )
    premature_blue_episodes = sum(
        bool(report["blue"]["premature_confirmation_frames"])
        for report in episode_reports
    )
    if not tip_probe_negative_control["negative_control_pass"]:
        errors.append({"code": "tip_probe_negative_control_semantics_failed"})
    safe = not errors
    contact_confirmed_episodes = sum(
        report["held_red"]["contact_confirmed"] for report in episode_reports
    )
    runtime_close_before_pickup_episodes = sum(
        report["close"]["runtime_gate_confirmation_offset_frames"] is not None
        and report["close"]["runtime_gate_confirmation_offset_frames"] < 0
        for report in episode_reports
    )
    state_machine_ordered_complete_episodes = sum(
        report["state_machine_replay"]["ordered_complete"]
        for report in episode_reports
    )
    state_machine_fault_free_episodes = sum(
        report["state_machine_replay"]["fault_free"]
        for report in episode_reports
    )
    source_binding_after = _capture_source_binding(
        root,
        dataset_repo_id=dataset_repo_id,
        controller_contract=Path(controller_contract),
        expected_task=expected_task,
        config=config,
        checkpoint=checkpoint,
    )
    if source_binding_after != source_binding_before:
        raise VisualLatchAuditInputError(
            "dataset, controller contract, visual profile, or implementation changed "
            "during the offline replay"
        )
    source_binding = source_binding_before
    tip_probe_passed = sum(
        bool(report["trace_envelope_pass"])
        for report in tip_probe_episode_reports
    )
    tip_probe_takeovers = sum(
        report["takeover_confirmation_frame"] is not None
        for report in tip_probe_episode_reports
    )
    tip_probe_outcomes = sum(
        report["outcome_frame"] is not None
        for report in tip_probe_episode_reports
    )
    if tip_probe_open_only:
        tip_probe_budget_passed = 0
        tip_probe_descents: list[float] = []
        tip_probe_required_steps: list[int] = []
    else:
        tip_probe_budget_passed = sum(
            bool(report["checks"]["within_recorded_base_z_budget"])
            for report in tip_probe_episode_reports
        )
        tip_probe_descents = [
            float(report["required_recorded_base_z_descent_m"])
            for report in tip_probe_episode_reports
            if report["required_recorded_base_z_descent_m"] is not None
        ]
        tip_probe_required_steps = [
            int(report["required_descent_steps"])
            for report in tip_probe_episode_reports
            if report["required_descent_steps"] is not None
        ]
    tip_probe_summary = {
        "purpose": "bounded_no-close_first_probe",
        "allow_close": False,
        "close_authorized": False,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "demonstration_episode_count": total,
        "takeover_confirmed_episodes": tip_probe_takeovers,
        "probe_outcome_confirmed_episodes": tip_probe_outcomes,
        "recorded_base_z_budget_passed_episodes": tip_probe_budget_passed,
        "trace_envelope_passed_episodes": tip_probe_passed,
        "required_recorded_base_z_descent_m": _number_stats(
            tip_probe_descents
        ),
        "required_descent_steps": _number_stats(tip_probe_required_steps),
        "configured_max_total_descent_m": tip_probe_config.max_total_descent_m,
        "configured_descent_step_m": tip_probe_config.descent_step_m,
        "configured_maximum_descent_steps": (
            0
            if tip_probe_open_only
            else int(
                math.ceil(
                    tip_probe_config.max_total_descent_m
                    / tip_probe_config.descent_step_m
                )
            )
        ),
        "negative_control_pass": tip_probe_negative_control[
            "negative_control_pass"
        ],
        "negative_control": tip_probe_negative_control,
        "limitation": (
            "The demonstration replay is a recorded trace envelope and the old "
            "ACT is a fail-closed negative control. Neither contains camera "
            "observations caused by the counterfactual commands, so hardware "
            "completion and closed-loop convergence remain unvalidated."
        ),
    }
    recorded_closed_capture_observed = sum(
        bool(report.get("recorded_closed_capture_observed", False))
        for report in tip_probe_episode_reports
    )
    if tip_probe_open_only:
        axis_authority = _open_only_axis_authority()
        tip_probe_summary = {
            "purpose": "axis_partitioned_closed_capture_open_only_probe",
            "axis_authority": axis_authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "demonstration_episode_count": total,
            "demonstration_envelope_checked_episodes": tip_probe_passed,
            "demonstration_validation_role": (
                "observational_validation_not_deployment_positive_control"
            ),
            "recorded_closed_capture_observed_episodes": (
                recorded_closed_capture_observed
            ),
            "deployment_positive_control_source": (
                "pinned_historical_act_v1_eight_frame_hardware_window"
            ),
            "configured_max_total_descent_m": 0.0,
            "configured_descent_step_m": 0.0,
            "configured_seeking_timeout_ms": (
                tip_probe_config.seeking_timeout_ms
            ),
            "negative_control_pass": tip_probe_negative_control[
                "negative_control_pass"
            ],
            "negative_control": tip_probe_negative_control,
            "limitation": (
                f"The {total} demonstration "
                f"{'episode is' if total == 1 else 'episodes are'} visual "
                "envelopes only. "
                "The pinned "
                "historical ACT exposes real pre-fix jaw and TCP behavior and "
                "drives a schema-v6 arm6 axis-authority replay. Pre-takeover policy "
                "numeric state/action contain J1-J6 only; jaw pixels may remain "
                "visible to the VLM but there is no numeric policy jaw input or "
                "output. Pre-takeover policy arm motion and TCP descent remain allowed "
                "and are not described "
                "as whole-pipeline zero motion. Hardware response to the "
                "counterfactual open command remains unvalidated."
            ),
        }
        tip_profile_semantics = (
            "policy_arm_before_takeover_tip_capture_hold_then_arm_hold_tip_open"
        )
        tip_profile_authority = {
            "axis_authority": axis_authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
        summary_tip = {
            "demonstration_envelope_checked_episodes": tip_probe_passed,
            "recorded_closed_capture_observed_episodes": (
                recorded_closed_capture_observed
            ),
            "negative_control_pass": tip_probe_negative_control[
                "negative_control_pass"
            ],
            **tip_profile_authority,
        }
        top_tip_authority = {"axis_authority": axis_authority}
    else:
        tip_profile_semantics = "bounded_no-close_recorded_trace_envelope"
        tip_profile_authority = {
            "allow_close": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
        summary_tip = {
            "trace_envelope_passed_episodes": tip_probe_passed,
            "recorded_base_z_budget_passed_episodes": tip_probe_budget_passed,
            "negative_control_pass": tip_probe_negative_control[
                "negative_control_pass"
            ],
            "allow_close": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
        top_tip_authority = {}


    return {
        "schema_version": _audit_schema_version(config),
        "kind": "firefly_y6_fullcycle_v4_visual_latch_replay",
        "read_only": True,
        "safe_for_deployment": safe,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        **top_tip_authority,
        "dataset": {
            "root": str(root),
            "repo_id": dataset_repo_id,
            "episodes": total,
            "frames": sum(len(rows) for rows in frames.values()),
            "fps": DEFAULT_FPS,
            "wrist_video_key": WRIST_VIDEO_KEY,
            "recorded_rotation_degrees": 180,
            "offline_rotation_applied": False,
        },
        "pseudo_labels": {
            "source": "four canonical action[6] threshold crossings with hysteresis",
            "jaw_crossing_rad": jaw_crossing_rad,
            "jaw_open_confirm_rad": jaw_open_confirm_rad,
            "jaw_closed_confirm_rad": jaw_closed_confirm_rad,
            "ordered_crossings": [
                "pregrasp_open (rising)",
                "pickup_close (falling)",
                "release_open (rising)",
                "return_close (falling)",
            ],
            "limitation": (
                "No explicit phase labels exist; pre-transport means strictly before "
                "pickup-close, where transport is physically impossible. Shallow "
                "threshold chatter is not treated as a phase transition."
            ),
        },
        "profile": {
            "windows": asdict(windows),
            "detector_config": _detector_config_from_source_binding(
                source_binding
            ),
            "close_confirmation_frames": config.close_confirm_frames,
            "blue_confirmation_frames": config.box_confirm_frames,
            "runtime_profile": {
                "policy_close_intent_max_rad": (
                    config.pickup_policy_close_intent_max_rad
                ),
                "policy_close_intent_timeout_ms": (
                    config.pickup_policy_close_intent_hold_ms
                ),
                "held_visual_loss": (
                    "closed jaw; no phase/release advancement; fault after "
                    "held_red_loss_fault_ms"
                ),
                "release_gate": "fresh held-red AND blue",
                "source_binding_checked": True,
            },
            "state_machine_replay": {
                "implementation": (
                    "VisualGraspReleaseLatch.step + compose_visual_action"
                ),
                "fk_urdf": fk_binding,
                "required_milestones": list(STATE_MACHINE_MILESTONES),
            },
            "tip_probe": {
                "config": asdict(tip_probe_config),
                "semantics": tip_profile_semantics,
                **tip_profile_authority,
            },
        },
        "source_binding": source_binding,
        "tip_probe": tip_probe_summary,
        "summary": {
            "verdict": "passed" if safe else "failed",
            "episode_count": total,
            "error_count": len(errors),
            "close_detected_episodes": close_detected,
            "close_detection_rate": close_detected / total,
            "close_confirmation_offset_frames": _number_stats(close_offsets),
            "runtime_close_latch_offset_frames": _number_stats(
                runtime_close_offsets
            ),
            "policy_close_intent_offset_frames": _number_stats(
                policy_close_intent_offsets
            ),
            "runtime_close_before_pickup_crossing_episodes": (
                runtime_close_before_pickup_episodes
            ),
            "premature_close_confirmation_episodes": premature_close_episodes,
            "contact_confirmed_episodes": contact_confirmed_episodes,
            "contact_confirm_offset_frames": _number_stats(contact_offsets),
            "held_red_timeout_safe_episodes": held_safe,
            "held_red_timeout_safe_rate": held_safe / total,
            "held_red_coverage": {
                "min": min(held_coverages),
                "median": statistics.median(held_coverages),
                "max": max(held_coverages),
            },
            "premature_pretransport_blue_confirmation_episodes": premature_blue_episodes,
            "release_window_detected_episodes": release_detected,
            "release_window_detection_rate": release_detected / total,
            "release_confirmation_offset_frames": _number_stats(release_offsets),
            "state_machine_ordered_complete_episodes": (
                state_machine_ordered_complete_episodes
            ),
            "state_machine_fault_free_episodes": (
                state_machine_fault_free_episodes
            ),
            "tip_probe": summary_tip,
            "state_machine_grasp_latched_offset_frames": _number_stats(
                state_machine_grasp_latch_offsets
            ),
            "state_machine_grasp_confirmed_offset_frames": _number_stats(
                state_machine_grasp_confirm_offsets
            ),
            "state_machine_release_unlatched_offset_frames": _number_stats(
                state_machine_release_unlatch_offsets
            ),
            "state_machine_release_confirmed_offset_frames": _number_stats(
                state_machine_release_confirm_offsets
            ),
            "state_machine_home_closing_offset_frames": _number_stats(
                state_machine_home_close_offsets
            ),
            "state_machine_task_done_offset_frames": _number_stats(
                state_machine_done_offsets
            ),
            "rejection_reason_counts": {
                "close_window": dict(sorted(close_rejections.items())),
                "held_interval": dict(sorted(held_rejections.items())),
                "release_window": dict(sorted(blue_rejections.items())),
            },
            "red_contour_diagnostics": {
                "definition": (
                    "Contours after configured red HSV, morphology, and excluded "
                    "ROIs; ambiguity candidates meet "
                    "red_ambiguity_area_fraction_min. bbox_norm is left/top/width/height."
                ),
                "ambiguity_area_fraction_min": (
                    config.red_ambiguity_area_fraction_min
                ),
                "close_window": _contour_window_summary(
                    close_contour_diagnostics
                ),
                "held_interval": _contour_window_summary(
                    held_contour_diagnostics
                ),
            },
            "errors": errors,
        },
        "episodes": episode_reports,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument(
        "--controller-contract", type=Path, default=DEFAULT_CONTROLLER_CONTRACT
    )
    parser.add_argument("--expected-task", default=DEFAULT_EXPECTED_TASK)
    parser.add_argument("--arm-urdf", type=Path, default=DEFAULT_ARM_URDF_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Required for schema-6 open_only: the explicit attested 020000 "
            "arm6_seeking pretrained_model directory. It is verified read-only."
        ),
    )
    parser.add_argument(
        "--tip-probe-negative-control",
        type=Path,
        default=DEFAULT_TIP_PROBE_NEGATIVE_CONTROL,
        help=(
            "Historical ACT JSONL used only for a pinned visual-boundary and "
            "axis-authority replay; it is never treated as a counterfactual "
            "hardware rollout."
        ),
    )
    parser.add_argument(
        "--tip-probe-mode",
        choices=tuple(mode.value for mode in TipServoMode),
        default=TipServoMode.DISABLED.value,
        help=(
            "Exact read-only provenance value for the audited deployment "
            "profile. Use execute only when generating the frozen report for "
            "an execute-mode first probe; this flag never commands hardware."
        ),
    )
    parser.add_argument(
        "--tip-probe-profile",
        choices=tuple(profile.value for profile in TipServoProbeProfile),
        default=TipServoProbeProfile.BOUNDED_DESCENT.value,
        help=(
            "Exact read-only tip-probe profile provenance. open_only selects "
            "the pinned schema-6 arm6 axis-authority audit and never commands hardware."
        ),
    )
    parser.add_argument("--jaw-crossing-rad", type=float, default=DEFAULT_JAW_CROSSING_RAD)
    parser.add_argument(
        "--red-ambiguity-area-fraction-min",
        type=float,
        help="Offline threshold override; the exact value is recorded in the report.",
    )
    parser.add_argument("--output", type=Path, help="Also write the strict JSON report here.")
    return parser.parse_args(argv)


def _strict_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tip_probe_mode = TipServoMode(args.tip_probe_mode)
    tip_probe_profile = TipServoProbeProfile(args.tip_probe_profile)
    output_schema_version = (
        TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION
        if tip_probe_profile is TipServoProbeProfile.OPEN_ONLY
        else (
            TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION
            if tip_probe_mode is TipServoMode.EXECUTE
            else VISUAL_LATCH_AUDIT_SCHEMA_VERSION
        )
    )
    try:
        tip_config = (
            _open_only_tip_servo_config()
            if tip_probe_profile is TipServoProbeProfile.OPEN_ONLY
            else TipServoConfig(mode=tip_probe_mode)
        )
        if tip_config.mode is not tip_probe_mode:
            tip_config = replace(tip_config, mode=tip_probe_mode)
        config = VisualPhaseLatchConfig(enabled=True, tip_servo=tip_config)
        negative_control_path = args.tip_probe_negative_control
        if (
            tip_probe_profile is TipServoProbeProfile.OPEN_ONLY
            and negative_control_path == DEFAULT_TIP_PROBE_NEGATIVE_CONTROL
        ):
            negative_control_path = (
                DEFAULT_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL
            )
        if args.red_ambiguity_area_fraction_min is not None:
            config = replace(
                config,
                red_ambiguity_area_fraction_min=(
                    args.red_ambiguity_area_fraction_min
                ),
            )
        report = audit_firefly_y6_visual_latch(
            args.root,
            expected_episodes=args.expected_episodes,
            dataset_repo_id=args.dataset_repo_id,
            controller_contract=args.controller_contract,
            expected_task=args.expected_task,
            arm_urdf_path=args.arm_urdf,
            jaw_crossing_rad=args.jaw_crossing_rad,
            config=config,
            tip_probe_negative_control_path=negative_control_path,
            checkpoint=args.checkpoint,
        )
        output = _strict_json(report)
    except (VisualLatchAuditInputError, ValueError) as exc:
        report = {
            "schema_version": output_schema_version,
            "kind": "firefly_y6_fullcycle_v4_visual_latch_replay",
            "read_only": True,
            "safe_for_deployment": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "summary": {
                "verdict": "error",
                "error_count": 1,
                "errors": [{"code": "audit_input_error", "message": str(exc)}],
            },
        }
        output = _strict_json(report)
        print(output, end="")
        return 2
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        except OSError as exc:
            print(f"cannot write audit report {args.output}: {exc}", file=sys.stderr)
            return 2
    print(output, end="")
    return 0 if report["safe_for_deployment"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
