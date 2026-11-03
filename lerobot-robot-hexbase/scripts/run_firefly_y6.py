#!/usr/bin/env python
"""Run the Firefly Y6 deployment ladder with explicit hardware safety gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
import random
import logging
import math
import signal
import sys
import time
from pathlib import Path

import draccus
import numpy as np
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from lerobot_robot_hexbase.firefly_deployment import (
    FireflyDeploymentConfig,
    GraspPhaseCorrector,
    InferenceWorker,
    StageVideoRecorder,
    adapt_action_chunk_slew_rate,
    action_dict,
    assess_act_start_pose,
    capture_deployment_sources,
    hold_action,
    invalidate_stage_reviews,
    interpolate_actions,
    mark_stage_reviewed,
    require_acceptance_order,
    require_arm6_policy_stage_gate,
    require_arm_hold_timeout_override_gate,
    require_hardware_safety_gate,
    require_tip_servo_execute_probe_gate,
    require_tip_servo_open_only_probe_gate,
    require_unchanged_deployment_sources,
    robot_config_for_stage,
    stage_run_label,
    validate_replay_start,
    visual_phase_latch_implementation_manifest,
    visual_phase_latch_provenance,
)
from lerobot_robot_hexbase.hex_zenoh_follower import HexZenohFollower, SafetyStopError
from lerobot_robot_hexbase.firefly_arm_only_policy import (
    ARM6_SEEKING_ROLE,
    ArmOnlyPolicyRole,
    expand_arm6_action_chunk,
    extract_arm6_observation,
    resolve_arm6_takeover_boundary,
    validate_arm6_action_chunk,
)
from lerobot_robot_hexbase.firefly_visual_latch import (
    BlobEvidence,
    D405VisualDetector,
    FrameStatus,
    LatchDecision,
    VisualEvidence,
    VisualGraspReleaseLatch,
    VisualLatchPhase,
    compose_visual_action,
)
from lerobot_robot_hexbase.firefly_tip_servo import (
    DlsIkConfig,
    TipServoDecision,
    TipServoError,
    TipServoFrameStatus,
    TipServoIntent,
    TipServoMode,
    TipServoObservation,
    TipServoPhase,
    TipServoProbeProfile,
    TipServoState,
    advance_tip_servo,
    resolve_open_only_jaw_target,
)
from lerobot_robot_hexbase.smolvlm_bundle import validate_materialized_smolvlm_bundle


_FOLLOWER_STATE_FEATURES = tuple(
    [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
)
_POLICY_IMAGE_FEATURES = ("observation.images.scene", "observation.images.wrist")
_LEGACY_ACTION7_ROLE = "legacy_action7"
_ARM6_CAPTURE_JAW_RANGE_RAD = (0.285, 0.325)
_ACT_START_RESET_MAX_FRAMES = 120
_ACT_START_RESET_SETTLE_FRAMES = 30
_ACT_START_RESET_REQUIRED_PASS_FRAMES = 3
_ACT_START_RESET_MAX_TRAVEL_RAD = (0.20, 0.30, 0.25, 0.36, 0.15, 0.30, 0.10)
_ACT_START_RESET_CANDIDATE_LEAD_BUDGET_RAD = (
    0.013,
    0.025,
    0.012,
    0.025,
    0.013,
    0.013,
    0.013,
)
_ACT_START_RESET_WAYPOINT_STEP_CAP_RAD = (
    0.003,
    0.003,
    0.002,
    0.003,
    0.003,
    0.003,
    0.003,
)
_ACT_START_RESET_STALL_FRAMES = 90
_ACT_START_RESET_MAX_CONTROL_FRAMES = 900
_ACT_START_EE_PREOPEN_MAX_TRAVEL_RAD = 0.080
_ACT_START_EE_PREOPEN_MAX_FRAMES = 27
_ACT_START_EE_PREOPEN_SETTLE_FRAMES = 30
_ACT_START_EE_PREOPEN_REQUIRED_PASS_FRAMES = 3
_VISUAL_REPLAN_EVENTS = frozenset(
    {
        "grasp_latched",
        "grasp_confirmed",
        "lift_confirmed",
        "transport_confirmed",
        "box_aligned",
        "release_unlatched",
        "release_confirmed",
        "home_closing_started",
        "task_done",
        "fault_hold",
    }
)
_VISUAL_TRANSITION_EVENT = "visual_phase_transition"
_J23_TARGET_INDICES = (1, 2)
_J23_TARGET_STEP_LIMITS_RAD = (0.012, 0.015)
_J23_TARGET_DELTA_STEP_LIMIT_RAD = 0.005
_J23_TARGET_EPSILON_RAD = 1e-12


class _PreHardwareInterrupt(RuntimeError):
    """Abort immediately when a signal arrives before hardware is connected."""


@dataclass(frozen=True, slots=True)
class PolicyRuntimeContract:
    """Checkpoint schema cross-check for an explicitly authorized runtime role."""

    policy_type: str
    runtime_role: str
    observation_state_dimension: int
    policy_action_dimension: int


@dataclass(frozen=True, slots=True)
class Arm6SeekingRuntime:
    """Run-local immutable jaw authority bound before the first inference."""

    contract: PolicyRuntimeContract
    startup_accepted_reference7: tuple[float, ...]
    capture_jaw_target_rad: float
    capture_jaw_range_rad: tuple[float, float]

    def __post_init__(self) -> None:
        expected_contract = PolicyRuntimeContract(
            policy_type="smolvla",
            runtime_role=ARM6_SEEKING_ROLE,
            observation_state_dimension=6,
            policy_action_dimension=6,
        )
        if self.contract != expected_contract:
            raise ValueError("arm6 runtime contract is inconsistent")
        if (
            type(self.capture_jaw_range_rad) is not tuple
            or self.capture_jaw_range_rad != _ARM6_CAPTURE_JAW_RANGE_RAD
        ):
            raise ValueError(
                "arm6 runtime capture interval must be exactly "
                f"{_ARM6_CAPTURE_JAW_RANGE_RAD}"
            )
        if type(self.startup_accepted_reference7) is not tuple:
            raise ValueError("arm6 startup accepted reference must be immutable")
        reference = _finite_reference7(self.startup_accepted_reference7)
        capture = self.capture_jaw_target_rad
        if type(capture) is not float:
            raise ValueError("arm6 runtime capture jaw must be finite")
        if not math.isfinite(capture):
            raise ValueError("arm6 runtime capture jaw must be finite")
        if not math.isclose(
            capture,
            reference[6],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "arm6 runtime capture jaw differs from its startup reference"
            )
        lower, upper = _ARM6_CAPTURE_JAW_RANGE_RAD
        if not lower <= capture <= upper:
            raise ValueError("arm6 runtime capture jaw is outside its fixed interval")

    def telemetry(self) -> dict[str, object]:
        return {
            "policy_role": self.contract.runtime_role,
            "observation_state_dimension": self.contract.observation_state_dimension,
            "raw_policy_action_dimension": self.contract.policy_action_dimension,
            "candidate_action_dimension": 7,
            "jaw_candidate_source": "startup_accepted_reference",
            "capture_jaw_target_rad": self.capture_jaw_target_rad,
        }


@dataclass(frozen=True)
class VisualControlCycle:
    """One fail-closed visual decision composed onto a policy/hold action."""

    action: tuple[float, ...]
    decision: LatchDecision
    telemetry: dict[str, object]
    requires_replan: bool
    replan_reason: str | None = None
    discard_policy_chunk: bool = False
    smoother_reset_required: bool = False
    arm_owner: str = "policy"
    jaw_owner: str = "policy"


class VisualActionSource(StrEnum):
    """Identify who owns a visual-runtime arm command this control cycle."""

    FRESH_POLICY = "fresh_policy"
    INFERENCE_HOLD = "inference_hold"
    TIP_SERVO = "tip_servo"


class J23TargetSmoother:
    """Bound J2/J3 command velocity and direct reversals at 30 Hz.

    The limiter is deliberately downstream of the visual action composer, so
    it sees the exact arm target that would otherwise be published. A safety
    HOLD always wins: it publishes the accepted reference exactly and clears
    the stored step. Likewise, a newly requested direction cannot replace a
    non-zero opposite step in one cycle; one exact-reference cycle separates
    the directions.

    A causal limiter cannot simultaneously preserve a hard acceleration bound
    and avoid overshooting an arbitrarily moving target. When the remaining
    target distance is shorter than the acceleration-feasible step, this class
    clamps to the target and records an urgent-deceleration exception rather
    than moving past it. Normal tracking cycles retain the exact 0.005-rad
    delta-step bound.
    """

    def __init__(self) -> None:
        self._previous_steps = np.zeros(2, dtype=np.float64)

    def apply(
        self,
        desired_action,
        command_reference,
        *,
        forced_reset_reason: str | None = None,
    ) -> tuple[tuple[float, ...], dict[str, object]]:
        try:
            desired = np.asarray(desired_action, dtype=np.float64)
            reference = np.asarray(command_reference, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "J2/J3 target smoothing requires finite 7-vectors"
            ) from exc
        if (
            desired.shape != (7,)
            or reference.shape != (7,)
            or not np.isfinite(desired).all()
            or not np.isfinite(reference).all()
        ):
            raise ValueError("J2/J3 target smoothing requires finite 7-vectors")
        if forced_reset_reason not in {None, "nonfresh_hold", "visual_arm_hold"}:
            raise ValueError("unknown J2/J3 target smoothing reset reason")

        output = desired.copy()
        previous_steps = self._previous_steps.copy()
        actual_steps = np.zeros(2, dtype=np.float64)
        reasons: list[str] = ["passthrough", "passthrough"]
        reset_reasons: list[str | None] = [None, None]
        urgent_deceleration = [False, False]

        if forced_reset_reason is not None:
            for local_index, joint_index in enumerate(_J23_TARGET_INDICES):
                output[joint_index] = reference[joint_index]
                reasons[local_index] = forced_reset_reason
                reset_reasons[local_index] = forced_reset_reason
                urgent_deceleration[local_index] = bool(
                    abs(previous_steps[local_index])
                    > _J23_TARGET_DELTA_STEP_LIMIT_RAD
                    + _J23_TARGET_EPSILON_RAD
                )
            self._previous_steps.fill(0.0)
        else:
            for local_index, (joint_index, step_limit) in enumerate(
                zip(
                    _J23_TARGET_INDICES,
                    _J23_TARGET_STEP_LIMITS_RAD,
                    strict=True,
                )
            ):
                error = float(desired[joint_index] - reference[joint_index])
                previous_step = float(previous_steps[local_index])
                reset_reason: str | None = None

                if abs(error) <= _J23_TARGET_EPSILON_RAD:
                    step = 0.0
                    reset_reason = "target_reached"
                elif previous_step * error < 0.0:
                    # Never jump directly from one direction into the other.
                    step = 0.0
                    reset_reason = "target_reversal"
                else:
                    lower = max(
                        -step_limit,
                        previous_step - _J23_TARGET_DELTA_STEP_LIMIT_RAD,
                    )
                    upper = min(
                        step_limit,
                        previous_step + _J23_TARGET_DELTA_STEP_LIMIT_RAD,
                    )
                    step = float(np.clip(error, lower, upper))

                    # Acceleration feasibility can demand a step longer than
                    # the remaining distance. Target monotonicity is the
                    # safety invariant, so stop at the target and audit the
                    # exceptional deceleration instead of overshooting it.
                    if step * error < 0.0 or abs(step) > abs(error):
                        step = error
                        reset_reason = "target_reached"

                if reset_reason is not None:
                    reasons[local_index] = reset_reason
                    reset_reasons[local_index] = reset_reason
                    urgent_deceleration[local_index] = bool(
                        abs(step - previous_step)
                        > _J23_TARGET_DELTA_STEP_LIMIT_RAD
                        + _J23_TARGET_EPSILON_RAD
                    )
                    next_previous_step = 0.0
                else:
                    delta_step = abs(step - previous_step)
                    if (
                        abs(step) >= step_limit - _J23_TARGET_EPSILON_RAD
                        and abs(error) > abs(step) + _J23_TARGET_EPSILON_RAD
                    ):
                        reasons[local_index] = "step_limited"
                    elif (
                        delta_step
                        >= _J23_TARGET_DELTA_STEP_LIMIT_RAD
                        - _J23_TARGET_EPSILON_RAD
                        and abs(error - step) > _J23_TARGET_EPSILON_RAD
                    ):
                        reasons[local_index] = "delta_step_limited"
                    next_previous_step = step

                output[joint_index] = reference[joint_index] + step
                actual_steps[local_index] = step
                self._previous_steps[local_index] = next_previous_step

        limited = [
            bool(
                abs(output[joint_index] - desired[joint_index])
                > _J23_TARGET_EPSILON_RAD
            )
            for joint_index in _J23_TARGET_INDICES
        ]
        telemetry: dict[str, object] = {
            "raw": [float(desired[index]) for index in _J23_TARGET_INDICES],
            "smoothed": [float(output[index]) for index in _J23_TARGET_INDICES],
            "previous_step": [float(value) for value in previous_steps],
            "step": [float(value) for value in actual_steps],
            "reason": reasons,
            "reset_reason": reset_reasons,
            "limited": [bool(value) for value in limited],
            "step_limits_rad": [
                float(value) for value in _J23_TARGET_STEP_LIMITS_RAD
            ],
            "delta_step_limit_rad": float(_J23_TARGET_DELTA_STEP_LIMIT_RAD),
            "urgent_deceleration_exception": [
                bool(value) for value in urgent_deceleration
            ],
            "urgent_deceleration_exception_count": int(
                sum(urgent_deceleration)
            ),
        }
        return tuple(float(value) for value in output), telemetry


class VisualLatchRuntime:
    """Bind the pure visual latch to one follower observation at 30 Hz.

    The wrist image and its side-channel metadata are sampled by the caller in
    the same thread. FK always uses measured J1--J6 from that observation. This
    adapter never rotates the already-correct D405 image. Its visual state
    machine changes J1--J6 only for an explicit arm HOLD; final J2/J3 target
    smoothing is applied separately at the publish boundary.
    """

    def __init__(self, config, *, home_center_rad, home_tolerance_rad):
        if not config.enabled:
            raise ValueError("VisualLatchRuntime requires an enabled visual latch")
        self.config = config
        self.detector = D405VisualDetector(config)
        self.latch = VisualGraspReleaseLatch(config)
        self.tip_config = config.tip_servo
        self.tip_state = TipServoState()
        self._tip_ik_config: DlsIkConfig | None = None
        self._last_tip_decision: TipServoDecision | None = None
        self.replan_epoch = 0
        self._last_tcp_position: np.ndarray | None = None
        self._last_tcp_sample_ns: int | None = None
        self._last_measured: np.ndarray | None = None
        self._last_measured_ns: int | None = None
        self._last_arm_hold: bool | None = None
        self._pickup_policy_close_intent_until_ns: int | None = None
        self.j23_target_smoother = J23TargetSmoother()
        self._start_envelope_center = self._validated_home_vector(
            home_center_rad, "home_center_rad"
        )
        self._start_envelope_tolerance = self._validated_home_vector(
            home_tolerance_rad, "home_tolerance_rad", positive=True
        )
        self._home_center: np.ndarray | None = None

    @staticmethod
    def _validated_home_vector(values, name: str, *, positive: bool = False):
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"visual latch {name} must contain seven finite values"
            ) from exc
        if vector.shape != (7,) or not np.isfinite(vector).all():
            raise ValueError(f"visual latch {name} must contain seven finite values")
        if positive and np.any(vector <= 0):
            raise ValueError(f"visual latch {name} must be positive")
        return vector

    @property
    def phase(self) -> VisualLatchPhase:
        return self.latch.phase

    @property
    def tip_mode(self) -> TipServoMode:
        return self.tip_config.mode

    @property
    def tip_exclusive_active(self) -> bool:
        return bool(
            self.tip_mode is TipServoMode.EXECUTE
            and self.tip_state.phase
            not in {TipServoPhase.SEEKING, TipServoPhase.FAULT_HOLD}
        )

    @property
    def tip_probe_complete(self) -> bool:
        return self.tip_state.phase is TipServoPhase.PROBE_COMPLETE

    def _tip_dls_config(self, robot: HexZenohFollower) -> DlsIkConfig:
        if self._tip_ik_config is not None:
            return self._tip_ik_config
        arm_limits, _ee_limits = robot.controller_limits
        defaults = DlsIkConfig()
        configured_steps = tuple(
            float(value) for value in robot.config.resolved_arm_step_limits_rad()
        )
        velocity_steps = tuple(
            float(value) / float(robot.config.command_hz)
            for value in arm_limits.vel_max
        )
        step_limits = tuple(
            min(default, configured, velocity)
            for default, configured, velocity in zip(
                defaults.joint_step_limit_rad,
                configured_steps,
                velocity_steps,
                strict=True,
            )
        )
        self._tip_ik_config = replace(
            defaults,
            joint_lower_rad=arm_limits.pos_min,
            joint_upper_rad=arm_limits.pos_max,
            joint_step_limit_rad=step_limits,
        )
        return self._tip_ik_config

    def _advance_tip_planner(
        self,
        robot: HexZenohFollower,
        evidence: VisualEvidence,
        measured: np.ndarray,
        *,
        arm_reference: np.ndarray,
        arm_speed_rad_s: float | None,
        jaw_speed_rad_s: float | None,
        action_source: VisualActionSource,
    ) -> tuple[TipServoDecision | None, str | None]:
        mode = self.tip_mode
        if mode is TipServoMode.DISABLED:
            return None, None
        if (
            mode is TipServoMode.EXECUTE
            and self.tip_state.phase is not TipServoPhase.SEEKING
            and action_source is not VisualActionSource.TIP_SERVO
        ):
            reason = "tip_servo_action_source_violation"
            self.tip_state = replace(
                self.tip_state,
                phase=TipServoPhase.FAULT_HOLD,
                last_evaluated_at_ns=evidence.evaluated_at_ns,
                fault_reason=reason,
            )
            return None, reason
        try:
            frame_status = TipServoFrameStatus(evidence.frame_status.value)
            observation = TipServoObservation(
                frame_id=max(0, int(evidence.frame_id)),
                evaluated_at_ns=int(evidence.evaluated_at_ns),
                frame_status=frame_status,
                candidate_count=int(evidence.red.candidate_count),
                red_centroid_norm=evidence.red.centroid_norm,
                red_area_fraction=(
                    None
                    if evidence.red.centroid_norm is None
                    else float(evidence.red.contour_area_fraction)
                ),
                arm_q_rad=tuple(float(value) for value in measured[:6]),
                arm_reference_q_rad=tuple(
                    float(value) for value in arm_reference[:6]
                ),
                jaw_reference_rad=float(arm_reference[6]),
                jaw_rad=float(measured[6]),
                arm_speed_rad_s=(
                    self.tip_config.max_arm_speed_rad_s + 1.0
                    if arm_speed_rad_s is None
                    else float(arm_speed_rad_s)
                ),
                jaw_speed_rad_s=(
                    self.tip_config.max_jaw_speed_rad_s + 1.0
                    if jaw_speed_rad_s is None
                    else float(jaw_speed_rad_s)
                ),
            )
            next_state, tip_decision = advance_tip_servo(
                self.tip_state,
                observation,
                fk=lambda arm_q: robot.get_tcp_pose(arm_q),
                config=self.tip_config,
                ik_config=self._tip_dls_config(robot),
                takeover_authorized=(
                    mode is not TipServoMode.EXECUTE
                    or action_source is VisualActionSource.FRESH_POLICY
                ),
            )
        except Exception as exc:
            reason = f"tip_servo_planner_{type(exc).__name__}"
            if mode is TipServoMode.ADVISORY:
                return None, f"advisory:{reason}"
            self.tip_state = replace(
                self.tip_state,
                phase=TipServoPhase.FAULT_HOLD,
                last_evaluated_at_ns=evidence.evaluated_at_ns,
                fault_reason=reason,
            )
            return None, reason
        self.tip_state = next_state
        self._last_tip_decision = tip_decision
        return (
            tip_decision,
            (
                tip_decision.fault_reason
                if tip_decision.phase_after is TipServoPhase.FAULT_HOLD
                else None
            ),
        )

    def compose_cycle(
        self,
        robot: HexZenohFollower,
        observation: dict,
        policy_action,
        command_reference,
        *,
        action_source: VisualActionSource = VisualActionSource.FRESH_POLICY,
        arm6_runtime: Arm6SeekingRuntime | None = None,
    ) -> VisualControlCycle:
        feature_names = tuple(robot.action_features)
        measured = np.asarray(
            [float(observation[name]) for name in feature_names], dtype=np.float64
        )
        reference = np.asarray(command_reference, dtype=np.float64)
        if measured.shape != (7,) or not np.isfinite(measured).all():
            raise ValueError("visual latch measured state must be a finite 7-vector")
        if reference.shape != (7,) or not np.isfinite(reference).all():
            raise ValueError("visual latch command reference must be a finite 7-vector")
        try:
            policy = np.asarray(policy_action, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "visual latch policy action must be a finite 7-vector"
            ) from exc
        if policy.shape != (7,) or not np.isfinite(policy).all():
            raise ValueError("visual latch policy action must be a finite 7-vector")
        if not isinstance(action_source, VisualActionSource):
            raise TypeError("action_source must be a VisualActionSource")
        if arm6_runtime is not None:
            if not isinstance(arm6_runtime, Arm6SeekingRuntime):
                raise TypeError("arm6_runtime must be an Arm6SeekingRuntime")
            if arm6_runtime.contract.runtime_role != ARM6_SEEKING_ROLE:
                raise ValueError("arm6_runtime does not carry arm6_seeking authority")
            if (
                action_source is not VisualActionSource.TIP_SERVO
                and not math.isclose(
                    float(policy[6]),
                    arm6_runtime.capture_jaw_target_rad,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "arm6 candidate jaw differs from the immutable startup capture jaw"
                )
        policy_has_jaw_authority = arm6_runtime is None
        policy_action_is_fresh = action_source is VisualActionSource.FRESH_POLICY
        if self._home_center is None:
            start_error = np.abs(
                measured[:6] - self._start_envelope_center[:6]
            )
            if np.any(start_error > self._start_envelope_tolerance[:6]):
                failed = [
                    index + 1
                    for index, value in enumerate(
                        start_error > self._start_envelope_tolerance[:6]
                    )
                    if bool(value)
                ]
                raise ValueError(
                    "visual latch cannot bind the run-start home reference; "
                    f"measured start is outside the demonstrated envelope at {failed}"
                )
            # The task says “return to its initial pose”, so bind this run's
            # measured J1--J6 rather than the broad 50-demo population center.
            self._home_center = measured.copy()

        now_ns = time.perf_counter_ns()
        pregrasp_open_ready = bool(
            measured[6] >= self.config.pregrasp_open_ready_rad
            or reference[6] >= self.config.pregrasp_open_ready_rad
            or self.latch.pregrasp_open_seen
        )
        if not pregrasp_open_ready:
            # The policy begins every demonstration with a closed jaw.  That
            # initial state is not pickup intent and must never survive into
            # the first open-jaw cycle through the bounded intent cache.
            self._pickup_policy_close_intent_until_ns = None
        pickup_policy_close_intent_observed = bool(
            policy_has_jaw_authority
            and pregrasp_open_ready
            and policy_action_is_fresh
            and policy[6] <= self.config.pickup_policy_close_intent_max_rad
        )
        if pickup_policy_close_intent_observed:
            hold_ns = int(
                self.config.pickup_policy_close_intent_hold_ms * 1_000_000.0
            )
            self._pickup_policy_close_intent_until_ns = now_ns + hold_ns
        intent_until_ns = self._pickup_policy_close_intent_until_ns
        pickup_policy_close_intent = bool(
            intent_until_ns is not None and now_ns < intent_until_ns
        )
        pickup_policy_close_intent_remaining_ms = (
            0.0
            if not pickup_policy_close_intent
            else (intent_until_ns - now_ns) / 1_000_000.0
        )
        if not pickup_policy_close_intent:
            self._pickup_policy_close_intent_until_ns = None
        metadata = robot.get_camera_frame_metadata().get("wrist")
        frame_id = -1
        captured_at_ns = -1
        metadata_fault: str | None = None
        if metadata is None:
            metadata_fault = "wrist_metadata_missing"
        elif metadata.sequence is None or metadata.captured_at_s is None:
            metadata_fault = "wrist_timestamp_unsupported"
        else:
            frame_id = int(metadata.sequence)
            captured_at_ns = int(
                round(float(metadata.captured_at_s) * 1_000_000_000)
            )

        evidence = self.detector.detect(
            observation.get("wrist"),
            frame_id=frame_id,
            captured_at_ns=captured_at_ns,
            now_ns=now_ns,
        )
        if evidence.frame_status in {
            FrameStatus.INVALID,
            FrameStatus.OUT_OF_ORDER,
            FrameStatus.STALE,
            FrameStatus.FUTURE,
        }:
            metadata_fault = (
                metadata_fault or f"wrist_frame_{evidence.frame_status.value}"
            )

        tcp_position: np.ndarray | None = None
        tcp_rotation: np.ndarray | None = None
        tcp_speed_m_s: float | None = None
        tcp_z_m: float | None = None
        fk_fault: str | None = None
        try:
            position, rotation = robot.get_tcp_pose(measured[:6])
            tcp_position = np.asarray(position, dtype=np.float64)
            tcp_rotation = np.asarray(rotation, dtype=np.float64)
            if tcp_position.shape != (3,) or not np.isfinite(tcp_position).all():
                raise ValueError("TCP FK returned a non-finite position")
            if tcp_rotation.shape != (3, 3) or not np.isfinite(tcp_rotation).all():
                raise ValueError("TCP FK returned a non-finite rotation")
            tcp_z_m = float(tcp_position[2])
            if (
                self._last_tcp_position is not None
                and self._last_tcp_sample_ns is not None
            ):
                elapsed_s = (
                    now_ns - self._last_tcp_sample_ns
                ) / 1_000_000_000.0
                if elapsed_s > 0:
                    tcp_speed_m_s = float(
                        np.linalg.norm(tcp_position - self._last_tcp_position)
                        / elapsed_s
                    )
        except Exception as exc:
            fk_fault = f"tcp_fk_{type(exc).__name__}"

        arm_speed_rad_s: float | None = None
        jaw_speed_rad_s: float | None = None
        if self._last_measured is not None and self._last_measured_ns is not None:
            elapsed_s = (now_ns - self._last_measured_ns) / 1_000_000_000.0
            if elapsed_s > 0:
                arm_speed_rad_s = float(
                    np.max(np.abs(measured[:6] - self._last_measured[:6]))
                    / elapsed_s
                )
                jaw_speed_rad_s = float(
                    abs(measured[6] - self._last_measured[6]) / elapsed_s
                )
        home_error = np.abs(measured[:6] - self._home_center[:6])
        home_policy_close_intent = bool(
            policy_has_jaw_authority
            and policy[6] <= self.config.home_policy_close_intent_max_rad
        )
        home_ready = bool(
            arm_speed_rad_s is not None
            and arm_speed_rad_s <= self.config.home_max_arm_speed_rad_s
            and np.all(home_error <= self.config.home_arm_tolerance_rad)
            and home_policy_close_intent
        )

        tip_takeover_reference_settled = bool(
            arm_speed_rad_s is not None
            and arm_speed_rad_s <= self.tip_config.max_arm_speed_rad_s
            and np.all(
                np.abs(measured[:6] - reference[:6])
                <= np.asarray(
                    self.tip_config.arm_target_settle_rad,
                    dtype=np.float64,
                )
            )
        )
        tip_geometry_rebound = False
        if (
            self.tip_mode is TipServoMode.EXECUTE
            and self.tip_state.phase is TipServoPhase.OPENING
            and tip_takeover_reference_settled
            and tcp_position is not None
            and tcp_rotation is not None
        ):
            # Coarse capture must happen before the policy overshoots.  First
            # HOLD the accepted target, then bind descent geometry only after
            # the measured arm has settled to that target.
            accepted_arm = tuple(float(value) for value in reference[:6])
            measured_arm = tuple(float(value) for value in measured[:6])
            self.tip_state = replace(
                self.tip_state,
                takeover_q_rad=measured_arm,
                takeover_position_m=tuple(
                    float(value) for value in tcp_position
                ),
                takeover_rotation=tuple(
                    tuple(float(value) for value in row)
                    for row in tcp_rotation
                ),
                last_arm_target_q_rad=accepted_arm,
            )
            tip_geometry_rebound = True
        tip_decision, tip_planner_fault = self._advance_tip_planner(
            robot,
            evidence,
            measured,
            arm_reference=reference,
            arm_speed_rad_s=arm_speed_rad_s,
            jaw_speed_rad_s=jaw_speed_rad_s,
            action_source=action_source,
        )
        if (
            tip_decision is not None
            and tip_decision.event == "coarse_takeover"
            and self.tip_mode is TipServoMode.EXECUTE
        ):
            # The takeover geometry is measured, but the first published arm
            # HOLD must remain the last accepted controller target.
            accepted_arm = tuple(float(value) for value in reference[:6])
            self.tip_state = replace(
                self.tip_state,
                last_arm_target_q_rad=accepted_arm,
            )
            tip_decision = replace(
                tip_decision,
                proposed_arm_target_q_rad=accepted_arm,
                commanded_arm_target_q_rad=accepted_arm,
            )
            self._last_tip_decision = tip_decision
        if self.tip_mode is TipServoMode.EXECUTE:
            # This bounded probe owns pregrasp and never masquerades a policy
            # EE sample as permission to enter the legacy close path.
            self._pickup_policy_close_intent_until_ns = None
            pickup_policy_close_intent_observed = False
            pickup_policy_close_intent = False
            pickup_policy_close_intent_remaining_ms = 0.0

        open_only_execute = bool(
            self.tip_mode is TipServoMode.EXECUTE
            and self.tip_config.probe_profile is TipServoProbeProfile.OPEN_ONLY
        )
        open_only_jaw_target: float | None = None
        open_only_jaw_guard_passed = True
        if open_only_execute:
            capture_jaw = self.tip_state.capture_jaw_target_rad
            try:
                if capture_jaw is None:
                    raise TipServoError(
                        "OPEN_ONLY planner did not bind the accepted capture jaw target"
                    )
                open_only_jaw_target = resolve_open_only_jaw_target(
                    capture_jaw_target_rad=capture_jaw,
                    previous_accepted_jaw_target_rad=float(reference[6]),
                    open_commanded=bool(self.tip_state.open_commanded),
                    config=self.tip_config,
                    ee_step_limit_rad=self.config.ee_step_limit_rad,
                )
            except TipServoError as exc:
                # Never correct an authority violation by moving the jaw back
                # toward capture: that could itself be a close. Publish the
                # already accepted full action once, then let the primary latch
                # stop both controller sessions below.
                open_only_jaw_guard_passed = False
                open_only_jaw_target = float(reference[6])
                reason = f"open_only_jaw_guard_{type(exc).__name__}"
                tip_planner_fault = tip_planner_fault or reason
                if self.tip_state.phase is not TipServoPhase.FAULT_HOLD:
                    self.tip_state = replace(
                        self.tip_state,
                        phase=TipServoPhase.FAULT_HOLD,
                        last_evaluated_at_ns=evidence.evaluated_at_ns,
                        fault_reason=reason,
                    )
                tip_decision = None

        faults = [reason for reason in (metadata_fault, fk_fault) if reason]
        if tip_planner_fault and self.tip_mode is TipServoMode.EXECUTE:
            faults.append(tip_planner_fault)
        decision = self.latch.step(
            evidence,
            measured,
            reference,
            tcp_z_m=tcp_z_m,
            tcp_speed_m_s=tcp_speed_m_s,
            home_ready=home_ready,
            policy_close_intent=pickup_policy_close_intent,
            external_fault_reason=("+".join(faults) if faults else None),
        )
        arm_hold_edge = (
            self._last_arm_hold is not None
            and bool(decision.arm_hold) != self._last_arm_hold
        )
        requires_replan = bool(
            decision.event in _VISUAL_REPLAN_EVENTS or arm_hold_edge
        )
        tip_takeover = bool(
            tip_decision is not None
            and tip_decision.event == "coarse_takeover"
            and self.tip_mode is TipServoMode.EXECUTE
        )
        tip_probe_terminal = bool(
            tip_decision is not None
            and tip_decision.phase_after is TipServoPhase.PROBE_COMPLETE
            and self.tip_mode is TipServoMode.EXECUTE
        )
        arm6_takeover = (
            None
            if arm6_runtime is None
            else resolve_arm6_takeover_boundary(
                role=ArmOnlyPolicyRole.ARM6_SEEKING,
                takeover_boundary=tip_takeover,
            )
        )
        discard_policy_chunk = (
            tip_takeover
            if arm6_takeover is None
            else arm6_takeover.discard_policy_chunk
        )
        smoother_reset_required = bool(
            (
                tip_takeover
                if arm6_takeover is None
                else arm6_takeover.smoother_reset_required
            )
            or (
                tip_probe_terminal
                and tip_decision.phase_before is not TipServoPhase.PROBE_COMPLETE
            )
        )
        replan_reason = (
            "tip_servo_takeover"
            if tip_takeover
            else (
                "tip_probe_complete"
                if tip_probe_terminal
                else (
                    decision.event
                    if decision.event in _VISUAL_REPLAN_EVENTS
                    else (
                        "arm_hold_entered"
                        if arm_hold_edge and decision.arm_hold
                        else ("arm_hold_released" if arm_hold_edge else None)
                    )
                )
            )
        )
        if requires_replan or discard_policy_chunk:
            self.replan_epoch += 1
        self._last_arm_hold = bool(decision.arm_hold)

        composed = compose_visual_action(
            policy,
            reference,
            decision,
            ee_step_limit_rad=self.config.ee_step_limit_rad,
        )
        action = np.asarray(composed.action, dtype=np.float64)
        arm_owner = "policy"
        jaw_owner = (
            "visual_latch" if decision.ee_target_rad is not None else "policy"
        )
        if arm6_runtime is not None:
            # The seventh candidate element is not a model prediction. The
            # primary latch has no permission to reinterpret it as jaw intent.
            action[6] = arm6_runtime.capture_jaw_target_rad
            jaw_owner = "startup_capture_hold"
        tip_ee_step_limited = False
        if open_only_execute:
            assert open_only_jaw_target is not None
            action[6] = open_only_jaw_target
            jaw_owner = (
                "tip_fault_hold"
                if self.tip_state.phase is TipServoPhase.FAULT_HOLD
                else (
                    "tip_open"
                    if self.tip_state.open_commanded
                    else "tip_capture_hold"
                )
            )
            planner_jaw = (
                None
                if tip_decision is None
                else tip_decision.commanded_jaw_target_rad
            )
            if planner_jaw is not None:
                tip_ee_step_limited = not math.isclose(
                    float(planner_jaw),
                    float(open_only_jaw_target),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
        if tip_decision is not None and self.tip_mode is TipServoMode.EXECUTE:
            tip_arm = tip_decision.commanded_arm_target_q_rad
            tip_jaw = tip_decision.commanded_jaw_target_rad
            if tip_arm is not None:
                action[:6] = np.asarray(tip_arm, dtype=np.float64)
                arm_owner = "visual_tip_servo"
            if tip_jaw is not None and not open_only_execute:
                jaw_delta = float(tip_jaw) - float(reference[6])
                limited_jaw_delta = float(
                    np.clip(
                        jaw_delta,
                        -self.config.ee_step_limit_rad,
                        self.config.ee_step_limit_rad,
                    )
                )
                action[6] = float(reference[6] + limited_jaw_delta)
                tip_ee_step_limited = not math.isclose(
                    jaw_delta,
                    limited_jaw_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                jaw_owner = "visual_tip_servo"
            if tip_takeover or tip_probe_terminal:
                action[:6] = reference[:6]
                arm_owner = "hold"
            elif tip_decision.event in {
                "visual_frame_hold",
                "red_area_drawdown_hold",
            }:
                action[:6] = reference[:6]
                arm_owner = "hold"
        if (
            open_only_execute
            and self.tip_state.phase is not TipServoPhase.SEEKING
        ):
            # OPEN_ONLY grants no post-takeover arm-motion authority.  Make
            # the published command and its ownership telemetry unambiguous
            # even if a future planner decision proposes another arm target.
            action[:6] = reference[:6]
            arm_owner = "hold"
        # Freeze the arm whenever the primary state machine requires HOLD.
        if composed.arm_hold or requires_replan:
            action[:6] = reference[:6]
            arm_owner = "hold"

        effective_arm_hold = bool(
            composed.arm_hold
            or requires_replan
            or arm_owner == "hold"
        )
        telemetry = self._telemetry(
            evidence,
            decision,
            tcp_position=tcp_position,
            tcp_speed_m_s=tcp_speed_m_s,
            home_ready=home_ready,
            home_error=home_error,
            home_policy_close_intent=home_policy_close_intent,
            pickup_policy_close_intent_observed=(
                pickup_policy_close_intent_observed
            ),
            pickup_policy_close_intent=pickup_policy_close_intent,
            pickup_policy_close_intent_remaining_ms=(
                pickup_policy_close_intent_remaining_ms
            ),
            policy_action_is_fresh=bool(policy_action_is_fresh),
            action_source=action_source,
            arm_hold=effective_arm_hold,
            replan_reason=replan_reason,
            ee_step_limited=bool(
                (
                    tip_ee_step_limited
                    if open_only_execute
                    else composed.ee_step_limited or tip_ee_step_limited
                )
            ),
        )
        if self.tip_mode is not TipServoMode.DISABLED:
            # The follower bounds command-cycle telemetry to 128 primitive
            # nodes.  During this no-close probe, blue-box and home vectors are
            # unreachable; trade those fields for compact, command-relevant
            # tip-servo evidence without raising that safety limit.
            telemetry.pop("blue", None)
            telemetry.pop("home_reference_arm_rad", None)
            telemetry.pop("home_error_arm_rad", None)
            ik = None if tip_decision is None else tip_decision.ik_result
            telemetry["tip_servo"] = {
                "mode": self.tip_mode.value,
                "profile": self.tip_config.probe_profile.value,
                "phase_before": (
                    self.tip_state.phase.value
                    if tip_decision is None
                    else tip_decision.phase_before.value
                ),
                "phase_after": self.tip_state.phase.value,
                "event": None if tip_decision is None else tip_decision.event,
                "fault_reason": (
                    tip_planner_fault
                    if tip_decision is None
                    else tip_decision.fault_reason
                ),
                "intent": (
                    None if tip_decision is None else tip_decision.intent.value
                ),
                "commanded_descent_m": float(
                    self.tip_state.commanded_descent_m
                ),
                "area_drop_streak": int(self.tip_state.area_drop_streak),
                "probe_success_streak": int(
                    self.tip_state.probe_success_streak
                ),
                "commanded_arm_target_q_rad": (
                    None
                    if tip_decision is None
                    or tip_decision.commanded_arm_target_q_rad is None
                    else [
                        float(value)
                        for value in tip_decision.commanded_arm_target_q_rad
                    ]
                ),
                "commanded_jaw_target_rad": (
                    None
                    if tip_decision is None
                    else tip_decision.commanded_jaw_target_rad
                ),
                "ik_condition_number": (
                    None
                    if ik is None
                    else self._finite_or_none(ik.condition_number)
                ),
                "ik_position_residual_m": (
                    None
                    if ik is None
                    else self._finite_or_none(ik.position_residual_m)
                ),
                "ik_orientation_residual_rad": (
                    None
                    if ik is None
                    else self._finite_or_none(
                        ik.orientation_residual_rad
                    )
                ),
                "ik_reason": None if ik is None else ik.reason,
                "arm_owner": arm_owner,
                "discard_policy_chunk": discard_policy_chunk,
                "smoother_reset_required": smoother_reset_required,
                "probe_close_allowed": bool(self.tip_config.allow_close),
                "takeover_reference_settled": tip_takeover_reference_settled,
                "takeover_geometry_rebound": tip_geometry_rebound,
            }
            if open_only_execute:
                # OPEN_ONLY cannot invoke IK and never enters the legacy grasp
                # streak machine. Trade those unreachable/null nodes for the
                # axis-authority evidence below while preserving the follower's
                # fixed 128-item telemetry ceiling.
                telemetry.pop("streaks", None)
                for key in (
                    "ik_condition_number",
                    "ik_position_residual_m",
                    "ik_orientation_residual_rad",
                    "ik_reason",
                ):
                    telemetry["tip_servo"].pop(key, None)
                open_only_authority_telemetry: dict[str, object] = {
                    "jaw_owner": jaw_owner,
                    "capture_jaw_target_rad": (
                        self.tip_state.capture_jaw_target_rad
                    ),
                    "main_latch_jaw_target_rad_ignored": (
                        None
                        if decision.ee_target_rad is None
                        else float(decision.ee_target_rad)
                    ),
                    "planner_jaw_target_rad": (
                        None
                        if tip_decision is None
                        else tip_decision.commanded_jaw_target_rad
                    ),
                    "published_jaw_target_rad": float(action[6]),
                    "previous_accepted_jaw_target_rad": float(reference[6]),
                    "no_close_guard_passed": bool(open_only_jaw_guard_passed),
                    "seeking_elapsed_ms": (
                        None
                        if self.tip_state.seeking_started_at_ns is None
                        else max(
                            0.0,
                            (
                                evidence.evaluated_at_ns
                                - self.tip_state.seeking_started_at_ns
                            )
                            / 1_000_000.0,
                        )
                    ),
                    "seeking_timeout_ms": float(
                        self.tip_config.seeking_timeout_ms
                    ),
                }
                if arm6_runtime is None:
                    open_only_authority_telemetry["raw_policy_jaw_target_rad"] = float(
                        policy[6]
                    )
                else:
                    open_only_authority_telemetry.update(
                        {
                            "policy_role": ARM6_SEEKING_ROLE,
                            "jaw_candidate_source": (
                                "tip_servo_exclusive_reference"
                                if action_source is VisualActionSource.TIP_SERVO
                                else "startup_accepted_reference"
                            ),
                        }
                    )
                telemetry["tip_servo"].update(open_only_authority_telemetry)
        if tcp_position is not None:
            self._last_tcp_position = tcp_position
            self._last_tcp_sample_ns = now_ns
        self._last_measured = measured.copy()
        self._last_measured_ns = now_ns
        return VisualControlCycle(
            action=tuple(float(value) for value in action),
            decision=decision,
            telemetry=telemetry,
            requires_replan=requires_replan,
            replan_reason=replan_reason,
            discard_policy_chunk=discard_policy_chunk,
            smoother_reset_required=smoother_reset_required,
            arm_owner=arm_owner,
            jaw_owner=jaw_owner,
        )

    def assert_open_only_publish_contract(
        self,
        action,
        command_reference,
        telemetry: dict[str, object],
    ) -> None:
        """Re-check the OPEN_ONLY jaw lane at the final publish boundary."""

        if not (
            self.tip_mode is TipServoMode.EXECUTE
            and self.tip_config.probe_profile is TipServoProbeProfile.OPEN_ONLY
        ):
            return
        candidate = np.asarray(action, dtype=np.float64)
        reference = np.asarray(command_reference, dtype=np.float64)
        if (
            candidate.shape != (7,)
            or reference.shape != (7,)
            or not np.isfinite(candidate).all()
            or not np.isfinite(reference).all()
        ):
            raise TipServoError(
                "OPEN_ONLY final publish assertion requires finite 7-vectors"
            )
        tip_payload = telemetry.get("tip_servo")
        if not isinstance(tip_payload, dict):
            raise TipServoError("OPEN_ONLY final command omits tip-servo telemetry")
        guard_passed = tip_payload.get("no_close_guard_passed")
        if type(guard_passed) is not bool:
            raise TipServoError("OPEN_ONLY final command omits the no-close guard")
        if not guard_passed:
            if (
                self.tip_state.phase is not TipServoPhase.FAULT_HOLD
                or self.latch.phase is not VisualLatchPhase.FAULT_HOLD
                or not np.allclose(candidate, reference, rtol=0.0, atol=1e-12)
            ):
                raise TipServoError(
                    "OPEN_ONLY guard failure may publish only one exact accepted-target FAULT_HOLD"
                )
            expected_jaw = float(reference[6])
        else:
            capture = self.tip_state.capture_jaw_target_rad
            if capture is None:
                raise TipServoError(
                    "OPEN_ONLY final command has no immutable capture target"
                )
            expected_jaw = resolve_open_only_jaw_target(
                capture_jaw_target_rad=capture,
                previous_accepted_jaw_target_rad=float(reference[6]),
                open_commanded=bool(self.tip_state.open_commanded),
                config=self.tip_config,
                ee_step_limit_rad=self.config.ee_step_limit_rad,
            )
            if candidate[6] < reference[6] - 1e-12:
                raise TipServoError("OPEN_ONLY final jaw target would close")
        if not math.isclose(
            float(candidate[6]), expected_jaw, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TipServoError(
                "OPEN_ONLY final jaw target differs from the pure authority resolver"
            )
        if self.tip_state.phase is not TipServoPhase.SEEKING:
            if not np.allclose(
                candidate[:6], reference[:6], rtol=0.0, atol=1e-12
            ):
                raise TipServoError(
                    "OPEN_ONLY post-takeover arm target differs from the accepted HOLD reference"
                )
            if tip_payload.get("arm_owner") != "hold":
                raise TipServoError(
                    "OPEN_ONLY post-takeover arm ownership is not HOLD"
                )
        published = tip_payload.get("published_jaw_target_rad")
        if (
            isinstance(published, bool)
            or not isinstance(published, (int, float))
            or not math.isclose(
                float(published),
                float(candidate[6]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise TipServoError(
                "OPEN_ONLY published jaw telemetry differs from the final command"
            )

    @staticmethod
    def _finite_or_none(value):
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    @staticmethod
    def _blob_telemetry(blob: BlobEvidence) -> dict[str, object]:
        return {
            "accepted": blob.accepted,
            "held_accepted": blob.held_accepted,
            "mask_area_fraction": VisualLatchRuntime._finite_or_none(
                blob.mask_area_fraction
            ),
            "roi_mask_fraction": VisualLatchRuntime._finite_or_none(
                blob.roi_mask_fraction
            ),
            "contour_area_fraction": VisualLatchRuntime._finite_or_none(
                blob.contour_area_fraction
            ),
            "centroid_norm": (
                None if blob.centroid_norm is None else list(blob.centroid_norm)
            ),
            "bbox_norm": (
                None if blob.bbox_norm is None else list(blob.bbox_norm)
            ),
            "aspect_ratio": VisualLatchRuntime._finite_or_none(blob.aspect_ratio),
            "solidity": VisualLatchRuntime._finite_or_none(blob.solidity),
            "candidate_count": blob.candidate_count,
            "rejection_reasons": list(blob.rejection_reasons),
            "held_rejection_reasons": list(blob.held_rejection_reasons),
        }

    def _telemetry(
        self,
        evidence: VisualEvidence,
        decision: LatchDecision,
        *,
        tcp_position: np.ndarray | None,
        tcp_speed_m_s: float | None,
        home_ready: bool,
        home_error: np.ndarray,
        home_policy_close_intent: bool,
        pickup_policy_close_intent_observed: bool,
        pickup_policy_close_intent: bool,
        pickup_policy_close_intent_remaining_ms: float,
        policy_action_is_fresh: bool,
        action_source: VisualActionSource,
        arm_hold: bool,
        replan_reason: str | None,
        ee_step_limited: bool,
    ) -> dict[str, object]:
        return {
            "phase_before": decision.phase_before.value,
            "phase_after": decision.phase_after.value,
            "event": decision.event,
            "fault_reason": decision.fault_reason,
            "frame": {
                "id": evidence.frame_id,
                "captured_at_ns": evidence.captured_at_ns,
                "evaluated_at_ns": evidence.evaluated_at_ns,
                "age_ms": self._finite_or_none(evidence.frame_age_ms),
                "status": evidence.frame_status.value,
            },
            "red": self._blob_telemetry(evidence.red),
            "blue": self._blob_telemetry(evidence.blue),
            "streaks": {
                "close": decision.close_streak,
                "grasp": decision.grasp_streak,
                "lift": decision.lift_streak,
                "transport": decision.transport_streak,
                "box": decision.box_streak,
                "settle": decision.settle_streak,
                "release": decision.release_streak,
                "home_ready": decision.home_ready_streak,
                "home_settle": decision.home_settle_streak,
            },
            "pregrasp_open_seen": decision.pregrasp_open_seen,
            "grasp_latched": decision.grasp_latched,
            "release_started": decision.release_started,
            "contact_gap_rad": self._finite_or_none(decision.contact_gap_rad),
            "held_contact_ok": decision.held_contact_ok,
            "arm_speed_rad_s": self._finite_or_none(decision.arm_speed_rad_s),
            "jaw_speed_rad_s": self._finite_or_none(decision.jaw_speed_rad_s),
            "tcp_position_m": (
                None
                if tcp_position is None
                else [float(value) for value in tcp_position]
            ),
            "tcp_speed_m_s": self._finite_or_none(tcp_speed_m_s),
            "tcp_delta_z_m": self._finite_or_none(decision.tcp_delta_z_m),
            "home_ready": home_ready,
            "home_policy_close_intent": home_policy_close_intent,
            "pickup_policy_close_intent": {
                "sampled": policy_action_is_fresh,
                "action_source": action_source.value,
                "observed": pickup_policy_close_intent_observed,
                "active": pickup_policy_close_intent,
                "remaining_ms": self._finite_or_none(
                    pickup_policy_close_intent_remaining_ms
                ),
            },
            "home_reference_arm_rad": [
                float(value) for value in self._home_center[:6]
            ],
            "home_error_arm_rad": [float(value) for value in home_error],
            "visual_usable": decision.visual_usable,
            "arm_hold": arm_hold,
            "replan_reason": replan_reason,
            "ee_step_limited": ee_step_limited,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("shadow", "hold", "replay", "act"), required=True)
    parser.add_argument(
        "--independent-estop-verified",
        action="store_true",
        help=(
            "Ephemeral operator assertion required for every persistent-HOLD execute stage "
            "(hold/replay/ACT), and for RAMP_STOP replay/ACT; pc-always-clear.yaml is insufficient."
        ),
    )
    parser.add_argument(
        "--act-duration-s",
        type=float,
        choices=(3.0, 10.0, 30.0, 65.0, 90.0),
        help="Override the reviewed ACT duration tier from the config.",
    )
    parser.add_argument("--hold-duration-s", type=float, default=10.0)
    parser.add_argument(
        "--allow-arm-hold-timeout-without-ramp-stop",
        action="store_true",
        help=(
            "Fresh per-run risk acknowledgement for a fingerprint-bound persistent HOLD profile; "
            "required together with --independent-estop-verified for every execute stage."
        ),
    )
    parser.add_argument(
        "--mark-reviewed",
        action="store_true",
        help="Do not connect hardware; approve an already completed stage after reviewing its logs/video.",
    )
    parser.add_argument(
        "--operator-authorized-skip-replay-for-act-3s",
        action="store_true",
        help=(
            "One-shot operator authorization to run only the 3-second policy stage after reviewed "
            "shadow and HOLD, without a replay review. The resulting run can never be marked reviewed."
        ),
    )
    parser.add_argument(
        "--operator-authorized-direct-act-30s",
        action="store_true",
        help=(
            "One-shot operator authorization to run only the 30-second policy stage after reviewed "
            "shadow and HOLD, skipping only absent replay/3-second/10-second review entries. "
            "The resulting run can never be marked reviewed."
        ),
    )
    parser.add_argument(
        "--operator-authorized-direct-act-65s",
        action="store_true",
        help=(
            "One-shot operator authorization to run only the 65-second policy stage after reviewed "
            "shadow and HOLD, skipping only absent replay/3-second/10-second/30-second review "
            "entries. The resulting run can never be marked reviewed."
        ),
    )
    parser.add_argument(
        "--operator-authorized-direct-act-90s",
        action="store_true",
        help=(
            "One-shot operator authorization to run only the 90-second policy stage, skipping "
            "only absent ladder review entries (including shadow/HOLD). Existing ledger entries "
            "are never ignored, and the resulting run can never be marked reviewed."
        ),
    )
    parser.add_argument(
        "--operator-authorized-act-start-reset",
        action="store_true",
        help=(
            "One-shot authorization to move only out-of-gate joints from the freshly measured "
            "pose to their fingerprint-bound ACT demonstration start centers before a non-review "
            "90-second ACT. "
            "The reset uses the existing per-joint slew, tracking, watchdog, holder, CAN, and "
            "URDF gates in the same control session as ACT."
        ),
    )
    parser.add_argument(
        "--operator-authorized-act-ee-preopen",
        action="store_true",
        help=(
            "One-shot authorization for an exact recovery profile to hold all arm joints and "
            "open only the EE to its fingerprint-bound pregrasp target before a non-review "
            "90-second ACT. The preopen uses the existing step, tracking, watchdog, holder, "
            "CAN, and URDF gates in the same control session as ACT."
        ),
    )
    parser.add_argument(
        "--operator-authorized-tip-servo-execute",
        action="store_true",
        help=(
            "One-shot authorization for the fingerprint-bound, bounded-descent, "
            "open-jaw, no-close tip-servo probe during a non-review 30-second ACT. "
            "It does not authorize OPEN_ONLY or any wider descent profile."
        ),
    )
    parser.add_argument(
        "--operator-authorized-tip-servo-open-only",
        action="store_true",
        help=(
            "One-shot authorization for the exact fingerprint-bound, closed-capture, "
            "zero-descent OPEN_ONLY tip-servo probe during a non-review 30-second ACT. "
            "It does not authorize bounded descent, grasp closure, reset, or preopen."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> FireflyDeploymentConfig:
    with path.open(encoding="utf-8") as stream:
        return draccus.load(FireflyDeploymentConfig, stream)


def _policy_runtime_contract(
    policy_type: str,
    expected_runtime_role: str,
) -> PolicyRuntimeContract:
    if type(expected_runtime_role) is not str or expected_runtime_role not in {
        _LEGACY_ACTION7_ROLE,
        ARM6_SEEKING_ROLE,
    }:
        raise ValueError(
            "expected_runtime_role must be exactly "
            f"{_LEGACY_ACTION7_ROLE!r} or {ARM6_SEEKING_ROLE!r}"
        )
    if expected_runtime_role == ARM6_SEEKING_ROLE and policy_type != "smolvla":
        raise ValueError("arm6_seeking runtime role accepts only a SmolVLA checkpoint")
    dimension = 6 if expected_runtime_role == ARM6_SEEKING_ROLE else 7
    return PolicyRuntimeContract(
        policy_type=policy_type,
        runtime_role=expected_runtime_role,
        observation_state_dimension=dimension,
        policy_action_dimension=dimension,
    )


def _validate_policy_schema(
    policy_config,
    *,
    expected_runtime_role: str = _LEGACY_ACTION7_ROLE,
) -> PolicyRuntimeContract:
    from lerobot.configs import FeatureType

    policy_type = policy_config.type
    contract = _policy_runtime_contract(policy_type, expected_runtime_role)
    expected_inputs = {"observation.state", *_POLICY_IMAGE_FEATURES}
    input_features = policy_config.input_features
    output_features = policy_config.output_features
    if not isinstance(input_features, dict) or set(input_features) != expected_inputs:
        actual = sorted(input_features) if isinstance(input_features, dict) else input_features
        raise ValueError(
            f"{policy_type} checkpoint input schema is {actual!r}; expected exactly "
            f"{sorted(expected_inputs)!r}"
        )
    if not isinstance(output_features, dict) or set(output_features) != {"action"}:
        actual = sorted(output_features) if isinstance(output_features, dict) else output_features
        raise ValueError(
            f"{policy_type} checkpoint output schema is {actual!r}; expected exactly ['action']"
        )
    required_features = {
        "observation.state": (
            FeatureType.STATE,
            (contract.observation_state_dimension,),
        ),
        "observation.images.scene": (FeatureType.VISUAL, (3, 480, 640)),
        "observation.images.wrist": (FeatureType.VISUAL, (3, 480, 640)),
        "action": (FeatureType.ACTION, (contract.policy_action_dimension,)),
    }
    all_features = {**input_features, **output_features}
    for name, (expected_type, expected_shape) in required_features.items():
        feature = all_features[name]
        shape = tuple(feature.shape)
        invalid_shape = expected_shape is not None and shape != expected_shape
        if feature.type is not expected_type or invalid_shape:
            raise ValueError(
                f"{policy_type} checkpoint feature {name!r} has type={feature.type!s}, "
                f"shape={shape}; expected type={expected_type!s}, "
                f"shape={expected_shape or 'trained visual shape'}"
            )
    return contract


def policy_runtime_contract(infer) -> PolicyRuntimeContract:
    """Return explicit callable authority; unannotated legacy callables stay 7-D."""

    contract = getattr(infer, "policy_runtime_contract", None)
    if contract is None:
        return PolicyRuntimeContract(
            policy_type="legacy_unannotated",
            runtime_role=_LEGACY_ACTION7_ROLE,
            observation_state_dimension=7,
            policy_action_dimension=7,
        )
    if not isinstance(contract, PolicyRuntimeContract):
        raise TypeError("inference callable has an invalid policy_runtime_contract")
    expected = _policy_runtime_contract(contract.policy_type, contract.runtime_role)
    if contract != expected:
        raise ValueError("inference callable policy_runtime_contract is inconsistent")
    return contract


def _finite_reference7(raw_reference) -> tuple[float, ...]:
    if isinstance(raw_reference, (str, bytes, bytearray, Mapping)):
        raise ValueError("startup accepted reference must be a finite 7-vector")
    try:
        values = list(raw_reference)
    except TypeError as exc:
        raise ValueError("startup accepted reference must be a finite 7-vector") from exc
    if len(values) != 7:
        raise ValueError("startup accepted reference must be a finite 7-vector")
    result: list[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("startup accepted reference must be a finite 7-vector")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "startup accepted reference must be a finite 7-vector"
            ) from exc
        if not math.isfinite(converted):
            raise ValueError("startup accepted reference must be a finite 7-vector")
        result.append(converted)
    return tuple(result)


def bind_arm6_seeking_runtime(infer, startup_accepted_reference) -> Arm6SeekingRuntime:
    """Synchronously bind capture jaw authority before any arm6 submission."""

    contract = policy_runtime_contract(infer)
    if contract.runtime_role != ARM6_SEEKING_ROLE:
        raise ValueError("capture jaw binding requires explicit arm6_seeking authority")
    reference = _finite_reference7(startup_accepted_reference)
    capture = reference[6]
    lower, upper = _ARM6_CAPTURE_JAW_RANGE_RAD
    if not lower <= capture <= upper:
        raise ValueError(
            "startup accepted capture jaw is outside the arm6_seeking interval "
            f"[{lower}, {upper}]"
        )
    # Cross-check the adapter contract during binding, before an inference
    # worker can receive its first observation.
    expand_arm6_action_chunk(
        ((0.0,) * 6,),
        n_steps=1,
        role=ArmOnlyPolicyRole.ARM6_SEEKING,
        capture_jaw_target_rad=capture,
        capture_jaw_range_rad=_ARM6_CAPTURE_JAW_RANGE_RAD,
    )
    return Arm6SeekingRuntime(
        contract=contract,
        startup_accepted_reference7=reference,
        capture_jaw_target_rad=capture,
        capture_jaw_range_rad=_ARM6_CAPTURE_JAW_RANGE_RAD,
    )


def load_policy(
    checkpoint: Path,
    device: str,
    n_action_steps: int,
    task: str,
    smolvla_vlm_snapshot: str | None = None,
    inference_seed: int | None = None,
    expected_runtime_role: str = _LEGACY_ACTION7_ROLE,
):
    """Load an ACT or SmolVLA checkpoint and return a 10-step inference callable."""
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    policy_type = policy_config.type
    if policy_type not in {"act", "smolvla"}:
        raise ValueError(
            f"unsupported deployment policy type {policy_type!r}; expected 'act' or 'smolvla'"
        )
    if n_action_steps != 10:
        raise ValueError("Firefly policy deployment is fixed at 10 action steps per replan")
    if inference_seed is not None and (
        isinstance(inference_seed, bool)
        or not isinstance(inference_seed, int)
        or not 0 <= inference_seed <= 0xFFFFFFFF
    ):
        raise ValueError("inference_seed must be an unsigned 32-bit integer or null")
    chunk_size = getattr(policy_config, "chunk_size", None)
    runtime_contract = _validate_policy_schema(
        policy_config,
        expected_runtime_role=expected_runtime_role,
    )
    if not isinstance(chunk_size, int) or chunk_size < n_action_steps:
        raise ValueError(
            f"{policy_type} checkpoint chunk_size={chunk_size!r} cannot provide "
            f"{n_action_steps} deployment steps"
        )
    if policy_type == "smolvla" and not task.strip():
        raise ValueError("SmolVLA deployment requires a non-empty task in the deployment config")

    vlm_snapshot: Path | None = None
    if policy_type == "act":
        if smolvla_vlm_snapshot is not None:
            raise ValueError("ACT inference must not configure a SmolVLM snapshot")
    else:
        if smolvla_vlm_snapshot is None:
            raise ValueError("SmolVLA inference requires a materialized SmolVLM snapshot")
        vlm_snapshot, _vlm_manifest = validate_materialized_smolvlm_bundle(
            Path(smolvla_vlm_snapshot)
        )
        policy_config.vlm_model_name = str(vlm_snapshot)
        policy_config.load_vlm_weights = False

    policy_config.device = device
    policy_config.n_action_steps = n_action_steps
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(
        checkpoint, config=policy_config, local_files_only=True, strict=True
    )
    preprocessor_overrides = {"device_processor": {"device": device}}
    if vlm_snapshot is not None:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": str(vlm_snapshot)
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=preprocessor_overrides,
    )
    inference_task = task if policy_type == "smolvla" else None

    inference_generation = 0

    def infer(raw_observation: dict) -> list[tuple[float, ...]]:
        nonlocal inference_generation
        observation = {}
        if inference_seed is not None:
            chunk_seed = inference_seed + inference_generation
            random.seed(chunk_seed)
            np.random.seed(chunk_seed % (2**32))
            torch.manual_seed(chunk_seed)
            if torch.device(device).type == "cuda":
                torch.cuda.manual_seed_all(chunk_seed)
        inference_generation += 1

        if runtime_contract.runtime_role == ARM6_SEEKING_ROLE:
            state = extract_arm6_observation(
                raw_observation,
                role=ArmOnlyPolicyRole.ARM6_SEEKING,
            )
        else:
            try:
                state = [raw_observation[name] for name in _FOLLOWER_STATE_FEATURES]
            except KeyError as exc:
                raise KeyError(
                    f"policy observation is missing follower state {exc.args[0]!r}"
                ) from exc
        observation["observation.state"] = np.asarray(state, dtype=np.float32)
        for camera_name in ("scene", "wrist"):
            if camera_name not in raw_observation:
                raise KeyError(f"policy observation is missing camera {camera_name!r}")
            observation[f"observation.images.{camera_name}"] = np.asarray(raw_observation[camera_name])
        prepared = prepare_observation_for_inference(
            observation,
            torch.device(device),
            task=inference_task,
            robot_type="hex_zenoh_follower",
        )
        prepared = preprocessor(prepared)
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type,
            enabled=bool(policy_config.use_amp and torch.device(device).type == "cuda"),
        ):
            normalized = policy.predict_action_chunk(prepared)
        if normalized.ndim != 3 or normalized.shape[0] != 1:
            raise ValueError(
                f"{policy_type} returned shape {tuple(normalized.shape)}, expected [1, steps, action_dim]"
            )
        if normalized.shape[1] < n_action_steps:
            raise ValueError(
                f"{policy_type} returned {normalized.shape[1]} steps, expected at least {n_action_steps}"
            )
        if (
            runtime_contract.runtime_role == ARM6_SEEKING_ROLE
            and normalized.shape[2] != 6
        ):
            raise ValueError(
                f"{policy_type} arm6_seeking returned normalized action dimension "
                f"{normalized.shape[2]}, expected 6"
            )
        normalized = normalized[:, :n_action_steps]
        processed = [postprocessor(normalized[:, index, :]) for index in range(n_action_steps)]
        chunk = torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy()
        expected_shape = (n_action_steps, runtime_contract.policy_action_dimension)
        if chunk.shape != expected_shape:
            raise ValueError(
                f"{policy_type} postprocessor returned shape {tuple(chunk.shape)}, "
                f"expected [{n_action_steps}, {runtime_contract.policy_action_dimension}]"
            )
        if runtime_contract.runtime_role == ARM6_SEEKING_ROLE:
            validated = validate_arm6_action_chunk(
                chunk.tolist(),
                n_steps=n_action_steps,
                role=ArmOnlyPolicyRole.ARM6_SEEKING,
            )
            return list(validated)
        if not np.isfinite(chunk).all():
            raise ValueError(f"{policy_type} returned NaN or infinite action values")
        return [tuple(float(value) for value in row) for row in chunk]

    infer.policy_runtime_contract = runtime_contract
    return infer


def load_episode_actions(config: FireflyDeploymentConfig) -> list[tuple[float, ...]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        config.dataset_repo_id,
        root=Path(config.dataset_root).expanduser(),
        episodes=[config.replay_episode],
        download_videos=False,
    )
    actions = dataset.select_columns("action")["action"]
    return [tuple(float(value) for value in action) for action in actions]


def build_visual_latch_runtime(config) -> VisualLatchRuntime | None:
    visual = getattr(config, "visual_phase_latch", None)
    if visual is None or not visual.enabled:
        return None
    phase_config = getattr(config, "grasp_phase_correction", None)
    center = getattr(phase_config, "act_start_center_rad", ())
    tolerance = getattr(phase_config, "act_start_tolerance_rad", ())
    return VisualLatchRuntime(
        visual,
        home_center_rad=center,
        home_tolerance_rad=tolerance,
    )


def send_visual_control_cycle(
    robot: HexZenohFollower,
    runtime: VisualLatchRuntime,
    observation: dict,
    policy_action,
    command_reference,
    *,
    inference_latency_s: float | None = None,
    action_source: VisualActionSource = VisualActionSource.FRESH_POLICY,
    arm6_runtime: Arm6SeekingRuntime | None = None,
) -> VisualControlCycle:
    """Compose, preflight, publish, and audit exactly one visual control cycle."""

    feature_names = tuple(robot.action_features)
    reference = tuple(float(value) for value in command_reference)
    cycle = runtime.compose_cycle(
        robot,
        observation,
        policy_action,
        reference,
        action_source=action_source,
        arm6_runtime=arm6_runtime,
    )
    forced_reset_reason = None
    if (
        cycle.decision.arm_hold
        or cycle.requires_replan
        or cycle.smoother_reset_required
        or cycle.decision.phase_after is VisualLatchPhase.FAULT_HOLD
    ):
        forced_reset_reason = "visual_arm_hold"
    elif action_source is VisualActionSource.INFERENCE_HOLD:
        forced_reset_reason = "nonfresh_hold"
    try:
        smoothed_action, smoothing_telemetry = runtime.j23_target_smoother.apply(
            cycle.action,
            reference,
            forced_reset_reason=forced_reset_reason,
        )
        cycle_telemetry = dict(cycle.telemetry)
        cycle_telemetry["j23_target_smoothing"] = smoothing_telemetry
        cycle = VisualControlCycle(
            action=smoothed_action,
            decision=cycle.decision,
            telemetry=cycle_telemetry,
            requires_replan=cycle.requires_replan,
            replan_reason=cycle.replan_reason,
            discard_policy_chunk=cycle.discard_policy_chunk,
            smoother_reset_required=cycle.smoother_reset_required,
            arm_owner=cycle.arm_owner,
            jaw_owner=cycle.jaw_owner,
        )
        runtime.assert_open_only_publish_contract(
            cycle.action,
            reference,
            cycle.telemetry,
        )
        robot.preflight_actions([reference, cycle.action])
    except Exception as exc:
        reason = (
            "visual control could not construct a preflight-safe command: "
            f"{type(exc).__name__}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason) from exc
    robot.send_action_with_metadata(
        action_dict(feature_names, cycle.action),
        inference_latency_s=inference_latency_s,
        visual_telemetry=cycle.telemetry,
    )

    def record_transition() -> None:
        """Record secondary phase telemetry after the command-cycle record."""

        if cycle.decision.phase_before is not cycle.decision.phase_after:
            robot.record_event(
                _VISUAL_TRANSITION_EVENT,
                phase_before=cycle.decision.phase_before.value,
                phase_after=cycle.decision.phase_after.value,
                transition=cycle.decision.event,
                visual_telemetry=cycle.telemetry,
            )
        elif cycle.replan_reason in {"arm_hold_entered", "arm_hold_released"}:
            robot.record_event(
                "visual_arm_hold_transition",
                transition=cycle.replan_reason,
                phase=cycle.decision.phase_after.value,
                visual_telemetry=cycle.telemetry,
            )

        tip_payload = cycle.telemetry.get("tip_servo")
        if (
            isinstance(tip_payload, dict)
            and tip_payload.get("phase_before")
            != tip_payload.get("phase_after")
        ):
            robot.record_event(
                "tip_servo_transition",
                phase_before=tip_payload.get("phase_before"),
                phase_after=tip_payload.get("phase_after"),
                transition=tip_payload.get("event"),
                visual_telemetry=cycle.telemetry,
            )

    if cycle.decision.phase_after is VisualLatchPhase.FAULT_HOLD:
        reason = (
            "visual phase latch entered fail-closed HOLD: "
            f"{cycle.decision.fault_reason or 'unspecified visual fault'}"
        )
        # The explicit FAULT_HOLD target has already been published. Release
        # both controller sessions before attempting secondary transition
        # telemetry: a full disk or malformed telemetry path must never defer
        # the safety stop to the command watchdog or outer disconnect cleanup.
        robot.stop(reason)
        try:
            record_transition()
        except Exception:
            logging.exception(
                "visual FAULT_HOLD transition telemetry failed after sessions were released"
            )
        raise SafetyStopError(robot.stopped_reason or reason)
    record_transition()
    return cycle


def wait_for_initial_chunk(
    robot: HexZenohFollower,
    worker: InferenceWorker,
    observation: dict,
    recorder: StageVideoRecorder,
    stopping: list[bool],
    visual_runtime: VisualLatchRuntime | None = None,
    arm6_runtime: Arm6SeekingRuntime | None = None,
) -> tuple | None:
    worker.submit(observation)
    submitted_visual_epoch = (
        None if visual_runtime is None else visual_runtime.replan_epoch
    )
    period_s = 1.0 / robot.config.command_hz
    while not stopping[0]:
        result = worker.get_nowait()
        if result is not None:
            if result.error is not None:
                raise result.error
            if (
                visual_runtime is not None
                and submitted_visual_epoch != visual_runtime.replan_epoch
            ):
                # A close/release/home transition occurred while inference was
                # running. Discard this semantically stale chunk and infer once
                # more from the latest measured state and image.
                worker.submit(observation)
                submitted_visual_epoch = visual_runtime.replan_epoch
                continue
            return result
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        if stopping[0]:
            return None
        hold = hold_action(robot, observation)
        if visual_runtime is None:
            robot.send_action(hold)
        else:
            reference_values = tuple(
                float(hold[name]) for name in robot.action_features
            )
            if arm6_runtime is None:
                send_visual_control_cycle(
                    robot,
                    visual_runtime,
                    observation,
                    reference_values,
                    reference_values,
                    action_source=VisualActionSource.INFERENCE_HOLD,
                )
            else:
                send_visual_control_cycle(
                    robot,
                    visual_runtime,
                    observation,
                    reference_values,
                    reference_values,
                    action_source=VisualActionSource.INFERENCE_HOLD,
                    arm6_runtime=arm6_runtime,
                )
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    return None


def run_tip_servo_probe_exclusive(
    robot: HexZenohFollower,
    runtime: VisualLatchRuntime,
    recorder: StageVideoRecorder,
    stopping: list[bool],
    *,
    deadline_monotonic_s: float,
    arm6_runtime: Arm6SeekingRuntime | None = None,
) -> bool:
    """Run the reviewed open-jaw descent probe without submitting inference."""

    if runtime.tip_mode is not TipServoMode.EXECUTE:
        raise ValueError("exclusive tip-servo loop requires execute mode")
    if runtime.tip_config.allow_close:
        raise ValueError("the first exclusive tip-servo probe must keep close disabled")
    if not runtime.tip_exclusive_active:
        raise ValueError("exclusive tip-servo loop requires an active takeover")

    feature_names = tuple(robot.action_features)
    period_s = 1.0 / robot.config.command_hz
    while not stopping[0]:
        if time.monotonic() >= deadline_monotonic_s:
            reason = (
                "tip-servo probe exceeded the ACT deadline before reaching "
                "the reviewed visual progress gate"
            )
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        reference = robot.command_target or hold_action(robot, observation)
        reference_values = tuple(
            float(reference[name]) for name in feature_names
        )
        cycle = send_visual_control_cycle(
            robot,
            runtime,
            observation,
            reference_values,
            reference_values,
            action_source=VisualActionSource.TIP_SERVO,
            arm6_runtime=arm6_runtime,
        )
        if runtime.tip_probe_complete:
            robot.record_event(
                "tip_servo_probe_complete",
                commanded_descent_m=runtime.tip_state.commanded_descent_m,
                close_allowed=runtime.tip_config.allow_close,
                terminal_action=cycle.action,
                visual_telemetry=cycle.telemetry,
            )
            return True
        if not runtime.tip_exclusive_active:
            reason = (
                "tip-servo exclusive ownership ended without a reviewed "
                f"probe terminal; phase={runtime.tip_state.phase.value}"
            )
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    return False


def _resolved_arm_step_limits(config) -> tuple[float, ...]:
    resolver = getattr(config, "resolved_arm_step_limits_rad", None)
    if callable(resolver):
        return tuple(float(value) for value in resolver())
    return (float(config.arm_step_limit_rad),) * 6


def prepare_policy_action_chunk(
    config,
    robot,
    raw_actions,
    reference_values,
    corrector: GraspPhaseCorrector | None = None,
    arm6_runtime: Arm6SeekingRuntime | None = None,
):
    """Optionally adapt one inferred chunk and expose auditable telemetry."""
    arm6_telemetry: dict[str, object] | None = None
    source_actions = raw_actions
    if arm6_runtime is not None:
        if not isinstance(arm6_runtime, Arm6SeekingRuntime):
            raise TypeError("arm6_runtime must be an Arm6SeekingRuntime")
        reference7 = _finite_reference7(reference_values)
        if not math.isclose(
            reference7[6],
            arm6_runtime.capture_jaw_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "arm6 command reference differs from the immutable capture jaw"
            )
        expanded = expand_arm6_action_chunk(
            raw_actions,
            n_steps=10,
            role=ArmOnlyPolicyRole.ARM6_SEEKING,
            capture_jaw_target_rad=arm6_runtime.capture_jaw_target_rad,
            capture_jaw_range_rad=arm6_runtime.capture_jaw_range_rad,
        )
        source_actions = expanded.candidate7
        arm6_telemetry = arm6_runtime.telemetry()
        if corrector is not None:
            correction_enabled = getattr(
                getattr(corrector, "config", None), "enabled", None
            )
            if correction_enabled is not False:
                raise ValueError(
                    "arm6_seeking forbids an enabled or unverified legacy jaw corrector"
                )
    if corrector is None:
        assisted_actions = source_actions
        correction_telemetry = {
            "enabled": False,
            "phase_before": "not_applied",
            "phase_after": "not_applied",
            "event": None,
            "transformed": False,
        }
    else:
        assisted_actions, correction_telemetry = corrector.apply(
            source_actions, reference_values
        )

    if arm6_runtime is not None:
        checked_assisted = tuple(_finite_reference7(row) for row in assisted_actions)
        if len(checked_assisted) != 10 or any(
            not math.isclose(
                row[6],
                arm6_runtime.capture_jaw_target_rad,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in checked_assisted
        ):
            raise ValueError("arm6 assisted chunk changed the immutable capture jaw")
        assisted_actions = checked_assisted

    enabled = bool(getattr(config, "enable_action_slew_rate_adapter", False))
    if not enabled:
        actions = assisted_actions
        telemetry = {
            "raw_actions": raw_actions,
            "assisted_actions": assisted_actions,
            "adapted_actions": actions,
            "grasp_phase_correction": correction_telemetry,
            "action_slew_rate": {
                "enabled": False,
                "step_limited": [False] * len(actions),
                "limited_value_counts": [0] * len(actions),
                "limited_step_count": 0,
                "limited_value_count": 0,
            },
        }
        if arm6_telemetry is not None:
            telemetry["arm6_authority"] = arm6_telemetry
        return actions, telemetry

    result = adapt_action_chunk_slew_rate(
        assisted_actions,
        reference_values,
        arm_step_limit_rad=robot.config.arm_step_limit_rad,
        ee_step_limit_rad=robot.config.ee_step_limit_rad,
        arm_step_limits_rad=_resolved_arm_step_limits(robot.config),
    )
    actions = result.adapted_chunk
    if arm6_runtime is not None and any(
        not math.isclose(
            row[6],
            arm6_runtime.capture_jaw_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in actions
    ):
        raise ValueError("arm6 slew adapter changed the immutable capture jaw")
    telemetry = {
        "raw_actions": raw_actions,
        "assisted_actions": assisted_actions,
        "adapted_actions": actions,
        "grasp_phase_correction": correction_telemetry,
        "action_slew_rate": {
            "enabled": True,
            "arm_step_limit_rad": robot.config.arm_step_limit_rad,
            "arm_step_limits_rad": _resolved_arm_step_limits(robot.config),
            "ee_step_limit_rad": robot.config.ee_step_limit_rad,
            "step_limited": result.step_limited,
            "limited_value_counts": result.limited_value_counts,
            "limited_step_count": result.limited_step_count,
            "limited_value_count": result.limited_value_count,
            "arm_limited_value_count": result.arm_limited_value_count,
            "ee_limited_value_count": result.ee_limited_value_count,
            "max_raw_arm_step_rad": result.max_raw_arm_step_rad,
            "max_raw_ee_step_rad": result.max_raw_ee_step_rad,
            "max_adapted_arm_step_rad": result.max_adapted_arm_step_rad,
            "max_adapted_ee_step_rad": result.max_adapted_ee_step_rad,
        },
    }
    if arm6_telemetry is not None:
        telemetry["arm6_authority"] = arm6_telemetry
    return actions, telemetry


def run_shadow(config, robot, infer, recorder, stopping) -> None:
    period_s = 1.0 / config.fps
    runtime_contract = policy_runtime_contract(infer)
    arm6_runtime: Arm6SeekingRuntime | None = None
    while not stopping[0]:
        observation = robot.get_observation()
        recorder.add_observation(observation)
        feature_names = tuple(robot.action_features)
        if runtime_contract.runtime_role == ARM6_SEEKING_ROLE:
            reference = robot.command_target
            if reference is None:
                reference = robot.prime_shadow_reference()
            if arm6_runtime is None:
                arm6_runtime = bind_arm6_seeking_runtime(
                    infer,
                    tuple(reference[name] for name in feature_names),
                )
            started_at = time.monotonic()
            raw_chunk = infer(observation)
            latency_s = time.monotonic() - started_at
        else:
            # Preserve the legacy 7-D call order and reference semantics.
            started_at = time.monotonic()
            raw_chunk = infer(observation)
            latency_s = time.monotonic() - started_at
            reference = robot.command_target
            if reference is None:
                reference = robot.prime_shadow_reference()
        reference_values = tuple(reference[name] for name in feature_names)
        chunk, action_telemetry = prepare_policy_action_chunk(
            config,
            robot,
            raw_chunk,
            reference_values,
            arm6_runtime=arm6_runtime,
        )
        robot.record_event(
            "inference_chunk",
            latency_s=latency_s,
            steps=len(chunk),
            reference=reference_values,
            **action_telemetry,
        )
        for values in chunk[: config.n_action_steps]:
            if stopping[0]:
                break
            cycle_start = time.monotonic()
            latest = robot.get_observation()
            recorder.add_observation(latest)
            try:
                robot.send_action_with_metadata(
                    action_dict(feature_names, values),
                    inference_latency_s=latency_s,
                )
            except (ValueError, TimeoutError) as exc:
                logging.error("shadow safety rejection: %s", exc)
            remaining = period_s - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)


def run_hold(
    config,
    robot,
    recorder,
    stopping,
    duration_s: float,
    *,
    independent_power_cut_verified: bool,
) -> bool:
    period_s = 1.0 / config.fps
    deadline = time.monotonic() + duration_s
    target = robot.command_target
    if target is None:
        raise SafetyStopError("stationary hold has no startup measured-state target")
    robot.record_event(
        "hold_stage_start",
        action_source="startup_measured_state",
        model_inference_enabled=False,
        requested_duration_s=duration_s,
        target=target,
        experimental_hold_override=robot.config.allow_arm_hold_timeout_without_ramp_stop,
        independent_physical_power_cut_confirmed=independent_power_cut_verified,
    )
    while not stopping[0] and time.monotonic() < deadline:
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        robot.send_action(target)
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    return not stopping[0] and time.monotonic() >= deadline


def run_replay(config, robot, recorder, stopping, source) -> bool:
    replay = interpolate_actions(source, config.replay_speed)
    observation = robot.get_observation()
    measured = [observation[name] for name in robot.action_features]
    validate_replay_start(
        measured,
        replay[0],
        arm_dof=len(robot.config.arm_joint_names),
    )
    robot.preflight_actions([tuple(measured), *replay])
    period_s = 1.0 / config.fps
    for values in replay:
        if stopping[0]:
            return False
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        robot.send_action(action_dict(tuple(robot.action_features), values))
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    return True


def build_act_start_reset_trajectory(
    start: tuple[float, ...],
    target: tuple[float, ...],
    step_limits: tuple[float, ...],
    *,
    max_frames: int = _ACT_START_RESET_MAX_FRAMES,
) -> list[tuple[float, ...]]:
    """Build one synchronized, monotonic joint-space reset trajectory."""
    if not start or len(start) != len(target) or len(start) != len(step_limits):
        raise ValueError("ACT start reset vectors must have the same non-zero length")
    if max_frames <= 0:
        raise ValueError("ACT start reset max_frames must be positive")
    if not all(np.isfinite(value) for value in (*start, *target, *step_limits)):
        raise ValueError("ACT start reset vectors must contain only finite values")
    if any(limit <= 0.0 for limit in step_limits):
        raise ValueError("ACT start reset step limits must be positive")
    deltas = tuple(goal - current for current, goal in zip(start, target, strict=True))
    frame_count = max(
        1,
        max(
            int(np.ceil(abs(delta) / limit))
            for delta, limit in zip(deltas, step_limits, strict=True)
        ),
    )
    if frame_count > max_frames:
        raise ValueError(
            f"ACT start reset requires {frame_count} frames, exceeding the hard {max_frames}-frame envelope"
        )
    return [
        tuple(
            goal if frame == frame_count else current + delta * frame / frame_count
            for current, goal, delta in zip(start, target, deltas, strict=True)
        )
        for frame in range(1, frame_count + 1)
    ]


def run_act_start_reset(config, robot, recorder, stopping) -> None:
    """Move only failed start-gate joints to center without releasing control."""
    correction_config = config.grasp_phase_correction
    center = tuple(float(value) for value in correction_config.act_start_center_rad)
    feature_names = tuple(robot.action_features)
    candidate_lead_budget = tuple(
        float(value) for value in _ACT_START_RESET_CANDIDATE_LEAD_BUDGET_RAD
    )
    waypoint_step_cap = tuple(
        float(value) for value in _ACT_START_RESET_WAYPOINT_STEP_CAP_RAD
    )
    tracking_limits = (
        *(float(robot.config.arm_tracking_error_rad) for _ in range(6)),
        float(robot.config.ee_tracking_error_rad),
    )
    if (
        len(candidate_lead_budget) != len(feature_names)
        or len(waypoint_step_cap) != len(feature_names)
        or len(tracking_limits) != len(feature_names)
        or any(
            not np.isfinite(waypoint_cap) or waypoint_cap <= 0.0
            for waypoint_cap in waypoint_step_cap
        )
        or any(
            not np.isfinite(budget)
            or budget <= 0.0
            or budget >= tracking_limit
            for budget, tracking_limit in zip(
                candidate_lead_budget, tracking_limits, strict=True
            )
        )
    ):
        raise ValueError(
            "ACT start reset waypoint caps and candidate lead budgets must be "
            "finite and positive, and lead budgets must be strictly below the "
            "corresponding follower tracking limits"
        )
    reference = robot.command_target
    if reference is None:
        raise SafetyStopError("ACT start reset has no acquired startup measured-state target")
    start = tuple(float(reference[name]) for name in feature_names)
    if len(center) != len(feature_names):
        raise ValueError(
            f"ACT start reset center has {len(center)} joints, expected {len(feature_names)}"
        )
    start_assessment = assess_act_start_pose(correction_config, start)
    failed_indices = tuple(int(index) for index in start_assessment["failed_indices"])
    if 2 in failed_indices and center[2] > start[2]:
        reason = (
            "ACT start reset refuses unvalidated positive joint_3 motion; "
            f"measured={start[2]:.6f}, center={center[2]:.6f}. "
            "Position joint_3 inside the fingerprint-bound start gate while STANDBY."
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    failed_index_set = set(failed_indices)
    target = tuple(
        goal if index in failed_index_set else current
        for index, (current, goal) in enumerate(zip(start, center, strict=True))
    )
    travel = tuple(abs(goal - current) for current, goal in zip(start, target, strict=True))
    if len(travel) != len(_ACT_START_RESET_MAX_TRAVEL_RAD) or any(
        delta > maximum + 1e-9
        for delta, maximum in zip(
            travel, _ACT_START_RESET_MAX_TRAVEL_RAD, strict=True
        )
    ):
        reason = (
            "ACT start reset measured pose exceeds the hard per-joint travel envelope; "
            f"travel_rad={travel}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    follower_step_limits = (
        *tuple(float(value) for value in robot.config.resolved_arm_step_limits_rad()),
        float(robot.config.ee_step_limit_rad),
    )
    reset_step_limits = tuple(
        min(step_limit, lead_budget, waypoint_cap)
        for step_limit, lead_budget, waypoint_cap in zip(
            follower_step_limits,
            candidate_lead_budget,
            waypoint_step_cap,
            strict=True,
        )
    )
    trajectory = build_act_start_reset_trajectory(start, target, reset_step_limits)
    robot.preflight_actions([start, *trajectory])
    robot.record_event(
        "act_start_reset_begin",
        start=start,
        demonstration_center=center,
        target=target,
        corrected_indices=failed_indices,
        preserved_indices=tuple(
            index for index in range(len(start)) if index not in failed_index_set
        ),
        travel_rad=travel,
        frames=len(trajectory),
        follower_step_limits_rad=follower_step_limits,
        waypoint_step_cap_rad=waypoint_step_cap,
        step_limits_rad=reset_step_limits,
        max_frames=_ACT_START_RESET_MAX_FRAMES,
        candidate_lead_budget_rad=candidate_lead_budget,
        stall_frames=_ACT_START_RESET_STALL_FRAMES,
        max_control_frames=_ACT_START_RESET_MAX_CONTROL_FRAMES,
    )
    period_s = 1.0 / config.fps
    trajectory_index = 0
    control_frames = 0
    stall_frames = 0
    last_target = start
    while trajectory_index < len(trajectory):
        if stopping[0]:
            reason = "operator interrupt during ACT start reset"
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        if control_frames >= _ACT_START_RESET_MAX_CONTROL_FRAMES:
            reason = (
                "ACT start reset exceeded the hard feedback-control frame limit "
                f"{_ACT_START_RESET_MAX_CONTROL_FRAMES} at nominal waypoint "
                f"{trajectory_index}/{len(trajectory)}"
            )
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        measured = tuple(float(observation[name]) for name in feature_names)
        candidate = trajectory[trajectory_index]
        ready_to_advance = all(
            abs(measured[index] - candidate[index])
            <= candidate_lead_budget[index]
            for index in failed_indices
        )
        if ready_to_advance:
            values = candidate
            trajectory_index += 1
            stall_frames = 0
        else:
            values = last_target
            stall_frames += 1
            if stall_frames >= _ACT_START_RESET_STALL_FRAMES:
                errors = tuple(
                    abs(measured[index] - last_target[index])
                    for index in failed_indices
                )
                reason = (
                    "ACT start reset made no feedback-paced progress for "
                    f"{_ACT_START_RESET_STALL_FRAMES} frames at nominal waypoint "
                    f"{trajectory_index}/{len(trajectory)}; failed-joint errors={errors}"
                )
                robot.stop(reason)
                raise SafetyStopError(robot.stopped_reason or reason)
        robot.send_action(action_dict(feature_names, values))
        last_target = values
        control_frames += 1
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)

    pass_streak = 0
    final_assessment = None
    for settle_frame in range(1, _ACT_START_RESET_SETTLE_FRAMES + 1):
        if stopping[0]:
            reason = "operator interrupt while settling ACT start reset"
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        measured = tuple(float(observation[name]) for name in feature_names)
        final_assessment = assess_act_start_pose(correction_config, measured)
        pass_streak = pass_streak + 1 if final_assessment["passed"] else 0
        robot.send_action(action_dict(feature_names, target))
        if pass_streak >= _ACT_START_RESET_REQUIRED_PASS_FRAMES:
            robot.record_event(
                "act_start_reset_complete",
                target=target,
                measured=measured,
                movement_frames=len(trajectory),
                movement_control_frames=control_frames,
                movement_hold_frames=control_frames - len(trajectory),
                settle_frames=settle_frame,
                consecutive_gate_pass_frames=pass_streak,
                assessment=final_assessment,
            )
            return
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    reason = (
        "ACT start reset did not settle inside the fingerprint-bound demonstration gate "
        f"within {_ACT_START_RESET_SETTLE_FRAMES} frames; assessment={final_assessment}"
    )
    robot.stop(reason)
    raise SafetyStopError(robot.stopped_reason or reason)


def run_act_ee_preopen(config, robot, recorder, stopping) -> None:
    """Hold the recovery arm pose while opening only the EE in the ACT session."""
    correction_config = config.grasp_phase_correction
    if not bool(correction_config.act_start_ee_preopen_enabled):
        reason = "ACT start EE preopen was invoked without its fingerprint-bound profile"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    if stopping[0]:
        reason = "operator interrupt before ACT start EE preopen"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)

    feature_names = tuple(robot.action_features)
    if feature_names != _FOLLOWER_STATE_FEATURES:
        reason = (
            "ACT start EE preopen requires the exact ordered seven-joint follower schema; "
            f"received {feature_names!r}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    reference = robot.command_target
    if reference is None:
        reason = "ACT start EE preopen has no acquired startup measured-state target"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    start = tuple(float(reference[name]) for name in feature_names)

    target_value = correction_config.pregrasp_open_target_rad
    ready_value = correction_config.pregrasp_open_ready_rad
    if target_value is None or ready_value is None:
        reason = "ACT start EE preopen profile has no target/ready pair"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    target_ee_rad = float(target_value)
    ready_ee_rad = float(ready_value)
    if not np.isfinite(target_ee_rad) or not np.isfinite(ready_ee_rad):
        reason = "ACT start EE preopen target/ready values must be finite"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)

    observation = robot.get_observation()
    recorder.add_observation(observation)
    try:
        measured_start = tuple(float(observation[name]) for name in feature_names)
        start_assessment = assess_act_start_pose(correction_config, measured_start)
    except (KeyError, TypeError, ValueError) as exc:
        reason = f"ACT recovery start measurement is invalid: {exc}"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason) from exc
    robot.record_event("act_recovery_start_preflight", **start_assessment)
    if not start_assessment["passed"]:
        failed = ",".join(
            str(index + 1) for index in start_assessment["failed_indices"]
        )
        reason = (
            "ACT recovery measured start pose is outside the fingerprint-bound gate "
            f"at 1-based joint indices {failed}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)

    if start[6] > target_ee_rad:
        reason = (
            "ACT start EE preopen refuses closing motion; "
            f"startup_target={start[6]:.6f}, open_target={target_ee_rad:.6f}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    travel_rad = target_ee_rad - start[6]
    if travel_rad > _ACT_START_EE_PREOPEN_MAX_TRAVEL_RAD + 1e-12:
        reason = (
            "ACT start EE preopen exceeds the fingerprint-bound opening travel; "
            f"travel={travel_rad:.6f}, maximum={_ACT_START_EE_PREOPEN_MAX_TRAVEL_RAD:.6f}"
        )
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)

    target = (*start[:6], target_ee_rad)
    step_limits = (
        *_resolved_arm_step_limits(robot.config),
        float(robot.config.ee_step_limit_rad),
    )
    try:
        trajectory = build_act_start_reset_trajectory(
            start,
            target,
            step_limits,
            max_frames=_ACT_START_EE_PREOPEN_MAX_FRAMES,
        )
        robot.preflight_actions([start, *trajectory])
    except ValueError as exc:
        reason = f"ACT start EE preopen trajectory preflight failed: {exc}"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason) from exc

    settle_tolerance_rad = target_ee_rad - ready_ee_rad
    if settle_tolerance_rad < 0.0:
        reason = "ACT start EE preopen ready value exceeds its target"
        robot.stop(reason)
        raise SafetyStopError(robot.stopped_reason or reason)
    robot.record_event(
        "act_start_ee_preopen_begin",
        start=start,
        target=target,
        opening_travel_rad=travel_rad,
        movement_frames=len(trajectory),
        step_limits_rad=step_limits,
        max_opening_travel_rad=_ACT_START_EE_PREOPEN_MAX_TRAVEL_RAD,
        max_movement_frames=_ACT_START_EE_PREOPEN_MAX_FRAMES,
        ready_ee_rad=ready_ee_rad,
        settle_tolerance_rad=settle_tolerance_rad,
        settle_frames=_ACT_START_EE_PREOPEN_SETTLE_FRAMES,
        required_settle_pass_frames=_ACT_START_EE_PREOPEN_REQUIRED_PASS_FRAMES,
    )

    period_s = 1.0 / config.fps
    for values in trajectory:
        if stopping[0]:
            reason = "operator interrupt during ACT start EE preopen"
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        robot.send_action(action_dict(feature_names, values))
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)

    pass_streak = 0
    final_measured = None
    final_assessment = None
    final_ee_error_rad = None
    for settle_frame in range(1, _ACT_START_EE_PREOPEN_SETTLE_FRAMES + 1):
        if stopping[0]:
            reason = "operator interrupt while settling ACT start EE preopen"
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        cycle_start = time.monotonic()
        observation = robot.get_observation()
        recorder.add_observation(observation)
        try:
            final_measured = tuple(float(observation[name]) for name in feature_names)
            final_assessment = assess_act_start_pose(correction_config, final_measured)
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"ACT start EE preopen settle measurement is invalid: {exc}"
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason) from exc
        final_ee_error_rad = abs(final_measured[6] - target_ee_rad)
        opening_ready = bool(
            final_measured[6] >= ready_ee_rad
            and final_ee_error_rad <= settle_tolerance_rad + 1e-12
        )
        pass_streak = (
            pass_streak + 1
            if opening_ready and final_assessment["passed"]
            else 0
        )
        robot.send_action(action_dict(feature_names, target))
        if pass_streak >= _ACT_START_EE_PREOPEN_REQUIRED_PASS_FRAMES:
            robot.record_event(
                "act_start_ee_preopen_complete",
                target=target,
                measured=final_measured,
                movement_frames=len(trajectory),
                settle_frames=settle_frame,
                consecutive_pass_frames=pass_streak,
                ready_ee_rad=ready_ee_rad,
                ee_error_rad=final_ee_error_rad,
                assessment=final_assessment,
            )
            return
        remaining = period_s - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)

    reason = (
        "ACT start EE preopen did not settle at the fingerprint-bound opening target "
        f"within {_ACT_START_EE_PREOPEN_SETTLE_FRAMES} frames; "
        f"measured={final_measured}, ee_error_rad={final_ee_error_rad}, "
        f"assessment={final_assessment}"
    )
    robot.stop(reason)
    raise SafetyStopError(robot.stopped_reason or reason)


def run_act(config, robot, infer, recorder, stopping, duration_s: float) -> bool:
    synchronous_hold = (
        getattr(config, "act_chunk_inference_mode", "pipelined")
        == "synchronous_hold"
    )
    visual_runtime = build_visual_latch_runtime(config)
    if visual_runtime is not None and not synchronous_hold:
        raise ValueError(
            "visual phase latch requires synchronous_hold inference scheduling"
        )
    runtime_contract = policy_runtime_contract(infer)
    arm6_runtime: Arm6SeekingRuntime | None = None
    worker = InferenceWorker(infer)
    correction_config = getattr(config, "grasp_phase_correction", None)
    corrector = (
        GraspPhaseCorrector(correction_config)
        if correction_config is not None
        else None
    )
    try:
        observation = robot.get_observation()
        recorder.add_observation(observation)
        feature_names = tuple(robot.action_features)
        if correction_config is not None and getattr(
            correction_config, "act_start_center_rad", None
        ):
            measured_start = tuple(
                float(observation[name]) for name in feature_names
            )
            start_assessment = assess_act_start_pose(
                correction_config, measured_start
            )
            robot.record_event("act_start_pose_preflight", **start_assessment)
            if not start_assessment["passed"]:
                failed = ",".join(
                    str(index + 1) for index in start_assessment["failed_indices"]
                )
                reason = (
                    "ACT measured start pose is outside the fingerprint-bound "
                    f"demonstration gate at 1-based joint indices {failed}"
                )
                robot.stop(reason)
                raise SafetyStopError(robot.stopped_reason or reason)

        if runtime_contract.runtime_role == ARM6_SEEKING_ROLE:
            startup_reference = robot.command_target or hold_action(robot, observation)
            startup_reference_values = tuple(
                startup_reference[name] for name in feature_names
            )
            arm6_runtime = bind_arm6_seeking_runtime(
                infer,
                startup_reference_values,
            )

        if visual_runtime is None:
            if arm6_runtime is None:
                chunk = wait_for_initial_chunk(
                    robot, worker, observation, recorder, stopping
                )
            else:
                chunk = wait_for_initial_chunk(
                    robot,
                    worker,
                    observation,
                    recorder,
                    stopping,
                    arm6_runtime=arm6_runtime,
                )
        else:
            if arm6_runtime is None:
                chunk = wait_for_initial_chunk(
                    robot,
                    worker,
                    observation,
                    recorder,
                    stopping,
                    visual_runtime,
                )
            else:
                chunk = wait_for_initial_chunk(
                    robot,
                    worker,
                    observation,
                    recorder,
                    stopping,
                    visual_runtime,
                    arm6_runtime,
                )
        if chunk is None:
            return False
        if visual_runtime is not None and visual_runtime.phase is VisualLatchPhase.DONE:
            return True
        deadline = time.monotonic() + duration_s
        period_s = 1.0 / config.fps
        while not stopping[0] and time.monotonic() < deadline:
            if len(chunk.actions) != config.n_action_steps:
                robot.stop(
                    f"policy returned {len(chunk.actions)} actions, expected {config.n_action_steps}"
                )
                raise SafetyStopError(robot.stopped_reason or "invalid policy chunk")
            latest_observation = robot.get_observation()
            reference = robot.command_target or hold_action(robot, latest_observation)
            reference_values = tuple(reference[name] for name in feature_names)
            actions, action_telemetry = prepare_policy_action_chunk(
                config,
                robot,
                chunk.actions,
                reference_values,
                corrector=corrector,
                arm6_runtime=arm6_runtime,
            )
            correction_telemetry = action_telemetry["grasp_phase_correction"]
            if correction_telemetry.get("progress_timeout", False):
                robot.record_event(
                    "inference_chunk",
                    generation=chunk.generation,
                    latency_s=chunk.latency_s,
                    steps=len(actions),
                    reference=reference_values,
                    **action_telemetry,
                )
                timeout_kind = correction_telemetry["progress_timeout_kind"]
                deadline_fields = {
                    "raw_open": "raw_open_progress_deadline_chunks",
                    "pickup": "pickup_progress_deadline_chunks",
                    "release": "release_progress_deadline_chunks",
                }
                timeout_descriptions = {
                    "raw_open": "produced no full-chunk raw EE opening intent",
                    "pickup": "did not reach and confirm the demonstrated pickup pose",
                    "release": "did not reach and confirm the blue-box release phase",
                }
                if timeout_kind not in deadline_fields:
                    raise ValueError(
                        f"unknown grasp phase progress timeout {timeout_kind!r}"
                    )
                deadline_chunks = correction_telemetry[
                    deadline_fields[timeout_kind]
                ]
                reason = (
                    f"policy {timeout_descriptions[timeout_kind]} within "
                    f"{deadline_chunks} inference chunks; corrected actions were not sent"
                )
                robot.stop(reason)
                raise SafetyStopError(robot.stopped_reason or reason)

            robot.preflight_actions([reference_values, *actions])
            robot.record_event(
                "inference_chunk",
                generation=chunk.generation,
                latency_s=chunk.latency_s,
                steps=len(actions),
                reference=reference_values,
                **action_telemetry,
            )
            if not synchronous_hold:
                worker.submit(latest_observation)
            for index, values in enumerate(actions):
                if stopping[0]:
                    return False
                if time.monotonic() >= deadline:
                    if (
                        visual_runtime is not None
                        and visual_runtime.tip_mode is TipServoMode.EXECUTE
                    ):
                        reason = (
                            "30-second tip-servo ACT ended before the reviewed "
                            "probe progress gate; "
                            f"terminal tip phase={visual_runtime.tip_state.phase.value}"
                        )
                        robot.stop(reason)
                        raise SafetyStopError(robot.stopped_reason or reason)
                    if (
                        visual_runtime is not None
                        and duration_s >= 90.0
                        and visual_runtime.phase is not VisualLatchPhase.DONE
                    ):
                        reason = (
                            "90-second visual ACT ended before task_done; "
                            f"terminal phase={visual_runtime.phase.value}"
                        )
                        robot.stop(reason)
                        raise SafetyStopError(robot.stopped_reason or reason)
                    return True
                cycle_start = time.monotonic()
                observation = robot.get_observation()
                recorder.add_observation(observation)
                visual_cycle = None
                if visual_runtime is None:
                    robot.send_action_with_metadata(
                        action_dict(feature_names, values),
                        inference_latency_s=(
                            chunk.latency_s if index == 0 else None
                        ),
                    )
                else:
                    live_reference = robot.command_target or hold_action(
                        robot, observation
                    )
                    live_reference_values = tuple(
                        float(live_reference[name]) for name in feature_names
                    )
                    if arm6_runtime is None:
                        visual_cycle = send_visual_control_cycle(
                            robot,
                            visual_runtime,
                            observation,
                            values,
                            live_reference_values,
                            inference_latency_s=(
                                chunk.latency_s if index == 0 else None
                            ),
                        )
                    else:
                        visual_cycle = send_visual_control_cycle(
                            robot,
                            visual_runtime,
                            observation,
                            values,
                            live_reference_values,
                            inference_latency_s=(
                                chunk.latency_s if index == 0 else None
                            ),
                            arm6_runtime=arm6_runtime,
                        )
                remaining = period_s - (time.monotonic() - cycle_start)
                if remaining > 0:
                    time.sleep(remaining)
                if (
                    visual_cycle is not None
                    and visual_cycle.decision.phase_after is VisualLatchPhase.DONE
                ):
                    return True
                if (
                    visual_cycle is not None
                    and (
                        visual_cycle.requires_replan
                        or visual_cycle.discard_policy_chunk
                    )
                ):
                    break
            if (
                visual_runtime is not None
                and bool(
                    getattr(
                        visual_runtime,
                        "tip_exclusive_active",
                        False,
                    )
                )
            ):
                if arm6_runtime is None:
                    probe_completed = run_tip_servo_probe_exclusive(
                        robot,
                        visual_runtime,
                        recorder,
                        stopping,
                        deadline_monotonic_s=deadline,
                    )
                else:
                    probe_completed = run_tip_servo_probe_exclusive(
                        robot,
                        visual_runtime,
                        recorder,
                        stopping,
                        deadline_monotonic_s=deadline,
                        arm6_runtime=arm6_runtime,
                    )
                if probe_completed:
                    return True
                return False
            if synchronous_hold:
                # The pipelined schedule predicts from the observation taken
                # before executing the current 10-step chunk. On a moving arm,
                # that makes the next absolute-action chunk one full horizon
                # stale and can repeatedly reverse J2/J3 at chunk boundaries.
                # Hold the last accepted target and infer from the measured
                # state after the chunk instead. wait_for_initial_chunk keeps
                # publishing HOLD at command_hz while inference runs.
                next_observation = robot.get_observation()
                recorder.add_observation(next_observation)
                if visual_runtime is None:
                    if arm6_runtime is None:
                        next_chunk = wait_for_initial_chunk(
                            robot,
                            worker,
                            next_observation,
                            recorder,
                            stopping,
                        )
                    else:
                        next_chunk = wait_for_initial_chunk(
                            robot,
                            worker,
                            next_observation,
                            recorder,
                            stopping,
                            arm6_runtime=arm6_runtime,
                        )
                else:
                    if arm6_runtime is None:
                        next_chunk = wait_for_initial_chunk(
                            robot,
                            worker,
                            next_observation,
                            recorder,
                            stopping,
                            visual_runtime,
                        )
                    else:
                        next_chunk = wait_for_initial_chunk(
                            robot,
                            worker,
                            next_observation,
                            recorder,
                            stopping,
                            visual_runtime,
                            arm6_runtime,
                        )
                if next_chunk is None:
                    return False
                if (
                    visual_runtime is not None
                    and visual_runtime.phase is VisualLatchPhase.DONE
                ):
                    return True
            else:
                next_chunk = worker.get_nowait()
                if next_chunk is None:
                    robot.stop(
                        "next policy chunk was not ready after 10 command steps; old actions were not reused"
                    )
                    raise SafetyStopError(robot.stopped_reason or "policy chunk deadline missed")
                if next_chunk.error is not None:
                    robot.stop(f"policy inference failed: {next_chunk.error}")
                    raise next_chunk.error
            chunk = next_chunk
        if (
            visual_runtime is not None
            and visual_runtime.tip_mode is TipServoMode.EXECUTE
        ):
            reason = (
                "30-second tip-servo ACT ended before the reviewed probe "
                "progress gate; "
                f"terminal tip phase={visual_runtime.tip_state.phase.value}"
            )
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        if (
            visual_runtime is not None
            and duration_s >= 90.0
            and visual_runtime.phase is not VisualLatchPhase.DONE
        ):
            reason = (
                "90-second visual ACT ended before task_done; "
                f"terminal phase={visual_runtime.phase.value}"
            )
            robot.stop(reason)
            raise SafetyStopError(robot.stopped_reason or reason)
        return not stopping[0]
    finally:
        try:
            if stopping[0]:
                robot.stop("operator interrupt during policy execution")
        finally:
            worker.close()



def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.act_duration_s is not None:
        config.act_duration_s = args.act_duration_s
        config.__post_init__()
    run_label = stage_run_label(args.stage, config.act_duration_s)
    tip_config = config.visual_phase_latch.tip_servo
    tip_execute_profile = tip_config.mode is TipServoMode.EXECUTE
    tip_bounded_execute_profile = bool(
        tip_execute_profile
        and tip_config.probe_profile is TipServoProbeProfile.BOUNDED_DESCENT
    )
    tip_open_only_execute_profile = bool(
        tip_execute_profile
        and tip_config.probe_profile is TipServoProbeProfile.OPEN_ONLY
    )
    tip_execute_authorized = bool(
        getattr(args, "operator_authorized_tip_servo_execute", False)
    )
    tip_open_only_authorized = bool(
        getattr(args, "operator_authorized_tip_servo_open_only", False)
    )
    if config.policy_runtime_role == ARM6_SEEKING_ROLE:
        require_arm6_policy_stage_gate(
            config,
            args.stage,
            run_label,
            mark_reviewed=args.mark_reviewed,
            operator_authorized_tip_servo_open_only=tip_open_only_authorized,
            operator_authorized_tip_servo_execute=tip_execute_authorized,
            operator_authorized_skip_replay_for_act_3s=(
                args.operator_authorized_skip_replay_for_act_3s
            ),
            operator_authorized_direct_act_30s=(
                args.operator_authorized_direct_act_30s
            ),
            operator_authorized_direct_act_65s=(
                args.operator_authorized_direct_act_65s
            ),
            operator_authorized_direct_act_90s=(
                args.operator_authorized_direct_act_90s
            ),
            operator_authorized_act_start_reset=(
                args.operator_authorized_act_start_reset
            ),
            operator_authorized_act_ee_preopen=(
                args.operator_authorized_act_ee_preopen
            ),
        )
    if tip_execute_authorized and tip_open_only_authorized:
        raise PermissionError(
            "bounded-descent and OPEN_ONLY tip-servo authorizations are mutually exclusive"
        )
    if tip_execute_authorized and not tip_bounded_execute_profile:
        raise PermissionError(
            "--operator-authorized-tip-servo-execute is valid only for the exact "
            "bounded-descent execute profile; it cannot authorize OPEN_ONLY"
        )
    if tip_open_only_authorized and not tip_open_only_execute_profile:
        raise PermissionError(
            "--operator-authorized-tip-servo-open-only is valid only for the exact "
            "OPEN_ONLY execute profile"
        )
    if tip_bounded_execute_profile:
        require_tip_servo_execute_probe_gate(
            args.stage,
            visual_phase_latch=config.visual_phase_latch,
            operator_authorized=tip_execute_authorized,
        )
    if tip_open_only_execute_profile:
        require_tip_servo_open_only_probe_gate(
            args.stage,
            visual_phase_latch=config.visual_phase_latch,
            operator_authorized=tip_open_only_authorized,
        )
    if tip_execute_authorized and (
        not tip_bounded_execute_profile
        or args.stage != "act"
        or args.mark_reviewed
        or run_label != "act_30s"
    ):
        raise PermissionError(
            "--operator-authorized-tip-servo-execute is valid only for the "
            "exact execute profile during a non-review 30-second ACT"
        )
    if tip_open_only_authorized and (
        not tip_open_only_execute_profile
        or args.stage != "act"
        or args.mark_reviewed
        or run_label != "act_30s"
    ):
        raise PermissionError(
            "--operator-authorized-tip-servo-open-only is valid only for the "
            "exact OPEN_ONLY execute profile during a non-review 30-second ACT"
        )
    if tip_execute_profile and tip_config.allow_close:
        raise PermissionError(
            "tip-servo execute profiles must keep grasp closure disabled"
        )
    if (
        tip_execute_profile
        and args.stage == "act"
        and (
            args.mark_reviewed
            or run_label != "act_30s"
            or (
                tip_bounded_execute_profile
                and not tip_execute_authorized
            )
            or (
                tip_open_only_execute_profile
                and not tip_open_only_authorized
            )
        )
    ):
        raise PermissionError(
            "tip-servo execute requires a fresh one-shot authorization for a "
            "non-review 30-second ACT"
        )
    if (tip_execute_authorized or tip_open_only_authorized) and any(
        (
            args.operator_authorized_skip_replay_for_act_3s,
            args.operator_authorized_direct_act_30s,
            args.operator_authorized_direct_act_65s,
            args.operator_authorized_direct_act_90s,
            args.operator_authorized_act_start_reset,
            args.operator_authorized_act_ee_preopen,
        )
    ):
        raise PermissionError(
            "tip-servo execute authorizations are mutually exclusive with all "
            "skip, direct-ACT, reset, and preopen authorizations"
        )

    replay_skip_authorized = args.operator_authorized_skip_replay_for_act_3s
    if replay_skip_authorized and (args.mark_reviewed or run_label != "act_3s"):
        raise PermissionError(
            "--operator-authorized-skip-replay-for-act-3s is valid only for a non-review "
            "--stage act --act-duration-s 3 run"
        )
    direct_act_30s_authorized = args.operator_authorized_direct_act_30s
    if direct_act_30s_authorized and (args.mark_reviewed or run_label != "act_30s"):
        raise PermissionError(
            "--operator-authorized-direct-act-30s is valid only for a non-review "
            "--stage act --act-duration-s 30 run"
        )
    direct_act_65s_authorized = args.operator_authorized_direct_act_65s
    if direct_act_65s_authorized and (args.mark_reviewed or run_label != "act_65s"):
        raise PermissionError(
            "--operator-authorized-direct-act-65s is valid only for a non-review "
            "--stage act --act-duration-s 65 run"
        )
    direct_act_90s_authorized = args.operator_authorized_direct_act_90s
    if direct_act_90s_authorized and (args.mark_reviewed or run_label != "act_90s"):
        raise PermissionError(
            "--operator-authorized-direct-act-90s is valid only for a non-review "
            "--stage act --act-duration-s 90 run"
        )
    act_start_reset_authorized = args.operator_authorized_act_start_reset
    if act_start_reset_authorized and (args.mark_reviewed or run_label != "act_90s"):
        raise PermissionError(
            "--operator-authorized-act-start-reset is valid only for a non-review "
            "--stage act --act-duration-s 90 run"
        )
    act_ee_preopen_authorized = bool(
        getattr(args, "operator_authorized_act_ee_preopen", False)
    )
    act_ee_preopen_profile_enabled = bool(
        config.grasp_phase_correction.act_start_ee_preopen_enabled
    )
    if act_ee_preopen_authorized and (
        args.mark_reviewed or run_label != "act_90s"
    ):
        raise PermissionError(
            "--operator-authorized-act-ee-preopen is valid only for a non-review "
            "--stage act --act-duration-s 90 run"
        )
    if act_ee_preopen_authorized and act_start_reset_authorized:
        raise PermissionError(
            "ACT start reset and deterministic EE preopen authorizations are mutually exclusive"
        )
    if act_ee_preopen_authorized != act_ee_preopen_profile_enabled:
        raise PermissionError(
            "deterministic ACT start EE preopen requires both the exact fingerprint-bound "
            "profile and a fresh --operator-authorized-act-ee-preopen assertion"
        )
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = capture_deployment_sources(config)
    deployment_fingerprint = source_before.deployment_fingerprint
    if args.mark_reviewed:
        ledger = mark_stage_reviewed(
            output_dir,
            run_label,
            deployment_fingerprint,
            require_video=bool(config.record_video and config.robot.cameras),
            verify_before_commit=lambda: require_unchanged_deployment_sources(
                source_before, capture_deployment_sources(config)
            ),
        )
        logging.info("approved %s in %s", run_label, ledger)
        return 0
    require_acceptance_order(
        output_dir,
        run_label,
        deployment_fingerprint,
        operator_authorized_skip_replay_for_act_3s=replay_skip_authorized,
        operator_authorized_direct_act_30s=direct_act_30s_authorized,
        operator_authorized_tip_servo_probe_act_30s=tip_execute_authorized,
        operator_authorized_tip_servo_open_only_act_30s=(
            tip_open_only_authorized
        ),
        operator_authorized_direct_act_65s=direct_act_65s_authorized,
        operator_authorized_direct_act_90s=direct_act_90s_authorized,
    )
    require_arm_hold_timeout_override_gate(
        args.stage,
        hold_profile_selected=config.robot.arm_timeout_behavior == "hold",
        risk_acknowledged=args.allow_arm_hold_timeout_without_ramp_stop,
        independent_estop_verified=args.independent_estop_verified,
        hold_duration_s=args.hold_duration_s,
    )
    require_hardware_safety_gate(args.stage, args.independent_estop_verified)
    telemetry_path = output_dir / f"{run_label}.jsonl"
    stage_robot_config = robot_config_for_stage(
        config,
        args.stage,
        telemetry_path,
    )
    stopping = [False]
    hardware_connected = [False]

    def request_stop(_signum, _frame):
        stopping[0] = True
        if not hardware_connected[0]:
            raise _PreHardwareInterrupt

    infer = None
    replay_source = None
    robot = None
    recorder = None
    try:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if args.stage in ("shadow", "act"):
            checkpoint = Path(config.checkpoint).expanduser()
            device = "cuda"
            infer = load_policy(
                checkpoint,
                device,
                config.n_action_steps,
                config.task,
                config.smolvla_vlm_snapshot,
                inference_seed=config.inference_seed,
                expected_runtime_role=config.policy_runtime_role,
            )
        if args.stage == "replay":
            # Materialize parquet before taking either controller session. Once
            # execute mode starts, the foreground loop may never block on I/O.
            replay_source = load_episode_actions(config)
        if stopping[0]:
            logging.info("deployment interrupted during offline preload; hardware was not connected")
            return 130
        source_after = capture_deployment_sources(config)
        if stopping[0]:
            logging.info("deployment interrupted during source verification; hardware was not connected")
            return 130
        require_unchanged_deployment_sources(source_before, source_after)
        require_acceptance_order(
            output_dir,
            run_label,
            deployment_fingerprint,
            operator_authorized_skip_replay_for_act_3s=replay_skip_authorized,
            operator_authorized_direct_act_30s=direct_act_30s_authorized,
            operator_authorized_tip_servo_probe_act_30s=tip_execute_authorized,
            operator_authorized_tip_servo_open_only_act_30s=(
                tip_open_only_authorized
            ),
            operator_authorized_direct_act_65s=direct_act_65s_authorized,
            operator_authorized_direct_act_90s=direct_act_90s_authorized,
        )
        if stopping[0]:
            logging.info("deployment interrupted during acceptance verification; hardware was not connected")
            return 130
        invalidate_stage_reviews(output_dir, run_label)
        telemetry_path.unlink(missing_ok=True)
        robot = HexZenohFollower(stage_robot_config)
        recorder = StageVideoRecorder(output_dir, run_label, config.fps, config.record_video)
        robot.record_event(
            "deployment_provenance",
            deployment_fingerprint=deployment_fingerprint,
            deployment_source_stability_digest=source_before.stability_digest,
            checkpoint=str(Path(config.checkpoint).expanduser().absolute()),
            policy_runtime_role=config.policy_runtime_role,
            dataset_repo_id=config.dataset_repo_id,
            dataset_root=str(Path(config.dataset_root).expanduser().absolute()),
            dataset_controller_contract=str(
                Path(config.dataset_controller_contract).expanduser().absolute()
            ),
            task=config.task,
            arm_timeout_behavior=config.robot.arm_timeout_behavior,
            arm_hold_timeout_opt_in=config.robot.allow_arm_hold_timeout_without_ramp_stop,
            arm_hold_timeout_risk_acknowledged=args.allow_arm_hold_timeout_without_ramp_stop,
            independent_physical_power_cut_confirmed=args.independent_estop_verified,
            action_slew_rate_adapter_enabled=config.enable_action_slew_rate_adapter,
            act_chunk_inference_mode=getattr(
                config, "act_chunk_inference_mode", "pipelined"
            ),
            grasp_phase_correction=config.grasp_phase_correction.provenance(),
            visual_phase_latch=visual_phase_latch_provenance(
                config.visual_phase_latch
            ),
            visual_phase_latch_implementation=(
                visual_phase_latch_implementation_manifest()
                if config.visual_phase_latch.enabled
                else None
            ),
            visual_latch_audit=source_before.visual_latch_audit,
            inference_seed=config.inference_seed,
            inference_seed_strategy=(
                "base_seed_plus_zero_based_inference_generation"
            ),
            action_slew_rate_arm_step_limit_rad=config.robot.arm_step_limit_rad,
            action_slew_rate_arm_step_limits_rad=(
                config.robot.resolved_arm_step_limits_rad()
            ),
            action_slew_rate_ee_step_limit_rad=config.robot.ee_step_limit_rad,
            ee_closing_tracking_error_rad=config.robot.ee_closing_tracking_error_rad,
            operator_authorized_skip_replay_for_act_3s=replay_skip_authorized,
            operator_authorized_direct_act_30s=direct_act_30s_authorized,
            operator_authorized_direct_act_65s=direct_act_65s_authorized,
            operator_authorized_direct_act_90s=direct_act_90s_authorized,
            operator_authorized_act_start_reset=act_start_reset_authorized,
            operator_authorized_act_ee_preopen=act_ee_preopen_authorized,
            operator_authorized_tip_servo_probe_act_30s=tip_execute_authorized,
            operator_authorized_tip_servo_open_only_act_30s=(
                tip_open_only_authorized
            ),
        )
        if stopping[0]:
            logging.info("deployment interrupted before hardware connection")
            return 130
        robot.connect()
        hardware_connected[0] = True
        completed = False
        if args.stage == "shadow":
            run_shadow(config, robot, infer, recorder, stopping)
            completed = True
        elif args.stage == "hold":
            completed = run_hold(
                config,
                robot,
                recorder,
                stopping,
                args.hold_duration_s,
                independent_power_cut_verified=args.independent_estop_verified,
            )
        elif args.stage == "replay":
            completed = run_replay(config, robot, recorder, stopping, replay_source)
        else:
            if act_start_reset_authorized:
                run_act_start_reset(config, robot, recorder, stopping)
            elif act_ee_preopen_authorized:
                run_act_ee_preopen(config, robot, recorder, stopping)
            completed = run_act(config, robot, infer, recorder, stopping, config.act_duration_s)
        if completed:
            robot.record_event("stage_complete", stage=run_label)
        else:
            robot.record_event("stage_interrupted", stage=run_label)
    except _PreHardwareInterrupt:
        logging.info("deployment interrupted before hardware connection")
        return 130
    except (SafetyStopError, PermissionError, ConnectionError, TimeoutError, ValueError) as exc:
        logging.error("deployment stopped: %s", exc)
        return 2
    finally:
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            if recorder is not None:
                recorder.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        exit_code = main()
    except (SafetyStopError, PermissionError, ConnectionError, TimeoutError, ValueError, FileNotFoundError) as exc:
        logging.error("deployment refused before hardware connection: %s", exc)
        exit_code = 2
    sys.exit(exit_code)
