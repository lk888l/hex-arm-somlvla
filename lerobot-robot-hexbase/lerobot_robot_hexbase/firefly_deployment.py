"""Staged Firefly Y6 shadow, hold, replay, and policy deployment utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import stat
import threading
import time
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from lerobot.robots.config import RobotConfig

from .firefly_training_attestation import (
    TrainingAttestationError,
    capture_training_inputs,
    verify_training_attestation,
)
from .firefly_tip_servo import (
    TipServoConfig,
    TipServoMode,
    TipServoProbeProfile,
    resolve_open_only_jaw_target,
)
from .firefly_visual_latch import VisualPhaseLatchConfig
from .hex_zenoh_follower import HexZenohFollower, HexZenohFollowerConfig, SafetyStopError
from .smolvlm_bundle import validate_materialized_smolvlm_bundle


Stage = Literal["shadow", "hold", "replay", "act"]
_MOTION_STAGES = frozenset({"replay", "act"})
_ACT_DURATION_TIERS_S = (3.0, 10.0, 30.0, 65.0, 90.0)
_ACCEPTANCE_ORDER = (
    "shadow",
    "hold",
    "replay",
    "act_3s",
    "act_10s",
    "act_30s",
    "act_65s",
    "act_90s",
)
_REVIEW_SCHEMA_VERSION = 4
_DEPLOYMENT_FINGERPRINT_SCHEMA_VERSION = 6
_DEPLOYMENT_SOURCE_GUARD_SCHEMA_VERSION = 3
_EXPECTED_ARM_JOINTS = tuple(f"joint_{index}" for index in range(1, 7))
_EXPECTED_EE_JOINTS = ("jaw_joint1",)
_FULLCYCLE_V4_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4"
_ARM_ONLY_V1_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4_arm_only_v1"
_LEGACY_POLICY_RUNTIME_ROLE = "legacy_action7"
_ARM6_POLICY_RUNTIME_ROLE = "arm6_seeking"
_FULLCYCLE_V4_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
_FULLCYCLE_V4_TOTAL_EPISODES = 50
_FULLCYCLE_V4_TOTAL_FRAMES = 91836
_VISUAL_LATCH_AUDIT_SCHEMA_VERSION = 2
_TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION = 3
_TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION = 6
_TIP_SERVO_NEGATIVE_CONTROL_SHA256 = "cf46da3d3ee50cea27ff3a2743fb9ce90a7d10e72e10c7508f30dab043273943"
_TIP_SERVO_NEGATIVE_CONTROL_SIZE = 11278473
_TIP_SERVO_NEGATIVE_CONTROL_RECORDS = 2932
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SHA256 = (
    "e31d0834d29c6c06dd74224ebda525f926bf8d69ae5a6ae65b5abdff7a352a99"
)
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE = 3960524
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS = 976
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_COMMAND_CYCLES = 906
_TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID = 487
_TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID = 494
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "firefly_deployment_hold_fullcycle_v4_100k_tip_probe_v1"
    / "act_30s.jsonl"
)
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_WINDOW_STATISTICS = {
    "observed_x_norm": {
        "min": 0.5239913324787889,
        "max": 0.5297030325554855,
    },
    "observed_y_norm": {
        "min": 0.44874340405581986,
        "max": 0.45366030222943005,
    },
    "observed_area_fraction": {
        "min": 0.017776692708333333,
        "max": 0.018248697916666667,
    },
    "observed_jaw_rad": {
        "min": 0.30454832315444946,
        "max": 0.3047424554824829,
    },
    "observed_max_arm_speed_rad_s": 0.06498425491022561,
}
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_HISTORICAL_STATISTICS = {
    "policy_arm_motion_observed": True,
    "minimum_measured_tcp_delta_z_m": -0.02712412178516388,
    "final_jaw_target_min_rad": 0.28363317251205444,
    "final_jaw_target_max_rad": 0.41,
    "final_jaw_target_below_capture_cycles": 549,
    "final_jaw_target_above_capture_cycles": 344,
}
_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_REPLAY_STATISTICS = {
    "capture_jaw_target_rad": 0.30926513671875,
    "replayed_seeking_cycles": 491,
    "published_jaw_min_rad": 0.30926513671875,
    "published_jaw_max_rad": 0.30926513671875,
    "suppressed_lower_jaw_proposal_cycles": 463,
    "suppressed_early_open_proposal_cycles": 15,
    "takeover_boundary_published_jaw_target_rad": 0.30926513671875,
    "takeover_boundary_still_capture_hold": True,
    "first_open_requires_takeover_and_settle": True,
    "opened_targets_rad": [
        0.30926513671875,
        0.32576513671875,
        0.34226513671875003,
        0.35876513671875004,
        0.37526513671875006,
        0.3917651367187501,
        0.4082651367187501,
        0.41,
    ],
    "opened_target_monotonic_nondecreasing": True,
    "opened_target_step_limit_pass": True,
    "open_target_reached": True,
    "adversarial_reclose_proposal_rad": 0.28363317251205444,
    "adversarial_reclose_resolved_target_rad": 0.34226513671875003,
    "adversarial_reclose_suppressed": True,
    "fault_hold_non_decreasing": True,
    "seeking_timeout_ms": 20_000.0,
    "seeking_timeout_holds_capture": True,
    "counterfactual_open_response_modeled": False,
}
_VISUAL_LATCH_AUDIT_KIND = "firefly_y6_fullcycle_v4_visual_latch_replay"
_FULLCYCLE_V4_START_CENTER_RAD = (
    -0.047621,
    -1.372725,
    2.938167,
    0.375869,
    -0.050262,
    -0.059902,
    0.314108,
)
_FULLCYCLE_V4_START_TOLERANCE_RAD = (
    0.163024,
    0.180633,
    0.121334,
    0.106397,
    0.116914,
    0.121981,
    0.032197,
)


def _validate_sha256_hex(value: object, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


@dataclass
class GraspPhaseCorrectionConfig:
    """One-shot, data-derived EE phase correction for the validated fast90 profile."""

    enabled: bool = False
    pickup_center_rad: list[float] = field(default_factory=list)
    pickup_tolerance_rad: list[float] = field(default_factory=list)
    premature_close_threshold_rad: float = 0.310
    pickup_open_threshold_rad: float = 0.350
    pickup_confirm_chunks: int = 6
    close_target_rad: float = 0.320
    pregrasp_open_target_rad: float | None = None
    pregrasp_open_ready_rad: float | None = None
    raw_open_progress_deadline_chunks: int | None = None
    pickup_progress_deadline_chunks: int | None = None
    release_progress_deadline_chunks: int | None = None
    act_start_center_rad: list[float] = field(default_factory=list)
    act_start_tolerance_rad: list[float] = field(default_factory=list)
    act_start_ee_preopen_enabled: bool = False
    release_joint_1_min_rad: float = 0.800
    release_joint_2_min_rad: float = -0.200
    release_joint_3_min_rad: float = 1.900
    release_joint_3_max_rad: float = 2.600
    release_joint_4_max_rad: float = 0.400
    release_open_threshold_rad: float = 0.360
    release_confirm_chunks: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("grasp phase correction enabled must be a boolean")
        if not isinstance(self.act_start_ee_preopen_enabled, bool):
            raise ValueError("ACT start EE preopen enabled must be a boolean")
        for name, values in (
            ("pickup_center_rad", self.pickup_center_rad),
            ("pickup_tolerance_rad", self.pickup_tolerance_rad),
            ("act_start_center_rad", self.act_start_center_rad),
            ("act_start_tolerance_rad", self.act_start_tolerance_rad),
        ):
            if values and (
                len(values) != (7 if name.startswith("act_start_") else 6)
                or not all(math.isfinite(float(value)) for value in values)
            ):
                expected = "seven" if name.startswith("act_start_") else "six"
                raise ValueError(f"{name} must contain {expected} finite values")
        if self.enabled and (
            len(self.pickup_center_rad) != 6 or len(self.pickup_tolerance_rad) != 6
        ):
            raise ValueError(
                "enabled grasp phase correction requires six pickup centers and tolerances"
            )
        if self.pickup_tolerance_rad and not all(
            float(value) > 0 for value in self.pickup_tolerance_rad
        ):
            raise ValueError("pickup_tolerance_rad values must be positive")
        if bool(self.act_start_center_rad) != bool(self.act_start_tolerance_rad):
            raise ValueError(
                "act_start_center_rad and act_start_tolerance_rad must be configured together"
            )
        if self.act_start_tolerance_rad and not all(
            float(value) > 0 for value in self.act_start_tolerance_rad
        ):
            raise ValueError("act_start_tolerance_rad values must be positive")
        # A start-pose gate is useful independently of action correction. In
        # particular, fullcycle-v4 uses the demonstrated 50-episode start
        # envelope while leaving every policy action dimension untouched.
        if (
            self.pregrasp_open_target_rad is not None
            or self.pregrasp_open_ready_rad is not None
            or self.act_start_ee_preopen_enabled
        ) and not self.enabled:
            raise ValueError(
                "ACT start/open assistance is valid only when grasp phase correction is enabled"
            )
        if self.act_start_ee_preopen_enabled and (
            not self.act_start_center_rad
            or self.pregrasp_open_target_rad is None
            or self.pregrasp_open_ready_rad is None
        ):
            raise ValueError(
                "ACT start EE preopen requires the start gate and pregrasp open target/ready values"
            )
        for name in (
            "premature_close_threshold_rad",
            "pickup_open_threshold_rad",
            "close_target_rad",
            "release_joint_1_min_rad",
            "release_joint_2_min_rad",
            "release_joint_3_min_rad",
            "release_joint_3_max_rad",
            "release_joint_4_max_rad",
            "release_open_threshold_rad",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not (
            0.0 <= self.premature_close_threshold_rad
            < self.close_target_rad
            < self.pickup_open_threshold_rad
            <= self.release_open_threshold_rad
            <= 0.6
        ):
            raise ValueError("grasp phase EE thresholds are not strictly ordered")
        if self.pregrasp_open_target_rad is not None and not (
            self.pickup_open_threshold_rad
            <= float(self.pregrasp_open_target_rad)
            <= 0.6
        ):
            raise ValueError(
                "pregrasp_open_target_rad must be between pickup_open_threshold_rad and 0.6"
            )
        if bool(self.pregrasp_open_target_rad is not None) != bool(
            self.pregrasp_open_ready_rad is not None
        ):
            raise ValueError(
                "pregrasp_open_target_rad and pregrasp_open_ready_rad must be configured together"
            )
        if self.pregrasp_open_ready_rad is not None and not (
            self.pickup_open_threshold_rad
            <= float(self.pregrasp_open_ready_rad)
            <= float(self.pregrasp_open_target_rad)
        ):
            raise ValueError(
                "pregrasp_open_ready_rad must be between pickup_open_threshold_rad "
                "and pregrasp_open_target_rad"
            )
        if self.release_joint_3_min_rad >= self.release_joint_3_max_rad:
            raise ValueError("release joint 3 bounds are not ordered")
        for name in ("pickup_confirm_chunks", "release_confirm_chunks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "raw_open_progress_deadline_chunks",
            "pickup_progress_deadline_chunks",
            "release_progress_deadline_chunks",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or null")

    def provenance(self) -> dict[str, object]:
        # Preserve the already-reviewed v1/v2 fingerprint payload when the
        # v3-only fields remain at their defaults.
        optional_defaults = {
            "pregrasp_open_target_rad": None,
            "pregrasp_open_ready_rad": None,
            "raw_open_progress_deadline_chunks": None,
            "pickup_progress_deadline_chunks": None,
            "release_progress_deadline_chunks": None,
            "act_start_center_rad": [],
            "act_start_tolerance_rad": [],
            "act_start_ee_preopen_enabled": False,
        }
        payload = {}
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if (
                field_.name in optional_defaults
                and value == optional_defaults[field_.name]
            ):
                continue
            payload[field_.name] = _canonical_fingerprint_value(value)
        return payload


@dataclass
class FireflyDeploymentConfig:
    """Single checked-in configuration shared by all deployment stages."""

    robot: RobotConfig
    checkpoint: str = "CHANGE_ME_VALIDATED_V2_CHECKPOINT"
    policy_runtime_role: str = _LEGACY_POLICY_RUNTIME_ROLE
    smolvla_vlm_snapshot: str | None = None
    dataset_repo_id: str = "local/firefly_y6_manual_validated_v2"
    dataset_root: str = "CHANGE_ME_VALIDATED_V2_DATASET_ROOT"
    dataset_controller_contract: str = "CHANGE_ME_VALIDATED_V2_CONTROLLER_CONTRACT"
    visual_audit_source_repo_id: str | None = None
    visual_audit_source_root: str | None = None
    arm6_run_commit_sha: str | None = None
    task: str = "Pick up the object and place it in the target area."
    fps: int = 30
    n_action_steps: int = 10
    inference_seed: int | None = None
    act_chunk_inference_mode: str = "pipelined"
    enable_action_slew_rate_adapter: bool = False
    grasp_phase_correction: GraspPhaseCorrectionConfig = field(
        default_factory=GraspPhaseCorrectionConfig
    )
    visual_phase_latch: VisualPhaseLatchConfig = field(
        default_factory=VisualPhaseLatchConfig
    )
    visual_latch_audit_path: str | None = None
    visual_latch_audit_sha256: str | None = None
    replay_episode: int = 0
    replay_speed: float = 0.5
    act_duration_s: float = 3.0
    output_dir: str = "outputs/firefly_deployment"
    record_video: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.robot, HexZenohFollowerConfig):
            raise ValueError("robot must be a hex_zenoh_follower configuration")
        if tuple(self.robot.arm_joint_names) != _EXPECTED_ARM_JOINTS:
            raise ValueError(
                f"deployment arm joint order must be {list(_EXPECTED_ARM_JOINTS)!r}"
            )
        if tuple(self.robot.ee_joint_names) != _EXPECTED_EE_JOINTS:
            raise ValueError(
                f"deployment EE joint order must be {list(_EXPECTED_EE_JOINTS)!r}"
            )
        if type(self.policy_runtime_role) is not str or self.policy_runtime_role not in {
            _LEGACY_POLICY_RUNTIME_ROLE,
            _ARM6_POLICY_RUNTIME_ROLE,
        }:
            raise ValueError(
                "policy_runtime_role must be exactly "
                f"{_LEGACY_POLICY_RUNTIME_ROLE!r} or {_ARM6_POLICY_RUNTIME_ROLE!r}"
            )
        visual_source_repo_configured = (
            type(self.visual_audit_source_repo_id) is str
            and bool(self.visual_audit_source_repo_id)
        )
        visual_source_root_configured = (
            type(self.visual_audit_source_root) is str
            and bool(self.visual_audit_source_root)
        )
        if visual_source_repo_configured != visual_source_root_configured:
            raise ValueError(
                "visual_audit_source_repo_id and visual_audit_source_root "
                "must be configured together"
            )
        if self.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE:
            if not visual_source_repo_configured:
                raise ValueError(
                    "arm6_seeking requires a distinct 7-D visual audit source repo and root"
                )
            assert isinstance(self.visual_audit_source_root, str)
            if not Path(self.visual_audit_source_root).expanduser().is_absolute():
                raise ValueError("visual_audit_source_root must be an absolute path")
            if self.visual_audit_source_repo_id != _FULLCYCLE_V4_DATASET_REPO_ID:
                raise ValueError(
                    "arm6_seeking visual audit source repo must be the original "
                    "fullcycle-v4 7-D dataset"
                )
            if self.dataset_repo_id != _ARM_ONLY_V1_DATASET_REPO_ID:
                raise ValueError(
                    "arm6_seeking policy dataset repo must be the arm-only-v1 projection"
                )
            if self.task != _FULLCYCLE_V4_TASK:
                raise ValueError(
                    "arm6_seeking task must exactly preserve the fullcycle-v4 task"
                )
            if _absolute_without_resolving(Path(self.dataset_root)) == (
                _absolute_without_resolving(Path(self.visual_audit_source_root))
            ):
                raise ValueError(
                    "arm6_seeking policy dataset and 7-D visual audit source must be distinct"
                )
            _validate_sha256_hex(
                self.arm6_run_commit_sha,
                label="arm6_run_commit_sha",
            )
        elif (
            visual_source_repo_configured
            or self.arm6_run_commit_sha is not None
        ):
            raise ValueError(
                "visual audit source separation and arm6_run_commit_sha are valid only "
                "for policy_runtime_role='arm6_seeking'"
            )
        if self.fps != 30:
            raise ValueError("Firefly deployment command rate is fixed at 30 Hz")
        if self.n_action_steps != 10:
            raise ValueError("initial policy deployment is fixed at n_action_steps=10")
        if self.inference_seed is not None and (
            isinstance(self.inference_seed, bool)
            or not isinstance(self.inference_seed, int)
            or not 0 <= self.inference_seed <= 0xFFFFFFFF
        ):
            raise ValueError("inference_seed must be an unsigned 32-bit integer or null")
        if self.act_chunk_inference_mode not in ("pipelined", "synchronous_hold"):
            raise ValueError(
                "act_chunk_inference_mode must be 'pipelined' or 'synchronous_hold'"
            )
        if not isinstance(self.enable_action_slew_rate_adapter, bool):
            raise ValueError("enable_action_slew_rate_adapter must be a boolean")
        if not isinstance(self.grasp_phase_correction, GraspPhaseCorrectionConfig):
            raise ValueError(
                "grasp_phase_correction must be a GraspPhaseCorrectionConfig"
            )
        if not isinstance(self.visual_phase_latch, VisualPhaseLatchConfig):
            raise ValueError("visual_phase_latch must be a VisualPhaseLatchConfig")
        if not isinstance(self.visual_phase_latch.enabled, bool):
            raise ValueError("visual phase latch enabled must be a boolean")
        if self.visual_phase_latch.enabled and self.grasp_phase_correction.enabled:
            raise ValueError(
                "visual phase latch and fixed-joint GraspPhaseCorrector are mutually exclusive"
            )
        audit_path_configured = isinstance(self.visual_latch_audit_path, str) and bool(
            self.visual_latch_audit_path.strip()
        )
        audit_digest_configured = isinstance(
            self.visual_latch_audit_sha256, str
        ) and bool(self.visual_latch_audit_sha256)
        if audit_path_configured != audit_digest_configured:
            raise ValueError(
                "visual_latch_audit_path and visual_latch_audit_sha256 must be configured together"
            )
        if self.visual_phase_latch.enabled and not audit_path_configured:
            raise ValueError(
                "enabled visual phase latch requires a fingerprint-bound offline audit path and SHA-256"
            )
        if audit_path_configured:
            assert isinstance(self.visual_latch_audit_path, str)
            if not Path(self.visual_latch_audit_path).expanduser().is_absolute():
                raise ValueError("visual_latch_audit_path must be an absolute path")
            _validate_sha256_hex(
                self.visual_latch_audit_sha256,
                label="visual_latch_audit_sha256",
            )
        if self.replay_episode != 0:
            raise ValueError("first hardware replay is fixed to episode 0")
        if self.replay_speed != 0.5:
            raise ValueError("first hardware replay is fixed to 0.5x speed")
        if self.act_duration_s not in _ACT_DURATION_TIERS_S:
            raise ValueError(f"ACT duration must be one of {_ACT_DURATION_TIERS_S}")
        if self.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE:
            if self.act_duration_s != 30.0:
                raise ValueError("arm6_seeking ACT duration is fixed at 30 seconds")
            if (
                not self.visual_phase_latch.enabled
                or self.visual_phase_latch.tip_servo
                != _fullcycle_v4_open_only_tip_servo_config()
            ):
                raise ValueError(
                    "arm6_seeking requires the exact enabled OPEN_ONLY zero-arm, "
                    "zero-descent, no-close visual profile"
                )


def _fullcycle_v4_open_only_tip_servo_config() -> TipServoConfig:
    """Return the one exact closed-capture, jaw-open-only probe profile."""

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


def _schema6_open_only_axis_authority() -> dict[str, object]:
    """Return the exact arm6 policy/tip ownership contract emitted by the auditor."""

    arm_axes = [f"joint_{index}.pos" for index in range(1, 7)]
    return {
        "scope": "axis_partitioned_open_only",
        "audit_schema_version": 6,
        "legacy_schema5_authorized": False,
        "policy_role": _ARM6_POLICY_RUNTIME_ROLE,
        "policy_numeric_observation_axes": arm_axes,
        "policy_numeric_action_axes": list(arm_axes),
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


def require_tip_servo_execute_probe_gate(
    stage: Stage,
    *,
    visual_phase_latch: VisualPhaseLatchConfig,
    operator_authorized: bool,
) -> None:
    """Require a fresh, act-only authorization for the bounded no-close probe."""

    if not isinstance(visual_phase_latch, VisualPhaseLatchConfig):
        raise TypeError("tip-servo execute gate requires VisualPhaseLatchConfig")
    if type(operator_authorized) is not bool:
        raise ValueError("tip-servo execute authorization must be a boolean")
    tip = visual_phase_latch.tip_servo
    if tip.mode is not TipServoMode.EXECUTE:
        if operator_authorized:
            raise PermissionError(
                "tip-servo execute authorization is invalid unless mode=execute"
            )
        return
    expected = TipServoConfig(
        mode=TipServoMode.EXECUTE,
        allow_close=False,
        max_total_descent_m=0.025,
    )
    if tip != expected:
        raise PermissionError(
            "tip-servo execute is restricted to the exact reviewed 25-mm, "
            "allow_close=false probe profile"
        )
    if stage != "act":
        if operator_authorized:
            raise PermissionError(
                "tip-servo execute authorization is valid only for the ACT stage"
            )
        return
    if not operator_authorized:
        raise PermissionError(
            "tip-servo execute requires a fresh "
            "--operator-authorized-tip-servo-execute acknowledgement"
        )
def require_tip_servo_open_only_probe_gate(
    stage: Stage,
    *,
    visual_phase_latch: VisualPhaseLatchConfig,
    operator_authorized: bool,
) -> None:
    """Require a fresh, act-only authorization for the zero-arm open-only probe."""

    if not isinstance(visual_phase_latch, VisualPhaseLatchConfig):
        raise TypeError("tip-servo open-only gate requires VisualPhaseLatchConfig")
    if type(operator_authorized) is not bool:
        raise ValueError("tip-servo open-only authorization must be a boolean")
    tip = visual_phase_latch.tip_servo
    if tip.mode is not TipServoMode.EXECUTE:
        if operator_authorized:
            raise PermissionError(
                "tip-servo open-only authorization is invalid unless mode=execute"
            )
        return
    if tip != _fullcycle_v4_open_only_tip_servo_config():
        raise PermissionError(
            "tip-servo open-only execute is restricted to the exact reviewed "
            "closed-jaw capture, zero-arm, zero-descent, allow_close=false profile"
        )
    if stage != "act":
        if operator_authorized:
            raise PermissionError(
                "tip-servo open-only authorization is valid only for the ACT stage"
            )
        return
    if not operator_authorized:
        raise PermissionError(
            "tip-servo open-only execute requires a fresh "
            "--operator-authorized-tip-servo-open-only acknowledgement"
        )


def require_arm6_policy_stage_gate(
    config: FireflyDeploymentConfig,
    stage: Stage,
    run_label: str,
    *,
    mark_reviewed: bool,
    operator_authorized_tip_servo_open_only: bool,
    operator_authorized_tip_servo_execute: bool,
    operator_authorized_skip_replay_for_act_3s: bool,
    operator_authorized_direct_act_30s: bool,
    operator_authorized_direct_act_65s: bool,
    operator_authorized_direct_act_90s: bool,
    operator_authorized_act_start_reset: bool,
    operator_authorized_act_ee_preopen: bool,
) -> None:
    """Restrict arm6 policy authority to shadow, HOLD, or one OPEN_ONLY ACT."""

    if not isinstance(config, FireflyDeploymentConfig):
        raise TypeError("arm6 stage gate requires FireflyDeploymentConfig")
    booleans = {
        "mark_reviewed": mark_reviewed,
        "operator_authorized_tip_servo_open_only": (
            operator_authorized_tip_servo_open_only
        ),
        "operator_authorized_tip_servo_execute": operator_authorized_tip_servo_execute,
        "operator_authorized_skip_replay_for_act_3s": (
            operator_authorized_skip_replay_for_act_3s
        ),
        "operator_authorized_direct_act_30s": operator_authorized_direct_act_30s,
        "operator_authorized_direct_act_65s": operator_authorized_direct_act_65s,
        "operator_authorized_direct_act_90s": operator_authorized_direct_act_90s,
        "operator_authorized_act_start_reset": operator_authorized_act_start_reset,
        "operator_authorized_act_ee_preopen": operator_authorized_act_ee_preopen,
    }
    for name, value in booleans.items():
        if type(value) is not bool:
            raise ValueError(f"arm6 stage gate {name} must be a boolean")
    if config.policy_runtime_role != _ARM6_POLICY_RUNTIME_ROLE:
        return
    if (
        not config.visual_phase_latch.enabled
        or config.visual_phase_latch.tip_servo
        != _fullcycle_v4_open_only_tip_servo_config()
    ):
        raise PermissionError(
            "arm6_seeking permits only the exact enabled OPEN_ONLY zero-arm, "
            "zero-descent, no-close profile"
        )
    if stage not in {"shadow", "hold", "act"}:
        raise PermissionError("arm6_seeking forbids dataset replay")
    forbidden_authorizations = {
        name: value
        for name, value in booleans.items()
        if name
        not in {
            "mark_reviewed",
            "operator_authorized_tip_servo_open_only",
        }
        and value
    }
    if forbidden_authorizations:
        raise PermissionError(
            "arm6_seeking forbids replay skips, direct ACT, bounded descent, "
            "reset, and preopen authorizations"
        )
    if stage == "act":
        if (
            run_label != "act_30s"
            or mark_reviewed
            or not operator_authorized_tip_servo_open_only
        ):
            raise PermissionError(
                "arm6_seeking ACT is limited to one non-review 30-second OPEN_ONLY "
                "run with a fresh one-shot authorization"
            )
    elif operator_authorized_tip_servo_open_only:
        raise PermissionError(
            "arm6_seeking OPEN_ONLY authorization is valid only for the 30-second ACT"
        )




def require_hardware_safety_gate(stage: Stage, independent_estop_verified: bool) -> None:
    """Block model or replay motion unless the operator gives an ephemeral CLI confirmation."""
    if stage in _MOTION_STAGES and not independent_estop_verified:
        raise PermissionError(
            f"stage {stage!r} is disabled: verify an independent physical power-cut or emergency-stop "
            "and pass --independent-estop-verified. pc-always-clear.yaml does not satisfy this gate."
        )


def require_arm_hold_timeout_override_gate(
    stage: Stage,
    *,
    hold_profile_selected: bool,
    risk_acknowledged: bool,
    independent_estop_verified: bool,
    hold_duration_s: float,
) -> None:
    """Require an ephemeral risk acknowledgement for every HOLD execute run."""
    if stage == "shadow":
        if risk_acknowledged:
            raise PermissionError(
                "the arm HOLD risk acknowledgement is execute-only and is invalid for shadow"
            )
        return
    if stage == "hold" and (not math.isfinite(hold_duration_s) or hold_duration_s <= 0):
        raise ValueError("hold duration must be positive and finite")
    if not hold_profile_selected:
        if risk_acknowledged:
            raise PermissionError(
                "--allow-arm-hold-timeout-without-ramp-stop is invalid for a RAMP_STOP profile"
            )
        return
    if not risk_acknowledged:
        raise PermissionError(
            "the persistent arm HOLD profile requires a fresh "
            "--allow-arm-hold-timeout-without-ramp-stop acknowledgement for every execute run"
        )
    if not independent_estop_verified:
        raise PermissionError(
            "the persistent arm HOLD profile requires an independent physical power cut and "
            "--independent-estop-verified"
        )
    if stage == "hold" and hold_duration_s > 1.0:
        raise PermissionError("the initial arm HOLD stage is limited to 1.0 second")


def stage_run_label(stage: Stage, act_duration_s: float) -> str:
    if stage != "act":
        return stage
    if act_duration_s not in _ACT_DURATION_TIERS_S:
        raise ValueError(f"ACT duration must be one of {_ACT_DURATION_TIERS_S}")
    return f"act_{act_duration_s:g}s"


def require_acceptance_order(
    output_dir: Path,
    run_label: str,
    deployment_fingerprint: str,
    *,
    operator_authorized_skip_replay_for_act_3s: bool = False,
    operator_authorized_direct_act_30s: bool = False,
    operator_authorized_tip_servo_probe_act_30s: bool = False,
    operator_authorized_tip_servo_open_only_act_30s: bool = False,
    operator_authorized_direct_act_65s: bool = False,
    operator_authorized_direct_act_90s: bool = False,
) -> None:
    """Require telemetry-bound human reviews for every preceding hardware stage."""
    _validate_deployment_fingerprint(deployment_fingerprint)
    if run_label not in _ACCEPTANCE_ORDER:
        raise ValueError(f"unknown deployment stage label {run_label!r}")
    if not isinstance(operator_authorized_skip_replay_for_act_3s, bool):
        raise ValueError("operator replay-skip authorization must be a boolean")
    if operator_authorized_skip_replay_for_act_3s and run_label != "act_3s":
        raise PermissionError(
            "the operator-authorized replay skip is valid only for the 3-second policy run"
        )
    if not isinstance(operator_authorized_direct_act_30s, bool):
        raise ValueError("operator direct-ACT-30s authorization must be a boolean")
    if operator_authorized_direct_act_30s and run_label != "act_30s":
        raise PermissionError(
            "the operator-authorized direct ACT run is valid only for the 30-second policy run"
        )
    if not isinstance(operator_authorized_tip_servo_probe_act_30s, bool):
        raise ValueError("tip-servo probe ACT-30s authorization must be a boolean")
    if (
        operator_authorized_tip_servo_probe_act_30s
        and run_label != "act_30s"
    ):
        raise PermissionError(
            "the tip-servo probe authorization is valid only for act_30s"
        )
    if (
        operator_authorized_tip_servo_probe_act_30s
        and operator_authorized_direct_act_30s
    ):
        raise PermissionError(
            "the tip-servo probe authorization cannot reuse or combine with the "
            "legacy direct-ACT-30s authorization"
        )
    if not isinstance(operator_authorized_tip_servo_open_only_act_30s, bool):
        raise ValueError("tip-servo open-only ACT-30s authorization must be a boolean")
    if (
        operator_authorized_tip_servo_open_only_act_30s
        and run_label != "act_30s"
    ):
        raise PermissionError(
            "the tip-servo open-only authorization is valid only for act_30s"
        )
    if operator_authorized_tip_servo_open_only_act_30s and (
        operator_authorized_direct_act_30s
        or operator_authorized_tip_servo_probe_act_30s
    ):
        raise PermissionError(
            "the tip-servo open-only authorization is independent and cannot "
            "reuse or combine with legacy ACT-30s authorizations"
        )
    if not isinstance(operator_authorized_direct_act_65s, bool):
        raise ValueError("operator direct-ACT-65s authorization must be a boolean")
    if operator_authorized_direct_act_65s and run_label != "act_65s":
        raise PermissionError(
            "the operator-authorized direct ACT run is valid only for the 65-second policy run"
        )
    if not isinstance(operator_authorized_direct_act_90s, bool):
        raise ValueError("operator direct-ACT-90s authorization must be a boolean")
    if operator_authorized_direct_act_90s and run_label != "act_90s":
        raise PermissionError(
            "the operator-authorized direct ACT run is valid only for the 90-second policy run"
        )
    index = _ACCEPTANCE_ORDER.index(run_label)
    if index == 0:
        return
    ledger_path = output_dir / "acceptance.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if not isinstance(ledger, dict):
        raise PermissionError(f"stage {run_label!r} is locked by malformed ledger {ledger_path}")
    for previous in _ACCEPTANCE_ORDER[:index]:
        if (
            previous == "replay"
            and operator_authorized_skip_replay_for_act_3s
            and previous not in ledger
        ):
            continue
        if (
            previous in {"replay", "act_3s", "act_10s"}
            and (
                operator_authorized_direct_act_30s
                or operator_authorized_tip_servo_probe_act_30s
                or operator_authorized_tip_servo_open_only_act_30s
            )
            and previous not in ledger
        ):
            continue
        if (
            previous in {"replay", "act_3s", "act_10s", "act_30s"}
            and operator_authorized_direct_act_65s
            and previous not in ledger
        ):
            continue
        if operator_authorized_direct_act_90s and previous not in ledger:
            # This fresh authorization skips absent stages only; it never
            # fabricates or ignores an existing ledger entry.
            continue
        entry = ledger.get(previous)
        if not isinstance(entry, dict) or not entry.get("reviewed", False):
            raise PermissionError(
                f"stage {run_label!r} is locked until {previous!r} logs and video have been reviewed; "
                f"then run this command with --stage {previous.split('_')[0]} --mark-reviewed"
            )
        if entry.get("schema_version") != _REVIEW_SCHEMA_VERSION:
            raise PermissionError(
                f"stage {run_label!r} is locked because the {previous!r} review marker is legacy "
                "and is not bound to safe telemetry and deployment provenance; review a new run"
            )
        if entry.get("deployment_fingerprint") != deployment_fingerprint:
            raise PermissionError(
                f"stage {run_label!r} is locked because the {previous!r} review belongs to "
                "a different checkpoint, task, dataset, or robot configuration; restart at shadow"
            )
        try:
            audit = _audit_stage_telemetry(output_dir, previous)
        except (FileNotFoundError, ValueError) as exc:
            raise PermissionError(
                f"stage {run_label!r} is locked because {previous!r} telemetry is no longer valid: {exc}"
            ) from exc
        if audit["deployment_fingerprint"] != deployment_fingerprint:
            raise PermissionError(
                f"stage {run_label!r} is locked because {previous!r} telemetry belongs to "
                "a different deployment fingerprint; restart at shadow"
            )
        if entry.get("telemetry_sha256") != audit["telemetry_sha256"]:
            raise PermissionError(
                f"stage {run_label!r} is locked because {previous!r} telemetry changed after review"
            )
        if entry.get("successful_cycle_count") != audit["successful_cycle_count"]:
            raise PermissionError(
                f"stage {run_label!r} is locked because the {previous!r} review summary is inconsistent"
            )
        videos = entry.get("videos", [])
        if not isinstance(videos, list):
            raise PermissionError(
                f"stage {run_label!r} is locked because the {previous!r} video review is malformed"
            )
        for video in videos:
            if not isinstance(video, dict) or "name" not in video or "sha256" not in video:
                raise PermissionError(
                    f"stage {run_label!r} is locked because the {previous!r} video review is malformed"
                )
            video_path = output_dir / video["name"]
            try:
                current_video_digest = _sha256_file(video_path)
            except (FileNotFoundError, PermissionError, ValueError):
                current_video_digest = None
            if current_video_digest != video["sha256"]:
                raise PermissionError(
                    f"stage {run_label!r} is locked because reviewed video {video_path.name!r} "
                    "is missing or changed"
                )


@dataclass(frozen=True)
class CheckpointProvenance:
    checkpoint: Path
    policy_type: Literal["act", "smolvla"]
    dataset_repo_id: str
    dataset_root: Path
    smolvla_vlm_snapshot: Path | None


@dataclass(frozen=True)
class DeploymentSourceCapture:
    deployment_fingerprint: str
    stability_digest: str
    visual_latch_audit: dict[str, object] | None = None
    arm6_attestation: dict[str, object] | None = None


@dataclass(frozen=True)
class _Arm6DeploymentBinding:
    selected_checkpoint: Path
    source_binding: dict[str, object]
    controlled_artifacts: tuple[dict[str, object], ...]


Arm6AttestationLoader = Callable[[], object]


def _load_fixed_arm6_attestation() -> object:
    """Lazy-load the fixed consumer so legacy deployments retain no dependency."""

    try:
        from .firefly_arm6_attestation import (
            FireflyArm6AttestationError,
            capture_firefly_arm6_attestation,
        )
    except ImportError as exc:
        raise PermissionError(
            "arm6_seeking attestation consumer is unavailable"
        ) from exc
    try:
        return capture_firefly_arm6_attestation()
    except FireflyArm6AttestationError as exc:
        raise PermissionError(f"arm6_seeking attestation refused: {exc}") from exc


def _validate_arm6_attestation_binding(
    config: FireflyDeploymentConfig,
    loader: Arm6AttestationLoader,
) -> _Arm6DeploymentBinding:
    """Cross-bind the fixed consumer result to the deployment configuration."""

    try:
        capture = loader()
        selected_checkpoint_value = capture.selected_checkpoint
        selected_step = capture.selected_step
        source_binding_value = capture.source_binding()
        controlled_artifacts_value = capture.controlled_artifacts()
    except (AttributeError, TypeError, ValueError) as exc:
        raise PermissionError(
            "arm6_seeking consumer returned an invalid capture API"
        ) from exc
    if type(selected_step) is not int or selected_step != 20_000:
        raise PermissionError("arm6_seeking consumer must select checkpoint step 20000")
    if not isinstance(selected_checkpoint_value, Path):
        raise PermissionError("arm6_seeking selected_checkpoint must be a Path")
    selected_checkpoint = _require_no_symlink_components(
        selected_checkpoint_value,
        label="arm6 selected checkpoint",
    )
    configured_checkpoint = _require_no_symlink_components(
        Path(config.checkpoint),
        label="configured arm6 checkpoint",
    )
    if selected_checkpoint != configured_checkpoint:
        raise PermissionError(
            "configured checkpoint is not the consumer-selected arm6 checkpoint"
        )
    if not isinstance(source_binding_value, dict) or set(source_binding_value) != {
        "policy_attestation",
        "projection_binding",
    }:
        raise PermissionError(
            "arm6 source_binding must contain exactly policy_attestation and projection_binding"
        )
    source_binding = _canonical_fingerprint_value(source_binding_value)
    if not isinstance(source_binding, dict):
        raise PermissionError("arm6 source_binding is not canonical")
    policy = source_binding.get("policy_attestation")
    projection = source_binding.get("projection_binding")
    if not isinstance(policy, dict) or not isinstance(projection, dict):
        raise PermissionError("arm6 source_binding sections must be objects")
    for name, expected in (
        ("policy_schema_role", "arm_only_6d"),
        ("authorized_runtime_role", _ARM6_POLICY_RUNTIME_ROLE),
    ):
        actual = policy.get(name)
        if type(actual) is not str or actual != expected:
            raise PermissionError(
                f"arm6 policy_attestation.{name} must be exactly {expected!r}"
            )
    run_commit = policy.get("run_commit")
    selected = policy.get("selected_checkpoint")
    if not isinstance(run_commit, dict) or not isinstance(selected, dict):
        raise PermissionError(
            "arm6 policy attestation lacks run_commit or selected_checkpoint"
        )
    run_commit_sha = run_commit.get("sha256")
    _validate_sha256_hex(run_commit_sha, label="consumer run commit SHA-256")
    if run_commit_sha != config.arm6_run_commit_sha:
        raise PermissionError(
            "configured arm6_run_commit_sha does not match the verified run marker"
        )
    if (
        type(selected.get("step")) is not int
        or selected.get("step") != 20_000
        or selected.get("directory") != "020000"
        or selected.get("pretrained_model") != str(selected_checkpoint)
    ):
        raise PermissionError(
            "arm6 policy attestation selected checkpoint is not canonical 020000"
        )
    destination = projection.get("destination")
    source = projection.get("source")
    video_identity = projection.get("video_byte_identity")
    if (
        not isinstance(destination, dict)
        or not isinstance(source, dict)
        or not isinstance(video_identity, dict)
    ):
        raise PermissionError(
            "arm6 projection binding lacks source, destination, or video identity"
        )
    expected_roots = (
        (
            destination,
            config.dataset_repo_id,
            _require_no_symlink_components(
                Path(config.dataset_root), label="arm6 policy dataset"
            ),
            "destination",
        ),
        (
            source,
            config.visual_audit_source_repo_id,
            _require_no_symlink_components(
                Path(config.visual_audit_source_root),
                label="arm6 visual audit source",
            ),
            "source",
        ),
    )
    for section, expected_repo, expected_root, label in expected_roots:
        if section.get("repo_id") != expected_repo:
            raise PermissionError(
                f"arm6 projection {label} repo_id differs from deployment config"
            )
        root_value = section.get("root")
        if type(root_value) is not str or _require_no_symlink_components(
            Path(root_value),
            label=f"arm6 projection {label}",
        ) != expected_root:
            raise PermissionError(
                f"arm6 projection {label} root differs from deployment config"
            )
    if video_identity.get("verified") is not True:
        raise PermissionError("arm6 projection video byte identity is not verified")
    policy_input_digest = policy.get("input_digest")
    _validate_sha256_hex(policy_input_digest, label="arm6 policy input digest")
    projection_controller = projection.get("controller_contract")
    if not isinstance(projection_controller, dict):
        raise PermissionError("arm6 projection controller_contract must be an object")
    configured_controller = _require_no_symlink_components(
        Path(config.dataset_controller_contract),
        label="arm6 deployment controller contract",
    )
    controller_digest, controller_identity = _hash_regular_file(
        configured_controller
    )
    if (
        projection_controller.get("path") != str(configured_controller)
        or type(projection_controller.get("size")) is not int
        or projection_controller.get("size") != controller_identity[3]
        or projection_controller.get("sha256") != controller_digest
    ):
        raise PermissionError(
            "arm6 projection controller contract differs from deployment config"
        )

    expected_roles = {
        "child_attestation_010000",
        "child_attestation_020000",
        "materializer_script",
        "model_010000",
        "model_020000",
        "producer_script",
        "projection_manifest",
        "run_commit",
    }
    if not isinstance(controlled_artifacts_value, list):
        raise PermissionError("arm6 controlled_artifacts must be a list")
    normalized_artifacts: list[dict[str, object]] = []
    previous: tuple[str, str] | None = None
    for index, item in enumerate(controlled_artifacts_value):
        if not isinstance(item, dict) or set(item) != {
            "role",
            "path",
            "size",
            "sha256",
        }:
            raise PermissionError(
                f"arm6 controlled_artifacts[{index}] is not an exact record"
            )
        role = item.get("role")
        path_value = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if type(role) is not str or not role:
            raise PermissionError("arm6 controlled artifact role must be non-empty")
        if type(path_value) is not str or not Path(path_value).is_absolute():
            raise PermissionError(
                "arm6 controlled artifact path must be an absolute string"
            )
        if type(size) is not int or size < 0:
            raise PermissionError(
                "arm6 controlled artifact size must be a non-negative integer"
            )
        _validate_sha256_hex(digest, label=f"arm6 controlled artifact {role} SHA-256")
        order_key = (role, path_value)
        if previous is not None and order_key <= previous:
            raise PermissionError(
                "arm6 controlled artifacts must be unique and sorted by role/path"
            )
        previous = order_key
        path = _require_no_symlink_components(
            Path(path_value), label=f"arm6 controlled artifact {role}"
        )
        current_digest, identity = _hash_regular_file(path)
        if current_digest != digest or identity[3] != size:
            raise PermissionError(
                f"arm6 controlled artifact {role} changed after consumer verification"
            )
        normalized_artifacts.append(
            {
                "role": role,
                "path": str(path),
                "size": size,
                "sha256": digest,
            }
        )
    if {item["role"] for item in normalized_artifacts} != expected_roles:
        raise PermissionError(
            "arm6 controlled artifacts do not cover the exact marker/children/models/"
            "projection/producer set"
        )
    artifacts_by_role = {
        str(item["role"]): item for item in normalized_artifacts
    }

    def bind_reference(role: str, value: object, *, label: str) -> None:
        if not isinstance(value, dict):
            raise PermissionError(f"arm6 {label} must be an object")
        path_value = value.get("path")
        size_value = value.get("size")
        sha_value = value.get("sha256")
        if type(path_value) is not str or not Path(path_value).is_absolute():
            raise PermissionError(f"arm6 {label}.path must be an absolute string")
        if type(size_value) is not int or size_value < 0:
            raise PermissionError(f"arm6 {label}.size must be a non-negative integer")
        _validate_sha256_hex(sha_value, label=f"arm6 {label} SHA-256")
        expected = {
            "role": role,
            "path": path_value,
            "size": size_value,
            "sha256": sha_value,
        }
        if not _type_strict_equal(artifacts_by_role.get(role), expected):
            raise PermissionError(
                f"arm6 controlled artifact {role} differs from {label}"
            )

    bind_reference("run_commit", run_commit, label="policy run_commit")
    projection_manifest = projection.get("manifest")
    projection_materializer = projection.get("materializer")
    bind_reference(
        "projection_manifest",
        projection_manifest,
        label="projection manifest",
    )
    bind_reference(
        "materializer_script",
        projection_materializer,
        label="projection materializer",
    )
    producer_verify = policy.get("producer_verify_run")
    if not isinstance(producer_verify, dict):
        raise PermissionError("arm6 producer_verify_run must be an object")
    bind_reference(
        "producer_script",
        producer_verify.get("script"),
        label="producer verifier script",
    )
    children = policy.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise PermissionError("arm6 policy attestation must bind exactly two children")
    children_by_step: dict[int, dict] = {}
    for child in children:
        if not isinstance(child, dict) or type(child.get("step")) is not int:
            raise PermissionError("arm6 child records must have strict integer steps")
        children_by_step[child["step"]] = child
    if set(children_by_step) != {10_000, 20_000}:
        raise PermissionError("arm6 children must be exactly checkpoints 010000/020000")
    for step, directory in ((10_000, "010000"), (20_000, "020000")):
        child = children_by_step[step]
        if child.get("directory") != directory:
            raise PermissionError(
                f"arm6 child {step} directory must be exactly {directory}"
            )
        bind_reference(
            f"child_attestation_{directory}",
            child.get("attestation"),
            label=f"child {directory} attestation",
        )
        bind_reference(
            f"model_{directory}",
            child.get("model"),
            label=f"child {directory} model",
        )
    selected_child = children_by_step[20_000]
    if (
        selected_child.get("pretrained_model") != str(selected_checkpoint)
        or selected.get("attestation_sha256")
        != selected_child["attestation"].get("sha256")
        or selected.get("checkpoint_files_sha256")
        != selected_child.get("checkpoint_files_sha256")
        or selected.get("model_sha256") != selected_child["model"].get("sha256")
    ):
        raise PermissionError(
            "arm6 selected checkpoint differs from the bound 020000 child"
        )
    return _Arm6DeploymentBinding(
        selected_checkpoint=selected_checkpoint,
        source_binding=source_binding,
        controlled_artifacts=tuple(normalized_artifacts),
    )


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} path component does not exist: {current}") from exc
        if stat.S_ISLNK(mode):
            raise PermissionError(
                f"{label} must use an immutable explicit path; symbolic link component rejected: "
                f"{current}"
            )
    return absolute


def _regular_tree_entries(
    root: Path, *, label: str
) -> tuple[Path, list[Path], list[Path]]:
    root = _require_no_symlink_components(root, label=label)
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} directory does not exist: {root}") from exc
    if not stat.S_ISDIR(root_mode):
        raise ValueError(f"{label} must be a directory: {root}")

    directories = [root]
    files: list[Path] = []

    def visit(directory: Path) -> None:
        before = _directory_identity(directory)
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"cannot enumerate {label} directory {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ValueError(f"cannot stat {label} entry {path}: {exc}") from exc
            if stat.S_ISLNK(mode):
                raise PermissionError(f"{label} tree contains symbolic link: {path}")
            if stat.S_ISDIR(mode):
                directories.append(path)
                visit(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            else:
                raise PermissionError(
                    f"{label} tree contains a non-regular filesystem entry: {path}"
                )
        if _directory_identity(directory) != before:
            raise PermissionError(f"{label} directory changed while enumerating: {directory}")

    visit(root)
    return root, directories, files


def _directory_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required directory does not exist: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise PermissionError(f"required source must be a non-symlink directory: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required regular file does not exist: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise PermissionError(f"required source must be a regular non-symlink file: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_bytes(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    path = _absolute_without_resolving(path)
    before = _regular_file_identity(path)
    if before[3] > max_bytes:
        raise ValueError(
            f"deployment metadata file is too large ({before[3]} bytes, limit {max_bytes}): {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PermissionError(f"cannot safely open deployment source {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before:
            raise PermissionError(f"deployment source changed while opening: {path}")
        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    f"deployment metadata file exceeded {max_bytes} bytes while reading: {path}"
                )
        after_open = os.fstat(descriptor)
        after_open_identity = (
            after_open.st_dev,
            after_open.st_ino,
            after_open.st_mode,
            after_open.st_size,
            after_open.st_mtime_ns,
            after_open.st_ctime_ns,
        )
        if after_open_identity != before:
            raise PermissionError(f"deployment source changed while reading: {path}")
    finally:
        os.close(descriptor)
    if _regular_file_identity(path) != before:
        raise PermissionError(f"deployment source was replaced while reading: {path}")
    return b"".join(chunks)


def _hash_regular_file(
    path: Path,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    path = _absolute_without_resolving(path)
    before = _regular_file_identity(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PermissionError(f"cannot safely open deployment source {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before:
            raise PermissionError(f"deployment source changed while opening: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
        after_open_identity = (
            after_open.st_dev,
            after_open.st_ino,
            after_open.st_mode,
            after_open.st_size,
            after_open.st_mtime_ns,
            after_open.st_ctime_ns,
        )
        if after_open_identity != before:
            raise PermissionError(f"deployment source changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if _regular_file_identity(path) != before:
        raise PermissionError(f"deployment source was replaced while hashing: {path}")
    return digest.hexdigest(), before


def _sha256_file(path: Path) -> str:
    return _hash_regular_file(path)[0]


def _read_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical_fingerprint_value(value):
    if is_dataclass(value):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": {
                item.name: _canonical_fingerprint_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Enum):
        return _canonical_fingerprint_value(value.value)
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("deployment fingerprint mappings require string keys")
        return {
            key: _canonical_fingerprint_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("deployment fingerprint values must be finite")
        return value
    raise TypeError(f"unsupported deployment fingerprint value {type(value).__name__}")


def _type_strict_equal(actual: object, expected: object) -> bool:
    """Compare canonical JSON-like values without Python's bool/int aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _type_strict_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _type_strict_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def visual_phase_latch_provenance(
    config: VisualPhaseLatchConfig,
) -> dict[str, object]:
    """Return a JSON-safe, type-bound record of every visual latch constant."""

    if not isinstance(config, VisualPhaseLatchConfig):
        raise TypeError("visual phase latch provenance requires VisualPhaseLatchConfig")
    payload = _canonical_fingerprint_value(config)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass invariant
        raise TypeError("visual phase latch provenance was not an object")
    return payload


def visual_phase_latch_implementation_manifest() -> dict[str, object]:
    """Hash every local implementation file which can change visual actuation."""

    package_dir = Path(__file__).parent
    repository_dir = package_dir.parent
    sources = {
        "deployment_guard": package_dir / "firefly_deployment.py",
        "arm6_action_adapter": package_dir / "firefly_arm_only_policy.py",
        "arm6_attestation_consumer": package_dir / "firefly_arm6_attestation.py",
        "visual_state_machine": package_dir / "firefly_visual_latch.py",
        "tip_servo_planner": package_dir / "firefly_tip_servo.py",
        "act_control_loop": repository_dir / "scripts" / "run_firefly_y6.py",
        "offline_visual_audit": (
            repository_dir / "scripts" / "audit_firefly_y6_visual_latch.py"
        ),
        "follower": package_dir / "hex_zenoh_follower.py",
        "fk": package_dir / "arm_gravity.py",
        "payload_gravity": package_dir / "payload_gravity.py",
    }
    manifest: dict[str, object] = {}
    for name, path in sources.items():
        digest, identity = _hash_regular_file(path)
        manifest[name] = {
            "path": str(_absolute_without_resolving(path)),
            "size": identity[3],
            "sha256": digest,
        }
    return manifest


def _decode_strict_json_object(data: bytes, *, label: str, path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _reject_visual_audit(reason: str) -> None:
    raise PermissionError(f"visual latch offline audit rejected: {reason}")


def _open_only_source_mapping(value: object, *, label: str) -> dict:
    if not isinstance(value, dict):
        _reject_visual_audit(f"{label} must be an object in the pinned v1 source")
    return value


def _open_only_source_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _reject_visual_audit(f"{label} must be finite in the pinned v1 source")
    return float(value)


def _open_only_source_vector(
    value: object,
    *,
    label: str,
    length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        _reject_visual_audit(
            f"{label} must contain exactly {length} values in the pinned v1 source"
        )
    return tuple(
        _open_only_source_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _require_open_only_source_statistics(
    actual: object,
    expected: object,
    *,
    label: str,
) -> None:
    """Compare report statistics to validator-owned, source-derived values."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            _reject_visual_audit(
                f"{label} does not exactly match the pinned v1 source fields"
            )
        for name, expected_value in expected.items():
            _require_open_only_source_statistics(
                actual[name],
                expected_value,
                label=f"{label}.{name}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _reject_visual_audit(
                f"{label} does not exactly match the pinned v1 source sequence"
            )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _require_open_only_source_statistics(
                actual_value,
                expected_value,
                label=f"{label}[{index}]",
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        _reject_visual_audit(
            f"{label} must match the pinned v1 source value {expected!r}, "
            f"found {actual!r}"
        )


def _read_pinned_open_only_negative_control(source: dict) -> list[dict]:
    """Open and re-hash the one reviewed historical OPEN_ONLY v1 trace."""

    label = "tip_probe.negative_control.source"
    source_path = source.get("path")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not Path(source_path).is_absolute()
    ):
        _reject_visual_audit(f"{label}.path must be a non-empty absolute path")
    configured_path = _absolute_without_resolving(Path(source_path))
    expected_path = _absolute_without_resolving(
        _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_PATH
    )
    if configured_path != expected_path:
        _reject_visual_audit(
            f"{label}.path must be the exact pinned real v1 source {expected_path}"
        )
    try:
        payload = _read_regular_bytes(
            expected_path,
            max_bytes=_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        _reject_visual_audit(f"cannot safely read pinned real v1 source: {exc}")
    actual_size = len(payload)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    for name, actual, expected in (
        ("size", actual_size, _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE),
        ("sha256", actual_sha256, _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SHA256),
    ):
        declared = source.get(name)
        if type(declared) is not type(expected) or declared != expected:
            _reject_visual_audit(
                f"{label}.{name} must be exactly {expected!r}, found {declared!r}"
            )
        if actual != expected:
            _reject_visual_audit(
                f"pinned real v1 source {name} changed: expected {expected!r}, "
                f"found {actual!r}"
            )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _reject_visual_audit(f"pinned real v1 source is not UTF-8: {exc}")
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        _reject_visual_audit("pinned real v1 source contains a blank JSONL record")
    if len(lines) != _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS:
        _reject_visual_audit(
            "pinned real v1 source record count changed: expected "
            f"{_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS}, found {len(lines)}"
        )
    declared_records = source.get("jsonl_records")
    if (
        type(declared_records) is not int
        or declared_records != _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS
    ):
        _reject_visual_audit(
            f"{label}.jsonl_records must be exactly "
            f"{_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS}"
        )

    records: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _reject_visual_audit(
                f"pinned real v1 source line {line_number} is invalid JSON: {exc}"
            )
        if not isinstance(record, dict):
            _reject_visual_audit(
                f"pinned real v1 source line {line_number} is not an object"
            )
        records.append(record)
    command_cycles = sum(
        record.get("event") == "command_cycle" for record in records
    )
    if command_cycles != _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_COMMAND_CYCLES:
        _reject_visual_audit(
            "pinned real v1 source command-cycle count changed: expected "
            f"{_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_COMMAND_CYCLES}, "
            f"found {command_cycles}"
        )
    return records


def _recompute_open_only_negative_control_statistics(
    records: list[dict],
    *,
    tip: TipServoConfig,
    ee_step_limit_rad: float,
) -> dict[str, object]:
    """Derive every schema-6 source statistic without trusting the report."""

    command_records = [
        record for record in records if record.get("event") == "command_cycle"
    ]
    acquired = [
        record for record in records if record.get("event") == "control_acquired"
    ]
    if len(acquired) != 1:
        _reject_visual_audit(
            "pinned real v1 source must contain exactly one control_acquired event"
        )
    capture = _open_only_source_vector(
        acquired[0].get("hold_target"),
        label="control_acquired.hold_target",
        length=7,
    )[6]
    if tip.coarse_capture_jaw_rad is None or not (
        tip.coarse_capture_jaw_rad[0]
        <= capture
        <= tip.coarse_capture_jaw_rad[1]
    ):
        _reject_visual_audit(
            "pinned real v1 source capture jaw is outside the reviewed gate"
        )

    parsed: list[dict[str, object]] = []
    final_jaws: list[float] = []
    arm_targets: list[tuple[float, ...]] = []
    tcp_z_values: list[float] = []
    arm_commands = 0
    descent_commands = 0
    jaw_commands = 0
    close_commands = 0
    for index, record in enumerate(command_records):
        visual = _open_only_source_mapping(
            record.get("visual_telemetry"),
            label=f"command_cycle[{index}].visual_telemetry",
        )
        frame = _open_only_source_mapping(
            visual.get("frame"), label=f"command_cycle[{index}].frame"
        )
        red = _open_only_source_mapping(
            visual.get("red"), label=f"command_cycle[{index}].red"
        )
        tip_telemetry = _open_only_source_mapping(
            visual.get("tip_servo"),
            label=f"command_cycle[{index}].tip_servo",
        )
        target = _open_only_source_vector(
            record.get("target"),
            label=f"command_cycle[{index}].target",
            length=7,
        )
        measured = _open_only_source_vector(
            record.get("measured"),
            label=f"command_cycle[{index}].measured",
            length=7,
        )
        frame_id = frame.get("id")
        candidate_count = red.get("candidate_count")
        if type(frame_id) is not int or frame_id < 0:
            _reject_visual_audit(
                f"command_cycle[{index}].frame.id must be nonnegative"
            )
        if type(candidate_count) is not int or candidate_count < 0:
            _reject_visual_audit(
                f"command_cycle[{index}].red.candidate_count must be nonnegative"
            )
        centroid_raw = red.get("centroid_norm")
        centroid = (
            None
            if centroid_raw is None
            else _open_only_source_vector(
                centroid_raw,
                label=f"command_cycle[{index}].red.centroid_norm",
                length=2,
            )
        )
        area = (
            None
            if centroid is None
            else _open_only_source_number(
                red.get("contour_area_fraction"),
                label=f"command_cycle[{index}].red.contour_area_fraction",
            )
        )
        descent = _open_only_source_number(
            tip_telemetry.get("commanded_descent_m", 0.0),
            label=f"command_cycle[{index}].tip_servo.commanded_descent_m",
        )
        arm_commands += tip_telemetry.get("commanded_arm_target_q_rad") is not None
        jaw_commands += tip_telemetry.get("commanded_jaw_target_rad") is not None
        descent_commands += abs(descent) > 1e-12
        close_commands += tip_telemetry.get("probe_close_allowed") is not False
        final_jaws.append(target[6])
        arm_targets.append(target[:6])
        tcp_position = visual.get("tcp_position_m")
        if tcp_position is not None:
            tcp_z_values.append(
                _open_only_source_vector(
                    tcp_position,
                    label=f"command_cycle[{index}].tcp_position_m",
                    length=3,
                )[2]
            )
        parsed.append(
            {
                "record": record,
                "visual": visual,
                "frame_id": frame_id,
                "fresh": frame.get("status") == "fresh_unique",
                "candidate_count": candidate_count,
                "centroid": centroid,
                "area": area,
                "target": target,
                "measured": measured,
            }
        )

    window = [
        item
        for item in parsed
        if _TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID
        <= item["frame_id"]
        <= _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
    ]
    expected_frame_ids = list(
        range(
            _TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID,
            _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID + 1,
        )
    )
    frame_ids = [item["frame_id"] for item in window]
    consecutive = frame_ids == expected_frame_ids
    x_values: list[float] = []
    y_values: list[float] = []
    area_values: list[float] = []
    jaw_values: list[float] = []
    arm_speeds: list[float] = []
    action_sources: list[str] = []
    gate_flags: list[bool] = []
    all_safety_ok = True
    for item in window:
        visual = item["visual"]
        record = item["record"]
        centroid = item["centroid"]
        area = item["area"]
        measured = item["measured"]
        if centroid is None or area is None:
            _reject_visual_audit(
                "pinned real v1 calibrated window omits red geometry"
            )
        arm_speed = _open_only_source_number(
            visual.get("arm_speed_rad_s"),
            label=f"frame[{item['frame_id']}].arm_speed_rad_s",
        )
        pickup = _open_only_source_mapping(
            visual.get("pickup_policy_close_intent"),
            label=f"frame[{item['frame_id']}].pickup_policy_close_intent",
        )
        action_source = pickup.get("action_source")
        if action_source not in {"fresh_policy", "inference_hold"}:
            _reject_visual_audit(
                "pinned real v1 calibrated window has invalid action_source"
            )
        x_values.append(centroid[0])
        y_values.append(centroid[1])
        area_values.append(area)
        jaw_values.append(measured[6])
        arm_speeds.append(arm_speed)
        action_sources.append(action_source)
        all_safety_ok = all_safety_ok and record.get("safety_ok") is True
        gate_flags.append(
            bool(
                item["fresh"]
                and item["candidate_count"] == 1
                and tip.coarse_x_norm[0] <= centroid[0] <= tip.coarse_x_norm[1]
                and tip.coarse_y_norm[0] <= centroid[1] <= tip.coarse_y_norm[1]
                and tip.coarse_area_fraction[0] <= area
                and area < tip.coarse_area_fraction[1]
                and tip.coarse_capture_jaw_rad[0]
                <= measured[6]
                <= tip.coarse_capture_jaw_rad[1]
                and arm_speed <= tip.max_arm_speed_rad_s + 1e-12
            )
        )
    source_counts = {
        name: action_sources.count(name) for name in sorted(set(action_sources))
    }
    first_confirmed_index = None
    streak = 0
    for index, matched in enumerate(gate_flags):
        streak = streak + 1 if matched else 0
        if first_confirmed_index is None and streak >= tip.takeover_confirm_frames:
            first_confirmed_index = index
    would_takeover_index = (
        None
        if first_confirmed_index is None
        else next(
            (
                index
                for index in range(first_confirmed_index, len(window))
                if action_sources[index] == "fresh_policy"
            ),
            None,
        )
    )

    boundary_index = next(
        (
            index
            for index, item in enumerate(parsed)
            if item["frame_id"] == _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
        ),
        None,
    )
    if boundary_index is None:
        _reject_visual_audit(
            "pinned real v1 source omits the takeover-boundary frame"
        )
    replay_records = parsed[: boundary_index + 1]
    suppressed_lower = sum(
        item["target"][6] < capture - 1e-12 for item in replay_records
    )
    suppressed_early_open = sum(
        item["target"][6] > capture + 1e-12 for item in replay_records
    )
    replayed_targets: list[float] = []
    previous = capture
    try:
        for _item in replay_records:
            previous = resolve_open_only_jaw_target(
                capture_jaw_target_rad=capture,
                previous_accepted_jaw_target_rad=previous,
                open_commanded=False,
                config=tip,
                ee_step_limit_rad=ee_step_limit_rad,
            )
            replayed_targets.append(previous)
        takeover_target = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture,
            previous_accepted_jaw_target_rad=previous,
            open_commanded=False,
            config=tip,
            ee_step_limit_rad=ee_step_limit_rad,
        )
        opened_targets = [takeover_target]
        previous = takeover_target
        for _ in range(32):
            previous = resolve_open_only_jaw_target(
                capture_jaw_target_rad=capture,
                previous_accepted_jaw_target_rad=previous,
                open_commanded=True,
                config=tip,
                ee_step_limit_rad=ee_step_limit_rad,
            )
            opened_targets.append(previous)
            if math.isclose(
                previous,
                tip.pregrasp_open_target_rad,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                break
        after_reclose = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture,
            previous_accepted_jaw_target_rad=opened_targets[1],
            open_commanded=True,
            config=tip,
            ee_step_limit_rad=ee_step_limit_rad,
        )
        fault_hold = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture,
            previous_accepted_jaw_target_rad=opened_targets[-1],
            open_commanded=True,
            config=tip,
            ee_step_limit_rad=ee_step_limit_rad,
        )
        timeout_hold = resolve_open_only_jaw_target(
            capture_jaw_target_rad=capture,
            previous_accepted_jaw_target_rad=capture,
            open_commanded=False,
            config=tip,
            ee_step_limit_rad=ee_step_limit_rad,
        )
    except Exception as exc:
        _reject_visual_audit(
            f"pinned real v1 axis-authority replay failed: {type(exc).__name__}"
        )

    open_steps = [
        after - before
        for before, after in zip(opened_targets, opened_targets[1:])
    ]
    historical_lower = sum(value < capture - 1e-12 for value in final_jaws)
    historical_higher = sum(value > capture + 1e-12 for value in final_jaws)
    policy_arm_motion = any(
        max(
            abs(value - start)
            for value, start in zip(target, arm_targets[0], strict=True)
        )
        > 1e-6
        for target in arm_targets[1:]
    )
    minimum_tcp_delta = min(value - tcp_z_values[0] for value in tcp_z_values)
    source_pass = bool(
        consecutive
        and all(item["fresh"] for item in window)
        and all_safety_ok
        and all(gate_flags)
        and source_counts == {"fresh_policy": 3, "inference_hold": 5}
        and first_confirmed_index is not None
        and window[first_confirmed_index]["frame_id"] == 492
        and would_takeover_index is not None
        and window[would_takeover_index]["frame_id"]
        == _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
        and arm_commands == 0
        and descent_commands == 0
        and jaw_commands == 0
        and close_commands == 0
        and historical_lower > 0
        and historical_higher > 0
        and policy_arm_motion
        and minimum_tcp_delta < -0.020
        and suppressed_lower > 0
        and suppressed_early_open > 0
        and all(value == capture for value in replayed_targets)
        and opened_targets[1] > capture
        and all(step >= -1e-12 for step in open_steps)
        and all(step <= ee_step_limit_rad + 1e-12 for step in open_steps)
        and opened_targets[-1] == tip.pregrasp_open_target_rad
        and after_reclose >= opened_targets[1] - 1e-12
        and fault_hold >= opened_targets[-1] - 1e-12
        and timeout_hold == capture
    )
    if not source_pass:
        _reject_visual_audit(
            "pinned real v1 source no longer satisfies the schema-6 negative-control contract"
        )

    return {
        "visual_command_cycles": len(command_records),
        "calibrated_window_start_frame_id": (
            _TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID
        ),
        "calibrated_window_end_frame_id": (
            _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID
        ),
        "calibrated_window_frames": len(window),
        "calibrated_window_consecutive": consecutive,
        "calibrated_window_all_fresh_unique": all(
            item["fresh"] for item in window
        ),
        "calibrated_window_all_closed_capture_gate": all(gate_flags),
        "calibrated_window_action_source_counts": source_counts,
        "first_confirmed_hold_frame_id": window[first_confirmed_index]["frame_id"],
        "would_takeover_boundary_frame_id": window[would_takeover_index]["frame_id"],
        "would_takeover_boundary_action_source": action_sources[
            would_takeover_index
        ],
        "actual_tip_servo_arm_command_cycles": arm_commands,
        "actual_tip_servo_descent_command_cycles": descent_commands,
        "actual_tip_servo_jaw_command_cycles": jaw_commands,
        "actual_tip_servo_close_command_cycles": close_commands,
        **{
            "observed_x_norm": {"min": min(x_values), "max": max(x_values)},
            "observed_y_norm": {"min": min(y_values), "max": max(y_values)},
            "observed_area_fraction": {
                "min": min(area_values),
                "max": max(area_values),
            },
            "observed_jaw_rad": {
                "min": min(jaw_values),
                "max": max(jaw_values),
            },
            "observed_max_arm_speed_rad_s": max(arm_speeds),
        },
        "historical_out_of_scope": {
            "policy_arm_motion_observed": policy_arm_motion,
            "minimum_measured_tcp_delta_z_m": minimum_tcp_delta,
            "final_jaw_target_min_rad": min(final_jaws),
            "final_jaw_target_max_rad": max(final_jaws),
            "final_jaw_target_below_capture_cycles": historical_lower,
            "final_jaw_target_above_capture_cycles": historical_higher,
        },
        "axis_authority_replay": {
            "capture_jaw_target_rad": capture,
            "replayed_seeking_cycles": len(replayed_targets),
            "published_jaw_min_rad": min(replayed_targets),
            "published_jaw_max_rad": max(replayed_targets),
            "suppressed_lower_jaw_proposal_cycles": suppressed_lower,
            "suppressed_early_open_proposal_cycles": suppressed_early_open,
            "takeover_boundary_published_jaw_target_rad": takeover_target,
            "takeover_boundary_still_capture_hold": takeover_target == capture,
            "first_open_requires_takeover_and_settle": (
                opened_targets[0] == capture and opened_targets[1] > capture
            ),
            "opened_targets_rad": opened_targets,
            "opened_target_monotonic_nondecreasing": all(
                step >= -1e-12 for step in open_steps
            ),
            "opened_target_step_limit_pass": all(
                step <= ee_step_limit_rad + 1e-12 for step in open_steps
            ),
            "open_target_reached": (
                opened_targets[-1] == tip.pregrasp_open_target_rad
            ),
            "adversarial_reclose_proposal_rad": min(final_jaws),
            "adversarial_reclose_resolved_target_rad": after_reclose,
            "adversarial_reclose_suppressed": (
                after_reclose >= opened_targets[1] - 1e-12
            ),
            "fault_hold_non_decreasing": (
                fault_hold >= opened_targets[-1] - 1e-12
            ),
            "seeking_timeout_ms": tip.seeking_timeout_ms,
            "seeking_timeout_holds_capture": timeout_hold == capture,
            "counterfactual_open_response_modeled": False,
        },
    }


def _visual_latch_audit_schema_version(
    config: FireflyDeploymentConfig,
) -> int:
    tip = config.visual_phase_latch.tip_servo
    if (
        tip.mode is TipServoMode.EXECUTE
        and tip.probe_profile is TipServoProbeProfile.OPEN_ONLY
    ):
        return _TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION
    if tip.mode is TipServoMode.EXECUTE:
        return _TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION
    return _VISUAL_LATCH_AUDIT_SCHEMA_VERSION


def validate_visual_latch_audit_report(
    report: dict,
    *,
    config: FireflyDeploymentConfig,
    dataset_root: Path,
    training_input_digest: str,
    implementation_manifest: dict[str, object],
    arm6_source_binding: dict[str, object] | None = None,
) -> None:
    """Validate the complete offline visual replay proof before hardware exists."""

    if not isinstance(report, dict):
        _reject_visual_audit("report must be a JSON object")
    try:
        _canonical_fingerprint_value(report)
    except (TypeError, ValueError) as exc:
        _reject_visual_audit(f"report is not finite canonical JSON: {exc}")

    expected_header = {
        "schema_version": _visual_latch_audit_schema_version(config),
        "kind": _VISUAL_LATCH_AUDIT_KIND,
        "read_only": True,
        "safe_for_deployment": True,
    }
    for name, expected in expected_header.items():
        if report.get(name) != expected or type(report.get(name)) is not type(expected):
            _reject_visual_audit(
                f"{name} must be exactly {expected!r}, found {report.get(name)!r}"
            )

    dataset_info = _read_json_object(
        dataset_root / "meta" / "info.json", label="deployment dataset info"
    )
    expected_episodes = dataset_info.get("total_episodes")
    expected_frames = dataset_info.get("total_frames")
    expected_fps = dataset_info.get("fps")
    for name, value in (
        ("total_episodes", expected_episodes),
        ("total_frames", expected_frames),
        ("fps", expected_fps),
    ):
        if type(value) is not int or value <= 0:
            _reject_visual_audit(f"dataset info {name} must be a positive integer")
    visual_source_repo_id = (
        config.visual_audit_source_repo_id
        if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE
        else config.dataset_repo_id
    )
    if visual_source_repo_id == _FULLCYCLE_V4_DATASET_REPO_ID and (
        expected_episodes != _FULLCYCLE_V4_TOTAL_EPISODES
        or expected_frames != _FULLCYCLE_V4_TOTAL_FRAMES
        or expected_fps != 30
    ):
        _reject_visual_audit(
            "fullcycle-v4 dataset totals must be exactly "
            f"{_FULLCYCLE_V4_TOTAL_EPISODES} episodes, "
            f"{_FULLCYCLE_V4_TOTAL_FRAMES} frames, 30 fps"
        )

    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        _reject_visual_audit("dataset must be an object")
    expected_dataset = {
        "root": str(dataset_root),
        "episodes": expected_episodes,
        "frames": expected_frames,
        "fps": config.fps,
        "wrist_video_key": "observation.images.wrist",
        "recorded_rotation_degrees": config.visual_phase_latch.d405_rotation_degrees,
        "offline_rotation_applied": False,
    }
    for name, expected in expected_dataset.items():
        actual = dataset.get(name)
        if actual != expected or type(actual) is not type(expected):
            _reject_visual_audit(
                f"dataset.{name} must be exactly {expected!r}, found {actual!r}"
            )

    visual_profile = visual_phase_latch_provenance(config.visual_phase_latch)
    visual_fields = visual_profile.get("fields")
    profile = report.get("profile")
    if not isinstance(profile, dict):
        _reject_visual_audit("profile must be an object")
    if not _type_strict_equal(
        _canonical_fingerprint_value(profile.get("detector_config")),
        visual_fields,
    ):
        _reject_visual_audit(
            "profile.detector_config does not exactly match deployed visual provenance"
        )
    for name, expected in (
        ("close_confirmation_frames", config.visual_phase_latch.close_confirm_frames),
        ("blue_confirmation_frames", config.visual_phase_latch.box_confirm_frames),
    ):
        actual = profile.get(name)
        if type(actual) is not int or actual != expected:
            _reject_visual_audit(f"profile.{name} must be exactly {expected}")

    audit_script = implementation_manifest.get("offline_visual_audit")
    if not isinstance(audit_script, dict):
        _reject_visual_audit("implementation manifest omits offline_visual_audit")
    audit_script_sha256 = audit_script.get("sha256")
    try:
        _validate_sha256_hex(
            audit_script_sha256, label="offline visual audit implementation SHA-256"
        )
    except ValueError as exc:
        _reject_visual_audit(str(exc))
    expected_source_binding = {
        "training_input_digest": training_input_digest,
        "visual_profile": visual_profile,
        "visual_implementation_manifest": implementation_manifest,
        "audit_script_sha256": audit_script_sha256,
    }
    if _visual_latch_audit_schema_version(config) == 6:
        if config.policy_runtime_role != _ARM6_POLICY_RUNTIME_ROLE:
            _reject_visual_audit(
                "schema-6 OPEN_ONLY authority requires policy_runtime_role="
                "'arm6_seeking' and the frozen arm6 attestation consumer"
            )
        if not isinstance(arm6_source_binding, dict) or set(
            arm6_source_binding
        ) != {"policy_attestation", "projection_binding"}:
            _reject_visual_audit(
                "schema-6 OPEN_ONLY arm6 source binding must contain exactly "
                "policy_attestation and projection_binding"
            )
        expected_source_binding.update(arm6_source_binding)
    if not _type_strict_equal(
        _canonical_fingerprint_value(report.get("source_binding")),
        _canonical_fingerprint_value(expected_source_binding),
    ):
        _reject_visual_audit(
            "source_binding must exactly match the training inputs, visual profile, "
            "visual implementation manifest, and audit script"
        )

    summary = report.get("summary")
    if not isinstance(summary, dict):
        _reject_visual_audit("summary must be an object")
    expected_summary_counts = {
        "verdict": "passed",
        "episode_count": expected_episodes,
        "error_count": 0,
        "runtime_close_before_pickup_crossing_episodes": 0,
        "premature_close_confirmation_episodes": 0,
        "premature_pretransport_blue_confirmation_episodes": 0,
        "close_detected_episodes": expected_episodes,
        "contact_confirmed_episodes": expected_episodes,
        "held_red_timeout_safe_episodes": expected_episodes,
        "release_window_detected_episodes": expected_episodes,
        "state_machine_ordered_complete_episodes": expected_episodes,
        "state_machine_fault_free_episodes": expected_episodes,
    }
    for name, expected in expected_summary_counts.items():
        actual = summary.get(name)
        if actual != expected or type(actual) is not type(expected):
            _reject_visual_audit(
                f"summary.{name} must be exactly {expected!r}, found {actual!r}"
            )
    if summary.get("errors") != []:
        _reject_visual_audit("summary.errors must be an empty array")
    for name in (
        "close_detection_rate",
        "held_red_timeout_safe_rate",
        "release_window_detection_rate",
    ):
        actual = summary.get(name)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            _reject_visual_audit(f"summary.{name} must be exactly 1.0")

    _validate_tip_servo_probe_audit_contract(
        report,
        config=config,
        expected_episodes=expected_episodes,
    )
    tip_probe_execute = (
        config.visual_phase_latch.tip_servo.mode is TipServoMode.EXECUTE
    )

    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        _reject_visual_audit(
            f"episodes must contain exactly {expected_episodes} per-episode records"
        )
    required_checks = (
        "no_premature_close_confirmation",
        "close_detected_in_window",
        "contact_confirmed",
        "held_red_survives_loss_timeout",
        "no_pretransport_blue_confirmation",
        "blue_detected_in_release_window",
        "state_machine_ordered_complete",
        "state_machine_fault_free",
    )
    seen_indexes: set[int] = set()
    episode_frame_total = 0
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            _reject_visual_audit(f"episodes[{expected_index}] must be an object")
        episode_index = episode.get("episode_index")
        if type(episode_index) is not int or episode_index != expected_index:
            _reject_visual_audit(
                f"episodes[{expected_index}].episode_index must be {expected_index}"
            )
        if episode_index in seen_indexes:
            _reject_visual_audit(f"duplicate episode index {episode_index}")
        seen_indexes.add(episode_index)
        frames = episode.get("frames")
        if type(frames) is not int or frames <= 0:
            _reject_visual_audit(
                f"episodes[{expected_index}].frames must be a positive integer"
            )
        episode_frame_total += frames
        if episode.get("safe") is not True:
            _reject_visual_audit(f"episodes[{expected_index}].safe must be true")
        checks = episode.get("checks")
        if not isinstance(checks, dict) or any(
            checks.get(name) is not True for name in required_checks
        ):
            _reject_visual_audit(
                f"episodes[{expected_index}].checks must pass every deployment check"
            )
        if tip_probe_execute:
            _validate_tip_servo_probe_episode(
                episode,
                episode_index=expected_index,
            )
    if episode_frame_total != expected_frames:
        _reject_visual_audit(
            f"per-episode frame sum {episode_frame_total} does not match {expected_frames}"
        )


def _validate_tip_servo_open_only_audit_contract(
    report: dict,
    *,
    config: FireflyDeploymentConfig,
    expected_episodes: int,
) -> None:
    """Require schema-6 evidence for axis-partitioned OPEN_ONLY authority."""

    tip = config.visual_phase_latch.tip_servo
    if tip.mode is not TipServoMode.EXECUTE:
        return
    if tip != _fullcycle_v4_open_only_tip_servo_config():
        _reject_visual_audit(
            "schema-6 open-only audit requires the exact reviewed axis-partitioned profile"
        )

    def require_mapping(value: object, label: str) -> dict:
        if not isinstance(value, dict):
            _reject_visual_audit(f"{label} must be an object")
        return value

    def require_exact(mapping: dict, name: str, expected: object, label: str) -> None:
        actual = mapping.get(name)
        if not _type_strict_equal(actual, expected):
            _reject_visual_audit(
                f"{label}.{name} must be exactly {expected!r}, found {actual!r}"
            )

    def finite_number(value: object, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            _reject_visual_audit(f"{label} must be a finite number")
        return float(value)

    def require_number(
        mapping: dict, name: str, expected: float, label: str
    ) -> None:
        actual = finite_number(mapping.get(name), f"{label}.{name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            _reject_visual_audit(
                f"{label}.{name} must be exactly {expected!r}, found {actual!r}"
            )

    authority = _schema6_open_only_axis_authority()
    require_exact(report, "axis_authority", authority, "report")
    require_exact(
        report,
        "hardware_completion_validated",
        False,
        "report",
    )
    require_exact(
        report,
        "closed_loop_convergence_claimed",
        False,
        "report",
    )
    for ambiguous in (
        "jaw_open_authorized",
        "arm_motion_authorized",
        "descent_authorized",
        "close_authorized",
    ):
        if ambiguous in report:
            _reject_visual_audit(
                f"report must not expose ambiguous unscoped field {ambiguous!r}"
            )

    profile = require_mapping(report.get("profile"), "profile")
    profile_tip = require_mapping(profile.get("tip_probe"), "profile.tip_probe")
    profile_config = require_mapping(
        profile_tip.get("config"), "profile.tip_probe.config"
    )
    expected_config = {
        "mode": TipServoMode.EXECUTE.value,
        "probe_profile": TipServoProbeProfile.OPEN_ONLY.value,
        "coarse_x_norm": [0.50, 0.545],
        "coarse_y_norm": [0.42, 0.56],
        "coarse_area_fraction": [0.017, 0.029],
        "coarse_capture_jaw_rad": [0.285, 0.325],
        "tracking_x_norm": [0.40, 0.56],
        "tracking_y_norm": [0.40, 0.62],
        "tracking_area_fraction": [0.014, 0.041],
        "max_arm_speed_rad_s": 0.07,
        "max_takeover_reference_error_rad": 0.035,
        "arm_target_settle_rad": [0.002] * 6,
        "opening_timeout_ms": 1500.0,
        "seeking_timeout_ms": 20000.0,
        "descent_step_m": 0.0,
        "max_total_descent_m": 0.0,
        "allow_close": False,
    }
    for name, expected in expected_config.items():
        actual = profile_config.get(name)
        if isinstance(expected, float):
            require_number(profile_config, name, expected, "profile.tip_probe.config")
        elif actual != expected or type(actual) is not type(expected):
            _reject_visual_audit(
                f"profile.tip_probe.config.{name} must be exactly {expected!r}, "
                f"found {actual!r}"
            )
    require_exact(
        profile_tip,
        "semantics",
        "policy_arm_before_takeover_tip_capture_hold_then_arm_hold_tip_open",
        "profile.tip_probe",
    )
    require_exact(
        profile_tip,
        "axis_authority",
        authority,
        "profile.tip_probe",
    )
    require_exact(
        profile_tip,
        "hardware_completion_validated",
        False,
        "profile.tip_probe",
    )
    require_exact(
        profile_tip,
        "closed_loop_convergence_claimed",
        False,
        "profile.tip_probe",
    )

    aggregate = require_mapping(report.get("tip_probe"), "tip_probe")
    require_exact(aggregate, "axis_authority", authority, "tip_probe")
    require_exact(
        aggregate,
        "hardware_completion_validated",
        False,
        "tip_probe",
    )
    require_exact(
        aggregate,
        "closed_loop_convergence_claimed",
        False,
        "tip_probe",
    )
    require_exact(
        aggregate,
        "demonstration_episode_count",
        expected_episodes,
        "tip_probe",
    )
    require_exact(
        aggregate,
        "demonstration_envelope_checked_episodes",
        expected_episodes,
        "tip_probe",
    )
    observed_episodes = aggregate.get("recorded_closed_capture_observed_episodes")
    if (
        type(observed_episodes) is not int
        or not 0 <= observed_episodes <= expected_episodes
    ):
        _reject_visual_audit(
            "tip_probe.recorded_closed_capture_observed_episodes must be an "
            "integer within the dataset episode count"
        )
    require_number(
        aggregate, "configured_max_total_descent_m", 0.0, "tip_probe"
    )
    require_number(aggregate, "configured_descent_step_m", 0.0, "tip_probe")
    require_number(
        aggregate, "configured_seeking_timeout_ms", 20000.0, "tip_probe"
    )
    require_exact(aggregate, "negative_control_pass", True, "tip_probe")

    negative = require_mapping(
        aggregate.get("negative_control"), "tip_probe.negative_control"
    )
    source = require_mapping(
        negative.get("source"), "tip_probe.negative_control.source"
    )
    pinned_records = _read_pinned_open_only_negative_control(source)
    recomputed = _recompute_open_only_negative_control_statistics(
        pinned_records,
        tip=tip,
        ee_step_limit_rad=config.visual_phase_latch.ee_step_limit_rad,
    )
    for name, expected in (
        *tuple(
            _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_WINDOW_STATISTICS.items()
        ),
        (
            "historical_out_of_scope",
            _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_HISTORICAL_STATISTICS,
        ),
        (
            "axis_authority_replay",
            _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_REPLAY_STATISTICS,
        ),
    ):
        _require_open_only_source_statistics(
            recomputed.get(name),
            expected,
            label=f"validator.pinned_v1.{name}",
        )
    for name, expected in recomputed.items():
        _require_open_only_source_statistics(
            negative.get(name),
            expected,
            label=f"tip_probe.negative_control.{name}",
        )
    require_exact(
        negative,
        "semantics",
        "historical_act_axis_authority_counterfactual_replay",
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "axis_authority",
        authority,
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "hardware_completion_validated",
        False,
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "closed_loop_convergence_claimed",
        False,
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "negative_control_pass",
        True,
        "tip_probe.negative_control",
    )
    exact_negative = {
        "visual_command_cycles": _TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_COMMAND_CYCLES,
        "calibrated_window_start_frame_id": _TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID,
        "calibrated_window_end_frame_id": _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID,
        "calibrated_window_frames": 8,
        "calibrated_window_consecutive": True,
        "calibrated_window_all_fresh_unique": True,
        "calibrated_window_all_closed_capture_gate": True,
        "calibrated_window_action_source_counts": {
            "fresh_policy": 3,
            "inference_hold": 5,
        },
        "first_confirmed_hold_frame_id": 492,
        "would_takeover_boundary_frame_id": _TIP_SERVO_OPEN_ONLY_WINDOW_END_FRAME_ID,
        "would_takeover_boundary_action_source": "fresh_policy",
        "actual_tip_servo_arm_command_cycles": 0,
        "actual_tip_servo_descent_command_cycles": 0,
        "actual_tip_servo_jaw_command_cycles": 0,
        "actual_tip_servo_close_command_cycles": 0,
        "expected_outcome": (
            "arm6_policy_j1_j6_search_with_external_capture_jaw_hold_then_arm_hold_and_tip_open"
        ),
    }
    for name, expected in exact_negative.items():
        require_exact(negative, name, expected, "tip_probe.negative_control")

    for name, bounds in (
        ("observed_x_norm", (0.50, 0.545)),
        ("observed_y_norm", (0.42, 0.56)),
        ("observed_area_fraction", (0.017, 0.029)),
        ("observed_jaw_rad", (0.285, 0.325)),
    ):
        stats = require_mapping(
            negative.get(name), f"tip_probe.negative_control.{name}"
        )
        lower = finite_number(
            stats.get("min"),
            f"tip_probe.negative_control.{name}.min",
        )
        upper = finite_number(
            stats.get("max"),
            f"tip_probe.negative_control.{name}.max",
        )
        if not bounds[0] <= lower <= upper <= bounds[1]:
            _reject_visual_audit(
                f"tip_probe.negative_control.{name} must remain inside "
                f"{bounds!r}"
            )
    maximum_speed = finite_number(
        negative.get("observed_max_arm_speed_rad_s"),
        "tip_probe.negative_control.observed_max_arm_speed_rad_s",
    )
    if not 0.0 <= maximum_speed <= 0.07 + 1e-12:
        _reject_visual_audit(
            "tip_probe.negative_control.observed_max_arm_speed_rad_s must "
            "remain within [0, 0.07]"
        )

    historical = require_mapping(
        negative.get("historical_out_of_scope"),
        "tip_probe.negative_control.historical_out_of_scope",
    )
    require_exact(
        historical,
        "policy_arm_motion_observed",
        True,
        "tip_probe.negative_control.historical_out_of_scope",
    )
    minimum_tcp_delta = finite_number(
        historical.get("minimum_measured_tcp_delta_z_m"),
        "tip_probe.negative_control.historical_out_of_scope.minimum_measured_tcp_delta_z_m",
    )
    if minimum_tcp_delta >= -0.020:
        _reject_visual_audit(
            "historical negative control must disclose more than 20 mm of "
            "pre-takeover measured TCP descent"
        )
    historical_jaw_min = finite_number(
        historical.get("final_jaw_target_min_rad"),
        "tip_probe.negative_control.historical_out_of_scope.final_jaw_target_min_rad",
    )
    historical_jaw_max = finite_number(
        historical.get("final_jaw_target_max_rad"),
        "tip_probe.negative_control.historical_out_of_scope.final_jaw_target_max_rad",
    )
    for name in (
        "final_jaw_target_below_capture_cycles",
        "final_jaw_target_above_capture_cycles",
    ):
        value = historical.get(name)
        if type(value) is not int or value <= 0:
            _reject_visual_audit(
                f"tip_probe.negative_control.historical_out_of_scope.{name} "
                "must be a positive integer"
            )

    replay = require_mapping(
        negative.get("axis_authority_replay"),
        "tip_probe.negative_control.axis_authority_replay",
    )
    capture = finite_number(
        replay.get("capture_jaw_target_rad"),
        "tip_probe.negative_control.axis_authority_replay.capture_jaw_target_rad",
    )
    if not 0.285 <= capture <= 0.325:
        _reject_visual_audit(
            "axis-authority replay capture jaw must remain in [0.285, 0.325]"
        )
    if not historical_jaw_min < capture < historical_jaw_max:
        _reject_visual_audit(
            "historical negative control must contain both lower-jaw and "
            "early-open proposals around capture"
        )
    replayed_cycles = replay.get("replayed_seeking_cycles")
    if type(replayed_cycles) is not int or replayed_cycles <= 0:
        _reject_visual_audit(
            "axis-authority replay must include positive seeking cycles"
        )
    require_number(
        replay,
        "published_jaw_min_rad",
        capture,
        "tip_probe.negative_control.axis_authority_replay",
    )
    require_number(
        replay,
        "published_jaw_max_rad",
        capture,
        "tip_probe.negative_control.axis_authority_replay",
    )
    require_number(
        replay,
        "takeover_boundary_published_jaw_target_rad",
        capture,
        "tip_probe.negative_control.axis_authority_replay",
    )
    for name in (
        "suppressed_lower_jaw_proposal_cycles",
        "suppressed_early_open_proposal_cycles",
    ):
        value = replay.get(name)
        if type(value) is not int or value <= 0:
            _reject_visual_audit(
                f"tip_probe.negative_control.axis_authority_replay.{name} "
                "must be a positive integer"
            )
    for name in (
        "takeover_boundary_still_capture_hold",
        "first_open_requires_takeover_and_settle",
        "opened_target_monotonic_nondecreasing",
        "opened_target_step_limit_pass",
        "open_target_reached",
        "adversarial_reclose_suppressed",
        "fault_hold_non_decreasing",
        "seeking_timeout_holds_capture",
    ):
        require_exact(
            replay,
            name,
            True,
            "tip_probe.negative_control.axis_authority_replay",
        )
    opened = replay.get("opened_targets_rad")
    if not isinstance(opened, list) or len(opened) < 2:
        _reject_visual_audit(
            "axis-authority replay opened_targets_rad must contain a transition"
        )
    opened_values = [
        finite_number(
            value,
            f"tip_probe.negative_control.axis_authority_replay.opened_targets_rad[{index}]",
        )
        for index, value in enumerate(opened)
    ]
    if (
        not math.isclose(opened_values[0], capture, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            opened_values[-1],
            tip.pregrasp_open_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or any(
            after < before - 1e-12
            or after - before > config.visual_phase_latch.ee_step_limit_rad + 1e-12
            for before, after in zip(
                opened_values,
                opened_values[1:],
            )
        )
    ):
        _reject_visual_audit(
            "axis-authority replay open targets violate capture, monotonicity, "
            "step limit, or final open target"
        )
    require_number(
        replay,
        "seeking_timeout_ms",
        20000.0,
        "tip_probe.negative_control.axis_authority_replay",
    )
    require_exact(
        replay,
        "counterfactual_open_response_modeled",
        False,
        "tip_probe.negative_control.axis_authority_replay",
    )

    runtime_replay = require_mapping(
        negative.get("runtime_authority_replay"),
        "tip_probe.negative_control.runtime_authority_replay",
    )
    first_open_target = capture + 0.0165
    fault_published_target = capture + 0.033
    expected_runtime_replay = {
        "implementation": (
            "scripts.run_firefly_y6.VisualLatchRuntime.compose_cycle"
        ),
        "software_only": True,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "boundary_events": (
            ["coarse_candidate"] * 5
            + ["coarse_ready_waiting_authorization"] * 2
            + ["coarse_takeover"]
        ),
        "boundary_sequence_exact": True,
        "takeover_discard_policy_chunk": True,
        "takeover_smoother_reset_required": True,
        "takeover_replan_reason": "tip_servo_takeover",
        "takeover_arm_owner": "hold",
        "post_takeover_arm_owner_values": ["hold"],
        "post_takeover_arm_exact_accepted_reference": True,
        "post_takeover_arm_owner_hold": True,
        "tampered_final_arm_rejected": True,
        "takeover_boundary_still_capture_hold": True,
        "first_open_requires_takeover_and_settle": True,
        "first_open_target_rad": first_open_target,
        "policy_role": _ARM6_POLICY_RUNTIME_ROLE,
        "numeric_policy_observation_dimension": 6,
        "raw_policy_action_dimension": 6,
        "numeric_policy_jaw_input": False,
        "numeric_policy_jaw_output": False,
        "arm6_authority": {
            "policy_role": _ARM6_POLICY_RUNTIME_ROLE,
            "observation_state_dimension": 6,
            "raw_policy_action_dimension": 6,
            "candidate_action_dimension": 7,
            "jaw_candidate_source": "startup_accepted_reference",
            "capture_jaw_target_rad": capture,
        },
        "arm6_adapter_contract_pass": True,
        "raw_policy_jaw_telemetry_absent": True,
        "policy_role_telemetry_exact": True,
        "jaw_candidate_source_telemetry_exact": True,
        "seven_dimensional_policy_output_rejected": True,
        "synthesized_jaw_tamper_rejected": True,
        "main_latch_jaw_suppressed": True,
        "fault_hold_non_decreasing": True,
        "fault_published_jaw_target_rad": fault_published_target,
        "seeking_timeout_holds_capture": True,
        "zero_explicit_descent": True,
        "ik_never_invoked": True,
        "close_never_authorized": True,
        "runtime_authority_replay_pass": True,
        "limitation": (
            "Pure software composition replay; it verifies command ownership "
            "and state transitions, not controller or hardware response. Runtime "
            "arm_owner is required to remain hold and every post-takeover arm "
            "target must equal the accepted reference exactly."
        ),
    }
    if set(runtime_replay) != set(expected_runtime_replay):
        _reject_visual_audit(
            "tip_probe.negative_control.runtime_authority_replay must contain "
            "exactly the deterministic schema-6 runtime replay fields"
        )
    for name, expected in expected_runtime_replay.items():
        require_exact(
            runtime_replay,
            name,
            expected,
            "tip_probe.negative_control.runtime_authority_replay",
        )

    summary = require_mapping(report.get("summary"), "summary")
    summary_tip = require_mapping(summary.get("tip_probe"), "summary.tip_probe")
    require_exact(
        summary_tip,
        "demonstration_envelope_checked_episodes",
        expected_episodes,
        "summary.tip_probe",
    )
    require_exact(
        summary_tip,
        "recorded_closed_capture_observed_episodes",
        observed_episodes,
        "summary.tip_probe",
    )
    require_exact(summary_tip, "negative_control_pass", True, "summary.tip_probe")
    require_exact(
        summary_tip,
        "axis_authority",
        authority,
        "summary.tip_probe",
    )
    require_exact(
        summary_tip,
        "hardware_completion_validated",
        False,
        "summary.tip_probe",
    )
    require_exact(
        summary_tip,
        "closed_loop_convergence_claimed",
        False,
        "summary.tip_probe",
    )


def _validate_tip_servo_open_only_episode(
    episode: dict,
    *,
    episode_index: int,
) -> None:
    checks = episode.get("checks")
    required_checks = (
        "tip_probe_open_only_envelope_pass",
        "tip_probe_axis_authority_namespaced",
        "tip_probe_policy_arm_only_before_takeover",
        "tip_probe_jaw_isolated_from_policy_and_main",
        "tip_probe_tip_zero_explicit_descent_authority",
        "tip_probe_close_not_authorized",
        "tip_probe_hardware_completion_not_claimed",
    )
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in required_checks
    ):
        _reject_visual_audit(
            f"episodes[{episode_index}] open-only checks must all be true"
        )

    trace = episode.get("tip_probe_open_only_envelope")
    if not isinstance(trace, dict):
        _reject_visual_audit(
            f"episodes[{episode_index}].tip_probe_open_only_envelope must be an object"
        )
    authority = _schema6_open_only_axis_authority()
    expected = {
        "semantics": (
            "recorded_demo_visual_envelope_axis_authority_not_hardware_response"
        ),
        "axis_authority": authority,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "counterfactual_open_response_modeled": False,
        "trace_envelope_pass": True,
    }
    for name, value in expected.items():
        actual = trace.get(name)
        if not _type_strict_equal(actual, value):
            _reject_visual_audit(
                f"episodes[{episode_index}].tip_probe_open_only_envelope.{name} "
                f"must be exactly {value!r}, found {actual!r}"
            )
    observed = trace.get("recorded_closed_capture_observed")
    if type(observed) is not bool:
        _reject_visual_audit(
            f"episodes[{episode_index}] recorded_closed_capture_observed must be boolean"
        )
    frame = trace.get("recorded_closed_capture_confirmation_frame")
    if observed:
        if type(frame) is not int or frame < 0:
            _reject_visual_audit(
                f"episodes[{episode_index}] observed capture needs a nonnegative frame"
            )
    elif frame is not None:
        _reject_visual_audit(
            f"episodes[{episode_index}] absent capture confirmation must be null"
        )
    trace_checks = trace.get("checks")
    required_trace_checks = (
        "observational_trace_only",
        "axis_authority_namespaced",
        "tip_zero_arm_authority",
        "tip_zero_explicit_descent_authority",
        "jaw_close_not_authorized",
        "hardware_completion_not_claimed",
    )
    if not isinstance(trace_checks, dict) or any(
        trace_checks.get(name) is not True for name in required_trace_checks
    ):
        _reject_visual_audit(
            f"episodes[{episode_index}] open-only trace checks must all be true"
        )


def _validate_tip_servo_probe_audit_contract(
    report: dict,
    *,
    config: FireflyDeploymentConfig,
    expected_episodes: int,
) -> None:
    """Require the schema-3 evidence which authorizes only a no-close probe."""

    if config.visual_phase_latch.tip_servo.mode is not TipServoMode.EXECUTE:
        return
    if (
        config.visual_phase_latch.tip_servo.probe_profile
        is TipServoProbeProfile.OPEN_ONLY
    ):
        _validate_tip_servo_open_only_audit_contract(
            report,
            config=config,
            expected_episodes=expected_episodes,
        )
        return

    def require_mapping(value: object, label: str) -> dict:
        if not isinstance(value, dict):
            _reject_visual_audit(f"{label} must be an object")
        return value

    def require_exact(mapping: dict, name: str, expected: object, label: str) -> None:
        actual = mapping.get(name)
        if not _type_strict_equal(actual, expected):
            _reject_visual_audit(
                f"{label}.{name} must be exactly {expected!r}, found {actual!r}"
            )

    def require_number(
        mapping: dict, name: str, expected: float, label: str
    ) -> None:
        actual = mapping.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            _reject_visual_audit(
                f"{label}.{name} must be exactly {expected!r}, found {actual!r}"
            )

    for name in (
        "hardware_completion_validated",
        "closed_loop_convergence_claimed",
    ):
        require_exact(report, name, False, "report")

    profile = require_mapping(report.get("profile"), "profile")
    profile_tip = require_mapping(profile.get("tip_probe"), "profile.tip_probe")
    profile_tip_config = require_mapping(
        profile_tip.get("config"), "profile.tip_probe.config"
    )
    require_exact(
        profile_tip_config,
        "mode",
        TipServoMode.EXECUTE.value,
        "profile.tip_probe.config",
    )
    require_exact(
        profile_tip_config, "allow_close", False, "profile.tip_probe.config"
    )
    require_number(
        profile_tip_config,
        "max_total_descent_m",
        0.025,
        "profile.tip_probe.config",
    )
    require_number(
        profile_tip_config,
        "descent_step_m",
        0.002,
        "profile.tip_probe.config",
    )
    require_exact(
        profile_tip,
        "semantics",
        "bounded_no-close_recorded_trace_envelope",
        "profile.tip_probe",
    )
    for name in (
        "allow_close",
        "hardware_completion_validated",
        "closed_loop_convergence_claimed",
    ):
        require_exact(profile_tip, name, False, "profile.tip_probe")

    tip_probe = require_mapping(report.get("tip_probe"), "tip_probe")
    for name in (
        "allow_close",
        "close_authorized",
        "hardware_completion_validated",
        "closed_loop_convergence_claimed",
    ):
        require_exact(tip_probe, name, False, "tip_probe")
    require_exact(
        tip_probe,
        "takeover_confirmed_episodes",
        expected_episodes,
        "tip_probe",
    )
    require_exact(
        tip_probe,
        "probe_outcome_confirmed_episodes",
        expected_episodes,
        "tip_probe",
    )
    require_exact(
        tip_probe,
        "demonstration_episode_count",
        expected_episodes,
        "tip_probe",
    )
    require_exact(
        tip_probe,
        "trace_envelope_passed_episodes",
        expected_episodes,
        "tip_probe",
    )
    require_exact(
        tip_probe,
        "recorded_base_z_budget_passed_episodes",
        expected_episodes,
        "tip_probe",
    )
    require_number(
        tip_probe, "configured_max_total_descent_m", 0.025, "tip_probe"
    )
    require_number(tip_probe, "configured_descent_step_m", 0.002, "tip_probe")
    require_exact(
        tip_probe, "configured_maximum_descent_steps", 13, "tip_probe"
    )
    require_exact(tip_probe, "negative_control_pass", True, "tip_probe")

    negative = require_mapping(
        tip_probe.get("negative_control"), "tip_probe.negative_control"
    )
    require_exact(
        negative,
        "semantics",
        "old_policy_trace_expected_safe_noncompletion",
        "tip_probe.negative_control",
    )
    for name in (
        "allow_close",
        "close_authorized",
        "hardware_completion_validated",
        "closed_loop_convergence_claimed",
        "post_takeover_recorded_frames_used_for_convergence",
    ):
        require_exact(negative, name, False, "tip_probe.negative_control")
    require_exact(
        negative,
        "probe_progress_confirmation_frame_id",
        None,
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "visual_would_close_confirmation_frame_id",
        None,
        "tip_probe.negative_control",
    )
    require_exact(
        negative,
        "expected_outcome",
        "captured_then_no_depth_progress",
        "tip_probe.negative_control",
    )
    require_exact(
        negative, "negative_control_pass", True, "tip_probe.negative_control"
    )
    visual_command_cycles = negative.get("visual_command_cycles")
    if type(visual_command_cycles) is not int or visual_command_cycles <= 0:
        _reject_visual_audit(
            "tip_probe.negative_control.visual_command_cycles must be a positive integer"
        )
    takeover_frame_id = negative.get("takeover_frame_id")
    if type(takeover_frame_id) is not int or takeover_frame_id < 0:
        _reject_visual_audit(
            "tip_probe.negative_control.takeover_frame_id must be a nonnegative integer"
        )
    maximum_area = negative.get("maximum_post_takeover_red_area_fraction")
    if (
        isinstance(maximum_area, bool)
        or not isinstance(maximum_area, (int, float))
        or not math.isfinite(float(maximum_area))
        or not 0.0 <= float(maximum_area) < 0.020
    ):
        _reject_visual_audit(
            "tip_probe.negative_control.maximum_post_takeover_red_area_fraction "
            "must be finite and below 0.020"
        )
    source = require_mapping(
        negative.get("source"), "tip_probe.negative_control.source"
    )
    source_path = source.get("path")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not Path(source_path).is_absolute()
    ):
        _reject_visual_audit(
            "tip_probe.negative_control.source.path must be a non-empty absolute path"
        )
    require_exact(
        source,
        "size",
        _TIP_SERVO_NEGATIVE_CONTROL_SIZE,
        "tip_probe.negative_control.source",
    )
    require_exact(
        source,
        "jsonl_records",
        _TIP_SERVO_NEGATIVE_CONTROL_RECORDS,
        "tip_probe.negative_control.source",
    )
    require_exact(
        source,
        "sha256",
        _TIP_SERVO_NEGATIVE_CONTROL_SHA256,
        "tip_probe.negative_control.source",
    )

    summary = require_mapping(report.get("summary"), "summary")
    summary_tip = require_mapping(summary.get("tip_probe"), "summary.tip_probe")
    require_exact(
        summary_tip,
        "trace_envelope_passed_episodes",
        expected_episodes,
        "summary.tip_probe",
    )
    require_exact(
        summary_tip,
        "recorded_base_z_budget_passed_episodes",
        expected_episodes,
        "summary.tip_probe",
    )
    require_exact(
        summary_tip, "negative_control_pass", True, "summary.tip_probe"
    )
    for name in (
        "allow_close",
        "hardware_completion_validated",
        "closed_loop_convergence_claimed",
    ):
        require_exact(summary_tip, name, False, "summary.tip_probe")


def _validate_tip_servo_probe_episode(
    episode: dict,
    *,
    episode_index: int,
) -> None:
    if "tip_probe_open_only_envelope" in episode:
        _validate_tip_servo_open_only_episode(
            episode, episode_index=episode_index
        )
        return
    checks = episode.get("checks")
    if not isinstance(checks, dict):
        _reject_visual_audit(f"episodes[{episode_index}].checks must be an object")
    for name in (
        "tip_probe_trace_envelope_pass",
        "tip_probe_close_not_authorized",
        "tip_probe_hardware_completion_not_claimed",
    ):
        if checks.get(name) is not True:
            _reject_visual_audit(
                f"episodes[{episode_index}].checks.{name} must be true"
            )

    trace = episode.get("tip_probe_trace_envelope")
    if not isinstance(trace, dict):
        _reject_visual_audit(
            f"episodes[{episode_index}].tip_probe_trace_envelope must be an object"
        )
    expected = {
        "semantics": "recorded_success_path_geometric_envelope_only",
        "allow_close": False,
        "close_authorized": False,
        "hardware_completion_validated": False,
        "closed_loop_convergence_claimed": False,
        "settling_and_command_response_modeled": False,
        "trace_envelope_pass": True,
    }
    for name, value in expected.items():
        actual = trace.get(name)
        if actual != value or type(actual) is not type(value):
            _reject_visual_audit(
                f"episodes[{episode_index}].tip_probe_trace_envelope.{name} "
                f"must be exactly {value!r}, found {actual!r}"
            )
    descent = trace.get("required_recorded_base_z_descent_m")
    outcome = trace.get("outcome")
    if outcome not in {"probe_progress_complete", "visual_would_close"}:
        _reject_visual_audit(
            f"episodes[{episode_index}] tip-probe outcome is not an approved confirmation"
        )
    for name in ("takeover_confirmation_frame", "outcome_frame"):
        value = trace.get(name)
        if type(value) is not int or value < 0:
            _reject_visual_audit(
                f"episodes[{episode_index}].tip_probe_trace_envelope.{name} "
                "must be a nonnegative integer"
            )
    outcome_confirmation_name = (
        "probe_progress_confirmation_frame"
        if outcome == "probe_progress_complete"
        else "visual_would_close_confirmation_frame"
    )
    outcome_confirmation = trace.get(outcome_confirmation_name)
    if type(outcome_confirmation) is not int or outcome_confirmation < 0:
        _reject_visual_audit(
            f"episodes[{episode_index}].tip_probe_trace_envelope."
            f"{outcome_confirmation_name} must be a nonnegative integer"
        )
    trace_checks = trace.get("checks")
    required_trace_checks = (
        "takeover_confirmed",
        "probe_outcome_confirmed",
        "within_recorded_base_z_budget",
        "tracking_corridor_continuous",
        "area_drop_fault_free",
        "allow_close_false",
        "close_not_authorized",
        "hardware_completion_not_claimed",
    )
    if not isinstance(trace_checks, dict) or any(
        trace_checks.get(name) is not True for name in required_trace_checks
    ):
        _reject_visual_audit(
            f"episodes[{episode_index}] tip-probe trace checks must all be true"
        )
    drop_streak = trace.get("max_significant_drop_streak_frames")
    if type(drop_streak) is not int or not 0 <= drop_streak < 3:
        _reject_visual_audit(
            f"episodes[{episode_index}] tip-probe area-drop streak must be below 3"
        )
    if (
        isinstance(descent, bool)
        or not isinstance(descent, (int, float))
        or not math.isfinite(float(descent))
        or not 0.0 <= float(descent) <= 0.025 + 1e-12
    ):
        _reject_visual_audit(
            f"episodes[{episode_index}] tip-probe descent must be within [0, 0.025]"
        )
    steps = trace.get("required_descent_steps")
    if type(steps) is not int or not 0 <= steps <= 13:
        _reject_visual_audit(
            f"episodes[{episode_index}] tip-probe required_descent_steps "
            "must be an integer within [0, 13]"
        )


def _capture_visual_latch_audit(
    config: FireflyDeploymentConfig,
    *,
    dataset_root: Path,
    training_input_digest: str,
    implementation_manifest: dict[str, object],
    arm6_source_binding: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not config.visual_phase_latch.enabled:
        return None
    if not isinstance(config.visual_latch_audit_path, str):  # __post_init__ invariant
        raise ValueError("enabled visual phase latch has no audit report path")
    audit_path = _require_no_symlink_components(
        Path(config.visual_latch_audit_path), label="visual latch offline audit"
    )
    configured_digest = config.visual_latch_audit_sha256
    _validate_sha256_hex(configured_digest, label="visual_latch_audit_sha256")
    digest, identity = _hash_regular_file(audit_path)
    if digest != configured_digest:
        _reject_visual_audit(
            f"configured SHA-256 {configured_digest} does not match report {digest}"
        )
    report_bytes = _read_regular_bytes(audit_path, max_bytes=128 * 1024 * 1024)
    if hashlib.sha256(report_bytes).hexdigest() != digest:
        _reject_visual_audit("report changed between hashing and reading")
    if _regular_file_identity(audit_path) != identity:
        _reject_visual_audit("report was replaced during validation")
    report = _decode_strict_json_object(
        report_bytes, label="visual latch offline audit", path=audit_path
    )
    validate_visual_latch_audit_report(
        report,
        config=config,
        dataset_root=dataset_root,
        training_input_digest=training_input_digest,
        implementation_manifest=implementation_manifest,
        arm6_source_binding=arm6_source_binding,
    )
    return {
        "configured_path": config.visual_latch_audit_path,
        "canonical_path": str(audit_path),
        "size": identity[3],
        "sha256": digest,
        "schema_version": _visual_latch_audit_schema_version(config),
        "kind": _VISUAL_LATCH_AUDIT_KIND,
    }


def validate_checkpoint_provenance(
    checkpoint: Path,
    *,
    expected_dataset_repo_id: str,
    expected_dataset_root: Path,
    expected_smolvla_vlm_snapshot: Path | None,
) -> CheckpointProvenance:
    checkpoint, _directories, files = _regular_tree_entries(
        checkpoint, label="deployment checkpoint"
    )
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {checkpoint}")
    train = _read_json_object(checkpoint / "train_config.json", label="checkpoint train_config")
    policy_config = _read_json_object(checkpoint / "config.json", label="checkpoint config")
    processor = _read_json_object(
        checkpoint / "policy_preprocessor.json", label="checkpoint policy preprocessor"
    )

    dataset = train.get("dataset")
    policy = train.get("policy")
    if not isinstance(dataset, dict) or not isinstance(policy, dict):
        raise ValueError("checkpoint train_config needs dataset and policy objects")
    repo_id = dataset.get("repo_id")
    root_value = dataset.get("root")
    policy_type = policy.get("type")
    if repo_id != expected_dataset_repo_id:
        raise PermissionError(
            f"checkpoint was trained on dataset repo {repo_id!r}, expected "
            f"{expected_dataset_repo_id!r}"
        )
    if "episodes" not in dataset or dataset["episodes"] is not None:
        raise PermissionError(
            "deployment checkpoint must be trained on the complete dataset (dataset.episodes=null)"
        )
    if dataset.get("streaming", False) is not False:
        raise PermissionError("deployment checkpoint dataset.streaming must be false")
    if not isinstance(root_value, str) or not root_value or not Path(root_value).is_absolute():
        raise ValueError("checkpoint train_config dataset.root must be a non-empty absolute path")
    trained_root = _require_no_symlink_components(
        Path(root_value), label="checkpoint training dataset"
    )
    expected_root = _require_no_symlink_components(
        expected_dataset_root, label="deployment dataset"
    )
    if trained_root.resolve(strict=True) != expected_root.resolve(strict=True):
        raise PermissionError(
            f"checkpoint was trained on dataset root {trained_root}, expected {expected_root}"
        )
    if policy_type not in {"act", "smolvla"}:
        raise ValueError(
            f"checkpoint train_config policy type {policy_type!r} is not ACT or SmolVLA"
        )
    if policy_config.get("type") != policy_type:
        raise PermissionError(
            "checkpoint config policy type does not match its train_config provenance"
        )
    if train.get("rename_map") != {}:
        raise PermissionError("deployment checkpoint train_config rename_map must be empty")
    steps = processor.get("steps")
    if not isinstance(steps, list):
        raise ValueError("checkpoint policy_preprocessor steps must be a list")
    rename_steps = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("registry_name") == "rename_observations_processor"
    ]
    if len(rename_steps) != 1 or rename_steps[0].get("config") != {"rename_map": {}}:
        raise PermissionError(
            "deployment checkpoint preprocessor must contain exactly one empty rename_map"
        )
    vlm_snapshot: Path | None = None
    if policy_type == "act":
        if expected_smolvla_vlm_snapshot is not None:
            raise PermissionError("ACT deployment must not configure smolvla_vlm_snapshot")
    else:
        if expected_smolvla_vlm_snapshot is None:
            raise PermissionError(
                "SmolVLA deployment requires a pinned materialized smolvla_vlm_snapshot"
            )
        vlm_snapshot, _vlm_manifest = validate_materialized_smolvlm_bundle(
            expected_smolvla_vlm_snapshot
        )
        expected_vlm_name = str(vlm_snapshot)
        identifiers = {
            "checkpoint config.vlm_model_name": policy_config.get("vlm_model_name"),
            "checkpoint train_config policy.vlm_model_name": policy.get("vlm_model_name"),
        }
        mismatched_identifiers = [
            f"{label}={value!r}"
            for label, value in identifiers.items()
            if value != expected_vlm_name
        ]
        if mismatched_identifiers:
            raise PermissionError(
                "SmolVLA checkpoint external VLM identity is not the pinned local bundle: "
                + "; ".join(mismatched_identifiers)
            )
        if policy_config.get("load_vlm_weights") is not False or policy.get(
            "load_vlm_weights"
        ) is not False:
            raise PermissionError(
                "SmolVLA deployment checkpoint must set load_vlm_weights=false; "
                "strict checkpoint loading supplies every trained VLM weight"
            )
        tokenizer_steps = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor"
        ]
        if (
            len(tokenizer_steps) != 1
            or not isinstance(tokenizer_steps[0].get("config"), dict)
            or tokenizer_steps[0]["config"].get("tokenizer_name") != expected_vlm_name
        ):
            raise PermissionError(
                "SmolVLA checkpoint tokenizer must use the same pinned local VLM bundle"
            )
    return CheckpointProvenance(
        checkpoint=checkpoint,
        policy_type=policy_type,
        dataset_repo_id=repo_id,
        dataset_root=trained_root,
        smolvla_vlm_snapshot=vlm_snapshot,
    )


def _manifest_entry(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    digest, identity = _hash_regular_file(path)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": identity[3],
        "sha256": digest,
    }


def _checkpoint_manifest(checkpoint: Path) -> list[dict[str, int | str]]:
    checkpoint, _directories, files = _regular_tree_entries(
        checkpoint, label="deployment checkpoint"
    )
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {checkpoint}")
    return [
        _manifest_entry(path, relative_to=checkpoint)
        for path in sorted(files, key=lambda item: item.relative_to(checkpoint).as_posix())
    ]


def _dataset_control_manifest(dataset_root: Path) -> list[dict[str, int | str]]:
    """Hash metadata and parquet used by replay/training, excluding large videos."""
    dataset_root = _require_no_symlink_components(dataset_root, label="deployment dataset")
    info_path = dataset_root / "meta" / "info.json"
    _regular_file_identity(info_path)
    files: list[Path] = []
    for directory, label in (
        (dataset_root / "meta", "deployment dataset metadata"),
        (dataset_root / "data", "deployment dataset parquet"),
    ):
        _root, _directories, tree_files = _regular_tree_entries(directory, label=label)
        files.extend(tree_files)
    files.sort(key=lambda path: path.relative_to(dataset_root).as_posix())
    if not any(path.suffix == ".parquet" for path in files):
        raise FileNotFoundError(f"dataset contains no parquet action data: {dataset_root}")
    return [
        _manifest_entry(path, relative_to=dataset_root)
        for path in files
    ]


def _stability_item(path: Path, *, root: Path, kind: str) -> dict[str, int | str]:
    identity = _directory_identity(path) if kind == "directory" else _regular_file_identity(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "dev": identity[0],
        "ino": identity[1],
        "mode": identity[2],
        "size": identity[3],
        "mtime_ns": identity[4],
        "ctime_ns": identity[5],
    }


def _parent_can_replace_entry(path: Path) -> bool:
    """Return whether this process can plausibly replace ``path`` through its parent."""
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required source ancestor does not exist: {parent}") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise PermissionError(f"required source ancestor is not a directory: {parent}")
    try:
        accessible = os.access(parent, os.W_OK | os.X_OK, effective_ids=True)
    except TypeError:  # pragma: no cover - platforms without effective_ids support
        accessible = os.access(parent, os.W_OK | os.X_OK)
    effective_uid = os.geteuid()
    return accessible or effective_uid == 0 or parent_stat.st_uid == effective_uid


def _ancestor_stability_records(path: Path) -> list[dict[str, int | str]]:
    """Bind ancestor identities without reacting to unrelated directory contents."""
    absolute = _absolute_without_resolving(path)
    records: list[dict[str, int | str]] = []
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current /= component
        identity = _directory_identity(current)
        record: dict[str, int | str] = {
            "path": str(current),
            "dev": identity[0],
            "ino": identity[1],
            "mode": identity[2],
        }
        # Renaming an entry can update its ctime even when the original tree is
        # restored before the post-load capture. Only bind ctime where this
        # process could plausibly perform that replacement; binding mtime/size
        # for /home-like ancestors would create unrelated false positives.
        if _parent_can_replace_entry(current):
            record["ctime_ns"] = identity[5]
        records.append(record)
    return records


def _deployment_stability_records(
    checkpoint: Path,
    dataset_root: Path,
    contract_path: Path,
    smolvla_vlm_snapshot: Path | None,
    visual_latch_audit_path: Path | None = None,
    visual_audit_source_root: Path | None = None,
    arm6_controlled_artifacts: tuple[dict[str, object], ...] = (),
    verify_arm6_artifact_hashes: bool = False,
) -> dict[str, object]:
    checkpoint, checkpoint_dirs, checkpoint_files = _regular_tree_entries(
        checkpoint, label="deployment checkpoint"
    )
    dataset_root = _require_no_symlink_components(dataset_root, label="deployment dataset")
    dataset_records: list[dict[str, int | str]] = [
        _stability_item(dataset_root, root=dataset_root, kind="directory")
    ]
    for directory, label in (
        (dataset_root / "meta", "deployment dataset metadata"),
        (dataset_root / "data", "deployment dataset parquet"),
    ):
        _root, directories, files = _regular_tree_entries(directory, label=label)
        dataset_records.extend(
            _stability_item(path, root=dataset_root, kind="directory")
            for path in directories
        )
        dataset_records.extend(
            _stability_item(path, root=dataset_root, kind="file") for path in files
        )
    contract_path = _require_no_symlink_components(
        contract_path, label="deployment controller contract"
    )
    contract_identity = _regular_file_identity(contract_path)
    records: dict[str, object] = {
        "schema_version": _DEPLOYMENT_SOURCE_GUARD_SCHEMA_VERSION,
        "checkpoint_ancestors": _ancestor_stability_records(checkpoint),
        "checkpoint_root": str(checkpoint),
        "checkpoint": sorted(
            [
                *(
                    _stability_item(path, root=checkpoint, kind="directory")
                    for path in checkpoint_dirs
                ),
                *(
                    _stability_item(path, root=checkpoint, kind="file")
                    for path in checkpoint_files
                ),
            ],
            key=lambda item: (str(item["path"]), str(item["kind"])),
        ),
        "dataset_root": str(dataset_root),
        "dataset_ancestors": _ancestor_stability_records(dataset_root),
        "dataset": sorted(
            dataset_records, key=lambda item: (str(item["path"]), str(item["kind"]))
        ),
        "controller_contract": {
            "path": str(contract_path),
            "dev": contract_identity[0],
            "ino": contract_identity[1],
            "mode": contract_identity[2],
            "size": contract_identity[3],
            "mtime_ns": contract_identity[4],
            "ctime_ns": contract_identity[5],
        },
        "controller_contract_ancestors": _ancestor_stability_records(contract_path),
    }
    if smolvla_vlm_snapshot is not None:
        vlm_root, vlm_dirs, vlm_files = _regular_tree_entries(
            smolvla_vlm_snapshot, label="materialized SmolVLM bundle"
        )
        records["smolvla_vlm_snapshot_root"] = str(vlm_root)
        records["smolvla_vlm_snapshot_ancestors"] = _ancestor_stability_records(vlm_root)
        records["smolvla_vlm_snapshot"] = sorted(
            [
                *(
                    _stability_item(path, root=vlm_root, kind="directory")
                    for path in vlm_dirs
                ),
                *(
                    _stability_item(path, root=vlm_root, kind="file")
                    for path in vlm_files
                ),
            ],
            key=lambda item: (str(item["path"]), str(item["kind"])),
        )
    if visual_latch_audit_path is not None:
        audit_path = _require_no_symlink_components(
            visual_latch_audit_path, label="visual latch offline audit"
        )
        audit_identity = _regular_file_identity(audit_path)
        records["visual_latch_audit"] = {
            "path": str(audit_path),
            "dev": audit_identity[0],
            "ino": audit_identity[1],
            "mode": audit_identity[2],
            "size": audit_identity[3],
            "mtime_ns": audit_identity[4],
            "ctime_ns": audit_identity[5],
        }
        records["visual_latch_audit_ancestors"] = _ancestor_stability_records(
            audit_path
        )
    if visual_audit_source_root is not None:
        policy_root, policy_dirs, policy_files = _regular_tree_entries(
            dataset_root,
            label="arm6 policy projection dataset",
        )
        records["arm6_policy_dataset_complete"] = sorted(
            [
                *(
                    _stability_item(path, root=policy_root, kind="directory")
                    for path in policy_dirs
                ),
                *(
                    _stability_item(path, root=policy_root, kind="file")
                    for path in policy_files
                ),
            ],
            key=lambda item: (str(item["path"]), str(item["kind"])),
        )
        visual_root, visual_dirs, visual_files = _regular_tree_entries(
            visual_audit_source_root,
            label="arm6 visual audit source dataset",
        )
        records["visual_audit_source_root"] = str(visual_root)
        records["visual_audit_source_ancestors"] = _ancestor_stability_records(
            visual_root
        )
        records["visual_audit_source"] = sorted(
            [
                *(
                    _stability_item(path, root=visual_root, kind="directory")
                    for path in visual_dirs
                ),
                *(
                    _stability_item(path, root=visual_root, kind="file")
                    for path in visual_files
                ),
            ],
            key=lambda item: (str(item["path"]), str(item["kind"])),
        )
    if arm6_controlled_artifacts:
        artifact_records: list[dict[str, int | str]] = []
        for artifact in arm6_controlled_artifacts:
            role = artifact["role"]
            path = _require_no_symlink_components(
                Path(str(artifact["path"])),
                label=f"arm6 controlled artifact {role}",
            )
            if verify_arm6_artifact_hashes:
                current_digest, identity = _hash_regular_file(path)
                if (
                    current_digest != artifact["sha256"]
                    or identity[3] != artifact["size"]
                ):
                    raise PermissionError(
                        f"arm6 controlled artifact {role} changed during deployment capture"
                    )
            else:
                identity = _regular_file_identity(path)
            artifact_records.append(
                {
                    "role": str(role),
                    "path": str(path),
                    "dev": identity[0],
                    "ino": identity[1],
                    "mode": identity[2],
                    "size": identity[3],
                    "mtime_ns": identity[4],
                    "ctime_ns": identity[5],
                }
            )
        records["arm6_controlled_artifacts"] = sorted(
            artifact_records,
            key=lambda item: (str(item["role"]), str(item["path"])),
        )
    return records


def capture_deployment_sources(
    config: FireflyDeploymentConfig,
    *,
    arm6_attestation_loader: Arm6AttestationLoader = _load_fixed_arm6_attestation,
) -> DeploymentSourceCapture:
    """Validate and bind every offline source before any hardware connection."""
    checkpoint = _absolute_without_resolving(Path(config.checkpoint))
    dataset_root = _absolute_without_resolving(Path(config.dataset_root))
    contract_path = _absolute_without_resolving(Path(config.dataset_controller_contract))
    vlm_snapshot = (
        _absolute_without_resolving(Path(config.smolvla_vlm_snapshot))
        if config.smolvla_vlm_snapshot is not None
        else None
    )
    arm6_binding = None
    visual_dataset_root = dataset_root
    visual_dataset_repo_id = config.dataset_repo_id
    if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE:
        assert config.visual_audit_source_root is not None
        assert config.visual_audit_source_repo_id is not None
        visual_dataset_root = _absolute_without_resolving(
            Path(config.visual_audit_source_root)
        )
        visual_dataset_repo_id = config.visual_audit_source_repo_id
    visual_latch_audit_path = None
    if config.visual_phase_latch.enabled:
        if not isinstance(config.visual_latch_audit_path, str) or not (
            config.visual_latch_audit_path.strip()
        ):
            raise ValueError(
                "enabled visual phase latch requires visual_latch_audit_path"
            )
        _validate_sha256_hex(
            config.visual_latch_audit_sha256,
            label="visual_latch_audit_sha256",
        )
        configured_audit_path = Path(config.visual_latch_audit_path).expanduser()
        if not configured_audit_path.is_absolute():
            raise ValueError("visual_latch_audit_path must be an absolute path")
        visual_latch_audit_path = _absolute_without_resolving(
            configured_audit_path
        )
    if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE:
        stability_before_consumer = _deployment_stability_records(
            checkpoint,
            dataset_root,
            contract_path,
            vlm_snapshot,
            visual_latch_audit_path,
            visual_dataset_root,
        )
        arm6_binding = _validate_arm6_attestation_binding(
            config,
            arm6_attestation_loader,
        )
        checkpoint = arm6_binding.selected_checkpoint
        stability_after_consumer = _deployment_stability_records(
            checkpoint,
            dataset_root,
            contract_path,
            vlm_snapshot,
            visual_latch_audit_path,
            visual_dataset_root,
        )
        if stability_before_consumer != stability_after_consumer:
            raise PermissionError(
                "arm6 checkpoint, projection, visual source, or controller changed "
                "during consumer verification"
            )
    stability_before = _deployment_stability_records(
        checkpoint,
        dataset_root,
        contract_path,
        vlm_snapshot,
        visual_latch_audit_path,
        (
            visual_dataset_root
            if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE
            else None
        ),
        () if arm6_binding is None else arm6_binding.controlled_artifacts,
        False,
    )
    provenance = validate_checkpoint_provenance(
        checkpoint,
        expected_dataset_repo_id=config.dataset_repo_id,
        expected_dataset_root=dataset_root,
        expected_smolvla_vlm_snapshot=vlm_snapshot,
    )
    try:
        visual_training_inputs = capture_training_inputs(
            visual_dataset_root,
            visual_dataset_repo_id,
            controller_contract=contract_path,
            expected_task=config.task,
        )
        if arm6_binding is None:
            policy_training_inputs = visual_training_inputs
            if provenance.policy_type == "smolvla":
                verify_training_attestation(checkpoint, policy_training_inputs)
            policy_training_input_digest = policy_training_inputs.digest
        else:
            policy_attestation = arm6_binding.source_binding["policy_attestation"]
            if not isinstance(policy_attestation, dict):  # validated above
                raise PermissionError("arm6 policy_attestation must be an object")
            policy_training_input_digest = policy_attestation.get("input_digest")
            _validate_sha256_hex(
                policy_training_input_digest,
                label="arm6 policy input digest",
            )
            projection_binding = arm6_binding.source_binding["projection_binding"]
            if not isinstance(projection_binding, dict):  # validated above
                raise PermissionError("arm6 projection_binding must be an object")
            projection_source = projection_binding.get("source")
            if not isinstance(projection_source, dict):  # validated above
                raise PermissionError("arm6 projection source must be an object")
            upstream_digest = projection_source.get(
                "upstream_training_input_digest"
            )
            _validate_sha256_hex(
                upstream_digest,
                label="arm6 projection upstream training input digest",
            )
            if upstream_digest != visual_training_inputs.digest:
                raise PermissionError(
                    "arm6 projection upstream digest differs from the verified "
                    "7-D visual audit source"
                )
    except TrainingAttestationError as exc:
        raise PermissionError(str(exc)) from exc
    visual_implementation_manifest = None
    visual_latch_audit = None
    if config.visual_phase_latch.enabled:
        visual_implementation_manifest = visual_phase_latch_implementation_manifest()
        visual_latch_audit = _capture_visual_latch_audit(
            config,
            dataset_root=visual_dataset_root,
            training_input_digest=visual_training_inputs.digest,
            implementation_manifest=visual_implementation_manifest,
            arm6_source_binding=(
                None if arm6_binding is None else arm6_binding.source_binding
            ),
        )
    checkpoint_manifest = _checkpoint_manifest(checkpoint)
    dataset_manifest = _dataset_control_manifest(dataset_root)
    contract_digest, contract_identity = _hash_regular_file(contract_path)
    vlm_manifest = None
    if provenance.smolvla_vlm_snapshot is not None:
        _validated_vlm_root, vlm_manifest = validate_materialized_smolvlm_bundle(
            provenance.smolvla_vlm_snapshot
        )
    stability_after = _deployment_stability_records(
        checkpoint,
        dataset_root,
        contract_path,
        provenance.smolvla_vlm_snapshot,
        visual_latch_audit_path,
        (
            visual_dataset_root
            if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE
            else None
        ),
        () if arm6_binding is None else arm6_binding.controlled_artifacts,
        arm6_binding is not None,
    )
    if stability_before != stability_after:
        raise PermissionError(
            "deployment checkpoint, dataset, or controller contract changed during validation; "
            "hardware was not connected"
        )
    if (
        visual_implementation_manifest is not None
        and visual_phase_latch_implementation_manifest()
        != visual_implementation_manifest
    ):
        raise PermissionError(
            "visual latch implementation changed during audit validation; "
            "hardware was not connected"
        )
    control = {
        "fps": config.fps,
        "n_action_steps": config.n_action_steps,
        "enable_action_slew_rate_adapter": config.enable_action_slew_rate_adapter,
        "grasp_phase_correction": config.grasp_phase_correction.provenance(),
        "record_video": config.record_video,
    }
    # Preserve every existing reviewed fingerprint while binding every visual
    # detector/state-machine constant for deployments which explicitly opt in.
    if config.visual_phase_latch.enabled:
        control["visual_phase_latch"] = visual_phase_latch_provenance(
            config.visual_phase_latch
        )
        control["visual_phase_latch_implementation"] = visual_implementation_manifest
        control["visual_latch_audit"] = visual_latch_audit
    if config.inference_seed is not None:
        control["inference_seed"] = config.inference_seed
        control["inference_seed_strategy"] = (
            "base_seed_plus_zero_based_inference_generation"
        )
    # Preserve existing reviewed fingerprints. Only deployments opting into
    # fresh-observation synchronous replanning bind the new scheduling field.
    if config.act_chunk_inference_mode != "pipelined":
        control["act_chunk_inference_mode"] = config.act_chunk_inference_mode

    robot_fingerprint = _canonical_fingerprint_value(config.robot)
    # Preserve every existing reviewed fingerprint. The selector did not
    # exist for legacy profiles; only the new non-default envelope is material
    # deployment provenance.
    if config.robot.motion_envelope == "initial":
        robot_fingerprint["fields"].pop("motion_envelope", None)
    # Camera freshness became a follower-wide hard safety limit after the
    # reviewed legacy deployments.  Omit only its exact default to preserve
    # those fingerprints; an enabled visual profile binds the same 100 ms
    # value independently and its allowlist rejects any drift.
    if math.isclose(
        float(config.robot.camera_max_age_s), 0.100, rel_tol=0.0, abs_tol=1e-12
    ):
        robot_fingerprint["fields"].pop("camera_max_age_s", None)

    payload = {
        "schema_version": _DEPLOYMENT_FINGERPRINT_SCHEMA_VERSION,
        "policy_runtime_role": config.policy_runtime_role,
        "checkpoint": {
            "configured_path": str(Path(config.checkpoint).expanduser()),
            "canonical_path": str(provenance.checkpoint),
            "policy_type": provenance.policy_type,
            "files": checkpoint_manifest,
            "training_input_digest": policy_training_input_digest,
        },
        "smolvla_vlm_snapshot": (
            None
            if provenance.smolvla_vlm_snapshot is None
            else {
                "configured_path": config.smolvla_vlm_snapshot,
                "canonical_path": str(provenance.smolvla_vlm_snapshot),
                "manifest": vlm_manifest,
            }
        ),
        "task": config.task,
        "dataset": {
            "repo_id": config.dataset_repo_id,
            "configured_root": str(Path(config.dataset_root).expanduser()),
            "canonical_root": str(dataset_root),
            "control_files": dataset_manifest,
            "controller_contract": {
                "configured_path": str(Path(config.dataset_controller_contract).expanduser()),
                "canonical_path": str(contract_path),
                "size": contract_identity[3],
                "sha256": contract_digest,
            },
            "replay_episode": config.replay_episode,
            "replay_speed": config.replay_speed,
        },
        "visual_audit_source": (
            None
            if arm6_binding is None
            else {
                "repo_id": visual_dataset_repo_id,
                "configured_root": config.visual_audit_source_root,
                "canonical_root": str(visual_dataset_root),
                "training_input_digest": visual_training_inputs.digest,
                "control_files": _dataset_control_manifest(visual_dataset_root),
            }
        ),
        "arm6_attestation": (
            None
            if arm6_binding is None
            else {
                "configured_run_commit_sha256": config.arm6_run_commit_sha,
                "source_binding": arm6_binding.source_binding,
                "controlled_artifacts": list(arm6_binding.controlled_artifacts),
            }
        ),
        "control": control,
        "robot": robot_fingerprint,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    stability_encoded = json.dumps(
        stability_after,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return DeploymentSourceCapture(
        deployment_fingerprint=hashlib.sha256(encoded).hexdigest(),
        stability_digest=hashlib.sha256(stability_encoded).hexdigest(),
        visual_latch_audit=visual_latch_audit,
        arm6_attestation=(
            None if arm6_binding is None else arm6_binding.source_binding
        ),
    )


def compute_deployment_fingerprint(config: FireflyDeploymentConfig) -> str:
    """Validate then hash the exact policy, dataset, task, contract, and robot."""
    return capture_deployment_sources(config).deployment_fingerprint


def require_unchanged_deployment_sources(
    before: DeploymentSourceCapture, after: DeploymentSourceCapture
) -> None:
    if before != after:
        raise PermissionError(
            "deployment sources changed during offline preload; hardware was not connected"
        )


def _validate_deployment_fingerprint(value: str) -> None:
    _validate_sha256_hex(value, label="deployment fingerprint")

def _audit_stage_telemetry(output_dir: Path, run_label: str) -> dict[str, int | str | bool]:
    telemetry_path = output_dir / f"{run_label}.jsonl"
    try:
        telemetry_bytes = _read_regular_bytes(
            telemetry_path, max_bytes=128 * 1024 * 1024
        )
        telemetry_text = telemetry_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{run_label!r} telemetry is not UTF-8") from exc
    counts = {
        "stage_complete": 0,
        "stage_interrupted": 0,
        "safety_rejection": 0,
        "safety_stop": 0,
        "deployment_provenance": 0,
    }
    successful_cycle_event = "shadow_cycle" if run_label == "shadow" else "command_cycle"
    successful_cycle_count = 0
    first_wall_time_ns: int | None = None
    deployment_fingerprint: str | None = None
    operator_authorized_skip_replay_for_act_3s = False
    operator_authorized_direct_act_30s = False
    operator_authorized_tip_servo_probe_act_30s = False
    operator_authorized_tip_servo_open_only_act_30s = False
    operator_authorized_direct_act_65s = False
    operator_authorized_direct_act_90s = False
    for line_number, line in enumerate(telemetry_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{run_label!r} telemetry line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"{run_label!r} telemetry line {line_number} is not an object")
        wall_time_ns = event.get("wall_time_ns")
        if isinstance(wall_time_ns, int):
            first_wall_time_ns = (
                wall_time_ns
                if first_wall_time_ns is None
                else min(first_wall_time_ns, wall_time_ns)
            )
        event_name = event.get("event")
        if event_name in counts:
            counts[event_name] += 1
        if event_name == "deployment_provenance":
            value = event.get("deployment_fingerprint")
            _validate_deployment_fingerprint(value)
            if deployment_fingerprint is not None and deployment_fingerprint != value:
                raise ValueError(
                    f"{run_label!r} telemetry contains conflicting deployment fingerprints"
                )
            deployment_fingerprint = value
            replay_skip_value = event.get(
                "operator_authorized_skip_replay_for_act_3s", False
            )
            if not isinstance(replay_skip_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean operator replay-skip marker"
                )
            operator_authorized_skip_replay_for_act_3s = replay_skip_value
            direct_act_30s_value = event.get("operator_authorized_direct_act_30s", False)
            if not isinstance(direct_act_30s_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean "
                    "operator direct-ACT-30s marker"
                )
            operator_authorized_direct_act_30s = direct_act_30s_value
            tip_probe_act_30s_value = event.get(
                "operator_authorized_tip_servo_probe_act_30s", False
            )
            if not isinstance(tip_probe_act_30s_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean "
                    "tip-servo probe ACT-30s marker"
                )
            operator_authorized_tip_servo_probe_act_30s = (
                tip_probe_act_30s_value
            )
            tip_open_only_act_30s_value = event.get(
                "operator_authorized_tip_servo_open_only_act_30s", False
            )
            if not isinstance(tip_open_only_act_30s_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean "
                    "tip-servo open-only ACT-30s marker"
                )
            operator_authorized_tip_servo_open_only_act_30s = (
                tip_open_only_act_30s_value
            )
            direct_act_65s_value = event.get("operator_authorized_direct_act_65s", False)
            if not isinstance(direct_act_65s_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean "
                    "operator direct-ACT-65s marker"
                )
            operator_authorized_direct_act_65s = direct_act_65s_value
            direct_act_90s_value = event.get("operator_authorized_direct_act_90s", False)
            if not isinstance(direct_act_90s_value, bool):
                raise ValueError(
                    f"{run_label!r} deployment provenance has a non-boolean "
                    "operator direct-ACT-90s marker"
                )
            operator_authorized_direct_act_90s = direct_act_90s_value

        if event_name == successful_cycle_event:
            if event.get("safety_ok") is not True:
                raise ValueError(
                    f"{run_label!r} contains a {successful_cycle_event} without safety_ok=true"
                )
            successful_cycle_count += 1
        if event_name == "stage_complete" and event.get("stage") != run_label:
            raise ValueError(
                f"{run_label!r} stage_complete names {event.get('stage')!r} instead"
            )
    if counts["stage_complete"] != 1:
        raise ValueError(
            f"{run_label!r} requires exactly one stage_complete, found {counts['stage_complete']}"
        )
    for failure_event in ("safety_rejection", "safety_stop", "stage_interrupted"):
        if counts[failure_event]:
            raise ValueError(
                f"{run_label!r} contains {counts[failure_event]} {failure_event} event(s) "
                "and cannot be approved"
            )
    if counts["deployment_provenance"] != 1 or deployment_fingerprint is None:
        raise ValueError(
            f"{run_label!r} requires exactly one deployment_provenance event, "
            f"found {counts['deployment_provenance']}"
        )
    if operator_authorized_skip_replay_for_act_3s:
        if run_label != "act_3s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope operator-authorized replay-skip marker"
            )
        raise ValueError(
            "'act_3s' used the one-shot operator-authorized replay skip and cannot be approved; "
            "complete and review replay, then run a new 3-second policy stage"
        )
    if operator_authorized_direct_act_30s:
        if run_label != "act_30s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope operator-authorized direct-ACT-30s marker"
            )
        raise ValueError(
            "'act_30s' used the one-shot operator-authorized direct ACT run and cannot be "
            "approved; complete and review the normal deployment ladder, then run a new "
            "30-second policy stage"
        )
    if operator_authorized_tip_servo_probe_act_30s:
        if run_label != "act_30s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope tip-servo probe ACT-30s marker"
            )
        raise ValueError(
            "'act_30s' used the one-shot tip-servo probe authorization and cannot "
            "be approved or promoted as normal policy evidence"
        )
    if operator_authorized_tip_servo_open_only_act_30s:
        if run_label != "act_30s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope tip-servo open-only ACT-30s marker"
            )
        raise ValueError(
            "'act_30s' used the one-shot tip-servo open-only authorization and "
            "cannot be approved or promoted as normal policy evidence"
        )
    if operator_authorized_direct_act_65s:
        if run_label != "act_65s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope operator-authorized direct-ACT-65s marker"
            )
        raise ValueError(
            "act_65s used the one-shot operator-authorized direct ACT run and cannot be "
            "approved; complete and review the normal deployment ladder, then run a new "
            "65-second policy stage"
        )
    if operator_authorized_direct_act_90s:
        if run_label != "act_90s":
            raise ValueError(
                f"{run_label!r} contains an out-of-scope operator-authorized direct-ACT-90s marker"
            )
        raise ValueError(
            "act_90s used the one-shot operator-authorized direct ACT run and cannot be "
            "approved; complete and review the normal deployment ladder, then run a new "
            "90-second policy stage"
        )
    if successful_cycle_count == 0:
        raise ValueError(
            f"{run_label!r} contains no successful {successful_cycle_event} and cannot be approved"
        )
    if first_wall_time_ns is None:
        raise ValueError(f"{run_label!r} telemetry has no wall_time_ns provenance")
    return {
        **counts,
        "successful_cycle_event": successful_cycle_event,
        "successful_cycle_count": successful_cycle_count,
        "first_wall_time_ns": first_wall_time_ns,
        "deployment_fingerprint": deployment_fingerprint,
        "operator_authorized_skip_replay_for_act_3s": False,
        "operator_authorized_direct_act_30s": False,
        "operator_authorized_tip_servo_probe_act_30s": False,
        "operator_authorized_tip_servo_open_only_act_30s": False,
        "operator_authorized_direct_act_65s": False,
        "operator_authorized_direct_act_90s": False,
        "telemetry_sha256": hashlib.sha256(telemetry_bytes).hexdigest(),
    }


def invalidate_stage_reviews(output_dir: Path, run_label: str) -> None:
    """Invalidate this stage and every downstream review before a new run."""
    if run_label not in _ACCEPTANCE_ORDER:
        raise ValueError(f"unknown deployment stage label {run_label!r}")
    ledger_path = output_dir / "acceptance.json"
    if not ledger_path.exists():
        return
    ledger = json.loads(ledger_path.read_text())
    if not isinstance(ledger, dict):
        raise ValueError(f"invalid acceptance ledger {ledger_path}")
    changed = False
    start = _ACCEPTANCE_ORDER.index(run_label)
    for label in _ACCEPTANCE_ORDER[start:]:
        changed = ledger.pop(label, None) is not None or changed
    if not changed:
        return
    temporary_path = ledger_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(ledger_path)


def mark_stage_reviewed(
    output_dir: Path,
    run_label: str,
    deployment_fingerprint: str,
    *,
    require_video: bool,
    verify_before_commit: Callable[[], None] | None = None,
) -> Path:
    """Bind an explicit human review to safe, immutable run artifacts."""
    require_acceptance_order(output_dir, run_label, deployment_fingerprint)
    audit = _audit_stage_telemetry(output_dir, run_label)
    if audit["deployment_fingerprint"] != deployment_fingerprint:
        raise ValueError(
            f"{run_label!r} telemetry belongs to a different deployment fingerprint"
        )
    videos = []
    for video_path in sorted(output_dir.glob(f"{run_label}_*.mp4")):
        video_digest, video_identity = _hash_regular_file(video_path)
        if video_identity[4] < audit["first_wall_time_ns"]:
            continue
        videos.append({"name": video_path.name, "sha256": video_digest})
    if require_video and not videos:
        raise FileNotFoundError(
            f"no video created by the reviewed {run_label!r} run was found in {output_dir}"
        )
    if verify_before_commit is not None:
        verify_before_commit()
        require_acceptance_order(output_dir, run_label, deployment_fingerprint)
    ledger_path = output_dir / "acceptance.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if not isinstance(ledger, dict):
        raise ValueError(f"invalid acceptance ledger {ledger_path}")
    ledger[run_label] = {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "reviewed": True,
        "reviewed_at_ns": time.time_ns(),
        "deployment_fingerprint": deployment_fingerprint,
        "telemetry_sha256": audit["telemetry_sha256"],
        "successful_cycle_event": audit["successful_cycle_event"],
        "successful_cycle_count": audit["successful_cycle_count"],
        "safety_rejection_count": audit["safety_rejection"],
        "safety_stop_count": audit["safety_stop"],
        "videos": videos,
    }
    temporary_path = ledger_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(ledger_path)
    return ledger_path


@dataclass(frozen=True)
class ActionSlewRateResult:
    """A policy chunk made continuous from one accepted command reference."""

    adapted_chunk: tuple[tuple[float, ...], ...]
    step_limited: tuple[bool, ...]
    limited_value_counts: tuple[int, ...]
    limited_step_count: int
    limited_value_count: int
    arm_limited_value_count: int
    ee_limited_value_count: int
    max_raw_arm_step_rad: float
    max_raw_ee_step_rad: float
    max_adapted_arm_step_rad: float
    max_adapted_ee_step_rad: float


def assess_act_start_pose(
    config: GraspPhaseCorrectionConfig,
    measured,
) -> dict[str, object]:
    """Evaluate the optional one-shot measured 7D ACT start-pose gate."""

    enabled = bool(config.act_start_center_rad)
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "measured_rad": None,
            "center_rad": None,
            "tolerance_rad": None,
            "absolute_error_rad": None,
            "failed_indices": [],
        }
    values = np.asarray(measured, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("ACT start-pose gate requires a finite measured vector shaped (7,)")
    center = np.asarray(config.act_start_center_rad, dtype=np.float64)
    tolerance = np.asarray(config.act_start_tolerance_rad, dtype=np.float64)
    absolute_error = np.abs(values - center)
    within = absolute_error <= tolerance
    return {
        "enabled": True,
        "passed": bool(np.all(within)),
        "measured_rad": [float(value) for value in values],
        "center_rad": [float(value) for value in center],
        "tolerance_rad": [float(value) for value in tolerance],
        "absolute_error_rad": [float(value) for value in absolute_error],
        "failed_indices": [
            int(index) for index, accepted in enumerate(within) if not accepted
        ],
    }


class GraspPhaseCorrector:
    """Correct one early-close phase only at the demonstrated pickup and release poses."""

    def __init__(self, config: GraspPhaseCorrectionConfig):
        self.config = config
        self.phase = "monitoring" if config.enabled else "disabled"
        self.premature_close_seen = False
        self.pickup_streak = 0
        self.release_streak = 0
        self.chunk_count = 0
        self.raw_open_seen = False
        self.latched_at_chunk = None

    def apply(self, raw_actions, reference) -> tuple[tuple[tuple[float, ...], ...], dict]:
        expected_shape = (10, 7)
        source = np.asarray(raw_actions, dtype=np.float64)
        previous = np.asarray(reference, dtype=np.float64)
        if source.shape != expected_shape or not np.isfinite(source).all():
            raise ValueError(
                f"grasp phase correction requires a finite action chunk shaped {expected_shape}"
            )
        if previous.shape != (7,) or not np.isfinite(previous).all():
            raise ValueError(
                "grasp phase correction requires a finite reference shaped (7,)"
            )

        raw_ee_min = float(np.min(source[:, 6]))
        raw_ee_max = float(np.max(source[:, 6]))
        phase_before = self.phase
        event = None
        pickup_gate = False
        release_gate = False
        commanded_open_ready = None
        pregrasp_open_active = False
        progress_timeout = False
        progress_timeout_kind = None
        assisted = source.copy()

        if self.config.enabled:
            self.chunk_count += 1
            if raw_ee_min >= self.config.pickup_open_threshold_rad:
                self.raw_open_seen = True

            center = np.asarray(self.config.pickup_center_rad, dtype=np.float64)
            tolerance = np.asarray(
                self.config.pickup_tolerance_rad, dtype=np.float64
            )
            pickup_gate = bool(
                np.all(np.abs(previous[:6] - center) <= tolerance)
            )
            release_gate = bool(
                previous[0] >= self.config.release_joint_1_min_rad
                and previous[1] >= self.config.release_joint_2_min_rad
                and self.config.release_joint_3_min_rad
                <= previous[2]
                <= self.config.release_joint_3_max_rad
                and previous[3] <= self.config.release_joint_4_max_rad
            )
            v3_open_assistance = self.config.pregrasp_open_target_rad is not None
            if v3_open_assistance:
                commanded_open_ready = bool(
                    previous[6] >= float(self.config.pregrasp_open_ready_rad)
                )

            if self.phase == "monitoring":
                if (
                    v3_open_assistance
                    and pickup_gate
                    and not self.premature_close_seen
                    and self.raw_open_seen
                ):
                    self.phase = "natural_passthrough"
                    event = "natural_pickup_observed"
                else:
                    if (
                        raw_ee_min <= self.config.premature_close_threshold_rad
                        and not pickup_gate
                    ):
                        if not self.premature_close_seen:
                            event = "premature_close_observed"
                        self.premature_close_seen = True

                    if v3_open_assistance:
                        pickup_candidate = (
                            self.premature_close_seen
                            and pickup_gate
                            and commanded_open_ready
                        )
                    else:
                        pickup_candidate = (
                            self.premature_close_seen
                            and pickup_gate
                            and raw_ee_min >= self.config.pickup_open_threshold_rad
                        )
                    self.pickup_streak = (
                        self.pickup_streak + 1 if pickup_candidate else 0
                    )
                    if self.pickup_streak >= self.config.pickup_confirm_chunks:
                        self.phase = "latched"
                        self.latched_at_chunk = self.chunk_count
                        event = "grasp_latched"
            elif self.phase == "latched":
                release_candidate = (
                    release_gate
                    and raw_ee_min >= self.config.release_open_threshold_rad
                )
                self.release_streak = (
                    self.release_streak + 1 if release_candidate else 0
                )
                if self.release_streak >= self.config.release_confirm_chunks:
                    self.phase = "released"
                    event = "release_unlatched"

            pregrasp_open_active = bool(
                v3_open_assistance
                and self.phase == "monitoring"
                and self.premature_close_seen
            )
            if pregrasp_open_active:
                assisted[:, 6] = np.maximum(
                    assisted[:, 6], float(self.config.pregrasp_open_target_rad)
                )
            if self.phase == "latched":
                if v3_open_assistance:
                    assisted[:, 6] = self.config.close_target_rad
                else:
                    assisted[:, 6] = np.minimum(
                        assisted[:, 6], self.config.close_target_rad
                    )

            raw_open_deadline = self.config.raw_open_progress_deadline_chunks
            pickup_deadline = self.config.pickup_progress_deadline_chunks
            release_deadline = self.config.release_progress_deadline_chunks
            if (
                raw_open_deadline is not None
                and self.phase == "monitoring"
                and self.chunk_count >= raw_open_deadline
                and not self.raw_open_seen
            ):
                progress_timeout_kind = "raw_open"
            elif (
                pickup_deadline is not None
                and self.phase == "monitoring"
                and self.chunk_count >= pickup_deadline
            ):
                progress_timeout_kind = "pickup"
            elif (
                release_deadline is not None
                and self.phase == "latched"
                and self.latched_at_chunk is not None
                and self.chunk_count - self.latched_at_chunk >= release_deadline
            ):
                progress_timeout_kind = "release"
            progress_timeout = progress_timeout_kind is not None
            if progress_timeout:
                event = f"{progress_timeout_kind}_progress_timeout"

        assisted_chunk = tuple(
            tuple(float(value) for value in row) for row in assisted
        )
        telemetry = {
            "enabled": self.config.enabled,
            "phase_before": phase_before,
            "phase_after": self.phase,
            "event": event,
            "premature_close_seen": self.premature_close_seen,
            "pickup_gate": pickup_gate,
            "pickup_confirm_streak": self.pickup_streak,
            "release_gate": release_gate,
            "release_confirm_streak": self.release_streak,
            "raw_ee_min_rad": raw_ee_min,
            "raw_ee_max_rad": raw_ee_max,
            "close_target_rad": self.config.close_target_rad,
            "pregrasp_open_active": pregrasp_open_active,
            "pregrasp_open_target_rad": self.config.pregrasp_open_target_rad,
            "pregrasp_open_ready_rad": self.config.pregrasp_open_ready_rad,
            "commanded_open_ready": commanded_open_ready,
            "chunk_count": self.chunk_count,
            "raw_open_seen": self.raw_open_seen,
            "raw_open_progress_deadline_chunks":
                self.config.raw_open_progress_deadline_chunks,
            "pickup_progress_deadline_chunks":
                self.config.pickup_progress_deadline_chunks,
            "release_progress_deadline_chunks":
                self.config.release_progress_deadline_chunks,
            "latched_at_chunk": self.latched_at_chunk,
            "progress_timeout": progress_timeout,
            "progress_timeout_kind": progress_timeout_kind,
            "transformed": not np.array_equal(assisted, source),
            "assisted_ee_min_rad": float(np.min(assisted[:, 6])),
            "assisted_ee_max_rad": float(np.max(assisted[:, 6])),
        }
        return assisted_chunk, telemetry


def adapt_action_chunk_slew_rate(
    raw_actions,
    reference,
    *,
    arm_step_limit_rad: float,
    ee_step_limit_rad: float,
    arm_step_limits_rad: tuple[float, ...] | list[float] | None = None,
) -> ActionSlewRateResult:
    """Clip a Firefly SmolVLA 10x7 chunk without weakening follower checks.

    Each raw target is compared with the preceding *adapted* target, beginning
    at ``reference``. This makes every returned frame continuous across both
    steps and chunk boundaries. The follower remains responsible for its
    independent position, live-velocity, tracking, freshness, and ownership
    checks.
    """

    expected_shape = (10, 7)
    try:
        source = np.asarray(raw_actions, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"raw action chunk must have shape {expected_shape}") from exc
    if source.shape != expected_shape:
        raise ValueError(
            f"raw action chunk has shape {tuple(source.shape)}, expected {expected_shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("raw action chunk contains NaN or infinite value")

    try:
        previous = np.asarray(reference, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action slew-rate reference must have shape (7,)") from exc
    if previous.shape != (7,):
        raise ValueError(
            f"action slew-rate reference has shape {tuple(previous.shape)}, expected (7,)"
        )
    if not np.isfinite(previous).all():
        raise ValueError("action slew-rate reference contains NaN or infinite value")

    for name, value in (
        ("arm_step_limit_rad", arm_step_limit_rad),
        ("ee_step_limit_rad", ee_step_limit_rad),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if arm_step_limits_rad is None:
        resolved_arm_limits = (float(arm_step_limit_rad),) * 6
    else:
        try:
            resolved_arm_limits = tuple(float(value) for value in arm_step_limits_rad)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "arm_step_limits_rad must contain six positive finite values"
            ) from exc
        if len(resolved_arm_limits) != 6 or not all(
            math.isfinite(value) and value > 0 for value in resolved_arm_limits
        ):
            raise ValueError(
                "arm_step_limits_rad must contain six positive finite values"
            )
        if any(value > arm_step_limit_rad for value in resolved_arm_limits):
            raise ValueError(
                "arm_step_limits_rad values must not exceed arm_step_limit_rad"
            )
    limits = np.asarray(
        [*resolved_arm_limits, float(ee_step_limit_rad)],
        dtype=np.float64,
    )

    adapted_rows: list[tuple[float, ...]] = []
    step_limited: list[bool] = []
    limited_value_counts: list[int] = []
    arm_limited_value_count = 0
    ee_limited_value_count = 0
    max_raw_arm_step_rad = 0.0
    max_raw_ee_step_rad = 0.0
    max_adapted_arm_step_rad = 0.0
    max_adapted_ee_step_rad = 0.0
    for raw in source:
        delta = raw - previous
        limited_mask = np.abs(delta) > limits
        clipped_delta = np.clip(delta, -limits, limits)
        # Preserve an untouched raw value exactly; only clipped coordinates
        # need the reference-plus-delta calculation.
        adapted = np.where(limited_mask, previous + clipped_delta, raw)

        count = int(np.count_nonzero(limited_mask))
        step_limited.append(count > 0)
        limited_value_counts.append(count)
        arm_limited_value_count += int(np.count_nonzero(limited_mask[:6]))
        ee_limited_value_count += int(np.count_nonzero(limited_mask[6:]))
        max_raw_arm_step_rad = max(max_raw_arm_step_rad, float(np.max(np.abs(delta[:6]))))
        max_raw_ee_step_rad = max(max_raw_ee_step_rad, float(np.max(np.abs(delta[6:]))))
        max_adapted_arm_step_rad = max(
            max_adapted_arm_step_rad, float(np.max(np.abs(adapted[:6] - previous[:6])))
        )
        max_adapted_ee_step_rad = max(
            max_adapted_ee_step_rad, float(np.max(np.abs(adapted[6:] - previous[6:])))
        )
        adapted_rows.append(tuple(float(value) for value in adapted))
        previous = adapted

    limited_value_count = sum(limited_value_counts)
    return ActionSlewRateResult(
        adapted_chunk=tuple(adapted_rows),
        step_limited=tuple(step_limited),
        limited_value_counts=tuple(limited_value_counts),
        limited_step_count=sum(step_limited),
        limited_value_count=limited_value_count,
        arm_limited_value_count=arm_limited_value_count,
        ee_limited_value_count=ee_limited_value_count,
        max_raw_arm_step_rad=max_raw_arm_step_rad,
        max_raw_ee_step_rad=max_raw_ee_step_rad,
        max_adapted_arm_step_rad=max_adapted_arm_step_rad,
        max_adapted_ee_step_rad=max_adapted_ee_step_rad,
    )


def action_dict(feature_names: tuple[str, ...] | list[str], values) -> dict[str, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != len(feature_names):
        raise ValueError(f"action has {len(vector)} values, expected {len(feature_names)}")
    return dict(zip(feature_names, vector, strict=True))


def interpolate_actions(actions, speed: float = 0.5) -> list[tuple[float, ...]]:
    """Linearly resample a 30 Hz trajectory for slower 30 Hz playback."""
    source = np.asarray(actions, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("actions must be a non-empty [frames, joints] array")
    if not np.isfinite(source).all():
        raise ValueError("actions contain NaN or infinite value")
    if not 0 < speed <= 1:
        raise ValueError("replay speed must be in (0, 1]")
    if source.shape[0] == 1:
        return [tuple(float(value) for value in source[0])]
    positions = np.arange(0.0, source.shape[0] - 1, speed, dtype=np.float64)
    if positions.size == 0 or positions[-1] != source.shape[0] - 1:
        positions = np.append(positions, source.shape[0] - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, source.shape[0] - 1)
    alpha = (positions - lower)[:, None]
    result = source[lower] * (1.0 - alpha) + source[upper] * alpha
    return [tuple(float(value) for value in row) for row in result]


def validate_replay_start(
    measured,
    first_action,
    *,
    arm_dof: int,
    arm_tolerance_rad: float = 0.05,
    ee_tolerance_rad: float = 0.02,
) -> None:
    measured_vector = np.asarray(measured, dtype=np.float64)
    action_vector = np.asarray(first_action, dtype=np.float64)
    if measured_vector.shape != action_vector.shape or measured_vector.ndim != 1:
        raise ValueError("measured state and first replay action must be equal-length vectors")
    arm_error = float(np.max(np.abs(measured_vector[:arm_dof] - action_vector[:arm_dof])))
    ee_error = float(np.max(np.abs(measured_vector[arm_dof:] - action_vector[arm_dof:])))
    if arm_error > arm_tolerance_rad or ee_error > ee_tolerance_rad:
        raise SafetyStopError(
            f"episode start mismatch: arm={arm_error:.6f} rad (limit {arm_tolerance_rad:.6f}), "
            f"EE={ee_error:.6f} (limit {ee_tolerance_rad:.6f}); position with teleoperation, "
            "then stop arm_teleop before retrying"
        )


@dataclass(frozen=True)
class InferenceChunk:
    generation: int
    actions: tuple[tuple[float, ...], ...]
    latency_s: float
    error: BaseException | None = None


class InferenceWorker:
    """One-slot observation worker; the 30 Hz sender never performs model inference."""

    def __init__(self, infer):
        self._infer = infer
        self._requests: queue.Queue[tuple[int, dict] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceChunk] = queue.Queue(maxsize=1)
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run, name="firefly-policy-inference", daemon=True
        )
        self._thread.start()

    def submit(self, observation: dict) -> int:
        if self._requests.full():
            raise RuntimeError("an ACT inference request is already pending")
        self._generation += 1
        self._requests.put_nowait((self._generation, observation))
        return self._generation

    def get(self, timeout_s: float | None = None) -> InferenceChunk:
        return self._results.get(timeout=timeout_s)

    def get_nowait(self) -> InferenceChunk | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            generation, observation = request
            started_at = time.monotonic()
            try:
                actions = tuple(tuple(float(value) for value in row) for row in self._infer(observation))
                result = InferenceChunk(generation, actions, time.monotonic() - started_at)
            except BaseException as exc:  # surfaced synchronously to the command thread
                result = InferenceChunk(generation, (), time.monotonic() - started_at, error=exc)
            try:
                self._results.put_nowait(result)
            except queue.Full:
                _ = self._results.get_nowait()
                self._results.put_nowait(result)


class StageVideoRecorder:
    """Optional per-camera MP4 recorder for mandatory post-tier review."""

    def __init__(self, output_dir: Path, stage: str, fps: int, enabled: bool):
        self.output_dir = output_dir
        self.stage = stage
        self.fps = fps
        self.enabled = enabled
        self._writers: dict[str, object] = {}

    def add_observation(self, observation: dict) -> None:
        if not self.enabled:
            return
        import cv2

        for name, image in observation.items():
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                continue
            writer = self._writers.get(name)
            if writer is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / f"{self.stage}_{name}.mp4"
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer {path}")
                self._writers[name] = writer
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()


def _require_fullcycle_v4_visual_phase_latch(
    config: FireflyDeploymentConfig,
) -> None:
    """Allow only the reviewed D405/RGB latch and its exact camera contract."""

    visual = config.visual_phase_latch
    if not visual.enabled:
        return
    expected_disabled = VisualPhaseLatchConfig(enabled=True)
    expected_probe = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(
            mode=TipServoMode.EXECUTE,
            allow_close=False,
            max_total_descent_m=0.025,
        ),
    )
    # The legacy constructor above remains readable for archived schema-3
    # reports, but it is deliberately not allowlisted for new fullcycle runs.
    expected_probe = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=_fullcycle_v4_open_only_tip_servo_config(),
    )
    actual_provenance = visual_phase_latch_provenance(visual)
    approved_provenance = {
        json.dumps(visual_phase_latch_provenance(candidate), sort_keys=True)
        for candidate in (expected_disabled, expected_probe)
    }
    if json.dumps(actual_provenance, sort_keys=True) not in approved_provenance:
        raise PermissionError(
            "fullcycle_v4 visual phase latch requires the exact calibrated "
            "D405 HSV/ROI/EE/FK/fail-closed profile, optionally with only the "
            "exact zero-arm, zero-descent, jaw-open-only tip-servo execute probe"
        )
    if not math.isclose(
        float(visual.ee_step_limit_rad),
        float(config.robot.ee_step_limit_rad),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PermissionError(
            "visual phase latch EE slew must exactly match the follower EE step limit"
        )
    if not math.isclose(
        float(config.robot.camera_max_age_s) * 1000.0,
        float(visual.max_frame_age_ms),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PermissionError(
            "visual phase latch camera freshness must exactly match the follower limit"
        )

    wrist = config.robot.cameras.get("wrist")
    if wrist is None:
        raise PermissionError("visual phase latch requires the configured wrist D405")
    rotation = getattr(wrist, "rotation", None)
    rotation_value = getattr(rotation, "value", rotation)
    color_mode = getattr(wrist, "color_mode", None)
    color_mode_value = getattr(color_mode, "value", color_mode)
    camera_contract = {
        "type": getattr(wrist, "type", None),
        "serial": str(getattr(wrist, "serial_number_or_name", "")),
        "rotation": rotation_value,
        "width": getattr(wrist, "width", None),
        "height": getattr(wrist, "height", None),
        "fps": getattr(wrist, "fps", None),
        "color_mode": color_mode_value,
        "use_rgb": getattr(wrist, "use_rgb", None),
        "use_depth": getattr(wrist, "use_depth", None),
    }
    expected_contract = {
        "type": "intelrealsense",
        "serial": visual.d405_serial,
        "rotation": visual.d405_rotation_degrees,
        "width": visual.expected_width,
        "height": visual.expected_height,
        "fps": config.fps,
        "color_mode": "rgb",
        "use_rgb": True,
        "use_depth": False,
    }
    if camera_contract != expected_contract:
        raise PermissionError(
            "visual phase latch requires the exact wrist D405 serial, RGB 640x480@30, "
            "rotation=180, depth-disabled camera contract; "
            f"actual={camera_contract!r}"
        )


def robot_config_for_stage(
    config: FireflyDeploymentConfig,
    stage: Stage,
    telemetry_path: Path,
) -> HexZenohFollowerConfig:
    ramp_stop_values = {
        "arm_step_limit_rad": 0.035,
        "ee_step_limit_rad": 0.012,
        "arm_tracking_error_rad": 0.150,
        "ee_tracking_error_rad": 0.050,
        "arm_kp": 40.0,
        "arm_kd": 2.0,
        "ee_kp": 150.0,
        "ee_kd_default_scale": 0.5,
    }
    hold_values = {
        "arm_tracking_error_rad": 0.050,
        "ee_tracking_error_rad": 0.020,
        "ee_kp": 75.0,
        "ee_kd_default_scale": 0.5,
    }
    fullcycle_v4_values = {
        "arm_tracking_error_rad": 0.150,
        "ee_tracking_error_rad": 0.050,
        "arm_kp": 40.0,
        "arm_kd": 2.0,
        "ee_kp": 150.0,
        "ee_kd_default_scale": 0.5,
    }
    hold_step_profiles = (
        (0.005, 0.002),
        (0.002, 0.002),
    )
    fast_arm_step_profile = (0.003, 0.003, 0.002, 0.003, 0.003, 0.003)
    if config.robot.mode != "shadow":
        raise PermissionError("deployment YAML robot.mode must remain 'shadow'")
    hold_profile = config.robot.arm_timeout_behavior == "hold"
    expected_values = dict(hold_values if hold_profile else ramp_stop_values)
    profile_name = "persistent HOLD" if hold_profile else "RAMP_STOP"
    if not hold_profile and (
        config.robot.arm_step_limits_rad
        or config.robot.ee_closing_tracking_error_rad is not None
        or config.grasp_phase_correction.enabled
        or config.visual_phase_latch.enabled
    ):
        raise PermissionError(
            "per-joint slew, directional EE tracking, grasp correction, and visual latch are "
            "approved only for the fingerprint-bound persistent HOLD profile"
        )
    if hold_profile:
        configured_step_profile = (
            float(config.robot.arm_step_limit_rad),
            float(config.robot.ee_step_limit_rad),
        )
        configured_arm_steps = config.robot.resolved_arm_step_limits_rad()
        fullcycle_v4_step_profile = (
            config.robot.motion_envelope == "fullcycle_v4"
            and not config.robot.arm_step_limits_rad
            and configured_step_profile == (0.0435, 0.0165)
        )
        if config.visual_phase_latch.enabled and not fullcycle_v4_step_profile:
            raise PermissionError(
                "visual phase latch is approved only for the exact fullcycle_v4 "
                "persistent-HOLD motion profile"
            )
        legacy_step_profile = (
            not config.robot.arm_step_limits_rad
            and configured_step_profile in hold_step_profiles
        )
        fast_step_profile = (
            configured_step_profile == (0.003, 0.003)
            and len(configured_arm_steps) == len(fast_arm_step_profile)
            and all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    configured_arm_steps, fast_arm_step_profile, strict=True
                )
            )
        )
        if not legacy_step_profile and not fast_step_profile and not fullcycle_v4_step_profile:
            raise PermissionError(
                "HOLD slew must be exactly one of the approved scalar pairs "
                f"{hold_step_profiles!r} for the fingerprint-bound persistent HOLD "
                "profile, or the exact fingerprint-bound fast per-joint profile "
                f"{fast_arm_step_profile!r}/EE 0.003, or the attested fullcycle-v4 "
                "0.0435/0.0165 profile"
            )
        if config.robot.motion_envelope == "fullcycle_v4" and not fullcycle_v4_step_profile:
            raise PermissionError(
                "fullcycle_v4 motion envelope must use the exact scalar 0.0435/0.0165 profile"
            )
        if fullcycle_v4_step_profile:
            expected_start_profile = GraspPhaseCorrectionConfig(
                enabled=False,
                act_start_center_rad=list(_FULLCYCLE_V4_START_CENTER_RAD),
                act_start_tolerance_rad=list(_FULLCYCLE_V4_START_TOLERANCE_RAD),
            )
            visual_source_repo_id = (
                config.visual_audit_source_repo_id
                if config.policy_runtime_role == _ARM6_POLICY_RUNTIME_ROLE
                else config.dataset_repo_id
            )
            if visual_source_repo_id != _FULLCYCLE_V4_DATASET_REPO_ID:
                raise PermissionError(
                    "fullcycle_v4 motion envelope requires the exact attested "
                    "fullcycle-v4 dataset as its original 7-D visual audit source"
                )
            if config.task != _FULLCYCLE_V4_TASK:
                raise PermissionError(
                    "fullcycle_v4 motion envelope requires the exact full-cycle task text"
                )
            if not config.enable_action_slew_rate_adapter:
                raise PermissionError(
                    "fullcycle_v4 motion envelope requires the cross-chunk action slew adapter"
                )
            if config.grasp_phase_correction.provenance() != expected_start_profile.provenance():
                raise PermissionError(
                    "fullcycle_v4 must disable fixed-joint grasp correction and use the exact "
                    "50-episode demonstrated ACT start-pose gate"
                )
            _require_fullcycle_v4_visual_phase_latch(config)
            if config.inference_seed != 1000:
                raise PermissionError(
                    "fullcycle_v4 deployment requires deterministic inference_seed=1000"
                )
            if config.act_chunk_inference_mode != "synchronous_hold":
                raise PermissionError(
                    "fullcycle_v4 deployment requires fresh-observation synchronous HOLD "
                    "inference between action chunks"
                )
            if not (
                config.robot.enable_arm_gravity_ff
                and config.robot.enable_ee_payload_gravity_ff
            ):
                raise PermissionError(
                    "fullcycle_v4 motion envelope requires the validated arm+GR80 payload "
                    "gravity feed-forward profile"
                )
            if config.robot.ee_closing_tracking_error_rad is None or not math.isclose(
                float(config.robot.ee_closing_tracking_error_rad),
                0.060,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise PermissionError(
                    "fullcycle_v4 motion envelope requires closing-contact EE tracking "
                    "threshold exactly 0.060 rad"
                )
            expected_values = dict(fullcycle_v4_values)
        if fast_step_profile:
            if not (
                config.robot.enable_arm_gravity_ff
                and config.robot.enable_ee_payload_gravity_ff
            ):
                raise PermissionError(
                    "the fast per-joint profile requires full arm+GR80 payload gravity feed-forward"
                )
            if not math.isclose(float(config.robot.ee_kp), 150.0):
                raise PermissionError("the fast per-joint profile requires ee_kp exactly 150.0")
            if config.robot.ee_closing_tracking_error_rad is None or not math.isclose(
                float(config.robot.ee_closing_tracking_error_rad),
                0.060,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise PermissionError(
                    "the fast per-joint profile requires closing-contact EE tracking "
                    "threshold exactly 0.060 rad"
                )
            expected_values["ee_kp"] = 150.0
        elif (
            config.robot.ee_closing_tracking_error_rad is not None
            and not fullcycle_v4_step_profile
        ):
            raise PermissionError(
                "directional EE tracking is approved only for the exact fast per-joint profile"
            )
        if config.grasp_phase_correction.enabled:
            if (
                stage != "act"
                or not fast_step_profile
                or not config.enable_action_slew_rate_adapter
            ):
                raise PermissionError(
                    "grasp phase correction is approved only for ACT with the exact "
                    "fast90 slew profile and action adapter"
                )
            expected_phase_correction_v1 = GraspPhaseCorrectionConfig(
                enabled=True,
                pickup_center_rad=[
                    0.395,
                    -0.148,
                    2.324,
                    0.623,
                    0.299,
                    1.031,
                ],
                pickup_tolerance_rad=[
                    0.100,
                    0.100,
                    0.100,
                    0.120,
                    0.100,
                    0.150,
                ],
            )
            expected_phase_correction_v2 = replace(
                expected_phase_correction_v1,
                pickup_open_threshold_rad=0.340,
            )
            expected_phase_correction_v3 = replace(
                expected_phase_correction_v2,
                pregrasp_open_target_rad=0.380,
                pregrasp_open_ready_rad=0.370,
                raw_open_progress_deadline_chunks=45,
                pickup_progress_deadline_chunks=180,
                release_progress_deadline_chunks=90,
                act_start_center_rad=[
                    -0.0077796346,
                    -1.4772257805,
                    3.0181598663,
                    0.3821436167,
                    -0.0341406241,
                    -0.0972255841,
                    0.3245764077,
                ],
                act_start_tolerance_rad=[
                    0.100,
                    0.100,
                    0.080,
                    0.150,
                    0.080,
                    0.180,
                    0.080,
                ],
            )
            expected_phase_correction_v5 = replace(
                expected_phase_correction_v3,
                act_start_tolerance_rad=[
                    0.100,
                    0.100,
                    0.150,
                    0.150,
                    0.080,
                    0.180,
                    0.080,
                ],
                act_start_ee_preopen_enabled=True,
            )
            configured_profile = (
                config.grasp_phase_correction.provenance(),
                config.inference_seed,
            )
            expected_profiles = (
                (expected_phase_correction_v1.provenance(), None),
                (expected_phase_correction_v2.provenance(), None),
                (expected_phase_correction_v3.provenance(), 1000),
                (expected_phase_correction_v5.provenance(), 1000),
            )
            if configured_profile not in expected_profiles:
                raise PermissionError(
                    "grasp phase correction and inference seed must match an exact "
                    "fingerprint-bound fast90 v1, v2, v3, or v5 profile"
                )
        configured_arm_kp = config.robot.resolved_arm_kp()
        configured_arm_kd = config.robot.resolved_arm_kd()
        vector_gain_profile = isinstance(config.robot.arm_kp, list)
        if vector_gain_profile != isinstance(config.robot.arm_kd, list):
            raise PermissionError("arm_kp and arm_kd must both be scalars or both be lists")
        experimental_arm_kp = (60.0, 60.0, 72.0, 72.0, 60.0, 60.0)
        experimental_arm_kd = (2.5, 2.5, 2.75, 2.75, 2.5, 2.5)
        if vector_gain_profile:
            configured_phase_profile = (
                config.grasp_phase_correction.provenance(),
                config.inference_seed,
            )
            if not (
                stage == "act"
                and fast_step_profile
                and config.robot.enable_arm_gravity_ff
                and config.robot.enable_ee_payload_gravity_ff
                and config.grasp_phase_correction.enabled
                and configured_phase_profile
                == (expected_phase_correction_v5.provenance(), 1000)
                and configured_arm_kp == experimental_arm_kp
                and configured_arm_kd == experimental_arm_kd
            ):
                raise PermissionError(
                    "per-joint arm gains require ACT with the exact v5 recovery-preopen "
                    "profile and KP=(60,60,72,72,60,60), "
                    "KD=(2.5,2.5,2.75,2.75,2.5,2.5)"
                )
        else:
            configured_gain_profile = (configured_arm_kp[0], configured_arm_kd[0])
            approved_gain_profiles = [(60.0, 2.5)] if fast_step_profile else [(40.0, 2.0)]
            if (not fast_step_profile and
                config.robot.enable_arm_gravity_ff
                and config.robot.enable_ee_payload_gravity_ff
                and configured_step_profile == (0.002, 0.002)
            ):
                approved_gain_profiles.append((60.0, 2.5))
            if configured_gain_profile not in approved_gain_profiles:
                raise PermissionError(
                    "arm_kp/arm_kd must be exactly one of "
                    f"{tuple(approved_gain_profiles)!r} for the fingerprint-bound "
                    "persistent HOLD profile"
                )
            expected_values["arm_kp"], expected_values["arm_kd"] = configured_gain_profile
    if config.robot.enable_arm_gravity_ff:
        if not hold_profile:
            raise PermissionError(
                "arm gravity feed-forward is approved only for the fingerprint-bound "
                "persistent HOLD profile"
            )
        legacy_payload_step_profile = (
            not config.robot.arm_step_limits_rad
            and configured_step_profile == (0.002, 0.002)
        )
        if (
            not legacy_payload_step_profile
            and not fast_step_profile
            and not fullcycle_v4_step_profile
        ):
            raise PermissionError(
                "arm gravity feed-forward requires the exact legacy 0.002/0.002 profile "
                "or the fingerprint-bound fast per-joint/fullcycle-v4 profile"
            )
        if not config.robot.enable_ee_payload_gravity_ff:
            raise PermissionError(
                "arm-only gravity feed-forward is not approved for deployment; "
                "the fingerprint-bound GR80 payload model is required"
            )
        expected_gravity_cap = 8.0
        if not math.isclose(
            float(config.robot.arm_gravity_ff_max_abs_nm), expected_gravity_cap
        ):
            raise PermissionError(
                "arm_gravity_ff_max_abs_nm must be exactly "
                f"{expected_gravity_cap} for the fingerprint-bound gravity profile"
            )
        if config.robot.enable_ee_payload_gravity_ff:
            expected_payload_strings = {
                "expected_arm_urdf_sha256":
                    "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291",
                "expected_ee_urdf_sha256":
                    "f74b3e76b14175c788c5ef70dd0c1941958d229a8461da6f20e17543e4ba1114",
                "expected_arm_urdf_root_link": "base_link",
                "expected_arm_urdf_tip_link": "link_6",
                "expected_ee_urdf_root_link": "base_link",
                "expected_ee_urdf_tip_link": "tcp",
            }
            for name, expected in expected_payload_strings.items():
                if getattr(config.robot, name) != expected:
                    raise PermissionError(
                        f"{name} must match the fingerprint-bound GR80 payload profile"
                    )
            if not math.isclose(
                float(config.robot.expected_ee_payload_mass_kg),
                0.41,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise PermissionError(
                    "expected_ee_payload_mass_kg must be exactly 0.41"
                )
            expected_payload_com = (
                0.005537804878,
                0.000026829268,
                0.048889970714,
            )
            if len(config.robot.expected_ee_payload_com_xyz_m) != 3 or any(
                not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    config.robot.expected_ee_payload_com_xyz_m,
                    expected_payload_com,
                    strict=True,
                )
            ):
                raise PermissionError(
                    "expected_ee_payload_com_xyz_m must match the fingerprint-bound "
                    "strictly parsed GR80 inertial"
                )
    for name, expected in expected_values.items():
        if not math.isclose(float(getattr(config.robot, name)), expected):
            raise PermissionError(
                f"{name} must be {expected} for the fingerprint-bound {profile_name} profile"
            )

    mode: Literal["shadow", "execute"] = "shadow" if stage == "shadow" else "execute"
    return replace(config.robot, mode=mode, telemetry_path=str(telemetry_path))


def hold_action(robot: HexZenohFollower, observation: dict) -> dict[str, float]:
    target = robot.command_target
    if target is not None:
        return target
    return {name: float(observation[name]) for name in robot.action_features}
