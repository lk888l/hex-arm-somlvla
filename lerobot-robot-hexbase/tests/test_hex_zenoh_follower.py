from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

import lerobot_robot_hexbase.hex_zenoh_follower as follower_module
import lerobot_robot_hexbase.firefly_deployment as deployment_module
from lerobot_robot_hexbase.arm_gravity import SerialRevoluteGravityModel
from lerobot_robot_hexbase.payload_gravity import (
    LumpedInertial,
    with_tip_payload,
)
from lerobot_robot_hexbase.firefly_deployment import (
    FireflyDeploymentConfig,
    GraspPhaseCorrectionConfig,
    InferenceWorker,
    adapt_action_chunk_slew_rate,
    capture_deployment_sources,
    compute_deployment_fingerprint,
    invalidate_stage_reviews,
    interpolate_actions,
    mark_stage_reviewed,
    require_acceptance_order,
    require_arm6_policy_stage_gate,
    require_tip_servo_open_only_probe_gate,
    require_arm_hold_timeout_override_gate,
    require_hardware_safety_gate,
    require_tip_servo_execute_probe_gate,
    require_unchanged_deployment_sources,
    robot_config_for_stage,
    validate_replay_start,
    visual_phase_latch_provenance,
)
from lerobot_robot_hexbase.firefly_tip_servo import (
    TipServoConfig,
    TipServoMode,
    TipServoProbeProfile,
)
from lerobot_robot_hexbase.firefly_visual_latch import VisualPhaseLatchConfig
from lerobot_robot_hexbase.hex_zenoh_follower import (
    HexZenohFollower,
    HexZenohFollowerConfig,
    SafetyStopError,
)
from lerobot_robot_hexbase.zenoh_helpers import arm_pb2

import common_pb2
import ee_pb2
import robot_pb2


ARM_PREFIX = "hexmeow/test-controller/arm1"
EE_PREFIX = "hexmeow/test-controller/ee1"
ARM_NAMES = [f"joint_{index}" for index in range(1, 7)]
EE_NAMES = ["jaw_joint1"]
DEPLOYMENT_FINGERPRINT = "a" * 64
DEPLOYMENT_TASK = "Move the red block."
ARM_URDF_XML = """\
<robot name="test_arm">
  <link name="base_link"/>
  <link name="link_1"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <link name="link_2"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <link name="link_3"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <link name="link_4"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <link name="link_5"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <link name="link_6"><inertial><origin xyz="0.05 0 0"/><mass value="0.1"/></inertial></link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/><child link="link_2"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint_3" type="revolute">
    <parent link="link_2"/><child link="link_3"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint_4" type="revolute">
    <parent link="link_3"/><child link="link_4"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint_5" type="revolute">
    <parent link="link_4"/><child link="link_5"/>
    <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
  </joint>
  <joint name="joint_6" type="revolute">
    <parent link="link_5"/><child link="link_6"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""
ARM_URDF_SHA256 = hashlib.sha256(ARM_URDF_XML.encode("utf-8")).hexdigest()
EE_PAYLOAD_URDF_XML = """\
<robot name="test_ee">
  <link name="base_link">
    <inertial>
      <origin xyz="0.005537804878 0.000026829268 0.048889970714"/>
      <mass value="0.41"/>
    </inertial>
  </link>
  <link name="tcp"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="base_link"/><child link="tcp"/><origin xyz="0 0 0.13"/>
  </joint>
</robot>
"""
EE_PAYLOAD_URDF_SHA256 = hashlib.sha256(
    EE_PAYLOAD_URDF_XML.encode("utf-8")
).hexdigest()


def _write_valid_deployment_assets(tmp_path: Path, *, policy_type: str = "act"):
    dataset = tmp_path / "dataset"
    data_dir = dataset / "data" / "chunk-000"
    episode_dir = dataset / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    actions = [
        [0.01 * frame, 0.1, 1.0, 0.2, -0.2, 0.05, 0.3 + 0.005 * frame]
        for frame in range(3)
    ]
    states = [[value - 0.003 for value in action] for action in actions]
    pq.write_table(
        pa.table(
            {
                "action": pa.array(actions, type=pa.list_(pa.float32(), 7)),
                "observation.state": pa.array(states, type=pa.list_(pa.float32(), 7)),
                "timestamp": pa.array([0.0, 1 / 30, 2 / 30], type=pa.float32()),
                "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                "episode_index": pa.array([0, 0, 0], type=pa.int64()),
                "index": pa.array([0, 1, 2], type=pa.int64()),
                "task_index": pa.array([0, 0, 0], type=pa.int64()),
            }
        ),
        data_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "tasks": pa.array([[DEPLOYMENT_TASK]], type=pa.list_(pa.string())),
                "length": pa.array([3], type=pa.int64()),
            }
        ),
        episode_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([DEPLOYMENT_TASK], type=pa.string()),
            }
        ),
        dataset / "meta" / "tasks.parquet",
    )
    joint_names = [
        *[f"joint_{index}.pos" for index in range(1, 7)],
        "jaw_joint1.pos",
    ]
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": joint_names},
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": joint_names,
        },
    }
    for image_key in ("observation.images.scene", "observation.images.wrist"):
        features[image_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False, "video.fps": 30},
        }
        video = dataset / "videos" / image_key / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"test-only-placeholder")
    for scalar, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[scalar] = {"dtype": dtype, "shape": [1], "names": None}
    info_path = dataset / "meta" / "info.json"
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "features": features,
                "total_episodes": 1,
                "total_frames": 3,
                "total_tasks": 1,
                "robot_type": "hex_zenoh_observer",
            }
        )
    )
    contract = tmp_path / "controller_contract.json"
    contract.write_text(
        json.dumps(
            {
                "source": "saved live test contract",
                "joint_names": joint_names,
                "pos_min": [-2.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.0],
                "pos_max": [2.0, 2.0, 3.2, 2.0, 2.0, 2.0, 0.6],
                "effective_step_limits": [0.0435] * 6 + [0.0165],
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    vlm_snapshot = tmp_path / "materialized-smolvlm"
    if policy_type == "smolvla":
        vlm_snapshot.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model-v1")
    policy_values = {"type": policy_type}
    if policy_type == "smolvla":
        policy_values.update(
            {
                "vlm_model_name": str(vlm_snapshot),
                "load_vlm_weights": False,
            }
        )
    (checkpoint / "config.json").write_text(json.dumps(policy_values))
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": "local/validated_v2",
                    "root": str(dataset),
                    "episodes": None,
                },
                "policy": policy_values,
                "rename_map": {},
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {"rename_map": {}},
                    },
                    *(
                        [
                            {
                                "registry_name": "tokenizer_processor",
                                "config": {"tokenizer_name": str(vlm_snapshot)},
                            }
                        ]
                        if policy_type == "smolvla"
                        else []
                    ),
                ]
            }
        )
    )
    config = FireflyDeploymentConfig(
        robot=_config("shadow"),
        checkpoint=str(checkpoint),
        smolvla_vlm_snapshot=str(vlm_snapshot) if policy_type == "smolvla" else None,
        dataset_repo_id="local/validated_v2",
        dataset_root=str(dataset),
        dataset_controller_contract=str(contract),
        task=DEPLOYMENT_TASK,
    )
    return config, checkpoint, dataset, contract, model, info_path


def _write_valid_visual_latch_audit(
    config: FireflyDeploymentConfig,
    path: Path,
    *,
    arm6_source_binding: dict | None = None,
) -> dict:
    dataset_root = Path(
        config.visual_audit_source_root
        if config.policy_runtime_role == "arm6_seeking"
        else config.dataset_root
    )
    dataset_repo_id = (
        config.visual_audit_source_repo_id
        if config.policy_runtime_role == "arm6_seeking"
        else config.dataset_repo_id
    )
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    episode_count = int(info["total_episodes"])
    frame_count = int(info["total_frames"])
    visual_profile = visual_phase_latch_provenance(config.visual_phase_latch)
    implementation_manifest = (
        deployment_module.visual_phase_latch_implementation_manifest()
    )
    training_inputs = deployment_module.capture_training_inputs(
        dataset_root,
        dataset_repo_id,
        controller_contract=Path(config.dataset_controller_contract),
        expected_task=config.task,
    )
    base_frames, remainder = divmod(frame_count, episode_count)
    checks = {
        "no_premature_close_confirmation": True,
        "close_detected_in_window": True,
        "contact_confirmed": True,
        "held_red_survives_loss_timeout": True,
        "no_pretransport_blue_confirmation": True,
        "blue_detected_in_release_window": True,
        "state_machine_ordered_complete": True,
        "state_machine_fault_free": True,
    }
    report = {
        "schema_version": (
            6
            if (
                config.visual_phase_latch.tip_servo.mode is TipServoMode.EXECUTE
                and config.visual_phase_latch.tip_servo.probe_profile
                is TipServoProbeProfile.OPEN_ONLY
            )
            else 3 if config.visual_phase_latch.tip_servo.mode is TipServoMode.EXECUTE else 2
        ),
        "kind": "firefly_y6_fullcycle_v4_visual_latch_replay",
        "read_only": True,
        "safe_for_deployment": True,
        "source_binding": {
            "training_input_digest": training_inputs.digest,
            "visual_profile": visual_profile,
            "visual_implementation_manifest": implementation_manifest,
            "audit_script_sha256": implementation_manifest[
                "offline_visual_audit"
            ]["sha256"],
        },
        "dataset": {
            "root": str(dataset_root),
            "episodes": episode_count,
            "frames": frame_count,
            "fps": int(info["fps"]),
            "wrist_video_key": "observation.images.wrist",
            "recorded_rotation_degrees": 180,
            "offline_rotation_applied": False,
        },
        "profile": {
            "detector_config": visual_profile["fields"],
            "close_confirmation_frames": (
                config.visual_phase_latch.close_confirm_frames
            ),
            "blue_confirmation_frames": config.visual_phase_latch.box_confirm_frames,
        },
        "summary": {
            "verdict": "passed",
            "episode_count": episode_count,
            "error_count": 0,
            "errors": [],
            "runtime_close_before_pickup_crossing_episodes": 0,
            "premature_close_confirmation_episodes": 0,
            "premature_pretransport_blue_confirmation_episodes": 0,
            "close_detected_episodes": episode_count,
            "contact_confirmed_episodes": episode_count,
            "held_red_timeout_safe_episodes": episode_count,
            "release_window_detected_episodes": episode_count,
            "state_machine_ordered_complete_episodes": episode_count,
            "state_machine_fault_free_episodes": episode_count,
            "close_detection_rate": 1.0,
            "held_red_timeout_safe_rate": 1.0,
            "release_window_detection_rate": 1.0,
        },
        "episodes": [
            {
                "episode_index": index,
                "frames": base_frames + (1 if index < remainder else 0),
                "safe": True,
                "checks": checks,
            }
            for index in range(episode_count)
        ],
    }
    if report["schema_version"] == 6:
        if config.policy_runtime_role == "arm6_seeking":
            assert isinstance(arm6_source_binding, dict)
            report["source_binding"].update(arm6_source_binding)
    if (
        config.visual_phase_latch.tip_servo.mode is TipServoMode.EXECUTE
        and config.visual_phase_latch.tip_servo.probe_profile
        is TipServoProbeProfile.OPEN_ONLY
    ):
        authority = deployment_module._schema6_open_only_axis_authority()
        trace_checks = {
            "observational_trace_only": True,
            "axis_authority_namespaced": True,
            "tip_zero_arm_authority": True,
            "tip_zero_explicit_descent_authority": True,
            "jaw_close_not_authorized": True,
            "hardware_completion_not_claimed": True,
        }
        trace = {
            "semantics": (
                "recorded_demo_visual_envelope_axis_authority_not_hardware_response"
            ),
            "axis_authority": authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "counterfactual_open_response_modeled": False,
            "recorded_closed_capture_observed": False,
            "recorded_closed_capture_confirmation_frame": None,
            "checks": trace_checks,
            "trace_envelope_pass": True,
        }
        checks.update(
            {
                "tip_probe_open_only_envelope_pass": True,
                "tip_probe_axis_authority_namespaced": True,
                "tip_probe_policy_arm_only_before_takeover": True,
                "tip_probe_jaw_isolated_from_policy_and_main": True,
                "tip_probe_tip_zero_explicit_descent_authority": True,
                "tip_probe_close_not_authorized": True,
                "tip_probe_hardware_completion_not_claimed": True,
            }
        )
        for episode in report["episodes"]:
            episode["tip_probe_open_only_envelope"] = trace
        report["axis_authority"] = authority
        report["hardware_completion_validated"] = False
        report["closed_loop_convergence_claimed"] = False
        report["profile"]["tip_probe"] = {
            "config": asdict(config.visual_phase_latch.tip_servo),
            "semantics": (
                "policy_arm_before_takeover_tip_capture_hold_then_arm_hold_tip_open"
            ),
            "axis_authority": authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
        negative = {
            "semantics": "historical_act_axis_authority_counterfactual_replay",
            "axis_authority": authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "visual_command_cycles": (
                deployment_module._TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_COMMAND_CYCLES
            ),
            "calibrated_window_start_frame_id": 487,
            "calibrated_window_end_frame_id": 494,
            "calibrated_window_frames": 8,
            "calibrated_window_consecutive": True,
            "calibrated_window_all_fresh_unique": True,
            "calibrated_window_all_closed_capture_gate": True,
            "calibrated_window_action_source_counts": {
                "fresh_policy": 3,
                "inference_hold": 5,
            },
            "first_confirmed_hold_frame_id": 492,
            "would_takeover_boundary_frame_id": 494,
            "would_takeover_boundary_action_source": "fresh_policy",
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
            "actual_tip_servo_arm_command_cycles": 0,
            "actual_tip_servo_descent_command_cycles": 0,
            "actual_tip_servo_jaw_command_cycles": 0,
            "actual_tip_servo_close_command_cycles": 0,
            "historical_out_of_scope": {
                "policy_arm_motion_observed": True,
                "minimum_measured_tcp_delta_z_m": -0.02712412178516388,
                "final_jaw_target_min_rad": 0.28363317251205444,
                "final_jaw_target_max_rad": 0.41,
                "final_jaw_target_below_capture_cycles": 549,
                "final_jaw_target_above_capture_cycles": 344,
            },
            "axis_authority_replay": {
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
            },
            "runtime_authority_replay": {
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
                "first_open_target_rad": 0.32576513671875,
                "policy_role": "arm6_seeking",
                "numeric_policy_observation_dimension": 6,
                "raw_policy_action_dimension": 6,
                "numeric_policy_jaw_input": False,
                "numeric_policy_jaw_output": False,
                "arm6_authority": {
                    "policy_role": "arm6_seeking",
                    "observation_state_dimension": 6,
                    "raw_policy_action_dimension": 6,
                    "candidate_action_dimension": 7,
                    "jaw_candidate_source": "startup_accepted_reference",
                    "capture_jaw_target_rad": 0.30926513671875,
                },
                "arm6_adapter_contract_pass": True,
                "raw_policy_jaw_telemetry_absent": True,
                "policy_role_telemetry_exact": True,
                "jaw_candidate_source_telemetry_exact": True,
                "seven_dimensional_policy_output_rejected": True,
                "synthesized_jaw_tamper_rejected": True,
                "main_latch_jaw_suppressed": True,
                "fault_hold_non_decreasing": True,
                "fault_published_jaw_target_rad": 0.34226513671875003,
                "seeking_timeout_holds_capture": True,
                "zero_explicit_descent": True,
                "ik_never_invoked": True,
                "close_never_authorized": True,
                "runtime_authority_replay_pass": True,
                "limitation": (
                    "Pure software composition replay; it verifies command "
                    "ownership and state transitions, not controller or hardware "
                    "response. Runtime arm_owner is required to remain hold and "
                    "every post-takeover arm target must equal the accepted "
                    "reference exactly."
                ),
            },
            "expected_outcome": (
                "arm6_policy_j1_j6_search_with_external_capture_jaw_hold_then_arm_hold_and_tip_open"
            ),
            "negative_control_pass": True,
            "source": {
                "path": str(
                    (
                        Path(__file__).parents[1]
                        / "outputs"
                        / "firefly_deployment_hold_fullcycle_v4_100k_tip_probe_v1"
                        / "act_30s.jsonl"
                    ).resolve()
                ),
                "size": (
                    deployment_module._TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE
                ),
                "sha256": (
                    deployment_module._TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SHA256
                ),
                "jsonl_records": (
                    deployment_module._TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_RECORDS
                ),
            },
        }
        report["tip_probe"] = {
            "axis_authority": authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "demonstration_episode_count": episode_count,
            "demonstration_envelope_checked_episodes": episode_count,
            "recorded_closed_capture_observed_episodes": 0,
            "configured_max_total_descent_m": 0.0,
            "configured_descent_step_m": 0.0,
            "configured_seeking_timeout_ms": 20_000.0,
            "negative_control_pass": True,
            "negative_control": negative,
        }
        report["summary"]["tip_probe"] = {
            "axis_authority": authority,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "demonstration_envelope_checked_episodes": episode_count,
            "recorded_closed_capture_observed_episodes": 0,
            "negative_control_pass": True,
        }
    elif config.visual_phase_latch.tip_servo.mode is TipServoMode.EXECUTE:
        tip_trace_checks = {
            "takeover_confirmed": True,
            "probe_outcome_confirmed": True,
            "within_recorded_base_z_budget": True,
            "tracking_corridor_continuous": True,
            "area_drop_fault_free": True,
            "allow_close_false": True,
            "close_not_authorized": True,
            "hardware_completion_not_claimed": True,
        }
        tip_trace = {
            "semantics": "recorded_success_path_geometric_envelope_only",
            "allow_close": False,
            "close_authorized": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "settling_and_command_response_modeled": False,
            "takeover_confirmation_frame": 0,
            "probe_progress_confirmation_frame": 0,
            "visual_would_close_confirmation_frame": None,
            "outcome": "probe_progress_complete",
            "outcome_frame": 0,
            "required_recorded_base_z_descent_m": 0.0,
            "required_descent_steps": 0,
            "max_significant_drop_streak_frames": 0,
            "checks": tip_trace_checks,
            "trace_envelope_pass": True,
        }
        checks.update(
            {
                "tip_probe_trace_envelope_pass": True,
                "tip_probe_close_not_authorized": True,
                "tip_probe_hardware_completion_not_claimed": True,
            }
        )
        for episode in report["episodes"]:
            episode["tip_probe_trace_envelope"] = tip_trace
        report["hardware_completion_validated"] = False
        report["closed_loop_convergence_claimed"] = False
        report["profile"]["tip_probe"] = {
            "config": {
                "mode": "execute",
                "allow_close": False,
                "max_total_descent_m": 0.025,
                "descent_step_m": 0.002,
            },
            "semantics": "bounded_no-close_recorded_trace_envelope",
            "allow_close": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
        report["tip_probe"] = {
            "allow_close": False,
            "close_authorized": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
            "demonstration_episode_count": episode_count,
            "takeover_confirmed_episodes": episode_count,
            "probe_outcome_confirmed_episodes": episode_count,
            "trace_envelope_passed_episodes": episode_count,
            "recorded_base_z_budget_passed_episodes": episode_count,
            "configured_max_total_descent_m": 0.025,
            "configured_descent_step_m": 0.002,
            "configured_maximum_descent_steps": 13,
            "negative_control_pass": True,
            "negative_control": {
                "semantics": "old_policy_trace_expected_safe_noncompletion",
                "allow_close": False,
                "close_authorized": False,
                "hardware_completion_validated": False,
                "closed_loop_convergence_claimed": False,
                "visual_command_cycles": 1,
                "takeover_frame_id": 0,
                "probe_progress_confirmation_frame_id": None,
                "visual_would_close_confirmation_frame_id": None,
                "maximum_post_takeover_red_area_fraction": 0.018,
                "expected_outcome": "captured_then_no_depth_progress",
                "negative_control_pass": True,
                "post_takeover_recorded_frames_used_for_convergence": False,
                "source": {
                    "path": "/tmp/official-old-act-negative-control.jsonl",
                    "size": deployment_module._TIP_SERVO_NEGATIVE_CONTROL_SIZE,
                    "sha256": deployment_module._TIP_SERVO_NEGATIVE_CONTROL_SHA256,
                    "jsonl_records": (
                        deployment_module._TIP_SERVO_NEGATIVE_CONTROL_RECORDS
                    ),
                },
            },
        }
        report["summary"]["tip_probe"] = {
            "trace_envelope_passed_episodes": episode_count,
            "recorded_base_z_budget_passed_episodes": episode_count,
            "negative_control_pass": True,
            "allow_close": False,
            "hardware_completion_validated": False,
            "closed_loop_convergence_claimed": False,
        }
    encoded = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    config.visual_latch_audit_path = str(path.absolute())
    config.visual_latch_audit_sha256 = hashlib.sha256(encoded).hexdigest()
    return report


def _arm6_stage_gate_config(tmp_path: Path) -> FireflyDeploymentConfig:
    config, _checkpoint, visual_dataset, _contract, _model, _info = (
        _write_valid_deployment_assets(tmp_path, policy_type="smolvla")
    )
    policy_dataset = tmp_path / "arm-only-policy-dataset"
    return deployment_module.replace(
        config,
        policy_runtime_role="arm6_seeking",
        dataset_repo_id="local/firefly_y6_fullcycle_v4_arm_only_v1",
        dataset_root=str(policy_dataset),
        visual_audit_source_repo_id="local/firefly_y6_fullcycle_v4",
        visual_audit_source_root=str(visual_dataset),
        arm6_run_commit_sha="a" * 64,
        task=deployment_module._FULLCYCLE_V4_TASK,
        act_duration_s=30.0,
        visual_phase_latch=VisualPhaseLatchConfig(
            enabled=True,
            tip_servo=deployment_module._fullcycle_v4_open_only_tip_servo_config(),
        ),
        visual_latch_audit_path=str(tmp_path / "schema6-audit.json"),
        visual_latch_audit_sha256="b" * 64,
    )


def _call_arm6_stage_gate(
    config: FireflyDeploymentConfig,
    stage: str,
    run_label: str,
    **overrides,
) -> None:
    arguments = {
        "mark_reviewed": False,
        "operator_authorized_tip_servo_open_only": False,
        "operator_authorized_tip_servo_execute": False,
        "operator_authorized_skip_replay_for_act_3s": False,
        "operator_authorized_direct_act_30s": False,
        "operator_authorized_direct_act_65s": False,
        "operator_authorized_direct_act_90s": False,
        "operator_authorized_act_start_reset": False,
        "operator_authorized_act_ee_preopen": False,
    }
    arguments.update(overrides)
    require_arm6_policy_stage_gate(config, stage, run_label, **arguments)


@dataclass
class _FakeArm6AttestationCapture:
    selected_checkpoint: Path
    selected_step: int
    binding: dict
    artifacts: list[dict]

    def source_binding(self) -> dict:
        return json.loads(json.dumps(self.binding))

    def controlled_artifacts(self) -> list[dict]:
        return json.loads(json.dumps(self.artifacts))


def _arm6_file_record(role: str, path: Path) -> dict:
    data = path.read_bytes()
    return {
        "role": role,
        "path": str(path.absolute()),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _arm6_reference(record: dict) -> dict:
    return {
        "path": record["path"],
        "size": record["size"],
        "sha256": record["sha256"],
    }


def _write_arm6_deployment_assets(monkeypatch, tmp_path: Path):
    (
        base_config,
        base_checkpoint,
        visual_dataset,
        contract,
        _model,
        _info,
    ) = _write_valid_deployment_assets(tmp_path, policy_type="smolvla")
    policy_dataset = tmp_path / "arm-only-policy-dataset"
    shutil.copytree(visual_dataset, policy_dataset)

    policy_data = policy_dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(policy_data)
    for feature in ("action", "observation.state"):
        values = [row[:6] for row in table[feature].to_pylist()]
        column = pa.array(values, type=pa.list_(pa.float32(), 6))
        index = table.schema.get_field_index(feature)
        table = table.set_column(index, feature, column)
    pq.write_table(table, policy_data)
    policy_info_path = policy_dataset / "meta" / "info.json"
    policy_info = json.loads(policy_info_path.read_text())
    for feature in ("action", "observation.state"):
        policy_info["features"][feature]["shape"] = [6]
        policy_info["features"][feature]["names"] = policy_info["features"][feature][
            "names"
        ][:6]
    policy_info_path.write_text(json.dumps(policy_info))

    for dataset in (visual_dataset, policy_dataset):
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([0], type=pa.int64()),
                    "task": pa.array(
                        [deployment_module._FULLCYCLE_V4_TASK], type=pa.string()
                    ),
                }
            ),
            dataset / "meta" / "tasks.parquet",
        )
        episode_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episode_table = pq.read_table(episode_path)
        task_column = pa.array(
            [[deployment_module._FULLCYCLE_V4_TASK]], type=pa.list_(pa.string())
        )
        episode_table = episode_table.set_column(
            episode_table.schema.get_field_index("tasks"), "tasks", task_column
        )
        pq.write_table(episode_table, episode_path)
    visual_info_path = visual_dataset / "meta" / "info.json"
    visual_info = json.loads(visual_info_path.read_text())
    visual_info["total_episodes"] = deployment_module._FULLCYCLE_V4_TOTAL_EPISODES
    visual_info["total_frames"] = deployment_module._FULLCYCLE_V4_TOTAL_FRAMES
    visual_info_path.write_text(json.dumps(visual_info))

    checkpoints = tmp_path / "arm6-run" / "checkpoints"
    checkpoint_010000 = checkpoints / "010000" / "pretrained_model"
    checkpoint_020000 = checkpoints / "020000" / "pretrained_model"
    shutil.copytree(base_checkpoint, checkpoint_010000)
    shutil.copytree(base_checkpoint, checkpoint_020000)
    for step, checkpoint in ((10_000, checkpoint_010000), (20_000, checkpoint_020000)):
        train_config_path = checkpoint / "train_config.json"
        train_config = json.loads(train_config_path.read_text())
        train_config["dataset"].update(
            {
                "repo_id": "local/firefly_y6_fullcycle_v4_arm_only_v1",
                "root": str(policy_dataset),
            }
        )
        train_config_path.write_text(json.dumps(train_config))
        (checkpoint / "model.safetensors").write_bytes(
            f"offline-arm6-model-{step}".encode()
        )
        (checkpoint / "firefly_child_attestation.json").write_text(
            json.dumps({"step": step, "fixture": True})
        )

    producer_script = tmp_path / "run2-producer.py"
    materializer_script = tmp_path / "arm6-materializer.py"
    projection_manifest = policy_dataset / "meta" / "firefly_arm_only_v1_manifest.json"
    run_commit = tmp_path / "arm6-run" / "firefly_arm6_run_commit.json"
    producer_script.write_text("# fixture producer\n")
    materializer_script.write_text("# fixture materializer\n")
    projection_manifest.write_text('{"fixture":"projection"}\n')
    run_commit.write_text('{"fixture":"run-commit"}\n')
    artifacts = sorted(
        [
            _arm6_file_record(
                "child_attestation_010000",
                checkpoint_010000 / "firefly_child_attestation.json",
            ),
            _arm6_file_record(
                "child_attestation_020000",
                checkpoint_020000 / "firefly_child_attestation.json",
            ),
            _arm6_file_record("materializer_script", materializer_script),
            _arm6_file_record("model_010000", checkpoint_010000 / "model.safetensors"),
            _arm6_file_record("model_020000", checkpoint_020000 / "model.safetensors"),
            _arm6_file_record("producer_script", producer_script),
            _arm6_file_record("projection_manifest", projection_manifest),
            _arm6_file_record("run_commit", run_commit),
        ],
        key=lambda item: (item["role"], item["path"]),
    )
    by_role = {item["role"]: item for item in artifacts}
    input_digest = "c" * 64
    children = []
    for step, directory, checkpoint in (
        (10_000, "010000", checkpoint_010000),
        (20_000, "020000", checkpoint_020000),
    ):
        children.append(
            {
                "step": step,
                "directory": directory,
                "pretrained_model": str(checkpoint),
                "attestation": _arm6_reference(
                    by_role[f"child_attestation_{directory}"]
                ),
                "checkpoint_files_sha256": str(step // 10_000) * 64,
                "model": _arm6_reference(by_role[f"model_{directory}"]),
            }
        )
    binding = {
        "policy_attestation": {
            "schema_version": 1,
            "kind": "firefly_smolvla_arm_only_v1_attestation_capture",
            "policy_schema_role": "arm_only_6d",
            "authorized_runtime_role": "arm6_seeking",
            "limitations": {
                "numeric_state_and_action_exclude_jaw": True,
                "source_video_pixels_may_show_gripper": True,
            },
            "run_id": input_digest,
            "core_input_digest": "b" * 64,
            "input_digest": input_digest,
            "run_commit": {
                **_arm6_reference(by_role["run_commit"]),
                "document_sha256": "d" * 64,
                "children_sha256": "e" * 64,
            },
            "children": children,
            "selected_checkpoint": {
                "step": 20_000,
                "directory": "020000",
                "pretrained_model": str(checkpoint_020000),
                "attestation_sha256": children[1]["attestation"]["sha256"],
                "checkpoint_files_sha256": children[1]["checkpoint_files_sha256"],
                "model_sha256": children[1]["model"]["sha256"],
            },
            "producer_verify_run": {
                "script": _arm6_reference(by_role["producer_script"]),
                "status": "verified-read-only-no-training-started",
                "result_sha256": "f" * 64,
            },
        },
        "projection_binding": {
            "schema_version": 1,
            "kind": "firefly_y6_arm_only_dataset_projection",
            "projection_version": "arm_only_v1",
            "manifest": _arm6_reference(by_role["projection_manifest"]),
            "materializer": _arm6_reference(by_role["materializer_script"]),
            "source": {
                "repo_id": "local/firefly_y6_fullcycle_v4",
                "root": str(visual_dataset),
                "upstream_training_input_digest": "6" * 64,
                "files_manifest_sha256": "1" * 64,
                "directories_sha256": "2" * 64,
                "file_count": 6,
                "directory_count": 10,
            },
            "destination": {
                "repo_id": "local/firefly_y6_fullcycle_v4_arm_only_v1",
                "root": str(policy_dataset),
                "files_manifest_sha256": "3" * 64,
                "directories_sha256": "2" * 64,
                "file_count": 6,
                "directory_count": 10,
            },
            "transform": {"projected_dimensions": [7, 6]},
            "controller_contract": _arm6_reference(
                _arm6_file_record("controller_contract", contract)
            ),
            "verification": {"exact_first_six_float32_bits": True},
            "video_byte_identity": {
                "verified": True,
                "file_count": 2,
                "files_sha256": "5" * 64,
            },
        },
    }
    fake_capture = _FakeArm6AttestationCapture(
        selected_checkpoint=checkpoint_020000,
        selected_step=20_000,
        binding=binding,
        artifacts=artifacts,
    )
    monkeypatch.setattr(
        deployment_module,
        "validate_materialized_smolvlm_bundle",
        lambda path: (Path(path), {"revision": "offline-test"}),
    )
    visual_digest = "6" * 64

    def capture_visual_inputs(root, repo_id, **_kwargs):
        assert Path(root) == visual_dataset
        assert repo_id == "local/firefly_y6_fullcycle_v4"
        return SimpleNamespace(digest=visual_digest)

    monkeypatch.setattr(
        deployment_module,
        "capture_training_inputs",
        capture_visual_inputs,
    )
    config = deployment_module.replace(
        base_config,
        checkpoint=str(checkpoint_020000),
        policy_runtime_role="arm6_seeking",
        dataset_repo_id="local/firefly_y6_fullcycle_v4_arm_only_v1",
        dataset_root=str(policy_dataset),
        visual_audit_source_repo_id="local/firefly_y6_fullcycle_v4",
        visual_audit_source_root=str(visual_dataset),
        arm6_run_commit_sha=by_role["run_commit"]["sha256"],
        task=deployment_module._FULLCYCLE_V4_TASK,
        act_duration_s=30.0,
        visual_phase_latch=VisualPhaseLatchConfig(
            enabled=True,
            tip_servo=deployment_module._fullcycle_v4_open_only_tip_servo_config(),
        ),
        visual_latch_audit_path=str(tmp_path / "arm6-schema6-audit.json"),
        visual_latch_audit_sha256="7" * 64,
    )
    _write_valid_visual_latch_audit(
        config,
        Path(config.visual_latch_audit_path),
        arm6_source_binding=binding,
    )
    return config, fake_capture, policy_dataset, visual_dataset, by_role


def _arm6_audit_fixture(monkeypatch, tmp_path: Path):
    config, consumer, policy_dataset, visual_dataset, artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    audit_path = Path(config.visual_latch_audit_path)
    report = json.loads(audit_path.read_text())
    return (
        config,
        consumer,
        audit_path,
        report,
        policy_dataset,
        visual_dataset,
        artifacts,
    )


def _capture_arm6_fixture(
    config: FireflyDeploymentConfig,
    consumer: _FakeArm6AttestationCapture,
):
    return capture_deployment_sources(
        config,
        arm6_attestation_loader=lambda: consumer,
    )


def test_arm6_config_requires_strict_role_separated_visual_source_and_commit(tmp_path):
    config = _arm6_stage_gate_config(tmp_path)
    assert config.policy_runtime_role == "arm6_seeking"
    assert config.dataset_repo_id.endswith("_arm_only_v1")
    assert config.visual_audit_source_repo_id == "local/firefly_y6_fullcycle_v4"
    assert config.dataset_root != config.visual_audit_source_root
    assert config.arm6_run_commit_sha == "a" * 64

    with pytest.raises(ValueError, match="policy_runtime_role"):
        deployment_module.replace(config, policy_runtime_role=1)
    with pytest.raises(ValueError, match="configured together"):
        deployment_module.replace(config, visual_audit_source_root=None)
    with pytest.raises(ValueError, match="arm6_run_commit_sha"):
        deployment_module.replace(config, arm6_run_commit_sha="not-a-digest")
    with pytest.raises(ValueError, match="must be distinct"):
        deployment_module.replace(
            config,
            dataset_root=config.visual_audit_source_root,
        )
    with pytest.raises(ValueError, match="exactly preserve"):
        deployment_module.replace(config, task="A different task")


def test_arm6_stage_gate_allows_only_shadow_hold_and_one_shot_open_only_act30(
    tmp_path,
):
    config = _arm6_stage_gate_config(tmp_path)
    _call_arm6_stage_gate(config, "shadow", "shadow")
    _call_arm6_stage_gate(config, "hold", "hold")
    _call_arm6_stage_gate(
        config,
        "act",
        "act_30s",
        operator_authorized_tip_servo_open_only=True,
    )

    with pytest.raises(PermissionError, match="forbids dataset replay"):
        _call_arm6_stage_gate(config, "replay", "replay")
    with pytest.raises(PermissionError, match="one non-review 30-second"):
        _call_arm6_stage_gate(
            config,
            "act",
            "act_10s",
            operator_authorized_tip_servo_open_only=True,
        )
    with pytest.raises(PermissionError, match="one non-review 30-second"):
        _call_arm6_stage_gate(config, "act", "act_30s")
    for authorization in (
        "operator_authorized_tip_servo_execute",
        "operator_authorized_skip_replay_for_act_3s",
        "operator_authorized_direct_act_30s",
        "operator_authorized_act_start_reset",
        "operator_authorized_act_ee_preopen",
    ):
        with pytest.raises(PermissionError, match="forbids"):
            _call_arm6_stage_gate(
                config,
                "act",
                "act_30s",
                operator_authorized_tip_servo_open_only=True,
                **{authorization: True},
            )

    with pytest.raises(ValueError, match="cannot allow_close"):
        deployment_module.replace(
            config.visual_phase_latch.tip_servo,
            allow_close=True,
        )
    config.visual_phase_latch = deployment_module.replace(
        config.visual_phase_latch,
        tip_servo=TipServoConfig(
            mode=TipServoMode.EXECUTE,
            allow_close=False,
            max_total_descent_m=0.025,
        ),
    )
    with pytest.raises(PermissionError, match="no-close"):
        _call_arm6_stage_gate(
            config,
            "act",
            "act_30s",
            operator_authorized_tip_servo_open_only=True,
        )


def test_arm6_capture_binds_consumer_policy_projection_visual_source_and_manifest(
    monkeypatch,
    tmp_path,
):
    config, consumer, policy_dataset, visual_dataset, artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    capture = capture_deployment_sources(
        config,
        arm6_attestation_loader=lambda: consumer,
    )

    assert len(capture.deployment_fingerprint) == 64
    assert len(capture.stability_digest) == 64
    assert capture.arm6_attestation == consumer.binding
    assert Path(config.dataset_root) == policy_dataset
    assert Path(config.visual_audit_source_root) == visual_dataset
    assert artifacts["run_commit"]["sha256"] == config.arm6_run_commit_sha
    audit = json.loads(Path(config.visual_latch_audit_path).read_text())
    assert audit["schema_version"] == 6
    assert audit["source_binding"]["policy_attestation"] == consumer.binding[
        "policy_attestation"
    ]
    assert audit["source_binding"]["projection_binding"] == consumer.binding[
        "projection_binding"
    ]
    assert audit["source_binding"]["projection_binding"]["source"] == {
        **consumer.binding["projection_binding"]["source"],
    }
    implementation = audit["source_binding"]["visual_implementation_manifest"]
    assert set(implementation) >= {
        "arm6_action_adapter",
        "arm6_attestation_consumer",
    }
    audit["source_binding"]["projection_binding"]["video_byte_identity"][
        "verified"
    ] = 1
    _rewrite_visual_latch_audit(
        config,
        Path(config.visual_latch_audit_path),
        audit,
    )
    with pytest.raises(PermissionError, match="source_binding"):
        capture_deployment_sources(
            config,
            arm6_attestation_loader=lambda: consumer,
        )


@pytest.mark.parametrize(
    "role",
    [
        "run_commit",
        "child_attestation_010000",
        "child_attestation_020000",
        "model_010000",
        "model_020000",
        "projection_manifest",
    ],
)
def test_arm6_capture_rejects_changed_controlled_artifact(
    monkeypatch,
    tmp_path,
    role,
):
    config, consumer, _policy_dataset, _visual_dataset, artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    Path(artifacts[role]["path"]).write_bytes(b"tampered-after-consumer-capture")

    with pytest.raises(PermissionError, match="changed after consumer verification"):
        capture_deployment_sources(
            config,
            arm6_attestation_loader=lambda: consumer,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda consumer: consumer.binding["policy_attestation"].__setitem__(
                "authorized_runtime_role", "legacy_action7"
            ),
            "authorized_runtime_role",
        ),
        (
            lambda consumer: consumer.binding["projection_binding"]["source"].__setitem__(
                "repo_id", "local/wrong"
            ),
            "source repo_id",
        ),
        (
            lambda consumer: consumer.binding["projection_binding"][
                "video_byte_identity"
            ].__setitem__("verified", False),
            "video byte identity",
        ),
        (
            lambda consumer: consumer.binding["projection_binding"]["source"].__setitem__(
                "upstream_training_input_digest", "0" * 64
            ),
            "upstream digest differs",
        ),
        (
            lambda consumer: consumer.binding["projection_binding"][
                "controller_contract"
            ].__setitem__("sha256", "0" * 64),
            "controller contract differs",
        ),
        (
            lambda consumer: setattr(consumer, "selected_step", 10_000),
            "step 20000",
        ),
    ],
)
def test_arm6_capture_rejects_consumer_binding_drift(
    monkeypatch,
    tmp_path,
    mutation,
    match,
):
    config, consumer, _policy_dataset, _visual_dataset, _artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    mutation(consumer)

    with pytest.raises(PermissionError, match=match):
        capture_deployment_sources(
            config,
            arm6_attestation_loader=lambda: consumer,
        )


def test_arm6_capture_rejects_out_of_band_run_commit_pin(
    monkeypatch,
    tmp_path,
):
    config, consumer, _policy_dataset, _visual_dataset, _artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    config.arm6_run_commit_sha = "0" * 64

    with pytest.raises(PermissionError, match="configured arm6_run_commit_sha"):
        capture_deployment_sources(
            config,
            arm6_attestation_loader=lambda: consumer,
        )


def test_arm6_stability_binds_complete_visual_source_tree(
    monkeypatch,
    tmp_path,
):
    config, consumer, _policy_dataset, visual_dataset, _artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    before = capture_deployment_sources(
        config,
        arm6_attestation_loader=lambda: consumer,
    )
    scene_video = (
        visual_dataset
        / "videos"
        / "observation.images.scene"
        / "chunk-000"
        / "file-000.mp4"
    )
    scene_video.write_bytes(scene_video.read_bytes() + b"-changed")
    after = capture_deployment_sources(
        config,
        arm6_attestation_loader=lambda: consumer,
    )
    assert after.stability_digest != before.stability_digest
    with pytest.raises(PermissionError, match="changed during offline preload"):
        require_unchanged_deployment_sources(before, after)


def test_arm6_capture_rejects_visual_source_change_during_consumer_verification(
    monkeypatch,
    tmp_path,
):
    config, consumer, _policy_dataset, visual_dataset, _artifacts = (
        _write_arm6_deployment_assets(monkeypatch, tmp_path)
    )
    wrist_video = (
        visual_dataset
        / "videos"
        / "observation.images.wrist"
        / "chunk-000"
        / "file-000.mp4"
    )

    def changing_loader():
        wrist_video.write_bytes(wrist_video.read_bytes() + b"-changed-in-consumer")
        return consumer

    with pytest.raises(PermissionError, match="changed during consumer verification"):
        capture_deployment_sources(
            config,
            arm6_attestation_loader=changing_loader,
        )


def _rewrite_visual_latch_audit(
    config: FireflyDeploymentConfig,
    path: Path,
    report: dict,
) -> None:
    encoded = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    config.visual_latch_audit_sha256 = hashlib.sha256(encoded).hexdigest()


def _provenance_event(
    fingerprint: str = DEPLOYMENT_FINGERPRINT,
    *,
    operator_authorized_skip_replay_for_act_3s: bool = False,
    operator_authorized_direct_act_30s: bool = False,
    operator_authorized_tip_servo_probe_act_30s: bool = False,
    operator_authorized_direct_act_65s: bool = False,
    operator_authorized_tip_servo_open_only_act_30s: bool = False,
) -> str:
    return json.dumps(
        {
            "event": "deployment_provenance",
            "deployment_fingerprint": fingerprint,
            "operator_authorized_skip_replay_for_act_3s": (
                operator_authorized_skip_replay_for_act_3s
            ),
            "operator_authorized_direct_act_30s": operator_authorized_direct_act_30s,
            "operator_authorized_tip_servo_probe_act_30s": (
                operator_authorized_tip_servo_probe_act_30s
            ),
            "operator_authorized_direct_act_65s": operator_authorized_direct_act_65s,
            "operator_authorized_tip_servo_open_only_act_30s": (
                operator_authorized_tip_servo_open_only_act_30s
            ),
            "wall_time_ns": 0,
        }
    ) + "\n"


def _write_completed_stage(
    output_dir: Path,
    run_label: str,
    *,
    operator_authorized_skip_replay_for_act_3s: bool = False,
    operator_authorized_direct_act_30s: bool = False,
    operator_authorized_tip_servo_probe_act_30s: bool = False,
    operator_authorized_direct_act_65s: bool = False,
    operator_authorized_tip_servo_open_only_act_30s: bool = False,
) -> None:
    successful_cycle_event = "shadow_cycle" if run_label == "shadow" else "command_cycle"
    (output_dir / f"{run_label}.jsonl").write_text(
        _provenance_event(
            operator_authorized_skip_replay_for_act_3s=(
                operator_authorized_skip_replay_for_act_3s
            ),
            operator_authorized_direct_act_30s=operator_authorized_direct_act_30s,
            operator_authorized_tip_servo_probe_act_30s=(
                operator_authorized_tip_servo_probe_act_30s
            ),
            operator_authorized_direct_act_65s=operator_authorized_direct_act_65s,
            operator_authorized_tip_servo_open_only_act_30s=(
                operator_authorized_tip_servo_open_only_act_30s
            ),
        )
        + json.dumps(
            {
                "event": successful_cycle_event,
                "safety_ok": True,
                "wall_time_ns": 1,
            }
        )
        + "\n"
        + json.dumps(
            {"event": "stage_complete", "stage": run_label, "wall_time_ns": 2}
        )
        + "\n"
    )


def _write_and_review_stage(output_dir: Path, run_label: str) -> None:
    _write_completed_stage(output_dir, run_label)
    mark_stage_reviewed(
        output_dir,
        run_label,
        DEPLOYMENT_FINGERPRINT,
        require_video=False,
    )


class _Payload:
    def __init__(self, data: bytes):
        self._data = data

    def to_bytes(self) -> bytes:
        return self._data


class _Sample:
    def __init__(self, data: bytes = b"", *, kind: str = "PUT"):
        self.payload = _Payload(data)
        self.kind = kind


class _Reply:
    def __init__(self, data: bytes):
        self.ok = _Sample(data)


class _Resource:
    def __init__(self):
        self.closed = False

    def undeclare(self) -> None:
        self.closed = True


class _Token(_Resource):
    def __init__(self, key: str):
        super().__init__()
        self.key = key


class _Liveliness:
    def __init__(self):
        self.tokens: list[_Token] = []
        self.subscribers: dict[str, object] = {}

    def declare_token(self, key: str) -> _Token:
        token = _Token(key)
        self.tokens.append(token)
        return token

    def declare_subscriber(self, key: str, callback, *, history: bool = False) -> _Resource:
        assert history is True
        self.subscribers[key] = callback
        callback(_Sample(kind="PUT"))
        return _Resource()


def _joint_state(values: list[float]) -> bytes:
    return arm_pb2.JointState(q=values).SerializeToString()


def _robot_status(mode: int, holder: int) -> bytes:
    return robot_pb2.RobotStatus(mode=mode, session_holder=holder).SerializeToString()


def _arm_description(
    supported_timeouts: list[int] | None = None,
    tau_max: list[float] | None = None,
) -> bytes:
    return arm_pb2.ArmDescription(
        dof=6,
        joint_names=ARM_NAMES,
        pos_min=[-3.2] * 6,
        pos_max=[3.2] * 6,
        vel_max=[2.0] * 6,
        tau_max=[20.0] * 6 if tau_max is None else tau_max,
        default_kp=[30.0] * 6,
        default_kd=[3.0] * 6,
        supported_timeouts=(
            [arm_pb2.TIMEOUT_BEHAVIOR_RAMP_STOP]
            if supported_timeouts is None
            else supported_timeouts
        ),
    ).SerializeToString()


def _ee_description() -> bytes:
    return ee_pb2.EeDescription(
        dof=1,
        joint_names=EE_NAMES,
        pos_min=[-0.5],
        pos_max=[0.5],
        vel_max=[1.0],
        tau_max=[10.0],
        default_kp=[100.0],
        default_kd=[4.0],
        supported_timeouts=[arm_pb2.TIMEOUT_BEHAVIOR_HOLD],
    ).SerializeToString()


class _FakeSession:
    def __init__(
        self,
        *,
        ee_acquire_ok: bool = True,
        arm_holder: int = 0,
        missing_query: str | None = None,
        arm_timeouts: list[int] | None = None,
        arm_urdf_xml: str = ARM_URDF_XML,
        ee_urdf_xml: str = EE_PAYLOAD_URDF_XML,
        arm_tau_max: list[float] | None = None,
    ):
        self.ee_acquire_ok = ee_acquire_ok
        self.missing_query = missing_query
        self.arm_timeouts = arm_timeouts
        self.arm_urdf_xml = arm_urdf_xml
        self.ee_urdf_xml = ee_urdf_xml
        self.arm_tau_max = arm_tau_max
        self.closed = False
        self.subscribers: dict[str, object] = {}
        self.resources: list[_Resource] = []
        self.liveliness_api = _Liveliness()
        self.get_calls: list[tuple[str, bytes]] = []
        self.put_calls: list[tuple[str, bytes]] = []
        self.operations: list[tuple[str, str]] = []
        self.timeline: list[tuple[str, str]] = []
        self.release_ids: list[int] = []
        self.acquire_requests: list[tuple[str, robot_pb2.AcquireSessionRequest]] = []
        self.holders = {ARM_PREFIX: arm_holder, EE_PREFIX: 0}
        self.initial_payloads = {
            f"{ARM_PREFIX}/arm/joint_state": _joint_state([0.0] * 6),
            f"{EE_PREFIX}/ee/joint_state": _joint_state([0.0]),
            f"{ARM_PREFIX}/status": _robot_status(
                common_pb2.ROBOT_MODE_RUNNING if arm_holder else common_pb2.ROBOT_MODE_STANDBY,
                arm_holder,
            ),
            f"{EE_PREFIX}/status": _robot_status(common_pb2.ROBOT_MODE_STANDBY, 0),
        }

    def declare_subscriber(self, key: str, callback) -> _Resource:
        self.subscribers[key] = callback
        resource = _Resource()
        self.resources.append(resource)
        callback(_Sample(self.initial_payloads[key]))
        return resource

    def liveliness(self) -> _Liveliness:
        return self.liveliness_api

    def _reply(self, message) -> list[_Reply]:
        return [_Reply(message.SerializeToString())]

    def get(self, key: str, *, payload: bytes, timeout: float):
        assert timeout > 0
        self.get_calls.append((key, payload))
        self.timeline.append(("get", key))
        if key == self.missing_query:
            return []
        if key == f"{ARM_PREFIX}/arm/description":
            return [_Reply(_arm_description(self.arm_timeouts, self.arm_tau_max))]
        if key == f"{EE_PREFIX}/ee/description":
            return [_Reply(_ee_description())]
        if key == f"{ARM_PREFIX}/arm/urdf":
            return self._reply(
                arm_pb2.UrdfResource(
                    xml=self.arm_urdf_xml,
                    root_link="base_link",
                    tip_link="link_6",
                )
            )
        if key == f"{EE_PREFIX}/ee/urdf":
            return self._reply(
                arm_pb2.UrdfResource(
                    xml=self.ee_urdf_xml,
                    root_link="base_link",
                    tip_link="tcp",
                )
            )
        if key.endswith("/rpc/acquire_session"):
            prefix = key.removesuffix("/rpc/acquire_session")
            request = robot_pb2.AcquireSessionRequest()
            request.ParseFromString(payload)
            self.acquire_requests.append((prefix, request))
            self.timeline.append(("acquire", prefix))
            if prefix == EE_PREFIX and not self.ee_acquire_ok:
                return self._reply(
                    robot_pb2.AcquireSessionResponse(
                        ok=False,
                        error="busy",
                        current_holder=77,
                        current_holder_name="arm_teleop",
                    )
                )
            session_id = 11 if prefix == ARM_PREFIX else 22
            self.holders[prefix] = session_id
            return self._reply(robot_pb2.AcquireSessionResponse(ok=True, session_id=session_id))
        if key.endswith("/rpc/set_mode"):
            prefix = key.removesuffix("/rpc/set_mode")
            self.operations.append(("set_mode", prefix))
            request = robot_pb2.SetModeRequest()
            request.ParseFromString(payload)
            assert request.mode == common_pb2.OPERATING_MODE_ACTIVE
            self.emit_status(prefix, common_pb2.ROBOT_MODE_RUNNING, request.session_id)
            return self._reply(common_pb2.GenericResponse(ok=True))
        if key.endswith("/rpc/release_session"):
            prefix = key.removesuffix("/rpc/release_session")
            request = robot_pb2.ReleaseSessionRequest()
            request.ParseFromString(payload)
            self.release_ids.append(request.session_id)
            self.holders[prefix] = 0
            return self._reply(common_pb2.GenericResponse(ok=True))
        raise AssertionError(f"unexpected query {key}")

    def put(self, key: str, payload: bytes) -> None:
        self.put_calls.append((key, payload))
        self.operations.append(("put", key))
        self.timeline.append(("put", key))

    def emit_state(self, prefix: str, values: list[float]) -> None:
        suffix = "/arm/joint_state" if prefix == ARM_PREFIX else "/ee/joint_state"
        self.subscribers[prefix + suffix](_Sample(_joint_state(values)))

    def emit_status(self, prefix: str, mode: int, holder: int) -> None:
        self.subscribers[f"{prefix}/status"](_Sample(_robot_status(mode, holder)))

    def close(self) -> None:
        self.closed = True


def _config(mode: str = "execute", **kwargs) -> HexZenohFollowerConfig:
    return HexZenohFollowerConfig(
        id="test-follower",
        arm_prefix=ARM_PREFIX,
        ee_prefix=EE_PREFIX,
        mode=mode,
        connect_timeout_s=0.1,
        rpc_timeout_s=0.1,
        max_state_age_s=0.1,
        max_status_age_s=0.3,
        max_send_gap_s=kwargs.pop("max_send_gap_s", 0.25),
        zenoh_settle_time_s=0,
        **kwargs,
    )


def _connect(monkeypatch, session: _FakeSession, mode: str = "execute", **kwargs) -> HexZenohFollower:
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config(mode, **kwargs))
    robot.connect()
    return robot


def _payload_gravity_kwargs(**overrides):
    values = {
        "enable_arm_gravity_ff": True,
        "expected_arm_urdf_sha256": ARM_URDF_SHA256,
        "enable_ee_payload_gravity_ff": True,
        "expected_ee_urdf_sha256": EE_PAYLOAD_URDF_SHA256,
        "expected_arm_urdf_root_link": "base_link",
        "expected_arm_urdf_tip_link": "link_6",
        "expected_ee_urdf_root_link": "base_link",
        "expected_ee_urdf_tip_link": "tcp",
        "expected_ee_payload_mass_kg": 0.41,
        "expected_ee_payload_com_xyz_m": [
            0.005537804878,
            0.000026829268,
            0.048889970714,
        ],
        "arm_gravity_ff_max_abs_nm": 8.0,
    }
    values.update(overrides)
    return values


def _zero_action(robot: HexZenohFollower) -> dict[str, float]:
    return dict.fromkeys(robot.action_features, 0.0)


class _TimestampedCamera:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_color_frame = np.zeros((2, 3, 3), dtype=np.uint8)
        self.latest_timestamp = time.perf_counter()
        self.is_connected = True
        self.max_age_ms_calls: list[int] = []

    def read_latest(self, max_age_ms: int = 500):
        self.max_age_ms_calls.append(max_age_ms)
        with self.frame_lock:
            return self.latest_color_frame

    def advance(self) -> None:
        with self.frame_lock:
            self.latest_color_frame = np.full(
                (2, 3, 3), len(self.max_age_ms_calls), dtype=np.uint8
            )
            self.latest_timestamp = time.perf_counter()

    def disconnect(self) -> None:
        self.is_connected = False


class _RgbCameraConfig:
    use_rgb = True
    use_depth = False
    height = 2
    width = 3


def test_camera_max_age_has_a_hard_100ms_limit():
    assert _config("shadow").camera_max_age_s == pytest.approx(0.100)
    with pytest.raises(ValueError, match="camera_max_age_s.*100 ms"):
        _config("shadow", camera_max_age_s=0.101)
    with pytest.raises(ValueError, match="camera_max_age_s.*positive"):
        _config("shadow", camera_max_age_s=0.0)


def test_camera_freshness_side_channel_does_not_change_observation_schema(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, "shadow")
    camera = _TimestampedCamera()
    robot.cameras["wrist"] = camera
    robot.config.cameras["wrist"] = _RgbCameraConfig()
    try:
        first_observation = robot.get_observation()
        first = robot.get_camera_frame_metadata()["wrist"]

        assert "wrist" in first_observation
        assert set(first_observation) == set(robot.observation_features)
        assert not any("metadata" in key for key in first_observation)
        assert first.sequence == 1
        assert first.is_new_frame is True
        assert first.timestamp_source == "lerobot_host_perf_counter"
        assert first.age_s is not None and 0 <= first.age_s <= 0.100
        assert camera.max_age_ms_calls == [100]

        robot.get_observation()
        duplicate = robot.get_camera_frame_metadata()["wrist"]
        assert duplicate.sequence == 1
        assert duplicate.is_new_frame is False

        camera.advance()
        robot.get_observation()
        advanced = robot.get_camera_frame_metadata()["wrist"]
        assert advanced.sequence == 2
        assert advanced.is_new_frame is True

        external_copy = robot.get_camera_frame_metadata()
        external_copy.clear()
        assert robot.get_camera_frame_metadata()["wrist"] == advanced
    finally:
        robot.disconnect()


def test_camera_freshness_rejects_stale_or_regressed_host_timestamps(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, "shadow")
    camera = _TimestampedCamera()
    robot.cameras["wrist"] = camera
    robot.config.cameras["wrist"] = _RgbCameraConfig()
    try:
        camera.latest_timestamp = time.perf_counter() - 0.2
        with pytest.raises(TimeoutError, match="latest frame is too old"):
            robot.get_observation()

        camera.latest_timestamp = time.perf_counter()
        robot.get_observation()
        previous_timestamp = camera.latest_timestamp
        camera.latest_color_frame = np.ones((2, 3, 3), dtype=np.uint8)
        camera.latest_timestamp = previous_timestamp - 0.001
        with pytest.raises(RuntimeError, match="camera timestamp regressed"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_tcp_pose_requires_and_uses_hash_validated_live_urdf(monkeypatch):
    disabled = HexZenohFollower(_config("shadow"))
    with pytest.raises(RuntimeError, match="enable_arm_gravity_ff"):
        disabled.get_tcp_pose([0.0] * 6)

    session = _FakeSession()
    robot = _connect(
        monkeypatch,
        session,
        "shadow",
        enable_arm_gravity_ff=True,
        expected_arm_urdf_sha256=ARM_URDF_SHA256,
    )
    try:
        position, rotation = robot.get_tcp_pose([0.0] * 6)
        assert position == pytest.approx((0.0, 0.0, 0.73), abs=1e-6)
        np.testing.assert_allclose(np.asarray(rotation), np.eye(3), atol=1e-6)
        with pytest.raises(ValueError, match="q must contain exactly 6"):
            robot.get_tcp_pose([0.0] * 5)

        robot._arm_gravity_urdf_sha256 = "0" * 64
        with pytest.raises(PermissionError, match="no longer matches"):
            robot.get_tcp_pose([0.0] * 6)
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    ("mode", "cycle_event"),
    (("shadow", "shadow_cycle"), ("execute", "command_cycle")),
)
def test_visual_telemetry_is_nested_in_successful_cycle_event(
    monkeypatch, tmp_path, mode, cycle_event
):
    telemetry_path = tmp_path / f"{mode}.jsonl"
    session = _FakeSession()
    robot = _connect(
        monkeypatch,
        session,
        mode,
        telemetry_path=str(telemetry_path),
    )
    visual = {
        "phase": "closing",
        "red": {"cx": 0.44, "stable": True},
        "votes": [1, None, "d405"],
    }
    try:
        target = robot.command_target or _zero_action(robot)
        robot.send_action_with_metadata(
            target,
            inference_latency_s=0.012,
            visual_telemetry=visual,
        )
        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        cycle = next(event for event in events if event["event"] == cycle_event)
        assert cycle["visual_telemetry"] == visual
        assert cycle["inference_latency_s"] == pytest.approx(0.012)
    finally:
        robot.disconnect()


@dataclass
class _NonJsonVisualEvidence:
    confidence: float


@pytest.mark.parametrize(
    ("visual", "message"),
    (
        ({"mask": np.zeros((2, 2), dtype=np.uint8)}, "ndarray"),
        ({"object": _NonJsonVisualEvidence(0.9)}, "_NonJsonVisualEvidence"),
        ({"confidence": float("nan")}, "NaN or infinite"),
        ({1: "non-string"}, "non-string mapping key"),
        ({"tuple": (1, 2)}, "tuple"),
        (["not", "a", "mapping"], "must be a mapping"),
        ({"too_long": "x" * 513}, "512-character"),
        ({"too_many": list(range(128))}, "128-item"),
    ),
)
def test_invalid_visual_telemetry_fails_before_hardware_publish(
    monkeypatch, visual, message
):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, "execute")
    published_before = len(session.put_calls)
    target = robot.command_target
    assert target is not None
    try:
        with pytest.raises(SafetyStopError, match=message):
            robot.send_action_with_metadata(target, visual_telemetry=visual)
        assert len(session.put_calls) == published_before
        assert session.release_ids[-2:] == [22, 11]
    finally:
        robot.disconnect()


def test_dual_session_success_and_wire_contract(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    try:
        assert robot.is_connected
        assert [prefix for prefix, _request in session.acquire_requests] == [ARM_PREFIX, EE_PREFIX]
        liveliness_keys = [request.liveliness_key for _prefix, request in session.acquire_requests]
        assert all(liveliness_keys)
        assert len(set(liveliness_keys)) == 2
        assert len(session.put_calls) == 2
        assert session.operations == [
            ("set_mode", EE_PREFIX),
            ("set_mode", ARM_PREFIX),
            ("put", f"{ARM_PREFIX}/arm/command"),
            ("put", f"{EE_PREFIX}/ee/command"),
        ]
        arm_command = arm_pb2.JointTrajectory()
        arm_command.ParseFromString(session.put_calls[0][1])
        ee_command = arm_pb2.JointTrajectory()
        ee_command.ParseFromString(session.put_calls[1][1])
        assert arm_command.session_id == 11
        assert list(arm_command.points[0].q) == pytest.approx([0.0] * 6)
        assert list(arm_command.points[0].dq) == pytest.approx([0.0] * 6)
        assert list(arm_command.points[0].kp) == pytest.approx([40.0] * 6)
        assert list(arm_command.points[0].kd) == pytest.approx([2.0] * 6)
        assert list(arm_command.points[0].tau_ff) == pytest.approx([0.0] * 6)
        assert list(arm_command.t_from_start_ns) == [33_333_333]
        assert arm_command.on_timeout == arm_pb2.TIMEOUT_BEHAVIOR_RAMP_STOP
        assert arm_command.header.seq == ee_command.header.seq
        assert arm_command.header.stamp_ns > 0
        assert ee_command.session_id == 22
        assert list(ee_command.points[0].kp) == pytest.approx([150.0])
        assert list(ee_command.points[0].kd) == pytest.approx([2.0])
        assert ee_command.on_timeout == arm_pb2.TIMEOUT_BEHAVIOR_HOLD
        assert list(ee_command.t_from_start_ns) == [33_333_333]
    finally:
        robot.disconnect()
    assert session.release_ids[-2:] == [22, 11]
    assert all(token.closed for token in session.liveliness_api.tokens)


def test_enabled_gravity_ff_binds_live_urdf_and_uses_fresh_q_on_wire(
    monkeypatch, tmp_path
):
    telemetry_path = tmp_path / "gravity.jsonl"
    session = _FakeSession()
    initial_arm = [0.0, -0.4, 0.7, -0.2, 0.1, -0.1]
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(initial_arm)
    robot = _connect(
        monkeypatch,
        session,
        enable_arm_gravity_ff=True,
        expected_arm_urdf_sha256=ARM_URDF_SHA256,
        arm_gravity_ff_max_abs_nm=6.0,
        telemetry_path=str(telemetry_path),
    )
    model = SerialRevoluteGravityModel.from_urdf_xml(ARM_URDF_XML, ARM_NAMES)
    try:
        assert f"{ARM_PREFIX}/arm/urdf" in [key for key, _payload in session.get_calls]
        initial_expected = model.gravity_torque(initial_arm)
        initial_command = arm_pb2.JointTrajectory()
        initial_command.ParseFromString(session.put_calls[0][1])
        assert list(initial_command.points[0].tau_ff) == pytest.approx(
            initial_expected, abs=2e-6
        )
        assert any(abs(value) > 1e-3 for value in initial_expected)

        fresh_arm = [0.0, -0.399, 0.699, -0.199, 0.101, -0.099]
        session.emit_state(ARM_PREFIX, fresh_arm)
        target = robot.command_target
        assert target is not None
        robot.send_action(target)
        fresh_expected = model.gravity_torque(fresh_arm)
        fresh_command = arm_pb2.JointTrajectory()
        fresh_command.ParseFromString(session.put_calls[-2][1])
        assert list(fresh_command.points[0].tau_ff) == pytest.approx(
            fresh_expected, abs=2e-6
        )
        assert fresh_expected != initial_expected

        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        loaded = next(
            event for event in events if event["event"] == "arm_gravity_ff_model_loaded"
        )
        assert loaded["live_urdf_sha256"] == ARM_URDF_SHA256
        assert loaded["command_published"] is False
        assert loaded["session_acquired"] is False
        preflight = next(
            event for event in events if event["event"] == "arm_gravity_ff_preflight"
        )
        assert preflight["reference_q_rad"] == pytest.approx(initial_arm)
        assert preflight["tau_nm"] == pytest.approx(initial_expected, abs=2e-6)
        assert preflight["live_urdf_sha256"] == ARM_URDF_SHA256
        assert preflight["clipped"] is False
        assert preflight["command_published"] is False
        assert preflight["session_acquired"] is False
        contract = next(event for event in events if event["event"] == "controller_contract")
        assert contract["arm_gravity_ff_enabled"] is True
        assert contract["arm_gravity_ff_live_urdf_sha256"] == ARM_URDF_SHA256
        assert contract["arm_gravity_ff_max_abs_nm"] == 6.0
        acquired = next(event for event in events if event["event"] == "control_acquired")
        assert acquired["arm_gravity_ff_reference_q_rad"] == pytest.approx(initial_arm)
        assert acquired["arm_gravity_ff_tau_nm"] == pytest.approx(
            initial_expected, abs=2e-6
        )
        assert acquired["arm_gravity_ff_live_urdf_sha256"] == ARM_URDF_SHA256
        assert acquired["arm_gravity_ff_clipped"] is False
        cycle = next(event for event in events if event["event"] == "command_cycle")
        assert cycle["arm_gravity_ff_reference_q_rad"] == pytest.approx(fresh_arm)
        assert cycle["arm_gravity_ff_tau_nm"] == pytest.approx(
            fresh_expected, abs=2e-6
        )
        assert cycle["arm_gravity_ff_live_urdf_sha256"] == ARM_URDF_SHA256
        assert cycle["arm_gravity_ff_clipped"] is False
    finally:
        robot.disconnect()


def test_payload_gravity_queries_both_urdfs_and_wires_fresh_enhanced_tau(
    monkeypatch, tmp_path
):
    telemetry_path = tmp_path / "payload-gravity.jsonl"
    session = _FakeSession()
    initial_arm = [0.0, -0.4, 0.7, -0.2, 0.1, -0.1]
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(initial_arm)
    robot = _connect(
        monkeypatch,
        session,
        telemetry_path=str(telemetry_path),
        **_payload_gravity_kwargs(),
    )
    payload = LumpedInertial(
        0.41,
        (0.005537804878, 0.000026829268, 0.048889970714),
    )
    model = with_tip_payload(
        SerialRevoluteGravityModel.from_urdf_string(ARM_URDF_XML, ARM_NAMES),
        payload,
    )
    try:
        arm_query = ("get", f"{ARM_PREFIX}/arm/urdf")
        ee_query = ("get", f"{EE_PREFIX}/ee/urdf")
        arm_acquire = ("acquire", ARM_PREFIX)
        assert arm_query in session.timeline
        assert ee_query in session.timeline
        assert session.timeline.index(arm_query) < session.timeline.index(ee_query)
        assert session.timeline.index(ee_query) < session.timeline.index(arm_acquire)

        startup = arm_pb2.JointTrajectory()
        startup.ParseFromString(session.put_calls[0][1])
        assert list(startup.points[0].tau_ff) == pytest.approx(
            model.gravity_torque(initial_arm), abs=2e-6
        )

        fresh_arm = [0.0, -0.399, 0.699, -0.199, 0.101, -0.099]
        session.emit_state(ARM_PREFIX, fresh_arm)
        target = robot.command_target
        assert target is not None
        robot.send_action(target)
        fresh = arm_pb2.JointTrajectory()
        fresh.ParseFromString(session.put_calls[-2][1])
        assert list(fresh.points[0].tau_ff) == pytest.approx(
            model.gravity_torque(fresh_arm), abs=2e-6
        )

        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        loaded = next(
            event for event in events if event["event"] == "arm_gravity_ff_model_loaded"
        )
        assert loaded["live_urdf_sha256"] == ARM_URDF_SHA256
        assert loaded["live_ee_urdf_sha256"] == EE_PAYLOAD_URDF_SHA256
        assert loaded["payload_mass_kg"] == pytest.approx(0.41, abs=1e-6)
        assert loaded["payload_com_xyz_m"] == pytest.approx(
            [0.005537804878, 0.000026829268, 0.048889970714],
            abs=1e-7,
        )
        assert loaded["payload_mount_transform"] == "arm_tip_to_ee_root_identity"
        preflight = next(
            event for event in events if event["event"] == "arm_gravity_ff_preflight"
        )
        assert preflight["live_ee_urdf_sha256"] == EE_PAYLOAD_URDF_SHA256
        cycle = next(event for event in events if event["event"] == "command_cycle")
        assert cycle["ee_payload_gravity_ff_live_urdf_sha256"] == EE_PAYLOAD_URDF_SHA256
        assert cycle["arm_gravity_ff_tau_nm"] == pytest.approx(
            model.gravity_torque(fresh_arm), abs=2e-6
        )
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"expected_ee_urdf_sha256": "0" * 64}, "EE URDF SHA-256"),
        ({"expected_ee_urdf_root_link": "wrong_root"}, "root_link"),
        ({"expected_ee_urdf_tip_link": "wrong_tip"}, "tip_link"),
        ({"expected_ee_payload_mass_kg": 0.42}, "payload mass"),
        (
            {
                "expected_ee_payload_com_xyz_m": [
                    0.005637804878,
                    0.000026829268,
                    0.048889970714,
                ]
            },
            "payload COM",
        ),
    ),
)
def test_payload_binding_mismatch_fails_before_any_acquire_or_publish(
    monkeypatch, override, message
):
    session = _FakeSession()
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config(**_payload_gravity_kwargs(**override)))
    with pytest.raises(PermissionError, match=message):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls
    assert not session.release_ids
    assert not [item for item in session.operations if item[0] == "set_mode"]


def test_payload_enhanced_torque_cap_fails_before_acquire(monkeypatch):
    session = _FakeSession()
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(
        [0.0, -0.4, 0.7, -0.2, 0.1, -0.1]
    )
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(**_payload_gravity_kwargs(arm_gravity_ff_max_abs_nm=0.001))
    )
    with pytest.raises(ValueError, match="exceeds effective"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls
    assert not [item for item in session.operations if item[0] == "set_mode"]


@pytest.mark.parametrize(
    ("attribute", "tampered"),
    (
        ("_ee_payload_urdf_sha256", "0" * 64),
        ("_ee_payload", None),
    ),
)
def test_payload_binding_is_rechecked_before_every_publish(
    monkeypatch, attribute, tampered
):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, **_payload_gravity_kwargs())
    published_before = len(session.put_calls)
    setattr(robot, attribute, tampered)
    target = robot.command_target
    assert target is not None
    with pytest.raises(SafetyStopError, match="EE"):
        robot.send_action(target)
    assert len(session.put_calls) == published_before
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_gravity_ff_urdf_hash_mismatch_fails_before_acquire(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(
            "shadow",
            enable_arm_gravity_ff=True,
            expected_arm_urdf_sha256="0" * 64,
        )
    )
    with pytest.raises(PermissionError, match="does not match"):
        robot.connect()
    assert f"{ARM_PREFIX}/arm/urdf" in [key for key, _payload in session.get_calls]
    assert not session.acquire_requests
    assert not session.put_calls


def test_gravity_ff_rejects_wrong_live_joint_chain_before_acquire(monkeypatch):
    bad_xml = ARM_URDF_XML.replace('name="joint_6"', 'name="wrong_joint_6"')
    session = _FakeSession(arm_urdf_xml=bad_xml)
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(
            "shadow",
            enable_arm_gravity_ff=True,
            expected_arm_urdf_sha256=hashlib.sha256(
                bad_xml.encode("utf-8")
            ).hexdigest(),
        )
    )
    with pytest.raises(ValueError, match="joint order"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls


@pytest.mark.parametrize(
    ("configured_cap", "live_tau_max"),
    (
        (0.01, None),
        (6.0, [0.01] * 6),
    ),
)
def test_gravity_ff_torque_gate_rejects_before_acquire_or_publish(
    monkeypatch, configured_cap, live_tau_max
):
    session = _FakeSession(arm_tau_max=live_tau_max)
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(
        [0.0, -0.4, 0.7, -0.2, 0.1, -0.1]
    )
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(
            enable_arm_gravity_ff=True,
            expected_arm_urdf_sha256=ARM_URDF_SHA256,
            arm_gravity_ff_max_abs_nm=configured_cap,
        )
    )
    with pytest.raises(ValueError, match="exceeds effective"):
        robot.connect()
    assert not session.put_calls
    assert not session.acquire_requests
    assert not session.release_ids
    assert not [operation for operation in session.operations if operation[0] == "set_mode"]


def test_partial_acquire_failure_rolls_back_first_session(monkeypatch):
    session = _FakeSession(ee_acquire_ok=False)
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config())
    with pytest.raises(ConnectionError, match="arm_teleop"):
        robot.connect()
    assert session.release_ids == [11]
    assert not session.put_calls
    assert all(token.closed for token in session.liveliness_api.tokens)


def test_rpc_timeout_fails_without_acquiring(monkeypatch):
    session = _FakeSession(missing_query=f"{ARM_PREFIX}/arm/description")
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config("shadow"))
    with pytest.raises(TimeoutError, match="no successful reply"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls


def test_gravity_ff_config_requires_explicit_enable_hash_and_bounded_cap():
    assert _config("shadow").enable_arm_gravity_ff is False
    with pytest.raises(ValueError, match="requires expected_arm_urdf_sha256"):
        _config("shadow", enable_arm_gravity_ff=True)
    with pytest.raises(ValueError, match="invalid unless"):
        _config("shadow", expected_arm_urdf_sha256="0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _config(
            "shadow",
            enable_arm_gravity_ff=True,
            expected_arm_urdf_sha256="A" * 64,
        )
    with pytest.raises(ValueError, match="hard safety maximum"):
        _config("shadow", arm_gravity_ff_max_abs_nm=6.01)


def test_shadow_records_missing_execute_timeout_without_acquiring(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "shadow.jsonl"
    session = _FakeSession(
        arm_timeouts=[
            arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
            arm_pb2.TIMEOUT_BEHAVIOR_FAULT,
        ]
    )
    robot = _connect(
        monkeypatch,
        session,
        "shadow",
        telemetry_path=str(telemetry_path),
    )
    try:
        assert robot.is_connected
        assert not session.acquire_requests
        assert not session.put_calls
        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        warning = next(event for event in events if event["event"] == "controller_capability_warning")
        assert warning["execute_allowed"] is False
        assert warning["missing_timeouts"][0]["required"] == "TIMEOUT_BEHAVIOR_RAMP_STOP"
        contract = next(event for event in events if event["event"] == "controller_contract")
        assert contract["arm_joint_names"] == ARM_NAMES
        assert contract["arm_pos_min"] == pytest.approx([-3.2] * 6)
        assert contract["arm_pos_max"] == pytest.approx([3.2] * 6)
        assert contract["arm_vel_max"] == pytest.approx([2.0] * 6)
        assert contract["ee_joint_names"] == EE_NAMES
        assert contract["ee_pos_min"] == pytest.approx([-0.5])
        assert contract["ee_pos_max"] == pytest.approx([0.5])
        assert contract["ee_vel_max"] == pytest.approx([1.0])
    finally:
        robot.disconnect()


def test_shadow_accepts_persistent_hold_profile_without_acquiring(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "shadow-hold.jsonl"
    session = _FakeSession(
        arm_timeouts=[
            arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
            arm_pb2.TIMEOUT_BEHAVIOR_FAULT,
        ]
    )
    robot = _connect(
        monkeypatch,
        session,
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        telemetry_path=str(telemetry_path),
    )
    try:
        assert robot.is_connected
        assert not session.acquire_requests
        assert not session.put_calls
        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        assert not [
            event for event in events if event["event"] == "controller_capability_warning"
        ]
        contract = next(event for event in events if event["event"] == "controller_contract")
        assert contract["arm_timeout"] == "TIMEOUT_BEHAVIOR_HOLD"
        assert contract["experimental_hold_override"] is True
    finally:
        robot.disconnect()


def test_execute_rejects_missing_ramp_stop_before_acquiring(monkeypatch):
    session = _FakeSession(
        arm_timeouts=[
            arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
            arm_pb2.TIMEOUT_BEHAVIOR_FAULT,
        ]
    )
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config("execute"))
    with pytest.raises(ValueError, match="TIMEOUT_BEHAVIOR_RAMP_STOP"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls


def test_execute_rejects_missing_hold_before_acquiring(monkeypatch):
    session = _FakeSession(
        arm_timeouts=[
            arm_pb2.TIMEOUT_BEHAVIOR_RAMP_STOP,
            arm_pb2.TIMEOUT_BEHAVIOR_FAULT,
        ]
    )
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(
            "execute",
            arm_timeout_behavior="hold",
            allow_arm_hold_timeout_without_ramp_stop=True,
        )
    )
    with pytest.raises(ValueError, match="TIMEOUT_BEHAVIOR_HOLD"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls


def test_existing_holder_is_rejected_before_acquire(monkeypatch):
    session = _FakeSession(arm_holder=91)
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(_config())
    with pytest.raises(ConnectionError, match="arm_teleop"):
        robot.connect()
    assert not session.acquire_requests
    assert not session.put_calls


def test_liveliness_disappearance_stops_and_releases(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    key, callback = next(iter(session.liveliness_api.subscribers.items()))
    callback(_Sample(kind="DELETE"))
    assert robot.stopped_reason == f"session liveliness disappeared: {key}"
    assert session.release_ids[-2:] == [22, 11]
    assert not robot.is_connected
    robot.disconnect()


def test_runtime_holder_loss_stops_and_releases(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    session.emit_status(ARM_PREFIX, common_pb2.ROBOT_MODE_STANDBY, 0)
    with pytest.raises(SafetyStopError, match="lost arm control"):
        robot.send_action(_zero_action(robot))
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_malformed_status_after_valid_stops_without_publishing(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    initial_puts = len(session.put_calls)
    session.subscribers[f"{ARM_PREFIX}/status"](_Sample(b"\xff"))
    with pytest.raises(SafetyStopError, match=r"invalid follower arm RobotStatus"):
        robot.send_action(_zero_action(robot))
    assert len(session.put_calls) == initial_puts
    assert session.release_ids[-2:] == [22, 11]
    assert robot.stopped_reason is not None
    robot.disconnect()




def test_publication_failure_stops_and_releases(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)

    def fail_put(_key, _payload):
        raise RuntimeError("transport closed")

    session.put = fail_put
    with pytest.raises(SafetyStopError, match="command publication failed"):
        robot.send_action(_zero_action(robot))
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda action: action.pop("joint_1.pos"), "missing"),
        (lambda action: action.__setitem__("joint_1.pos", float("nan")), "NaN"),
        (lambda action: action.__setitem__("joint_1.pos", 4.0), "outside controller range"),
        (lambda action: action.__setitem__("joint_1.pos", 0.1), "step"),
    ],
)
def test_shadow_rejects_invalid_actions_without_publishing(monkeypatch, mutate, match):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, "shadow")
    try:
        action = _zero_action(robot)
        mutate(action)
        with pytest.raises(ValueError, match=match):
            robot.send_action(action)
        assert not session.acquire_requests
        assert not session.put_calls
        assert robot.stopped_reason is None
    finally:
        robot.disconnect()


def test_execute_stale_state_stops_and_releases(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    robot._arm_state.update([0.0] * 6, received_at=time.monotonic() - 1)
    with pytest.raises(SafetyStopError, match="stale follower arm"):
        robot.send_action(_zero_action(robot))
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_execute_measured_position_outside_controller_range_stops(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    session.emit_state(
        ARM_PREFIX,
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    with pytest.raises(SafetyStopError, match="measured position"):
        robot.send_action(_zero_action(robot))
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_tracking_error_for_three_frames_stops(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    action = _zero_action(robot)
    session.emit_state(ARM_PREFIX, [0.2] + [0.0] * 5)
    robot.send_action(action)
    session.emit_state(ARM_PREFIX, [0.2] + [0.0] * 5)
    robot.send_action(action)
    session.emit_state(ARM_PREFIX, [0.2] + [0.0] * 5)
    with pytest.raises(SafetyStopError, match="three consecutive"):
        robot.send_action(action)
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_250ms_command_watchdog_stops_and_releases(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session)
    deadline = time.monotonic() + 0.6
    while robot.stopped_reason is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert robot.stopped_reason is not None
    assert "250 ms" in robot.stopped_reason
    assert session.release_ids[-2:] == [22, 11]
    robot.disconnect()


def test_preflight_uses_controller_limits_and_velocity(monkeypatch):
    session = _FakeSession()
    robot = _connect(monkeypatch, session, "shadow")
    try:
        robot.preflight_actions([[0.0] * 7, [0.01] * 7])
        with pytest.raises(ValueError, match="trajectory frame 1"):
            robot.preflight_actions([[0.0] * 7, [0.1] + [0.0] * 6])
    finally:
        robot.disconnect()


def test_replay_interpolation_and_start_gate():
    result = interpolate_actions([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], speed=0.5)
    assert result == pytest.approx([(0.0, 0.0), (0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0)])
    validate_replay_start([0.0] * 7, [0.04] * 6 + [0.01], arm_dof=6)
    with pytest.raises(SafetyStopError, match="episode start mismatch"):
        validate_replay_start([0.0] * 7, [0.06] + [0.0] * 6, arm_dof=6)


def test_action_slew_rate_adapter_clips_sequentially_and_reports_statistics():
    raw = [
        [0.20, -0.20, 0.04, -0.04, 0.0, 0.0, 0.03],
        [0.04, -0.04, 0.06, -0.06, 0.0, 0.0, 0.005],
        *([[0.04, -0.04, 0.06, -0.06, 0.0, 0.0, 0.005]] * 8),
    ]

    result = adapt_action_chunk_slew_rate(
        raw,
        [0.0] * 7,
        arm_step_limit_rad=0.05,
        ee_step_limit_rad=0.01,
    )

    assert result.adapted_chunk[0] == pytest.approx(
        (0.05, -0.05, 0.04, -0.04, 0.0, 0.0, 0.01)
    )
    assert result.adapted_chunk[1] == pytest.approx(tuple(raw[1]))
    trajectory = [[0.0] * 7, *result.adapted_chunk]
    deltas = abs(np.diff(np.asarray(trajectory), axis=0))
    assert np.all(deltas[:, :6] <= 0.05)
    assert np.all(deltas[:, 6:] <= 0.01)
    assert result.step_limited == (True, *(False,) * 9)
    assert result.limited_value_counts == (3, *(0,) * 9)
    assert result.limited_step_count == 1
    assert result.limited_value_count == 3
    assert result.arm_limited_value_count == 2
    assert result.ee_limited_value_count == 1
    assert result.max_raw_arm_step_rad == pytest.approx(0.20)
    assert result.max_raw_ee_step_rad == pytest.approx(0.03)
    assert result.max_adapted_arm_step_rad == pytest.approx(0.05)
    assert result.max_adapted_ee_step_rad == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("raw", "reference", "match"),
    [
        ([[0.0] * 7] * 9, [0.0] * 7, "shape"),
        ([[0.0] * 6] * 10, [0.0] * 7, "shape"),
        ([[float("nan")] * 7] * 10, [0.0] * 7, "NaN or infinite"),
        ([[0.0] * 7] * 10, [0.0] * 6, "reference.*shape"),
        ([[0.0] * 7] * 10, [0.0] * 6 + [float("inf")], "reference.*NaN or infinite"),
    ],
)
def test_action_slew_rate_adapter_rejects_bad_shape_or_nonfinite(raw, reference, match):
    with pytest.raises(ValueError, match=match):
        adapt_action_chunk_slew_rate(
            raw,
            reference,
            arm_step_limit_rad=0.05,
            ee_step_limit_rad=0.01,
        )


def test_motion_stages_require_independent_estop_confirmation():
    require_hardware_safety_gate("shadow", False)
    require_hardware_safety_gate("hold", False)
    with pytest.raises(PermissionError, match="pc-always-clear"):
        require_hardware_safety_gate("replay", False)
    require_hardware_safety_gate("act", True)


def test_tip_servo_execute_probe_gate_is_exact_act_only_and_fresh():
    disabled = VisualPhaseLatchConfig(enabled=True)
    require_tip_servo_execute_probe_gate(
        "shadow", visual_phase_latch=disabled, operator_authorized=False
    )
    with pytest.raises(PermissionError, match="unless mode=execute"):
        require_tip_servo_execute_probe_gate(
            "act", visual_phase_latch=disabled, operator_authorized=True
        )

    execute = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(
            mode=TipServoMode.EXECUTE,
            allow_close=False,
            max_total_descent_m=0.025,
        ),
    )
    for stage in ("shadow", "hold"):
        require_tip_servo_execute_probe_gate(
            stage, visual_phase_latch=execute, operator_authorized=False
        )
        with pytest.raises(PermissionError, match="only for the ACT stage"):
            require_tip_servo_execute_probe_gate(
                stage, visual_phase_latch=execute, operator_authorized=True
            )
    with pytest.raises(PermissionError, match="fresh"):
        require_tip_servo_execute_probe_gate(
            "act", visual_phase_latch=execute, operator_authorized=False
        )
    require_tip_servo_execute_probe_gate(
        "act", visual_phase_latch=execute, operator_authorized=True
    )

    drifted = deployment_module.replace(
        execute,
        tip_servo=deployment_module.replace(
            execute.tip_servo, max_total_descent_m=0.024
        ),
    )
    with pytest.raises(PermissionError, match="exact reviewed"):
        require_tip_servo_execute_probe_gate(
            "act", visual_phase_latch=drifted, operator_authorized=True
        )

def test_tip_servo_open_only_probe_gate_is_independent_exact_and_fresh():
    visual = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=deployment_module._fullcycle_v4_open_only_tip_servo_config(),
    )
    for stage in ("shadow", "hold"):
        require_tip_servo_open_only_probe_gate(
            stage, visual_phase_latch=visual, operator_authorized=False
        )
        with pytest.raises(PermissionError, match="only for the ACT stage"):
            require_tip_servo_open_only_probe_gate(
                stage, visual_phase_latch=visual, operator_authorized=True
            )
    with pytest.raises(PermissionError, match="fresh"):
        require_tip_servo_open_only_probe_gate(
            "act", visual_phase_latch=visual, operator_authorized=False
        )
    require_tip_servo_open_only_probe_gate(
        "act", visual_phase_latch=visual, operator_authorized=True
    )

    legacy = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(mode=TipServoMode.EXECUTE),
    )
    with pytest.raises(PermissionError, match="exact reviewed"):
        require_tip_servo_open_only_probe_gate(
            "act", visual_phase_latch=legacy, operator_authorized=True
        )
    drifted = deployment_module.replace(
        visual,
        tip_servo=deployment_module.replace(
            visual.tip_servo, max_arm_speed_rad_s=0.069
        ),
    )
    with pytest.raises(PermissionError, match="exact reviewed"):
        require_tip_servo_open_only_probe_gate(
            "act", visual_phase_latch=drifted, operator_authorized=True
        )



def test_hard_safety_constants_cannot_be_relaxed():
    with pytest.raises(ValueError, match="hard safety maximum"):
        _config("shadow", arm_step_limit_rad=0.036)
    with pytest.raises(ValueError, match="fixed at 30"):
        _config("shadow", command_hz=50.0)


def test_acceptance_order_requires_explicit_review(tmp_path):
    require_acceptance_order(tmp_path, "shadow", DEPLOYMENT_FINGERPRINT)
    with pytest.raises(PermissionError, match="shadow"):
        require_acceptance_order(tmp_path, "hold", DEPLOYMENT_FINGERPRINT)
    telemetry = tmp_path / "shadow.jsonl"
    telemetry.write_text(
        _provenance_event()
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )
    ledger = mark_stage_reviewed(
        tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
    )
    assert ledger.is_file()
    entry = json.loads(ledger.read_text())["shadow"]
    assert entry["schema_version"] == 4
    assert entry["deployment_fingerprint"] == DEPLOYMENT_FINGERPRINT
    assert entry["successful_cycle_count"] == 1
    assert entry["safety_rejection_count"] == 0
    require_acceptance_order(tmp_path, "hold", DEPLOYMENT_FINGERPRINT)
    with pytest.raises(PermissionError, match="different checkpoint"):
        require_acceptance_order(tmp_path, "hold", "b" * 64)
    (tmp_path / "hold.jsonl").write_text('{"event":"connected","wall_time_ns":3}\n')
    with pytest.raises(ValueError, match="exactly one stage_complete"):
        mark_stage_reviewed(
            tmp_path, "hold", DEPLOYMENT_FINGERPRINT, require_video=False
        )


def test_operator_authorization_skips_only_a_completely_missing_replay_review(tmp_path):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")

    with pytest.raises(PermissionError, match="replay"):
        require_acceptance_order(tmp_path, "act_3s", DEPLOYMENT_FINGERPRINT)

    require_acceptance_order(
        tmp_path,
        "act_3s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_skip_replay_for_act_3s=True,
    )


@pytest.mark.parametrize(
    ("reviewed_stages", "missing_stage"),
    (((), "shadow"), (("shadow",), "hold")),
)
def test_operator_replay_skip_still_requires_reviewed_shadow_and_hold(
    tmp_path, reviewed_stages, missing_stage
):
    for run_label in reviewed_stages:
        _write_and_review_stage(tmp_path, run_label)

    with pytest.raises(PermissionError, match=missing_stage):
        require_acceptance_order(
            tmp_path,
            "act_3s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_skip_replay_for_act_3s=True,
        )


@pytest.mark.parametrize(
    "run_label", ("shadow", "hold", "replay", "act_10s", "act_30s", "act_65s")
)
def test_operator_replay_skip_is_scoped_only_to_act_3s(tmp_path, run_label):
    with pytest.raises(PermissionError, match="only for the 3-second policy run"):
        require_acceptance_order(
            tmp_path,
            run_label,
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_skip_replay_for_act_3s=True,
        )


def test_operator_replay_skip_does_not_hide_a_damaged_replay_review(tmp_path):
    for run_label in ("shadow", "hold", "replay"):
        _write_and_review_stage(tmp_path, run_label)
    with (tmp_path / "replay.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"event":"post_review_change","wall_time_ns":3}\n')

    with pytest.raises(PermissionError, match="changed after review"):
        require_acceptance_order(
            tmp_path,
            "act_3s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_skip_replay_for_act_3s=True,
        )


def test_operator_replay_skip_act_3s_telemetry_can_never_be_reviewed(tmp_path):
    for run_label in ("shadow", "hold", "replay"):
        _write_and_review_stage(tmp_path, run_label)
    _write_completed_stage(
        tmp_path,
        "act_3s",
        operator_authorized_skip_replay_for_act_3s=True,
    )
    ledger_before = (tmp_path / "acceptance.json").read_bytes()

    with pytest.raises(ValueError, match="cannot be approved"):
        mark_stage_reviewed(
            tmp_path,
            "act_3s",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
        )

    assert (tmp_path / "acceptance.json").read_bytes() == ledger_before
    assert "act_3s" not in json.loads(ledger_before)


def test_operator_direct_act_30s_skips_only_completely_missing_ladder_entries(tmp_path):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")

    with pytest.raises(PermissionError, match="replay"):
        require_acceptance_order(tmp_path, "act_30s", DEPLOYMENT_FINGERPRINT)

    require_acceptance_order(
        tmp_path,
        "act_30s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_direct_act_30s=True,
    )


@pytest.mark.parametrize(
    ("reviewed_stages", "missing_stage"),
    (((), "shadow"), (("shadow",), "hold")),
)
def test_operator_direct_act_30s_still_requires_shadow_and_hold(
    tmp_path, reviewed_stages, missing_stage
):
    for run_label in reviewed_stages:
        _write_and_review_stage(tmp_path, run_label)

    with pytest.raises(PermissionError, match=missing_stage):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_30s=True,
        )


@pytest.mark.parametrize(
    "run_label", ("shadow", "hold", "replay", "act_3s", "act_10s", "act_65s")
)
def test_operator_direct_act_30s_is_scoped_only_to_act_30s(tmp_path, run_label):
    with pytest.raises(PermissionError, match="only for the 30-second policy run"):
        require_acceptance_order(
            tmp_path,
            run_label,
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_30s=True,
        )


@pytest.mark.parametrize("previous", ("replay", "act_3s", "act_10s"))
def test_operator_direct_act_30s_does_not_hide_existing_unreviewed_entry(
    tmp_path, previous
):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")
    ledger_path = tmp_path / "acceptance.json"
    ledger = json.loads(ledger_path.read_text())
    ledger[previous] = {"reviewed": False}
    ledger_path.write_text(json.dumps(ledger))

    with pytest.raises(PermissionError, match=previous):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_30s=True,
        )


def test_operator_direct_act_30s_telemetry_can_never_be_reviewed(tmp_path):
    for run_label in ("shadow", "hold", "replay", "act_3s", "act_10s"):
        _write_and_review_stage(tmp_path, run_label)
    _write_completed_stage(
        tmp_path,
        "act_30s",
        operator_authorized_direct_act_30s=True,
    )
    ledger_before = (tmp_path / "acceptance.json").read_bytes()

    with pytest.raises(ValueError, match="cannot be approved"):
        mark_stage_reviewed(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
        )

    assert (tmp_path / "acceptance.json").read_bytes() == ledger_before
    assert "act_30s" not in json.loads(ledger_before)


def test_tip_servo_probe_act_30s_requires_reviewed_shadow_and_hold(tmp_path):
    with pytest.raises(PermissionError, match="shadow"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_tip_servo_probe_act_30s=True,
        )
    _write_and_review_stage(tmp_path, "shadow")
    with pytest.raises(PermissionError, match="hold"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_tip_servo_probe_act_30s=True,
        )
    _write_and_review_stage(tmp_path, "hold")
    require_acceptance_order(
        tmp_path,
        "act_30s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_tip_servo_probe_act_30s=True,
    )


def test_tip_servo_probe_act_30s_is_distinct_and_never_reviewable(tmp_path):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")
    with pytest.raises(PermissionError, match="cannot reuse"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_30s=True,
            operator_authorized_tip_servo_probe_act_30s=True,
        )

    for run_label in ("replay", "act_3s", "act_10s"):
        _write_and_review_stage(tmp_path, run_label)
    _write_completed_stage(
        tmp_path,
        "act_30s",
        operator_authorized_tip_servo_probe_act_30s=True,
    )
    ledger_before = (tmp_path / "acceptance.json").read_bytes()
    with pytest.raises(ValueError, match="cannot be approved"):
        mark_stage_reviewed(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
        )
    assert (tmp_path / "acceptance.json").read_bytes() == ledger_before


def test_tip_servo_open_only_act_30s_is_independent_and_never_reviewable(
    tmp_path,
):
    with pytest.raises(PermissionError, match="shadow"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_tip_servo_open_only_act_30s=True,
        )
    _write_and_review_stage(tmp_path, "shadow")
    with pytest.raises(PermissionError, match="hold"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_tip_servo_open_only_act_30s=True,
        )
    _write_and_review_stage(tmp_path, "hold")
    require_acceptance_order(
        tmp_path,
        "act_30s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_tip_servo_open_only_act_30s=True,
    )
    with pytest.raises(PermissionError, match="independent"):
        require_acceptance_order(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_tip_servo_probe_act_30s=True,
            operator_authorized_tip_servo_open_only_act_30s=True,
        )

    for run_label in ("replay", "act_3s", "act_10s"):
        _write_and_review_stage(tmp_path, run_label)
    _write_completed_stage(
        tmp_path,
        "act_30s",
        operator_authorized_tip_servo_open_only_act_30s=True,
    )
    with pytest.raises(ValueError, match="cannot be approved"):
        mark_stage_reviewed(
            tmp_path,
            "act_30s",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
        )


def test_operator_direct_act_65s_skips_only_completely_missing_ladder_entries(tmp_path):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")

    with pytest.raises(PermissionError, match="replay"):
        require_acceptance_order(tmp_path, "act_65s", DEPLOYMENT_FINGERPRINT)

    require_acceptance_order(
        tmp_path,
        "act_65s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_direct_act_65s=True,
    )


@pytest.mark.parametrize(
    ("reviewed_stages", "missing_stage"),
    (((), "shadow"), (("shadow",), "hold")),
)
def test_operator_direct_act_65s_still_requires_shadow_and_hold(
    tmp_path, reviewed_stages, missing_stage
):
    for run_label in reviewed_stages:
        _write_and_review_stage(tmp_path, run_label)

    with pytest.raises(PermissionError, match=missing_stage):
        require_acceptance_order(
            tmp_path,
            "act_65s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_65s=True,
        )


@pytest.mark.parametrize(
    "run_label", ("shadow", "hold", "replay", "act_3s", "act_10s", "act_30s")
)
def test_operator_direct_act_65s_is_scoped_only_to_act_65s(tmp_path, run_label):
    with pytest.raises(PermissionError, match="only for the 65-second policy run"):
        require_acceptance_order(
            tmp_path,
            run_label,
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_65s=True,
        )


@pytest.mark.parametrize("previous", ("replay", "act_3s", "act_10s", "act_30s"))
def test_operator_direct_act_65s_does_not_hide_existing_unreviewed_entry(
    tmp_path, previous
):
    _write_and_review_stage(tmp_path, "shadow")
    _write_and_review_stage(tmp_path, "hold")
    ledger_path = tmp_path / "acceptance.json"
    ledger = json.loads(ledger_path.read_text())
    ledger[previous] = {"reviewed": False}
    ledger_path.write_text(json.dumps(ledger))

    with pytest.raises(PermissionError, match=previous):
        require_acceptance_order(
            tmp_path,
            "act_65s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_65s=True,
        )


def test_operator_direct_act_65s_telemetry_can_never_be_reviewed(tmp_path):
    for run_label in ("shadow", "hold", "replay", "act_3s", "act_10s", "act_30s"):
        _write_and_review_stage(tmp_path, run_label)
    _write_completed_stage(
        tmp_path,
        "act_65s",
        operator_authorized_direct_act_65s=True,
    )
    ledger_before = (tmp_path / "acceptance.json").read_bytes()

    with pytest.raises(ValueError, match="cannot be approved"):
        mark_stage_reviewed(
            tmp_path,
            "act_65s",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
        )

    assert (tmp_path / "acceptance.json").read_bytes() == ledger_before
    assert "act_65s" not in json.loads(ledger_before)


def test_acceptance_rejects_completed_shadow_with_safety_rejections(tmp_path):
    (tmp_path / "shadow.jsonl").write_text(
        _provenance_event()
        + '{"event":"safety_rejection","safety_ok":false,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )
    with pytest.raises(ValueError, match="safety_rejection"):
        mark_stage_reviewed(
            tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
        )
    assert not (tmp_path / "acceptance.json").exists()


def test_acceptance_source_recheck_failure_does_not_write_ledger(tmp_path):
    (tmp_path / "shadow.jsonl").write_text(
        _provenance_event()
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )

    def source_changed():
        raise PermissionError("deployment source changed during review")

    with pytest.raises(PermissionError, match="changed during review"):
        mark_stage_reviewed(
            tmp_path,
            "shadow",
            DEPLOYMENT_FINGERPRINT,
            require_video=False,
            verify_before_commit=source_changed,
        )
    assert not (tmp_path / "acceptance.json").exists()


def test_acceptance_rejects_legacy_or_changed_telemetry(tmp_path):
    ledger = tmp_path / "acceptance.json"
    ledger.write_text('{"shadow":{"reviewed":true}}\n')
    with pytest.raises(PermissionError, match="legacy"):
        require_acceptance_order(tmp_path, "hold", DEPLOYMENT_FINGERPRINT)

    (tmp_path / "shadow.jsonl").write_text(
        _provenance_event()
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )
    mark_stage_reviewed(
        tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
    )
    (tmp_path / "shadow.jsonl").write_text(
        _provenance_event()
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":3}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":4}\n'
    )
    with pytest.raises(PermissionError, match="changed after review"):
        require_acceptance_order(tmp_path, "hold", DEPLOYMENT_FINGERPRINT)


def test_acceptance_requires_a_successful_cycle_and_matching_stage(tmp_path):
    telemetry = tmp_path / "shadow.jsonl"
    telemetry.write_text(
        _provenance_event()
        + '{"event":"stage_complete","stage":"shadow","wall_time_ns":1}\n'
    )
    with pytest.raises(ValueError, match="no successful shadow_cycle"):
        mark_stage_reviewed(
            tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
        )
    telemetry.write_text(
        _provenance_event()
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"hold","wall_time_ns":2}\n'
    )
    with pytest.raises(ValueError, match="names 'hold'"):
        mark_stage_reviewed(
            tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
        )


def test_acceptance_rejects_missing_or_different_deployment_provenance(tmp_path):
    telemetry = tmp_path / "shadow.jsonl"
    telemetry.write_text(
        '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )
    with pytest.raises(ValueError, match="deployment_provenance"):
        mark_stage_reviewed(
            tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
        )
    telemetry.write_text(
        _provenance_event("b" * 64)
        + '{"event":"shadow_cycle","safety_ok":true,"wall_time_ns":1}\n'
        '{"event":"stage_complete","stage":"shadow","wall_time_ns":2}\n'
    )
    with pytest.raises(ValueError, match="different deployment fingerprint"):
        mark_stage_reviewed(
            tmp_path, "shadow", DEPLOYMENT_FINGERPRINT, require_video=False
        )


def test_deployment_fingerprint_binds_checkpoint_dataset_task_and_robot(tmp_path):
    config, _checkpoint, _dataset, contract, model, info_path = (
        _write_valid_deployment_assets(tmp_path)
    )
    baseline = compute_deployment_fingerprint(config)

    model.write_bytes(b"model-v2")
    assert compute_deployment_fingerprint(config) != baseline
    model.write_bytes(b"model-v1")
    info = json.loads(info_path.read_text())
    info["fingerprint_test_note"] = "safe metadata change"
    info_path.write_text(json.dumps(info))
    assert compute_deployment_fingerprint(config) != baseline
    info.pop("fingerprint_test_note")
    info_path.write_text(json.dumps(info))
    config.task = "Move the blue block."
    with pytest.raises(PermissionError, match="expected_task_mismatch"):
        compute_deployment_fingerprint(config)
    config.task = DEPLOYMENT_TASK
    contract_value = json.loads(contract.read_text())
    contract_value["source"] = "different saved live contract provenance"
    contract.write_text(json.dumps(contract_value))
    assert compute_deployment_fingerprint(config) != baseline
    contract_value["source"] = "saved live test contract"
    contract.write_text(json.dumps(contract_value))
    config.robot.arm_kp = 39.0
    assert compute_deployment_fingerprint(config) != baseline
    config.robot = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_step_limit_rad=0.005,
        ee_step_limit_rad=0.002,
        arm_tracking_error_rad=0.050,
        ee_tracking_error_rad=0.020,
        arm_kp=20.0,
        arm_kd=2.0,
        ee_kp=75.0,
        ee_kd_default_scale=0.5,
    )
    assert compute_deployment_fingerprint(config) != baseline


def test_deployment_fingerprint_binds_enabled_visual_latch_and_preserves_disabled_default(
    tmp_path,
):
    config, *_assets = _write_valid_deployment_assets(tmp_path)
    disabled_fingerprint = compute_deployment_fingerprint(config)
    assert config.visual_phase_latch == VisualPhaseLatchConfig()

    enabled = VisualPhaseLatchConfig(enabled=True)
    with pytest.raises(ValueError, match="offline audit path"):
        deployment_module.replace(config, visual_phase_latch=enabled)
    config.visual_phase_latch = enabled
    audit_path = tmp_path / "visual-latch-audit.json"
    _write_valid_visual_latch_audit(config, audit_path)
    enabled_capture = capture_deployment_sources(config)
    enabled_fingerprint = enabled_capture.deployment_fingerprint
    assert enabled_fingerprint != disabled_fingerprint
    assert enabled_capture.visual_latch_audit == {
        "configured_path": str(audit_path.absolute()),
        "canonical_path": str(audit_path.absolute()),
        "size": audit_path.stat().st_size,
        "sha256": config.visual_latch_audit_sha256,
        "schema_version": 2,
        "kind": "firefly_y6_fullcycle_v4_visual_latch_replay",
    }
    provenance = visual_phase_latch_provenance(enabled)
    assert provenance["fields"]["enabled"] is True
    assert provenance["fields"]["d405_serial"] == "323622270296"

    config.visual_phase_latch = deployment_module.replace(
        enabled, blue_global_fraction_min=0.69
    )
    with pytest.raises(PermissionError, match="detector_config"):
        compute_deployment_fingerprint(config)
    _write_valid_visual_latch_audit(config, audit_path)
    assert compute_deployment_fingerprint(config) != enabled_fingerprint

def test_tip_servo_execute_requires_schema3_and_strict_probe_contract(tmp_path):
    config, *_assets = _write_valid_deployment_assets(tmp_path)
    config.visual_phase_latch = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(
            mode=TipServoMode.EXECUTE,
            allow_close=False,
            max_total_descent_m=0.025,
        ),
    )
    audit_path = tmp_path / "tip-probe-audit.json"
    report = _write_valid_visual_latch_audit(config, audit_path)
    capture = capture_deployment_sources(config)
    assert capture.visual_latch_audit["schema_version"] == 3

    report["schema_version"] = 2
    _rewrite_visual_latch_audit(config, audit_path, report)
    with pytest.raises(PermissionError, match="schema_version"):
        capture_deployment_sources(config)

    report["schema_version"] = 3
    report["tip_probe"]["configured_max_total_descent_m"] = 0.026
    _rewrite_visual_latch_audit(config, audit_path, report)
    with pytest.raises(PermissionError, match="configured_max_total_descent_m"):
        capture_deployment_sources(config)


def test_tip_servo_open_only_requires_schema6_axis_authority_contract(
    monkeypatch,
    tmp_path,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    capture = _capture_arm6_fixture(config, consumer)
    assert capture.visual_latch_audit["schema_version"] == 6

    report["schema_version"] = 5
    _rewrite_visual_latch_audit(config, audit_path, report)
    with pytest.raises(PermissionError, match="schema_version"):
        _capture_arm6_fixture(config, consumer)

    report["schema_version"] = 6
    report["axis_authority"]["policy_jaw_authorized"] = True
    _rewrite_visual_latch_audit(config, audit_path, report)
    with pytest.raises(PermissionError, match="axis_authority"):
        _capture_arm6_fixture(config, consumer)

    report["axis_authority"]["policy_jaw_authorized"] = 0
    _rewrite_visual_latch_audit(config, audit_path, report)
    with pytest.raises(PermissionError, match="axis_authority"):
        _capture_arm6_fixture(config, consumer)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda report: report.__setitem__(
                "arm_motion_authorized", False
            ),
            "ambiguous unscoped",
        ),
        (
            lambda report: report["tip_probe"]["negative_control"][
                "axis_authority_replay"
            ].__setitem__(
                "published_jaw_min_rad", 0.308
            ),
            "published_jaw_min_rad",
        ),
        (
            lambda report: report["tip_probe"]["negative_control"][
                "axis_authority_replay"
            ]["opened_targets_rad"].__setitem__(
                1, 0.30
            ),
            "opened_targets_rad.*pinned v1 source value",
        ),
        (
            lambda report: report["tip_probe"]["negative_control"][
                "axis_authority_replay"
            ].__setitem__(
                "seeking_timeout_ms", 30_000.0
            ),
            "seeking_timeout_ms",
        ),
    ],
)
def test_tip_servo_open_only_schema6_rejects_scope_and_jaw_guard_tampering(
    monkeypatch,
    tmp_path,
    mutation,
    match,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    mutation(report)
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(PermissionError, match=match):
        _capture_arm6_fixture(config, consumer)


def test_tip_servo_open_only_schema6_rejects_missing_pinned_source(
    tmp_path,
    monkeypatch,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    missing = (tmp_path / "missing-pinned-v1.jsonl").resolve()
    monkeypatch.setattr(
        deployment_module,
        "_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_PATH",
        missing,
    )
    report["tip_probe"]["negative_control"]["source"]["path"] = str(missing)
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(
        PermissionError,
        match="cannot safely read pinned real v1 source.*does not exist",
    ):
        _capture_arm6_fixture(config, consumer)


def test_tip_servo_open_only_schema6_rehashes_pinned_source_bytes(
    tmp_path,
    monkeypatch,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    counterfeit = (tmp_path / "counterfeit-pinned-v1.jsonl").resolve()
    counterfeit.write_bytes(
        b"x" * deployment_module._TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_SIZE
    )
    monkeypatch.setattr(
        deployment_module,
        "_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL_PATH",
        counterfeit,
    )
    report["tip_probe"]["negative_control"]["source"]["path"] = str(
        counterfeit
    )
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(PermissionError, match="source sha256 changed"):
        _capture_arm6_fixture(config, consumer)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda negative: negative["observed_x_norm"].__setitem__(
                "min", 0.5240
            ),
            "observed_x_norm.min.*pinned v1 source value",
        ),
        (
            lambda negative: negative["historical_out_of_scope"].__setitem__(
                "minimum_measured_tcp_delta_z_m", -0.026
            ),
            "minimum_measured_tcp_delta_z_m.*pinned v1 source value",
        ),
        (
            lambda negative: negative["axis_authority_replay"].__setitem__(
                "replayed_seeking_cycles", 492
            ),
            "replayed_seeking_cycles.*pinned v1 source value",
        ),
    ],
)
def test_tip_servo_open_only_schema6_rejects_forged_source_statistics(
    monkeypatch,
    tmp_path,
    mutation,
    match,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    mutation(report["tip_probe"]["negative_control"])
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(PermissionError, match=match):
        _capture_arm6_fixture(config, consumer)


def test_tip_servo_open_only_schema6_requires_runtime_authority_replay(
    monkeypatch,
    tmp_path,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    report["tip_probe"]["negative_control"].pop(
        "runtime_authority_replay"
    )
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(
        PermissionError,
        match="runtime_authority_replay must be an object",
    ):
        _capture_arm6_fixture(config, consumer)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda runtime: runtime["boundary_events"].__setitem__(
                7, "coarse_candidate"
            ),
            "runtime_authority_replay.boundary_events",
        ),
        (
            lambda runtime: runtime.__setitem__(
                "post_takeover_arm_owner_values", ["hold", "policy"]
            ),
            "post_takeover_arm_owner_values",
        ),
        (
            lambda runtime: runtime.__setitem__(
                "first_open_target_rad",
                runtime["first_open_target_rad"] + 0.0001,
            ),
            "first_open_target_rad",
        ),
        (
            lambda runtime: runtime.__setitem__(
                "fault_published_jaw_target_rad",
                runtime["fault_published_jaw_target_rad"] - 0.0001,
            ),
            "fault_published_jaw_target_rad",
        ),
        (
            lambda runtime: runtime.__setitem__(
                "tampered_final_arm_rejected", False
            ),
            "tampered_final_arm_rejected",
        ),
        (
            lambda runtime: runtime.__setitem__(
                "runtime_authority_replay_pass", False
            ),
            "runtime_authority_replay_pass",
        ),
        (
            lambda runtime: runtime.__setitem__("unexpected_field", True),
            "exactly the deterministic schema-6 runtime replay fields",
        ),
    ],
)
def test_tip_servo_open_only_schema6_rejects_runtime_replay_tampering(
    monkeypatch,
    tmp_path,
    mutation,
    match,
):
    config, consumer, audit_path, report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    mutation(
        report["tip_probe"]["negative_control"][
            "runtime_authority_replay"
        ]
    )
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(PermissionError, match=match):
        _capture_arm6_fixture(config, consumer)


def test_tip_servo_open_only_rejects_retained_real_schema4_artifact(
    monkeypatch,
    tmp_path,
):
    config, consumer, _audit_path, _report, *_assets = _arm6_audit_fixture(
        monkeypatch,
        tmp_path,
    )
    schema4_path = (
        Path(__file__).parents[1]
        / "outputs"
        / "firefly_diagnostics"
        / "firefly_y6_tip_servo_open_only_fullcycle_v4_schema4_exact.json"
    ).resolve()
    schema4_bytes = schema4_path.read_bytes()
    config.visual_latch_audit_path = str(schema4_path)
    config.visual_latch_audit_sha256 = hashlib.sha256(schema4_bytes).hexdigest()

    with pytest.raises(
        PermissionError,
        match="schema_version must be exactly 6, found 4",
    ):
        _capture_arm6_fixture(config, consumer)



@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda report: report.__setitem__("safe_for_deployment", False),
            "safe_for_deployment",
        ),
        (
            lambda report: report["summary"].__setitem__("error_count", 1),
            "summary.error_count",
        ),
        (
            lambda report: report["dataset"].__setitem__("frames", 4),
            "dataset.frames",
        ),
        (
            lambda report: report["source_binding"].__setitem__(
                "training_input_digest", "0" * 64
            ),
            "source_binding",
        ),
        (
            lambda report: report["episodes"][0]["checks"].__setitem__(
                "state_machine_fault_free", False
            ),
            "checks",
        ),
    ],
)
def test_visual_latch_audit_semantic_drift_is_rejected(
    tmp_path, mutation, match
):
    config, *_assets = _write_valid_deployment_assets(tmp_path)
    config.visual_phase_latch = VisualPhaseLatchConfig(enabled=True)
    audit_path = tmp_path / "visual-latch-audit.json"
    report = _write_valid_visual_latch_audit(config, audit_path)
    mutation(report)
    _rewrite_visual_latch_audit(config, audit_path, report)

    with pytest.raises(PermissionError, match=match):
        capture_deployment_sources(config)


def test_visual_latch_audit_digest_and_symlink_are_rejected(tmp_path):
    config, *_assets = _write_valid_deployment_assets(tmp_path)
    config.visual_phase_latch = VisualPhaseLatchConfig(enabled=True)
    audit_path = tmp_path / "visual-latch-audit.json"
    _write_valid_visual_latch_audit(config, audit_path)
    configured_digest = config.visual_latch_audit_sha256

    audit_path.write_bytes(audit_path.read_bytes() + b" ")
    with pytest.raises(PermissionError, match="configured SHA-256"):
        capture_deployment_sources(config)

    _write_valid_visual_latch_audit(config, audit_path)
    config.visual_latch_audit_path = "relative-audit.json"
    with pytest.raises(ValueError, match="absolute path"):
        capture_deployment_sources(config)

    alias = tmp_path / "visual-latch-audit-link.json"
    alias.symlink_to(audit_path)
    config.visual_latch_audit_path = str(alias)
    with pytest.raises(PermissionError, match="symbolic link"):
        capture_deployment_sources(config)
    assert configured_digest is not None


@pytest.mark.parametrize("policy_type", ["act", "smolvla"])
def test_checkpoint_provenance_accepts_only_matching_dataset_and_policy(
    monkeypatch, tmp_path, policy_type
):
    config, checkpoint, _dataset, _contract, _model, _info = (
        _write_valid_deployment_assets(tmp_path, policy_type=policy_type)
    )
    if policy_type == "smolvla":
        monkeypatch.setattr(
            deployment_module,
            "validate_materialized_smolvlm_bundle",
            lambda path: (Path(path), {"revision": "test"}),
        )
        monkeypatch.setattr(
            deployment_module,
            "verify_training_attestation",
            lambda *_args: None,
        )
    assert len(compute_deployment_fingerprint(config)) == 64

    train_path = checkpoint / "train_config.json"
    train = json.loads(train_path.read_text())
    train["dataset"]["repo_id"] = "local/wrong"
    train_path.write_text(json.dumps(train))
    with pytest.raises(PermissionError, match="trained on dataset repo"):
        compute_deployment_fingerprint(config)


@pytest.mark.parametrize(
    ("dataset_change", "match"),
    [
        ("missing_episodes", "complete dataset"),
        ("episode_subset", "complete dataset"),
        ("streaming", "dataset.streaming must be false"),
    ],
)
def test_checkpoint_provenance_rejects_partial_or_streaming_dataset(
    tmp_path, dataset_change, match
):
    config, checkpoint, _dataset, _contract, _model, _info = (
        _write_valid_deployment_assets(tmp_path)
    )
    train_path = checkpoint / "train_config.json"
    train = json.loads(train_path.read_text())
    if dataset_change == "missing_episodes":
        train["dataset"].pop("episodes")
    elif dataset_change == "episode_subset":
        train["dataset"]["episodes"] = [0]
    else:
        train["dataset"]["streaming"] = True
    train_path.write_text(json.dumps(train))

    with pytest.raises(PermissionError, match=match):
        compute_deployment_fingerprint(config)


def test_checkpoint_ancestor_symlink_and_nonempty_rename_map_are_rejected(tmp_path):
    config, checkpoint, _dataset, _contract, _model, _info = (
        _write_valid_deployment_assets(tmp_path)
    )
    alias_parent = tmp_path / "last"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    config.checkpoint = str(alias_parent / checkpoint.name)
    with pytest.raises(PermissionError, match="symbolic link component"):
        compute_deployment_fingerprint(config)

    config.checkpoint = str(checkpoint)
    train_path = checkpoint / "train_config.json"
    train = json.loads(train_path.read_text())
    train["rename_map"] = {"observation.images.scene": "observation.images.wrist"}
    train_path.write_text(json.dumps(train))
    with pytest.raises(PermissionError, match="rename_map must be empty"):
        compute_deployment_fingerprint(config)


def test_source_guard_detects_same_content_file_replacement(tmp_path):
    config, _checkpoint, _dataset, _contract, model, _info = (
        _write_valid_deployment_assets(tmp_path)
    )
    before = capture_deployment_sources(config)
    replacement = model.with_suffix(".replacement")
    replacement.write_bytes(model.read_bytes())
    replacement.replace(model)
    after = capture_deployment_sources(config)

    assert after.deployment_fingerprint == before.deployment_fingerprint
    assert after.stability_digest != before.stability_digest
    with pytest.raises(PermissionError, match="changed during offline preload"):
        require_unchanged_deployment_sources(before, after)


def test_source_guard_detects_ancestor_rename_and_restore(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config, _checkpoint, _dataset, _contract, _model, _info = (
        _write_valid_deployment_assets(bundle)
    )
    before = capture_deployment_sources(config)

    parked_bundle = tmp_path / "bundle.parked"
    time.sleep(0.01)
    bundle.rename(parked_bundle)
    bundle.mkdir()
    bundle.rmdir()
    parked_bundle.rename(bundle)

    after = capture_deployment_sources(config)
    assert after.deployment_fingerprint == before.deployment_fingerprint
    assert after.stability_digest != before.stability_digest
    with pytest.raises(PermissionError, match="changed during offline preload"):
        require_unchanged_deployment_sources(before, after)


def test_rerun_invalidates_stage_and_downstream_reviews(tmp_path):
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "shadow": {"reviewed": True},
                "hold": {"reviewed": True},
                "replay": {"reviewed": True},
            }
        )
    )
    invalidate_stage_reviews(tmp_path, "hold")
    ledger = json.loads((tmp_path / "acceptance.json").read_text())
    assert set(ledger) == {"shadow"}


def test_inference_worker_returns_chunk_and_error():
    worker = InferenceWorker(lambda observation: [[observation["value"]]])
    try:
        generation = worker.submit({"value": 2.0})
        result = worker.get(timeout_s=0.2)
        assert result.generation == generation
        assert result.actions == ((2.0,),)
        assert result.error is None
    finally:
        worker.close()


def test_deployment_example_deserializes():
    path = Path(__file__).parents[1] / "configs" / "deploy_firefly_y6.example.yaml"
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)
    assert isinstance(config.robot, HexZenohFollowerConfig)
    assert config.robot.mode == "shadow"
    assert config.n_action_steps == 10
    assert config.enable_action_slew_rate_adapter is False
    assert config.replay_episode == 0
    assert config.replay_speed == 0.5
    assert config.act_duration_s == 3.0
    assert config.checkpoint == "CHANGE_ME_VALIDATED_V2_CHECKPOINT"
    assert config.dataset_repo_id == "local/firefly_y6_manual_validated_v2"
    assert config.robot.cameras["wrist"].rotation == 180
    assert config.visual_phase_latch.enabled is False


def test_hold_deployment_example_deserializes():
    path = Path(__file__).parents[1] / "configs" / "deploy_firefly_y6_hold.example.yaml"
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)
    assert config.robot.mode == "shadow"
    assert config.robot.arm_timeout_behavior == "hold"
    assert config.robot.allow_arm_hold_timeout_without_ramp_stop is True
    assert config.robot.arm_kp == 40.0
    assert config.robot.ee_kp == 75.0
    assert config.robot.arm_step_limit_rad == 0.005
    assert config.robot.ee_step_limit_rad == 0.002
    assert config.enable_action_slew_rate_adapter is True
    assert config.act_chunk_inference_mode == "pipelined"
    assert config.output_dir == "outputs/firefly_deployment_hold_validated_v2"
    assert config.visual_phase_latch.enabled is False


def test_persistent_arm_hold_profile_requires_fresh_execute_confirmations():
    require_arm_hold_timeout_override_gate(
        "shadow",
        hold_profile_selected=True,
        risk_acknowledged=False,
        independent_estop_verified=False,
        hold_duration_s=1.0,
    )
    with pytest.raises(PermissionError, match="execute-only"):
        require_arm_hold_timeout_override_gate(
            "shadow",
            hold_profile_selected=True,
            risk_acknowledged=True,
            independent_estop_verified=True,
            hold_duration_s=1.0,
        )
    for stage in ("hold", "replay", "act"):
        with pytest.raises(PermissionError, match="fresh"):
            require_arm_hold_timeout_override_gate(
                stage,
                hold_profile_selected=True,
                risk_acknowledged=False,
                independent_estop_verified=True,
                hold_duration_s=1.0,
            )
        with pytest.raises(PermissionError, match="physical power cut"):
            require_arm_hold_timeout_override_gate(
                stage,
                hold_profile_selected=True,
                risk_acknowledged=True,
                independent_estop_verified=False,
                hold_duration_s=1.0,
            )
        require_arm_hold_timeout_override_gate(
            stage,
            hold_profile_selected=True,
            risk_acknowledged=True,
            independent_estop_verified=True,
            hold_duration_s=1.0,
        )
    with pytest.raises(PermissionError, match="1.0 second"):
        require_arm_hold_timeout_override_gate(
            "hold",
            hold_profile_selected=True,
            risk_acknowledged=True,
            independent_estop_verified=True,
            hold_duration_s=1.01,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        require_arm_hold_timeout_override_gate(
            "hold",
            hold_profile_selected=True,
            risk_acknowledged=True,
            independent_estop_verified=True,
            hold_duration_s=float("nan"),
        )
    with pytest.raises(PermissionError, match="invalid for a RAMP_STOP"):
        require_arm_hold_timeout_override_gate(
            "replay",
            hold_profile_selected=False,
            risk_acknowledged=True,
            independent_estop_verified=True,
            hold_duration_s=1.0,
        )
    require_arm_hold_timeout_override_gate(
        "replay",
        hold_profile_selected=False,
        risk_acknowledged=False,
        independent_estop_verified=True,
        hold_duration_s=1.0,
    )


def test_arm_hold_timeout_requires_explicit_paired_opt_in():
    with pytest.raises(ValueError, match="requires explicit"):
        _config("execute", arm_timeout_behavior="hold")
    with pytest.raises(ValueError, match="invalid unless"):
        _config(
            "shadow",
            arm_timeout_behavior="ramp_stop",
            allow_arm_hold_timeout_without_ramp_stop=True,
        )
    persisted = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
    )
    assert persisted.mode == "shadow"
    assert persisted.arm_timeout_behavior == "hold"
    assert persisted.allow_arm_hold_timeout_without_ramp_stop is True


def test_hold_override_uses_low_gain_hold_wire_and_frozen_start_target(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "hold.jsonl"
    session = _FakeSession(
        arm_timeouts=[
            arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
            arm_pb2.TIMEOUT_BEHAVIOR_FAULT,
        ]
    )
    initial_arm = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    initial_ee = [0.2]
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(initial_arm)
    session.initial_payloads[f"{EE_PREFIX}/ee/joint_state"] = _joint_state(initial_ee)
    robot = _connect(
        monkeypatch,
        session,
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_kp=20.0,
        arm_kd=2.0,
        ee_kp=75.0,
        telemetry_path=str(telemetry_path),
    )
    try:
        assert len(session.put_calls) == 2
        arm_command = arm_pb2.JointTrajectory()
        arm_command.ParseFromString(session.put_calls[0][1])
        ee_command = arm_pb2.JointTrajectory()
        ee_command.ParseFromString(session.put_calls[1][1])
        assert arm_command.on_timeout == arm_pb2.TIMEOUT_BEHAVIOR_HOLD
        assert list(arm_command.points[0].q) == pytest.approx(initial_arm)
        assert list(arm_command.points[0].kp) == pytest.approx([20.0] * 6)
        assert list(arm_command.points[0].kd) == pytest.approx([2.0] * 6)
        assert ee_command.on_timeout == arm_pb2.TIMEOUT_BEHAVIOR_HOLD
        assert list(ee_command.points[0].q) == pytest.approx(initial_ee)
        assert list(ee_command.points[0].kp) == pytest.approx([75.0])

        frozen_target = robot.command_target
        assert frozen_target is not None
        session.emit_state(ARM_PREFIX, [0.11, -0.2, 0.3, -0.1, 0.2, -0.3])
        robot.send_action(frozen_target)
        repeated_arm_command = arm_pb2.JointTrajectory()
        repeated_arm_command.ParseFromString(session.put_calls[-2][1])
        assert list(repeated_arm_command.points[0].q) == pytest.approx(initial_arm)

        events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        contract = next(event for event in events if event["event"] == "controller_contract")
        assert contract["arm_timeout"] == "TIMEOUT_BEHAVIOR_HOLD"
        assert contract["experimental_hold_override"] is True
        assert contract["arm_kp"] == 20.0
        assert contract["arm_kd"] == 2.0
        assert contract["ee_kp"] == 75.0
    finally:
        robot.disconnect()


def test_hold_override_rejects_out_of_range_start_before_publish(monkeypatch):
    session = _FakeSession(arm_timeouts=[arm_pb2.TIMEOUT_BEHAVIOR_HOLD])
    session.initial_payloads[f"{ARM_PREFIX}/arm/joint_state"] = _joint_state(
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    monkeypatch.setattr(follower_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohFollower(
        _config(
            "execute",
            arm_timeout_behavior="hold",
            allow_arm_hold_timeout_without_ramp_stop=True,
            arm_kp=20.0,
            ee_kp=75.0,
        )
    )
    with pytest.raises(ValueError, match="outside controller range"):
        robot.connect()
    assert session.release_ids == [22, 11]
    assert not session.put_calls
    assert not [key for key, _payload in session.get_calls if key.endswith("/rpc/set_mode")]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm_kp": 40.1},
        {"arm_kd": 2.1},
        {"ee_kp": 150.1},
        {"ee_kd_default_scale": 0.51},
    ],
)
def test_gain_limits_can_only_be_lowered(kwargs):
    lowered = _config("shadow", arm_kp=20.0, arm_kd=2.0, ee_kp=75.0)
    assert lowered.arm_kp == 20.0
    assert lowered.ee_kp == 75.0
    with pytest.raises(ValueError, match="hard safety maximum"):
        _config("shadow", **kwargs)


@pytest.mark.parametrize(
    ("arm_step_limit_rad", "ee_step_limit_rad"),
    ((0.005, 0.002), (0.002, 0.002)),
)
def test_robot_config_for_stage_preserves_fingerprint_bound_hold_profile(
    tmp_path, arm_step_limit_rad, ee_step_limit_rad
):
    persisted = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_step_limit_rad=arm_step_limit_rad,
        ee_step_limit_rad=ee_step_limit_rad,
        arm_tracking_error_rad=0.050,
        ee_tracking_error_rad=0.020,
        arm_kp=40.0,
        arm_kd=2.0,
        ee_kp=75.0,
        ee_kd_default_scale=0.5,
    )
    deployment = FireflyDeploymentConfig(robot=persisted)
    for stage in ("shadow", "hold", "replay", "act"):
        result = robot_config_for_stage(
            deployment,
            stage,
            tmp_path / f"{stage}.jsonl",
        )
        assert result.mode == ("shadow" if stage == "shadow" else "execute")
        assert result.arm_timeout_behavior == "hold"
        assert result.allow_arm_hold_timeout_without_ramp_stop is True
        assert result.arm_kp == 40.0
        assert result.arm_kd == 2.0
        assert result.ee_kp == 75.0
        assert result.arm_step_limit_rad == arm_step_limit_rad
        assert result.ee_step_limit_rad == ee_step_limit_rad
        assert result.arm_tracking_error_rad == 0.050
        assert result.ee_tracking_error_rad == 0.020


@pytest.mark.parametrize(
    ("arm_step_limit_rad", "ee_step_limit_rad"),
    ((0.003, 0.002), (0.005, 0.003), (0.002, 0.001)),
)
def test_robot_config_for_stage_rejects_unapproved_hold_step_pair(
    tmp_path, arm_step_limit_rad, ee_step_limit_rad
):
    unapproved = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_step_limit_rad=arm_step_limit_rad,
        ee_step_limit_rad=ee_step_limit_rad,
        arm_tracking_error_rad=0.050,
        ee_tracking_error_rad=0.020,
        arm_kp=40.0,
        arm_kd=2.0,
        ee_kp=75.0,
        ee_kd_default_scale=0.5,
    )
    deployment = FireflyDeploymentConfig(robot=unapproved)
    with pytest.raises(PermissionError, match="must be exactly one of"):
        robot_config_for_stage(deployment, "hold", tmp_path / "hold.jsonl")


def test_robot_config_for_stage_rejects_nonconservative_hold_profile(tmp_path):
    unsafe = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
    )
    deployment = FireflyDeploymentConfig(robot=unsafe)
    with pytest.raises(PermissionError, match="fingerprint-bound persistent HOLD"):
        robot_config_for_stage(deployment, "hold", tmp_path / "hold.jsonl")


def test_robot_config_for_stage_requires_exact_gr80_payload_gravity_profile(tmp_path):
    arm_only = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_step_limit_rad=0.002,
        ee_step_limit_rad=0.002,
        arm_tracking_error_rad=0.050,
        ee_tracking_error_rad=0.020,
        arm_kp=40.0,
        arm_kd=2.0,
        ee_kp=75.0,
        ee_kd_default_scale=0.5,
        enable_arm_gravity_ff=True,
        expected_arm_urdf_sha256=(
            "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291"
        ),
        arm_gravity_ff_max_abs_nm=6.0,
    )
    with pytest.raises(PermissionError, match="arm-only"):
        robot_config_for_stage(
            FireflyDeploymentConfig(robot=arm_only),
            "hold",
            tmp_path / "arm-only-gravity.jsonl",
        )

    approved = deployment_module.replace(
        arm_only,
        enable_ee_payload_gravity_ff=True,
        expected_ee_urdf_sha256=(
            "f74b3e76b14175c788c5ef70dd0c1941958d229a8461da6f20e17543e4ba1114"
        ),
        expected_arm_urdf_root_link="base_link",
        expected_arm_urdf_tip_link="link_6",
        expected_ee_urdf_root_link="base_link",
        expected_ee_urdf_tip_link="tcp",
        expected_ee_payload_mass_kg=0.41,
        expected_ee_payload_com_xyz_m=[
            0.005537804878,
            0.000026829268,
            0.048889970714,
        ],
        arm_gravity_ff_max_abs_nm=8.0,
    )
    result = robot_config_for_stage(
        FireflyDeploymentConfig(robot=approved),
        "hold",
        tmp_path / "payload-gravity-hold.jsonl",
    )
    assert result.mode == "execute"
    assert result.enable_arm_gravity_ff is True
    assert result.enable_ee_payload_gravity_ff is True
    assert result.arm_gravity_ff_max_abs_nm == 8.0

    with pytest.raises(PermissionError, match="0.002/0.002"):
        robot_config_for_stage(
            FireflyDeploymentConfig(
                robot=deployment_module.replace(approved, arm_step_limit_rad=0.005)
            ),
            "hold",
            tmp_path / "wide-payload-gravity-hold.jsonl",
        )
    with pytest.raises(PermissionError, match="exactly 8.0"):
        robot_config_for_stage(
            FireflyDeploymentConfig(
                robot=deployment_module.replace(
                    approved, arm_gravity_ff_max_abs_nm=7.0
                )
            ),
            "hold",
            tmp_path / "low-cap-payload-gravity-hold.jsonl",
        )
    with pytest.raises(PermissionError, match="expected_ee_urdf_sha256"):
        robot_config_for_stage(
            FireflyDeploymentConfig(
                robot=deployment_module.replace(
                    approved, expected_ee_urdf_sha256="0" * 64
                )
            ),
            "hold",
            tmp_path / "wrong-ee-hash.jsonl",
        )


def test_v3_trimmed_slew002_local_yaml_is_an_independent_exact_hold_profile():
    root = Path(__file__).parents[1]
    old_path = root / "configs" / "deploy_firefly_y6_hold_validated_v3_trimmed.local.yaml"
    new_path = (
        root
        / "configs"
        / "deploy_firefly_y6_hold_validated_v3_trimmed_slew002.local.yaml"
    )
    with old_path.open() as stream:
        old = draccus.load(FireflyDeploymentConfig, stream)
    with new_path.open() as stream:
        new = draccus.load(FireflyDeploymentConfig, stream)

    assert old.robot.arm_step_limit_rad == 0.005
    assert old.robot.ee_step_limit_rad == 0.002
    assert new.robot.arm_step_limit_rad == 0.002
    assert new.robot.ee_step_limit_rad == 0.002
    assert new.robot.arm_tracking_error_rad == 0.050
    assert new.robot.ee_tracking_error_rad == 0.020
    assert new.robot.tracking_error_frames == 3
    assert new.robot.arm_kp == 40.0
    assert new.robot.arm_kd == 2.0
    assert new.robot.ee_kp == 75.0
    assert new.robot.client_name != old.robot.client_name
    assert new.output_dir != old.output_dir
    assert new.checkpoint == old.checkpoint
    assert new.dataset_repo_id == old.dataset_repo_id
    assert new.dataset_root == old.dataset_root
    assert new.dataset_controller_contract == old.dataset_controller_contract


def test_v3_trimmed_gravity_ff_local_yaml_is_independent_and_hash_bound():
    root = Path(__file__).parents[1]
    base_path = (
        root
        / "configs"
        / "deploy_firefly_y6_hold_validated_v3_trimmed_slew002.local.yaml"
    )
    gravity_path = (
        root
        / "configs"
        / "deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff.local.yaml"
    )
    with base_path.open() as stream:
        base = draccus.load(FireflyDeploymentConfig, stream)
    with gravity_path.open() as stream:
        gravity = draccus.load(FireflyDeploymentConfig, stream)

    assert base.robot.enable_arm_gravity_ff is False
    assert base.robot.expected_arm_urdf_sha256 is None
    assert gravity.robot.enable_arm_gravity_ff is True
    assert gravity.robot.expected_arm_urdf_sha256 == (
        "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291"
    )
    assert gravity.robot.arm_gravity_ff_max_abs_nm == 6.0
    assert gravity.robot.arm_step_limit_rad == 0.002
    assert gravity.robot.ee_step_limit_rad == 0.002
    assert gravity.robot.arm_tracking_error_rad == 0.050
    assert gravity.robot.ee_tracking_error_rad == 0.020
    assert gravity.robot.arm_kp == 40.0
    assert gravity.robot.arm_kd == 2.0
    assert gravity.robot.ee_kp == 75.0
    assert gravity.robot.client_name != base.robot.client_name
    assert gravity.output_dir != base.output_dir
    assert gravity.checkpoint == base.checkpoint
    assert gravity.dataset_root == base.dataset_root


def test_v3_trimmed_gr80_payload_gravity_yaml_is_independent_and_bound():
    root = Path(__file__).parents[1]
    arm_only_path = (
        root
        / "configs"
        / "deploy_firefly_y6_hold_validated_v3_trimmed_slew002_gravity_ff.local.yaml"
    )
    payload_path = (
        root
        / "configs"
        / (
            "deploy_firefly_y6_hold_validated_v3_trimmed_slew002_"
            "gravity_ff_gr80_payload.local.yaml"
        )
    )
    with arm_only_path.open() as stream:
        arm_only = draccus.load(FireflyDeploymentConfig, stream)
    with payload_path.open() as stream:
        payload = draccus.load(FireflyDeploymentConfig, stream)

    assert arm_only.robot.enable_ee_payload_gravity_ff is False
    assert payload.robot.enable_arm_gravity_ff is True
    assert payload.robot.enable_ee_payload_gravity_ff is True
    assert payload.robot.expected_arm_urdf_sha256 == (
        "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291"
    )
    assert payload.robot.expected_ee_urdf_sha256 == (
        "f74b3e76b14175c788c5ef70dd0c1941958d229a8461da6f20e17543e4ba1114"
    )
    assert payload.robot.expected_arm_urdf_tip_link == "link_6"
    assert payload.robot.expected_ee_urdf_root_link == "base_link"
    assert payload.robot.expected_ee_payload_mass_kg == pytest.approx(0.41)
    assert payload.robot.expected_ee_payload_com_xyz_m == pytest.approx(
        [0.005537804878, 0.000026829268, 0.048889970714]
    )
    assert payload.robot.arm_gravity_ff_max_abs_nm == 8.0
    assert payload.robot.arm_step_limit_rad == 0.002
    assert payload.robot.ee_step_limit_rad == 0.002
    assert payload.robot.arm_tracking_error_rad == 0.050
    assert payload.robot.ee_tracking_error_rad == 0.020
    assert payload.robot.arm_kp == 40.0
    assert payload.robot.arm_kd == 2.0
    assert payload.robot.client_name != arm_only.robot.client_name
    assert payload.output_dir != arm_only.output_dir
    assert payload.checkpoint == arm_only.checkpoint


def test_deployment_local_yaml_keeps_stock_execution_values():
    path = Path(__file__).parents[1] / "configs" / "deploy_firefly_y6.local.yaml"
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)
    assert config.robot.arm_timeout_behavior == "ramp_stop"
    assert config.robot.allow_arm_hold_timeout_without_ramp_stop is False
    assert config.robot.arm_kp == 40.0
    assert config.robot.arm_kd == 2.0
    assert config.robot.ee_kp == 150.0
    assert config.robot.arm_step_limit_rad == 0.035
    assert config.robot.ee_step_limit_rad == 0.012


def test_robot_config_for_stage_accepts_only_exact_elevated_gr80_gain_profile(tmp_path):
    baseline = _config(
        "shadow",
        arm_timeout_behavior="hold",
        allow_arm_hold_timeout_without_ramp_stop=True,
        arm_step_limit_rad=0.002,
        ee_step_limit_rad=0.002,
        arm_tracking_error_rad=0.050,
        ee_tracking_error_rad=0.020,
        arm_kp=60.0,
        arm_kd=2.5,
        ee_kp=75.0,
        ee_kd_default_scale=0.5,
        **_payload_gravity_kwargs(
            expected_arm_urdf_sha256=(
                "f923d1f790ffc3bd7561cf9bdf43d749515a6e157961ea7cb013bd186f550291"
            ),
            expected_ee_urdf_sha256=(
                "f74b3e76b14175c788c5ef70dd0c1941958d229a8461da6f20e17543e4ba1114"
            ),
        ),
    )
    result = robot_config_for_stage(
        FireflyDeploymentConfig(robot=baseline),
        "act",
        tmp_path / "elevated-payload-act.jsonl",
    )
    assert (result.arm_kp, result.arm_kd) == (60.0, 2.5)

    for arm_kp, arm_kd in ((60.0, 2.0), (40.0, 2.5)):
        with pytest.raises(ValueError, match="exact 60.0/2.5"):
            deployment_module.replace(baseline, arm_kp=arm_kp, arm_kd=arm_kd)


def test_v3_trimmed_gr80_payload_kp60_yaml_is_exact_and_stage_approved():
    root = Path(__file__).parents[1]
    path = (
        root
        / "configs"
        / (
            "deploy_firefly_y6_hold_validated_v3_trimmed_slew002_"
            "gravity_ff_gr80_payload_kp60_kd2p5.local.yaml"
        )
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)

    assert (config.robot.arm_kp, config.robot.arm_kd) == (60.0, 2.5)
    assert config.robot.enable_ee_payload_gravity_ff is True
    result = robot_config_for_stage(config, "act", Path("act.jsonl"))
    assert (result.arm_kp, result.arm_kd) == (60.0, 2.5)


def test_fast_per_joint_limits_drive_adapter_and_live_preflight(monkeypatch):
    arm_limits = [0.003, 0.003, 0.002, 0.003, 0.003, 0.003]
    config = _config(
        "shadow",
        arm_step_limit_rad=0.003,
        arm_step_limits_rad=arm_limits,
        ee_step_limit_rad=0.003,
    )
    assert config.resolved_arm_step_limits_rad() == pytest.approx(arm_limits)

    result = adapt_action_chunk_slew_rate(
        [[1.0] * 7] * 10,
        [0.0] * 7,
        arm_step_limit_rad=0.003,
        arm_step_limits_rad=arm_limits,
        ee_step_limit_rad=0.003,
    )
    assert result.adapted_chunk[0] == pytest.approx(
        (0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003)
    )

    session = _FakeSession()
    robot = _connect(
        monkeypatch,
        session,
        "shadow",
        arm_step_limit_rad=0.003,
        arm_step_limits_rad=arm_limits,
        ee_step_limit_rad=0.003,
    )
    try:
        accepted = [0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003]
        robot.preflight_actions([[0.0] * 7, accepted])
        rejected = list(accepted)
        rejected[2] = 0.0021
        with pytest.raises(ValueError, match=r"joint_3\.pos step"):
            robot.preflight_actions([[0.0] * 7, rejected])
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    ("closing_limit", "measured_ee", "should_stop", "direction"),
    [
        (0.060, 0.355, False, None),
        (0.060, 0.361, True, "closing-contact"),
        (0.060, 0.279, True, "opening"),
        (None, 0.321, True, "closing-contact"),
    ],
)
def test_directional_ee_tracking_uses_closing_limit_only_for_measured_above_target(
    closing_limit, measured_ee, should_stop, direction
):
    robot = HexZenohFollower(
        _config(
            "shadow",
            ee_tracking_error_rad=0.020,
            ee_closing_tracking_error_rad=closing_limit,
        )
    )
    robot._last_target = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.300)
    measured = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, measured_ee)
    if not should_stop:
        for _ in range(6):
            robot._check_tracking(measured)
        assert robot._ee_error_frames == 0
        return
    robot._check_tracking(measured)
    robot._check_tracking(measured)
    with pytest.raises(SafetyStopError, match=direction):
        robot._check_tracking(measured)


def test_fast90_yaml_is_exact_and_stage_approved():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / (
            "deploy_firefly_y6_hold_validated_v3_trimmed_fast90_per_joint_"
            "gravity_ff_gr80_payload_kp60_kd2p5_eekp150.local.yaml"
        )
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)

    assert config.act_duration_s == 90.0
    assert config.robot.arm_step_limit_rad == pytest.approx(0.003)
    assert config.robot.arm_step_limits_rad == pytest.approx(
        [0.003, 0.003, 0.002, 0.003, 0.003, 0.003]
    )
    assert config.robot.ee_step_limit_rad == pytest.approx(0.003)
    assert config.robot.arm_tracking_error_rad == pytest.approx(0.050)
    assert config.robot.ee_tracking_error_rad == pytest.approx(0.020)
    assert config.robot.ee_closing_tracking_error_rad == pytest.approx(0.060)
    assert (config.robot.arm_kp, config.robot.arm_kd, config.robot.ee_kp) == (
        60.0,
        2.5,
        150.0,
    )
    assert config.robot.enable_ee_payload_gravity_ff is True
    result = robot_config_for_stage(config, "act", Path("act_90s.jsonl"))
    assert result.mode == "execute"
    assert result.arm_step_limits_rad == pytest.approx(config.robot.arm_step_limits_rad)

    weakened = deployment_module.replace(
        config.robot, ee_closing_tracking_error_rad=0.059
    )
    with pytest.raises(PermissionError, match="exactly 0.060"):
        robot_config_for_stage(
            deployment_module.replace(config, robot=weakened),
            "act",
            Path("rejected.jsonl"),
        )


def test_fullcycle_v4_motion_envelope_requires_exact_recording_limits():
    config = _config(
        "shadow",
        motion_envelope="fullcycle_v4",
        arm_step_limit_rad=0.0435,
        ee_step_limit_rad=0.0165,
    )
    assert config.resolved_arm_step_limits_rad() == pytest.approx([0.0435] * 6)
    assert config.ee_step_limit_rad == pytest.approx(0.0165)

    with pytest.raises(ValueError, match="exact 0.0435/0.0165"):
        deployment_module.replace(config, ee_step_limit_rad=0.0164)
    with pytest.raises(ValueError, match="one scalar arm step limit"):
        deployment_module.replace(config, arm_step_limits_rad=[0.0435] * 6)


def test_fullcycle_v4_100k_yaml_is_exact_and_stage_approved():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)

    assert config.dataset_repo_id == "local/firefly_y6_fullcycle_v4"
    assert config.checkpoint.endswith("/checkpoints/100000/pretrained_model")
    assert config.robot.motion_envelope == "fullcycle_v4"
    assert config.robot.arm_step_limit_rad == pytest.approx(0.0435)
    assert config.robot.ee_step_limit_rad == pytest.approx(0.0165)
    assert config.robot.arm_tracking_error_rad == pytest.approx(0.150)
    assert config.robot.ee_tracking_error_rad == pytest.approx(0.050)
    assert config.robot.ee_closing_tracking_error_rad == pytest.approx(0.060)
    assert (config.robot.arm_kp, config.robot.arm_kd, config.robot.ee_kp) == (
        40.0,
        2.0,
        150.0,
    )
    assert config.robot.enable_ee_payload_gravity_ff is True
    assert config.robot.cameras["wrist"].rotation.name == "ROTATE_180"
    assert config.enable_action_slew_rate_adapter is True
    assert config.act_chunk_inference_mode == "synchronous_hold"
    assert config.grasp_phase_correction.enabled is False
    assert len(config.grasp_phase_correction.act_start_center_rad) == 7
    assert config.inference_seed == 1000

    for stage in ("shadow", "hold", "replay", "act"):
        result = robot_config_for_stage(config, stage, Path(f"{stage}.jsonl"))
        assert result.mode == ("shadow" if stage == "shadow" else "execute")
        assert result.motion_envelope == "fullcycle_v4"

    with pytest.raises(PermissionError, match="exact attested fullcycle-v4 dataset"):
        robot_config_for_stage(
            deployment_module.replace(config, dataset_repo_id="local/wrong"),
            "act",
            Path("wrong-dataset.jsonl"),
        )
    with pytest.raises(PermissionError, match="cross-chunk action slew adapter"):
        robot_config_for_stage(
            deployment_module.replace(config, enable_action_slew_rate_adapter=False),
            "act",
            Path("no-adapter.jsonl"),
        )
    with pytest.raises(PermissionError, match="fresh-observation synchronous HOLD"):
        robot_config_for_stage(
            deployment_module.replace(config, act_chunk_inference_mode="pipelined"),
            "act",
            Path("stale-pipeline.jsonl"),
        )
    with pytest.raises(PermissionError, match="disable fixed-joint grasp correction"):
        wrong_start = deployment_module.replace(
            config.grasp_phase_correction,
            act_start_tolerance_rad=[0.2] * 7,
        )
        robot_config_for_stage(
            deployment_module.replace(config, grasp_phase_correction=wrong_start),
            "act",
            Path("wrong-start.jsonl"),
        )


def test_fullcycle_v4_visual_latch_exact_profile_camera_and_provenance_are_approved():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)

    visual = config.visual_phase_latch
    assert visual == VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=deployment_module._fullcycle_v4_open_only_tip_servo_config(),
    )
    assert visual.tip_servo.probe_profile is TipServoProbeProfile.OPEN_ONLY
    assert visual.tip_servo.descent_step_m == pytest.approx(0.0)
    assert visual.tip_servo.max_total_descent_m == pytest.approx(0.0)
    assert visual.tip_servo.seeking_timeout_ms == pytest.approx(20_000.0)
    assert visual.tip_servo.coarse_capture_jaw_rad == pytest.approx(
        (0.285, 0.325)
    )
    assert config.visual_latch_audit_sha256 == (
        "e156e0b836cad9298af46ba8d0fbbfc4"
        "46e1b07d11affbd1913c004b5a260415"
    )
    assert config.output_dir.endswith("_tip_probe_v5")
    assert config.visual_latch_audit_path.endswith("_schema5_exact.json")
    assert visual.enabled is True
    assert visual.red_hsv_ranges == (
        ((0, 100, 60), (6, 255, 255)),
        ((174, 100, 60), (179, 255, 255)),
    )
    assert visual.red_center_roi_norm == pytest.approx((0.40, 0.49, 0.48, 0.60))
    assert visual.close_target_rad == pytest.approx(0.320)
    assert visual.release_target_rad == pytest.approx(0.420)
    assert visual.box_confirm_frames == 10
    assert config.robot.camera_max_age_s == pytest.approx(0.100)
    result = robot_config_for_stage(config, "act", Path("visual-act.jsonl"))
    assert result.mode == "execute"
    provenance = visual_phase_latch_provenance(visual)
    assert provenance["fields"]["blue_release_roi_fraction_min"] == pytest.approx(
        0.58
    )


def test_fullcycle_v4_real_yaml_schema5_audit_is_rejected():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)
    with pytest.raises(
        PermissionError,
        match="schema_version must be exactly 6, found 5",
    ):
        capture_deployment_sources(config)


def test_fullcycle_v4_visual_latch_rejects_profile_camera_and_freshness_drift():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)

    changed_visual = deployment_module.replace(
        config.visual_phase_latch, blue_global_fraction_min=0.69
    )
    with pytest.raises(PermissionError, match="exact calibrated"):
        robot_config_for_stage(
            deployment_module.replace(config, visual_phase_latch=changed_visual),
            "act",
            Path("wrong-visual.jsonl"),
        )

    wrist = config.robot.cameras["wrist"]
    wrong_wrist = deployment_module.replace(
        wrist, serial_number_or_name="000000000000"
    )
    wrong_cameras = dict(config.robot.cameras)
    wrong_cameras["wrist"] = wrong_wrist
    wrong_robot = deployment_module.replace(config.robot, cameras=wrong_cameras)
    with pytest.raises(PermissionError, match="exact wrist D405"):
        robot_config_for_stage(
            deployment_module.replace(config, robot=wrong_robot),
            "act",
            Path("wrong-camera.jsonl"),
        )

    stale_robot = deployment_module.replace(config.robot, camera_max_age_s=0.099)
    with pytest.raises(PermissionError, match="camera freshness"):
        robot_config_for_stage(
            deployment_module.replace(config, robot=stale_robot),
            "act",
            Path("wrong-freshness.jsonl"),
        )


def test_visual_latch_and_legacy_joint_corrector_are_mutually_exclusive():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
    )
    with path.open() as stream:
        config = draccus.load(FireflyDeploymentConfig, stream)
    legacy = GraspPhaseCorrectionConfig(
        enabled=True,
        pickup_center_rad=[0.0] * 6,
        pickup_tolerance_rad=[0.1] * 6,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        deployment_module.replace(config, grasp_phase_correction=legacy)


def test_direct_act_90s_skips_only_absent_entries_without_writing_ledger(tmp_path):
    require_acceptance_order(
        tmp_path,
        "act_90s",
        DEPLOYMENT_FINGERPRINT,
        operator_authorized_direct_act_90s=True,
    )
    assert not (tmp_path / "acceptance.json").exists()

    (tmp_path / "acceptance.json").write_text(
        json.dumps({"shadow": {"reviewed": False}})
    )
    with pytest.raises(PermissionError, match="shadow"):
        require_acceptance_order(
            tmp_path,
            "act_90s",
            DEPLOYMENT_FINGERPRINT,
            operator_authorized_direct_act_90s=True,
        )
