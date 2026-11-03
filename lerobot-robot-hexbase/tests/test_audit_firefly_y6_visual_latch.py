from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_visual_latch import (
    BlobEvidence,
    FrameStatus,
    VisualEvidence,
    VisualPhaseLatchConfig,
)
from lerobot_robot_hexbase.firefly_tip_servo import (
    TipServoConfig,
    TipServoMode,
    TipServoProbeProfile,
)
from scripts import audit_firefly_y6_visual_latch as audit


SCHEMA6_CHECKPOINT = Path(
    "/synthetic/run2/checkpoints/020000/pretrained_model"
)


def _schema6_policy_source_binding(
    *,
    source_repo_id: str = "local/firefly_y6_fullcycle_v4",
    source_root: str = "/dataset",
    upstream_digest: str = "a" * 64,
) -> dict:
    return {
        "policy_attestation": {
            "authorized_runtime_role": "arm6_seeking",
            "policy_schema_role": "arm_only_6d",
            "selected_checkpoint": {
                "step": 20_000,
                "directory": "020000",
                "pretrained_model": str(SCHEMA6_CHECKPOINT),
                "model_sha256": "1" * 64,
                "attestation_sha256": "2" * 64,
                "checkpoint_files_sha256": "3" * 64,
            },
        },
        "projection_binding": {
            "manifest_document_sha256": "4" * 64,
            "source": {
                "repo_id": source_repo_id,
                "root": source_root,
                "upstream_training_input_digest": upstream_digest,
            },
            "source_dimension": 7,
            "target_dimension": 6,
            "removed_feature": "jaw_joint1.pos",
            "removed_index": 6,
            "exact_first_six": True,
            "videos_byte_identical": True,
        },
    }


class _Schema6Capture:
    selected_checkpoint = SCHEMA6_CHECKPOINT
    selected_step = 20_000

    def source_binding(self) -> dict:
        return _schema6_policy_source_binding()


def _original_7d_features() -> dict:
    names = list(audit.FULLCYCLE_V4_ACTION_STATE_NAMES)
    return {
        "action": {"dtype": "float32", "shape": [7], "names": names},
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": names,
        },
        audit.WRIST_VIDEO_KEY: {
            "dtype": "video",
            "shape": [480, 640, 3],
        },
    }


def _actions(jaw: list[float]) -> list[tuple[float, ...]]:
    return [tuple([0.0] * 6 + [value]) for value in jaw]


def _small_config() -> VisualPhaseLatchConfig:
    return VisualPhaseLatchConfig(
        close_confirm_frames=3,
        box_confirm_frames=2,
        held_red_loss_fault_ms=100.0,
    )


def _safe_evidence():
    labels = audit.PhaseLabels(3, 10, 30, 40)
    close = [False] * 45
    held = [False] * 45
    blue = [False] * 45
    close[8:13] = [True] * 5
    held[10:30] = [True] * 20
    blue[27:35] = [True] * 8
    return labels, close, held, blue


def test_reconstructs_exact_four_alternating_jaw_crossings():
    labels = audit.reconstruct_phase_labels(
        _actions([0.30] * 3 + [0.41] * 3 + [0.32] * 3 + [0.42] * 3 + [0.31] * 3)
    )

    assert labels == audit.PhaseLabels(
        pregrasp_open_frame=3,
        pickup_close_frame=6,
        release_open_frame=9,
        return_close_frame=12,
    )


@pytest.mark.parametrize(
    "jaw, message",
    [
        ([0.30, 0.41, 0.32, 0.42], "complete confirmed"),
        ([0.41, 0.32, 0.42, 0.31, 0.42], "confirmed closed"),
        ([0.30, 0.375, 0.32, 0.42, 0.31], "complete confirmed"),
    ],
)
def test_reconstruction_fails_closed_on_noncanonical_cycle(jaw, message):
    with pytest.raises(audit.VisualLatchAuditInputError, match=message):
        audit.reconstruct_phase_labels(_actions(jaw))


def test_reconstruction_validates_all_seven_action_values():
    with pytest.raises(audit.VisualLatchAuditInputError, match="seven values"):
        audit.reconstruct_phase_labels([(0.0,)] * 5)


def test_reconstruction_uses_hysteresis_to_ignore_threshold_chatter():
    labels = audit.reconstruct_phase_labels(
        _actions(
            [0.30, 0.369, 0.371, 0.369, 0.371, 0.40]
            + [0.371, 0.369, 0.34]
            + [0.369, 0.371, 0.41]
            + [0.371, 0.369, 0.32]
        )
    )

    assert labels == audit.PhaseLabels(4, 7, 10, 13)


def test_reconstruction_ignores_shallow_post_release_jaw_correction():
    labels = audit.reconstruct_phase_labels(
        _actions(
            [0.30, 0.41, 0.32, 0.42, 0.365, 0.389, 0.32]
        )
    )

    assert labels == audit.PhaseLabels(1, 2, 3, 6)


def test_safe_evidence_requires_streaks_and_reports_offsets():
    labels, close, held, blue = _safe_evidence()

    report = audit.audit_evidence_flags(
        labels,
        close_flags=close,
        held_flags=held,
        blue_flags=blue,
        contact_confirm_frame=labels.pickup_close_frame,
        config=_small_config(),
        fps=10,
        windows=audit.AuditWindows(
            close_before_frames=4,
            close_after_frames=4,
            release_before_frames=5,
            release_after_frames=3,
        ),
    )

    assert report["safe"] is True
    assert report["close"]["confirmation_frame"] == 10
    assert report["close"]["confirmation_offset_frames"] == 0
    assert report["held_red"]["coverage"] == 1.0
    assert report["held_red"]["fault_run_frames"] == 3
    assert report["blue"]["release_confirmation_frame"] == 28
    assert report["blue"]["release_confirmation_offset_frames"] == -2


def test_premature_close_and_blue_streaks_fail_even_if_windows_detect():
    labels, close, held, blue = _safe_evidence()
    close[0:3] = [True] * 3
    blue[3:5] = [True] * 2

    report = audit.audit_evidence_flags(
        labels,
        close_flags=close,
        held_flags=held,
        blue_flags=blue,
        contact_confirm_frame=labels.pickup_close_frame,
        config=_small_config(),
        fps=10,
        windows=audit.AuditWindows(
            close_before_frames=4,
            close_after_frames=4,
            release_before_frames=5,
            release_after_frames=3,
        ),
    )

    assert report["safe"] is False
    assert report["close"]["premature_confirmation_frames"] == [2]
    assert report["blue"]["premature_confirmation_frames"] == [4]


def test_held_red_missing_run_uses_strict_elapsed_timeout():
    labels, close, held, blue = _safe_evidence()
    held[15:18] = [False] * 3

    report = audit.audit_evidence_flags(
        labels,
        close_flags=close,
        held_flags=held,
        blue_flags=blue,
        contact_confirm_frame=labels.pickup_close_frame,
        config=_small_config(),
        fps=10,
        windows=audit.AuditWindows(
            close_before_frames=4,
            close_after_frames=4,
            release_before_frames=5,
            release_after_frames=3,
        ),
    )

    assert report["held_red"]["fault_run_frames"] == 3
    assert report["held_red"]["max_consecutive_missing_frames"] == 3
    assert report["held_red"]["survives_loss_timeout"] is False
    assert report["safe"] is False


def test_rejection_reasons_are_aggregated_only_inside_phase_windows():
    labels, close, held, blue = _safe_evidence()
    empty = [()] * 45
    close_reasons = list(empty)
    close_reasons[7] = ("outside",)
    close_reasons[8] = ("ambiguous", "area")
    close_reasons[9] = ("ambiguous",)

    report = audit.audit_evidence_flags(
        labels,
        close_flags=close,
        held_flags=held,
        blue_flags=blue,
        contact_confirm_frame=labels.pickup_close_frame,
        close_reasons=close_reasons,
        config=_small_config(),
        fps=10,
        windows=audit.AuditWindows(
            close_before_frames=2,
            close_after_frames=1,
            release_before_frames=5,
            release_after_frames=3,
        ),
    )

    assert report["close"]["rejection_reason_counts_in_window"] == {
        "ambiguous": 2,
        "area": 1,
    }


def test_evidence_array_length_mismatch_fails_closed():
    labels, close, held, blue = _safe_evidence()
    with pytest.raises(audit.VisualLatchAuditInputError, match="unequal lengths"):
        audit.audit_evidence_flags(
            labels,
            close_flags=close[:-1],
            held_flags=held,
            blue_flags=blue,
            contact_confirm_frame=labels.pickup_close_frame,
            config=_small_config(),
        )


def test_contour_diagnostics_report_counts_and_rank_quantiles():
    def blob(area, cx, cy, left):
        return {
            "area_fraction": area,
            "centroid_norm": [cx, cy],
            "bbox_norm": [left, 0.2, 0.1, 0.2],
        }

    summary = audit._contour_window_summary(
        [
            {
                "positive_contour_count": 4,
                "ambiguity_candidates": [
                    blob(0.04, 0.4, 0.5, 0.3),
                    blob(0.01, 0.1, 0.8, 0.0),
                ],
            },
            {
                "positive_contour_count": 5,
                "ambiguity_candidates": [
                    blob(0.03, 0.5, 0.6, 0.4),
                    blob(0.02, 0.2, 0.9, 0.1),
                    blob(0.005, 0.9, 0.1, 0.8),
                ],
            },
            {
                "positive_contour_count": 4,
                "ambiguity_candidates": [blob(0.02, 0.6, 0.7, 0.5)],
            },
        ]
    )

    assert summary["positive_contour_count_distribution"] == {"4": 2, "5": 1}
    assert summary["ambiguity_candidate_count_distribution"] == {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert summary["largest_ambiguity_candidate"]["area_fraction"]["median"] == pytest.approx(0.03)
    assert summary["second_largest_ambiguity_candidate"]["frames_with_contour"] == 2
    assert summary["second_largest_ambiguity_candidate"]["centroid_norm"]["x"]["median"] == pytest.approx(0.15)


def test_runtime_proxy_applies_measured_arm_speed_open_and_contact_streaks():
    config = VisualPhaseLatchConfig(
        close_confirm_frames=3,
        grasp_confirm_frames=5,
    )
    rows = []
    for frame in range(20):
        command_jaw = 0.30 if frame == 0 else (0.41 if frame < 5 else 0.32)
        measured_jaw = 0.30 if frame == 0 else (0.41 if frame < 5 else 0.34)
        rows.append(
            {
                "action": [0.0] * 6 + [command_jaw],
                "observation.state": [0.0] * 6 + [measured_jaw],
            }
        )

    proxy = audit._runtime_gate_proxies(
        rows,
        close_visual_flags=[True] * len(rows),
        config=config,
        fps=30,
    )

    assert proxy["first_policy_close_intent_frame"] == 5
    assert proxy["close_latch_frame"] == 7
    assert proxy["arm_speed_rad_s_at_close_latch"] == pytest.approx(0.0)
    assert proxy["contact_confirm_frame"] == 15
    assert proxy["contact_gap_rad_at_confirmation"] == pytest.approx(0.02)
    assert proxy["simulated_command_ee_rad_at_contact_confirmation"] == pytest.approx(0.32)
    assert proxy["jaw_speed_rad_s_at_contact_confirmation"] == pytest.approx(0.0)


def test_exact_state_machine_replay_finishes_and_models_suppressed_human_release():
    config = VisualPhaseLatchConfig(
        enabled=True,
        close_confirm_frames=1,
        grasp_confirm_frames=1,
        lift_confirm_frames=1,
        transport_confirm_frames=1,
        box_confirm_frames=1,
        release_settle_frames=1,
        release_confirm_cycles=1,
        home_ready_confirm_cycles=1,
        home_settle_confirm_cycles=1,
        close_max_arm_speed_rad_s=1.0,
        jaw_settled_speed_max_rad_s=10.0,
        lift_height_m=0.02,
        transport_height_m=0.04,
        release_max_tcp_speed_m_s=10.0,
        home_max_arm_speed_rad_s=1.0,
    )
    jaws = [0.30, 0.41, 0.41, 0.36] + [0.40] * 5 + [0.314] * 10
    policy_jaws = [0.30, 0.41, 0.30] + [0.30] * (len(jaws) - 3)
    rows = [
        {
            "action": [0.0] * 6 + [policy_jaws[index]],
            "observation.state": [0.0] * 6 + [jaw],
        }
        for index, jaw in enumerate(jaws)
    ]
    red = BlobEvidence(accepted=True, held_accepted=True)
    absent = BlobEvidence()
    blue = BlobEvidence(accepted=True)
    evidences = [
        VisualEvidence(
            frame_id=index,
            captured_at_ns=(index + 1) * 33_333_333,
            evaluated_at_ns=(index + 1) * 33_333_333,
            frame_age_ms=0.0,
            frame_status=FrameStatus.FRESH_UNIQUE,
            red=red if index >= 2 else absent,
            blue=blue if index >= 6 else absent,
        )
        for index in range(len(rows))
    ]

    class FakeFk:
        def __init__(self):
            self.index = 0

        def end_effector_pose(self, _q, tool_offset):
            assert tuple(tool_offset) == audit.TCP_TOOL_OFFSET_M
            z = [0.0, 0.0, 0.0, 0.0, 0.03, 0.05] + [0.05] * 20
            result = ((0.0, 0.0, z[self.index]), ((1.0, 0.0, 0.0),) * 3)
            self.index += 1
            return result

    report = audit._state_machine_replay(
        rows,
        evidences,
        audit.PhaseLabels(1, 2, 8, 9),
        config=config,
        fk_model=FakeFk(),
    )

    assert report["ordered_complete"] is True
    assert report["fault_free"] is True
    assert report["final_phase"] == "done"
    assert report["counterfactual_held_ee"]["substituted_frames"] > 0
    assert list(report["milestone_frames"]) == list(audit.STATE_MACHINE_MILESTONES)


def _tip_probe_rows_and_evidence(*, excessive_descent: bool = False):
    rows = []
    evidences = []
    absent = BlobEvidence()
    for index in range(20):
        q0 = (
            max(0, index - 5) * 0.006
            if excessive_descent
            else index * 0.001
        )
        rows.append(
            {
                "action": [q0] + [0.0] * 5 + [0.41],
                "observation.state": [q0] + [0.0] * 5 + [0.41],
            }
        )
        area = 0.018 if index < 6 else 0.0205
        red = BlobEvidence(
            contour_area_fraction=area,
            centroid_norm=(0.46, 0.48),
            candidate_count=1,
        )
        evidences.append(
            VisualEvidence(
                frame_id=index,
                captured_at_ns=(index + 1) * 33_333_333,
                evaluated_at_ns=(index + 1) * 33_333_333,
                frame_age_ms=0.0,
                frame_status=FrameStatus.FRESH_UNIQUE,
                red=red,
                blue=absent,
            )
        )
    return rows, evidences


class _TipProbeFk:
    def end_effector_pose(self, q, tool_offset):
        assert tuple(tool_offset) == audit.TCP_TOOL_OFFSET_M
        return (
            (0.2, 0.1, 0.1 - float(q[0])),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_tip_probe_demo_audit_is_a_bounded_no_close_trace_envelope():
    rows, evidences = _tip_probe_rows_and_evidence()

    report = audit._tip_probe_demo_trace_envelope(
        rows,
        evidences,
        audit.PhaseLabels(2, 15, 17, 19),
        config=TipServoConfig(),
        fk_model=_TipProbeFk(),
    )

    assert report["takeover_confirmation_frame"] == 5
    assert report["probe_progress_confirmation_frame"] == 11
    assert report["outcome"] == "probe_progress_complete"
    assert report["required_recorded_base_z_descent_m"] == pytest.approx(0.006)
    assert report["required_descent_steps"] == 3
    assert report["trace_envelope_pass"] is True
    assert report["allow_close"] is False
    assert report["close_authorized"] is False
    assert report["hardware_completion_validated"] is False
    assert report["closed_loop_convergence_claimed"] is False
    assert report["settling_and_command_response_modeled"] is False


def test_tip_probe_demo_audit_rejects_a_trace_outside_25mm_budget():
    rows, evidences = _tip_probe_rows_and_evidence(excessive_descent=True)

    report = audit._tip_probe_demo_trace_envelope(
        rows,
        evidences,
        audit.PhaseLabels(2, 15, 17, 19),
        config=TipServoConfig(),
        fk_model=_TipProbeFk(),
    )

    assert report["required_recorded_base_z_descent_m"] == pytest.approx(0.036)
    assert report["checks"]["within_recorded_base_z_budget"] is False
    assert report["trace_envelope_pass"] is False
    assert report["close_authorized"] is False


def _tip_probe_command_cycle(frame_id, *, x=0.46, y=0.48, area=0.018):
    return {
        "event": "command_cycle",
        "visual_telemetry": {
            "frame": {"id": frame_id, "status": "fresh_unique"},
            "red": {
                "centroid_norm": [x, y],
                "contour_area_fraction": area,
                "candidate_count": 1,
            },
        },
    }


def test_tip_probe_old_act_negative_control_requires_safe_noncompletion():
    records = [
        _tip_probe_command_cycle(index)
        for index in range(6)
    ] + [
        _tip_probe_command_cycle(index, y=0.70, area=0.0185)
        for index in range(6, 12)
    ]

    report = audit._audit_tip_probe_negative_control_records(
        records, config=TipServoConfig()
    )

    assert report["takeover_frame_id"] == 5
    assert report["probe_progress_confirmation_frame_id"] is None
    assert report["visual_would_close_confirmation_frame_id"] is None
    assert report["first_tracking_exit_frame_id"] == 6
    assert report["negative_control_pass"] is True
    assert report["close_authorized"] is False
    assert report["post_takeover_recorded_frames_used_for_convergence"] is False
    assert report["hardware_completion_validated"] is False


def test_tip_probe_negative_control_fails_if_old_act_reaches_progress_gate():
    records = [
        _tip_probe_command_cycle(index)
        for index in range(6)
    ] + [
        _tip_probe_command_cycle(index, area=0.021)
        for index in range(6, 12)
    ]

    report = audit._audit_tip_probe_negative_control_records(
        records, config=TipServoConfig()
    )

    assert report["probe_progress_confirmation_frame_id"] == 11
    assert report["negative_control_pass"] is False
    assert report["close_authorized"] is False


def test_tip_probe_negative_control_loader_rejects_an_unpinned_trace(tmp_path):
    path = tmp_path / "replacement.jsonl"
    path.write_text(json.dumps(_tip_probe_command_cycle(0)) + "\n")

    with pytest.raises(
        audit.VisualLatchAuditInputError,
        match="pinned old ACT artifact identity",
    ):
        audit._load_tip_probe_negative_control(
            path,
            config=TipServoConfig(),
        )


def test_tip_probe_audit_rejects_any_profile_that_authorizes_close():
    rows, evidences = _tip_probe_rows_and_evidence()

    with pytest.raises(audit.VisualLatchAuditInputError, match="allow_close=false"):
        audit._tip_probe_demo_trace_envelope(
            rows,
            evidences,
            audit.PhaseLabels(2, 15, 17, 19),
            config=TipServoConfig(allow_close=True),
            fk_model=_TipProbeFk(),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (TipServoConfig(allow_close=True), "allow_close=false"),
        (TipServoConfig(descent_step_m=0.001), "descent_step_m=0.002"),
        (
            TipServoConfig(max_total_descent_m=0.030),
            "max_total_descent_m=0.025",
        ),
    ],
)
def test_schema_v3_tip_probe_profile_is_pinned(config, message):
    with pytest.raises(audit.VisualLatchAuditInputError, match=message):
        audit._validate_tip_probe_audit_config(config)


def test_schema_v3_tip_probe_profile_allows_execute_provenance_without_execution():
    tip_config = TipServoConfig(mode=TipServoMode.EXECUTE)

    audit._validate_tip_probe_audit_config(tip_config)

    assert tip_config.mode is TipServoMode.EXECUTE
    assert tip_config.allow_close is False
    assert tip_config.max_total_descent_m == pytest.approx(0.025)
    assert audit._audit_schema_version(
        VisualPhaseLatchConfig(enabled=True, tip_servo=tip_config)
    ) == audit.TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION == 3

def test_schema_v6_open_only_profile_is_exact_and_axis_partitioned():
    tip_config = audit._open_only_tip_servo_config()

    audit._validate_tip_probe_audit_config(tip_config)

    assert tip_config.probe_profile is TipServoProbeProfile.OPEN_ONLY
    assert tip_config.descent_step_m == pytest.approx(0.0)
    assert tip_config.max_total_descent_m == pytest.approx(0.0)
    assert tip_config.allow_close is False
    assert audit._audit_schema_version(
        VisualPhaseLatchConfig(enabled=True, tip_servo=tip_config)
    ) == audit.TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION == 6
    assert tip_config.seeking_timeout_ms == pytest.approx(20_000.0)

    with pytest.raises(audit.VisualLatchAuditInputError, match="exact"):
        audit._validate_tip_probe_audit_config(
            audit.replace(tip_config, max_arm_speed_rad_s=0.069)
        )


def test_schema_v6_demo_envelope_namespaces_authority_and_discloses_policy_arm():
    rows, evidences = _tip_probe_rows_and_evidence()
    report = audit._tip_probe_open_only_demo_envelope(
        rows,
        evidences,
        audit.PhaseLabels(2, 15, 17, 19),
        config=audit._open_only_tip_servo_config(),
    )

    assert report["trace_envelope_pass"] is True
    assert report["recorded_closed_capture_observed"] is False
    authority = report["axis_authority"]
    assert authority["audit_schema_version"] == 6
    assert authority["legacy_schema5_authorized"] is False
    assert authority["policy_role"] == "arm6_seeking"
    assert authority["policy_numeric_observation_dimension"] == 6
    assert authority["policy_numeric_action_dimension"] == 6
    assert authority["policy_numeric_jaw_input"] is False
    assert authority["policy_numeric_jaw_output"] is False
    assert authority["jaw_may_remain_visible_in_video"] is True
    assert authority["pre_takeover_policy_arm_motion_authorized"] is True
    assert authority["post_takeover_arm_motion_authorized"] is False
    assert authority["whole_pipeline_arm_motion_prohibited"] is False
    assert authority["whole_pipeline_tcp_descent_prohibited"] is False
    assert authority["policy_jaw_authorized"] is False
    assert authority["main_latch_jaw_authorized"] is False
    assert authority["tip_capture_hold_authorized"] is True
    assert authority["tip_explicit_descent_authorized"] is False
    assert authority["jaw_close_authorized"] is False
    assert report["counterfactual_open_response_modeled"] is False
    assert report["hardware_completion_validated"] is False
    assert report["closed_loop_convergence_claimed"] is False
    assert report["observational_window"] == {
        "start_frame": 0,
        "end_frame": 2,
        "evaluated_frames": 3,
        "coarse_gate_true_frames": 0,
        "maximum_arm_speed_rad_s": pytest.approx(0.03),
        "observed_jaw_rad": {"min": 0.41, "max": 0.41},
        "role": "dataset_observation_only_not_deployment_positive_control",
    }
    assert report["checks"]["observational_window_evaluated"] is True


def test_schema_v6_demo_reports_a_measured_closed_capture_window_when_present():
    rows, evidences = _tip_probe_rows_and_evidence()
    for row in rows:
        row["observation.state"][6] = 0.309
    for index, evidence in enumerate(evidences):
        evidences[index] = audit.replace(
            evidence,
            red=audit.replace(
                evidence.red,
                contour_area_fraction=0.018,
                centroid_norm=(0.52, 0.45),
                candidate_count=1,
            ),
        )

    report = audit._tip_probe_open_only_demo_envelope(
        rows,
        evidences,
        audit.PhaseLabels(15, 16, 17, 19),
        config=audit._open_only_tip_servo_config(),
    )

    assert report["trace_envelope_pass"] is True
    assert report["recorded_closed_capture_observed"] is True
    assert report["recorded_closed_capture_confirmation_frame"] == 5
    assert report["recorded_closed_capture_gate_frames"] == 16
    assert report["observational_window"]["coarse_gate_true_frames"] == 16


def _open_only_command_cycle(frame_id: int, action_source: str) -> dict:
    offset = frame_id - audit.TIP_SERVO_OPEN_ONLY_WINDOW_START_FRAME_ID
    target_jaw = 0.28 if offset % 2 == 0 else 0.41
    return {
        "event": "command_cycle",
        "safety_ok": True,
        "target": [offset * 0.001] + [0.0] * 5 + [target_jaw],
        "measured": [0.0] * 6 + [0.3046],
        "visual_telemetry": {
            "frame": {"id": frame_id, "status": "fresh_unique"},
            "red": {
                "centroid_norm": [0.5297 - offset * 0.0007, 0.451],
                "contour_area_fraction": 0.018,
                "candidate_count": 1,
            },
            "arm_speed_rad_s": 0.06,
            "tcp_position_m": [0.2, 0.0, 0.145 - offset * 0.004],
            "pickup_policy_close_intent": {
                "action_source": action_source,
            },
            "tip_servo": {
                "commanded_arm_target_q_rad": None,
                "commanded_descent_m": 0.0,
                "commanded_jaw_target_rad": None,
                "probe_close_allowed": False,
                "arm_owner": "policy",
                "discard_policy_chunk": False,
                "smoother_reset_required": False,
            },
        },
    }


def test_schema_v6_negative_control_replays_axis_authority_at_boundary():
    sources = [
        "fresh_policy",
        "fresh_policy",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "fresh_policy",
    ]
    records = [{
        "event": "control_acquired",
        "hold_target": [0.0] * 6 + [0.30926513671875],
    }, *[
        _open_only_command_cycle(frame_id, source)
        for frame_id, source in zip(range(487, 495), sources, strict=True)
    ]]

    report = audit._audit_tip_probe_open_only_negative_control_records(
        records,
        config=audit._open_only_tip_servo_config(),
    )

    assert report["negative_control_pass"] is True
    assert report["calibrated_window_frames"] == 8
    assert report["first_confirmed_hold_frame_id"] == 492
    assert report["would_takeover_boundary_frame_id"] == 494
    assert report["calibrated_window_action_source_counts"] == {
        "fresh_policy": 3,
        "inference_hold": 5,
    }
    assert report["actual_tip_servo_arm_command_cycles"] == 0
    assert report["actual_tip_servo_descent_command_cycles"] == 0
    assert report["actual_tip_servo_jaw_command_cycles"] == 0
    historical = report["historical_out_of_scope"]
    assert historical["policy_arm_motion_observed"] is True
    assert historical["minimum_measured_tcp_delta_z_m"] == pytest.approx(-0.028)
    assert historical["final_jaw_target_below_capture_cycles"] == 4
    assert historical["final_jaw_target_above_capture_cycles"] == 4
    replay = report["axis_authority_replay"]
    assert replay["published_jaw_min_rad"] == pytest.approx(
        replay["capture_jaw_target_rad"]
    )
    assert replay["published_jaw_max_rad"] == pytest.approx(
        replay["capture_jaw_target_rad"]
    )
    assert replay["takeover_boundary_still_capture_hold"] is True
    assert replay["first_open_requires_takeover_and_settle"] is True
    assert replay["opened_target_monotonic_nondecreasing"] is True
    assert replay["opened_target_step_limit_pass"] is True
    assert replay["adversarial_reclose_suppressed"] is True
    assert replay["seeking_timeout_holds_capture"] is True
    runtime = report["runtime_authority_replay"]
    assert runtime["runtime_authority_replay_pass"] is True
    assert runtime["post_takeover_arm_exact_accepted_reference"] is True
    assert runtime["post_takeover_arm_owner_values"] == ["hold"]
    assert runtime["tampered_final_arm_rejected"] is True
    assert runtime["takeover_discard_policy_chunk"] is True
    assert runtime["takeover_smoother_reset_required"] is True
    assert runtime["first_open_requires_takeover_and_settle"] is True
    assert runtime["fault_hold_non_decreasing"] is True
    assert runtime["policy_role"] == "arm6_seeking"
    assert runtime["numeric_policy_observation_dimension"] == 6
    assert runtime["raw_policy_action_dimension"] == 6
    assert runtime["numeric_policy_jaw_input"] is False
    assert runtime["numeric_policy_jaw_output"] is False
    assert runtime["arm6_adapter_contract_pass"] is True
    assert runtime["raw_policy_jaw_telemetry_absent"] is True
    assert runtime["policy_role_telemetry_exact"] is True
    assert runtime["jaw_candidate_source_telemetry_exact"] is True
    assert runtime["seven_dimensional_policy_output_rejected"] is True
    assert runtime["synthesized_jaw_tamper_rejected"] is True
    assert "adversarial_policy_jaw_target_rad" not in runtime
    assert "policy_jaw_suppressed" not in runtime


@pytest.mark.parametrize(
    "missing",
    [
        "control_acquired",
        "safety_ok",
        "target",
        "measured",
        "visual_telemetry",
        "frame",
        "frame.id",
        "frame.status",
        "red",
        "red.centroid_norm",
        "red.contour_area_fraction",
        "red.candidate_count",
        "arm_speed_rad_s",
        "tcp_position_m",
        "pickup_policy_close_intent",
        "pickup_policy_close_intent.action_source",
        "tip_servo",
    ],
)
def test_schema_v6_negative_control_fails_closed_on_missing_layer_evidence(
    missing,
):
    sources = [
        "fresh_policy",
        "fresh_policy",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "fresh_policy",
    ]
    records = [{
        "event": "control_acquired",
        "hold_target": [0.0] * 6 + [0.30926513671875],
    }, *[
        _open_only_command_cycle(frame_id, source)
        for frame_id, source in zip(range(487, 495), sources, strict=True)
    ]]
    if missing == "control_acquired":
        records = records[1:]
    elif "." in missing:
        parent, child = missing.split(".", maxsplit=1)
        records[1]["visual_telemetry"][parent].pop(child)
    elif missing in {"safety_ok", "target", "measured", "visual_telemetry"}:
        records[1].pop(missing)
    else:
        records[1]["visual_telemetry"].pop(missing)

    with pytest.raises(audit.VisualLatchAuditInputError):
        audit._audit_tip_probe_open_only_negative_control_records(
            records,
            config=audit._open_only_tip_servo_config(),
        )


@pytest.mark.parametrize(
    "missing",
    [
        "commanded_arm_target_q_rad",
        "commanded_jaw_target_rad",
        "commanded_descent_m",
        "probe_close_allowed",
        "arm_owner",
        "discard_policy_chunk",
        "smoother_reset_required",
    ],
)
def test_schema_v6_negative_control_rejects_missing_required_tip_telemetry(
    missing,
):
    sources = [
        "fresh_policy",
        "fresh_policy",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "fresh_policy",
    ]
    records = [{
        "event": "control_acquired",
        "hold_target": [0.0] * 6 + [0.30926513671875],
    }, *[
        _open_only_command_cycle(frame_id, source)
        for frame_id, source in zip(range(487, 495), sources, strict=True)
    ]]
    records[1]["visual_telemetry"]["tip_servo"].pop(missing)

    with pytest.raises(audit.VisualLatchAuditInputError, match="required fields"):
        audit._audit_tip_probe_open_only_negative_control_records(
            records,
            config=audit._open_only_tip_servo_config(),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record["target"].__setitem__(0, "0.0"), "JSON number"),
        (
            lambda record: record["visual_telemetry"]["frame"].__setitem__(
                "id", 487.5
            ),
            "frame id",
        ),
        (
            lambda record: record["visual_telemetry"]["red"].__setitem__(
                "candidate_count", True
            ),
            "candidate count",
        ),
        (
            lambda record: record["visual_telemetry"]["tip_servo"].__setitem__(
                "commanded_descent_m", "0.0"
            ),
            "JSON number",
        ),
        (
            lambda record: record["visual_telemetry"].__setitem__(
                "tcp_position_m", [0.2, 0.0, False]
            ),
            "JSON number",
        ),
        (lambda record: record.__setitem__("safety_ok", 1), "safety_ok"),
        (
            lambda record: record["visual_telemetry"]["frame"].__setitem__(
                "status", "fresh"
            ),
            "supported string",
        ),
        (
            lambda record: record["visual_telemetry"]["red"].__setitem__(
                "contour_area_fraction", "0.018"
            ),
            "JSON number",
        ),
        (
            lambda record: record["visual_telemetry"]["tip_servo"].__setitem__(
                "commanded_arm_target_q_rad", (0.0,) * 6
            ),
            "JSON array",
        ),
        (
            lambda record: record["visual_telemetry"]["tip_servo"].__setitem__(
                "commanded_jaw_target_rad", True
            ),
            "JSON number",
        ),
        (
            lambda record: record["visual_telemetry"]["tip_servo"].__setitem__(
                "arm_owner", 1
            ),
            "arm_owner",
        ),
        (
            lambda record: record["visual_telemetry"][
                "pickup_policy_close_intent"
            ].__setitem__("action_source", 1),
            "action source",
        ),
    ],
)
def test_schema_v6_negative_control_rejects_coercible_telemetry_types(
    mutate,
    message,
):
    sources = [
        "fresh_policy",
        "fresh_policy",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "fresh_policy",
    ]
    records = [{
        "event": "control_acquired",
        "hold_target": [0.0] * 6 + [0.30926513671875],
    }, *[
        _open_only_command_cycle(frame_id, source)
        for frame_id, source in zip(range(487, 495), sources, strict=True)
    ]]
    mutate(records[1])

    with pytest.raises(audit.VisualLatchAuditInputError, match=message):
        audit._audit_tip_probe_open_only_negative_control_records(
            records,
            config=audit._open_only_tip_servo_config(),
        )


def test_schema_v6_negative_control_cannot_pass_with_a_tip_jaw_command():
    sources = [
        "fresh_policy",
        "fresh_policy",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "inference_hold",
        "fresh_policy",
    ]
    records = [{
        "event": "control_acquired",
        "hold_target": [0.0] * 6 + [0.30926513671875],
    }, *[
        _open_only_command_cycle(frame_id, source)
        for frame_id, source in zip(range(487, 495), sources, strict=True)
    ]]
    records[1]["visual_telemetry"]["tip_servo"][
        "commanded_jaw_target_rad"
    ] = 0.31

    report = audit._audit_tip_probe_open_only_negative_control_records(
        records,
        config=audit._open_only_tip_servo_config(),
    )

    assert report["actual_tip_servo_jaw_command_cycles"] == 1
    assert report["negative_control_pass"] is False


def test_schema_v6_generator_binds_policy_projection_and_real_arm6_replay(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 30,
        "features": _original_7d_features(),
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    video_path = dataset_root / "synthetic-wrist.mp4"
    video_path.write_bytes(b"not-decoded-by-this-test")

    rows, evidences = _tip_probe_rows_and_evidence()
    for index, row in enumerate(rows):
        row.update({"episode_index": 0, "frame_index": index})
    labels = audit.PhaseLabels(2, 8, 14, 18)
    metadata = {0: {"length": len(rows)}}
    load_info = {**info, "total_episodes": 1}
    visual_config = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=audit._open_only_tip_servo_config(),
    )
    training_digest = "a" * 64
    implementation_manifest = audit.visual_phase_latch_implementation_manifest()
    visual_profile = audit.visual_phase_latch_provenance(visual_config)
    source_binding = {
        "training_input_digest": training_digest,
        "visual_profile": visual_profile,
        "visual_implementation_manifest": implementation_manifest,
        "audit_script_sha256": implementation_manifest[
            "offline_visual_audit"
        ]["sha256"],
        **_schema6_policy_source_binding(
            source_repo_id="local/schema6-roundtrip",
            source_root=str(dataset_root.absolute()),
            upstream_digest=training_digest,
        ),
    }

    close_flags = [True] * 20
    held_flags = [False] * 20
    held_flags[8:] = [True] * 12
    blue_flags = [False] * 20
    blue_flags[10:] = [True] * 10
    empty_reasons = [()] * 20
    detected = {
        "close_flags": close_flags,
        "held_flags": held_flags,
        "blue_flags": blue_flags,
        "close_reasons": empty_reasons,
        "held_reasons": empty_reasons,
        "blue_reasons": empty_reasons,
        "evidences": evidences,
        "frame_status_counts": {"fresh_unique": 20},
        "red_contours": [
            {"positive_contour_count": 0, "ambiguity_candidates": []}
            for _ in range(20)
        ],
    }
    runtime_proxy = {
        "close_gate_flags": close_flags,
        "first_policy_close_intent_frame": 8,
        "policy_close_intent_timeout_frames": 15,
        "close_latch_frame": 8,
        "arm_speed_rad_s_at_close_latch": 0.0,
        "contact_confirm_frame": 8,
        "contact_gap_rad_at_confirmation": 0.04,
        "simulated_command_ee_rad_at_contact_confirmation": 0.32,
        "jaw_speed_rad_s_at_contact_confirmation": 0.0,
    }
    state_machine = {
        "ordered_complete": True,
        "fault_free": True,
        "offsets_frames": {
            "grasp_latched_from_pickup_close": 0,
            "grasp_confirmed_from_pickup_close": 1,
            "release_unlatched_from_release_open": 0,
            "release_confirmed_from_release_open": 1,
            "home_closing_started_from_return_close": 0,
            "task_done_from_return_close": 1,
        },
    }

    source_capture_calls = []

    def capture_source(*_args, **_kwargs):
        source_capture_calls.append(True)
        return source_binding

    monkeypatch.setattr(audit, "_capture_source_binding", capture_source)
    monkeypatch.setattr(
        audit,
        "_load_dataset",
        lambda *_args, **_kwargs: (load_info, metadata, {0: rows}),
    )
    monkeypatch.setattr(
        audit,
        "reconstruct_phase_labels",
        lambda *_args, **_kwargs: labels,
    )
    monkeypatch.setattr(
        audit,
        "_wrist_video_path",
        lambda *_args, **_kwargs: video_path,
    )
    monkeypatch.setattr(
        audit,
        "_detect_episode",
        lambda *_args, **_kwargs: detected,
    )
    monkeypatch.setattr(
        audit,
        "_runtime_gate_proxies",
        lambda *_args, **_kwargs: runtime_proxy,
    )
    monkeypatch.setattr(
        audit,
        "_state_machine_replay",
        lambda *_args, **_kwargs: state_machine,
    )
    monkeypatch.setattr(
        audit,
        "_load_fk_model",
        lambda *_args, **_kwargs: (
            object(),
            {
                "path": "/synthetic/firefly.urdf",
                "size": 1,
                "sha256": "b" * 64,
                "joint_names": list(audit.ARM_JOINT_NAMES),
                "tcp_tool_offset_m": list(audit.TCP_TOOL_OFFSET_M),
            },
        ),
    )
    monkeypatch.setattr(
        audit,
        "_contour_window_summary",
        lambda diagnostics: {"frames": len(diagnostics)},
    )

    generated = audit.audit_firefly_y6_visual_latch(
        dataset_root,
        expected_episodes=1,
        dataset_repo_id="local/schema6-roundtrip",
        controller_contract=tmp_path / "controller-contract.json",
        expected_task="synthetic roundtrip",
        arm_urdf_path=tmp_path / "firefly.urdf",
        config=visual_config,
        tip_probe_negative_control_path=(
            audit.DEFAULT_TIP_SERVO_OPEN_ONLY_NEGATIVE_CONTROL
        ),
        checkpoint=SCHEMA6_CHECKPOINT,
    )
    report = json.loads(audit._strict_json(generated))

    assert report["safe_for_deployment"] is True, report["summary"]["errors"]
    assert source_capture_calls == [True, True]
    assert report["schema_version"] == 6
    assert report["dataset"]["repo_id"] == "local/schema6-roundtrip"
    assert report["source_binding"] == source_binding
    assert report["safe_for_deployment"] is True
    assert report["tip_probe"]["demonstration_validation_role"] == (
        "observational_validation_not_deployment_positive_control"
    )
    assert "The 1 demonstration episode is" in report["tip_probe"]["limitation"]
    assert report["tip_probe"]["negative_control"][
        "runtime_authority_replay"
    ]["runtime_authority_replay_pass"] is True
    runtime = report["tip_probe"]["negative_control"][
        "runtime_authority_replay"
    ]
    assert runtime["policy_role"] == "arm6_seeking"
    assert runtime["raw_policy_action_dimension"] == 6
    assert runtime["raw_policy_jaw_telemetry_absent"] is True
    assert "adversarial_policy_jaw_target_rad" not in runtime
    assert report["tip_probe"]["recorded_closed_capture_observed_episodes"] == 0



@pytest.mark.parametrize("mode", [TipServoMode.DISABLED, TipServoMode.ADVISORY])
def test_non_execute_visual_audit_keeps_schema_v2(mode):
    config = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(mode=mode),
    )

    assert audit._audit_schema_version(config) == 2


def test_schema6_source_contract_accepts_only_original_7d_jaw_evidence():
    info = {"features": _original_7d_features()}

    audit._require_original_fullcycle_v4_7d_source(info)


@pytest.mark.parametrize("feature_name", ["action", "observation.state"])
def test_schema6_source_contract_explicitly_rejects_arm_only_6d_projection(
    feature_name,
):
    features = _original_7d_features()
    features[feature_name] = {
        "dtype": "float32",
        "shape": [6],
        "names": list(audit.FULLCYCLE_V4_ACTION_STATE_NAMES[:6]),
    }

    with pytest.raises(
        audit.VisualLatchAuditInputError,
        match="original 7-D.*6-D arm-only projection is policy-only",
    ):
        audit._require_original_fullcycle_v4_7d_source({"features": features})


def test_schema6_audit_rejects_6d_root_before_negative_control_or_attestation(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "arm-only"
    (root / "meta").mkdir(parents=True)
    features = _original_7d_features()
    for name in ("action", "observation.state"):
        features[name] = {
            "dtype": "float32",
            "shape": [6],
            "names": list(audit.FULLCYCLE_V4_ACTION_STATE_NAMES[:6]),
        }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 1,
                "fps": 30,
                "features": features,
            }
        ),
        encoding="utf-8",
    )

    def forbidden_capture(*_args, **_kwargs):
        pytest.fail("attestation must not run for a 6-D visual source")

    monkeypatch.setattr(audit, "_capture_source_binding", forbidden_capture)

    with pytest.raises(
        audit.VisualLatchAuditInputError,
        match="original 7-D.*6-D arm-only projection is policy-only",
    ):
        audit.audit_firefly_y6_visual_latch(
            root,
            expected_episodes=1,
            config=VisualPhaseLatchConfig(
                enabled=True,
                tip_servo=audit._open_only_tip_servo_config(),
            ),
            checkpoint=SCHEMA6_CHECKPOINT,
            tip_probe_negative_control_path=tmp_path / "must-not-be-read.jsonl",
        )


def test_schema6_arm6_binding_delegates_to_consumer_and_preserves_exact_json():
    binding = audit._capture_arm6_source_binding(
        SCHEMA6_CHECKPOINT,
        capture_factory=_Schema6Capture,
    )

    assert binding == _schema6_policy_source_binding()


def test_schema6_default_consumer_entrypoint_is_lazy_and_fixture_injectable(
    monkeypatch,
):
    from lerobot_robot_hexbase import firefly_arm6_attestation as consumer

    monkeypatch.setattr(
        consumer,
        "capture_firefly_arm6_attestation",
        _Schema6Capture,
    )

    binding = audit._capture_arm6_source_binding(SCHEMA6_CHECKPOINT)

    assert binding == _schema6_policy_source_binding()


@pytest.mark.parametrize(
    ("checkpoint", "selected_checkpoint", "selected_step", "binding", "match"),
    [
        (
            Path("relative/checkpoint"),
            SCHEMA6_CHECKPOINT,
            20_000,
            _schema6_policy_source_binding(),
            "explicit absolute",
        ),
        (
            Path("/synthetic/wrong/pretrained_model"),
            SCHEMA6_CHECKPOINT,
            20_000,
            _schema6_policy_source_binding(),
            "exactly equal",
        ),
        (
            SCHEMA6_CHECKPOINT,
            SCHEMA6_CHECKPOINT,
            10_000,
            _schema6_policy_source_binding(),
            "exactly 20000",
        ),
        (
            SCHEMA6_CHECKPOINT,
            SCHEMA6_CHECKPOINT,
            20_000,
            {"policy_attestation": {}},
            "contain exactly",
        ),
        (
            SCHEMA6_CHECKPOINT,
            SCHEMA6_CHECKPOINT,
            20_000,
            {
                "policy_attestation": {"bad_tuple": (1, 2)},
                "projection_binding": {},
            },
            "non-JSON value",
        ),
    ],
)
def test_schema6_arm6_binding_fails_closed_on_path_step_or_shape_drift(
    checkpoint,
    selected_checkpoint,
    selected_step,
    binding,
    match,
):
    capture = SimpleNamespace(
        selected_checkpoint=selected_checkpoint,
        selected_step=selected_step,
        source_binding=lambda: binding,
    )

    with pytest.raises(audit.VisualLatchAuditInputError, match=match):
        audit._capture_arm6_source_binding(
            checkpoint,
            capture_factory=lambda: capture,
        )


def test_schema6_source_binding_merges_consumer_policy_and_projection(monkeypatch):
    monkeypatch.setattr(
        audit,
        "capture_training_inputs",
        lambda *args, **kwargs: SimpleNamespace(digest="a" * 64),
    )
    monkeypatch.setattr(
        audit,
        "visual_phase_latch_provenance",
        lambda _config: {"__type__": "schema6-profile", "fields": {}},
    )
    manifest = {"offline_visual_audit": {"sha256": "b" * 64}}
    monkeypatch.setattr(
        audit, "visual_phase_latch_implementation_manifest", lambda: manifest
    )

    binding = audit._capture_source_binding(
        Path("/dataset"),
        dataset_repo_id="local/firefly_y6_fullcycle_v4",
        controller_contract=Path("/contract.json"),
        expected_task="task",
        config=VisualPhaseLatchConfig(
            enabled=True,
            tip_servo=audit._open_only_tip_servo_config(),
        ),
        checkpoint=SCHEMA6_CHECKPOINT,
        arm6_attestation_capture=_Schema6Capture,
    )

    assert binding == {
        "training_input_digest": "a" * 64,
        "visual_profile": {"__type__": "schema6-profile", "fields": {}},
        "visual_implementation_manifest": manifest,
        "audit_script_sha256": "b" * 64,
        **_schema6_policy_source_binding(),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "local/wrong-source"),
        ("root", "/wrong/source"),
        ("upstream_training_input_digest", "f" * 64),
    ],
)
def test_schema6_source_binding_rejects_projection_source_drift(
    monkeypatch,
    field,
    value,
):
    monkeypatch.setattr(
        audit,
        "capture_training_inputs",
        lambda *args, **kwargs: SimpleNamespace(digest="a" * 64),
    )
    monkeypatch.setattr(
        audit,
        "visual_phase_latch_provenance",
        lambda _config: {"__type__": "schema6-profile", "fields": {}},
    )
    monkeypatch.setattr(
        audit,
        "visual_phase_latch_implementation_manifest",
        lambda: {"offline_visual_audit": {"sha256": "b" * 64}},
    )
    policy_binding = _schema6_policy_source_binding()
    policy_binding["projection_binding"]["source"][field] = value
    capture = SimpleNamespace(
        selected_checkpoint=SCHEMA6_CHECKPOINT,
        selected_step=20_000,
        source_binding=lambda: policy_binding,
    )

    with pytest.raises(
        audit.VisualLatchAuditInputError,
        match="does not match the audited original 7-D source",
    ):
        audit._capture_source_binding(
            Path("/dataset"),
            dataset_repo_id="local/firefly_y6_fullcycle_v4",
            controller_contract=Path("/contract.json"),
            expected_task="task",
            config=VisualPhaseLatchConfig(
                enabled=True,
                tip_servo=audit._open_only_tip_servo_config(),
            ),
            checkpoint=SCHEMA6_CHECKPOINT,
            arm6_attestation_capture=lambda: capture,
        )


def test_source_binding_uses_exact_manifest_audit_digest(monkeypatch):
    monkeypatch.setattr(
        audit,
        "capture_training_inputs",
        lambda *args, **kwargs: SimpleNamespace(digest="a" * 64),
    )
    monkeypatch.setattr(
        audit,
        "visual_phase_latch_provenance",
        lambda config: {"__type__": "profile", "fields": {"enabled": config.enabled}},
    )
    manifest = {"offline_visual_audit": {"sha256": "b" * 64}}
    monkeypatch.setattr(
        audit, "visual_phase_latch_implementation_manifest", lambda: manifest
    )

    binding = audit._capture_source_binding(
        Path("/dataset"),
        dataset_repo_id="local/example",
        controller_contract=Path("/contract.json"),
        expected_task="task",
        config=VisualPhaseLatchConfig(enabled=True),
    )

    assert binding == {
        "training_input_digest": "a" * 64,
        "visual_profile": {"__type__": "profile", "fields": {"enabled": True}},
        "visual_implementation_manifest": manifest,
        "audit_script_sha256": "b" * 64,
    }


def test_detector_config_uses_exact_typed_visual_provenance():
    config = VisualPhaseLatchConfig(
        enabled=True,
        tip_servo=TipServoConfig(mode=TipServoMode.EXECUTE),
    )
    profile = audit.visual_phase_latch_provenance(config)

    detector = audit._detector_config_from_source_binding(
        {"visual_profile": profile}
    )

    assert detector == profile["fields"]
    assert detector["tip_servo"]["__type__"].endswith(".TipServoConfig")
    assert detector["tip_servo"]["fields"]["mode"] == "execute"
    assert "mode" not in detector["tip_servo"]


@pytest.mark.parametrize(
    "binding",
    ({}, {"visual_profile": {}}, {"visual_profile": {"fields": []}}),
)
def test_detector_config_rejects_untyped_source_binding(binding):
    with pytest.raises(
        audit.VisualLatchAuditInputError,
        match="visual_profile",
    ):
        audit._detector_config_from_source_binding(binding)


def test_cli_defaults_pin_fullcycle_v4_dataset_and_episode_count():
    args = audit.parse_args([])

    assert args.root == audit.DEFAULT_DATASET_ROOT
    assert args.expected_episodes == 50
    assert args.jaw_crossing_rad == pytest.approx(0.370)
    assert args.dataset_repo_id == audit.DEFAULT_DATASET_REPO_ID
    assert args.controller_contract == audit.DEFAULT_CONTROLLER_CONTRACT
    assert args.arm_urdf == audit.DEFAULT_ARM_URDF_PATH
    assert args.checkpoint is None
    assert (
        args.tip_probe_negative_control
        == audit.DEFAULT_TIP_PROBE_NEGATIVE_CONTROL
    )
    assert args.tip_probe_mode == TipServoMode.DISABLED.value
    assert args.tip_probe_profile == TipServoProbeProfile.BOUNDED_DESCENT.value


def test_cli_can_bind_the_read_only_audit_to_execute_mode_provenance():
    args = audit.parse_args(["--tip-probe-mode", "execute"])

    assert args.tip_probe_mode == TipServoMode.EXECUTE.value


def test_cli_accepts_explicit_schema6_checkpoint_without_resolving_aliases():
    args = audit.parse_args(["--checkpoint", str(SCHEMA6_CHECKPOINT)])

    assert args.checkpoint == SCHEMA6_CHECKPOINT


def test_main_returns_input_error_as_strict_json(tmp_path, capsys):
    status = audit.main(["--root", str(tmp_path / "missing")])
    report = json.loads(capsys.readouterr().out)

    assert status == 2
    assert report["schema_version"] == audit.VISUAL_LATCH_AUDIT_SCHEMA_VERSION == 2
    assert report["safe_for_deployment"] is False
    assert report["hardware_completion_validated"] is False
    assert report["closed_loop_convergence_claimed"] is False
    assert report["summary"]["verdict"] == "error"
    assert report["summary"]["errors"][0]["code"] == "audit_input_error"


def test_main_execute_mode_input_error_uses_schema_v3(tmp_path, capsys):
    status = audit.main(
        [
            "--root",
            str(tmp_path / "missing"),
            "--tip-probe-mode",
            "execute",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert status == 2
    assert report["schema_version"] == audit.TIP_SERVO_PROBE_AUDIT_SCHEMA_VERSION == 3
    assert report["safe_for_deployment"] is False
    assert report["hardware_completion_validated"] is False
    assert report["closed_loop_convergence_claimed"] is False
def test_main_open_only_input_error_uses_schema_v6(tmp_path, capsys):
    status = audit.main(
        [
            "--root",
            str(tmp_path / "missing"),
            "--tip-probe-mode",
            "execute",
            "--tip-probe-profile",
            "open_only",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert status == 2
    assert (
        report["schema_version"]
        == audit.TIP_SERVO_OPEN_ONLY_AUDIT_SCHEMA_VERSION
        == 6
    )
    assert report["safe_for_deployment"] is False
    assert report["hardware_completion_validated"] is False
    assert report["closed_loop_convergence_claimed"] is False


def test_main_schema6_open_only_requires_checkpoint_before_dataset_read(
    tmp_path,
    capsys,
):
    status = audit.main(
        [
            "--root",
            str(tmp_path),
            "--tip-probe-mode",
            "execute",
            "--tip-probe-profile",
            "open_only",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert status == 2
    assert report["schema_version"] == 6
    assert "requires --checkpoint" in report["summary"]["errors"][0]["message"]
