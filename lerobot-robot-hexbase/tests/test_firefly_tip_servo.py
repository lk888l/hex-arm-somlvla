from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any

import numpy as np
import pytest

from lerobot_robot_hexbase.firefly_tip_servo import (
    DlsIkConfig,
    HARD_MAX_DESCENT_STEP_M,
    HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS,
    HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD,
    HARD_MAX_TOTAL_DESCENT_M,
    TipServoConfig,
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
    solve_full_pose_dls,
)


Q0 = (0.20, 0.10, 1.00, 0.10, -0.10, 0.20)


def _rx(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _ry(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rz(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _toy_fk(q: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float64)
    return values[:3].copy(), _rx(values[3]) @ _ry(values[4]) @ _rz(values[5])


def _toy_float32_fk(q: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float32)
    position = values[:3].copy()
    rotation = (
        _rx(float(values[3])) @ _ry(float(values[4])) @ _rz(float(values[5]))
    ).astype(np.float32)
    return position, rotation


def _singular_fk(q: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float64)
    return values[:3].copy(), np.eye(3)


def _observation(
    frame_id: int,
    *,
    evaluated_at_ns: int | None = None,
    status: TipServoFrameStatus = TipServoFrameStatus.FRESH_UNIQUE,
    candidate_count: int = 1,
    centroid: tuple[float, float] | None = (0.46, 0.50),
    area: float | None = 0.020,
    q: tuple[float, ...] = Q0,
    reference_q: tuple[float, ...] = Q0,
    jaw: float = 0.41,
    jaw_reference: float | None = None,
    arm_speed: float = 0.0,
    jaw_speed: float = 0.0,
) -> TipServoObservation:
    return TipServoObservation(
        frame_id=frame_id,
        evaluated_at_ns=(
            frame_id * 33_333_333 if evaluated_at_ns is None else evaluated_at_ns
        ),
        frame_status=status,
        candidate_count=candidate_count,
        red_centroid_norm=centroid,
        red_area_fraction=area,
        arm_q_rad=q,
        arm_reference_q_rad=reference_q,
        jaw_reference_rad=jaw if jaw_reference is None else jaw_reference,
        jaw_rad=jaw,
        arm_speed_rad_s=arm_speed,
        jaw_speed_rad_s=jaw_speed,
    )


def _config(**overrides: Any) -> TipServoConfig:
    values: dict[str, Any] = {"mode": TipServoMode.EXECUTE}
    values.update(overrides)
    return TipServoConfig(**values)


def _open_only_config(**overrides: Any) -> TipServoConfig:
    values: dict[str, Any] = {
        "mode": TipServoMode.EXECUTE,
        "probe_profile": TipServoProbeProfile.OPEN_ONLY,
        "coarse_capture_jaw_rad": (0.285, 0.325),
        "descent_step_m": 0.0,
        "max_total_descent_m": 0.0,
    }
    values.update(overrides)
    return TipServoConfig(**values)


def _active_state(
    *,
    phase: TipServoPhase = TipServoPhase.DESCENDING,
    q: tuple[float, ...] = Q0,
    descent_m: float = 0.0,
    peak_area: float = 0.020,
    frame_id: int = 10,
    evaluated_at_ns: int = 1_000_000_000,
) -> TipServoState:
    position, rotation = _toy_fk(Q0)
    return TipServoState(
        phase=phase,
        takeover_q_rad=Q0,
        takeover_position_m=tuple(float(value) for value in position),
        takeover_rotation=tuple(
            tuple(float(value) for value in row) for row in rotation
        ),
        last_arm_target_q_rad=q,
        takeover_jaw_target_rad=0.41,
        takeover_started_at_ns=max(0, evaluated_at_ns - 500_000_000),
        open_commanded=True,
        commanded_descent_m=descent_m,
        coarse_streak=6,
        open_streak=3,
        peak_red_area_fraction=peak_area,
        last_fresh_frame_id=frame_id,
        last_evaluated_at_ns=evaluated_at_ns,
    )


def _reach_descending(
    config: TipServoConfig | None = None,
) -> tuple[TipServoState, int]:
    config = _config() if config is None else config
    state = TipServoState()
    frame_id = 0
    for _ in range(config.takeover_confirm_frames):
        frame_id += 1
        state, decision = advance_tip_servo(
            state,
            _observation(frame_id),
            fk=_toy_fk,
            config=config,
        )
    assert state.phase is TipServoPhase.OPENING
    assert decision.event == "coarse_takeover"
    for _ in range(config.open_confirm_frames):
        frame_id += 1
        state, decision = advance_tip_servo(
            state,
            _observation(frame_id),
            fk=_toy_fk,
            config=config,
        )
    assert state.phase is TipServoPhase.DESCENDING
    assert decision.event == "open_stable"
    return state, frame_id


def _assert_primitive_tree(value: Any) -> None:
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for item in value:
            _assert_primitive_tree(item)
        return
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_primitive_tree(item)
        return
    pytest.fail(f"non-primitive telemetry value: {type(value).__name__}: {value!r}")


def test_default_profile_is_disabled_probe_only_and_uses_data_derived_gates():
    config = TipServoConfig()

    assert config.mode is TipServoMode.DISABLED
    assert config.probe_profile is TipServoProbeProfile.BOUNDED_DESCENT
    assert config.allow_close is False
    assert config.coarse_capture_jaw_rad is None
    assert config.coarse_x_norm == (0.42, 0.50)
    assert config.coarse_y_norm == (0.42, 0.56)
    assert config.coarse_area_fraction == (0.017, 0.029)
    assert config.takeover_confirm_frames == 6
    assert config.close_ready_x_norm == (0.43, 0.48)
    assert config.close_ready_y_norm == (0.51, 0.60)
    assert config.close_ready_area_fraction == (0.0295, 0.040)
    assert config.close_ready_confirm_frames == 9
    assert config.probe_success_confirm_frames == 6
    assert config.descent_step_m == 0.002
    assert config.max_total_descent_m == 0.025
    assert config.max_takeover_reference_error_rad == 0.035
    assert config.opening_timeout_ms == 1500.0
    assert config.probe_success_area_fraction == 0.020


def test_config_strictly_enforces_motion_hard_limits_and_nested_visual_gates():
    assert HARD_MAX_DESCENT_STEP_M == 0.002
    assert HARD_MAX_TOTAL_DESCENT_M == 0.070
    assert TipServoConfig(max_total_descent_m=0.070).max_total_descent_m == 0.070

    with pytest.raises(TipServoError, match="0.002-m hard limit"):
        TipServoConfig(descent_step_m=0.0020001)
    with pytest.raises(TipServoError, match="0.070-m hard limit"):
        TipServoConfig(max_total_descent_m=0.0700001)
    with pytest.raises(TipServoError, match="must not exceed max_total"):
        TipServoConfig(descent_step_m=0.002, max_total_descent_m=0.001)
    with pytest.raises(TipServoError, match="coarse_x_norm must fit"):
        TipServoConfig(coarse_x_norm=(0.39, 0.50))
    with pytest.raises(TipServoError, match="coarse_area_fraction must end"):
        TipServoConfig(coarse_area_fraction=(0.017, 0.030))
    with pytest.raises(TipServoError, match="allow_close must be boolean"):
        TipServoConfig(allow_close=1)  # type: ignore[arg-type]
    with pytest.raises(TipServoError, match="mode is unsupported"):
        TipServoConfig(mode="shadow")  # type: ignore[arg-type]


def test_open_only_profile_is_explicit_zero_descent_and_fingerprintable():
    config = TipServoConfig(
        mode="execute",
        probe_profile="open_only",
        coarse_capture_jaw_rad=(0.285, 0.325),
        descent_step_m=0,
        max_total_descent_m=0,
    )

    assert config.mode is TipServoMode.EXECUTE
    assert config.probe_profile is TipServoProbeProfile.OPEN_ONLY
    assert config.coarse_capture_jaw_rad == (0.285, 0.325)
    assert config.max_total_descent_m == 0.0
    assert config.allow_close is False


def test_open_only_profile_hard_binds_seeking_timeout_and_open_target():
    config = _open_only_config()

    assert config.seeking_timeout_ms == pytest.approx(
        HARD_MAX_OPEN_ONLY_SEEKING_TIMEOUT_MS
    )
    assert config.pregrasp_open_target_rad == pytest.approx(0.41)

    for timeout_ms in (19_999.999, 20_000.001, 30_000.0):
        with pytest.raises(TipServoError, match="exactly 20000 ms"):
            _open_only_config(seeking_timeout_ms=timeout_ms)
    with pytest.raises(TipServoError, match="0.41-rad hard limit"):
        _open_only_config(pregrasp_open_target_rad=0.410001)


def test_open_only_jaw_resolver_holds_then_opens_monotonically_by_one_step():
    config = _open_only_config()
    kwargs = {
        "capture_jaw_target_rad": 0.305,
        "config": config,
        "ee_step_limit_rad": 0.01,
    }

    assert resolve_open_only_jaw_target(
        **kwargs,
        previous_accepted_jaw_target_rad=0.305,
        open_commanded=False,
    ) == pytest.approx(0.305)
    assert resolve_open_only_jaw_target(
        **kwargs,
        previous_accepted_jaw_target_rad=0.305,
        open_commanded=True,
    ) == pytest.approx(0.315)
    assert resolve_open_only_jaw_target(
        **kwargs,
        previous_accepted_jaw_target_rad=0.405,
        open_commanded=True,
    ) == pytest.approx(0.41)

    with pytest.raises(TipServoError, match="before visual open authorization"):
        resolve_open_only_jaw_target(
            **kwargs,
            previous_accepted_jaw_target_rad=0.306,
            open_commanded=False,
        )
    for malicious_previous in (0.10, 0.42):
        with pytest.raises(TipServoError, match="no-close envelope"):
            resolve_open_only_jaw_target(
                **kwargs,
                previous_accepted_jaw_target_rad=malicious_previous,
                open_commanded=True,
            )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"probe_profile": "hover"}, "probe_profile is unsupported"),
        (
            {
                "probe_profile": TipServoProbeProfile.OPEN_ONLY,
                "descent_step_m": 0.0,
                "max_total_descent_m": 0.0,
            },
            "requires coarse_capture_jaw_rad",
        ),
        (
            {
                "probe_profile": TipServoProbeProfile.OPEN_ONLY,
                "coarse_capture_jaw_rad": (0.285, 0.325),
                "descent_step_m": 0.0,
            },
            "requires max_total_descent_m=0",
        ),
        (
            {
                "probe_profile": TipServoProbeProfile.OPEN_ONLY,
                "coarse_capture_jaw_rad": (0.285, 0.325),
                "descent_step_m": 0.0,
                "max_total_descent_m": 0.0,
                "allow_close": True,
            },
            "cannot allow_close",
        ),
        (
            {
                "probe_profile": TipServoProbeProfile.OPEN_ONLY,
                "coarse_capture_jaw_rad": (0.285, 0.38),
                "descent_step_m": 0.0,
                "max_total_descent_m": 0.0,
            },
            "must remain below pregrasp_open_ready_rad",
        ),
        (
            {
                "probe_profile": TipServoProbeProfile.OPEN_ONLY,
                "coarse_capture_jaw_rad": (0.285, 0.325),
                "max_total_descent_m": 0.0,
            },
            "requires descent_step_m=0",
        ),
        (
            {"descent_step_m": 0.0},
            "bounded_descent probe_profile requires positive descent_step_m",
        ),
        (
            {"max_total_descent_m": 0.0},
            "bounded_descent probe_profile requires positive",
        ),
        (
            {"max_takeover_reference_error_rad": 0.0350001},
            "0.035-rad hard limit",
        ),
        ({"opening_timeout_ms": 0.0}, "finite and positive"),
    ],
)
def test_probe_profile_rejects_illegal_authority_combinations(overrides, message):
    with pytest.raises(TipServoError, match=message):
        TipServoConfig(**overrides)


def test_closed_capture_interval_validation_preserves_legacy_open_capture_default():
    assert TipServoConfig().coarse_capture_jaw_rad is None
    assert TipServoConfig(
        coarse_capture_jaw_rad=(0.285, 0.325)
    ).coarse_capture_jaw_rad == (0.285, 0.325)

    for invalid in ((0.325, 0.285), (0.3, 0.3), (-0.1, 0.3), (0.2, math.inf)):
        with pytest.raises(TipServoError, match="coarse_capture_jaw_rad"):
            TipServoConfig(coarse_capture_jaw_rad=invalid)


def test_observation_requires_a_finite_six_axis_accepted_reference():
    with pytest.raises(TipServoError, match="arm_reference_q_rad.*exactly 6"):
        replace(_observation(1), arm_reference_q_rad=(0.0,) * 5)
    with pytest.raises(TipServoError, match="arm_reference_q_rad.*NaN"):
        replace(_observation(1), arm_reference_q_rad=(*Q0[:5], math.nan))


def test_ik_config_requires_six_axis_limits_margins_and_step_caps():
    with pytest.raises(TipServoError, match="exactly 6"):
        DlsIkConfig(joint_lower_rad=(0.0,) * 5)  # type: ignore[arg-type]
    with pytest.raises(TipServoError, match="invalid joint limits"):
        DlsIkConfig(joint_lower_rad=(0.0,) * 6, joint_upper_rad=(0.0,) * 6)
    with pytest.raises(TipServoError, match="must be in"):
        DlsIkConfig(joint_step_limit_rad=(0.05,) * 6)
    with pytest.raises(TipServoError, match="finite_difference_rad"):
        DlsIkConfig(finite_difference_rad=0.02)


def test_observation_normalizes_status_and_rejects_invalid_red_or_time_data():
    observation = replace(_observation(1), frame_status="fresh_unique")
    assert observation.frame_status is TipServoFrameStatus.FRESH_UNIQUE
    assert observation.arm_q_rad == Q0

    with pytest.raises(TipServoError, match="positive candidate_count requires"):
        _observation(1, centroid=None, area=None)
    with pytest.raises(TipServoError, match=r"inside \[0, 1\]"):
        _observation(1, centroid=(1.1, 0.5))
    with pytest.raises(TipServoError, match="nonnegative integer"):
        _observation(1, evaluated_at_ns=-1)
    missing = _observation(1, candidate_count=0, centroid=None, area=None)
    assert missing.red_centroid_norm is None


def test_full_pose_dls_moves_only_base_z_and_preserves_xy_and_rotation():
    position, rotation = _toy_fk(Q0)
    desired = position.copy()
    desired[2] -= 0.002

    result = solve_full_pose_dls(Q0, desired, rotation, fk=_toy_fk)

    assert result.ok is True
    assert result.reason is None
    assert result.q_rad is not None
    solved_position, solved_rotation = _toy_fk(result.q_rad)
    assert solved_position[0] == pytest.approx(position[0], abs=1e-9)
    assert solved_position[1] == pytest.approx(position[1], abs=1e-9)
    assert solved_position[2] == pytest.approx(position[2] - 0.002, abs=1e-7)
    assert solved_rotation == pytest.approx(rotation, abs=1e-9)
    for before, after, limit in zip(
        Q0, result.q_rad, DlsIkConfig().joint_step_limit_rad, strict=True
    ):
        assert abs(after - before) <= limit + 1e-12


def test_default_finite_difference_is_stable_with_float32_fk_arithmetic():
    position, rotation = _toy_float32_fk(Q0)
    desired = position.astype(np.float64)
    desired[2] -= 0.002

    config = DlsIkConfig()
    result = solve_full_pose_dls(Q0, desired, rotation, fk=_toy_float32_fk, config=config)

    assert config.finite_difference_rad == pytest.approx(1e-3)
    assert result.ok is True
    assert result.position_residual_m is not None
    assert result.position_residual_m <= config.residual_position_m
    assert result.orientation_residual_rad is not None
    assert result.orientation_residual_rad <= config.residual_orientation_rad


def test_full_pose_dls_rejects_singular_fk_bad_rotation_and_fk_exception():
    position, rotation = _toy_fk(Q0)
    desired = position.copy()
    desired[2] -= 0.002

    singular = solve_full_pose_dls(Q0, desired, rotation, fk=_singular_fk)
    assert singular.ok is False
    assert singular.reason == "jacobian_condition_gate"
    assert singular.condition_number == math.inf

    def bad_rotation_fk(q: Any) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(q[:3]), np.ones((3, 3))

    bad = solve_full_pose_dls(Q0, desired, rotation, fk=bad_rotation_fk)
    assert bad.ok is False
    assert bad.reason == "fk_result_invalid"

    def raising_fk(_q: Any) -> Any:
        raise RuntimeError("no model")

    raised = solve_full_pose_dls(Q0, desired, rotation, fk=raising_fk)
    assert raised.ok is False
    assert raised.reason == "fk_exception_RuntimeError"


def test_full_pose_dls_fails_when_step_or_joint_limits_prevent_residual_gate():
    position, rotation = _toy_fk(Q0)
    desired = position.copy()
    desired[2] -= 0.002
    tiny_steps = DlsIkConfig(
        joint_step_limit_rad=(0.0001,) * 6,
        finite_difference_rad=1e-6,
    )

    limited = solve_full_pose_dls(Q0, desired, rotation, fk=_toy_fk, config=tiny_steps)
    assert limited.ok is False
    assert limited.reason == "joint_or_step_limit_residual"
    assert limited.q_rad is not None
    assert max(abs(after - before) for before, after in zip(Q0, limited.q_rad)) <= 0.0001 + 1e-12

    outside = list(Q0)
    outside[2] = 0.0
    result = solve_full_pose_dls(outside, desired, rotation, fk=_toy_fk)
    assert result.ok is False
    assert result.reason == "input_q_outside_inset_joint_limits"


def test_disabled_has_no_state_proposal_command_or_fk_evaluation():
    state = TipServoState()

    def forbidden_fk(_q: Any) -> Any:
        raise AssertionError("disabled mode must not evaluate FK")

    next_state, decision = advance_tip_servo(
        state,
        _observation(1),
        fk=forbidden_fk,
        config=TipServoConfig(),
    )

    assert next_state is state
    assert decision.mode is TipServoMode.DISABLED
    assert decision.intent is TipServoIntent.OBSERVE
    assert decision.proposed_arm_target_q_rad is None
    assert decision.commanded_arm_target_q_rad is None
    assert decision.proposed_jaw_target_rad is None
    assert decision.close_authorized is False


def test_explicit_mode_must_match_fingerprinted_config_mode():
    with pytest.raises(TipServoError, match="exactly match"):
        advance_tip_servo(
            TipServoState(),
            _observation(1),
            fk=_toy_fk,
            config=TipServoConfig(mode=TipServoMode.ADVISORY),
            mode=TipServoMode.EXECUTE,
        )


def test_advisory_advances_counterfactual_state_but_never_emits_commands():
    state = _active_state()
    config = TipServoConfig(mode=TipServoMode.ADVISORY)

    next_state, decision = advance_tip_servo(
        state,
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.019),
        fk=_toy_fk,
        config=config,
    )

    assert next_state is not state
    assert next_state.commanded_descent_m == pytest.approx(0.002)
    assert decision.mode is TipServoMode.ADVISORY
    assert decision.phase_after is state.phase
    assert decision.intent is TipServoIntent.DESCEND
    assert decision.event == "advisory_descent_proposed"
    assert decision.proposed_arm_target_q_rad is not None
    assert decision.commanded_arm_target_q_rad is None
    assert decision.commanded_jaw_target_rad is None
    assert decision.commanded_descent_m == 0.0
    assert decision.close_authorized is False


def test_advisory_accumulates_unique_takeover_streak_without_commands():
    config = TipServoConfig(mode=TipServoMode.ADVISORY)
    state = TipServoState()
    decision = None

    for frame_id in range(1, config.takeover_confirm_frames + 1):
        state, decision = advance_tip_servo(
            state,
            _observation(frame_id),
            fk=_toy_fk,
            config=config,
        )

    assert decision is not None
    assert state.phase is TipServoPhase.OPENING
    assert state.coarse_streak == config.takeover_confirm_frames
    assert decision.event == "advisory_coarse_takeover"
    assert decision.proposed_arm_target_q_rad == Q0
    assert decision.commanded_arm_target_q_rad is None
    assert decision.commanded_jaw_target_rad is None


def test_execute_requires_six_unique_coarse_frames_then_open_stability():
    config = _config()
    state = TipServoState()

    for frame_id in range(1, 6):
        state, decision = advance_tip_servo(
            state, _observation(frame_id), fk=_toy_fk, config=config
        )
        assert state.phase is TipServoPhase.SEEKING
        assert decision.commanded_arm_target_q_rad is None
    state, decision = advance_tip_servo(
        state, _observation(6), fk=_toy_fk, config=config
    )
    assert state.phase is TipServoPhase.OPENING
    assert state.takeover_position_m == pytest.approx(Q0[:3])
    assert decision.event == "coarse_takeover"
    assert decision.commanded_arm_target_q_rad == Q0
    assert decision.commanded_jaw_target_rad == pytest.approx(0.41)

    for frame_id in (7, 8):
        state, decision = advance_tip_servo(
            state, _observation(frame_id), fk=_toy_fk, config=config
        )
        assert state.phase is TipServoPhase.OPENING
        assert decision.intent is TipServoIntent.OPEN
    state, decision = advance_tip_servo(
        state, _observation(9), fk=_toy_fk, config=config
    )
    assert state.phase is TipServoPhase.DESCENDING
    assert decision.event == "open_stable"


def test_unauthorized_fresh_frames_hold_ready_streak_until_current_gate_is_authorized():
    config = _open_only_config(takeover_confirm_frames=3)
    accepted = (Q0[0] + 0.02, *Q0[1:])
    state = TipServoState()
    fk_inputs = []

    def recording_fk(q):
        fk_inputs.append(tuple(q))
        return _toy_fk(q)

    decisions = []
    for frame_id in range(1, 6):
        state, decision = advance_tip_servo(
            state,
            _observation(
                frame_id,
                jaw=0.305,
                reference_q=accepted,
            ),
            fk=recording_fk,
            config=config,
            takeover_authorized=False,
        )
        decisions.append(decision)

    assert state.phase is TipServoPhase.SEEKING
    assert state.coarse_streak == config.takeover_confirm_frames
    assert [decision.event for decision in decisions] == [
        "coarse_candidate",
        "coarse_candidate",
        "coarse_ready_waiting_authorization",
        "coarse_ready_waiting_authorization",
        "coarse_ready_waiting_authorization",
    ]
    assert all(decision.commanded_arm_target_q_rad is None for decision in decisions)
    assert [decision.commanded_jaw_target_rad for decision in decisions] == (
        pytest.approx([0.305] * len(decisions))
    )
    assert state.capture_jaw_target_rad == pytest.approx(0.305)
    assert fk_inputs == []

    state, takeover = advance_tip_servo(
        state,
        _observation(6, jaw=0.305, reference_q=accepted),
        fk=recording_fk,
        config=config,
        takeover_authorized=True,
    )

    assert state.phase is TipServoPhase.OPENING
    assert state.takeover_q_rad == Q0
    assert state.last_arm_target_q_rad == accepted
    assert state.takeover_jaw_target_rad == pytest.approx(0.305)
    assert state.open_commanded is False
    assert takeover.event == "coarse_takeover"
    assert takeover.intent is TipServoIntent.HOLD
    assert takeover.commanded_arm_target_q_rad == accepted
    assert takeover.commanded_jaw_target_rad == pytest.approx(0.305)
    assert fk_inputs == [Q0]


def test_open_only_binds_accepted_capture_immutably_and_faults_at_exact_20s():
    config = _open_only_config()
    started_at_ns = 1_000_000_000
    state, first = advance_tip_servo(
        TipServoState(),
        _observation(
            1,
            evaluated_at_ns=started_at_ns,
            status=TipServoFrameStatus.DUPLICATE,
            candidate_count=0,
            centroid=None,
            area=None,
            jaw=0.10,
            jaw_reference=0.305,
        ),
        fk=_toy_fk,
        config=config,
    )

    assert state.phase is TipServoPhase.SEEKING
    assert state.capture_jaw_target_rad == pytest.approx(0.305)
    assert state.seeking_started_at_ns == started_at_ns
    assert first.commanded_jaw_target_rad == pytest.approx(0.305)

    state, before_deadline = advance_tip_servo(
        state,
        _observation(
            2,
            evaluated_at_ns=started_at_ns + 19_999_999_999,
            status=TipServoFrameStatus.DUPLICATE,
            candidate_count=0,
            centroid=None,
            area=None,
            jaw=0.50,
            jaw_reference=0.320,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.SEEKING
    assert state.capture_jaw_target_rad == pytest.approx(0.305)
    assert before_deadline.commanded_jaw_target_rad == pytest.approx(0.305)

    state, deadline = advance_tip_servo(
        state,
        _observation(
            3,
            evaluated_at_ns=started_at_ns + 20_000_000_000,
            status=TipServoFrameStatus.DUPLICATE,
            candidate_count=0,
            centroid=None,
            area=None,
            jaw=0.01,
            jaw_reference=0.01,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "seeking_timeout"
    assert deadline.intent is TipServoIntent.FAULT_HOLD
    assert deadline.commanded_jaw_target_rad == pytest.approx(0.305)

    state, malicious_after_fault = advance_tip_servo(
        state,
        _observation(
            4,
            evaluated_at_ns=started_at_ns + 20_001_000_000,
            jaw=0.0,
            jaw_reference=0.0,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.capture_jaw_target_rad == pytest.approx(0.305)
    assert malicious_after_fault.commanded_jaw_target_rad == pytest.approx(0.305)


def test_authorization_never_bypasses_current_reference_error_gate():
    config = _config(takeover_confirm_frames=2)
    state = TipServoState()
    fk_calls = 0

    def recording_fk(q):
        nonlocal fk_calls
        fk_calls += 1
        return _toy_fk(q)

    for frame_id in (1, 2):
        state, _ = advance_tip_servo(
            state,
            _observation(frame_id),
            fk=recording_fk,
            config=config,
            takeover_authorized=False,
        )
    assert state.coarse_streak == 2

    outside = (Q0[0] + 0.036, *Q0[1:])
    state, rejected = advance_tip_servo(
        state,
        _observation(3, reference_q=outside),
        fk=recording_fk,
        config=config,
        takeover_authorized=True,
    )
    assert state.phase is TipServoPhase.SEEKING
    assert state.coarse_streak == 0
    assert rejected.event is None
    assert fk_calls == 0

    boundary = (Q0[0] + HARD_MAX_TAKEOVER_REFERENCE_ERROR_RAD, *Q0[1:])
    boundary_config = _config(takeover_confirm_frames=1)
    accepted_state, accepted = advance_tip_servo(
        TipServoState(),
        _observation(4, reference_q=boundary),
        fk=recording_fk,
        config=boundary_config,
    )
    assert accepted_state.phase is TipServoPhase.OPENING
    assert accepted.commanded_arm_target_q_rad == boundary
    assert fk_calls == 1


def test_takeover_authorized_requires_an_exact_boolean_even_when_disabled():
    with pytest.raises(TipServoError, match="takeover_authorized must be boolean"):
        advance_tip_servo(
            TipServoState(),
            _observation(1),
            fk=_toy_fk,
            config=TipServoConfig(),
            takeover_authorized=1,  # type: ignore[arg-type]
        )


def test_capture_jaw_interval_can_select_closed_jaw_without_changing_legacy_gate():
    legacy = _config(takeover_confirm_frames=1)
    rejected, _ = advance_tip_servo(
        TipServoState(),
        _observation(1, jaw=0.379),
        fk=_toy_fk,
        config=legacy,
    )
    assert rejected.phase is TipServoPhase.SEEKING

    accepted_legacy, _ = advance_tip_servo(
        TipServoState(),
        _observation(1, jaw=legacy.pregrasp_open_ready_rad),
        fk=_toy_fk,
        config=legacy,
    )
    assert accepted_legacy.phase is TipServoPhase.OPENING

    closed = _config(
        takeover_confirm_frames=1,
        coarse_capture_jaw_rad=(0.285, 0.325),
    )
    for jaw in (0.285, 0.305, 0.325):
        state, decision = advance_tip_servo(
            TipServoState(),
            _observation(1, jaw=jaw),
            fk=_toy_fk,
            config=closed,
        )
        assert state.phase is TipServoPhase.OPENING
        assert decision.commanded_jaw_target_rad == pytest.approx(jaw)
    outside_state, _ = advance_tip_servo(
        TipServoState(),
        _observation(1, jaw=0.326),
        fk=_toy_fk,
        config=closed,
    )
    assert outside_state.phase is TipServoPhase.SEEKING


def test_open_only_holds_closed_until_settled_then_opens_and_completes_without_ik(
    monkeypatch,
):
    config = _open_only_config(
        takeover_confirm_frames=1,
        open_confirm_frames=2,
    )
    accepted = (Q0[0] + 0.02, *Q0[1:])

    def forbidden_ik(*_args, **_kwargs):
        raise AssertionError("OPEN_ONLY must never invoke DLS IK")

    monkeypatch.setattr(
        "lerobot_robot_hexbase.firefly_tip_servo.solve_full_pose_dls",
        forbidden_ik,
    )

    state, takeover = advance_tip_servo(
        TipServoState(),
        _observation(1, jaw=0.305, reference_q=accepted),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.OPENING
    assert state.takeover_q_rad == Q0
    assert state.last_arm_target_q_rad == accepted
    assert state.open_commanded is False
    assert takeover.intent is TipServoIntent.HOLD
    assert takeover.commanded_arm_target_q_rad == accepted
    assert takeover.commanded_jaw_target_rad == pytest.approx(0.305)

    state, trailing = advance_tip_servo(
        state,
        _observation(2, q=Q0, reference_q=accepted, jaw=0.305),
        fk=_toy_fk,
        config=config,
    )
    assert state.open_commanded is False
    assert trailing.event == "takeover_settle_hold"
    assert trailing.commanded_arm_target_q_rad == accepted
    assert trailing.commanded_jaw_target_rad == pytest.approx(0.305)

    state, moving = advance_tip_servo(
        state,
        _observation(
            3,
            q=accepted,
            reference_q=accepted,
            jaw=0.305,
            arm_speed=config.max_arm_speed_rad_s + 0.001,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.open_commanded is False
    assert moving.event == "takeover_settle_hold"
    assert moving.commanded_jaw_target_rad == pytest.approx(0.305)

    state, opening = advance_tip_servo(
        state,
        _observation(4, q=accepted, reference_q=accepted, jaw=0.305),
        fk=_toy_fk,
        config=config,
    )
    assert state.open_commanded is True
    assert state.open_streak == 0
    assert opening.event == "opening_hold"
    assert opening.intent is TipServoIntent.OPEN
    assert opening.commanded_arm_target_q_rad == accepted
    assert opening.commanded_jaw_target_rad == pytest.approx(0.41)

    # OPEN_ONLY completion deliberately needs low arm speed, not the bounded
    # descent profile's 1-mrad target-settle gate.  A 10-mrad residual is
    # therefore permitted only after opening has irreversibly begun.
    slow_but_not_settled = (accepted[0] + 0.010, *accepted[1:])
    open_decisions = []
    for frame_id in (5, 6):
        state, decision = advance_tip_servo(
            state,
            _observation(
                frame_id,
                q=slow_but_not_settled,
                reference_q=accepted,
                jaw=0.39,
                arm_speed=0.05,
                jaw_speed=0.01,
            ),
            fk=_toy_fk,
            config=config,
        )
        open_decisions.append(decision)

    assert state.phase is TipServoPhase.PROBE_COMPLETE
    assert state.commanded_descent_m == 0.0
    assert open_decisions[-1].event == "open_only_probe_complete"
    assert open_decisions[-1].intent is TipServoIntent.OPEN
    assert open_decisions[-1].proposed_descent_step_m == 0.0
    assert open_decisions[-1].ik_result is None
    assert open_decisions[-1].would_close is False
    assert open_decisions[-1].close_authorized is False
    assert all(
        decision.commanded_arm_target_q_rad == accepted
        for decision in (takeover, trailing, moving, opening, *open_decisions)
    )

    # PROBE_COMPLETE is irreversible even if later evidence is missing and
    # its timestamp is out of order.  It continues to hold the accepted arm
    # reference and the open jaw without reacquiring motion or close authority.
    terminal_state = state
    state, terminal = advance_tip_servo(
        state,
        _observation(
            7,
            evaluated_at_ns=1,
            candidate_count=0,
            centroid=None,
            area=None,
            q=Q0,
            reference_q=accepted,
            jaw=0.20,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state is terminal_state
    assert state.phase is TipServoPhase.PROBE_COMPLETE
    assert terminal.event is None
    assert terminal.commanded_arm_target_q_rad == accepted
    assert terminal.commanded_jaw_target_rad == pytest.approx(0.41)
    assert terminal.proposed_descent_step_m == 0.0
    assert terminal.would_close is False
    assert terminal.close_authorized is False


@pytest.mark.parametrize("open_only", [False, True])
def test_opening_wall_time_deadline_faults_at_exact_boundary(open_only):
    if open_only:
        config = _open_only_config(
            takeover_confirm_frames=1,
            opening_timeout_ms=1500.0,
        )
        jaw = 0.305
    else:
        config = _config(
            takeover_confirm_frames=1,
            opening_timeout_ms=1500.0,
        )
        jaw = 0.41
    accepted = (Q0[0] + 0.02, *Q0[1:])
    start_ns = 1_000_000_000
    state, _ = advance_tip_servo(
        TipServoState(),
        _observation(
            1,
            evaluated_at_ns=start_ns,
            reference_q=accepted,
            jaw=jaw,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.OPENING
    assert state.takeover_started_at_ns == start_ns

    state, before_deadline = advance_tip_servo(
        state,
        _observation(
            2,
            evaluated_at_ns=start_ns + 1_499_999_999,
            reference_q=accepted,
            jaw=jaw,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.OPENING
    assert before_deadline.event == "takeover_settle_hold"
    assert before_deadline.commanded_arm_target_q_rad == accepted
    assert before_deadline.commanded_jaw_target_rad == pytest.approx(jaw)

    state, deadline = advance_tip_servo(
        state,
        _observation(
            3,
            evaluated_at_ns=start_ns + 1_500_000_000,
            reference_q=accepted,
            jaw=jaw,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "opening_timeout"
    assert deadline.event == "tip_servo_fault"
    assert deadline.intent is TipServoIntent.FAULT_HOLD
    assert deadline.commanded_arm_target_q_rad == accepted


def test_opening_deadline_also_bounds_a_settled_arm_whose_jaw_never_opens():
    config = _open_only_config(
        takeover_confirm_frames=1,
        opening_timeout_ms=1500.0,
    )
    start_ns = 1_000_000_000
    state, _ = advance_tip_servo(
        TipServoState(),
        _observation(1, evaluated_at_ns=start_ns, jaw=0.305),
        fk=_toy_fk,
        config=config,
    )
    state, opening = advance_tip_servo(
        state,
        _observation(2, evaluated_at_ns=start_ns + 100_000_000, jaw=0.305),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.OPENING
    assert state.open_commanded is True
    assert opening.event == "opening_hold"
    assert opening.commanded_jaw_target_rad == pytest.approx(0.41)

    state, deadline = advance_tip_servo(
        state,
        _observation(3, evaluated_at_ns=start_ns + 1_500_000_000, jaw=0.305),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "opening_timeout"
    assert deadline.intent is TipServoIntent.FAULT_HOLD


def test_open_only_corrupt_descent_state_faults_without_calling_ik(monkeypatch):
    config = _open_only_config()

    def forbidden_ik(*_args, **_kwargs):
        raise AssertionError("OPEN_ONLY invalid-phase guard must run before IK")

    monkeypatch.setattr(
        "lerobot_robot_hexbase.firefly_tip_servo.solve_full_pose_dls",
        forbidden_ik,
    )
    state, decision = advance_tip_servo(
        replace(
            _active_state(phase=TipServoPhase.DESCENDING),
            capture_jaw_target_rad=0.305,
            seeking_started_at_ns=0,
        ),
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "open_only_invalid_phase"
    assert decision.ik_result is None
    assert decision.proposed_descent_step_m == 0.0
    assert decision.close_authorized is False


def test_arm_settle_regression_reaches_opening_descending_and_dls_without_zip_arity_error():
    config = _config(takeover_confirm_frames=1, open_confirm_frames=1)
    state, takeover = advance_tip_servo(
        TipServoState(),
        _observation(1),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.OPENING
    assert takeover.event == "coarse_takeover"

    state, opened = advance_tip_servo(
        state,
        _observation(2),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.DESCENDING
    assert opened.event == "open_stable"

    state, descent = advance_tip_servo(
        state,
        _observation(3, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert descent.intent is TipServoIntent.DESCEND
    assert descent.ik_result is not None and descent.ik_result.ok is True
    assert state.commanded_descent_m == pytest.approx(0.002)


def test_coarse_area_upper_bound_is_exclusive_and_invalid_frame_resets_streak():
    config = _config()
    state = TipServoState()
    state, _ = advance_tip_servo(
        state, _observation(1, area=0.028999), fk=_toy_fk, config=config
    )
    assert state.coarse_streak == 1
    state, _ = advance_tip_servo(
        state, _observation(2, area=0.029), fk=_toy_fk, config=config
    )
    assert state.coarse_streak == 0
    state, _ = advance_tip_servo(
        state,
        _observation(3, status=TipServoFrameStatus.DUPLICATE),
        fk=_toy_fk,
        config=config,
    )
    assert state.coarse_streak == 0


def test_execute_descends_two_mm_in_base_z_and_waits_for_previous_target():
    config = _config()
    state, frame_id = _reach_descending(config)
    frame_id += 1
    state, first = advance_tip_servo(
        state,
        _observation(frame_id, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert first.intent is TipServoIntent.DESCEND
    assert first.proposed_descent_step_m == pytest.approx(0.002)
    assert first.commanded_arm_target_q_rad is not None
    assert state.commanded_descent_m == pytest.approx(0.002)
    first_q = first.commanded_arm_target_q_rad
    first_position, first_rotation = _toy_fk(first_q)
    takeover_position, takeover_rotation = _toy_fk(Q0)
    assert first_position == pytest.approx(
        (takeover_position[0], takeover_position[1], takeover_position[2] - 0.002),
        abs=1e-7,
    )
    assert first_rotation == pytest.approx(takeover_rotation, abs=1e-9)

    frame_id += 1
    held_state, held = advance_tip_servo(
        state,
        _observation(frame_id, q=Q0, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert held.intent is TipServoIntent.HOLD
    assert held.event == "arm_target_settle_hold"
    assert held_state.commanded_descent_m == pytest.approx(0.002)
    assert held.commanded_arm_target_q_rad == first_q

    frame_id += 1
    second_state, second = advance_tip_servo(
        held_state,
        _observation(frame_id, q=first_q, area=0.0195),
        fk=_toy_fk,
        config=config,
    )
    assert second.intent is TipServoIntent.DESCEND
    assert second_state.commanded_descent_m == pytest.approx(0.004)
    assert second.commanded_arm_target_q_rad is not None
    second_position, second_rotation = _toy_fk(second.commanded_arm_target_q_rad)
    assert second_position == pytest.approx(
        (takeover_position[0], takeover_position[1], takeover_position[2] - 0.004),
        abs=1e-7,
    )
    assert second_rotation == pytest.approx(takeover_rotation, abs=1e-9)


@pytest.mark.parametrize(
    "status",
    [
        TipServoFrameStatus.DUPLICATE,
        TipServoFrameStatus.STALE,
        TipServoFrameStatus.OUT_OF_ORDER,
        TipServoFrameStatus.INVALID,
    ],
)
def test_active_nonfresh_frame_holds_first_then_faults_only_after_250ms(status):
    config = _config()
    state = _active_state()
    first = _observation(
        10,
        status=status,
        evaluated_at_ns=1_100_000_000,
    )
    state, decision = advance_tip_servo(state, first, fk=_toy_fk, config=config)
    assert state.phase is TipServoPhase.DESCENDING
    assert decision.intent is TipServoIntent.HOLD
    assert decision.vision_loss_elapsed_ms == 0.0

    state, decision = advance_tip_servo(
        state,
        replace(first, evaluated_at_ns=1_349_000_000),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.DESCENDING
    assert decision.vision_loss_elapsed_ms == pytest.approx(249.0)

    state, decision = advance_tip_servo(
        state,
        replace(first, evaluated_at_ns=1_350_000_000),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "fresh_visual_timeout"
    assert decision.intent is TipServoIntent.FAULT_HOLD


def test_fresh_frame_clears_a_short_visual_loss_timer_without_advancing_duplicate():
    config = _config()
    state = _active_state()
    state, duplicate = advance_tip_servo(
        state,
        _observation(
            10,
            status=TipServoFrameStatus.DUPLICATE,
            evaluated_at_ns=1_100_000_000,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert duplicate.intent is TipServoIntent.HOLD
    assert state.commanded_descent_m == 0.0

    state, fresh = advance_tip_servo(
        state,
        _observation(11, evaluated_at_ns=1_200_000_000, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert state.vision_loss_since_ns is None
    assert fresh.intent is TipServoIntent.DESCEND


def test_fresh_red_loss_or_tracking_corridor_escape_faults_immediately():
    config = _config()
    state = _active_state()
    lost, decision = advance_tip_servo(
        state,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            candidate_count=0,
            centroid=None,
            area=None,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert lost.phase is TipServoPhase.FAULT_HOLD
    assert lost.fault_reason == "red_outside_tracking_corridor"
    assert decision.commanded_arm_target_q_rad == Q0

    escaped, _ = advance_tip_servo(
        state,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            centroid=(0.53, 0.50),
        ),
        fk=_toy_fk,
        config=config,
    )
    assert escaped.phase is TipServoPhase.FAULT_HOLD
    assert escaped.fault_reason == "red_outside_tracking_corridor"


def test_area_drawdown_must_exceed_point002_for_three_fresh_frames_to_fault():
    config = _config()
    state = _active_state(peak_area=0.025)
    first_drop, first_decision = advance_tip_servo(
        state,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            area=0.0229,
            arm_speed=0.0,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert first_drop.phase is TipServoPhase.DESCENDING
    assert first_drop.area_drop_streak == 1
    assert first_drop.commanded_descent_m == state.commanded_descent_m
    assert first_drop.last_arm_target_q_rad == state.last_arm_target_q_rad
    assert first_decision.intent is TipServoIntent.HOLD
    assert first_decision.event == "red_area_drawdown_hold"
    assert first_decision.proposed_descent_step_m == 0.0
    assert first_decision.ik_result is None

    duplicate, duplicate_decision = advance_tip_servo(
        first_drop,
        _observation(
            11,
            status=TipServoFrameStatus.DUPLICATE,
            evaluated_at_ns=1_050_000_000,
            area=0.0229,
            arm_speed=0.0,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert duplicate.area_drop_streak == 1
    assert duplicate.commanded_descent_m == state.commanded_descent_m
    assert duplicate.last_arm_target_q_rad == state.last_arm_target_q_rad
    assert duplicate_decision.intent is TipServoIntent.HOLD
    assert duplicate_decision.proposed_descent_step_m == 0.0
    assert duplicate_decision.ik_result is None

    state, decision = advance_tip_servo(
        duplicate,
        _observation(
            12,
            evaluated_at_ns=1_066_666_666,
            area=0.0229,
            arm_speed=0.0,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.DESCENDING
    assert state.area_drop_streak == 2
    assert state.commanded_descent_m == 0.0
    assert decision.intent is TipServoIntent.HOLD
    assert decision.event == "red_area_drawdown_hold"
    state, decision = advance_tip_servo(
        state,
        _observation(
            13,
            evaluated_at_ns=1_099_999_999,
            area=0.0229,
            arm_speed=0.0,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "red_area_significant_drop"
    assert decision.area_drop_streak == 3

    small_drop = _active_state(peak_area=0.025)
    small_drop, _ = advance_tip_servo(
        small_drop,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            area=0.0231,
            arm_speed=0.2,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert small_drop.area_drop_streak == 0


def test_ik_condition_failure_enters_fault_hold_without_a_best_effort_target():
    config = _config()
    state = _active_state()
    state, decision = advance_tip_servo(
        state,
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.019),
        fk=_singular_fk,
        config=config,
    )

    assert state.phase is TipServoPhase.FAULT_HOLD
    assert state.fault_reason == "ik_jacobian_condition_gate"
    assert decision.intent is TipServoIntent.FAULT_HOLD
    assert decision.ik_result is not None
    assert decision.ik_result.ok is False
    assert decision.commanded_arm_target_q_rad == Q0


def test_default_probe_reports_would_close_but_never_authorizes_or_closes_jaw():
    config = _config(allow_close=False)
    state = _active_state(peak_area=0.033)
    decision = None
    for index in range(config.close_ready_confirm_frames):
        state, decision = advance_tip_servo(
            state,
            _observation(
                11 + index,
                evaluated_at_ns=1_033_333_333 + index * 33_333_333,
                centroid=(0.45, 0.55),
                area=0.033,
            ),
            fk=_toy_fk,
            config=config,
        )
        if index + 1 == config.probe_success_confirm_frames:
            assert state.phase is TipServoPhase.DESCENDING
            assert decision.event == "close_ready_confirming"
    assert decision is not None
    assert state.phase is TipServoPhase.PROBE_COMPLETE
    assert decision.intent is TipServoIntent.HOLD
    assert decision.event == "visual_would_close"
    assert decision.would_close is True
    assert decision.close_authorized is False
    assert decision.commanded_jaw_target_rad == pytest.approx(0.41)


def test_reviewed_full_execute_only_reports_authorized_visual_close_ready():
    config = _config(allow_close=True)
    state = replace(
        _active_state(peak_area=0.033),
        close_ready_streak=config.close_ready_confirm_frames - 1,
    )
    state, decision = advance_tip_servo(
        state,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            centroid=(0.45, 0.55),
            area=0.033,
        ),
        fk=_toy_fk,
        config=config,
    )

    assert state.phase is TipServoPhase.CLOSE_READY
    assert decision.intent is TipServoIntent.VISUAL_CLOSE_READY
    assert decision.event == "visual_close_ready"
    assert decision.would_close is True
    assert decision.close_authorized is True
    assert decision.commanded_jaw_target_rad == pytest.approx(0.41)


def test_advisory_never_authorizes_close_even_when_allow_close_is_true():
    config = TipServoConfig(mode=TipServoMode.ADVISORY, allow_close=True)
    state = replace(
        _active_state(peak_area=0.033),
        close_ready_streak=config.close_ready_confirm_frames - 1,
    )
    next_state, decision = advance_tip_servo(
        state,
        _observation(
            11,
            evaluated_at_ns=1_033_333_333,
            centroid=(0.45, 0.55),
            area=0.033,
        ),
        fk=_toy_fk,
        config=config,
    )
    assert next_state.phase is TipServoPhase.CLOSE_READY
    assert decision.would_close is True
    assert decision.close_authorized is False
    assert decision.commanded_arm_target_q_rad is None
    assert decision.commanded_jaw_target_rad is None


def test_probe_stops_early_after_six_settled_fresh_frames_at_area_point020():
    config = _config(allow_close=False)
    state = _active_state(peak_area=0.020)
    decision = None
    for index in range(config.probe_success_confirm_frames):
        state, decision = advance_tip_servo(
            state,
            _observation(
                11 + index,
                evaluated_at_ns=1_033_333_333 + index * 33_333_333,
                area=0.020,
            ),
            fk=_toy_fk,
            config=config,
        )
    assert decision is not None
    assert state.phase is TipServoPhase.PROBE_COMPLETE
    assert state.commanded_descent_m == 0.0
    assert state.probe_success_streak == 6
    assert decision.event == "probe_progress_complete"
    assert decision.close_authorized is False


def test_probe_at_descent_limit_requires_six_area_progress_frames_or_faults_below_it():
    config = _config(max_total_descent_m=0.004, allow_close=False)
    at_limit = _active_state(descent_m=0.004, peak_area=0.021)
    complete = at_limit
    decision = None
    for index in range(config.probe_success_confirm_frames):
        complete, decision = advance_tip_servo(
            complete,
            _observation(
                11 + index,
                evaluated_at_ns=1_033_333_333 + index * 33_333_333,
                area=0.021,
            ),
            fk=_toy_fk,
            config=config,
        )
    assert decision is not None
    assert complete.phase is TipServoPhase.PROBE_COMPLETE
    assert decision.event == "probe_progress_complete"
    assert decision.close_authorized is False

    failed, _ = advance_tip_servo(
        _active_state(descent_m=0.004, peak_area=0.019),
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.019),
        fk=_toy_fk,
        config=config,
    )
    assert failed.phase is TipServoPhase.FAULT_HOLD
    assert failed.fault_reason == "probe_descent_without_minimum_area_progress"

    full, _ = advance_tip_servo(
        _active_state(descent_m=0.004, peak_area=0.021),
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.021),
        fk=_toy_fk,
        config=_config(max_total_descent_m=0.004, allow_close=True),
    )
    assert full.phase is TipServoPhase.FAULT_HOLD
    assert full.fault_reason == "max_total_descent_without_close_ready"


def test_decision_telemetry_contains_only_json_primitive_values():
    config = _config()
    state = _active_state()
    _, decision = advance_tip_servo(
        state,
        _observation(11, evaluated_at_ns=1_033_333_333, area=0.019),
        fk=_toy_fk,
        config=config,
    )

    telemetry = decision.telemetry()
    _assert_primitive_tree(telemetry)
    json.dumps(telemetry, allow_nan=False)
    assert telemetry["mode"] == "execute"
    assert telemetry["ik"]["ok"] is True
    assert type(telemetry["proposed_arm_target_q_rad"]) is list

    position, rotation = _toy_fk(Q0)
    desired = position.copy()
    desired[2] -= 0.002
    singular = solve_full_pose_dls(Q0, desired, rotation, fk=_singular_fk)
    json.dumps(singular.telemetry(), allow_nan=False)
    assert singular.telemetry()["condition_number"] is None
