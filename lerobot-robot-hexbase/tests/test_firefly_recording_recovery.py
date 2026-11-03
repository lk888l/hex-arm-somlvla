from __future__ import annotations

from dataclasses import replace

import pytest

from lerobot_robot_hexbase.firefly_recording_recovery import (
    AXIS_NAMES,
    STAGE_ORDER,
    RecoveryPlan,
    RecoveryPlanningError,
    RecoveryProfile,
    audit_recovery_plan,
    assess_recording_boundaries,
    plan_recording_recovery,
    stage_contract,
)


def _profile(**overrides) -> RecoveryProfile:
    values = {
        "home": [0.20, 0.25, 0.30, 0.40, -0.20, 0.10, 0.15],
        "home_tolerance": [0.02] * 7,
        "position_min": [-2.0] * 7,
        "position_max": [2.0] * 7,
        "entry_min": [-0.50] * 7,
        "entry_max": [0.50] * 7,
        "recovery_min": [-1.0] * 7,
        "recovery_max": [1.0] * 7,
        "step_cap": [0.10] * 6 + [0.05],
        "j4_clearance": 0.40,
        "max_frames": 100,
        "max_total_travel_rad": 10.0,
        "allow_positive_j3_motion": True,
    }
    values.update(overrides)
    return RecoveryProfile(**values)


def _start() -> list[float]:
    return [-0.30, -0.40, 0.10, -0.20, 0.30, -0.40, 0.35]


def test_profile_normalizes_finite_seven_value_vectors():
    profile = _profile()

    assert profile.home == (0.20, 0.25, 0.30, 0.40, -0.20, 0.10, 0.15)
    assert profile.step_cap == (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05)

    with pytest.raises(RecoveryPlanningError, match="exactly seven"):
        _profile(home=[0.0] * 6)
    with pytest.raises(RecoveryPlanningError, match="NaN or infinity"):
        _profile(home=[0.0] * 6 + [float("nan")])
    with pytest.raises(RecoveryPlanningError, match="exactly seven"):
        plan_recording_recovery([0.0] * 6, profile)
    with pytest.raises(RecoveryPlanningError, match="NaN or infinity"):
        plan_recording_recovery([0.0] * 6 + [float("inf")], profile)


def test_profile_envelopes_are_nested_inside_inset_position_limits():
    with pytest.raises(RecoveryPlanningError, match="recovery envelope exceeds inset"):
        _profile(recovery_min=[-2.01] + [-1.0] * 6)
    with pytest.raises(RecoveryPlanningError, match="entry envelope exceeds recovery"):
        _profile(entry_max=[1.01] + [0.50] * 6)
    with pytest.raises(RecoveryPlanningError, match="home tolerance does not fit"):
        _profile(entry_max=[0.21] + [0.50] * 6)


def test_j4_clearance_is_the_home_value_and_cannot_be_lowered_later():
    with pytest.raises(RecoveryPlanningError, match="must equal home joint_4"):
        _profile(j4_clearance=0.41)


def test_plan_uses_only_fixed_stage_order_and_axis_authority():
    profile = _profile()
    plan = plan_recording_recovery(_start(), profile)

    assert plan.stages == STAGE_ORDER
    assert plan.final == profile.home
    assert plan.required_frames <= profile.max_frames
    assert plan.total_travel_rad <= profile.max_total_travel_rad
    assert [row["stage"] for row in stage_contract()] == list(STAGE_ORDER)
    assert stage_contract() == (
        {
            "stage": "j4_clearance",
            "axis_indexes": (3,),
            "axis_names": ("joint_4.pos",),
        },
        {
            "stage": "j2_home",
            "axis_indexes": (1,),
            "axis_names": ("joint_2.pos",),
        },
        {
            "stage": "j3_home",
            "axis_indexes": (2,),
            "axis_names": ("joint_3.pos",),
        },
        {
            "stage": "remaining_ee_home",
            "axis_indexes": (0, 4, 5, 6),
            "axis_names": (
                "joint_1.pos",
                "joint_5.pos",
                "joint_6.pos",
                "jaw_joint1.pos",
            ),
        },
    )

    previous = plan.start
    stage_indexes = []
    stage_targets = {
        "j4_clearance": {3: profile.home[3]},
        "j2_home": {1: profile.home[1]},
        "j3_home": {2: profile.home[2]},
        "remaining_ee_home": {
            0: profile.home[0],
            4: profile.home[4],
            5: profile.home[5],
            6: profile.home[6],
        },
    }
    for step in plan.steps:
        stage_indexes.append(STAGE_ORDER.index(step.stage))
        allowed = set(step.allowed_axis_indexes)
        changed = {
            index
            for index, (before, after) in enumerate(
                zip(previous, step.positions, strict=True)
            )
            if abs(after - before) > 1e-12
        }
        assert changed <= allowed
        assert step.allowed_axis_names == tuple(AXIS_NAMES[index] for index in allowed)
        for index, (before, after, cap) in enumerate(
            zip(previous, step.positions, profile.step_cap, strict=True)
        ):
            assert abs(after - before) <= cap + 1e-12
            if index in stage_targets[step.stage]:
                target = stage_targets[step.stage][index]
                assert abs(target - after) <= abs(target - before) + 1e-12
        previous = step.positions
    assert stage_indexes == sorted(stage_indexes)

    audit = audit_recovery_plan(plan, profile)
    assert audit["safe"] is True
    assert audit["errors"] == ()
    assert audit["observed_stages"] == STAGE_ORDER


def test_positive_j3_motion_requires_explicit_profile_authorization():
    profile = _profile(allow_positive_j3_motion=False)

    with pytest.raises(RecoveryPlanningError, match="positive J3.*explicit"):
        plan_recording_recovery(_start(), profile)

    negative_j3_start = list(profile.home)
    negative_j3_start[2] = 0.60
    plan = plan_recording_recovery(negative_j3_start, profile)
    assert plan.stages == ("j3_home",)
    assert plan.final == profile.home


def test_start_outside_wider_recovery_envelope_fails_closed():
    start = _start()
    start[0] = 1.0001

    with pytest.raises(RecoveryPlanningError, match="outside.*fail closed"):
        plan_recording_recovery(start, _profile())


def test_frame_and_total_travel_caps_are_hard_failures():
    with pytest.raises(RecoveryPlanningError, match="exceeding max_frames=1"):
        plan_recording_recovery(_start(), _profile(max_frames=1))
    with pytest.raises(RecoveryPlanningError, match="exceeds max_total_travel"):
        plan_recording_recovery(
            _start(),
            _profile(max_total_travel_rad=0.10),
        )


def test_audit_detects_a_forbidden_axis_change_in_a_tampered_plan():
    profile = _profile()
    plan = plan_recording_recovery(_start(), profile)
    first = plan.steps[0]
    tampered_positions = list(first.positions)
    tampered_positions[0] += 0.01
    tampered_step = replace(first, positions=tuple(tampered_positions))
    tampered = RecoveryPlan(
        start=plan.start,
        steps=(tampered_step,) + plan.steps[1:],
        total_travel_rad=plan.total_travel_rad,
    )

    audit = audit_recovery_plan(tampered, profile)
    assert audit["safe"] is False
    assert any("changes forbidden axis joint_1.pos" in error for error in audit["errors"])


def test_boundary_assessment_requires_continuous_head_tail_and_final_windows():
    profile = _profile()
    home = list(profile.home)
    entry = list(home)
    entry[0] += 0.10
    task_pose = list(home)
    task_pose[0] = 0.80
    samples = [home, entry, task_pose, entry, home, home]

    result = assess_recording_boundaries(
        samples,
        profile,
        head_entry_frames=2,
        tail_entry_frames=3,
        final_home_frames=2,
    )

    assert result.head_entry_consecutive == 2
    assert result.tail_entry_consecutive == 3
    assert result.final_home_consecutive == 2
    assert result.head_entry_passed is True
    assert result.tail_entry_passed is True
    assert result.final_home_passed is True
    assert result.ready_for_next_recording is True
    assert result.episode_boundaries_passed is True
    assert result.fail_closed is False


def test_a_single_final_home_sample_does_not_pass_the_stability_gate():
    profile = _profile()
    home = list(profile.home)
    entry = list(home)
    entry[0] += 0.10
    task_pose = list(home)
    task_pose[0] = 0.80

    result = assess_recording_boundaries(
        [home, home, task_pose, entry, home],
        profile,
        head_entry_frames=2,
        tail_entry_frames=2,
        final_home_frames=2,
    )

    assert result.head_entry_passed is True
    assert result.tail_entry_passed is True
    assert result.final_home_consecutive == 1
    assert result.final_home_passed is False
    assert result.ready_for_next_recording is False
    assert result.episode_boundaries_passed is False


def test_head_gate_is_independent_from_ready_for_next_recording():
    profile = _profile()
    home = list(profile.home)
    task_pose = list(home)
    task_pose[0] = 0.80

    result = assess_recording_boundaries(
        [home, task_pose, home, home],
        profile,
        head_entry_frames=2,
        tail_entry_frames=2,
        final_home_frames=2,
    )

    assert result.head_entry_consecutive == 1
    assert result.head_entry_passed is False
    assert result.ready_for_next_recording is True
    assert result.episode_boundaries_passed is False


def test_boundary_assessment_fails_closed_on_any_recovery_envelope_escape():
    profile = _profile()
    home = list(profile.home)
    outside = list(home)
    outside[0] = 1.0001

    result = assess_recording_boundaries(
        [home, home, outside, home, home],
        profile,
        head_entry_frames=2,
        tail_entry_frames=2,
        final_home_frames=2,
    )

    assert result.first_outside_recovery_index == 2
    assert result.fail_closed is True
    assert result.ready_for_next_recording is False
    assert result.episode_boundaries_passed is False


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        ([], "non-empty"),
        ([[0.0] * 6], "exactly seven"),
        ([[0.0] * 6 + [float("nan")]], "NaN or infinity"),
    ],
)
def test_boundary_assessment_rejects_invalid_sample_vectors(samples, match):
    with pytest.raises(RecoveryPlanningError, match=match):
        assess_recording_boundaries(
            samples,
            _profile(),
            head_entry_frames=1,
            tail_entry_frames=1,
            final_home_frames=1,
        )
