from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from lerobot_robot_hexbase.firefly_arm_only_policy import (
    ARM6_OBSERVATION_FEATURES,
    ARM6_SEEKING_ROLE,
    ArmOnlyPolicyError,
    ArmOnlyPolicyRole,
    expand_arm6_action_chunk,
    extract_arm6_observation,
    resolve_arm6_takeover_boundary,
    validate_arm6_action_chunk,
)


def _observation() -> dict[str, object]:
    return {
        **{
            name: float(index) / 10.0
            for index, name in enumerate(ARM6_OBSERVATION_FEATURES, start=1)
        },
        "jaw_joint1.pos": 0.31,
        "scene": object(),
        "wrist": object(),
    }


def _chunk(steps: int = 2) -> list[list[float]]:
    return [
        [float(step) + axis / 100.0 for axis in range(6)]
        for step in range(steps)
    ]


def test_extract_arm6_observation_reads_exactly_six_arm_axes_without_mutation():
    observation = _observation()
    before = dict(observation)

    result = extract_arm6_observation(
        observation,
        role=ArmOnlyPolicyRole.ARM6_SEEKING,
    )

    assert result == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    assert observation == before
    assert len(result) == 6


def test_extract_arm6_observation_never_reads_jaw():
    class ExplodesIfConverted:
        def __float__(self) -> float:
            raise AssertionError("jaw must not be read by the arm-only extractor")

    observation = _observation()
    observation["jaw_joint1.pos"] = ExplodesIfConverted()

    assert extract_arm6_observation(
        observation,
        role=ARM6_SEEKING_ROLE,
    ) == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))


@pytest.mark.parametrize("missing", ARM6_OBSERVATION_FEATURES)
def test_extract_arm6_observation_rejects_every_missing_axis(missing: str):
    observation = _observation()
    observation.pop(missing)

    with pytest.raises(ArmOnlyPolicyError, match="missing required feature"):
        extract_arm6_observation(observation, role=ARM6_SEEKING_ROLE)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "bad"])
def test_extract_arm6_observation_rejects_nonfinite_or_nonnumeric_arm(bad: object):
    observation = _observation()
    observation["joint_4.pos"] = bad

    with pytest.raises(ArmOnlyPolicyError, match="finite number"):
        extract_arm6_observation(observation, role=ARM6_SEEKING_ROLE)


@pytest.mark.parametrize("role", ["policy", "arm7_fullcycle", "", None])
def test_all_entry_points_reject_unsupported_or_missing_role(role: object):
    with pytest.raises(ArmOnlyPolicyError, match="arm6_seeking"):
        extract_arm6_observation(_observation(), role=role)  # type: ignore[arg-type]
    with pytest.raises(ArmOnlyPolicyError, match="arm6_seeking"):
        validate_arm6_action_chunk(_chunk(), n_steps=2, role=role)  # type: ignore[arg-type]
    with pytest.raises(ArmOnlyPolicyError, match="arm6_seeking"):
        expand_arm6_action_chunk(
            _chunk(),
            n_steps=2,
            role=role,  # type: ignore[arg-type]
            capture_jaw_target_rad=0.31,
            capture_jaw_range_rad=(0.285, 0.325),
        )


def test_validate_arm6_action_chunk_accepts_numpy_and_keeps_input_unchanged():
    raw = np.asarray(_chunk(3), dtype=np.float32)
    before = raw.copy()

    result = validate_arm6_action_chunk(
        raw,
        n_steps=3,
        role=ARM6_SEEKING_ROLE,
    )

    assert np.array_equal(raw, before)
    assert np.asarray(result).shape == (3, 6)
    assert np.asarray(result) == pytest.approx(before)


@pytest.mark.parametrize(
    ("raw", "n_steps"),
    [
        ([[0.0] * 7], 1),
        ([[0.0] * 5], 1),
        ([[0.0] * 6, [0.0] * 6], 1),
        ([[0.0] * 6], 2),
    ],
)
def test_validate_arm6_action_chunk_rejects_7d_missing_axes_and_wrong_steps(
    raw: list[list[float]], n_steps: int
):
    with pytest.raises(ArmOnlyPolicyError):
        validate_arm6_action_chunk(
            raw,
            n_steps=n_steps,
            role=ARM6_SEEKING_ROLE,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "bad"])
def test_validate_arm6_action_chunk_rejects_bad_values(bad: object):
    raw: list[list[object]] = [[0.0] * 6]
    raw[0][3] = bad

    with pytest.raises(ArmOnlyPolicyError, match="finite number"):
        validate_arm6_action_chunk(
            raw,
            n_steps=1,
            role=ARM6_SEEKING_ROLE,
        )


@pytest.mark.parametrize("n_steps", [0, -1, 1.0, True])
def test_validate_arm6_action_chunk_rejects_invalid_step_count(n_steps: object):
    with pytest.raises(ArmOnlyPolicyError, match="positive integer"):
        validate_arm6_action_chunk(
            _chunk(),
            n_steps=n_steps,  # type: ignore[arg-type]
            role=ARM6_SEEKING_ROLE,
        )


def test_expand_arm6_action_chunk_appends_exact_external_capture_and_reports_authority():
    raw = _chunk(2)
    before = deepcopy(raw)

    result = expand_arm6_action_chunk(
        raw,
        n_steps=2,
        role=ARM6_SEEKING_ROLE,
        capture_jaw_target_rad=0.30926513671875,
        capture_jaw_range_rad=(0.285, 0.325),
    )

    assert raw == before
    assert result.raw_arm_action6 == tuple(tuple(row) for row in raw)
    assert result.synthesized_capture_jaw == 0.30926513671875
    assert all(row[6] == 0.30926513671875 for row in result.candidate7)
    assert all(
        row[:6] == tuple(arm)
        for row, arm in zip(result.candidate7, raw, strict=True)
    )
    telemetry = result.telemetry()
    assert telemetry == {
        "policy_role": "arm6_seeking",
        "raw_arm_action6": raw,
        "synthesized_capture_jaw": 0.30926513671875,
        "candidate7": [list(row) for row in result.candidate7],
    }
    assert not any("policy_jaw" in key for key in telemetry)


@pytest.mark.parametrize("capture", [0.285, 0.325])
def test_expand_arm6_action_chunk_accepts_inclusive_capture_edges(capture: float):
    result = expand_arm6_action_chunk(
        _chunk(1),
        n_steps=1,
        role=ARM6_SEEKING_ROLE,
        capture_jaw_target_rad=capture,
        capture_jaw_range_rad=(0.285, 0.325),
    )
    assert result.candidate7[0][6] == capture


@pytest.mark.parametrize("capture", [0.284999, 0.325001, math.nan, math.inf, True])
def test_expand_arm6_action_chunk_rejects_invalid_capture(capture: object):
    with pytest.raises(ArmOnlyPolicyError):
        expand_arm6_action_chunk(
            _chunk(1),
            n_steps=1,
            role=ARM6_SEEKING_ROLE,
            capture_jaw_target_rad=capture,
            capture_jaw_range_rad=(0.285, 0.325),
        )


@pytest.mark.parametrize(
    "interval",
    [(0.325, 0.285), (0.31, 0.31), (0.285,), (0.285, 0.325, 0.4), (0.285, math.nan)],
)
def test_expand_arm6_action_chunk_rejects_invalid_capture_interval(interval: object):
    with pytest.raises(ArmOnlyPolicyError):
        expand_arm6_action_chunk(
            _chunk(1),
            n_steps=1,
            role=ARM6_SEEKING_ROLE,
            capture_jaw_target_rad=0.31,
            capture_jaw_range_rad=interval,
        )


def test_telemetry_is_detached_from_immutable_result():
    result = expand_arm6_action_chunk(
        _chunk(1),
        n_steps=1,
        role=ARM6_SEEKING_ROLE,
        capture_jaw_target_rad=0.31,
        capture_jaw_range_rad=(0.285, 0.325),
    )
    telemetry = result.telemetry()
    telemetry["raw_arm_action6"][0][0] = 999.0
    telemetry["candidate7"][0][6] = 999.0

    assert result.raw_arm_action6[0][0] == 0.0
    assert result.candidate7[0][6] == 0.31


def test_takeover_boundary_requires_chunk_discard_and_smoother_reset():
    before = resolve_arm6_takeover_boundary(
        role=ARM6_SEEKING_ROLE,
        takeover_boundary=False,
    )
    takeover = resolve_arm6_takeover_boundary(
        role=ARM6_SEEKING_ROLE,
        takeover_boundary=True,
    )

    assert before.arm_policy_authorized is True
    assert before.discard_policy_chunk is False
    assert before.smoother_reset_required is False
    assert before.replan_reason is None
    assert takeover.arm_policy_authorized is False
    assert takeover.discard_policy_chunk is True
    assert takeover.smoother_reset_required is True
    assert takeover.replan_reason == "tip_servo_takeover"


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_takeover_boundary_requires_real_boolean(value: object):
    with pytest.raises(ArmOnlyPolicyError, match="must be boolean"):
        resolve_arm6_takeover_boundary(
            role=ARM6_SEEKING_ROLE,
            takeover_boundary=value,  # type: ignore[arg-type]
        )
