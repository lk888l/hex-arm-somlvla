from __future__ import annotations

import pytest

from lerobot_robot_hexbase.shadow_envelope import EnvelopeContract, analyze_shadow_envelope


def _contract() -> EnvelopeContract:
    return EnvelopeContract.from_mapping(
        {
            "source": "test live description",
            "joint_names": [
                "joint_1.pos",
                "joint_2.pos",
                "joint_3.pos",
                "joint_4.pos",
                "joint_5.pos",
                "joint_6.pos",
                "jaw_joint1.pos",
            ],
            "pos_min": [-2.86, -1.57, 0.0, -1.57, -1.54, -2.79, 0.0],
            "pos_max": [2.86, 2.09, 3.14, 1.57, 1.54, 2.79, 0.6],
            "effective_step_limits": [0.035] * 6 + [0.012],
        }
    )


def _event(target, measured=None):
    return {
        "event": "safety_rejection",
        "target": target,
        "measured": [0.0] * 7 if measured is None else measured,
        "effective_step_limits": [0.035] * 6 + [0.012],
    }


def test_hypothetical_bounds_cannot_hide_first_step_rejection():
    measured = [0.0, -1.57, 3.14, 0.0, 0.0, 0.0, 0.0]
    events = [
        {"event": "inference_chunk"},
        _event([0.05, -1.58, 3.15, 0.0, 0.0, 0.0, 0.0], measured),
        _event([0.051, -1.579, 3.149, 0.0, 0.0, 0.0, 0.0], measured),
    ]
    report = analyze_shadow_envelope(
        events,
        _contract(),
        joint_2_min=-1.59,
        joint_3_max=3.16,
    )
    assert report["hypothetical_only"] is True
    assert report["eligible_for_execution"] is False
    assert report["controller_position_rejections"] == 2
    assert report["hypothetical_position_rejections"] == 0
    assert report["hypothetical_adapter_rejections"] == 2
    assert report["hypothetical_adapter_acceptances"] == 0
    assert report["open_loop_step_rejections"] == 1
    assert report["first_hypothetical_rejection"]["step_joints"] == ["joint_1.pos"]


@pytest.mark.parametrize(
    ("joint_2_min", "joint_3_max", "match"),
    [
        (-1.56, 3.16, "only expand outward"),
        (-1.59, 3.13, "only expand outward"),
        (-1.591, 3.16, "limited to 0.020"),
        (-1.59, 3.161, "limited to 0.020"),
        (float("nan"), 3.16, "finite"),
    ],
)
def test_diagnostic_expansion_is_narrow_and_finite(joint_2_min, joint_3_max, match):
    with pytest.raises(ValueError, match=match):
        analyze_shadow_envelope(
            [_event([0.0] * 7)],
            _contract(),
            joint_2_min=joint_2_min,
            joint_3_max=joint_3_max,
        )


def test_execute_telemetry_is_rejected():
    with pytest.raises(ValueError, match="execute command telemetry"):
        analyze_shadow_envelope(
            [{"event": "command_cycle", "target": [0.0] * 7, "measured": [0.0] * 7}],
            _contract(),
            joint_2_min=-1.59,
            joint_3_max=3.16,
        )
