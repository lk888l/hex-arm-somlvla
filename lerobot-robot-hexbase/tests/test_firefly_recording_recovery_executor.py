from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot_robot_hexbase.firefly_recording_recovery import (
    AXIS_NAMES,
    STAGE_ORDER,
    RecoveryProfile,
)
from lerobot_robot_hexbase.firefly_recording_recovery_executor import (
    RecoveryControlContract,
    RecoveryExecutionError,
    execute_connected_recovery,
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


def _control(**overrides) -> RecoveryControlContract:
    values = {
        "candidate_lead_budget": [0.01] * 7,
        "stall_frames": 5,
        "max_movement_control_frames": 200,
        "settle_frames": 10,
        "required_home_frames": 3,
        "minimum_progress_rad": 0.001,
    }
    values.update(overrides)
    return RecoveryControlContract(**values)


class _FakeRobot:
    def __init__(self, start, *, follow_commands: bool = True):
        self.config = SimpleNamespace(
            mode="execute",
            arm_tracking_error_rad=0.15,
            ee_tracking_error_rad=0.05,
            command_hz=30.0,
        )
        self.action_features = dict.fromkeys(AXIS_NAMES, float)
        self.is_connected = True
        self.stopped_reason = None
        self._measured = list(start)
        self.command_target = dict(zip(AXIS_NAMES, start, strict=True))
        self.follow_commands = follow_commands
        self.actions = []
        self.events = []
        self.preflight = None

    def get_observation(self):
        return dict(zip(AXIS_NAMES, self._measured, strict=True))

    def preflight_actions(self, actions):
        self.preflight = tuple(tuple(frame) for frame in actions)

    def send_action(self, action):
        values = tuple(action[name] for name in AXIS_NAMES)
        self.actions.append(values)
        self.command_target = dict(action)
        if self.follow_commands:
            self._measured = list(values)
        return dict(action)

    def record_event(self, event, **fields):
        self.events.append((event, fields))

    def stop(self, reason):
        self.stopped_reason = reason


def _start():
    return [-0.30, -0.40, 0.10, -0.20, 0.30, -0.40, 0.35]


def test_connected_core_is_feedback_paced_and_uses_fixed_stage_order():
    robot = _FakeRobot(_start())

    result = execute_connected_recovery(
        robot,
        _profile(),
        _control(),
        sleep=lambda _seconds: None,
    )

    assert result.plan.stages == STAGE_ORDER
    assert result.plan.final == _profile().home
    assert robot.preflight == (
        result.plan.start,
        *(step.positions for step in result.plan.steps),
    )
    assert robot.actions[-1] == _profile().home
    stage_events = [
        fields["stage"]
        for event, fields in robot.events
        if event == "recording_recovery_stage_begin"
    ]
    assert tuple(stage_events) == STAGE_ORDER
    assert robot.events[-1][0] == "recording_recovery_complete"
    assert result.consecutive_home_frames == 3


def test_connected_core_stops_if_start_is_outside_recovery_envelope():
    start = _start()
    start[0] = 1.01
    robot = _FakeRobot(start)

    with pytest.raises(RecoveryExecutionError, match="outside.*recovery envelope"):
        execute_connected_recovery(
            robot,
            _profile(),
            _control(),
            sleep=lambda _seconds: None,
        )

    assert robot.actions == []
    assert robot.stopped_reason is not None


def test_connected_core_stops_on_measured_feedback_stall():
    robot = _FakeRobot(_start(), follow_commands=False)

    with pytest.raises(RecoveryExecutionError, match="stalled for 3 frames"):
        execute_connected_recovery(
            robot,
            _profile(),
            _control(stall_frames=3),
            sleep=lambda _seconds: None,
        )

    assert robot.stopped_reason is not None
    assert len(robot.actions) == 4


def test_connected_core_requires_execute_mode_exact_schema_and_tight_lead_budget():
    robot = _FakeRobot(_start())
    robot.config.mode = "shadow"
    with pytest.raises(RecoveryExecutionError, match="execute-mode"):
        execute_connected_recovery(robot, _profile(), _control())

    robot = _FakeRobot(_start())
    robot.action_features = dict.fromkeys(reversed(AXIS_NAMES), float)
    with pytest.raises(RecoveryExecutionError, match="exact ordered"):
        execute_connected_recovery(robot, _profile(), _control())

    robot = _FakeRobot(_start())
    with pytest.raises(RecoveryExecutionError, match="strictly below"):
        execute_connected_recovery(
            robot,
            _profile(),
            _control(candidate_lead_budget=[0.01] * 6 + [0.05]),
        )


def test_control_contract_has_hard_rate_and_finite_positive_limits():
    with pytest.raises(ValueError, match="fixed at 30"):
        _control(command_hz=29.0)
    with pytest.raises(ValueError, match="positive"):
        _control(candidate_lead_budget=[0.01] * 6 + [0.0])
    with pytest.raises(ValueError, match="must not exceed"):
        _control(settle_frames=2, required_home_frames=3)
