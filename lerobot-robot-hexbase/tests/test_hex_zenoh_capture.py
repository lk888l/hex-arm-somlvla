from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import draccus
import pytest

from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import RecordConfig, record_loop

import lerobot_robot_hexbase.hex_zenoh_command_tap as command_tap_module
import lerobot_robot_hexbase.hex_zenoh_observer as observer_module
from lerobot_robot_hexbase import (
    HexZenohCommandTap,
    HexZenohCommandTapConfig,
    HexZenohObserver,
    HexZenohObserverConfig,
    UnsafeRecordedActionError,
)
from lerobot_robot_hexbase.zenoh_helpers import LatestJointVector, arm_pb2

CONTROLLER_ID = "test-controller"
ARM_PREFIX = f"hexmeow/{CONTROLLER_ID}/arm1"
EE_PREFIX = f"hexmeow/{CONTROLLER_ID}/ee1"


class _Payload:
    def __init__(self, data: bytes):
        self._data = data

    def to_bytes(self) -> bytes:
        return self._data


class _Sample:
    def __init__(self, data: bytes):
        self.payload = _Payload(data)


class _Reply:
    def __init__(self, data: bytes):
        self.ok = _Sample(data)
        self.err = None


class _Subscriber:
    def __init__(self):
        self.closed = False

    def undeclare(self) -> None:
        self.closed = True


def _arm_description() -> bytes:
    return arm_pb2.ArmDescription(
        dof=6,
        joint_names=[f"joint_{index}" for index in range(1, 7)],
        pos_min=[-2.86, -1.57, 0.0, -1.57, -1.54, -2.79],
        pos_max=[2.86, 2.09, 3.14, 1.57, 1.54, 2.79],
        vel_max=[2.0] * 6,
        tau_max=[1.0] * 6,
        default_kp=[40.0] * 6,
        default_kd=[2.0] * 6,
        motor_models=["fake"] * 6,
        supported_timeouts=[arm_pb2.TIMEOUT_BEHAVIOR_HOLD],
    ).SerializeToString()


def _ee_description() -> bytes:
    return command_tap_module.ee_pb2.EeDescription(
        dof=1,
        joint_names=["jaw_joint1"],
        pos_min=[0.0],
        pos_max=[0.6],
        vel_max=[0.6],
        tau_max=[1.0],
        default_kp=[150.0],
        default_kd=[2.0],
        motor_models=["fake"],
        supported_timeouts=[arm_pb2.TIMEOUT_BEHAVIOR_HOLD],
    ).SerializeToString()


class _FakeSession:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.declared_keys: list[str] = []
        self.callbacks: dict[str, object] = {}
        self.subscribers: list[_Subscriber] = []
        self.get_calls: list[tuple[str, bytes, float]] = []
        self.closed = False

    def get(self, key: str, *, payload: bytes, timeout: float):
        self.get_calls.append((key, payload, timeout))
        descriptions = {
            f"{ARM_PREFIX}/arm/description": _arm_description(),
            f"{EE_PREFIX}/ee/description": _ee_description(),
        }
        if key not in descriptions:
            raise AssertionError(f"unexpected read-only query: {key}")
        return [_Reply(descriptions[key])]

    def declare_subscriber(self, key: str, callback):
        self.declared_keys.append(key)
        self.callbacks[key] = callback
        subscriber = _Subscriber()
        self.subscribers.append(subscriber)
        callback(_Sample(self.payloads[key]))
        return subscriber

    def emit(self, key: str, payload: bytes) -> None:
        self.callbacks[key](_Sample(payload))

    def put(self, *_args, **_kwargs) -> None:
        raise AssertionError("read-only capture session attempted Zenoh put")

    def declare_publisher(self, *_args, **_kwargs):
        raise AssertionError("read-only capture session attempted to publish")

    def close(self) -> None:
        self.closed = True


def _joint_state(values: list[float], *, seq: int = 1) -> bytes:
    message = arm_pb2.JointState(q=values)
    message.header.seq = seq
    return message.SerializeToString()


def _joint_command(
    values: list[float],
    *,
    seq: int = 1,
    session_id: int = 11,
    point: arm_pb2.JointSetpoint | None = None,
) -> bytes:
    message = arm_pb2.JointTrajectory(
        session_id=session_id,
        points=[arm_pb2.JointSetpoint(q=values) if point is None else point],
        t_from_start_ns=[33_333_333],
        on_timeout=arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
    )
    message.header.seq = seq
    return message.SerializeToString()


def _multi_point_command(values: list[float]) -> bytes:
    return arm_pb2.JointTrajectory(
        session_id=11,
        points=[
            arm_pb2.JointSetpoint(q=values),
            arm_pb2.JointSetpoint(q=values),
        ],
        t_from_start_ns=[33_333_333, 66_666_666],
        on_timeout=arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
    ).SerializeToString()


def _robot_status(session_holder: int) -> bytes:
    return command_tap_module.robot_pb2.RobotStatus(
        mode=command_tap_module.common_pb2.ROBOT_MODE_RUNNING,
        session_holder=session_holder,
    ).SerializeToString()


ARM_SESSION = 11
EE_SESSION = 22
ARM_STATE = [0.10, -0.20, 1.00, 0.10, 0.20, -0.30]
ARM_TARGET = [0.11, -0.19, 1.01, 0.11, 0.21, -0.29]
EE_STATE = [0.30]
EE_TARGET = [0.305]


def _tap_payloads(
    *,
    arm_state: list[float] = ARM_STATE,
    ee_state: list[float] = EE_STATE,
    arm_target: list[float] = ARM_TARGET,
    ee_target: list[float] = EE_TARGET,
) -> dict[str, bytes]:
    return {
        f"{ARM_PREFIX}/arm/joint_state": _joint_state(arm_state),
        f"{EE_PREFIX}/ee/joint_state": _joint_state(ee_state),
        f"{ARM_PREFIX}/arm/command": _joint_command(
            arm_target, session_id=ARM_SESSION
        ),
        f"{EE_PREFIX}/ee/command": _joint_command(
            ee_target, session_id=EE_SESSION
        ),
        f"{ARM_PREFIX}/status": _robot_status(ARM_SESSION),
        f"{EE_PREFIX}/status": _robot_status(EE_SESSION),
    }


def _tap_config(**overrides) -> HexZenohCommandTapConfig:
    values = {
        "id": "test-command-tap",
        "arm_prefix": ARM_PREFIX,
        "ee_prefix": EE_PREFIX,
        "zenoh_settle_time_s": 0,
        "connect_timeout_s": 0.1,
        "max_command_age_s": 0.100,
        "max_state_age_s": 0.100,
        "max_status_age_s": 0.300,
    }
    values.update(overrides)
    return HexZenohCommandTapConfig(**values)


def _connected_tap(monkeypatch, payloads=None, **config_overrides):
    session = _FakeSession(_tap_payloads() if payloads is None else payloads)
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: session)
    teleop = HexZenohCommandTap(_tap_config(**config_overrides))
    teleop.connect()
    return teleop, session


def _update_command(cache, values, *, session_id, seq, received_at) -> None:
    cache.update(
        values,
        source_seq=seq,
        source_session_id=session_id,
        source_timeout_behavior=arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
        received_at=received_at,
    )


def test_observer_reads_follower_state_without_publishing(monkeypatch):
    session = _FakeSession(
        {
            f"{ARM_PREFIX}/arm/joint_state": _joint_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            f"{EE_PREFIX}/ee/joint_state": _joint_state([0.07]),
        }
    )
    monkeypatch.setattr(observer_module, "open_zenoh_session", lambda _endpoint: session)
    robot = HexZenohObserver(
        HexZenohObserverConfig(
            id="test-follower",
            arm_prefix=ARM_PREFIX,
            ee_prefix=EE_PREFIX,
            zenoh_settle_time_s=0,
            connect_timeout_s=0.1,
            max_state_age_s=1,
        )
    )

    robot.connect()
    observation = robot.get_observation()

    assert list(observation) == [
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "jaw_joint1.pos",
    ]
    assert list(observation.values()) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07])
    assert robot.send_action(observation) == pytest.approx(observation)
    assert session.declared_keys == [
        f"{ARM_PREFIX}/arm/joint_state",
        f"{EE_PREFIX}/ee/joint_state",
    ]

    robot.disconnect()
    assert session.closed
    assert all(subscriber.closed for subscriber in session.subscribers)


def test_command_tap_validates_requested_single_point_and_is_read_only(monkeypatch):
    session = _FakeSession(_tap_payloads())
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: session)
    teleop = HexZenohCommandTap(_tap_config())

    teleop.connect()
    assert teleop.get_action() == pytest.approx(
        {
            "joint_1.pos": 0.11,
            "joint_2.pos": -0.19,
            "joint_3.pos": 1.01,
            "joint_4.pos": 0.11,
            "joint_5.pos": 0.21,
            "joint_6.pos": -0.29,
            "jaw_joint1.pos": 0.305,
        }
    )
    assert [key for key, _payload, _timeout in session.get_calls] == [
        f"{ARM_PREFIX}/arm/description",
        f"{EE_PREFIX}/ee/description",
    ]
    assert session.declared_keys == [
        f"{ARM_PREFIX}/arm/joint_state",
        f"{EE_PREFIX}/ee/joint_state",
        f"{ARM_PREFIX}/arm/command",
        f"{EE_PREFIX}/ee/command",
        f"{ARM_PREFIX}/status",
        f"{EE_PREFIX}/status",
    ]

    teleop.disconnect()
    assert session.closed
    assert all(subscriber.closed for subscriber in session.subscribers)


def test_command_tap_rejects_multi_point_trajectory_and_rolls_back(monkeypatch):
    payloads = _tap_payloads()
    payloads[f"{ARM_PREFIX}/arm/command"] = _multi_point_command(ARM_TARGET)
    session = _FakeSession(payloads)
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: session)
    teleop = HexZenohCommandTap(_tap_config(connect_timeout_s=0.01))

    with pytest.raises(TimeoutError, match="recording requires exactly one"):
        teleop.connect()

    assert session.closed
    assert all(subscriber.closed for subscriber in session.subscribers)


def test_command_tap_rejects_target_inside_controller_limit_but_outside_recording_margin(
    monkeypatch,
):
    arm_state = ARM_STATE.copy()
    arm_state[1] = -1.56
    arm_target = ARM_TARGET.copy()
    arm_target[1] = -1.56
    teleop, _session = _connected_tap(
        monkeypatch,
        _tap_payloads(arm_state=arm_state, arm_target=arm_target),
    )

    try:
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"joint_2 requested target=.*outside.*inward_margin=0.020000",
        ):
            teleop.get_action()
        assert teleop._last_recorded is None
        assert teleop._session_pair is None
    finally:
        teleop.disconnect()


def test_command_tap_seeds_first_valid_command_despite_tracking_error(monkeypatch):
    arm_target = ARM_TARGET.copy()
    arm_target[2] = ARM_STATE[2] + 0.10
    teleop, _session = _connected_tap(
        monkeypatch,
        _tap_payloads(arm_target=arm_target),
    )

    try:
        action = teleop.get_action()
        assert action["joint_3.pos"] == pytest.approx(ARM_STATE[2] + 0.10)
        assert teleop._last_recorded == pytest.approx(arm_target + EE_TARGET)
        assert teleop._session_pair == (ARM_SESSION, EE_SESSION)
    finally:
        teleop.disconnect()


def test_command_tap_checks_cumulative_step_from_last_recorded_action(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        assert teleop.get_action()["joint_1.pos"] == pytest.approx(0.11)
        received_at = time.monotonic()
        intermediate = ARM_TARGET.copy()
        for value in (0.12, 0.13, 0.154):
            intermediate[0] = value
            _update_command(
                teleop._arm_command,
                intermediate,
                session_id=ARM_SESSION,
                seq=2,
                received_at=received_at,
            )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )

        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"joint_1 step from previous accepted command is 0.044",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_reseeds_step_reference_after_long_sampling_gap(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        assert teleop._last_accepted_at is not None
        teleop._last_accepted_at -= 2.0
        received_at = time.monotonic()
        arm_target = ARM_TARGET.copy()
        arm_target[1] -= 0.073
        _update_command(
            teleop._arm_command,
            arm_target,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)

        action = teleop.get_action()

        assert action["joint_2.pos"] == pytest.approx(arm_target[1])
        assert teleop._last_recorded == pytest.approx(arm_target + EE_TARGET)
    finally:
        teleop.disconnect()


def test_command_tap_uses_live_velocity_limit_when_tighter(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)
    assert teleop._arm_limits is not None
    teleop._arm_limits = replace(
        teleop._arm_limits,
        vel_max=(0.6,) + teleop._arm_limits.vel_max[1:],
    )

    try:
        teleop.get_action()
        received_at = time.monotonic()
        arm_target = ARM_TARGET.copy()
        arm_target[0] += 0.025
        _update_command(
            teleop._arm_command,
            arm_target,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"joint_1 step from previous accepted command.*limit 0.020",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_accepts_approved_scheduler_margin_steps(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        received_at = time.monotonic()
        arm_target = ARM_TARGET.copy()
        arm_target[5] += 0.043258175
        ee_target = [EE_TARGET[0] + 0.015648216]
        _update_command(
            teleop._arm_command,
            arm_target,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            ee_target,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)

        action = teleop.get_action()

        assert action["joint_6.pos"] == pytest.approx(arm_target[5])
        assert action["jaw_joint1.pos"] == pytest.approx(ee_target[0])
        assert teleop._last_recorded[-1] == pytest.approx(ee_target[0])
    finally:
        teleop.disconnect()


def test_command_tap_rejects_step_above_approved_scheduler_margin(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        received_at = time.monotonic()
        ee_target = [EE_TARGET[0] + 0.0166]
        _update_command(
            teleop._arm_command,
            ARM_TARGET,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            ee_target,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)

        with pytest.raises(
            UnsafeRecordedActionError,
            match=(
                r"jaw_joint1 step from previous accepted command.*"
                r"limit 0.016500000"
            ),
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_rejects_session_change_after_first_valid_command(monkeypatch):
    teleop, session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        assert teleop._last_accepted_at is not None
        teleop._last_accepted_at -= 2.0
        first_target = teleop._last_recorded
        received_at = time.monotonic()
        new_arm_session = ARM_SESSION + 1
        _update_command(
            teleop._arm_command,
            ARM_TARGET,
            session_id=new_arm_session,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)
        session.emit(f"{ARM_PREFIX}/status", _robot_status(new_arm_session))

        with pytest.raises(
            UnsafeRecordedActionError,
            match=(
                r"controller session changed during recording; "
                r"expected arm=11, EE=22, observed arm=12, EE=22"
            ),
        ):
            teleop.get_action()
        assert teleop._last_recorded == first_target
        assert teleop._session_pair == (ARM_SESSION, EE_SESSION)
    finally:
        teleop.disconnect()


def test_rejected_candidate_does_not_advance_step_reference(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        received_at = time.monotonic()
        rejected = ARM_TARGET.copy()
        rejected[0] = 0.155
        _update_command(
            teleop._arm_command,
            rejected,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        with pytest.raises(UnsafeRecordedActionError):
            teleop.get_action()

        later = ARM_TARGET.copy()
        later[0] = 0.18
        _update_command(
            teleop._arm_command,
            later,
            session_id=ARM_SESSION,
            seq=3,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=3,
            received_at=received_at,
        )
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"reference=0.109999999, target=0.180000000",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_malformed_command_after_valid_sample_never_reuses_old_action(monkeypatch):
    teleop, session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        session.emit(f"{ARM_PREFIX}/arm/command", b"\xff")
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"invalid follower arm command: invalid JointTrajectory protobuf",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_rejects_status_holder_that_does_not_match_command(monkeypatch):
    payloads = _tap_payloads()
    payloads[f"{ARM_PREFIX}/status"] = _robot_status(99)
    teleop, _session = _connected_tap(monkeypatch, payloads)

    try:
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"arm controller cannot accept command session_id=11:.*holder=99",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_invalid_non_q_setpoint_field_after_valid_never_reuses_old_action(monkeypatch):
    teleop, session = _connected_tap(monkeypatch)

    try:
        teleop.get_action()
        malformed = arm_pb2.JointSetpoint(q=ARM_TARGET, kp=[1.0])
        session.emit(
            f"{ARM_PREFIX}/arm/command",
            _joint_command(
                ARM_TARGET,
                seq=2,
                session_id=ARM_SESSION,
                point=malformed,
            ),
        )
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"JointSetpoint kp has 1 values, expected 6 or zero",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()



@pytest.mark.parametrize(
    ("field_name", "values", "match"),
    [
        ("kp", [-1.0] * 6, r"JointSetpoint kp contains a negative gain"),
        ("kd", [-1.0] * 6, r"JointSetpoint kd contains a negative gain"),
        ("dq", [2.1] * 6, r"requested dq=.*exceeds.*vel_max"),
        ("tau_ff", [1.1] * 6, r"requested tau_ff=.*exceeds.*tau_max"),
    ],
)
def test_controller_rejected_setpoint_vectors_never_become_actions(
    monkeypatch, field_name, values, match
):
    teleop, session = _connected_tap(monkeypatch)
    try:
        teleop.get_action()
        point = arm_pb2.JointSetpoint(q=ARM_TARGET, **{field_name: values})
        session.emit(
            f"{ARM_PREFIX}/arm/command",
            _joint_command(
                ARM_TARGET,
                seq=2,
                session_id=ARM_SESSION,
                point=point,
            ),
        )
        with pytest.raises(UnsafeRecordedActionError, match=match):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_valid_nonempty_setpoint_vectors_are_checked_and_accepted(monkeypatch):
    payloads = _tap_payloads()
    point = arm_pb2.JointSetpoint(
        q=ARM_TARGET,
        dq=[0.1] * 6,
        kp=[20.0] * 6,
        kd=[2.0] * 6,
        tau_ff=[0.5] * 6,
    )
    payloads[f"{ARM_PREFIX}/arm/command"] = _joint_command(
        ARM_TARGET, session_id=ARM_SESSION, point=point
    )
    teleop, _session = _connected_tap(monkeypatch, payloads)
    try:
        assert teleop.get_action()["joint_1.pos"] == pytest.approx(ARM_TARGET[0])
    finally:
        teleop.disconnect()


def test_malformed_status_after_valid_never_reuses_old_status(monkeypatch):
    teleop, session = _connected_tap(monkeypatch)
    try:
        teleop.get_action()
        session.emit(f"{ARM_PREFIX}/status", b"\xff")
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"invalid follower arm RobotStatus: invalid RobotStatus protobuf",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_unknown_advertised_timeout_is_rejected(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        assert teleop._arm_limits is not None
        teleop._arm_limits = replace(
            teleop._arm_limits,
            supported_timeouts=frozenset({99}),
        )
        received_at = time.monotonic()
        teleop._arm_command.update(
            ARM_TARGET,
            source_seq=2,
            source_session_id=ARM_SESSION,
            source_timeout_behavior=99,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"arm command has unknown timeout behavior 99",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()



def test_command_tap_rejects_non_running_or_stale_status(monkeypatch):
    teleop, session = _connected_tap(monkeypatch)
    try:
        session.emit(
            f"{ARM_PREFIX}/status",
            command_tap_module.robot_pb2.RobotStatus(
                mode=command_tap_module.common_pb2.ROBOT_MODE_STANDBY,
                session_holder=ARM_SESSION,
            ).SerializeToString(),
        )
        with pytest.raises(UnsafeRecordedActionError, match=r"mode=ROBOT_MODE_STANDBY"):
            teleop.get_action()

        session.emit(f"{ARM_PREFIX}/status", _robot_status(ARM_SESSION))
        with teleop._arm_status._lock:
            assert teleop._arm_status._snapshot is not None
            teleop._arm_status._snapshot = replace(
                teleop._arm_status._snapshot,
                received_at=time.monotonic() - 0.101,
            )
        assert teleop.get_action()["joint_1.pos"] == pytest.approx(ARM_TARGET[0])

        with teleop._arm_status._lock:
            assert teleop._arm_status._snapshot is not None
            teleop._arm_status._snapshot = replace(
                teleop._arm_status._snapshot,
                received_at=time.monotonic() - 0.301,
            )
        with pytest.raises(UnsafeRecordedActionError, match=r"stale follower arm RobotStatus"):
            teleop.get_action()
    finally:
        teleop.disconnect()


@pytest.mark.parametrize(
    ("timeout_behavior", "match"),
    [
        (
            arm_pb2.TIMEOUT_BEHAVIOR_UNSPECIFIED,
            r"arm command has no explicit timeout behavior",
        ),
        (
            arm_pb2.TIMEOUT_BEHAVIOR_RAMP_STOP,
            r"arm command timeout TIMEOUT_BEHAVIOR_RAMP_STOP is not advertised",
        ),
    ],
)
def test_command_tap_rejects_missing_or_unadvertised_timeout(
    monkeypatch, timeout_behavior, match
):
    teleop, _session = _connected_tap(monkeypatch)
    try:
        received_at = time.monotonic()
        teleop._arm_command.update(
            ARM_TARGET,
            source_seq=2,
            source_session_id=ARM_SESSION,
            source_timeout_behavior=timeout_behavior,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        with pytest.raises(UnsafeRecordedActionError, match=match):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_rejects_ee_command_state_receive_skew(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)
    try:
        command_at = time.monotonic() - 0.021
        _update_command(
            teleop._arm_command,
            ARM_TARGET,
            session_id=ARM_SESSION,
            seq=2,
            received_at=command_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=command_at,
        )
        teleop._arm_state.update(
            ARM_STATE, source_seq=2, received_at=command_at + 0.020
        )
        teleop._ee_state.update(
            EE_STATE, source_seq=2, received_at=command_at + 0.021
        )
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"EE command/joint_state pair receive skew is 21.000 ms",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()

def test_command_tap_rejects_command_state_receive_skew(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        received_at = time.monotonic()
        _update_command(
            teleop._arm_command,
            ARM_TARGET,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(
            ARM_STATE, source_seq=2, received_at=received_at - 0.021
        )
        teleop._ee_state.update(
            EE_STATE, source_seq=2, received_at=received_at - 0.021
        )

        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"arm command/joint_state pair receive skew is 21.000 ms",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_command_tap_rejects_arm_ee_receive_skew(monkeypatch):
    teleop, _session = _connected_tap(monkeypatch)

    try:
        received_at = time.monotonic()
        _update_command(
            teleop._arm_command,
            ARM_TARGET,
            session_id=ARM_SESSION,
            seq=10,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=11,
            received_at=received_at - 0.021,
        )
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"command pair receive skew is 21.000 ms.*arm_seq=10, ee_seq=11",
        ):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_latest_vector_rejects_new_error_after_valid_snapshot():
    vector = LatestJointVector(1, "EE command")
    vector.update([0.2])
    vector.remember_error("malformed replacement")

    with pytest.raises(ValueError, match="invalid EE command: malformed replacement"):
        vector.get_fresh(0.1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_command_age_s": 0.101}, r"must not exceed 0\.1"),
        ({"max_state_age_s": 0.101}, r"must not exceed 0\.1"),
        ({"max_status_age_s": 0.301}, r"must not exceed 0\.3"),
        ({"max_pair_skew_s": 0.021}, r"must not exceed 0\.02"),
        ({"arm_step_limit_rad": 0.0436}, r"must not exceed 0\.0435"),
        ({"ee_step_limit_rad": 0.0166}, r"must not exceed 0\.0165"),
        ({"record_hz": 29.0}, "record_hz must be 30"),
        ({"arm_position_margin_rad": 0.019}, r"must be at least 0\.02"),
        ({"ee_position_margin_rad": 0.004}, r"must be at least 0\.005"),
        ({"arm_position_margin_rad": -0.001}, "finite and non-negative"),
    ],
)
def test_command_tap_config_rejects_weakened_recording_contract(overrides, match):
    with pytest.raises(ValueError, match=match):
        _tap_config(**overrides)


def test_command_tap_exports_exact_live_recording_contract(monkeypatch):
    session = _FakeSession(_tap_payloads())
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: session)
    teleop = HexZenohCommandTap(_tap_config())

    with pytest.raises(ConnectionError, match="not connected"):
        teleop.controller_contract()

    try:
        teleop.connect()
        contract = teleop.controller_contract()
    finally:
        teleop.disconnect()

    assert contract["schema"] == "firefly_recording_contract_v1"
    assert contract["joint_names"] == [
        *(f"joint_{index}.pos" for index in range(1, 7)),
        "jaw_joint1.pos",
    ]
    assert contract["pos_min"] == pytest.approx([-2.86, -1.57, 0, -1.57, -1.54, -2.79, 0])
    assert contract["pos_max"] == pytest.approx([2.86, 2.09, 3.14, 1.57, 1.54, 2.79, 0.6])
    assert contract["position_margins"] == pytest.approx([0.02] * 6 + [0.005])
    assert contract["effective_step_limits"] == pytest.approx([0.0435] * 6 + [0.0165])


def test_invalid_dimension_is_reported_instead_of_becoming_ready():
    vector = LatestJointVector(6, "arm state")
    vector.update([1, 2])

    with pytest.raises(TimeoutError, match="expected 6 values, received 2"):
        vector.wait_ready(0.001)


def test_stale_sample_is_rejected():
    vector = LatestJointVector(1, "EE state")
    vector.update([0.2], received_at=time.monotonic() - 1)

    with pytest.raises(TimeoutError, match="stale EE state"):
        vector.get_fresh(0.1)


def test_example_record_config_deserializes():
    config_path = Path(__file__).parents[1] / "configs" / "record_firefly_y6.example.yaml"
    with config_path.open() as stream:
        config = draccus.load(RecordConfig, stream)

    assert isinstance(config.robot, HexZenohObserverConfig)
    assert isinstance(config.teleop, HexZenohCommandTapConfig)
    assert config.robot.arm_prefix == (
        "hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm1"
    )
    assert set(config.robot.cameras) == {"scene", "wrist"}
    assert all(camera.type == "intelrealsense" for camera in config.robot.cameras.values())
    assert config.robot.max_state_age_s == pytest.approx(0.1)
    assert config.robot.cameras["wrist"].rotation == 180
    assert config.teleop.max_command_age_s == pytest.approx(0.1)
    assert config.teleop.max_state_age_s == pytest.approx(0.1)
    assert config.teleop.max_status_age_s == pytest.approx(0.3)
    assert config.teleop.max_pair_skew_s == pytest.approx(0.02)
    assert config.teleop.record_hz == pytest.approx(30)
    assert config.teleop.arm_step_limit_rad == pytest.approx(0.0435)
    assert config.teleop.ee_step_limit_rad == pytest.approx(0.0165)
    assert config.teleop.arm_position_margin_rad == pytest.approx(0.02)
    assert config.teleop.ee_position_margin_rad == pytest.approx(0.005)
    assert config.dataset.repo_id == "local/firefly_y6_manual_validated_v2"
    assert str(config.dataset.root) == (
        "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2"
    )
    assert config.dataset.fps == 30
    assert config.dataset.episode_time_s == 600
    assert config.dataset.num_episodes == 20
    assert config.dataset.push_to_hub is False


class _FakeDataset:
    def __init__(self, feature_names: list[str]):
        vector_feature = {
            "dtype": "float32",
            "shape": (len(feature_names),),
            "names": feature_names,
        }
        self.fps = 30
        self.features = {
            "observation.state": vector_feature,
            "action": vector_feature,
        }
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)


def test_lerobot_record_loop_builds_state_and_validated_action_frames(monkeypatch):
    state_session = _FakeSession(
        {
            f"{ARM_PREFIX}/arm/joint_state": _joint_state(ARM_STATE),
            f"{EE_PREFIX}/ee/joint_state": _joint_state(EE_STATE),
        }
    )
    command_session = _FakeSession(_tap_payloads())
    monkeypatch.setattr(observer_module, "open_zenoh_session", lambda _endpoint: state_session)
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: command_session)

    common = {
        "arm_prefix": ARM_PREFIX,
        "ee_prefix": EE_PREFIX,
        "zenoh_settle_time_s": 0,
        "connect_timeout_s": 0.1,
    }
    robot = HexZenohObserver(HexZenohObserverConfig(max_state_age_s=1, **common))
    teleop = HexZenohCommandTap(_tap_config())
    names = list(robot.action_features)
    dataset = _FakeDataset(names)
    teleop_processor, action_processor, observation_processor = make_default_processors()

    try:
        robot.connect()
        teleop.connect()
        record_loop(
            robot=robot,
            events={"exit_early": False},
            fps=30,
            teleop_action_processor=teleop_processor,
            robot_action_processor=action_processor,
            robot_observation_processor=observation_processor,
            dataset=dataset,
            teleop=teleop,
            control_time_s=0.04,
            single_task="test task",
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert dataset.frames
    frame = dataset.frames[0]
    assert frame["observation.state"].tolist() == pytest.approx(ARM_STATE + EE_STATE)
    assert frame["action"].tolist() == pytest.approx(ARM_TARGET + EE_TARGET)
    assert frame["task"] == "test task"


def test_lerobot_record_loop_never_adds_unsafe_action_frame(monkeypatch):
    state_session = _FakeSession(
        {
            f"{ARM_PREFIX}/arm/joint_state": _joint_state(ARM_STATE),
            f"{EE_PREFIX}/ee/joint_state": _joint_state(EE_STATE),
        }
    )
    command_session = _FakeSession(_tap_payloads())
    monkeypatch.setattr(observer_module, "open_zenoh_session", lambda _endpoint: state_session)
    monkeypatch.setattr(command_tap_module, "open_zenoh_session", lambda _endpoint: command_session)

    common = {
        "arm_prefix": ARM_PREFIX,
        "ee_prefix": EE_PREFIX,
        "zenoh_settle_time_s": 0,
        "connect_timeout_s": 0.1,
    }
    robot = HexZenohObserver(HexZenohObserverConfig(max_state_age_s=1, **common))
    teleop = HexZenohCommandTap(_tap_config())
    dataset = _FakeDataset(list(robot.action_features))
    teleop_processor, action_processor, observation_processor = make_default_processors()

    try:
        robot.connect()
        teleop.connect()
        teleop.get_action()
        received_at = time.monotonic()
        unsafe_target = ARM_TARGET.copy()
        unsafe_target[0] += 0.10
        _update_command(
            teleop._arm_command,
            unsafe_target,
            session_id=ARM_SESSION,
            seq=2,
            received_at=received_at,
        )
        _update_command(
            teleop._ee_command,
            EE_TARGET,
            session_id=EE_SESSION,
            seq=2,
            received_at=received_at,
        )
        teleop._arm_state.update(ARM_STATE, source_seq=2, received_at=received_at)
        teleop._ee_state.update(EE_STATE, source_seq=2, received_at=received_at)
        with pytest.raises(
            UnsafeRecordedActionError,
            match=r"step from previous accepted command",
        ):
            record_loop(
                robot=robot,
                events={"exit_early": False},
                fps=30,
                teleop_action_processor=teleop_processor,
                robot_action_processor=action_processor,
                robot_observation_processor=observation_processor,
                dataset=dataset,
                teleop=teleop,
                control_time_s=0.04,
                single_task="test task",
            )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert dataset.frames == []
