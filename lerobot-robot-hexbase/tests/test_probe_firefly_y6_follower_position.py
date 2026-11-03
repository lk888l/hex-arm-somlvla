from __future__ import annotations

import math
import socket
import struct
from pathlib import Path

import pytest

from scripts import probe_firefly_y6_follower_position as probe


def _response(
    node_id: int,
    raw_revolutions: float = 0.125,
    *,
    command: int = probe.SDO_UPLOAD_RESPONSE_4_BYTES,
    index: int = probe.POSITION_INDEX,
    subindex: int = probe.POSITION_SUBINDEX,
    can_id: int | None = None,
    data_length: int = 8,
    value_bytes: bytes | None = None,
) -> bytes:
    if value_bytes is None:
        value_bytes = struct.pack("<f", raw_revolutions)
    data = bytes(
        (
            command,
            index & 0xFF,
            index >> 8,
            subindex,
        )
    ) + value_bytes
    return probe._CAN_FRAME.pack(
        probe.response_can_id(node_id) if can_id is None else can_id,
        data_length,
        data,
    )


class _FakeBus:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.timeouts: list[float | None] = []
        self.blocking: list[bool] = []
        self.sockopts: list[tuple[int, int, bytes | int]] = []
        self.bound: list[tuple[str]] = []
        self.closed = False

    def bind(self, address):
        self.bound.append(address)

    def close(self):
        self.closed = True

    def recv(self, _size):
        if not self.responses:
            raise socket.timeout("no response")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def setblocking(self, flag):
        self.blocking.append(flag)

    def setsockopt(self, level, option, value):
        self.sockopts.append((level, option, value))

    def settimeout(self, value):
        self.timeouts.append(value)


def _clock_for_six_nodes() -> callable:
    values = iter(value for node in range(6) for value in (node * 1.0, node * 1.0 + 0.01))
    return lambda: next(values)


def test_protocol_and_mapping_are_immutable() -> None:
    assert probe.CAN_INTERFACE == "can1"
    assert probe.NODE_IDS == (1, 2, 3, 4, 5, 6)
    assert probe.POSITION_INDEX == 0x6064
    assert probe.POSITION_SUBINDEX == 0
    assert probe.JOINT_SIGN == (-1, -1, 1, 1, 1, 1)
    assert probe.ZERO_OFFSET_RAD == (0, 0, math.pi / 2, 0, 0, 0)
    assert probe.sdo_upload_payload() == bytes.fromhex("4064600000000000")
    assert probe.printable_requests() == tuple(
        f"can1 node={node} can_id=0x{0x600 + node:03X} data=4064600000000000"
        for node in range(1, 7)
    )


def test_cli_has_no_hardware_or_protocol_overrides() -> None:
    assert probe.parse_args(["--print-requests"]).print_requests
    assert probe.parse_args(["--read"]).read

    with pytest.raises(SystemExit):
        probe.parse_args([])
    with pytest.raises(SystemExit):
        probe.parse_args(["--read", "--print-requests"])
    for forbidden in (
        ["--read", "--interface", "can0"],
        ["--read", "--node-id", "7"],
        ["--read", "--index", "0x6040"],
        ["--read", "--subindex", "1"],
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(forbidden)


def test_print_requests_never_preflights_opens_or_reads(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        probe,
        "run_preflight",
        lambda: pytest.fail("--print-requests must not inspect hardware"),
    )
    monkeypatch.setattr(
        probe,
        "open_can1_socket",
        lambda: pytest.fail("--print-requests must not open CAN"),
    )
    monkeypatch.setattr(
        probe,
        "read_positions",
        lambda: pytest.fail("--print-requests must not read CAN"),
    )

    assert probe.main(["--print-requests"]) == 0
    output = capsys.readouterr().out
    assert output.count("can1 node=") == 6
    assert "--read" in output


def test_process_preflight_detects_every_locked_controller_name(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    expected: list[tuple[int, str]] = []
    for pid, name in enumerate(sorted(probe.CONTROL_PROCESS_NAMES), start=100):
        pid_dir = proc / str(pid)
        pid_dir.mkdir()
        (pid_dir / "comm").write_text(name, encoding="utf-8")
        expected.append((pid, name))
    other = proc / "999"
    other.mkdir()
    (other / "comm").write_text("python", encoding="utf-8")

    assert probe.related_control_processes(proc, current_pid=1) == tuple(expected)
    with pytest.raises(probe.ProbeError, match="arm_controller"):
        probe.require_no_control_processes(proc)


def test_process_preflight_checks_executable_name_when_comm_is_truncated(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    pid_dir = proc / "123"
    pid_dir.mkdir(parents=True)
    (pid_dir / "comm").write_text("hex-hw-supervi", encoding="utf-8")
    executable = tmp_path / "hex-hw-supervisor"
    executable.write_text("", encoding="utf-8")
    (pid_dir / "exe").symlink_to(executable)

    assert probe.related_control_processes(proc, current_pid=1) == (
        (123, "hex-hw-supervisor"),
    )


def test_can1_must_be_up_and_must_be_a_can_interface(tmp_path: Path) -> None:
    net = tmp_path / "net"
    interface = net / "can1"
    interface.mkdir(parents=True)
    (interface / "flags").write_text("0x1\n", encoding="ascii")
    (interface / "type").write_text("280\n", encoding="ascii")

    probe.require_can1_up(net)

    (interface / "flags").write_text("0x0\n", encoding="ascii")
    with pytest.raises(probe.ProbeError, match="not UP"):
        probe.require_can1_up(net)
    (interface / "flags").write_text("0x1\n", encoding="ascii")
    (interface / "type").write_text("1\n", encoding="ascii")
    with pytest.raises(probe.ProbeError, match="ARPHRD"):
        probe.require_can1_up(net)


def test_open_socket_binds_only_can1_and_filters_only_sdo_responses() -> None:
    created: list[tuple[int, int, int]] = []
    bus = _FakeBus()

    def factory(family, kind, protocol):
        created.append((family, kind, protocol))
        return bus

    assert probe.open_can1_socket(factory) is bus
    assert created == [(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)]
    assert bus.bound == [("can1",)]
    assert len(bus.sockopts) == 1
    filters = bus.sockopts[0][2]
    assert isinstance(filters, bytes)
    assert len(filters) == 6 * probe._CAN_FILTER.size


def test_preexisting_sdo_response_is_rejected_as_not_fresh() -> None:
    bus = _FakeBus([_response(1)])
    with pytest.raises(probe.ProbeError, match="pre-existing.*freshness"):
        probe.require_empty_sdo_receive_queue(bus)
    assert bus.blocking == [False, True]

    empty = _FakeBus([BlockingIOError()])
    probe.require_empty_sdo_receive_queue(empty)
    assert empty.blocking == [False, True]


def test_parse_response_accepts_only_fresh_matching_6064_float() -> None:
    assert probe.parse_position_response(
        _response(3, 0.227164865),
        expected_node_id=3,
        sent_at=10.0,
        received_at=10.01,
    ) == pytest.approx(0.227164865, abs=1e-8)

    with pytest.raises(probe.ProbeError, match="not fresh"):
        probe.parse_position_response(
            _response(3),
            expected_node_id=3,
            sent_at=10.0,
            received_at=10.251,
        )
    with pytest.raises(probe.ProbeError, match="out-of-order"):
        probe.parse_position_response(
            _response(2),
            expected_node_id=3,
            sent_at=10.0,
            received_at=10.01,
        )
    with pytest.raises(probe.ProbeError, match="wrong object"):
        probe.parse_position_response(
            _response(3, index=0x6040),
            expected_node_id=3,
            sent_at=10.0,
            received_at=10.01,
        )
    with pytest.raises(probe.ProbeError, match="command 0x4B"):
        probe.parse_position_response(
            _response(3, command=0x4B),
            expected_node_id=3,
            sent_at=10.0,
            received_at=10.01,
        )
    with pytest.raises(probe.ProbeError, match="non-finite"):
        probe.parse_position_response(
            _response(3, value_bytes=struct.pack("<f", math.nan)),
            expected_node_id=3,
            sent_at=10.0,
            received_at=10.01,
        )


def test_sdo_abort_is_rejected_with_abort_code() -> None:
    frame = _response(
        4,
        command=probe.SDO_ABORT_RESPONSE,
        value_bytes=(0x06020000).to_bytes(4, "little"),
    )
    with pytest.raises(probe.ProbeError, match="0x06020000"):
        probe.parse_position_response(
            frame,
            expected_node_id=4,
            sent_at=1.0,
            received_at=1.01,
        )


def test_six_reads_send_exactly_one_locked_upload_per_node() -> None:
    raw = (0.01, 0.20, 0.22, -0.01, 0.02, 0.03)
    bus = _FakeBus([_response(node, raw[node - 1]) for node in probe.NODE_IDS])
    guards: list[str] = []

    actual = probe.read_raw_motor_revolutions(
        bus,
        clock=_clock_for_six_nodes(),
        pre_send_guard=lambda: guards.append("checked"),
    )

    assert actual == pytest.approx(raw, abs=1e-7)
    assert bus.sent == [probe.packed_request(node) for node in probe.NODE_IDS]
    assert guards == ["checked"] * 6
    for frame in bus.sent:
        can_id, length, data = probe._CAN_FRAME.unpack(frame)
        assert can_id in range(0x601, 0x607)
        assert length == 8
        assert data == bytes.fromhex("4064600000000000")


def test_guard_failure_prevents_the_next_request() -> None:
    bus = _FakeBus([_response(1), _response(2)])
    checks = 0

    def guard() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise probe.ProbeError("controller appeared")

    clock_values = iter((0.0, 0.01, 1.0, 1.01))
    with pytest.raises(probe.ProbeError, match="controller appeared"):
        probe.read_raw_motor_revolutions(
            bus,
            clock=lambda: next(clock_values),
            pre_send_guard=guard,
        )
    assert bus.sent == [probe.packed_request(1), probe.packed_request(2)]


def test_timeout_and_out_of_order_response_stop_sequence() -> None:
    timeout_bus = _FakeBus([socket.timeout("late")])
    with pytest.raises(probe.ProbeError, match="timed out"):
        probe.read_raw_motor_revolutions(
            timeout_bus,
            clock=lambda: 0.0,
            pre_send_guard=lambda: None,
        )
    assert timeout_bus.sent == [probe.packed_request(1)]

    wrong_bus = _FakeBus([_response(2)])
    times = iter((0.0, 0.01))
    with pytest.raises(probe.ProbeError, match="out-of-order"):
        probe.read_raw_motor_revolutions(
            wrong_bus,
            clock=lambda: next(times),
            pre_send_guard=lambda: None,
        )
    assert wrong_bus.sent == [probe.packed_request(1)]


def test_integer_turn_resolution_requires_exactly_one_candidate() -> None:
    turns, q = probe.resolve_integer_turn(
        1.01,
        sign=-1,
        zero_offset_rad=0.0,
        lower_rad=-2.86,
        upper_rad=2.86,
    )
    assert turns == -1
    assert q == pytest.approx(-0.01 * math.tau)

    with pytest.raises(probe.ProbeError, match="candidates=0"):
        probe.resolve_integer_turn(
            0.5,
            sign=1,
            zero_offset_rad=0.0,
            lower_rad=-1.57,
            upper_rad=1.57,
        )
    with pytest.raises(probe.ProbeError, match="candidates=3"):
        probe.resolve_integer_turn(
            0.0,
            sign=1,
            zero_offset_rad=0.0,
            lower_rad=-math.tau,
            upper_rad=math.tau,
        )


def test_fixed_joint_mapping_and_live_limits() -> None:
    raw = (0.0025, 0.20, 0.227, -0.001, 0.002, 0.006)
    positions = probe.resolve_joint_positions(raw)

    assert [item.integer_turns for item in positions] == [0] * 6
    assert [item.joint_position_rad for item in positions] == pytest.approx(
        [
            -math.tau * raw[0],
            -math.tau * raw[1],
            math.pi / 2 + math.tau * raw[2],
            math.tau * raw[3],
            math.tau * raw[4],
            math.tau * raw[5],
        ]
    )

    # The observed powered-off J2 sample is a few milliradians past the fixed live
    # minimum and therefore must be reported as unsafe, never silently clamped.
    with pytest.raises(probe.ProbeError, match="candidates=0"):
        probe.resolve_joint_positions(
            (0.002591848, 0.25084579, 0.227164865, -0.001460254, 0.002845764, 0.00680542)
        )


def test_read_mode_failure_is_fail_closed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        probe,
        "read_positions",
        lambda: (_ for _ in ()).throw(probe.ProbeError("unsafe state")),
    )
    assert probe.main(["--read"]) == 2
    assert "READ REFUSED: unsafe state" in capsys.readouterr().err
