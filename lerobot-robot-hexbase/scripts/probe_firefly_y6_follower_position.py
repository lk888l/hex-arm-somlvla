#!/usr/bin/env python3
"""Fail-closed, read-only position probe for the Firefly Y6 follower on CAN1.

The command surface is deliberately locked.  ``--print-requests`` only prints the
six fixed CANopen SDO upload requests.  ``--read`` is the sole mode that opens
``can1`` and sends them.  There are no command-line overrides for the interface,
node IDs, object dictionary entry, joint mapping, or limits.

This tool never sends NMT, controlwords, PDOs, or SDO downloads.  It must only be
used while every Hex control process is stopped normally.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


CAN_INTERFACE = "can1"
NODE_IDS = (1, 2, 3, 4, 5, 6)
POSITION_INDEX = 0x6064
POSITION_SUBINDEX = 0x00

JOINT_SIGN = (-1, -1, 1, 1, 1, 1)
ZERO_OFFSET_RAD = (0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0)
LIVE_JOINT_LIMITS_RAD = (
    (-2.86, 2.86),
    (-1.57, 2.09),
    (0.0, 3.14),
    (-1.57, 1.57),
    (-1.54, 1.54),
    (-2.79, 2.79),
)

SDO_UPLOAD_COMMAND = 0x40
SDO_UPLOAD_RESPONSE_4_BYTES = 0x43
SDO_ABORT_RESPONSE = 0x80
SDO_REQUEST_BASE_ID = 0x600
SDO_RESPONSE_BASE_ID = 0x580
RESPONSE_TIMEOUT_S = 0.250

# These are the deployed processes that can own or command either physical arm.
# Exact executable/comm names are used; arbitrary process-name input is forbidden.
CONTROL_PROCESS_NAMES = frozenset(
    {
        "arm_controller",
        "arm_teleop",
        "ee_controller",
        "hex-controller",
        "hex-hw-supervisor",
        "hex-zenoh-policy",
        "launcher",
        "lift_controller",
    }
)

_CAN_FRAME = struct.Struct("=IB3x8s")
_CAN_FILTER = struct.Struct("=II")
_CAN_SFF_MASK = 0x7FF
_CAN_EFF_FLAG = 0x80000000
_CAN_RTR_FLAG = 0x40000000
_CAN_ERR_FLAG = 0x20000000
_CAN_ID_FLAGS = _CAN_EFF_FLAG | _CAN_RTR_FLAG | _CAN_ERR_FLAG
_IFF_UP = 0x1
_ARPHRD_CAN = 280


class ProbeError(RuntimeError):
    """A precondition or protocol invariant failed; no further request is safe."""


class _CanSocket(Protocol):
    def bind(self, address: tuple[str]) -> None: ...

    def close(self) -> None: ...

    def recv(self, size: int) -> bytes: ...

    def send(self, data: bytes) -> int: ...

    def setblocking(self, flag: bool) -> None: ...

    def setsockopt(self, level: int, option: int, value: bytes | int) -> None: ...

    def settimeout(self, value: float | None) -> None: ...


@dataclass(frozen=True)
class JointPosition:
    node_id: int
    raw_motor_revolutions: float
    integer_turns: int
    joint_position_rad: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--print-requests",
        action="store_true",
        help="Print the six immutable SDO upload frames; do not inspect or open CAN.",
    )
    action.add_argument(
        "--read",
        action="store_true",
        help="After fail-closed preflight, send only the six printed upload requests.",
    )
    return parser.parse_args(argv)


def sdo_upload_payload() -> bytes:
    return bytes(
        (
            SDO_UPLOAD_COMMAND,
            POSITION_INDEX & 0xFF,
            POSITION_INDEX >> 8,
            POSITION_SUBINDEX,
            0,
            0,
            0,
            0,
        )
    )


def request_can_id(node_id: int) -> int:
    if node_id not in NODE_IDS:
        raise ValueError("only the hard-coded follower node IDs 1..6 are allowed")
    return SDO_REQUEST_BASE_ID + node_id


def response_can_id(node_id: int) -> int:
    if node_id not in NODE_IDS:
        raise ValueError("only the hard-coded follower node IDs 1..6 are allowed")
    return SDO_RESPONSE_BASE_ID + node_id


def packed_request(node_id: int) -> bytes:
    return _CAN_FRAME.pack(request_can_id(node_id), 8, sdo_upload_payload())


def printable_requests() -> tuple[str, ...]:
    data = sdo_upload_payload().hex().upper()
    return tuple(
        f"{CAN_INTERFACE} node={node_id} can_id=0x{request_can_id(node_id):03X} data={data}"
        for node_id in NODE_IDS
    )


def _read_process_name(pid_dir: Path) -> str | None:
    comm_path = pid_dir / "comm"
    try:
        name = comm_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        name = ""
    except OSError as exc:
        raise ProbeError(f"cannot inspect {comm_path}: {exc}") from exc
    if name in CONTROL_PROCESS_NAMES:
        return name

    try:
        executable = (pid_dir / "exe").resolve(strict=True)
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError) as exc:
        # Kernel threads have no exe and are already covered by readable comm.  A
        # permission/loop error is different: absence of a controller is unprovable.
        raise ProbeError(f"cannot inspect {pid_dir / 'exe'}: {exc}") from exc
    return executable.name if executable.name in CONTROL_PROCESS_NAMES else None


def related_control_processes(
    proc_root: Path = Path("/proc"),
    *,
    current_pid: int | None = None,
) -> tuple[tuple[int, str], ...]:
    if current_pid is None:
        current_pid = os.getpid()
    try:
        entries: Iterable[Path] = proc_root.iterdir()
    except OSError as exc:
        raise ProbeError(f"cannot inspect running processes under {proc_root}: {exc}") from exc

    matches: list[tuple[int, str]] = []
    for pid_dir in entries:
        if not pid_dir.name.isdecimal():
            continue
        pid = int(pid_dir.name)
        if pid == current_pid:
            continue
        try:
            name = _read_process_name(pid_dir)
        except ProbeError:
            # A process may disappear between readdir and inspection.  Only that
            # expected race is safe to ignore.
            if not pid_dir.exists():
                continue
            raise
        if name is not None:
            matches.append((pid, name))
    return tuple(sorted(matches))


def require_no_control_processes(proc_root: Path = Path("/proc")) -> None:
    matches = related_control_processes(proc_root)
    if matches:
        detail = ", ".join(f"{name}(PID {pid})" for pid, name in matches)
        raise ProbeError(
            f"physical control process still running: {detail}; stop arm_teleop and "
            "launcher normally, then stop supervisor, before retrying"
        )


def require_can1_up(sys_class_net: Path = Path("/sys/class/net")) -> None:
    interface = sys_class_net / CAN_INTERFACE
    try:
        flags = int((interface / "flags").read_text(encoding="ascii").strip(), 0)
        hardware_type = int((interface / "type").read_text(encoding="ascii").strip(), 0)
    except (OSError, ValueError) as exc:
        raise ProbeError(f"cannot validate fixed interface {CAN_INTERFACE}: {exc}") from exc
    if hardware_type != _ARPHRD_CAN:
        raise ProbeError(
            f"fixed interface {CAN_INTERFACE} has ARPHRD type {hardware_type}, "
            f"expected CAN type {_ARPHRD_CAN}"
        )
    if flags & _IFF_UP == 0:
        raise ProbeError(f"fixed interface {CAN_INTERFACE} is not UP")


def run_preflight() -> None:
    require_no_control_processes()
    require_can1_up()


def _can_filters() -> bytes:
    exact_mask = _CAN_SFF_MASK | _CAN_ID_FLAGS
    return b"".join(
        _CAN_FILTER.pack(response_can_id(node_id), exact_mask) for node_id in NODE_IDS
    )


def open_can1_socket(
    socket_factory: Callable[[int, int, int], _CanSocket] = socket.socket,
) -> _CanSocket:
    try:
        bus = socket_factory(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sol_can_raw = getattr(socket, "SOL_CAN_RAW", 101)
        can_raw_filter = getattr(socket, "CAN_RAW_FILTER", 1)
        bus.setsockopt(sol_can_raw, can_raw_filter, _can_filters())
        bus.bind((CAN_INTERFACE,))
    except OSError as exc:
        try:
            bus.close()  # type: ignore[possibly-undefined]
        except (OSError, UnboundLocalError):
            pass
        raise ProbeError(f"cannot open locked raw CAN socket on {CAN_INTERFACE}: {exc}") from exc
    return bus


def require_empty_sdo_receive_queue(bus: _CanSocket) -> None:
    """Reject pre-existing SDO responses instead of mistaking them for fresh replies."""

    bus.setblocking(False)
    try:
        frame = bus.recv(_CAN_FRAME.size)
    except (BlockingIOError, socket.timeout):
        return
    finally:
        bus.setblocking(True)
    raise ProbeError(
        "pre-existing SDO response traffic was present before the first request "
        f"({len(frame)} bytes); freshness cannot be proven"
    )


def parse_position_response(
    frame: bytes,
    *,
    expected_node_id: int,
    sent_at: float,
    received_at: float,
) -> float:
    if not math.isfinite(sent_at) or not math.isfinite(received_at):
        raise ProbeError("non-finite response timestamps")
    age_s = received_at - sent_at
    if age_s < 0 or age_s > RESPONSE_TIMEOUT_S:
        raise ProbeError(
            f"node {expected_node_id} response is not fresh: age={age_s:.6f}s, "
            f"limit={RESPONSE_TIMEOUT_S:.3f}s"
        )
    if len(frame) != _CAN_FRAME.size:
        raise ProbeError(
            f"node {expected_node_id} returned malformed CAN frame length {len(frame)}"
        )
    can_id_with_flags, data_length, data = _CAN_FRAME.unpack(frame)
    if can_id_with_flags & _CAN_ID_FLAGS:
        raise ProbeError(
            f"node {expected_node_id} returned flagged/non-standard CAN ID "
            f"0x{can_id_with_flags:08X}"
        )
    expected_can_id = response_can_id(expected_node_id)
    if can_id_with_flags != expected_can_id:
        raise ProbeError(
            f"out-of-order SDO response 0x{can_id_with_flags:03X}; "
            f"expected 0x{expected_can_id:03X} for node {expected_node_id}"
        )
    if data_length != 8:
        raise ProbeError(
            f"node {expected_node_id} returned SDO payload length {data_length}, expected 8"
        )

    command = data[0]
    index = data[1] | (data[2] << 8)
    subindex = data[3]
    if command == SDO_ABORT_RESPONSE:
        abort_code = int.from_bytes(data[4:8], "little")
        raise ProbeError(
            f"node {expected_node_id} aborted SDO 0x{index:04X}:{subindex:02X} "
            f"with code 0x{abort_code:08X}"
        )
    if command != SDO_UPLOAD_RESPONSE_4_BYTES:
        raise ProbeError(
            f"node {expected_node_id} returned SDO command 0x{command:02X}, expected 0x43"
        )
    if index != POSITION_INDEX or subindex != POSITION_SUBINDEX:
        raise ProbeError(
            f"node {expected_node_id} returned wrong object 0x{index:04X}:{subindex:02X}; "
            f"expected 0x{POSITION_INDEX:04X}:{POSITION_SUBINDEX:02X}"
        )

    (raw_revolutions,) = struct.unpack("<f", data[4:8])
    if not math.isfinite(raw_revolutions):
        raise ProbeError(f"node {expected_node_id} returned non-finite motor position")
    return float(raw_revolutions)


def read_raw_motor_revolutions(
    bus: _CanSocket,
    *,
    clock: Callable[[], float] = time.monotonic,
    pre_send_guard: Callable[[], None] = run_preflight,
) -> tuple[float, ...]:
    values: list[float] = []
    request = sdo_upload_payload()
    for node_id in NODE_IDS:
        # Re-check immediately before every one of the only six permitted writes.
        pre_send_guard()
        sent_at = clock()
        frame = _CAN_FRAME.pack(request_can_id(node_id), 8, request)
        try:
            sent_bytes = bus.send(frame)
        except OSError as exc:
            raise ProbeError(f"node {node_id} SDO upload request failed: {exc}") from exc
        if sent_bytes != len(frame):
            raise ProbeError(
                f"node {node_id} SDO request was short: {sent_bytes}/{len(frame)} bytes"
            )
        bus.settimeout(RESPONSE_TIMEOUT_S)
        try:
            response = bus.recv(_CAN_FRAME.size)
        except (TimeoutError, socket.timeout) as exc:
            raise ProbeError(
                f"node {node_id} SDO response timed out after {RESPONSE_TIMEOUT_S:.3f}s"
            ) from exc
        except OSError as exc:
            raise ProbeError(f"node {node_id} SDO response failed: {exc}") from exc
        received_at = clock()
        values.append(
            parse_position_response(
                response,
                expected_node_id=node_id,
                sent_at=sent_at,
                received_at=received_at,
            )
        )
    return tuple(values)


def resolve_integer_turn(
    raw_revolutions: float,
    *,
    sign: int,
    zero_offset_rad: float,
    lower_rad: float,
    upper_rad: float,
) -> tuple[int, float]:
    """Choose the sole integer motor turn whose mapped joint lies in live limits."""

    values = (raw_revolutions, zero_offset_rad, lower_rad, upper_rad)
    if not all(math.isfinite(value) for value in values):
        raise ProbeError("non-finite value in integer-turn resolution")
    if sign not in {-1, 1}:
        raise ProbeError("joint sign must be exactly -1 or +1")
    if lower_rad > upper_rad:
        raise ProbeError("joint limit lower bound exceeds upper bound")

    tau = math.tau
    base = zero_offset_rad + sign * tau * raw_revolutions
    if sign == 1:
        first = math.ceil((lower_rad - base) / tau)
        last = math.floor((upper_rad - base) / tau)
    else:
        first = math.ceil((base - upper_rad) / tau)
        last = math.floor((base - lower_rad) / tau)

    candidates: list[tuple[int, float]] = []
    for turns in range(first, last + 1):
        position = base + sign * tau * turns
        if lower_rad <= position <= upper_rad and math.isfinite(position):
            candidates.append((turns, position))
    if len(candidates) != 1:
        raise ProbeError(
            "integer-turn resolution is not unique: "
            f"raw={raw_revolutions:+.9f} rev, limits=[{lower_rad:+.6f}, "
            f"{upper_rad:+.6f}] rad, candidates={len(candidates)}"
        )
    return candidates[0]


def resolve_joint_positions(raw_revolutions: Sequence[float]) -> tuple[JointPosition, ...]:
    if len(raw_revolutions) != len(NODE_IDS):
        raise ProbeError(
            f"expected {len(NODE_IDS)} motor positions, got {len(raw_revolutions)}"
        )
    positions: list[JointPosition] = []
    for index, (node_id, raw) in enumerate(zip(NODE_IDS, raw_revolutions, strict=True)):
        lower, upper = LIVE_JOINT_LIMITS_RAD[index]
        turns, position = resolve_integer_turn(
            float(raw),
            sign=JOINT_SIGN[index],
            zero_offset_rad=ZERO_OFFSET_RAD[index],
            lower_rad=lower,
            upper_rad=upper,
        )
        positions.append(
            JointPosition(
                node_id=node_id,
                raw_motor_revolutions=float(raw),
                integer_turns=turns,
                joint_position_rad=position,
            )
        )
    return tuple(positions)


def read_positions() -> tuple[JointPosition, ...]:
    run_preflight()
    bus = open_can1_socket()
    try:
        require_empty_sdo_receive_queue(bus)
        raw = read_raw_motor_revolutions(bus)
    finally:
        bus.close()
    return resolve_joint_positions(raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_requests:
        print("Only --read can open CAN or send these immutable SDO upload requests:")
        for line in printable_requests():
            print(line)
        return 0

    try:
        positions = read_positions()
    except ProbeError as exc:
        print(f"READ REFUSED: {exc}", file=sys.stderr)
        return 2

    print("Follower position read succeeded (q = sign*2pi*(raw_rev+turns)+offset):")
    for index, position in enumerate(positions, start=1):
        print(
            f"J{index}: node={position.node_id} "
            f"raw_rev={position.raw_motor_revolutions:+.9f} "
            f"turns={position.integer_turns:+d} "
            f"q_rad={position.joint_position_rad:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
