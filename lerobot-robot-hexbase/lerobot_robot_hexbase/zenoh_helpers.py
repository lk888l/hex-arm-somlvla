"""Shared, read-only Zenoh helpers for Hex LeRobot devices."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

import zenoh

# protoc emits flat imports such as "import common_pb2". Keep the generated
# directory on sys.path for both editable and wheel installs.
_PROTO_DIR = os.path.join(os.path.dirname(__file__), "proto")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import arm_pb2  # noqa: E402


def normalized_prefix(prefix: str, *, label: str) -> str:
    value = prefix.strip().rstrip("/")
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not value.startswith("hexmeow/"):
        raise ValueError(f"{label} must start with 'hexmeow/': {value!r}")
    return value


def joint_feature_names(arm_joint_names: Iterable[str], ee_joint_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(arm_joint_names) + tuple(ee_joint_names)
    if not names:
        raise ValueError("at least one arm or EE joint is required")
    if any(not name or "/" in name for name in names):
        raise ValueError(f"joint names must be non-empty and contain no '/': {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"arm and EE joint names must be unique: {names!r}")
    return tuple(f"{name}.pos" for name in names)


def open_zenoh_session(connect: str):
    config = zenoh.Config()
    config.insert_json5("mode", '"peer"')
    endpoint = connect.strip()
    if endpoint:
        config.insert_json5("connect/endpoints", f'["{endpoint}"]')
    return zenoh.open(config)


def sample_payload_bytes(sample) -> bytes:
    return bytes(sample.payload.to_bytes())


def close_zenoh_resource(resource) -> None:
    if resource is None:
        return
    for method_name in ("undeclare", "close"):
        method = getattr(resource, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception:
            pass
        return


@dataclass(frozen=True)
class JointVectorSnapshot:
    values: tuple[float, ...]
    received_at: float
    source_seq: int | None
    source_session_id: int | None = None
    source_timeout_behavior: int | None = None
    source_dq: tuple[float, ...] | None = None
    source_kp: tuple[float, ...] | None = None
    source_kd: tuple[float, ...] | None = None
    source_tau_ff: tuple[float, ...] | None = None


class LatestJointVector:
    """Thread-safe latest-value cache with dimensional and freshness checks."""

    def __init__(self, expected_size: int, label: str):
        if expected_size <= 0:
            raise ValueError(f"{label}: expected_size must be positive")
        self.expected_size = expected_size
        self.label = label
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._snapshot: JointVectorSnapshot | None = None
        self._last_error: str | None = None
        self._last_error_at: float | None = None

    def update(
        self,
        values: Iterable[float],
        *,
        source_seq: int | None = None,
        source_session_id: int | None = None,
        source_timeout_behavior: int | None = None,
        source_dq: tuple[float, ...] | None = None,
        source_kp: tuple[float, ...] | None = None,
        source_kd: tuple[float, ...] | None = None,
        source_tau_ff: tuple[float, ...] | None = None,
        received_at: float | None = None,
    ) -> None:
        vector = tuple(float(value) for value in values)
        if len(vector) != self.expected_size:
            self.remember_error(f"expected {self.expected_size} values, received {len(vector)}")
            return
        if not all(math.isfinite(value) for value in vector):
            self.remember_error("received NaN or infinite value")
            return
        snapshot = JointVectorSnapshot(
            values=vector,
            received_at=time.monotonic() if received_at is None else received_at,
            source_seq=source_seq,
            source_session_id=source_session_id,
            source_timeout_behavior=source_timeout_behavior,
            source_dq=source_dq,
            source_kp=source_kp,
            source_kd=source_kd,
            source_tau_ff=source_tau_ff,
        )
        with self._lock:
            self._snapshot = snapshot
            self._last_error = None
            self._last_error_at = None
            self._ready.set()

    def remember_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._last_error_at = time.monotonic()

    def wait_ready(self, timeout_s: float) -> None:
        if self._ready.wait(timeout_s):
            return
        with self._lock:
            detail = f": {self._last_error}" if self._last_error else ""
        raise TimeoutError(f"timed out waiting for {self.label}{detail}")

    def get_fresh(self, max_age_s: float) -> JointVectorSnapshot:
        with self._lock:
            snapshot = self._snapshot
            last_error = self._last_error
            last_error_at = self._last_error_at
        if last_error is not None and (
            snapshot is None
            or (
                last_error_at is not None and last_error_at >= snapshot.received_at
            )
        ):
            raise ValueError(f"invalid {self.label}: {last_error}")
        if snapshot is None:
            detail = f": {last_error}" if last_error else ""
            raise RuntimeError(f"no valid sample received for {self.label}{detail}")
        age_s = time.monotonic() - snapshot.received_at
        if age_s > max_age_s:
            raise TimeoutError(
                f"stale {self.label}: last valid sample is {age_s * 1000:.1f} ms old "
                f"(limit {max_age_s * 1000:.1f} ms)"
            )
        return snapshot


def decode_joint_state(sample, destination: LatestJointVector) -> None:
    message = arm_pb2.JointState()
    try:
        message.ParseFromString(sample_payload_bytes(sample))
    except Exception as exc:
        destination.remember_error(f"invalid JointState protobuf: {exc}")
        return
    sequence = message.header.seq if message.HasField("header") else None
    destination.update(message.q, source_seq=sequence)


def decode_joint_command(sample, destination: LatestJointVector) -> None:
    message = arm_pb2.JointTrajectory()
    try:
        message.ParseFromString(sample_payload_bytes(sample))
    except Exception as exc:
        destination.remember_error(f"invalid JointTrajectory protobuf: {exc}")
        return
    if not message.points:
        destination.remember_error("JointTrajectory contains no points")
        return
    if len(message.points) != 1:
        destination.remember_error(
            f"JointTrajectory contains {len(message.points)} points; recording requires exactly one"
        )
        return
    if len(message.t_from_start_ns) != 1 or message.t_from_start_ns[0] < 0:
        destination.remember_error(
            "JointTrajectory recording requires one non-negative t_from_start_ns"
        )
        return
    point = message.points[0]
    setpoint_vectors: dict[str, tuple[float, ...] | None] = {}
    for field_name in ("dq", "kp", "kd", "tau_ff"):
        values = tuple(float(value) for value in getattr(point, field_name))
        if values and len(values) != destination.expected_size:
            destination.remember_error(
                f"JointSetpoint {field_name} has {len(values)} values, "
                f"expected {destination.expected_size} or zero"
            )
            return
        if not all(math.isfinite(value) for value in values):
            destination.remember_error(
                f"JointSetpoint {field_name} contains NaN or infinite value"
            )
            return
        if field_name in ("kp", "kd") and any(value < 0 for value in values):
            destination.remember_error(
                f"JointSetpoint {field_name} contains a negative gain"
            )
            return
        setpoint_vectors[field_name] = values or None

    sequence = message.header.seq if message.HasField("header") else None
    destination.update(
        point.q,
        source_seq=sequence,
        source_session_id=int(message.session_id),
        source_timeout_behavior=int(message.on_timeout),
        source_dq=setpoint_vectors["dq"],
        source_kp=setpoint_vectors["kp"],
        source_kd=setpoint_vectors["kd"],
        source_tau_ff=setpoint_vectors["tau_ff"],
    )
