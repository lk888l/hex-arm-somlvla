"""Fail-closed read-only tap of requested follower commands for recording."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from .hex_zenoh_follower import DeviceLimits
from .zenoh_helpers import (
    LatestJointVector,
    arm_pb2,
    close_zenoh_resource,
    decode_joint_command,
    decode_joint_state,
    joint_feature_names,
    normalized_prefix,
    open_zenoh_session,
    sample_payload_bytes,
)

# Importing zenoh_helpers above adds the generated protobuf directory to
# sys.path; protoc emits these modules as flat imports.
import common_pb2  # noqa: E402
import ee_pb2  # noqa: E402
import robot_pb2  # noqa: E402


class UnsafeRecordedActionError(ValueError):
    """Requested target must not be committed as a training action."""


@dataclass(frozen=True)
class _RobotStatusSnapshot:
    mode: int
    session_holder: int
    received_at: float


class _LatestRobotStatus:
    def __init__(self, label: str):
        self.label = label
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._snapshot: _RobotStatusSnapshot | None = None
        self._last_error: str | None = None
        self._last_error_at: float | None = None

    def decode(self, sample) -> None:
        message = robot_pb2.RobotStatus()
        try:
            message.ParseFromString(sample_payload_bytes(sample))
        except Exception as exc:
            with self._lock:
                self._last_error = f"invalid RobotStatus protobuf: {exc}"
                self._last_error_at = time.monotonic()
            return
        snapshot = _RobotStatusSnapshot(
            mode=int(message.mode),
            session_holder=int(message.session_holder),
            received_at=time.monotonic(),
        )
        with self._lock:
            self._snapshot = snapshot
            self._last_error = None
            self._last_error_at = None
            self._ready.set()

    def wait_ready(self, timeout_s: float) -> None:
        if self._ready.wait(timeout_s):
            return
        with self._lock:
            detail = f": {self._last_error}" if self._last_error else ""
        raise TimeoutError(f"timed out waiting for {self.label}{detail}")

    def get_fresh(self, max_age_s: float) -> _RobotStatusSnapshot:
        with self._lock:
            snapshot = self._snapshot
            last_error = self._last_error
            last_error_at = self._last_error_at
        if last_error is not None and (
            snapshot is None
            or (
                last_error_at is not None
                and last_error_at >= snapshot.received_at
            )
        ):
            raise ValueError(f"invalid {self.label}: {last_error}")
        if snapshot is None:
            raise RuntimeError(f"no valid sample received for {self.label}")
        age_s = time.monotonic() - snapshot.received_at
        if age_s > max_age_s:
            raise TimeoutError(
                f"stale {self.label}: last valid sample is {age_s * 1000:.1f} ms old "
                f"(limit {max_age_s * 1000:.1f} ms)"
            )
        return snapshot



@TeleoperatorConfig.register_subclass("hex_zenoh_command_tap")
@dataclass
class HexZenohCommandTapConfig(TeleoperatorConfig):
    connect: str = "tcp/127.0.0.1:7447"
    arm_prefix: str = ""
    ee_prefix: str = ""
    arm_joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{index}" for index in range(1, 7)]
    )
    ee_joint_names: list[str] = field(default_factory=lambda: ["jaw_joint1"])
    connect_timeout_s: float = 5.0
    rpc_timeout_s: float = 1.0
    max_command_age_s: float = 0.100
    max_state_age_s: float = 0.100
    max_status_age_s: float = 0.300
    max_pair_skew_s: float = 0.020
    record_hz: float = 30.0
    max_step_reference_age_s: float = 1.0
    arm_step_limit_rad: float = 0.0435
    ee_step_limit_rad: float = 0.0165
    arm_position_margin_rad: float = 0.020
    ee_position_margin_rad: float = 0.005
    zenoh_settle_time_s: float = 0.2

    def __post_init__(self) -> None:
        self.arm_prefix = normalized_prefix(self.arm_prefix, label="arm_prefix")
        self.ee_prefix = normalized_prefix(self.ee_prefix, label="ee_prefix")
        joint_feature_names(self.arm_joint_names, self.ee_joint_names)
        if self.arm_prefix == self.ee_prefix:
            raise ValueError("arm_prefix and ee_prefix must identify different devices")
        positive_values = {
            "connect_timeout_s": self.connect_timeout_s,
            "rpc_timeout_s": self.rpc_timeout_s,
            "max_command_age_s": self.max_command_age_s,
            "max_state_age_s": self.max_state_age_s,
            "max_status_age_s": self.max_status_age_s,
            "max_pair_skew_s": self.max_pair_skew_s,
            "record_hz": self.record_hz,
            "max_step_reference_age_s": self.max_step_reference_age_s,
            "arm_step_limit_rad": self.arm_step_limit_rad,
            "ee_step_limit_rad": self.ee_step_limit_rad,
        }
        for label, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if not math.isclose(self.record_hz, 30.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("record_hz must be 30 for the Firefly recording contract")
        if not math.isclose(
            self.max_step_reference_age_s, 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "max_step_reference_age_s must remain 1.0 for the Firefly recording contract"
            )
        maximum_values = {
            "max_command_age_s": (self.max_command_age_s, 0.100),
            "max_state_age_s": (self.max_state_age_s, 0.100),
            "max_status_age_s": (self.max_status_age_s, 0.300),
            "max_pair_skew_s": (self.max_pair_skew_s, 0.020),
            "arm_step_limit_rad": (self.arm_step_limit_rad, 0.0435),
            "ee_step_limit_rad": (self.ee_step_limit_rad, 0.0165),
        }
        for label, (value, maximum) in maximum_values.items():
            if value > maximum:
                raise ValueError(f"{label} must not exceed {maximum}")
        if self.max_pair_skew_s > min(self.max_command_age_s, self.max_state_age_s):
            raise ValueError("max_pair_skew_s must not exceed either freshness limit")
        nonnegative_values = {
            "arm_position_margin_rad": self.arm_position_margin_rad,
            "ee_position_margin_rad": self.ee_position_margin_rad,
            "zenoh_settle_time_s": self.zenoh_settle_time_s,
        }
        for label, value in nonnegative_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        minimum_margins = {
            "arm_position_margin_rad": (self.arm_position_margin_rad, 0.020),
            "ee_position_margin_rad": (self.ee_position_margin_rad, 0.005),
        }
        for label, (value, minimum) in minimum_margins.items():
            if value < minimum:
                raise ValueError(f"{label} must be at least {minimum}")


class HexZenohCommandTap(Teleoperator):
    """Expose status-consistent, fail-closed requested targets for recording."""

    config_class = HexZenohCommandTapConfig
    name = "hex_zenoh_command_tap"

    def __init__(self, config: HexZenohCommandTapConfig):
        super().__init__(config)
        self.config = config
        self._feature_names = joint_feature_names(config.arm_joint_names, config.ee_joint_names)
        self._z = None
        self._resources: list[object] = []
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._arm_command = LatestJointVector(
            len(self.config.arm_joint_names), "follower arm command"
        )
        self._ee_command = LatestJointVector(
            len(self.config.ee_joint_names), "follower EE command"
        )
        self._arm_state = LatestJointVector(
            len(self.config.arm_joint_names), "follower arm joint_state"
        )
        self._ee_state = LatestJointVector(
            len(self.config.ee_joint_names), "follower EE joint_state"
        )
        self._arm_status = _LatestRobotStatus("follower arm RobotStatus")
        self._ee_status = _LatestRobotStatus("follower EE RobotStatus")
        self._arm_limits: DeviceLimits | None = None
        self._ee_limits: DeviceLimits | None = None
        self._last_recorded: tuple[float, ...] | None = None
        self._last_accepted_at: float | None = None
        self._session_pair: tuple[int, int] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._feature_names, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    def controller_contract(self) -> dict[str, object]:
        """Return the exact live limits used by this recording session."""
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected")
        assert self._arm_limits is not None
        assert self._ee_limits is not None

        position_margins = [self.config.arm_position_margin_rad] * len(
            self._arm_limits.joint_names
        ) + [self.config.ee_position_margin_rad] * len(self._ee_limits.joint_names)
        effective_step_limits = [
            min(self.config.arm_step_limit_rad, velocity / self.config.record_hz)
            for velocity in self._arm_limits.vel_max
        ] + [
            min(self.config.ee_step_limit_rad, velocity / self.config.record_hz)
            for velocity in self._ee_limits.vel_max
        ]
        return {
            "schema": "firefly_recording_contract_v1",
            "source": "live ArmDescription/EeDescription via HexZenohCommandTap",
            "captured_at_unix_ns": time.time_ns(),
            "arm_prefix": self.config.arm_prefix,
            "ee_prefix": self.config.ee_prefix,
            "record_hz": self.config.record_hz,
            "joint_names": list(self._feature_names),
            "pos_min": list(self._arm_limits.pos_min + self._ee_limits.pos_min),
            "pos_max": list(self._arm_limits.pos_max + self._ee_limits.pos_max),
            "position_margins": position_margins,
            "effective_step_limits": effective_step_limits,
        }

    @property
    def is_connected(self) -> bool:
        return (
            self._z is not None
            and len(self._resources) == 6
            and self._arm_limits is not None
            and self._ee_limits is not None
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self._z is not None:
            raise RuntimeError(f"{self} is already connected")
        self._resources = []
        self._reset_runtime_state()
        try:
            self._z = open_zenoh_session(self.config.connect)
            if self.config.zenoh_settle_time_s:
                time.sleep(self.config.zenoh_settle_time_s)
            self._load_descriptions()
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.arm_prefix}/arm/joint_state",
                    lambda sample: decode_joint_state(sample, self._arm_state),
                )
            )
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.ee_prefix}/ee/joint_state",
                    lambda sample: decode_joint_state(sample, self._ee_state),
                )
            )
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.arm_prefix}/arm/command",
                    lambda sample: decode_joint_command(sample, self._arm_command),
                )
            )
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.ee_prefix}/ee/command",
                    lambda sample: decode_joint_command(sample, self._ee_command),
                )
            )
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.arm_prefix}/status",
                    self._arm_status.decode,
                )
            )
            self._resources.append(
                self._z.declare_subscriber(
                    f"{self.config.ee_prefix}/status",
                    self._ee_status.decode,
                )
            )
            self._arm_state.wait_ready(self.config.connect_timeout_s)
            self._ee_state.wait_ready(self.config.connect_timeout_s)
            self._arm_command.wait_ready(self.config.connect_timeout_s)
            self._ee_command.wait_ready(self.config.connect_timeout_s)
            self._arm_status.wait_ready(self.config.connect_timeout_s)
            self._ee_status.wait_ready(self.config.connect_timeout_s)
        except Exception:
            self.disconnect()
            raise

    def _query_one(self, key: str, response_type):
        assert self._z is not None
        successful = []
        errors: list[str] = []
        try:
            replies = self._z.get(key, payload=b"", timeout=self.config.rpc_timeout_s)
            for reply in replies:
                sample = getattr(reply, "ok", None)
                if sample is None:
                    error = getattr(reply, "err", None)
                    if error is not None:
                        errors.append(str(error))
                    continue
                message = response_type()
                message.ParseFromString(sample_payload_bytes(sample))
                successful.append(message)
        except Exception as exc:
            raise TimeoutError(f"RPC/query failed for {key}: {exc}") from exc
        if len(successful) != 1:
            detail = f"; errors={errors!r}" if errors else ""
            raise TimeoutError(
                f"RPC/query expected exactly one successful reply for {key}, "
                f"received {len(successful)}{detail}"
            )
        return successful[0]

    def _load_descriptions(self) -> None:
        arm_description = self._query_one(
            f"{self.config.arm_prefix}/arm/description", arm_pb2.ArmDescription
        )
        ee_description = self._query_one(
            f"{self.config.ee_prefix}/ee/description", ee_pb2.EeDescription
        )
        self._arm_limits = DeviceLimits.from_message(
            "arm", arm_description, self.config.arm_joint_names
        )
        self._ee_limits = DeviceLimits.from_message(
            "EE", ee_description, self.config.ee_joint_names
        )
        for limits, margin in (
            (self._arm_limits, self.config.arm_position_margin_rad),
            (self._ee_limits, self.config.ee_position_margin_rad),
        ):
            for name, lower, upper in zip(
                limits.joint_names, limits.pos_min, limits.pos_max, strict=True
            ):
                if lower + margin >= upper - margin:
                    raise ValueError(
                        f"{limits.label} {name} range [{lower}, {upper}] "
                        f"is too narrow for inward recording margin {margin}"
                    )
        logging.info(
            "Recording contract loaded from controller: arm ranges=%s..%s, "
            "EE ranges=%s..%s, inward margins arm=%.6f EE=%.6f rad",
            self._arm_limits.pos_min,
            self._arm_limits.pos_max,
            self._ee_limits.pos_min,
            self._ee_limits.pos_max,
            self.config.arm_position_margin_rad,
            self.config.ee_position_margin_rad,
        )

    @staticmethod
    def _unsafe(message: str) -> UnsafeRecordedActionError:
        return UnsafeRecordedActionError(
            "unsafe requested follower action; recording stopped and the current "
            f"episode must be discarded: {message}"
        )

    def _fresh(self, cache: LatestJointVector, max_age_s: float):
        try:
            return cache.get_fresh(max_age_s)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            raise self._unsafe(str(exc)) from exc

    def _validate_pair_skew(self, label: str, first, second) -> None:
        skew_s = abs(first.received_at - second.received_at)
        if skew_s > self.config.max_pair_skew_s + 1e-9:
            raise self._unsafe(
                f"{label} pair receive skew is {skew_s * 1000:.3f} ms "
                f"(limit {self.config.max_pair_skew_s * 1000:.3f} ms); "
                f"arm_seq={first.source_seq}, ee_seq={second.source_seq}"
            )

    def _validate_controller_acceptance(
        self,
        device: str,
        command,
        status: _RobotStatusSnapshot,
        limits: DeviceLimits,
    ) -> None:
        session_id = command.source_session_id
        if session_id is None or session_id <= 0:
            raise self._unsafe(
                f"{device} command has invalid session_id={session_id}"
            )
        if status.mode != common_pb2.ROBOT_MODE_RUNNING or status.session_holder != session_id:
            try:
                mode_name = common_pb2.RobotMode.Name(status.mode)
            except ValueError:
                mode_name = f"UNKNOWN({status.mode})"
            raise self._unsafe(
                f"{device} controller cannot accept command session_id={session_id}: "
                f"mode={mode_name}, holder={status.session_holder}"
            )
        timeout_behavior = command.source_timeout_behavior
        if timeout_behavior in (None, arm_pb2.TIMEOUT_BEHAVIOR_UNSPECIFIED):
            raise self._unsafe(f"{device} command has no explicit timeout behavior")
        try:
            timeout_name = arm_pb2.TimeoutBehavior.Name(timeout_behavior)
        except ValueError as exc:
            raise self._unsafe(
                f"{device} command has unknown timeout behavior {timeout_behavior}"
            ) from exc
        if timeout_behavior not in limits.supported_timeouts:
            raise self._unsafe(
                f"{device} command timeout {timeout_name} is not advertised by controller"
            )

    def _validate_setpoint_vectors(self, device: str, command, limits: DeviceLimits) -> None:
        if command.source_dq is not None:
            for name, value, maximum in zip(
                limits.joint_names, command.source_dq, limits.vel_max, strict=True
            ):
                if abs(value) > maximum + 1e-9:
                    raise self._unsafe(
                        f"{device} {name} requested dq={value:.9f} exceeds "
                        f"controller vel_max={maximum:.9f}"
                    )
        if command.source_tau_ff is not None:
            for name, value, maximum in zip(
                limits.joint_names, command.source_tau_ff, limits.tau_max, strict=True
            ):
                if abs(value) > maximum + 1e-9:
                    raise self._unsafe(
                        f"{device} {name} requested tau_ff={value:.9f} exceeds "
                        f"controller tau_max={maximum:.9f}"
                    )

    def _validate_positions(
        self,
        device: str,
        values: tuple[float, ...],
        limits: DeviceLimits,
        margin: float,
        *,
        source_seq: int | None,
        measured: bool,
    ) -> None:
        for name, value, raw_lower, raw_upper in zip(
            limits.joint_names, values, limits.pos_min, limits.pos_max, strict=True
        ):
            lower = raw_lower if measured else raw_lower + margin
            upper = raw_upper if measured else raw_upper - margin
            if value < lower or value > upper:
                kind = "measured state" if measured else "requested target"
                raise self._unsafe(
                    f"{device} {name} {kind}={value:.9f} is outside "
                    f"[{lower:.9f}, {upper:.9f}]; controller range="
                    f"[{raw_lower:.9f}, {raw_upper:.9f}], inward_margin={margin:.6f}, "
                    f"seq={source_seq}"
                )

    def _validate_step(
        self,
        device: str,
        target: tuple[float, ...],
        reference: tuple[float, ...],
        limits: DeviceLimits,
        configured_limit: float,
        reference_label: str,
        source_seq: int | None,
    ) -> None:
        for name, value, previous, velocity in zip(
            limits.joint_names, target, reference, limits.vel_max, strict=True
        ):
            effective_limit = min(configured_limit, velocity / self.config.record_hz)
            delta = abs(value - previous)
            if delta > effective_limit + 1e-9:
                raise self._unsafe(
                    f"{device} {name} step from {reference_label} is {delta:.9f} rad "
                    f"(limit {effective_limit:.9f}); reference={previous:.9f}, "
                    f"target={value:.9f}, vel_max={velocity:.9f}, seq={source_seq}"
                )

    def get_action(self) -> dict:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected")
        assert self._arm_limits is not None
        assert self._ee_limits is not None

        arm_command = self._fresh(self._arm_command, self.config.max_command_age_s)
        ee_command = self._fresh(self._ee_command, self.config.max_command_age_s)
        arm_state = self._fresh(self._arm_state, self.config.max_state_age_s)
        ee_state = self._fresh(self._ee_state, self.config.max_state_age_s)
        arm_status = self._fresh(self._arm_status, self.config.max_status_age_s)
        ee_status = self._fresh(self._ee_status, self.config.max_status_age_s)
        self._validate_pair_skew("command", arm_command, ee_command)
        self._validate_pair_skew("joint_state", arm_state, ee_state)
        self._validate_pair_skew("arm command/joint_state", arm_command, arm_state)
        self._validate_pair_skew("EE command/joint_state", ee_command, ee_state)
        self._validate_controller_acceptance(
            "arm", arm_command, arm_status, self._arm_limits
        )
        self._validate_controller_acceptance(
            "EE", ee_command, ee_status, self._ee_limits
        )
        assert arm_command.source_session_id is not None
        assert ee_command.source_session_id is not None
        session_pair = (
            arm_command.source_session_id,
            ee_command.source_session_id,
        )
        if self._session_pair is not None and session_pair != self._session_pair:
            raise self._unsafe(
                "controller session changed during recording; "
                f"expected arm={self._session_pair[0]}, EE={self._session_pair[1]}, "
                f"observed arm={session_pair[0]}, EE={session_pair[1]}"
            )
        self._validate_setpoint_vectors("arm", arm_command, self._arm_limits)
        self._validate_setpoint_vectors("EE", ee_command, self._ee_limits)

        arm_target = arm_command.values
        ee_target = ee_command.values
        arm_measured = arm_state.values
        ee_measured = ee_state.values
        self._validate_positions(
            "arm", arm_measured, self._arm_limits, 0.0,
            source_seq=arm_state.source_seq, measured=True
        )
        self._validate_positions(
            "EE", ee_measured, self._ee_limits, 0.0,
            source_seq=ee_state.source_seq, measured=True
        )
        self._validate_positions(
            "arm", arm_target, self._arm_limits, self.config.arm_position_margin_rad,
            source_seq=arm_command.source_seq, measured=False
        )
        self._validate_positions(
            "EE", ee_target, self._ee_limits, self.config.ee_position_margin_rad,
            source_seq=ee_command.source_seq, measured=False
        )

        accepted_at = time.monotonic()
        reference_age_s = (
            accepted_at - self._last_accepted_at
            if self._last_accepted_at is not None
            else None
        )
        reference_is_contiguous = (
            self._last_recorded is not None
            and reference_age_s is not None
            and reference_age_s <= self.config.max_step_reference_age_s
        )
        if reference_is_contiguous:
            arm_size = len(self.config.arm_joint_names)
            arm_reference = self._last_recorded[:arm_size]
            ee_reference = self._last_recorded[arm_size:]
            self._validate_step(
                "arm", arm_target, arm_reference, self._arm_limits,
                self.config.arm_step_limit_rad, "previous accepted command",
                arm_command.source_seq
            )
            self._validate_step(
                "EE", ee_target, ee_reference, self._ee_limits,
                self.config.ee_step_limit_rad, "previous accepted command",
                ee_command.source_seq
            )
        elif self._last_recorded is not None:
            logging.warning(
                "Re-seeding requested-target step reference after %.3f s sampling gap; "
                "same controller sessions remain locked and all freshness/status/position "
                "checks passed",
                reference_age_s,
            )

        values = arm_target + ee_target
        self._session_pair = session_pair
        self._last_recorded = values
        self._last_accepted_at = accepted_at
        return {
            feature_name: value
            for feature_name, value in zip(self._feature_names, values, strict=True)
        }

    def send_feedback(self, feedback: dict) -> None:
        return None

    def disconnect(self) -> None:
        for resource in reversed(self._resources):
            close_zenoh_resource(resource)
        self._resources = []
        if self._z is not None:
            try:
                self._z.close()
            except Exception:
                pass
            self._z = None
        self._arm_limits = None
        self._ee_limits = None
        self._last_recorded = None
        self._last_accepted_at = None
        self._session_pair = None
