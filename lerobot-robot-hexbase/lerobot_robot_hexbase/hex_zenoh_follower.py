"""Safety-gated Firefly follower execution over the Hex Zenoh robot API."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .arm_gravity import SerialRevoluteGravityModel
from .payload_gravity import (
    LumpedInertial,
    lumped_inertial_from_urdf_string,
    serial_chain_endpoints,
    with_tip_payload,
)
from .zenoh_helpers import (
    LatestJointVector,
    arm_pb2,
    close_zenoh_resource,
    decode_joint_state,
    joint_feature_names,
    normalized_prefix,
    open_zenoh_session,
    sample_payload_bytes,
)

# Generated modules use flat imports. Importing zenoh_helpers above installs the
# generated directory on sys.path before these imports are evaluated.
import common_pb2  # noqa: E402
import ee_pb2  # noqa: E402
import robot_pb2  # noqa: E402


_CLIENT_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_VISUAL_TELEMETRY_MAX_DEPTH = 4
_VISUAL_TELEMETRY_MAX_ITEMS = 128
_VISUAL_TELEMETRY_MAX_STRING_CHARS = 512
_VISUAL_TELEMETRY_MAX_KEY_CHARS = 128


class SafetyStopError(RuntimeError):
    """Raised after command publication has been stopped and sessions released."""


@dataclass(frozen=True)
class CameraFrameMetadata:
    """Freshness evidence for one image returned by ``get_observation``.

    This is deliberately exposed through :meth:`get_camera_frame_metadata`
    rather than added to ``observation_features`` or the observation mapping.
    The policy therefore keeps the exact training-time schema while deployment
    safety code can reject stale or duplicate camera frames.

    The bundled LeRobot RealSense driver records a host ``perf_counter`` value
    after a frameset has been copied and post-processed. It does not expose the
    RealSense SDK frame number or device timestamp through the Camera API. For
    that driver, ``sequence`` is a process-local counter derived from changes in
    this host timestamp. Cameras without such a timestamp are still readable,
    but report ``sequence=None``/``is_new_frame=False`` so a visual latch can
    fail closed instead of treating each observation call as a new frame.
    """

    camera_name: str
    sequence: int | None
    captured_at_s: float | None
    observed_at_s: float
    age_s: float | None
    timestamp_source: str
    is_new_frame: bool


def _validated_visual_telemetry(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return a bounded plain-JSON copy of compact visual cycle evidence."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("visual_telemetry must be a mapping or None")

    item_count = [0]

    def normalize(item: object, *, path: str, depth: int) -> object:
        item_count[0] += 1
        if item_count[0] > _VISUAL_TELEMETRY_MAX_ITEMS:
            raise ValueError(
                "visual_telemetry exceeds the compact 128-item safety limit"
            )
        if depth > _VISUAL_TELEMETRY_MAX_DEPTH:
            raise ValueError(
                "visual_telemetry exceeds the maximum nesting depth of 4"
            )
        if item is None or type(item) is bool or type(item) is int:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{path} contains NaN or infinite float")
            return item
        if type(item) is str:
            if len(item) > _VISUAL_TELEMETRY_MAX_STRING_CHARS:
                raise ValueError(f"{path} exceeds the 512-character string limit")
            return item
        if type(item) is list:
            return [
                normalize(child, path=f"{path}[{index}]", depth=depth + 1)
                for index, child in enumerate(item)
            ]
        if isinstance(item, Mapping):
            normalized: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"{path} contains a non-string mapping key")
                if len(key) > _VISUAL_TELEMETRY_MAX_KEY_CHARS:
                    raise ValueError(f"{path} contains a key longer than 128 characters")
                normalized[key] = normalize(
                    child,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )
            return normalized
        raise ValueError(
            f"{path} contains unsupported non-JSON type {type(item).__name__}"
        )

    result = normalize(value, path="visual_telemetry", depth=0)
    assert isinstance(result, dict)
    return result


@dataclass(frozen=True)
class DeviceLimits:
    """Validated controller-provided command limits in wire joint order."""

    label: str
    joint_names: tuple[str, ...]
    pos_min: tuple[float, ...]
    pos_max: tuple[float, ...]
    vel_max: tuple[float, ...]
    tau_max: tuple[float, ...]
    default_kp: tuple[float, ...]
    default_kd: tuple[float, ...]
    supported_timeouts: frozenset[int]

    @classmethod
    def from_message(cls, label: str, message, expected_names: list[str]) -> DeviceLimits:
        dof = int(message.dof)
        fields = {
            "joint_names": tuple(message.joint_names),
            "pos_min": tuple(float(value) for value in message.pos_min),
            "pos_max": tuple(float(value) for value in message.pos_max),
            "vel_max": tuple(float(value) for value in message.vel_max),
            "tau_max": tuple(float(value) for value in message.tau_max),
            "default_kp": tuple(float(value) for value in message.default_kp),
            "default_kd": tuple(float(value) for value in message.default_kd),
        }
        if dof != len(expected_names):
            raise ValueError(f"{label} description DOF is {dof}, expected {len(expected_names)}")
        for field_name, values in fields.items():
            if len(values) != dof:
                raise ValueError(
                    f"{label} description {field_name} has {len(values)} entries, expected {dof}"
                )
        if fields["joint_names"] != tuple(expected_names):
            raise ValueError(
                f"{label} controller joint order {fields['joint_names']!r} does not match "
                f"configured order {tuple(expected_names)!r}"
            )
        numeric_values = (
            fields["pos_min"]
            + fields["pos_max"]
            + fields["vel_max"]
            + fields["tau_max"]
            + fields["default_kp"]
            + fields["default_kd"]
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"{label} description contains NaN or infinite limits/gains")
        for index, (lower, upper, velocity) in enumerate(
            zip(fields["pos_min"], fields["pos_max"], fields["vel_max"], strict=True)
        ):
            if lower >= upper:
                raise ValueError(f"{label} joint {index} has invalid position range [{lower}, {upper}]")
            if velocity <= 0:
                raise ValueError(f"{label} joint {index} has non-positive vel_max {velocity}")
        if any(value < 0 for value in fields["tau_max"]):
            raise ValueError(f"{label} description contains a negative tau_max")
        if any(value < 0 for value in fields["default_kp"] + fields["default_kd"]):
            raise ValueError(f"{label} description contains a negative default gain")
        return cls(
            label=label,
            joint_names=fields["joint_names"],
            pos_min=fields["pos_min"],
            pos_max=fields["pos_max"],
            vel_max=fields["vel_max"],
            tau_max=fields["tau_max"],
            default_kp=fields["default_kp"],
            default_kd=fields["default_kd"],
            supported_timeouts=frozenset(int(value) for value in message.supported_timeouts),
        )


@dataclass(frozen=True)
class RobotStatusSnapshot:
    mode: int
    session_holder: int
    received_at: float


class _LatestRobotStatus:
    def __init__(self, label: str):
        self.label = label
        self._condition = threading.Condition()
        self._snapshot: RobotStatusSnapshot | None = None
        self._last_error: str | None = None
        self._last_error_at: float | None = None

    def decode(self, sample) -> None:
        message = robot_pb2.RobotStatus()
        try:
            message.ParseFromString(sample_payload_bytes(sample))
        except Exception as exc:
            with self._condition:
                self._last_error = f"invalid RobotStatus protobuf: {exc}"
                self._last_error_at = time.monotonic()
                self._condition.notify_all()
            return
        with self._condition:
            self._snapshot = RobotStatusSnapshot(
                mode=int(message.mode),
                session_holder=int(message.session_holder),
                received_at=time.monotonic(),
            )
            self._last_error = None
            self._last_error_at = None
            self._condition.notify_all()

    def wait_ready(self, timeout_s: float) -> RobotStatusSnapshot:
        return self.wait_for(lambda _snapshot: True, timeout_s)

    def wait_for(self, predicate, timeout_s: float) -> RobotStatusSnapshot:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                snapshot = self._snapshot
                if self._last_error is not None and (
                    snapshot is None
                    or (
                        self._last_error_at is not None
                        and self._last_error_at >= snapshot.received_at
                    )
                ):
                    raise ValueError(f"invalid {self.label}: {self._last_error}")
                if snapshot is not None and predicate(snapshot):
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._last_error}" if self._last_error else ""
                    raise TimeoutError(f"timed out waiting for {self.label}{detail}")
                self._condition.wait(remaining)

    def get_fresh(self, max_age_s: float) -> RobotStatusSnapshot:
        with self._condition:
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


@RobotConfig.register_subclass("hex_zenoh_follower")
@dataclass
class HexZenohFollowerConfig(RobotConfig):
    """Zenoh endpoints and non-negotiable safety limits for Firefly execution."""

    connect: str = "tcp/127.0.0.1:7447"
    arm_prefix: str = ""
    ee_prefix: str = ""
    arm_joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{index}" for index in range(1, 7)]
    )
    ee_joint_names: list[str] = field(default_factory=lambda: ["jaw_joint1"])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Kept as str because the Draccus version bundled with LeRobot 0.6 does
    # not decode typing.Literal fields. __post_init__ enforces the two values.
    mode: str = "shadow"
    client_name: str = "lerobot_firefly_y6"
    connect_timeout_s: float = 5.0
    rpc_timeout_s: float = 1.0
    max_state_age_s: float = 0.100
    # LeRobot Camera.read_latest() otherwise defaults to allowing 500 ms.
    camera_max_age_s: float = 0.100
    max_status_age_s: float = 0.300
    max_send_gap_s: float = 0.250
    command_hz: float = 30.0
    # The wider envelope is reserved for the fullcycle-v4 deployment ladder.
    # robot_config_for_stage binds it to the attested dataset/checkpoint/task;
    # keeping the selector here makes the follower-side hard maximum explicit.
    motion_envelope: str = "initial"
    arm_step_limit_rad: float = 0.035
    # Optional per-joint limits in ``arm_joint_names`` order. The scalar
    # remains a backwards-compatible global cap/fallback.
    arm_step_limits_rad: list[float] = field(default_factory=list)
    ee_step_limit_rad: float = 0.012
    arm_tracking_error_rad: float = 0.150
    ee_tracking_error_rad: float = 0.050
    ee_closing_tracking_error_rad: float | None = None
    tracking_error_frames: int = 3
    # Scalars preserve historical profiles/fingerprints; a six-value list is
    # reserved for the exact fingerprint-bound J3/J4 experiment below.
    arm_kp: float | list[float] = 40.0
    arm_kd: float | list[float] = 2.0
    ee_kp: float = 150.0
    ee_kd_default_scale: float = 0.5
    arm_timeout_behavior: str = "ramp_stop"
    allow_arm_hold_timeout_without_ramp_stop: bool = False
    enable_arm_gravity_ff: bool = False
    expected_arm_urdf_sha256: str | None = None
    enable_ee_payload_gravity_ff: bool = False
    expected_ee_urdf_sha256: str | None = None
    expected_arm_urdf_root_link: str | None = None
    expected_arm_urdf_tip_link: str | None = None
    expected_ee_urdf_root_link: str | None = None
    expected_ee_urdf_tip_link: str | None = None
    expected_ee_payload_mass_kg: float | None = None
    expected_ee_payload_com_xyz_m: list[float] = field(default_factory=list)
    arm_gravity_ff_max_abs_nm: float = 6.0
    zenoh_settle_time_s: float = 0.2
    telemetry_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.arm_prefix = normalized_prefix(self.arm_prefix, label="arm_prefix")
        self.ee_prefix = normalized_prefix(self.ee_prefix, label="ee_prefix")
        joint_feature_names(self.arm_joint_names, self.ee_joint_names)
        if not self.arm_joint_names or not self.ee_joint_names:
            raise ValueError("hex_zenoh_follower requires both arm and EE joints")
        if self.arm_prefix == self.ee_prefix:
            raise ValueError("arm_prefix and ee_prefix must identify separate robot sessions")
        if self.mode not in ("shadow", "execute"):
            raise ValueError("mode must be 'shadow' or 'execute'")
        if self.motion_envelope not in ("initial", "fullcycle_v4"):
            raise ValueError("motion_envelope must be 'initial' or 'fullcycle_v4'")
        if self.arm_timeout_behavior not in ("ramp_stop", "hold"):
            raise ValueError("arm_timeout_behavior must be 'ramp_stop' or 'hold'")
        if self.arm_timeout_behavior == "hold" and not self.allow_arm_hold_timeout_without_ramp_stop:
            raise ValueError(
                "arm HOLD timeout requires explicit allow_arm_hold_timeout_without_ramp_stop"
            )
        if self.arm_timeout_behavior != "hold" and self.allow_arm_hold_timeout_without_ramp_stop:
            raise ValueError("arm HOLD timeout opt-in is invalid unless behavior is 'hold'")
        if not self.client_name.strip():
            raise ValueError("client_name must not be empty")
        if not isinstance(self.enable_arm_gravity_ff, bool):
            raise ValueError("enable_arm_gravity_ff must be a boolean")
        if self.enable_arm_gravity_ff:
            if not isinstance(self.expected_arm_urdf_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", self.expected_arm_urdf_sha256
            ) is None:
                raise ValueError(
                    "enabled arm gravity feed-forward requires expected_arm_urdf_sha256 "
                    "as a lowercase SHA-256 digest"
                )
        elif self.expected_arm_urdf_sha256 is not None:
            raise ValueError(
                "expected_arm_urdf_sha256 is invalid unless arm gravity feed-forward is enabled"
            )
        if not isinstance(self.enable_ee_payload_gravity_ff, bool):
            raise ValueError("enable_ee_payload_gravity_ff must be a boolean")
        payload_binding_values = (
            self.expected_ee_urdf_sha256,
            self.expected_arm_urdf_root_link,
            self.expected_arm_urdf_tip_link,
            self.expected_ee_urdf_root_link,
            self.expected_ee_urdf_tip_link,
            self.expected_ee_payload_mass_kg,
        )
        if self.enable_ee_payload_gravity_ff:
            if not self.enable_arm_gravity_ff:
                raise ValueError("EE payload gravity feed-forward requires arm gravity feed-forward")
            if not isinstance(self.expected_ee_urdf_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", self.expected_ee_urdf_sha256
            ) is None:
                raise ValueError(
                    "enabled EE payload gravity feed-forward requires "
                    "expected_ee_urdf_sha256 as a lowercase SHA-256 digest"
                )
            for name in (
                "expected_arm_urdf_root_link",
                "expected_arm_urdf_tip_link",
                "expected_ee_urdf_root_link",
                "expected_ee_urdf_tip_link",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty mount-binding link name")
            if self.expected_ee_payload_mass_kg is None or not math.isfinite(
                self.expected_ee_payload_mass_kg
            ) or self.expected_ee_payload_mass_kg <= 0:
                raise ValueError("expected_ee_payload_mass_kg must be positive and finite")
            if len(self.expected_ee_payload_com_xyz_m) != 3 or not all(
                math.isfinite(value) for value in self.expected_ee_payload_com_xyz_m
            ):
                raise ValueError(
                    "expected_ee_payload_com_xyz_m must contain exactly three finite values"
                )
        elif any(value is not None for value in payload_binding_values) or (
            self.expected_ee_payload_com_xyz_m
        ):
            raise ValueError(
                "EE payload URDF/mount/inertial bindings are invalid unless "
                "enable_ee_payload_gravity_ff is true"
            )
        positive = {
            "connect_timeout_s": self.connect_timeout_s,
            "rpc_timeout_s": self.rpc_timeout_s,
            "max_state_age_s": self.max_state_age_s,
            "camera_max_age_s": self.camera_max_age_s,
            "max_status_age_s": self.max_status_age_s,
            "max_send_gap_s": self.max_send_gap_s,
            "command_hz": self.command_hz,
            "arm_step_limit_rad": self.arm_step_limit_rad,
            "ee_step_limit_rad": self.ee_step_limit_rad,
            "arm_tracking_error_rad": self.arm_tracking_error_rad,
            "ee_tracking_error_rad": self.ee_tracking_error_rad,
            "ee_kp": self.ee_kp,
            "ee_kd_default_scale": self.ee_kd_default_scale,
            "arm_gravity_ff_max_abs_nm": self.arm_gravity_ff_max_abs_nm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.arm_step_limits_rad, list):
            raise ValueError("arm_step_limits_rad must be a list")
        if self.arm_step_limits_rad:
            if len(self.arm_step_limits_rad) != len(self.arm_joint_names):
                raise ValueError(
                    "arm_step_limits_rad must have one value per configured arm joint"
                )
            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 0
                for value in self.arm_step_limits_rad
            ):
                raise ValueError(
                    "arm_step_limits_rad must contain only positive finite values"
                )
            if any(value > self.arm_step_limit_rad for value in self.arm_step_limits_rad):
                raise ValueError(
                    "arm_step_limits_rad values must not exceed arm_step_limit_rad"
                )
            self.arm_step_limits_rad = [float(value) for value in self.arm_step_limits_rad]
        if self.motion_envelope == "fullcycle_v4":
            if self.arm_step_limits_rad:
                raise ValueError(
                    "fullcycle_v4 motion envelope requires one scalar arm step limit"
                )
            if not math.isclose(
                self.arm_step_limit_rad, 0.0435, rel_tol=0.0, abs_tol=1e-12
            ) or not math.isclose(
                self.ee_step_limit_rad, 0.0165, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "fullcycle_v4 motion envelope requires exact 0.0435/0.0165 "
                    "rad-per-30Hz-step limits"
                )
        if self.ee_closing_tracking_error_rad is not None:
            value = self.ee_closing_tracking_error_rad
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "ee_closing_tracking_error_rad must be positive and finite when configured"
                )
            if value < self.ee_tracking_error_rad:
                raise ValueError(
                    "ee_closing_tracking_error_rad must not be below ee_tracking_error_rad"
                )
            if value > 0.060:
                raise ValueError(
                    "ee_closing_tracking_error_rad must not exceed the hard safety maximum 0.06"
                )
        if self.max_state_age_s > 0.100:
            raise ValueError("max_state_age_s must not exceed the 100 ms safety limit")
        if self.camera_max_age_s > 0.100:
            raise ValueError("camera_max_age_s must not exceed the 100 ms safety limit")
        if self.max_send_gap_s > 0.250:
            raise ValueError("max_send_gap_s must not exceed the 250 ms watchdog limit")
        baseline_arm_gain_profile = (40.0, 2.0)
        elevated_payload_arm_gain_profile = (60.0, 2.5)
        kp_is_vector = isinstance(self.arm_kp, list)
        kd_is_vector = isinstance(self.arm_kd, list)
        if kp_is_vector != kd_is_vector:
            raise ValueError("arm_kp and arm_kd must both be scalars or both be lists")
        configured_arm_kp = self.resolved_arm_kp()
        configured_arm_kd = self.resolved_arm_kd()
        experimental_payload_arm_kp = (60.0, 60.0, 72.0, 72.0, 60.0, 60.0)
        experimental_payload_arm_kd = (2.5, 2.5, 2.75, 2.75, 2.5, 2.5)
        if kp_is_vector:
            if (
                not self.enable_ee_payload_gravity_ff
                or configured_arm_kp != experimental_payload_arm_kp
                or configured_arm_kd != experimental_payload_arm_kd
            ):
                raise ValueError(
                    "per-joint arm gains are allowed only for the exact full-EE-payload "
                    "J3/J4 experimental profile KP=(60,60,72,72,60,60), "
                    "KD=(2.5,2.5,2.75,2.75,2.5,2.5)"
                )
        else:
            configured_arm_gain_profile = (configured_arm_kp[0], configured_arm_kd[0])
            if (
                configured_arm_gain_profile[0] > baseline_arm_gain_profile[0]
                or configured_arm_gain_profile[1] > baseline_arm_gain_profile[1]
            ) and (
                not self.enable_ee_payload_gravity_ff
                or configured_arm_gain_profile != elevated_payload_arm_gain_profile
            ):
                raise ValueError(
                    "arm gains above the 40.0/2.0 baseline exceed the hard safety maximum; "
                    "only the exact 60.0/2.5 scalar full-EE-payload profile or the "
                    "exact J3/J4 experimental vector profile is allowed"
                )
        max_arm_step = 0.0435 if self.motion_envelope == "fullcycle_v4" else 0.035
        max_ee_step = 0.0165 if self.motion_envelope == "fullcycle_v4" else 0.012
        hard_maxima = {
            "arm_step_limit_rad": (self.arm_step_limit_rad, max_arm_step),
            "ee_step_limit_rad": (self.ee_step_limit_rad, max_ee_step),
            "arm_tracking_error_rad": (self.arm_tracking_error_rad, 0.150),
            "ee_tracking_error_rad": (self.ee_tracking_error_rad, 0.050),
            "ee_kp": (self.ee_kp, 150.0),
            "ee_kd_default_scale": (self.ee_kd_default_scale, 0.5),
        }
        for name, (value, maximum) in hard_maxima.items():
            if value > maximum:
                raise ValueError(f"{name} must not exceed the hard safety maximum {maximum}")
        gravity_cap_maximum = 8.0 if self.enable_ee_payload_gravity_ff else 6.0
        if self.arm_gravity_ff_max_abs_nm > gravity_cap_maximum:
            raise ValueError(
                "arm_gravity_ff_max_abs_nm must not exceed the hard safety maximum "
                f"{gravity_cap_maximum}"
            )
        exact_values = {"command_hz": (self.command_hz, 30.0)}
        for name, (value, expected) in exact_values.items():
            if not math.isclose(value, expected):
                raise ValueError(f"{name} is fixed at {expected} for the initial Firefly deployment")
        if self.tracking_error_frames != 3:
            raise ValueError("tracking_error_frames is fixed at three consecutive frames")
        if self.zenoh_settle_time_s < 0:
            raise ValueError("zenoh_settle_time_s must be non-negative")

    def resolved_arm_step_limits_rad(self) -> tuple[float, ...]:
        """Return configured arm slew limits in controller joint order."""
        if self.arm_step_limits_rad:
            return tuple(self.arm_step_limits_rad)
        return (float(self.arm_step_limit_rad),) * len(self.arm_joint_names)

    def _resolved_arm_gain(self, value: float | list[float], *, name: str) -> tuple[float, ...]:
        if isinstance(value, list):
            if len(value) != len(self.arm_joint_names):
                raise ValueError(f"{name} must have one value per configured arm joint")
            raw_values = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            raw_values = [value] * len(self.arm_joint_names)
        else:
            raise ValueError(f"{name} must be a positive finite scalar or list")
        if not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item > 0
            for item in raw_values
        ):
            raise ValueError(f"{name} must contain only positive finite values")
        return tuple(float(item) for item in raw_values)

    def resolved_arm_kp(self) -> tuple[float, ...]:
        """Return arm proportional gains in controller joint order."""
        return self._resolved_arm_gain(self.arm_kp, name="arm_kp")

    def resolved_arm_kd(self) -> tuple[float, ...]:
        """Return arm derivative gains in controller joint order."""
        return self._resolved_arm_gain(self.arm_kd, name="arm_kd")


class HexZenohFollower(Robot):
    """Reads follower sensors and, only in execute mode, owns both command sessions."""

    config_class = HexZenohFollowerConfig
    name = "hex_zenoh_follower"

    def __init__(self, config: HexZenohFollowerConfig):
        super().__init__(config)
        self.config = config
        self._feature_names = joint_feature_names(config.arm_joint_names, config.ee_joint_names)
        self._arm_state = LatestJointVector(len(config.arm_joint_names), "follower arm joint_state")
        self._ee_state = LatestJointVector(len(config.ee_joint_names), "follower EE joint_state")
        self._arm_status = _LatestRobotStatus("follower arm RobotStatus")
        self._ee_status = _LatestRobotStatus("follower EE RobotStatus")
        self.cameras = make_cameras_from_configs(config.cameras)
        self._z = None
        self._resources: list[object] = []
        self._liveliness_tokens: list[object] = []
        self._arm_session_id = 0
        self._ee_session_id = 0
        self._arm_limits: DeviceLimits | None = None
        self._ee_limits: DeviceLimits | None = None
        self._arm_gravity_model: SerialRevoluteGravityModel | None = None
        self._arm_gravity_urdf_sha256: str | None = None
        self._ee_payload_urdf_sha256: str | None = None
        self._ee_payload: LumpedInertial | None = None
        self._last_target: tuple[float, ...] | None = None
        self._last_send_at: float | None = None
        self._last_cycle_at: float | None = None
        self._arm_error_frames = 0
        self._ee_error_frames = 0
        self._sequence = 0
        self._clock_origin_ns = time.monotonic_ns()
        self._stopped_reason: str | None = None
        self._disconnecting = False
        self._lifecycle_lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._camera_metadata_lock = threading.Lock()
        self._camera_frame_metadata: dict[str, CameraFrameMetadata] = {}
        self._camera_capture_timestamps: dict[str, float] = {}
        self._camera_sequences: dict[str, int] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict.fromkeys(self._feature_names, float)
        for camera_name, camera_config in self.config.cameras.items():
            if getattr(camera_config, "use_rgb", True):
                features[camera_name] = (camera_config.height, camera_config.width, 3)
            if getattr(camera_config, "use_depth", False):
                features[f"{camera_name}_depth"] = (camera_config.height, camera_config.width, 1)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._feature_names, float)

    @property
    def is_connected(self) -> bool:
        base_connected = (
            self._z is not None
            and len(self._resources) >= 4
            and all(camera.is_connected for camera in self.cameras.values())
        )
        if self.config.mode == "shadow":
            return base_connected and self._arm_session_id == 0 and self._ee_session_id == 0
        return base_connected and self._arm_session_id != 0 and self._ee_session_id != 0

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped_reason

    @property
    def controller_limits(self) -> tuple[DeviceLimits, DeviceLimits]:
        if self._arm_limits is None or self._ee_limits is None:
            raise RuntimeError("controller descriptions have not been loaded")
        return self._arm_limits, self._ee_limits

    @property
    def command_target(self) -> dict[str, float] | None:
        """Return the last accepted target, primarily for stationary hold loops."""
        if self._last_target is None:
            return None
        return dict(zip(self._feature_names, self._last_target, strict=True))

    def get_camera_frame_metadata(self) -> dict[str, CameraFrameMetadata]:
        """Return freshness evidence for images from the last observation read.

        No metadata key is added to the policy observation mapping. A caller
        must require a non-``None`` monotonically increasing ``sequence`` before
        using a camera frame to advance a safety-critical visual state machine.
        """

        with self._camera_metadata_lock:
            return dict(self._camera_frame_metadata)

    def get_tcp_pose(
        self,
        measured_arm_q: Sequence[float],
        tool_offset: Sequence[float] = (0.0, 0.0, 0.13),
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
        """Return a measured TCP pose from the hash-validated live arm URDF.

        This accessor performs no device I/O and publishes no command. It is
        intentionally unavailable unless gravity/FK support was enabled and
        the live URDF digest still matches the configured expected digest.
        ``tool_offset`` is expressed in the arm terminal-link frame.
        """

        if not self.config.enable_arm_gravity_ff:
            raise RuntimeError(
                "TCP FK requires enable_arm_gravity_ff and a validated live URDF model"
            )
        if self._arm_gravity_model is None or self._arm_gravity_urdf_sha256 is None:
            raise RuntimeError("TCP FK requires a validated live arm URDF model")
        if self._arm_gravity_urdf_sha256 != self.config.expected_arm_urdf_sha256:
            raise PermissionError(
                "validated live arm URDF no longer matches the configured expected digest"
            )
        return self._arm_gravity_model.end_effector_pose(
            measured_arm_q,
            tool_offset=tool_offset,
        )

    def prime_shadow_reference(self) -> dict[str, float]:
        """Set the first shadow validation reference from one fresh state read.

        This never acquires a control session or publishes a command. It keeps
        the slew-rate adapter and follower validation on the same reference
        after a potentially slow first policy inference.
        """
        with self._lifecycle_lock:
            if self.config.mode != "shadow":
                raise PermissionError(
                    "shadow reference priming is valid only in shadow mode"
                )
            if self._z is None:
                raise ConnectionError(f"{self} is not connected")
            if self._last_target is None:
                arm = self._arm_state.get_fresh(self.config.max_state_age_s).values
                ee = self._ee_state.get_fresh(self.config.max_state_age_s).values
                measured = arm + ee
                self._validate_position_limits(
                    measured, value_label="measured position"
                )
                self._last_target = measured
                self.record_event("shadow_reference_primed", reference=measured)
            return dict(zip(self._feature_names, self._last_target, strict=True))

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self._z is not None:
            raise RuntimeError(f"{self} is already connected")
        self._disconnecting = False
        self._stopped_reason = None
        self._arm_state = LatestJointVector(len(self.config.arm_joint_names), "follower arm joint_state")
        self._ee_state = LatestJointVector(len(self.config.ee_joint_names), "follower EE joint_state")
        self._arm_status = _LatestRobotStatus("follower arm RobotStatus")
        self._ee_status = _LatestRobotStatus("follower EE RobotStatus")
        self._last_target = None
        self._last_send_at = None
        self._last_cycle_at = None
        self._arm_error_frames = 0
        self._ee_error_frames = 0
        self._sequence = 0
        self._clock_origin_ns = time.monotonic_ns()
        with self._camera_metadata_lock:
            self._camera_frame_metadata.clear()
            self._camera_capture_timestamps.clear()
            self._camera_sequences.clear()
        self._arm_gravity_model = None
        self._arm_gravity_urdf_sha256 = None
        self._ee_payload_urdf_sha256 = None
        self._ee_payload = None
        try:
            self._z = open_zenoh_session(self.config.connect)
            if self.config.zenoh_settle_time_s:
                time.sleep(self.config.zenoh_settle_time_s)
            self._declare_readers()
            for camera in self.cameras.values():
                camera.connect()
            self._arm_state.wait_ready(self.config.connect_timeout_s)
            self._ee_state.wait_ready(self.config.connect_timeout_s)
            self._arm_status.wait_ready(self.config.connect_timeout_s)
            self._ee_status.wait_ready(self.config.connect_timeout_s)
            self._load_descriptions()
            if self.config.enable_arm_gravity_ff:
                preflight_tau_ff, preflight_reference = (
                    self._arm_gravity_feed_forward_for_publish()
                )
                self.record_event(
                    "arm_gravity_ff_preflight",
                    reference_q_rad=preflight_reference,
                    tau_nm=preflight_tau_ff,
                    live_urdf_sha256=self._arm_gravity_urdf_sha256,
                    live_ee_urdf_sha256=self._ee_payload_urdf_sha256,
                    payload_mass_kg=(
                        None if self._ee_payload is None else self._ee_payload.mass
                    ),
                    payload_com_xyz_m=(
                        None if self._ee_payload is None else self._ee_payload.center_of_mass
                    ),
                    clipped=False,
                    command_published=False,
                    session_acquired=False,
                )
            if self.config.mode == "execute":
                self._acquire_both_and_hold()
                self._start_watchdog()
            self.record_event("connected", mode=self.config.mode)
        except Exception:
            self.disconnect()
            raise

    def _declare_readers(self) -> None:
        assert self._z is not None
        self._resources.extend(
            [
                self._z.declare_subscriber(
                    f"{self.config.arm_prefix}/arm/joint_state",
                    lambda sample: decode_joint_state(sample, self._arm_state),
                ),
                self._z.declare_subscriber(
                    f"{self.config.ee_prefix}/ee/joint_state",
                    lambda sample: decode_joint_state(sample, self._ee_state),
                ),
                self._z.declare_subscriber(f"{self.config.arm_prefix}/status", self._arm_status.decode),
                self._z.declare_subscriber(f"{self.config.ee_prefix}/status", self._ee_status.decode),
            ]
        )

    def _query_one(self, key: str, payload: bytes, response_type):
        assert self._z is not None
        try:
            replies = self._z.get(key, payload=payload, timeout=self.config.rpc_timeout_s)
            for reply in replies:
                sample = getattr(reply, "ok", None)
                if sample is None:
                    continue
                message = response_type()
                message.ParseFromString(sample_payload_bytes(sample))
                return message
        except Exception as exc:
            raise TimeoutError(f"RPC/query failed for {key}: {exc}") from exc
        raise TimeoutError(f"RPC/query returned no successful reply for {key}")

    def _load_descriptions(self) -> None:
        arm_description = self._query_one(
            f"{self.config.arm_prefix}/arm/description", b"", arm_pb2.ArmDescription
        )
        ee_description = self._query_one(
            f"{self.config.ee_prefix}/ee/description", b"", ee_pb2.EeDescription
        )
        self._arm_limits = DeviceLimits.from_message(
            "arm", arm_description, self.config.arm_joint_names
        )
        self._ee_limits = DeviceLimits.from_message("EE", ee_description, self.config.ee_joint_names)
        self._load_arm_gravity_model()
        arm_timeout = self._arm_timeout_behavior_value()
        required = {
            "arm": (self._arm_limits, arm_timeout),
            "EE": (self._ee_limits, arm_pb2.TIMEOUT_BEHAVIOR_HOLD),
        }
        self.record_event(
            "controller_contract",
            arm_tau_max=self._arm_limits.tau_max,
            mode=self.config.mode,
            arm_joint_names=self._arm_limits.joint_names,
            arm_pos_min=self._arm_limits.pos_min,
            arm_pos_max=self._arm_limits.pos_max,
            arm_vel_max=self._arm_limits.vel_max,
            ee_tau_max=self._ee_limits.tau_max,
            ee_joint_names=self._ee_limits.joint_names,
            ee_pos_min=self._ee_limits.pos_min,
            ee_pos_max=self._ee_limits.pos_max,
            ee_vel_max=self._ee_limits.vel_max,
            arm_timeout=arm_pb2.TimeoutBehavior.Name(arm_timeout),
            arm_advertised_timeouts=[
                arm_pb2.TimeoutBehavior.Name(value)
                for value in sorted(self._arm_limits.supported_timeouts)
            ],
            ee_timeout=arm_pb2.TimeoutBehavior.Name(arm_pb2.TIMEOUT_BEHAVIOR_HOLD),
            ee_advertised_timeouts=[
                arm_pb2.TimeoutBehavior.Name(value)
                for value in sorted(self._ee_limits.supported_timeouts)
            ],
            arm_kp=self.config.arm_kp,
            arm_kd=self.config.arm_kd,
            resolved_arm_kp=self.config.resolved_arm_kp(),
            resolved_arm_kd=self.config.resolved_arm_kd(),
            configured_arm_step_limit_rad=self.config.arm_step_limit_rad,
            configured_arm_step_limits_rad=self.config.resolved_arm_step_limits_rad(),
            configured_ee_step_limit_rad=self.config.ee_step_limit_rad,
            effective_step_limits=self._effective_step_limits(),
            ee_closing_tracking_error_rad=self.config.ee_closing_tracking_error_rad,
            ee_kp=self.config.ee_kp,
            ee_kd_default_scale=self.config.ee_kd_default_scale,
            ee_kd=[
                value * self.config.ee_kd_default_scale
                for value in self._ee_limits.default_kd
            ],
            experimental_hold_override=self.config.allow_arm_hold_timeout_without_ramp_stop,
            arm_gravity_ff_enabled=self.config.enable_arm_gravity_ff,
            arm_gravity_ff_algorithm=(
                (
                    "hex-arm-dynamics-0.1.0-compatible+"
                    "lumped_inertial_from_urdf_string+with_tip_payload"
                    if self.config.enable_ee_payload_gravity_ff
                    else "hex-arm-dynamics-0.1.0-compatible"
                )
                if self.config.enable_arm_gravity_ff
                else None
            ),
            arm_gravity_ff_expected_urdf_sha256=self.config.expected_arm_urdf_sha256,
            arm_gravity_ff_live_urdf_sha256=self._arm_gravity_urdf_sha256,
            ee_payload_gravity_ff_enabled=self.config.enable_ee_payload_gravity_ff,
            ee_payload_gravity_ff_expected_urdf_sha256=self.config.expected_ee_urdf_sha256,
            ee_payload_gravity_ff_live_urdf_sha256=self._ee_payload_urdf_sha256,
            ee_payload_expected_mass_kg=self.config.expected_ee_payload_mass_kg,
            ee_payload_expected_com_xyz_m=(
                self.config.expected_ee_payload_com_xyz_m
                if self.config.enable_ee_payload_gravity_ff else None
            ),
            ee_payload_live_mass_kg=(
                None if self._ee_payload is None else self._ee_payload.mass
            ),
            ee_payload_live_com_xyz_m=(
                None if self._ee_payload is None else self._ee_payload.center_of_mass
            ),
            payload_mount_arm_tip_link=self.config.expected_arm_urdf_tip_link,
            payload_mount_ee_root_link=self.config.expected_ee_urdf_root_link,
            payload_mount_transform=(
                "identity" if self.config.enable_ee_payload_gravity_ff else None
            ),
            arm_gravity_ff_max_abs_nm=self.config.arm_gravity_ff_max_abs_nm,
            arm_gravity_ff_gravity_m_s2=(
                [0.0, 0.0, -9.81] if self.config.enable_arm_gravity_ff else None
            ),
        )
        missing_timeouts: list[dict[str, object]] = []
        for label, (limits, timeout_behavior) in required.items():
            if timeout_behavior not in limits.supported_timeouts:
                timeout_name = arm_pb2.TimeoutBehavior.Name(timeout_behavior)
                missing_timeouts.append(
                    {
                        "device": label,
                        "required": timeout_name,
                        "advertised": [
                            arm_pb2.TimeoutBehavior.Name(value)
                            for value in sorted(limits.supported_timeouts)
                        ],
                    }
                )
        if not missing_timeouts:
            return
        detail = "; ".join(
            f"{item['device']} controller does not advertise required {item['required']}"
            for item in missing_timeouts
        )
        if self.config.mode == "execute":
            raise ValueError(detail)
        # Shadow never publishes a trajectory, so execute-only timeout
        # capabilities are diagnostic here. Keep the evidence in telemetry
        # while preserving the hard gate before any session can be acquired.
        self.record_event(
            "controller_capability_warning",
            mode="shadow",
            execute_allowed=False,
            missing_timeouts=missing_timeouts,
        )

    def _load_arm_gravity_model(self) -> None:
        if not self.config.enable_arm_gravity_ff:
            return
        resource = self._query_one(
            f"{self.config.arm_prefix}/arm/urdf", b"", arm_pb2.UrdfResource
        )
        xml = resource.xml
        if not xml.strip():
            raise ValueError("live arm URDF XML is empty; gravity feed-forward is disabled")
        digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        if digest != self.config.expected_arm_urdf_sha256:
            raise PermissionError(
                "live arm URDF SHA-256 does not match the fingerprint-bound expected digest: "
                f"expected {self.config.expected_arm_urdf_sha256}, got {digest}"
            )
        arm_xml_root, arm_xml_tip = serial_chain_endpoints(xml)
        if resource.root_link and resource.root_link != arm_xml_root:
            raise ValueError("live arm URDF declared root_link does not match its XML chain")
        if resource.tip_link and resource.tip_link != arm_xml_tip:
            raise ValueError("live arm URDF declared tip_link does not match its XML chain")
        model = SerialRevoluteGravityModel.from_urdf_string(
            xml, tuple(self.config.arm_joint_names)
        )
        if model.dof != len(self.config.arm_joint_names):
            raise ValueError(
                f"live arm gravity model DOF is {model.dof}, expected "
                f"{len(self.config.arm_joint_names)}"
            )

        payload = None
        ee_digest = None
        ee_resource = None
        algorithm = "hex-arm-dynamics-0.1.0-compatible"
        if self.config.enable_ee_payload_gravity_ff:
            if resource.root_link != self.config.expected_arm_urdf_root_link:
                raise PermissionError(
                    "live arm URDF root_link does not match the payload mount binding"
                )
            if resource.tip_link != self.config.expected_arm_urdf_tip_link:
                raise PermissionError(
                    "live arm URDF tip_link does not match the payload mount binding"
                )
            if resource.mount_links:
                raise ValueError(
                    "arm URDF advertises mount_links; non-identity payload mounts are unsupported"
                )
            ee_resource = self._query_one(
                f"{self.config.ee_prefix}/ee/urdf", b"", arm_pb2.UrdfResource
            )
            ee_xml = ee_resource.xml
            if not ee_xml.strip():
                raise ValueError("live EE URDF XML is empty; payload gravity is disabled")
            ee_digest = hashlib.sha256(ee_xml.encode("utf-8")).hexdigest()
            if ee_digest != self.config.expected_ee_urdf_sha256:
                raise PermissionError(
                    "live EE URDF SHA-256 does not match the fingerprint-bound expected digest: "
                    f"expected {self.config.expected_ee_urdf_sha256}, got {ee_digest}"
                )
            if ee_resource.root_link != self.config.expected_ee_urdf_root_link:
                raise PermissionError(
                    "live EE URDF root_link does not match the payload mount binding"
                )
            if ee_resource.tip_link != self.config.expected_ee_urdf_tip_link:
                raise PermissionError(
                    "live EE URDF tip_link does not match the payload mount binding"
                )
            if ee_resource.mount_links:
                raise ValueError(
                    "EE URDF advertises mount_links; non-identity payload mounts are unsupported"
                )
            payload = lumped_inertial_from_urdf_string(
                ee_xml,
                expected_root_link=self.config.expected_ee_urdf_root_link,
                expected_tip_link=self.config.expected_ee_urdf_tip_link,
            )
            assert self.config.expected_ee_payload_mass_kg is not None
            if not math.isclose(
                payload.mass,
                self.config.expected_ee_payload_mass_kg,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise PermissionError(
                    "live EE URDF lumped payload mass does not match the "
                    "fingerprint-bound expected mass"
                )
            for index, (actual, expected) in enumerate(
                zip(
                    payload.center_of_mass,
                    self.config.expected_ee_payload_com_xyz_m,
                    strict=True,
                )
            ):
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7):
                    raise PermissionError(
                        "live EE URDF lumped payload COM does not match the "
                        f"fingerprint-bound expected COM at axis {index}"
                    )
            model = with_tip_payload(model, payload)
            algorithm = (
                "hex-arm-dynamics-0.1.0-compatible+"
                "lumped_inertial_from_urdf_string+with_tip_payload"
            )

        self._arm_gravity_model = model
        self._arm_gravity_urdf_sha256 = digest
        self._ee_payload_urdf_sha256 = ee_digest
        self._ee_payload = payload
        self.record_event(
            "arm_gravity_ff_model_loaded",
            algorithm=algorithm,
            live_urdf_sha256=digest,
            live_ee_urdf_sha256=ee_digest,
            joint_names=model.joint_names,
            dof=model.dof,
            arm_root_link=arm_xml_root,
            arm_tip_link=arm_xml_tip,
            ee_root_link=(None if ee_resource is None else ee_resource.root_link),
            ee_tip_link=(None if ee_resource is None else ee_resource.tip_link),
            payload_mount_transform="arm_tip_to_ee_root_identity" if payload else None,
            payload_mass_kg=None if payload is None else payload.mass,
            payload_com_xyz_m=None if payload is None else payload.center_of_mass,
            gravity_m_s2=[0.0, 0.0, -9.81],
            max_abs_nm=self.config.arm_gravity_ff_max_abs_nm,
            command_published=False,
            session_acquired=False,
        )

    def _arm_timeout_behavior_value(self) -> int:
        if self.config.arm_timeout_behavior == "hold":
            return arm_pb2.TIMEOUT_BEHAVIOR_HOLD
        return arm_pb2.TIMEOUT_BEHAVIOR_RAMP_STOP

    def _new_liveliness_token(self, device_label: str) -> tuple[str, object]:
        assert self._z is not None
        safe_client = _CLIENT_COMPONENT_RE.sub("_", self.config.client_name.strip())
        key = f"hexmeow/clients/{safe_client}/{uuid.uuid4().hex}/{device_label}"
        token = self._z.liveliness().declare_token(key)
        self._liveliness_tokens.append(token)
        if not hasattr(self, "_liveliness_keys"):
            self._liveliness_keys: list[str] = []
        self._liveliness_keys.append(key)
        return key, token

    def _acquire_one(self, prefix: str, device_label: str) -> int:
        liveliness_key, _token = self._new_liveliness_token(device_label)
        request = robot_pb2.AcquireSessionRequest(
            client_name=self.config.client_name,
            liveliness_key=liveliness_key,
        )
        response = self._query_one(
            f"{prefix}/rpc/acquire_session",
            request.SerializeToString(),
            robot_pb2.AcquireSessionResponse,
        )
        if not response.ok or response.session_id == 0:
            holder_name = response.current_holder_name if response.HasField("current_holder_name") else "unknown"
            detail = response.error if response.HasField("error") else "controller rejected acquire"
            raise ConnectionError(
                f"{device_label} control is unavailable: {detail}; "
                f"holder={response.current_holder} name={holder_name!r}. "
                "Stop arm_teleop before execution."
            )
        return int(response.session_id)

    def _release_one(self, prefix: str, session_id: int) -> None:
        if session_id == 0 or self._z is None:
            return
        request = robot_pb2.ReleaseSessionRequest(session_id=session_id)
        try:
            response = self._query_one(
                f"{prefix}/rpc/release_session",
                request.SerializeToString(),
                common_pb2.GenericResponse,
            )
            if not response.ok:
                detail = response.error if response.HasField("error") else "controller rejected release"
                self._record_event_best_effort(
                    "release_error", prefix=prefix, session_id=session_id, detail=detail
                )
        except Exception as exc:
            self._record_event_best_effort(
                "release_error", prefix=prefix, session_id=session_id, detail=str(exc)
            )

    def _set_active(self, prefix: str, session_id: int) -> None:
        request = robot_pb2.SetModeRequest(
            session_id=session_id,
            mode=common_pb2.OPERATING_MODE_ACTIVE,
        )
        response = self._query_one(
            f"{prefix}/rpc/set_mode", request.SerializeToString(), common_pb2.GenericResponse
        )
        if not response.ok:
            detail = response.error if response.HasField("error") else "controller rejected ACTIVE"
            raise ConnectionError(f"failed to set {prefix} ACTIVE: {detail}")

    def _ensure_unoccupied(self) -> None:
        for label, status in (("arm", self._arm_status), ("EE", self._ee_status)):
            snapshot = status.get_fresh(self.config.max_status_age_s)
            if snapshot.session_holder != 0:
                raise ConnectionError(
                    f"{label} already has session holder {snapshot.session_holder}; "
                    "arm_teleop and all other controllers must be stopped first"
                )

    def _acquire_both_and_hold(self) -> None:
        self._ensure_unoccupied()
        arm_session_id = 0
        ee_session_id = 0
        try:
            arm_session_id = self._acquire_one(self.config.arm_prefix, "arm")
            self._arm_session_id = arm_session_id
            ee_session_id = self._acquire_one(self.config.ee_prefix, "ee")
            self._ee_session_id = ee_session_id
            arm = self._arm_state.get_fresh(self.config.max_state_age_s).values
            ee = self._ee_state.get_fresh(self.config.max_state_age_s).values
            initial_target = arm + ee
            self._validate_position_limits(initial_target, value_label="startup measured position")
            self._last_target = initial_target
            # Activate the lighter EE first, then minimize the arm's ACTIVE-without-setpoint gap.
            self._set_active(self.config.ee_prefix, ee_session_id)
            self._set_active(self.config.arm_prefix, arm_session_id)
            startup_tau_ff, startup_gravity_reference = self._publish_pair(self._last_target)
            running = common_pb2.ROBOT_MODE_RUNNING
            self._arm_status.wait_for(
                lambda status: status.mode == running and status.session_holder == arm_session_id,
                self.config.connect_timeout_s,
            )
            self._ee_status.wait_for(
                lambda status: status.mode == running and status.session_holder == ee_session_id,
                self.config.connect_timeout_s,
            )
            self._watch_liveliness()
            self.record_event(
                "control_acquired",
                arm_session_id=arm_session_id,
                ee_session_id=ee_session_id,
                hold_target=self._last_target,
                arm_gravity_ff_tau_nm=startup_tau_ff,
                arm_gravity_ff_reference_q_rad=startup_gravity_reference,
                arm_gravity_ff_live_urdf_sha256=self._arm_gravity_urdf_sha256,
                ee_payload_gravity_ff_live_urdf_sha256=self._ee_payload_urdf_sha256,
                ee_payload_mass_kg=(
                    None if self._ee_payload is None else self._ee_payload.mass
                ),
                ee_payload_com_xyz_m=(
                    None if self._ee_payload is None else self._ee_payload.center_of_mass
                ),
                arm_gravity_ff_clipped=False,
            )
        except Exception:
            self._release_one(self.config.ee_prefix, ee_session_id)
            self._release_one(self.config.arm_prefix, arm_session_id)
            self._arm_session_id = 0
            self._ee_session_id = 0
            self._close_liveliness_tokens()
            raise

    def _watch_liveliness(self) -> None:
        """Trip if either session-bound token unexpectedly disappears."""
        assert self._z is not None
        for key in getattr(self, "_liveliness_keys", ()):
            subscriber = self._z.liveliness().declare_subscriber(
                key,
                lambda sample, token_key=key: self._on_liveliness(token_key, sample),
                history=True,
            )
            self._resources.append(subscriber)

    def _on_liveliness(self, token_key: str, sample) -> None:
        kind = str(getattr(sample, "kind", "")).lower()
        if "delete" not in kind:
            return
        with self._lifecycle_lock:
            if self._disconnecting or self._stopped_reason is not None:
                return
        self._trip_safety_stop(f"session liveliness disappeared: {token_key}")

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="hex-follower-command-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        poll_s = min(0.025, self.config.max_send_gap_s / 4)
        while not self._watchdog_stop.wait(poll_s):
            with self._lifecycle_lock:
                if self._disconnecting or self._stopped_reason is not None:
                    return
                last_send_at = self._last_send_at
            if last_send_at is not None and time.monotonic() - last_send_at > self.config.max_send_gap_s:
                self._trip_safety_stop("command watchdog exceeded 250 ms send-gap limit")
                return

    def get_observation(self) -> dict:
        if self._z is None:
            raise ConnectionError(f"{self} is not connected")
        arm = self._arm_state.get_fresh(self.config.max_state_age_s).values
        ee = self._ee_state.get_fresh(self.config.max_state_age_s).values
        observation = {
            feature_name: value for feature_name, value in zip(self._feature_names, arm + ee, strict=True)
        }
        for camera_name, camera in self.cameras.items():
            camera_config = self.config.cameras[camera_name]
            captured_at_s: float | None = None
            timestamp_source = "unavailable"
            if getattr(camera_config, "use_rgb", True):
                observation[camera_name], captured_at_s, timestamp_source = (
                    self._read_latest_camera_channel(camera, read_depth=False)
                )
            if getattr(camera_config, "use_depth", False):
                depth_frame, depth_captured_at_s, depth_timestamp_source = (
                    self._read_latest_camera_channel(camera, read_depth=True)
                )
                observation[f"{camera_name}_depth"] = depth_frame
                if not getattr(camera_config, "use_rgb", True):
                    captured_at_s = depth_captured_at_s
                    timestamp_source = depth_timestamp_source
            self._record_camera_frame_metadata(
                camera_name,
                captured_at_s=captured_at_s,
                timestamp_source=timestamp_source,
            )
        return observation

    def _read_latest_camera_channel(
        self, camera, *, read_depth: bool
    ) -> tuple[object, float | None, str]:
        """Read one channel and, where supported, bind it to its host timestamp.

        LeRobot 0.6 has no public timestamp-returning Camera method. Its
        RealSense implementation retains ``latest_timestamp`` and the exact
        buffered ndarray under ``frame_lock``. We call the public read method,
        then verify under that lock that the timestamp belongs to the returned
        ndarray. A bounded retry handles a camera update racing the first read.
        """

        max_age_ms = max(1, math.ceil(self.config.camera_max_age_s * 1000.0))
        method_name = "read_latest_depth" if read_depth else "read_latest"
        reader = getattr(camera, method_name)
        frame_lock = getattr(camera, "frame_lock", None)
        frame_attribute = "latest_depth_frame" if read_depth else "latest_color_frame"
        has_host_timestamp = (
            frame_lock is not None
            and hasattr(camera, frame_attribute)
            and hasattr(camera, "latest_timestamp")
        )

        attempts = 3 if has_host_timestamp else 1
        for _attempt in range(attempts):
            frame = reader(max_age_ms=max_age_ms)
            if not has_host_timestamp:
                return frame, None, "unavailable"
            with frame_lock:
                buffered_frame = getattr(camera, frame_attribute)
                captured_at_s = getattr(camera, "latest_timestamp")
            if frame is not buffered_frame:
                continue
            if not isinstance(captured_at_s, (int, float)) or not math.isfinite(
                captured_at_s
            ):
                raise RuntimeError(f"{camera} returned an invalid host capture timestamp")
            age_s = time.perf_counter() - float(captured_at_s)
            if age_s < 0:
                raise RuntimeError(f"{camera} host capture timestamp is in the future")
            if age_s > self.config.camera_max_age_s:
                raise TimeoutError(
                    f"{camera} latest frame is too old: {age_s * 1000:.1f} ms "
                    f"(max allowed: {self.config.camera_max_age_s * 1000:.1f} ms)"
                )
            return frame, float(captured_at_s), "lerobot_host_perf_counter"
        raise RuntimeError(
            f"{camera} camera buffer changed during freshness sampling; "
            "could not bind the returned frame to a capture timestamp"
        )

    def _record_camera_frame_metadata(
        self,
        camera_name: str,
        *,
        captured_at_s: float | None,
        timestamp_source: str,
    ) -> None:
        observed_at_s = time.perf_counter()
        with self._camera_metadata_lock:
            if captured_at_s is None:
                metadata = CameraFrameMetadata(
                    camera_name=camera_name,
                    sequence=None,
                    captured_at_s=None,
                    observed_at_s=observed_at_s,
                    age_s=None,
                    timestamp_source=timestamp_source,
                    is_new_frame=False,
                )
            else:
                previous_timestamp = self._camera_capture_timestamps.get(camera_name)
                if previous_timestamp is not None and captured_at_s < previous_timestamp:
                    raise RuntimeError(
                        f"{camera_name} camera timestamp regressed from "
                        f"{previous_timestamp:.9f} to {captured_at_s:.9f}"
                    )
                is_new_frame = previous_timestamp is None or captured_at_s > previous_timestamp
                sequence = self._camera_sequences.get(camera_name, 0)
                if is_new_frame:
                    sequence += 1
                    self._camera_sequences[camera_name] = sequence
                    self._camera_capture_timestamps[camera_name] = captured_at_s
                metadata = CameraFrameMetadata(
                    camera_name=camera_name,
                    sequence=sequence,
                    captured_at_s=captured_at_s,
                    observed_at_s=observed_at_s,
                    age_s=observed_at_s - captured_at_s,
                    timestamp_source=timestamp_source,
                    is_new_frame=is_new_frame,
                )
            self._camera_frame_metadata[camera_name] = metadata

    def _validate_action_schema(self, action: dict) -> tuple[float, ...]:
        if not isinstance(action, Mapping):
            raise ValueError("action must be a mapping with the complete follower schema")
        expected = set(self._feature_names)
        received = set(action)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"action keys do not match follower schema; missing={missing}, extra={extra}")
        try:
            values = tuple(float(action[name]) for name in self._feature_names)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"action contains a non-numeric value: {exc}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action contains NaN or infinite value")
        return values

    def _validate_position_limits(
        self, values: tuple[float, ...], *, value_label: str = "target"
    ) -> None:
        arm_limits, ee_limits = self.controller_limits
        limits = tuple(zip(arm_limits.pos_min, arm_limits.pos_max, strict=True)) + tuple(
            zip(ee_limits.pos_min, ee_limits.pos_max, strict=True)
        )
        for name, value, (lower, upper) in zip(self._feature_names, values, limits, strict=True):
            if value < lower or value > upper:
                raise ValueError(
                    f"{name} {value_label} {value:.6f} is outside controller range [{lower}, {upper}]"
                )

    def _effective_step_limits(self) -> tuple[float, ...]:
        arm_limits, ee_limits = self.controller_limits
        arm = tuple(
            min(configured, velocity / self.config.command_hz)
            for configured, velocity in zip(
                self.config.resolved_arm_step_limits_rad(),
                arm_limits.vel_max,
                strict=True,
            )
        )
        ee = tuple(
            min(self.config.ee_step_limit_rad, velocity / self.config.command_hz)
            for velocity in ee_limits.vel_max
        )
        return arm + ee

    def _validate_step(self, values: tuple[float, ...], reference: tuple[float, ...]) -> None:
        for name, value, previous, limit in zip(
            self._feature_names, values, reference, self._effective_step_limits(), strict=True
        ):
            delta = abs(value - previous)
            if delta > limit + 1e-9:
                raise ValueError(
                    f"{name} step {delta:.6f} exceeds 30 Hz limit {limit:.6f}; target is not clipped"
                )

    def preflight_actions(self, actions: list[tuple[float, ...]] | list[list[float]]) -> None:
        """Validate a complete trajectory against live descriptions without publishing."""
        if not actions:
            raise ValueError("trajectory contains no actions")
        previous: tuple[float, ...] | None = None
        for index, raw_values in enumerate(actions):
            values = tuple(float(value) for value in raw_values)
            if len(values) != len(self._feature_names):
                raise ValueError(
                    f"trajectory frame {index} has {len(values)} values, expected {len(self._feature_names)}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"trajectory frame {index} contains NaN or infinite value")
            self._validate_position_limits(values)
            if previous is not None:
                try:
                    self._validate_step(values, previous)
                except ValueError as exc:
                    raise ValueError(f"trajectory frame {index}: {exc}") from exc
            previous = values

    def _check_tracking(self, measured: tuple[float, ...]) -> None:
        if self._last_target is None:
            return
        arm_count = len(self.config.arm_joint_names)
        arm_error = max(
            abs(actual - target)
            for actual, target in zip(measured[:arm_count], self._last_target[:arm_count], strict=True)
        )
        ee_deltas = tuple(
            actual - target
            for actual, target in zip(
                measured[arm_count:], self._last_target[arm_count:], strict=True
            )
        )
        closing_limit = self.config.ee_closing_tracking_error_rad
        ee_limits = tuple(
            (
                closing_limit
                if closing_limit is not None and delta > 0.0
                else self.config.ee_tracking_error_rad
            )
            for delta in ee_deltas
        )
        ee_violation = any(
            abs(delta) > limit
            for delta, limit in zip(ee_deltas, ee_limits, strict=True)
        )
        self._arm_error_frames = self._arm_error_frames + 1 if arm_error > self.config.arm_tracking_error_rad else 0
        self._ee_error_frames = self._ee_error_frames + 1 if ee_violation else 0
        if self._arm_error_frames >= self.config.tracking_error_frames:
            raise SafetyStopError(
                f"arm tracking error {arm_error:.6f} rad exceeded {self.config.arm_tracking_error_rad:.6f} "
                "for three consecutive frames"
            )
        if self._ee_error_frames >= self.config.tracking_error_frames:
            worst_index = max(
                range(len(ee_deltas)),
                key=lambda index: abs(ee_deltas[index]) - ee_limits[index],
            )
            direction = "closing-contact" if ee_deltas[worst_index] > 0.0 else "opening"
            raise SafetyStopError(
                f"EE {direction} tracking error {abs(ee_deltas[worst_index]):.6f} rad exceeded "
                f"{ee_limits[worst_index]:.6f} "
                "for three consecutive frames"
            )

    def _verify_ownership(self) -> None:
        running = common_pb2.ROBOT_MODE_RUNNING
        for label, status_cache, session_id in (
            ("arm", self._arm_status, self._arm_session_id),
            ("EE", self._ee_status, self._ee_session_id),
        ):
            status = status_cache.get_fresh(self.config.max_status_age_s)
            if status.mode != running or status.session_holder != session_id:
                raise SafetyStopError(
                    f"lost {label} control: mode={status.mode} holder={status.session_holder}, "
                    f"expected RUNNING holder={session_id}"
                )

    def send_action(self, action: dict) -> dict:
        return self.send_action_with_metadata(action)

    def send_action_with_metadata(
        self,
        action: dict,
        *,
        inference_latency_s: float | None = None,
        visual_telemetry: Mapping[str, object] | None = None,
    ) -> dict:
        started_at = time.monotonic()
        previous_cycle_at = self._last_cycle_at
        self._last_cycle_at = started_at
        validated_visual_telemetry: dict[str, object] | None = None
        measured: tuple[float, ...] | None = None
        values: tuple[float, ...] | None = None
        arm_tau_ff: tuple[float, ...] | None = None
        arm_gravity_reference: tuple[float, ...] | None = None
        try:
            if self._z is None:
                raise ConnectionError(f"{self} is not connected")
            if self._stopped_reason is not None:
                raise SafetyStopError(f"execution is stopped: {self._stopped_reason}")
            # Validate and detach telemetry before any command can be published.
            validated_visual_telemetry = _validated_visual_telemetry(
                visual_telemetry
            )
            values = self._validate_action_schema(action)
            arm = self._arm_state.get_fresh(self.config.max_state_age_s).values
            ee = self._ee_state.get_fresh(self.config.max_state_age_s).values
            measured = arm + ee
            self._validate_position_limits(measured, value_label="measured position")
            self._validate_position_limits(values)
            reference = self._last_target if self._last_target is not None else measured
            self._validate_step(values, reference)
            if self.config.mode == "execute":
                self._check_tracking(measured)
                if self._last_send_at is not None and started_at - self._last_send_at > self.config.max_send_gap_s:
                    raise SafetyStopError("command sender resumed after exceeding the 250 ms watchdog limit")
                self._verify_ownership()
                arm_tau_ff, arm_gravity_reference = self._publish_pair(values)
            else:
                self._last_target = values
            result = dict(zip(self._feature_names, values, strict=True))
            self.record_event(
                "command_cycle" if self.config.mode == "execute" else "shadow_cycle",
                control_period_s=(None if previous_cycle_at is None else started_at - previous_cycle_at),
                inference_latency_s=inference_latency_s,
                visual_telemetry=validated_visual_telemetry,
                target=values,
                measured=measured,
                effective_step_limits=self._effective_step_limits(),
                arm_gravity_ff_tau_nm=arm_tau_ff,
                arm_gravity_ff_reference_q_rad=arm_gravity_reference,
                arm_gravity_ff_live_urdf_sha256=self._arm_gravity_urdf_sha256,
                ee_payload_gravity_ff_live_urdf_sha256=self._ee_payload_urdf_sha256,
                ee_payload_mass_kg=(
                    None if self._ee_payload is None else self._ee_payload.mass
                ),
                ee_payload_com_xyz_m=(
                    None if self._ee_payload is None else self._ee_payload.center_of_mass
                ),
                arm_gravity_ff_clipped=False,
                step_limited=False,
                safety_ok=True,
            )
            return result
        except SafetyStopError as exc:
            self._trip_safety_stop(str(exc), target=values, measured=measured)
            raise
        except (TimeoutError, ValueError) as exc:
            self.record_event(
                "safety_rejection",
                target=values,
                measured=measured,
                control_period_s=(
                    None if previous_cycle_at is None else started_at - previous_cycle_at
                ),
                inference_latency_s=inference_latency_s,
                effective_step_limits=(
                    self._effective_step_limits()
                    if self._arm_limits is not None and self._ee_limits is not None
                    else None
                ),
                step_limited=False,
                safety_ok=False,
                reason=str(exc),
            )
            if self.config.mode == "execute":
                self._trip_safety_stop(str(exc), target=values, measured=measured)
                raise SafetyStopError(str(exc)) from exc
            raise
        except Exception as exc:
            if self.config.mode == "execute":
                reason = f"unexpected command-cycle failure: {exc}"
                self._trip_safety_stop(reason, target=values, measured=measured)
                raise SafetyStopError(reason) from exc
            raise

    def _arm_gravity_feed_forward_for_publish(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
        arm_count = len(self.config.arm_joint_names)
        if not self.config.enable_arm_gravity_ff:
            return (0.0,) * arm_count, None
        if self._arm_gravity_model is None or self._arm_gravity_urdf_sha256 is None:
            raise RuntimeError(
                "arm gravity feed-forward is enabled without a validated live URDF model"
            )
        if self._arm_gravity_urdf_sha256 != self.config.expected_arm_urdf_sha256:
            raise PermissionError(
                "validated live arm URDF no longer matches the configured expected digest"
            )
        if self.config.enable_ee_payload_gravity_ff:
            if self._ee_payload is None or self._ee_payload_urdf_sha256 is None:
                raise RuntimeError(
                    "EE payload gravity is enabled without a validated lumped payload"
                )
            if (
                self._ee_payload_urdf_sha256
                != self.config.expected_ee_urdf_sha256
            ):
                raise PermissionError(
                    "validated live EE URDF no longer matches the configured expected digest"
                )
        limits, _ee_limits = self.controller_limits
        measured_q = self._arm_state.get_fresh(self.config.max_state_age_s).values
        for name, value, lower, upper in zip(
            self.config.arm_joint_names,
            measured_q,
            limits.pos_min,
            limits.pos_max,
            strict=True,
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} gravity reference contains NaN or infinite value")
            if value < lower or value > upper:
                raise ValueError(
                    f"{name} gravity reference {value:.6f} is outside controller range "
                    f"[{lower}, {upper}]"
                )
        tau_ff = self._arm_gravity_model.gravity_torque(
            measured_q, gravity=(0.0, 0.0, -9.81)
        )
        if len(tau_ff) != arm_count:
            raise ValueError(
                f"arm gravity feed-forward returned {len(tau_ff)} values, expected {arm_count}"
            )
        cap = float(self.config.arm_gravity_ff_max_abs_nm)
        cap_maximum = 8.0 if self.config.enable_ee_payload_gravity_ff else 6.0
        if not math.isfinite(cap) or cap <= 0 or cap > cap_maximum:
            raise ValueError(
                "arm_gravity_ff_max_abs_nm changed outside the validated range "
                f"(0, {cap_maximum}]"
            )
        for name, value, live_maximum in zip(
            self.config.arm_joint_names, tau_ff, limits.tau_max, strict=True
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} gravity feed-forward contains NaN or infinite torque"
                )
            effective_maximum = min(float(live_maximum), cap)
            if abs(value) > effective_maximum:
                raise ValueError(
                    f"{name} gravity feed-forward {value:.6f} Nm exceeds effective "
                    f"{effective_maximum:.6f} Nm limit "
                    f"(live tau_max={live_maximum:.6f}, configured cap={cap:.6f}); "
                    "torque is not clipped"
                )
        return tuple(float(value) for value in tau_ff), tuple(measured_q)

    def _publish_pair(
        self, values: tuple[float, ...]
    ) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
        with self._lifecycle_lock:
            assert self._z is not None
            if self._disconnecting:
                raise SafetyStopError("disconnect is in progress; command publication is disabled")
            if self._stopped_reason is not None:
                raise SafetyStopError(f"execution is stopped: {self._stopped_reason}")
            if self._arm_session_id == 0 or self._ee_session_id == 0:
                raise SafetyStopError("cannot publish without both arm and EE sessions")
            arm_count = len(self.config.arm_joint_names)
            now_ns = time.monotonic_ns() - self._clock_origin_ns
            period_ns = round(1_000_000_000 / self.config.command_hz)
            self._sequence += 1
            arm_values = values[:arm_count]
            ee_values = values[arm_count:]
            assert self._ee_limits is not None
            arm_tau_ff, arm_gravity_reference = (
                self._arm_gravity_feed_forward_for_publish()
            )
            arm_command = arm_pb2.JointTrajectory(
                header=common_pb2.Header(seq=self._sequence, stamp_ns=now_ns),
                session_id=self._arm_session_id,
                points=[
                    arm_pb2.JointSetpoint(
                        q=arm_values,
                        dq=[0.0] * len(arm_values),
                        kp=list(self.config.resolved_arm_kp()),
                        kd=list(self.config.resolved_arm_kd()),
                        tau_ff=arm_tau_ff,
                    )
                ],
                t_from_start_ns=[period_ns],
                on_timeout=self._arm_timeout_behavior_value(),
            )
            ee_command = arm_pb2.JointTrajectory(
                header=common_pb2.Header(seq=self._sequence, stamp_ns=now_ns),
                session_id=self._ee_session_id,
                points=[
                    arm_pb2.JointSetpoint(
                        q=ee_values,
                        dq=[0.0] * len(ee_values),
                        kp=[self.config.ee_kp] * len(ee_values),
                        kd=[value * self.config.ee_kd_default_scale for value in self._ee_limits.default_kd],
                        tau_ff=[0.0] * len(ee_values),
                    )
                ],
                t_from_start_ns=[period_ns],
                on_timeout=arm_pb2.TIMEOUT_BEHAVIOR_HOLD,
            )
            try:
                self._z.put(f"{self.config.arm_prefix}/arm/command", arm_command.SerializeToString())
                self._z.put(f"{self.config.ee_prefix}/ee/command", ee_command.SerializeToString())
            except Exception as exc:
                raise SafetyStopError(f"command publication failed: {exc}") from exc
            self._last_target = values
            self._last_send_at = time.monotonic()
            return arm_tau_ff, arm_gravity_reference

    def stop(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("stop reason must not be empty")
        self._trip_safety_stop(reason)

    def _trip_safety_stop(self, reason: str, **telemetry) -> None:
        with self._lifecycle_lock:
            if self._stopped_reason is not None:
                return
            self._stopped_reason = reason
            self._watchdog_stop.set()
            self._record_event_best_effort("safety_stop", reason=reason, **telemetry)
            arm_session_id = self._arm_session_id
            ee_session_id = self._ee_session_id
            self._arm_session_id = 0
            self._ee_session_id = 0
            self._release_one(self.config.ee_prefix, ee_session_id)
            self._release_one(self.config.arm_prefix, arm_session_id)
            self._close_liveliness_tokens()

    def record_event(self, event: str, **fields) -> None:
        path = self.config.telemetry_path
        if not path:
            return
        payload = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_lock, destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=list) + "\n")

    def _record_event_best_effort(self, event: str, **fields) -> None:
        try:
            self.record_event(event, **fields)
        except Exception:
            # Cleanup must never be interrupted by a full disk or malformed
            # telemetry path. Normal command-cycle logging remains fail-closed.
            pass

    def _close_liveliness_tokens(self) -> None:
        while self._liveliness_tokens:
            close_zenoh_resource(self._liveliness_tokens.pop())
        if hasattr(self, "_liveliness_keys"):
            self._liveliness_keys.clear()

    def disconnect(self) -> None:
        with self._lifecycle_lock:
            self._disconnecting = True
            self._watchdog_stop.set()
            arm_session_id = self._arm_session_id
            ee_session_id = self._ee_session_id
            self._arm_session_id = 0
            self._ee_session_id = 0
            self._release_one(self.config.ee_prefix, ee_session_id)
            self._release_one(self.config.arm_prefix, arm_session_id)
            self._close_liveliness_tokens()
        thread = self._watchdog_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._watchdog_thread = None
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                pass
        while self._resources:
            close_zenoh_resource(self._resources.pop())
        if self._z is not None:
            try:
                self._z.close()
            except Exception:
                pass
            self._z = None
        self._last_target = None
        self._last_send_at = None
        self.record_event("disconnected", stopped_reason=self._stopped_reason)
