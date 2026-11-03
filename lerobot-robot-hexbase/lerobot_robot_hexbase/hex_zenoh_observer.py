"""Read-only Hex follower observation source for LeRobot dataset recording."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .zenoh_helpers import (
    LatestJointVector,
    close_zenoh_resource,
    decode_joint_state,
    joint_feature_names,
    normalized_prefix,
    open_zenoh_session,
)


@RobotConfig.register_subclass("hex_zenoh_observer")
@dataclass
class HexZenohObserverConfig(RobotConfig):
    """Static schema plus Zenoh endpoints for one follower arm and its EE."""

    connect: str = "tcp/127.0.0.1:7447"
    arm_prefix: str = ""
    ee_prefix: str = ""
    arm_joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{index}" for index in range(1, 7)]
    )
    ee_joint_names: list[str] = field(default_factory=lambda: ["jaw_joint1"])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    connect_timeout_s: float = 5.0
    max_state_age_s: float = 0.25
    zenoh_settle_time_s: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.arm_prefix = normalized_prefix(self.arm_prefix, label="arm_prefix")
        self.ee_prefix = normalized_prefix(self.ee_prefix, label="ee_prefix")
        joint_feature_names(self.arm_joint_names, self.ee_joint_names)
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        if self.max_state_age_s <= 0:
            raise ValueError("max_state_age_s must be positive")
        if self.zenoh_settle_time_s < 0:
            raise ValueError("zenoh_settle_time_s must be non-negative")


class HexZenohObserver(Robot):
    """Reads actual follower q and cameras without acquiring a control session."""

    config_class = HexZenohObserverConfig
    name = "hex_zenoh_observer"

    def __init__(self, config: HexZenohObserverConfig):
        super().__init__(config)
        self.config = config
        self._feature_names = joint_feature_names(config.arm_joint_names, config.ee_joint_names)
        self._arm_state = LatestJointVector(len(config.arm_joint_names), "follower arm joint_state")
        self._ee_state = LatestJointVector(len(config.ee_joint_names), "follower EE joint_state")
        self.cameras = make_cameras_from_configs(config.cameras)
        self._z = None
        self._arm_sub = None
        self._ee_sub = None

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
        return (
            self._z is not None
            and self._arm_sub is not None
            and self._ee_sub is not None
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        try:
            self._z = open_zenoh_session(self.config.connect)
            if self.config.zenoh_settle_time_s:
                time.sleep(self.config.zenoh_settle_time_s)
            self._arm_sub = self._z.declare_subscriber(
                f"{self.config.arm_prefix}/arm/joint_state",
                lambda sample: decode_joint_state(sample, self._arm_state),
            )
            self._ee_sub = self._z.declare_subscriber(
                f"{self.config.ee_prefix}/ee/joint_state",
                lambda sample: decode_joint_state(sample, self._ee_state),
            )
            for camera in self.cameras.values():
                camera.connect()
            self._arm_state.wait_ready(self.config.connect_timeout_s)
            self._ee_state.wait_ready(self.config.connect_timeout_s)
        except Exception:
            self.disconnect()
            raise

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected")
        arm = self._arm_state.get_fresh(self.config.max_state_age_s).values
        ee = self._ee_state.get_fresh(self.config.max_state_age_s).values
        observation = {
            feature_name: value for feature_name, value in zip(self._feature_names, arm + ee, strict=True)
        }
        for camera_name, camera in self.cameras.items():
            camera_config = self.config.cameras[camera_name]
            if getattr(camera_config, "use_rgb", True):
                observation[camera_name] = camera.read_latest()
            if getattr(camera_config, "use_depth", False):
                observation[f"{camera_name}_depth"] = camera.read_latest_depth()
        return observation

    def send_action(self, action: dict) -> dict:
        """Validate and return the observed command; never publish or acquire control."""
        expected = set(self._feature_names)
        received = set(action)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"action keys do not match Hex follower schema; missing={missing}, extra={extra}")
        result = {name: float(action[name]) for name in self._feature_names}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("action contains NaN or infinite value")
        return result

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                pass
        close_zenoh_resource(self._arm_sub)
        close_zenoh_resource(self._ee_sub)
        self._arm_sub = None
        self._ee_sub = None
        if self._z is not None:
            try:
                self._z.close()
            except Exception:
                pass
            self._z = None
