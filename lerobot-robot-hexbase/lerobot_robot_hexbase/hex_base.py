"""HexBase —— lerobot Robot,通过 hex-controller 的 robot_api(Zenoh)驱动全向底盘。

action/observation 用 lerobot 底盘惯例(对齐 lekiwi):
  x.vel (m/s 前进)、y.vel (m/s 左移)、theta.vel (**deg/s** 自转)
我们的 robot_api Twist.wz 是 rad/s,故在边界做 deg<->rad 转换。
纯底盘,无相机(VLA 阶段再加)。zenoh-python 底层即 Rust。
"""

import math
import os
import sys
import time
from dataclasses import dataclass

import zenoh

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig

# 生成的 protobuf(scripts/gen_proto.sh 放到 proto/,flat 互 import)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))
import common_pb2  # noqa: E402
import robot_pb2  # noqa: E402
import base_pb2  # noqa: E402

DEG = 180.0 / math.pi
RAD = math.pi / 180.0


@RobotConfig.register_subclass("hexbase")
@dataclass
class HexBaseConfig(RobotConfig):
    connect: str = ""          # 空=组播自动发现局域网
    robot_prefix: str = ""      # 空=自动发现首个 BASE


class HexBase(Robot):
    config_class = HexBaseConfig
    name = "hexbase"

    def __init__(self, config: HexBaseConfig):
        super().__init__(config)
        self.config = config
        self._z = None
        self._sub = None
        self._sid = 0
        self._prefix = None
        self._odom = (0.0, 0.0, 0.0)  # measured vx,vy,wz (rad/s)

    # ---- features ----
    @property
    def _ft(self):
        return {"x.vel": float, "y.vel": float, "theta.vel": float}

    @property
    def observation_features(self) -> dict:
        return self._ft

    @property
    def action_features(self) -> dict:
        return self._ft

    @property
    def is_connected(self) -> bool:
        return self._z is not None and self._sid != 0

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---- robot_api helpers ----
    def _query_one(self, key, payload, resp_cls):
        for reply in self._z.get(key, payload=payload):
            if reply.ok is not None:
                m = resp_cls()
                m.ParseFromString(bytes(reply.ok.payload.to_bytes()))
                return m
        return None

    def _discover(self):
        for reply in self._z.get("hexmeow/**/description"):
            if reply.ok is None:
                continue
            d = robot_pb2.RobotDescription()
            d.ParseFromString(bytes(reply.ok.payload.to_bytes()))
            if d.kind == common_pb2.ROBOT_KIND_BASE:
                k = str(reply.ok.key_expr)
                return k[: -len("/description")] if k.endswith("/description") else k
        return None

    def _on_odom(self, sample):
        o = base_pb2.Odometry()
        o.ParseFromString(bytes(sample.payload.to_bytes()))
        self._odom = (o.twist.vx, o.twist.vy, o.twist.wz)

    # ---- lifecycle ----
    def connect(self, calibrate: bool = True) -> None:
        conf = zenoh.Config()
        conf.insert_json5("mode", '"peer"')
        if self.config.connect:
            conf.insert_json5("connect/endpoints", '["%s"]' % self.config.connect)
        self._z = zenoh.open(conf)
        time.sleep(0.7)  # 等组播建链

        self._prefix = self.config.robot_prefix or self._discover()
        if not self._prefix:
            raise ConnectionError("没发现 BASE(控制器起了吗?MOCK=1?)")

        req = robot_pb2.AcquireSessionRequest(client_name="lerobot_hexbase")
        resp = self._query_one(f"{self._prefix}/rpc/acquire_session",
                               req.SerializeToString(), robot_pb2.AcquireSessionResponse)
        if resp is None or not resp.ok:
            raise ConnectionError(f"取控失败: {resp.error if resp else 'no reply'}")
        self._sid = resp.session_id

        # set ACTIVE
        sm = robot_pb2.SetModeRequest(session_id=self._sid, mode=common_pb2.OPERATING_MODE_ACTIVE)
        self._query_one(f"{self._prefix}/rpc/set_mode", sm.SerializeToString(), common_pb2.GenericResponse)

        self._sub = self._z.declare_subscriber(f"{self._prefix}/base/odom", self._on_odom)

    def get_observation(self) -> dict:
        vx, vy, wz = self._odom
        return {"x.vel": vx, "y.vel": vy, "theta.vel": wz * DEG}

    def send_action(self, action: dict) -> dict:
        # 原生键:x.vel/y.vel/theta.vel(theta deg/s)。
        # 兼容 keyboard_rover 的 {linear_velocity, angular_velocity}(m/s, rad/s) —— 它走 identity 处理器直达这里。
        if "linear_velocity" in action or "angular_velocity" in action:
            vx = float(action.get("linear_velocity", 0.0))
            vy = 0.0
            wz = float(action.get("angular_velocity", 0.0))  # rad/s
            th = wz * DEG
        else:
            vx = float(action.get("x.vel", 0.0))
            vy = float(action.get("y.vel", 0.0))
            th = float(action.get("theta.vel", 0.0))  # deg/s
            wz = th * RAD
        cmd = base_pb2.BaseCommand(session_id=self._sid,
                                   twist=base_pb2.Twist(vx=vx, vy=vy, wz=wz))
        self._z.put(f"{self._prefix}/base/cmd_vel", cmd.SerializeToString())
        return {"x.vel": vx, "y.vel": vy, "theta.vel": th}

    def disconnect(self) -> None:
        try:
            if self._sid:
                rel = robot_pb2.ReleaseSessionRequest(session_id=self._sid)
                self._query_one(f"{self._prefix}/rpc/release_session",
                                rel.SerializeToString(), common_pb2.GenericResponse)
        except Exception:
            pass
        self._sid = 0
        try:
            if self._z is not None:
                self._z.close()
        finally:
            self._z = None
