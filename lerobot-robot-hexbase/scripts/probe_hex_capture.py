#!/usr/bin/env python3
"""Probe Hex follower state and command topics before starting LeRobot recording."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

try:
    from lerobot_robot_hexbase import (
        HexZenohCommandTap,
        HexZenohCommandTapConfig,
        HexZenohObserver,
        HexZenohObserverConfig,
    )
except ModuleNotFoundError as exc:
    io_python = (
        _REPOSITORY_ROOT.parent / "miniforge3" / "envs" / "lerobot-io" / "bin" / "python"
    )
    recommended_python = io_python if io_python.is_file() else Path("python")
    raise SystemExit(
        f"Missing Python module {exc.name!r} in {sys.executable}.\n"
        "This probe must run in the lerobot-io environment; do not use sudo.\n"
        "Run:\n"
        f"  {recommended_python} scripts/probe_hex_capture.py --state-only"
    ) from None


def _format_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read actual follower q and validate requested targets published by arm_teleop."
    )
    parser.add_argument(
        "--controller-id",
        default="50e41fe1cf464c2581a44f1339e6eb4e",
        help="hex-controller ID used below hexmeow/<controller-id>/...",
    )
    parser.add_argument("--arm-index", default="arm1")
    parser.add_argument("--ee-index", default="ee1")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--max-age-s", type=float, default=0.1)
    parser.add_argument("--max-status-age-s", type=float, default=0.3)
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Do not wait for command topics; useful before arm_teleop is running.",
    )
    parser.add_argument(
        "--contract-output",
        type=Path,
        help="Write the live recording contract as JSON for post-recording audit.",
    )
    args = parser.parse_args()

    if (
        args.duration_s <= 0
        or args.hz <= 0
        or args.max_age_s <= 0
        or args.max_status_age_s <= 0
    ):
        parser.error("duration-s, hz, max-age-s, and max-status-age-s must be positive")
    if not args.state_only and not math.isclose(args.hz, 30.0, rel_tol=0.0, abs_tol=1e-9):
        parser.error("full state/action validation must run at 30 Hz")
    if args.state_only and args.contract_output is not None:
        parser.error("--contract-output requires full state/action validation")

    root = f"hexmeow/{args.controller_id}"
    arm_prefix = f"{root}/{args.arm_index}"
    ee_prefix = f"{root}/{args.ee_index}"
    common = {
        "connect": args.connect,
        "arm_prefix": arm_prefix,
        "ee_prefix": ee_prefix,
        "connect_timeout_s": 5,
        "zenoh_settle_time_s": 0.2,
    }

    robot = HexZenohObserver(
        HexZenohObserverConfig(
            id="hex-capture-probe",
            max_state_age_s=args.max_age_s,
            **common,
        )
    )
    command_tap = None
    if not args.state_only:
        command_tap = HexZenohCommandTap(
            HexZenohCommandTapConfig(
                id="hex-command-probe",
                max_command_age_s=args.max_age_s,
                max_state_age_s=args.max_age_s,
                max_status_age_s=args.max_status_age_s,
                record_hz=args.hz,
                **common,
            )
        )

    live_contract = None
    try:
        robot.connect()
        if command_tap is not None:
            command_tap.connect()
            if args.contract_output is not None:
                live_contract = command_tap.controller_contract()

        names = list(robot.action_features)
        print("Joint order:", ", ".join(names))
        print(
            "Units: rad; state=measured follower q; "
            "target=controller-validated requested JointTrajectory q"
        )
        started = time.monotonic()
        period_s = 1 / args.hz
        while time.monotonic() - started < args.duration_s:
            observation = robot.get_observation()
            state = [float(observation[name]) for name in names]
            line = f"t={time.monotonic() - started:7.3f}s state={_format_vector(state)}"
            if command_tap is not None:
                action = command_tap.get_action()
                target = [float(action[name]) for name in names]
                line += f" target={_format_vector(target)}"
            print(line)
            time.sleep(period_s)
        if live_contract is not None:
            output_path = args.contract_output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps(live_contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(output_path)
            print(f"Validated live controller contract: {output_path}")
    finally:
        if command_tap is not None:
            command_tap.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
