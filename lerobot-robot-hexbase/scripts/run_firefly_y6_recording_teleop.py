#!/usr/bin/env python3
"""Preflight and run the fingerprint-locked Firefly Y6 recording teleop client.

This wrapper deliberately exposes no gain, force-feedback, endpoint, or robot-prefix
overrides.  It is for the zero-force-feedback recording setup only.  ``--check`` and
``--print-command`` never execute ``arm_teleop``; only ``--execute`` can replace this
process with the reviewed binary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import socket
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


ARM_TELEOP = Path("/home/kk/kk_data/hex-arm/remote-control/arm_teleop")
ARM_TELEOP_SHA256 = "65e1a36dfd7ba264c69b5e905c16a6558e16cd09a44d86e4563180b81d903ab3"
ROUTER_HOST = "127.0.0.1"
ROUTER_PORT = 7447

ROBOT_PREFIXES = (
    "hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm0",
    "hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/arm1",
    "hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee0",
    "hexmeow/50e41fe1cf464c2581a44f1339e6eb4e/ee1",
)

# Keep this ordered: the printed command is intended to be human-auditable.
_FIXED_ENV = (
    ("CONNECT", "tcp/127.0.0.1:7447"),
    ("MODE", "follow"),
    ("LEADER_MODE", "gravity_comp"),
    ("GRAVITY_FF", "1"),
    ("RATE_HZ", "1000"),
    ("SECS", "0"),
    ("ARM_KP", "40,40,40,48,40,40"),
    ("ARM_KD", "2,2,2,2.4,2,2"),
    ("DFR_GAIN", "0"),
    ("DFR_TAU_LIMIT", "0"),
    ("DFR_LEADER_KD", "0"),
    ("EE_KP", "150"),
    ("EE_SQUEEZE", "0.04"),
    ("EE_DFR_GAIN", "0"),
    ("EE_DFR_TAU_LIMIT", "0"),
    ("EE_DFR_LEADER_KD", "0"),
)

_REMOVED_INHERITED_ENV = frozenset(
    {
        "ARM_KP_SCALE",
        "ARM_KD_SCALE",
        "EE_KD",
        "EE_KP_SCALE",
        "EE_KD_SCALE",
    }
)


class PreflightError(RuntimeError):
    """The recording teleop process is not safe to start."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check",
        action="store_true",
        help="Run all preflight checks and exit without executing arm_teleop.",
    )
    action.add_argument(
        "--print-command",
        action="store_true",
        help="Print the locked command and exit without executing or probing hardware.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Run all preflight checks, then explicitly execute the locked arm_teleop.",
    )
    parser.add_argument(
        "--align",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "Leader/follower alignment mode (default: relative). Both modes retain "
            "arm_teleop's built-in 15-degree startup hard gate."
        ),
    )
    return parser.parse_args(argv)


def locked_environment(align: str) -> dict[str, str]:
    if align not in {"relative", "absolute"}:
        raise ValueError("align must be 'relative' or 'absolute'")
    return {**dict(_FIXED_ENV), "ALIGN": align}


def command_argv() -> tuple[str, ...]:
    return (str(ARM_TELEOP), *ROBOT_PREFIXES)


def printable_command(align: str) -> str:
    assignments = [f"{name}={value}" for name, value in locked_environment(align).items()]
    return shlex.join([*assignments, *command_argv()])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_binary(path: Path = ARM_TELEOP) -> None:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise PreflightError(f"cannot stat locked arm_teleop binary {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PreflightError(f"locked arm_teleop path is not a regular file: {path}")
    if not os.access(path, os.X_OK):
        raise PreflightError(f"locked arm_teleop binary is not executable: {path}")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise PreflightError(f"cannot hash locked arm_teleop binary {path}: {exc}") from exc
    if actual != ARM_TELEOP_SHA256:
        raise PreflightError(
            "arm_teleop fingerprint mismatch: "
            f"expected {ARM_TELEOP_SHA256}, got {actual}; refusing to execute"
        )


def _process_executable(pid_dir: Path) -> Path | None:
    try:
        return (pid_dir / "exe").resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def arm_teleop_pids(
    proc_root: Path = Path("/proc"),
    *,
    current_pid: int | None = None,
) -> tuple[int, ...]:
    """Return PIDs whose executable is the locked binary or is named arm_teleop."""

    if current_pid is None:
        current_pid = os.getpid()
    matches: list[int] = []
    try:
        entries: Iterable[Path] = proc_root.iterdir()
    except OSError as exc:
        raise PreflightError(f"cannot inspect running processes under {proc_root}: {exc}") from exc
    locked_resolved = ARM_TELEOP.resolve(strict=False)
    for pid_dir in entries:
        if not pid_dir.name.isdecimal():
            continue
        pid = int(pid_dir.name)
        if pid == current_pid:
            continue
        executable = _process_executable(pid_dir)
        if executable is None:
            continue
        if executable == locked_resolved or executable.name == "arm_teleop":
            matches.append(pid)
    return tuple(sorted(matches))


def require_no_arm_teleop(proc_root: Path = Path("/proc")) -> None:
    running = arm_teleop_pids(proc_root)
    if running:
        formatted = ", ".join(str(pid) for pid in running)
        raise PreflightError(
            f"another arm_teleop process is already running (PID {formatted}); "
            "stop it normally before retrying"
        )


def require_router(*, timeout_s: float = 0.5) -> None:
    try:
        connection = socket.create_connection((ROUTER_HOST, ROUTER_PORT), timeout=timeout_s)
    except OSError as exc:
        raise PreflightError(
            f"Zenoh router is not reachable at tcp/{ROUTER_HOST}:{ROUTER_PORT}: {exc}; "
            "start/verify the supervisor and dual-arm launcher first"
        ) from exc
    connection.close()


def run_preflight() -> None:
    verify_binary()
    require_no_arm_teleop()
    require_router()


def execution_environment(
    align: str,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if inherited is None else inherited)
    for name in _REMOVED_INHERITED_ENV:
        environment.pop(name, None)
    environment.update(locked_environment(align))
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = printable_command(args.align)
    if args.print_command:
        print(command)
        print(
            "NOTE: ALIGN=relative and ALIGN=absolute both retain arm_teleop's "
            "built-in 15-degree startup hard gate.",
            file=sys.stderr,
        )
        return 0

    try:
        run_preflight()
    except PreflightError as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print("PRECHECK PASSED: binary fingerprint, port 7447, and process exclusivity")
        print(command)
        print(
            "No process was executed. The 15-degree startup hard gate remains active "
            f"with ALIGN={args.align}."
        )
        return 0

    print("PRECHECK PASSED; executing fingerprint-locked arm_teleop", flush=True)
    os.execve(
        str(ARM_TELEOP),
        list(command_argv()),
        execution_environment(args.align),
    )
    raise AssertionError("os.execve returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
