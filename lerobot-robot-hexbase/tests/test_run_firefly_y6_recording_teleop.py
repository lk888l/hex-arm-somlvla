from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import run_firefly_y6_recording_teleop as teleop


def test_locked_command_has_one_follow_mode_and_fixed_zero_feedback() -> None:
    command = teleop.printable_command("relative")
    words = command.split()

    assert words.count("MODE=follow") == 1
    assert not any(word == "MODE=dfr" for word in words)
    assert "LEADER_MODE=gravity_comp" in words
    assert "ALIGN=relative" in words
    assert "RATE_HZ=1000" in words
    assert "SECS=0" in words
    assert "ARM_KP=40,40,40,48,40,40" in words
    assert "ARM_KD=2,2,2,2.4,2,2" in words
    assert "DFR_GAIN=0" in words
    assert "DFR_TAU_LIMIT=0" in words
    assert "DFR_LEADER_KD=0" in words
    assert "EE_KP=150" in words
    assert "EE_SQUEEZE=0.04" in words
    assert "EE_DFR_GAIN=0" in words
    assert "EE_DFR_TAU_LIMIT=0" in words
    assert "EE_DFR_LEADER_KD=0" in words
    assert tuple(words[-4:]) == teleop.ROBOT_PREFIXES


def test_cli_allows_only_locked_actions_and_alignments() -> None:
    assert teleop.parse_args(["--check"]).align == "relative"
    assert teleop.parse_args(["--execute", "--align", "absolute"]).align == "absolute"

    with pytest.raises(SystemExit):
        teleop.parse_args(["--execute", "--align", "unsafe"])
    with pytest.raises(SystemExit):
        teleop.parse_args(["--execute", "--arm-kp", "99"])
    with pytest.raises(SystemExit):
        teleop.parse_args(["--check", "--execute"])
    with pytest.raises(SystemExit):
        teleop.parse_args([])


def test_print_command_never_runs_preflight_or_exec(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        teleop,
        "run_preflight",
        lambda: pytest.fail("--print-command must not run preflight"),
    )
    monkeypatch.setattr(
        teleop.os,
        "execve",
        lambda *_args: pytest.fail("--print-command must not exec"),
    )

    assert teleop.main(["--print-command", "--align", "absolute"]) == 0
    captured = capsys.readouterr()
    assert "ALIGN=absolute" in captured.out
    assert "15-degree" in captured.err


def test_check_runs_preflight_but_never_exec(monkeypatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr(teleop, "run_preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(
        teleop.os,
        "execve",
        lambda *_args: pytest.fail("--check must not exec"),
    )

    assert teleop.main(["--check"]) == 0
    assert calls == ["preflight"]
    assert "No process was executed" in capsys.readouterr().out


def test_execute_preflights_then_execs_locked_binary(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(teleop, "run_preflight", lambda: calls.append("preflight"))

    def fake_execve(path, argv, environment):
        calls.append((path, argv, environment))
        raise RuntimeError("exec sentinel")

    monkeypatch.setattr(teleop.os, "execve", fake_execve)

    with pytest.raises(RuntimeError, match="exec sentinel"):
        teleop.main(["--execute", "--align", "absolute"])

    assert calls[0] == "preflight"
    path, argv, environment = calls[1]
    assert path == str(teleop.ARM_TELEOP)
    assert tuple(argv) == teleop.command_argv()
    assert environment["MODE"] == "follow"
    assert environment["LEADER_MODE"] == "gravity_comp"
    assert environment["ALIGN"] == "absolute"
    assert environment["DFR_GAIN"] == "0"
    assert environment["EE_DFR_GAIN"] == "0"


def test_preflight_failure_prevents_execute(monkeypatch, capsys) -> None:
    def fail_preflight():
        raise teleop.PreflightError("not safe")

    monkeypatch.setattr(teleop, "run_preflight", fail_preflight)
    monkeypatch.setattr(
        teleop.os,
        "execve",
        lambda *_args: pytest.fail("failed preflight must not exec"),
    )

    assert teleop.main(["--execute"]) == 2
    assert "PRECHECK FAILED: not safe" in capsys.readouterr().err


def test_verify_binary_accepts_only_expected_fingerprint(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "arm_teleop"
    binary.write_bytes(b"reviewed binary")
    binary.chmod(0o755)
    expected = hashlib.sha256(b"reviewed binary").hexdigest()
    monkeypatch.setattr(teleop, "ARM_TELEOP_SHA256", expected)

    teleop.verify_binary(binary)

    binary.write_bytes(b"changed binary")
    with pytest.raises(teleop.PreflightError, match="fingerprint mismatch"):
        teleop.verify_binary(binary)


def test_arm_teleop_process_detection_uses_executable_name(tmp_path, monkeypatch) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "self").mkdir()
    (proc / "101").mkdir()
    (proc / "202").mkdir()
    other = tmp_path / "python"
    other.write_text("")
    running = tmp_path / "arm_teleop"
    running.write_text("")
    (proc / "101" / "exe").symlink_to(running)
    (proc / "202" / "exe").symlink_to(other)
    monkeypatch.setattr(teleop, "ARM_TELEOP", tmp_path / "locked" / "arm_teleop")

    assert teleop.arm_teleop_pids(proc, current_pid=999) == (101,)
    with pytest.raises(teleop.PreflightError, match="PID 101"):
        teleop.require_no_arm_teleop(proc)


def test_execution_environment_overrides_inherited_unsafe_values() -> None:
    environment = teleop.execution_environment(
        "relative",
        {
            "PATH": "/usr/bin",
            "MODE": "dfr",
            "DFR_GAIN": "9",
            "EE_DFR_GAIN": "9",
            "ARM_KP": "999",
            "RATE_HZ": "20",
            "SECS": "1",
            "ARM_KP_SCALE": "9",
            "ARM_KD_SCALE": "9",
            "EE_KD": "999",
            "EE_KP_SCALE": "9",
            "EE_KD_SCALE": "9",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["MODE"] == "follow"
    assert environment["DFR_GAIN"] == "0"
    assert environment["EE_DFR_GAIN"] == "0"
    assert environment["ARM_KP"] == "40,40,40,48,40,40"
    assert environment["RATE_HZ"] == "1000"
    assert environment["SECS"] == "0"
    assert "ARM_KP_SCALE" not in environment
    assert "ARM_KD_SCALE" not in environment
    assert "EE_KD" not in environment
    assert "EE_KP_SCALE" not in environment
    assert "EE_KD_SCALE" not in environment


def test_run_preflight_checks_binary_process_and_router_in_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(teleop, "verify_binary", lambda: calls.append("binary"))
    monkeypatch.setattr(teleop, "require_no_arm_teleop", lambda: calls.append("process"))
    monkeypatch.setattr(teleop, "require_router", lambda: calls.append("router"))

    teleop.run_preflight()

    assert calls == ["binary", "process", "router"]
