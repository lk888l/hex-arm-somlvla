from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_firefly_y6_recording_recovery as recovery


def _document() -> dict:
    return {
        "schema_version": 1,
        "profile_id": "offline-test-only",
        "commissioning": {name: False for name in recovery.COMMISSIONING_GATES},
        "planning": {
            "home": [0.20, 0.25, 0.30, 0.40, -0.20, 0.10, 0.15],
            "home_tolerance": [0.02] * 7,
            "position_min": [-2.0] * 7,
            "position_max": [2.0] * 7,
            "entry_min": [-0.50] * 7,
            "entry_max": [0.50] * 7,
            "recovery_min": [-1.0] * 7,
            "recovery_max": [1.0] * 7,
            "step_cap": [0.10] * 6 + [0.05],
            "j4_clearance": 0.40,
            "max_frames": 100,
            "max_total_travel_rad": 10.0,
            "allow_positive_j3_motion": True,
        },
        "control": {
            "candidate_lead_budget": [0.01] * 7,
            "command_hz": 30.0,
            "stall_frames": 90,
            "max_movement_control_frames": 1200,
            "settle_frames": 60,
            "required_home_frames": 5,
            "minimum_progress_rad": 0.0002,
        },
    }


def _write_profile(tmp_path: Path, document: dict | None = None) -> Path:
    path = tmp_path / "recording_recovery.local.yaml"
    path.write_text(yaml.safe_dump(document or _document()), encoding="utf-8")
    return path


def test_execute_refuses_before_reading_profile_or_touching_hardware(monkeypatch, capsys):
    monkeypatch.setattr(
        recovery,
        "load_profile",
        lambda *_args, **_kwargs: pytest.fail("--execute must refuse before profile loading"),
    )

    result = recovery.main(
        [
            "--profile",
            "/definitely/missing.local.yaml",
            "--execute",
            "--operator-authorized-staged-recovery",
            "--independent-estop-verified",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["execution_ready"] is False
    assert payload["hardware_io_performed"] is False
    assert "pre-ACTIVE" in payload["reason"]


def test_check_is_offline_and_reports_commissioning_blockers(tmp_path, capsys):
    path = _write_profile(tmp_path)

    assert recovery.main(["--profile", str(path), "--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_complete"] is True
    assert payload["commissioning_complete"] is False
    assert payload["execution_ready"] is False
    assert payload["hardware_io_performed"] is False
    assert any("home_validated" in blocker for blocker in payload["blockers"])


def test_plan_emits_audited_j4_j2_j3_remaining_waypoints(tmp_path, capsys):
    path = _write_profile(tmp_path)
    start = ["-0.30", "-0.40", "0.10", "-0.20", "0.30", "-0.40", "0.35"]

    assert recovery.main(
        ["--profile", str(path), "--plan", "--start", *start]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == [
        "j4_clearance",
        "j2_home",
        "j3_home",
        "remaining_ee_home",
    ]
    assert payload["audit"]["safe"] is True
    assert payload["required_frames"] == len(payload["waypoints"])
    assert payload["hardware_io_performed"] is False


def test_profile_digest_can_be_bound_and_mismatch_fails_closed(tmp_path, capsys):
    path = _write_profile(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert recovery.main(
        [
            "--profile",
            str(path),
            "--check",
            "--expected-profile-sha256",
            digest,
        ]
    ) == 0
    capsys.readouterr()

    assert recovery.main(
        [
            "--profile",
            str(path),
            "--check",
            "--expected-profile-sha256",
            "0" * 64,
        ]
    ) == 2
    payload = json.loads(capsys.readouterr().err)
    assert "SHA-256 mismatch" in payload["error"]
    assert payload["hardware_io_performed"] is False


def test_checked_in_example_contains_no_candidate_home_and_fails_closed(capsys):
    path = Path("configs/firefly_y6_recording_recovery.example.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["planning"]["home"] is None
    assert all(value is False for value in raw["commissioning"].values())
    assert recovery.main(["--profile", str(path), "--check"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_complete"] is False
    assert payload["execution_ready"] is False
    assert "missing planning.home" in payload["blockers"]


def test_plan_requires_start_and_execution_assertions_cannot_be_used_for_check():
    with pytest.raises(SystemExit):
        recovery.parse_args(["--profile", "x.yaml", "--plan"])
    with pytest.raises(SystemExit):
        recovery.parse_args(
            [
                "--profile",
                "x.yaml",
                "--check",
                "--operator-authorized-staged-recovery",
            ]
        )


def test_cli_has_no_hardware_follower_or_transport_import():
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    assert "import HexZenohFollower" not in source
    assert "import zenoh" not in source
    assert "import can" not in source
