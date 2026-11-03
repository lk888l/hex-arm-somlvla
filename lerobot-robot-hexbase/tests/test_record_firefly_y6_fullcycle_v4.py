from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import record_firefly_y6_fullcycle_v4 as recording


def _config(target: recording.RecordingTarget) -> dict:
    camera = {
        "type": "intelrealsense",
        "serial_number_or_name": "123456",
        "fps": 30,
        "width": 640,
        "height": 480,
        "use_rgb": True,
        "use_depth": False,
    }
    wrist = dict(camera)
    wrist["serial_number_or_name"] = "654321"
    wrist["rotation"] = 180
    return {
        "robot": {
            "type": "hex_zenoh_observer",
            "cameras": {"scene": camera, "wrist": wrist},
        },
        "teleop": {
            "type": "hex_zenoh_command_tap",
            "record_hz": 30.0,
            "arm_step_limit_rad": 0.0435,
            "ee_step_limit_rad": 0.0165,
        },
        "dataset": {
            "repo_id": target.repo_id,
            "root": str(target.dataset_root),
            "single_task": target.expected_task,
            "fps": 30,
            "episode_time_s": 600,
            "reset_time_s": 600,
            "num_episodes": 1,
            "video": True,
            "push_to_hub": False,
            "streaming_encoding": False,
            "num_image_writer_processes": 0,
            "num_image_writer_threads_per_camera": 4,
        },
        "display_data": False,
        "play_sounds": False,
        "resume": False,
    }


def _target(tmp_path: Path, *, materialized: bool = False) -> recording.RecordingTarget:
    target = recording.RecordingTarget(
        dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "record.yaml",
        controller_contract=tmp_path / "controller.json",
    )
    if materialized:
        target.dataset_root.mkdir()
    target.config_path.write_text(yaml.safe_dump(_config(target)), encoding="utf-8")
    return target


def _safe_report(episodes: int) -> dict:
    return {
        "checks": {
            "metadata_consistency": {
                "total_episodes": {"metadata": episodes, "actual": episodes}
            }
        },
        "summary": {"safe_for_training": True, "errors": []},
    }


def _capture_exec():
    calls: list[tuple[str, list[str]]] = []

    def execute(path: str, argv: list[str]) -> None:
        calls.append((path, list(argv)))

    return calls, execute


def test_checked_in_local_config_matches_fullcycle_contract():
    recording.validate_record_config(recording.RecordingTarget())


def test_new_dataset_records_exactly_one_episode(tmp_path):
    target = _target(tmp_path)
    calls, execute = _capture_exec()

    result = recording.run_recording(
        target,
        audit=lambda *_args, **_kwargs: pytest.fail("missing root must not be audited"),
        contract_loader=lambda _path: object(),
        protocol_validator=lambda *_args, **_kwargs: {"warnings": []},
        exec_function=execute,
        record_executable=tmp_path / "bin/lerobot-record",
    )

    assert result == 0
    assert calls == [
        (
            str(tmp_path / "bin/lerobot-record"),
            [
                str(tmp_path / "bin/lerobot-record"),
                f"--config_path={target.config_path}",
                "--resume=false",
                "--dataset.num_episodes=1",
            ],
        )
    ]


def test_existing_dataset_resumes_one_episode_not_all_remaining(tmp_path):
    target = _target(tmp_path, materialized=True)
    calls, execute = _capture_exec()

    result = recording.run_recording(
        target,
        audit=lambda *_args, **_kwargs: _safe_report(17),
        contract_loader=lambda _path: object(),
        protocol_validator=lambda *_args, **_kwargs: {"warnings": []},
        exec_function=execute,
        record_executable=tmp_path / "bin/lerobot-record",
    )

    assert result == 0
    assert calls[0][1][-2:] == ["--resume=true", "--dataset.num_episodes=1"]


def test_fifty_audited_episodes_exit_without_hardware_exec(tmp_path):
    target = _target(tmp_path, materialized=True)
    calls, execute = _capture_exec()

    result = recording.run_recording(
        target,
        audit=lambda *_args, **_kwargs: _safe_report(50),
        contract_loader=lambda _path: object(),
        protocol_validator=lambda *_args, **_kwargs: {"warnings": []},
        exec_function=execute,
        record_executable=tmp_path / "bin/lerobot-record",
    )

    assert result == 0
    assert calls == []


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("dataset", "episode_time_s", 60),
        ("dataset", "reset_time_s", 15),
        ("dataset", "num_episodes", 50),
        ("dataset", "video", False),
        ("dataset", "streaming_encoding", True),
        ("root", "display_data", True),
    ],
)
def test_long_manual_recording_contract_drift_is_rejected_before_contract(
    tmp_path, section, field, bad_value
):
    target = _target(tmp_path)
    raw = yaml.safe_load(target.config_path.read_text(encoding="utf-8"))
    if section == "root":
        raw[field] = bad_value
    else:
        raw[section][field] = bad_value
    target.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    calls, execute = _capture_exec()

    with pytest.raises(recording.RecordingPreflightError, match="record config"):
        recording.run_recording(
            target,
            audit=lambda *_args, **_kwargs: pytest.fail("audit must not run"),
            contract_loader=lambda _path: pytest.fail("contract must not load"),
            exec_function=execute,
            record_executable=tmp_path / "bin/lerobot-record",
        )

    assert calls == []


def test_formal_cli_has_no_target_or_batch_override():
    with pytest.raises(SystemExit) as raised:
        recording.parse_args(["--target-total", "100"])

    assert raised.value.code == 2
