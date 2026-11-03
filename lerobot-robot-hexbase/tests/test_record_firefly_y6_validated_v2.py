from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import record_firefly_y6_validated_v2 as recording


def _target(tmp_path: Path, *, episodes: int | None = None) -> recording.RecordingTarget:
    dataset_root = tmp_path / "dataset"
    if episodes is not None:
        dataset_root.mkdir()
    target = recording.RecordingTarget(
        dataset_root=dataset_root,
        repo_id=recording.REPO_ID,
        config_path=tmp_path / "record.yaml",
        controller_contract=tmp_path / "controller_contract_arm0435_ee0165.json",
        expected_task=recording.EXPECTED_TASK,
        target_total=20,
    )
    target.config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "repo_id": target.repo_id,
                    "root": str(target.dataset_root),
                    "single_task": target.expected_task,
                    "fps": 30,
                    "num_episodes": 20,
                },
                "teleop": {
                    "record_hz": 30.0,
                    "arm_step_limit_rad": 0.0435,
                    "ee_step_limit_rad": 0.0165,
                },
                "resume": False,
            }
        ),
        encoding="utf-8",
    )
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


def test_formal_cli_rejects_target_total_override():
    with pytest.raises(SystemExit) as raised:
        recording.parse_args(["--target-total", "100"])

    assert raised.value.code == 2


def test_new_dataset_starts_fresh_for_full_target_without_dataset_audit(tmp_path):
    target = _target(tmp_path)
    calls, execute = _capture_exec()
    loaded: list[Path] = []

    def unexpected_audit(*_args, **_kwargs):
        raise AssertionError("a missing dataset must not be audited")

    def load_contract(path: Path):
        loaded.append(path)
        return object()

    result = recording.run_recording(
        target,
        audit=unexpected_audit,
        contract_loader=load_contract,
        exec_function=execute,
        record_executable=tmp_path / "bin" / "lerobot-record",
    )

    assert result == 0
    assert loaded == [target.controller_contract]
    assert calls == [
        (
            str(tmp_path / "bin" / "lerobot-record"),
            [
                str(tmp_path / "bin" / "lerobot-record"),
                f"--config_path={target.config_path}",
                "--resume=false",
                "--dataset.num_episodes=20",
            ],
        )
    ]


def test_new_dataset_with_invalid_contract_is_refused_without_exec(tmp_path):
    target = _target(tmp_path)
    calls, execute = _capture_exec()

    def invalid_contract(_path: Path):
        raise recording.DatasetAuditInputError("missing or invalid contract")

    with pytest.raises(
        recording.RecordingPreflightError, match="controller contract validation"
    ):
        recording.run_recording(
            target,
            audit=lambda *_args, **_kwargs: pytest.fail("audit must not run"),
            contract_loader=invalid_contract,
            exec_function=execute,
            record_executable=tmp_path / "bin" / "lerobot-record",
        )

    assert calls == []


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("dataset", "root", "/tmp/not-the-audited-firefly-root"),
        ("dataset", "single_task", "A different task."),
        ("teleop", "arm_step_limit_rad", 0.0436),
    ],
    ids=("wrong-root", "wrong-task", "wrong-step"),
)
def test_mismatched_record_config_is_refused_before_contract_or_exec(
    tmp_path, section, field, replacement
):
    target = _target(tmp_path)
    raw = yaml.safe_load(target.config_path.read_text(encoding="utf-8"))
    raw[section][field] = replacement
    target.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    calls, execute = _capture_exec()

    with pytest.raises(recording.RecordingPreflightError, match="record config"):
        recording.run_recording(
            target,
            audit=lambda *_args, **_kwargs: pytest.fail("audit must not run"),
            contract_loader=lambda _path: pytest.fail("contract loader must not run"),
            exec_function=execute,
            record_executable=tmp_path / "bin" / "lerobot-record",
        )

    assert calls == []


def test_one_safe_episode_resumes_only_the_remaining_nineteen(tmp_path):
    target = _target(tmp_path, episodes=1)
    contract = object()
    audited: list[tuple] = []
    calls, execute = _capture_exec()

    def load_contract(path: Path):
        assert path == target.controller_contract
        return contract

    def audit(root, repo_id, *, controller_contract, expected_task):
        audited.append((root, repo_id, controller_contract, expected_task))
        return _safe_report(1)

    result = recording.run_recording(
        target,
        audit=audit,
        contract_loader=load_contract,
        exec_function=execute,
        record_executable=tmp_path / "bin" / "lerobot-record",
    )

    assert result == 0
    assert audited == [
        (target.dataset_root, target.repo_id, contract, recording.EXPECTED_TASK)
    ]
    assert calls[0][1][-2:] == ["--resume=true", "--dataset.num_episodes=19"]


def test_twenty_safe_episodes_exit_successfully_without_exec(tmp_path):
    target = _target(tmp_path, episodes=20)
    calls, execute = _capture_exec()

    result = recording.run_recording(
        target,
        audit=lambda *_args, **_kwargs: _safe_report(20),
        contract_loader=lambda _path: object(),
        exec_function=execute,
        record_executable=tmp_path / "bin" / "lerobot-record",
    )

    assert result == 0
    assert calls == []


@pytest.mark.parametrize(
    "report",
    [
        {
            "checks": {},
            "summary": {
                "safe_for_training": False,
                "errors": [{"code": "bad_episode", "message": "unsafe data"}],
            },
        },
        {"checks": {}, "summary": {"safe_for_training": True, "errors": []}},
    ],
    ids=("audit-failed", "audit-incomplete"),
)
def test_failed_or_incomplete_audit_refuses_resume(tmp_path, report):
    target = _target(tmp_path, episodes=1)
    calls, execute = _capture_exec()

    with pytest.raises(recording.RecordingPreflightError):
        recording.run_recording(
            target,
            audit=lambda *_args, **_kwargs: report,
            contract_loader=lambda _path: object(),
            exec_function=execute,
            record_executable=tmp_path / "bin" / "lerobot-record",
        )

    assert calls == []
