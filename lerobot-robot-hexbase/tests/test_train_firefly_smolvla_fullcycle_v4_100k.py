from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import train_firefly_smolvla_fullcycle_v4_100k as launcher


def _write_info(root: Path, *, total_frames: int, episodes: int = 50) -> None:
    path = root / "meta/info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fps": 30,
                "total_tasks": 1,
                "total_episodes": episodes,
                "total_frames": total_frames,
                "splits": {"train": f"0:{episodes}"},
            }
        ),
        encoding="utf-8",
    )


def _episode_rows(*, length: int = 1_350) -> list[dict]:
    return [
        {
            "episode_index": index,
            "tasks": [launcher.EXPECTED_TASK],
            "length": length,
        }
        for index in range(launcher.EXPECTED_EPISODES)
    ]


def _write_training_config(path: Path, dataset_root: Path) -> None:
    raw = yaml.safe_load(launcher.CONFIG.read_text(encoding="utf-8"))
    raw["dataset"]["root"] = str(dataset_root)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_checked_in_training_config_matches_pinned_baseline():
    launcher.validate_training_config()


def test_build_command_pins_policy_horizon_optimizer_and_schedule():
    command = launcher.build_command()

    expected = {
        f"--dataset-root={launcher.DATASET_ROOT}",
        f"--repo-id={launcher.REPO_ID}",
        f"--expected-task={launcher.EXPECTED_TASK}",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.optimizer_lr=0.0001",
        "--policy.scheduler_warmup_steps=2000",
        "--policy.scheduler_decay_steps=100000",
        "--steps=100000",
        "--save_freq=10000",
    }
    assert expected.issubset(command)


def test_dataset_gate_accepts_fifty_45_second_episodes(tmp_path):
    rows = _episode_rows(length=45 * launcher.FPS)
    _write_info(tmp_path, total_frames=sum(row["length"] for row in rows))

    launcher.validate_dataset_metadata(
        tmp_path,
        episode_loader=lambda _root: rows,
    )


@pytest.mark.parametrize(
    ("length", "message"),
    [
        (34 * launcher.FPS, "35-120s"),
        (121 * launcher.FPS, "35-120s"),
    ],
)
def test_dataset_gate_rejects_partial_or_overlong_episode(tmp_path, length, message):
    rows = _episode_rows()
    rows[7]["length"] = length
    _write_info(tmp_path, total_frames=sum(row["length"] for row in rows))

    with pytest.raises(RuntimeError, match=message):
        launcher.validate_dataset_metadata(
            tmp_path,
            episode_loader=lambda _root: rows,
        )


def test_dataset_gate_rejects_task_text_drift(tmp_path):
    rows = _episode_rows()
    rows[12]["tasks"] = ["Pick and place the block."]
    _write_info(tmp_path, total_frames=sum(row["length"] for row in rows))

    with pytest.raises(RuntimeError, match="task must be exactly"):
        launcher.validate_dataset_metadata(
            tmp_path,
            episode_loader=lambda _root: rows,
        )


def test_dataset_gate_rejects_wrong_count_before_episode_loading(tmp_path):
    _write_info(tmp_path, total_frames=1_000, episodes=49)

    with pytest.raises(RuntimeError, match="exactly 50"):
        launcher.validate_dataset_metadata(
            tmp_path,
            episode_loader=lambda _root: pytest.fail("episode loader must not run"),
        )


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("steps",), 30_000, "steps"),
        (("save_freq",), 5_000, "save_freq"),
        (("dataset", "image_transforms", "enable"), True, "image_transforms"),
    ],
)
def test_training_config_drift_is_rejected(tmp_path, path, bad_value, message):
    config = tmp_path / "train.yaml"
    _write_training_config(config, tmp_path / "dataset")
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        launcher.validate_training_config(config, dataset_root=tmp_path / "dataset")


def test_dry_run_does_not_touch_dataset_or_training_config(monkeypatch, capsys):
    monkeypatch.setattr(
        launcher,
        "validate_dataset_metadata",
        lambda: pytest.fail("dry-run must not inspect dataset"),
    )
    monkeypatch.setattr(
        launcher,
        "validate_training_config",
        lambda: pytest.fail("dry-run must not inspect config"),
    )

    assert launcher.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert str(launcher.DATASET_ROOT) in output
    assert "--steps=100000" in output
