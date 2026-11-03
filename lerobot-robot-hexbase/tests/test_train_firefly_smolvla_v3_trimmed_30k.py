from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import train_firefly_smolvla_v3_trimmed_30k as launcher


def test_build_command_pins_trimmed_dataset_and_formal_training_contract():
    command = launcher.build_command()

    assert f"--dataset-root={launcher.DATASET_ROOT}" in command
    assert f"--repo-id={launcher.REPO_ID}" in command
    assert f"--controller-contract={launcher.CONTROLLER_CONTRACT}" in command
    assert f"--smolvla-base={launcher.SMOLVLA_BASE}" in command
    assert f"--config_path={launcher.CONFIG}" in command
    assert "--steps=30000" in command
    assert "--policy.freeze_vision_encoder=true" in command
    assert "--policy.train_expert_only=true" in command
    assert "--policy.train_state_proj=true" in command
    assert launcher.REPO_ID == "local/firefly_y6_manual_validated_v3_trimmed"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_info(
    root: Path,
    *,
    total_episodes: int,
    train_split: str,
    total_frames: int = 24_000,
) -> None:
    info = root / "meta/info.json"
    info.parent.mkdir(parents=True)
    info.write_text(
        json.dumps(
            {
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "splits": {"train": train_split},
            }
        ),
        encoding="utf-8",
    )


def _manifest(root: Path, *, total_frames: int = 24_000) -> dict:
    document = {
        "schema_version": 1,
        "kind": "firefly_action_onset_trim",
        "source": {
            "repo_id": "local/firefly_y6_manual_validated_v2",
            "episode_count": 20,
        },
        "destination": {
            "repo_id": launcher.REPO_ID,
            "root": str(root),
            "episode_count": 20,
            "total_frames": total_frames,
        },
        "algorithm": {
            "threshold_rad": 0.005,
            "pre_roll_frames": 2,
            "validation_window_frames": 10,
        },
        "summary": {
            "episode_count": 20,
            "all_first10_has_motion": True,
            "all_first_action_state_within_contract": True,
        },
        "episodes": [
            {
                "episode_index": index,
                "first_10_has_motion": True,
                "first_action_state_within_contract": True,
            }
            for index in range(20)
        ],
    }
    document["document_sha256"] = _digest(document)
    return document


def _resign(document: dict) -> None:
    unsigned = {
        key: value
        for key, value in document.items()
        if key != "document_sha256"
    }
    document["document_sha256"] = _digest(unsigned)


def _write_manifest(root: Path, document: dict | None = None) -> dict:
    if document is None:
        document = _manifest(root)
    path = root / "meta" / launcher.TRIM_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def _write_valid_gate(root: Path, *, total_frames: int = 24_000) -> None:
    _write_info(
        root,
        total_episodes=20,
        train_split="0:20",
        total_frames=total_frames,
    )
    _write_manifest(root, _manifest(root, total_frames=total_frames))


@pytest.mark.parametrize("total_frames", [24_000, 30_000])
def test_validate_dataset_metadata_accepts_all_20_episodes(tmp_path, total_frames):
    _write_valid_gate(tmp_path, total_frames=total_frames)

    launcher.validate_dataset_metadata(tmp_path)


@pytest.mark.parametrize(
    ("total_episodes", "train_split", "message"),
    [(19, "0:19", "exactly 20"), (20, "0:19", "train split")],
)
def test_validate_dataset_metadata_rejects_incomplete_dataset(
    tmp_path, total_episodes, train_split, message
):
    _write_info(
        tmp_path, total_episodes=total_episodes, train_split=train_split
    )

    with pytest.raises(RuntimeError, match=message):
        launcher.validate_dataset_metadata(tmp_path)


@pytest.mark.parametrize("total_frames", [0, 30_001, True, 1.5])
def test_validate_dataset_metadata_rejects_frame_counts_without_full_epoch(
    tmp_path, total_frames
):
    _write_info(
        tmp_path,
        total_episodes=20,
        train_split="0:20",
        total_frames=total_frames,
    )

    with pytest.raises(RuntimeError, match="total_frames"):
        launcher.validate_dataset_metadata(tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "message"),
    [
        (None, "schema_version", 2, "schema_version"),
        (None, "kind", "wrong", "kind"),
        ("source", "repo_id", "local/wrong", "source repo_id"),
        ("source", "episode_count", 19, "source episode count"),
        ("destination", "repo_id", "local/wrong", "destination repo_id"),
        ("destination", "root", "/wrong", "destination root"),
        ("destination", "episode_count", 19, "destination episode count"),
        ("algorithm", "threshold_rad", 0.006, "threshold"),
        ("algorithm", "pre_roll_frames", 3, "pre-roll"),
        ("algorithm", "validation_window_frames", 9, "validation window"),
        ("summary", "episode_count", 19, "summary episode count"),
        ("summary", "all_first10_has_motion", False, "first-10 motion"),
        (
            "summary",
            "all_first_action_state_within_contract",
            False,
            "first action-state",
        ),
    ],
)
def test_validate_dataset_metadata_rejects_wrong_manifest_contract(
    tmp_path, section, field, bad_value, message
):
    _write_info(tmp_path, total_episodes=20, train_split="0:20")
    document = _manifest(tmp_path)
    target = document if section is None else document[section]
    target[field] = bad_value
    _resign(document)
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match=message):
        launcher.validate_dataset_metadata(tmp_path)


def test_validate_dataset_metadata_rejects_wrong_episode_list(tmp_path):
    _write_info(tmp_path, total_episodes=20, train_split="0:20")
    document = _manifest(tmp_path)
    document["episodes"].pop()
    _resign(document)
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match="exactly 20 entries"):
        launcher.validate_dataset_metadata(tmp_path)


def test_validate_dataset_metadata_rejects_noncontiguous_episode(tmp_path):
    _write_info(tmp_path, total_episodes=20, train_split="0:20")
    document = _manifest(tmp_path)
    document["episodes"][7]["episode_index"] = 8
    _resign(document)
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match="not contiguous"):
        launcher.validate_dataset_metadata(tmp_path)


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("first_10_has_motion", "episode 7 lacks first-10 motion"),
        (
            "first_action_state_within_contract",
            "episode 7 first action-state",
        ),
    ],
)
def test_validate_dataset_metadata_rejects_failed_episode_flag(
    tmp_path, flag, message
):
    _write_info(tmp_path, total_episodes=20, train_split="0:20")
    document = _manifest(tmp_path)
    document["episodes"][7][flag] = False
    _resign(document)
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match=message):
        launcher.validate_dataset_metadata(tmp_path)


def test_validate_dataset_metadata_rejects_frame_or_digest_mismatch(tmp_path):
    _write_info(tmp_path, total_episodes=20, train_split="0:20")
    document = _manifest(tmp_path, total_frames=23_999)
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match="destination total_frames"):
        launcher.validate_dataset_metadata(tmp_path)

    document = _manifest(tmp_path)
    document["document_sha256"] = "0" * 64
    _write_manifest(tmp_path, document)

    with pytest.raises(RuntimeError, match="document digest"):
        launcher.validate_dataset_metadata(tmp_path)


def test_dry_run_does_not_require_materialized_dataset(monkeypatch, capsys):
    def unexpected_validation():
        raise AssertionError("dry-run must not touch the dataset")

    monkeypatch.setattr(launcher, "validate_dataset_metadata", unexpected_validation)

    assert launcher.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert str(launcher.DATASET_ROOT) in output
    assert "--steps=30000" in output
