from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import lerobot_robot_hexbase.firefly_training_attestation as training_attestation
from lerobot_robot_hexbase.firefly_training_attestation import (
    ATTESTATION_FILENAME,
    TrainingAttestationError,
    TrainingInputChangedError,
    capture_training_inputs,
    run_attested_training,
    verify_training_attestation,
)


TASK = "Pick up the red block and place it in the blue tray."
REPO_ID = "local/firefly_test"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    files = {
        "meta/info.json": b'{"fps":30}',
        "meta/tasks.parquet": b"tasks",
        "data/chunk-000/file-000.parquet": b"actions-and-state",
        "videos/observation.images.scene/chunk-000/file-000.mp4": b"scene-video",
        "videos/observation.images.wrist/chunk-000/file-000.mp4": b"wrist-video",
    }
    for relative, content in files.items():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    contract = tmp_path / "controller_contract.json"
    contract.write_text('{"source":"test"}', encoding="utf-8")
    return dataset, contract


def _safe_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        training_attestation, "load_controller_contract", lambda _path: sentinel
    )

    def audit(dataset_root, repo_id, *, controller_contract, expected_task):
        assert controller_contract is sentinel
        assert repo_id == REPO_ID
        assert expected_task == TASK
        return {
            "schema_version": 1,
            "audit": "firefly_lerobot_dataset",
            "dataset": {"root": str(dataset_root), "repo_id": repo_id},
            "summary": {
                "safe_for_training": True,
                "verdict": "pass",
                "errors": [],
                "warnings": [],
            },
        }

    monkeypatch.setattr(training_attestation, "audit_firefly_dataset", audit)


@pytest.fixture
def pinned_base(tmp_path, monkeypatch):
    base = tmp_path / "pinned-base"
    base.mkdir()
    vlm = tmp_path / "pinned-vlm"
    vlm.mkdir()
    canonical_base = base.resolve(strict=True)
    canonical_vlm = vlm.resolve(strict=True)
    provenance = {
        "smolvlm_bundle": {"canonical_path": str(canonical_vlm)}
    }

    def validate(path):
        assert Path(path).resolve(strict=True) == canonical_base
        return canonical_base, provenance

    monkeypatch.setattr(
        training_attestation, "validate_materialized_smolvla_base", validate
    )
    return canonical_base, canonical_vlm


def _argument(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    return next(argument[len(prefix) :] for argument in command if argument.startswith(prefix))


def _write_checkpoint(
    output_dir: Path,
    step: str,
    *,
    token: str,
    dataset_root: Path,
    smolvla_base: Path,
    smolvlm_bundle: Path,
    notes: str | None = None,
) -> Path:
    model = output_dir / "checkpoints" / step / "pretrained_model"
    model.mkdir(parents=True)
    train_config = {
        "dataset": {
            "repo_id": REPO_ID,
            "root": str(dataset_root),
            "episodes": None,
            "revision": None,
            "streaming": False,
            "eval_split": 0.0,
        },
        "policy": {
            "type": "smolvla",
            "pretrained_path": str(smolvla_base),
            "load_vlm_weights": False,
            "vlm_model_name": str(smolvlm_bundle),
            "use_peft": False,
            "push_to_hub": False,
        },
        "wandb": {"enable": False, "notes": token if notes is None else notes},
        "job": {"target": "local"},
        "peft": None,
        "output_dir": str(output_dir),
        "resume": False,
        "save_checkpoint": True,
        "save_checkpoint_to_hub": False,
        "rename_map": {},
    }
    (model / "train_config.json").write_text(
        json.dumps(train_config), encoding="utf-8"
    )
    return model


def test_capture_hashes_meta_data_and_videos(monkeypatch, tmp_path):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)

    first = capture_training_inputs(
        dataset,
        REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
    )
    second = capture_training_inputs(
        dataset,
        REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
    )

    paths = {entry["path"] for entry in first.payload["dataset"]["files"]}
    assert "meta/info.json" in paths
    assert "data/chunk-000/file-000.parquet" in paths
    assert "videos/observation.images.scene/chunk-000/file-000.mp4" in paths
    assert first.digest == second.digest
    assert first.token == second.token


def test_successful_training_forces_contract_and_attests_every_checkpoint(
    monkeypatch, tmp_path, pinned_base
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    output = tmp_path / "formal-output"
    seen_command: list[str] = []

    def runner(command, *, check):
        assert check is False
        seen_command.extend(command)
        token = _argument(command, "wandb.notes")
        configured_output = Path(_argument(command, "output_dir"))
        configured_dataset = Path(_argument(command, "dataset.root"))
        _write_checkpoint(
            configured_output,
            "005000",
            token=token,
            dataset_root=configured_dataset,
            smolvla_base=pinned_base[0],
            smolvlm_bundle=pinned_base[1],
        )
        _write_checkpoint(
            configured_output,
            "010000",
            token=token,
            dataset_root=configured_dataset,
            smolvla_base=pinned_base[0],
            smolvlm_bundle=pinned_base[1],
        )
        (configured_output / "checkpoints" / "last").symlink_to("010000")
        return subprocess.CompletedProcess(command, 0)

    written = run_attested_training(
        dataset_root=dataset,
        repo_id=REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
        output_dir=output,
        smolvla_base=pinned_base[0],
        training_args=["--config_path=configs/train.yaml", "--steps=10000"],
        runner=runner,
    )

    assert seen_command[:3] == [
        str(Path(training_attestation.sys.executable)),
        "-m",
        "lerobot.scripts.lerobot_train",
    ]
    assert _argument(seen_command, "dataset.episodes") == "null"
    assert _argument(seen_command, "dataset.streaming") == "false"
    assert _argument(seen_command, "dataset.eval_split") == "0.0"
    assert _argument(seen_command, "policy.path") == str(pinned_base[0])
    assert _argument(seen_command, "policy.input_features") == "null"
    assert _argument(seen_command, "policy.output_features") == "null"
    assert _argument(seen_command, "policy.load_vlm_weights") == "false"
    assert _argument(seen_command, "policy.vlm_model_name") == str(pinned_base[1])
    assert _argument(seen_command, "policy.use_peft") == "false"
    assert _argument(seen_command, "policy.push_to_hub") == "false"
    assert _argument(seen_command, "peft") == "null"
    assert len(written) == 2
    current = capture_training_inputs(
        dataset,
        REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
    )
    for path in written:
        assert path.name == ATTESTATION_FILENAME
        document = verify_training_attestation(path.parent, current)
        assert document["verification"]["inputs_unchanged_after_training"] is True


def test_dataset_change_during_training_leaves_checkpoint_unattested(
    monkeypatch, tmp_path, pinned_base
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    output = tmp_path / "changed-output"

    def runner(command, *, check):
        token = _argument(command, "wandb.notes")
        _write_checkpoint(
            output,
            "005000",
            token=token,
            dataset_root=dataset,
            smolvla_base=pinned_base[0],
            smolvlm_bundle=pinned_base[1],
        )
        video = dataset / "videos/observation.images.scene/chunk-000/file-000.mp4"
        video.write_bytes(b"changed-during-training")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(TrainingInputChangedError, match="changed during training"):
        run_attested_training(
            dataset_root=dataset,
            repo_id=REPO_ID,
            controller_contract=contract,
            expected_task=TASK,
            output_dir=output,
            smolvla_base=pinned_base[0],
            training_args=["--config_path=configs/train.yaml"],
            runner=runner,
        )

    assert not list(output.rglob(ATTESTATION_FILENAME))


def test_failed_training_leaves_partial_checkpoint_unattested(
    monkeypatch, tmp_path, pinned_base
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    output = tmp_path / "failed-output"

    def runner(command, *, check):
        token = _argument(command, "wandb.notes")
        _write_checkpoint(
            output,
            "005000",
            token=token,
            dataset_root=dataset,
            smolvla_base=pinned_base[0],
            smolvlm_bundle=pinned_base[1],
        )
        return subprocess.CompletedProcess(command, 7)

    with pytest.raises(TrainingAttestationError, match="status 7"):
        run_attested_training(
            dataset_root=dataset,
            repo_id=REPO_ID,
            controller_contract=contract,
            expected_task=TASK,
            output_dir=output,
            smolvla_base=pinned_base[0],
            training_args=["--config_path=configs/train.yaml"],
            runner=runner,
        )

    assert not list(output.rglob(ATTESTATION_FILENAME))


def test_wrong_checkpoint_token_is_rejected_without_attestation(
    monkeypatch, tmp_path, pinned_base
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    output = tmp_path / "wrong-token-output"

    def runner(command, *, check):
        token = _argument(command, "wandb.notes")
        _write_checkpoint(
            output,
            "005000",
            token=token,
            dataset_root=dataset,
            smolvla_base=pinned_base[0],
            smolvlm_bundle=pinned_base[1],
            notes="untrusted-manual-note",
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(TrainingAttestationError, match="wandb.notes"):
        run_attested_training(
            dataset_root=dataset,
            repo_id=REPO_ID,
            controller_contract=contract,
            expected_task=TASK,
            output_dir=output,
            smolvla_base=pinned_base[0],
            training_args=["--config_path=configs/train.yaml"],
            runner=runner,
        )

    assert not list(output.rglob(ATTESTATION_FILENAME))


def test_missing_attestation_and_reserved_override_fail_closed(
    monkeypatch, tmp_path, pinned_base
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    capture = capture_training_inputs(
        dataset,
        REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
    )
    model = tmp_path / "standalone-model"
    model.mkdir()

    with pytest.raises(TrainingAttestationError, match="no formal Firefly training attestation"):
        changed = dataset / "videos/observation.images.wrist/chunk-000/file-000.mp4"
        changed.write_bytes(b"different-current-video")
        current = capture_training_inputs(
            dataset,
            REPO_ID,
            controller_contract=contract,
            expected_task=TASK,
        )
        verify_training_attestation(model, current)

    with pytest.raises(TrainingAttestationError, match="controlled by"):
        training_attestation.build_training_command(
            capture,
            tmp_path / "new-output",
            ["--config_path=configs/train.yaml", "--wandb.notes=manual"],
            smolvla_base=pinned_base[0],
        )


@pytest.mark.parametrize(
    "override",
    [
        "--policy.path=lerobot/smolvla_base",
        "--policy.pretrained_path=/tmp/untrusted",
        "--policy.input_features=null",
        "--policy.output_features",
        "--policy.load_vlm_weights=true",
        "--policy.vlm_model_name=HuggingFaceTB/unpinned",
        "--policy.use_peft=true",
        "--peft.type=lora",
    ],
)
def test_policy_source_and_schema_overrides_are_rejected(
    monkeypatch, tmp_path, pinned_base, override
):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    capture = capture_training_inputs(
        dataset,
        REPO_ID,
        controller_contract=contract,
        expected_task=TASK,
    )

    with pytest.raises(TrainingAttestationError, match="controlled by"):
        training_attestation.build_training_command(
            capture,
            tmp_path / "new-output",
            ["--config_path=configs/train.yaml", override],
            smolvla_base=pinned_base[0],
        )


def test_dataset_symlink_is_rejected(monkeypatch, tmp_path):
    _safe_audit(monkeypatch)
    dataset, contract = _write_inputs(tmp_path)
    target = dataset / "real-extra.mp4"
    target.write_bytes(b"extra")
    (dataset / "videos" / "linked.mp4").symlink_to(target)

    with pytest.raises(TrainingAttestationError, match="symbolic link"):
        capture_training_inputs(
            dataset,
            REPO_ID,
            controller_contract=contract,
            expected_task=TASK,
        )
