from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lerobot_robot_hexbase import firefly_arm6_attestation as arm6


def _bytes(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_inventory(root: Path, *, exclude: frozenset[str] = frozenset()):
    files = []
    directories = []
    for tree in ("meta", "data", "videos"):
        for current, names, filenames in os.walk(root / tree):
            names.sort()
            filenames.sort()
            current_path = Path(current)
            directories.append(current_path.relative_to(root).as_posix())
            for filename in filenames:
                path = current_path / filename
                relative = path.relative_to(root).as_posix()
                if relative in exclude:
                    continue
                data = path.read_bytes()
                files.append(
                    {
                        "path": relative,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    files.sort(key=lambda item: item["path"])
    directories.sort()
    return files, directories


def _smoke():
    return {
        "status": "passed",
        "policy_type": "smolvla",
        "input_state_dimension": 6,
        "output_action_dimension": 6,
        "selected_action_shape": [1, 6],
        "postprocessed_action_shape": [1, 6],
        "strict_weights": True,
        "local_files_only": True,
        "processor_roundtrip": True,
        "finite_action": True,
    }


def _checkpoint_files(model: Path):
    return sorted((_file(model / name) for name in arm6.REQUIRED_CHECKPOINT_FILES), key=lambda item: item["path"])


def _recommit(fixture: dict) -> None:
    marker_path = fixture["marker"]
    marker = json.loads(marker_path.read_text())
    for record in marker["children"]:
        directory = record["directory"]
        model = fixture["output"] / "checkpoints" / directory / "pretrained_model"
        child_path = model / arm6.CHILD_ATTESTATION_FILENAME
        child = json.loads(child_path.read_text())
        files = _checkpoint_files(model)
        child["checkpoint"]["files"] = files
        child["checkpoint"]["files_sha256"] = _digest(files)
        _write_json(child_path, child)
        child_data = child_path.read_bytes()
        record["attestation_size"] = len(child_data)
        record["attestation_sha256"] = hashlib.sha256(child_data).hexdigest()
        record["checkpoint_files_sha256"] = _digest(files)
        record["model_sha256"] = next(
            item["sha256"] for item in files if item["path"] == "model.safetensors"
        )
    marker["children_sha256"] = _digest(marker["children"])
    unsigned = dict(marker)
    unsigned.pop("document_sha256", None)
    marker["document_sha256"] = _digest(unsigned)
    _write_json(marker_path, marker)
    arm6.EXPECTED_RUN_COMMIT_SHA256 = hashlib.sha256(marker_path.read_bytes()).hexdigest()


@pytest.fixture
def fixed_run(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    producer = scripts / "train_firefly_smolvla_arm_only_v1_finetune_20k_run2.py"
    materializer = scripts / "materialize_firefly_y6_arm_only_v1.py"
    producer.write_text("# frozen producer fixture\n", encoding="utf-8")
    materializer.write_text("# frozen materializer fixture\n", encoding="utf-8")

    source = tmp_path / "datasets" / "source"
    destination = tmp_path / "datasets" / "destination"
    relative_files = [
        "videos/observation.images.scene/chunk-000/episode.mp4",
        "videos/observation.images.wrist/chunk-000/episode.mp4",
        "meta/info.json",
        *[f"data/chunk-000/part-{index:03d}.bin" for index in range(194)],
    ]
    assert len(relative_files) == 197
    for relative in relative_files:
        source_path = source / relative
        destination_path = destination / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("payload:" + relative).encode()
        source_path.write_bytes(payload)
        destination_path.write_bytes(payload)

    controller = repository / "outputs/firefly_diagnostics/controller.json"
    controller.parent.mkdir(parents=True)
    _write_json(controller, {"effective_step_limits": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]})
    controller_data = controller.read_bytes()
    source_files, source_directories = _tree_inventory(source)
    destination_files, destination_directories = _tree_inventory(destination)
    assert source_directories == destination_directories
    transform = arm6._expected_transform()
    manifest = {
        "schema_version": 1,
        "kind": arm6.PROJECTION_KIND,
        "projection_version": arm6.PROJECTION_VERSION,
        "source": {
            "repo_id": arm6.SOURCE_DATASET_REPO_ID,
            "root": str(source),
            "episode_count": arm6.EXPECTED_EPISODES,
            "frame_count": arm6.EXPECTED_FRAMES,
            "fps": arm6.EXPECTED_FPS,
            "task": arm6.EXPECTED_TASK,
            "files_manifest_sha256": _digest(source_files),
            "files": source_files,
            "directories_sha256": _digest(source_directories),
            "directories": source_directories,
            "upstream_training_input_digest": arm6.EXPECTED_UPSTREAM_INPUT_DIGEST,
        },
        "destination": {
            "repo_id": arm6.DESTINATION_DATASET_REPO_ID,
            "root": str(destination),
            "episode_count": arm6.EXPECTED_EPISODES,
            "frame_count": arm6.EXPECTED_FRAMES,
            "fps": arm6.EXPECTED_FPS,
            "task": arm6.EXPECTED_TASK,
            "files_manifest_sha256": _digest(destination_files),
            "files": destination_files,
            "directories_sha256": _digest(destination_directories),
            "directories": destination_directories,
        },
        "transform": transform,
        "controller_contract": {
            "path": str(controller),
            "size": len(controller_data),
            "sha256": hashlib.sha256(controller_data).hexdigest(),
            "retained_arm_effective_step_limits_rad": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "removed_jaw_effective_step_limit_rad": 0.07,
        },
        "verification": {
            "exact_first_six_float32_bits": True,
            "one_to_one_file_paths": True,
            "copied_files_byte_identical": True,
            "destination_files_independent_regular_single_link": True,
            "source_unchanged_before_publish": True,
        },
    }
    manifest["document_sha256"] = _digest(manifest)
    manifest_path = destination / arm6.PROJECTION_MANIFEST_RELATIVE
    _write_json(manifest_path, manifest)

    output = tmp_path / "outputs" / "firefly_run2"
    inputs = {
        "schema_version": 1,
        "kind": arm6.CHILD_ATTESTATION_KIND,
        "training_contract": {
            "policy_schema_role": arm6.POLICY_SCHEMA_ROLE,
            "authorized_runtime_role": arm6.AUTHORIZED_RUNTIME_ROLE,
            "observation_limitations": arm6.LIMITATIONS,
            "output_dir": str(output),
            "job_name": output.name,
            "steps": 20_000,
            "save_freq": 10_000,
        },
        "dataset": {
            "repo_id": arm6.DESTINATION_DATASET_REPO_ID,
            "root": str(destination),
            "arm_only_manifest": {
                "path": arm6.PROJECTION_MANIFEST_RELATIVE.as_posix(),
                "size": len(manifest_path.read_bytes()),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "document": manifest,
            },
        },
        "command": ["offline", "train"],
    }
    core = dict(inputs)
    core.pop("command")
    core_digest = _digest(core)
    input_digest = _digest(inputs)
    children = []
    for step, directory in arm6.EXPECTED_CHECKPOINTS:
        model = output / "checkpoints" / directory / "pretrained_model"
        model.mkdir(parents=True)
        _write_json(
            model / "config.json",
            {
                "type": "smolvla",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.scene": {"type": "VISUAL", "shape": [3, 480, 640]},
                    "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            },
        )
        _write_json(model / "train_config.json", {"step": step})
        (model / "model.safetensors").write_bytes(f"model:{step}".encode())
        _write_json(model / "policy_preprocessor.json", {})
        _write_json(model / "policy_postprocessor.json", {})
        (model / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(b"pre" + str(step).encode())
        (model / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"post" + str(step).encode())
        files = _checkpoint_files(model)
        child = {
            "schema_version": 1,
            "kind": arm6.CHILD_ATTESTATION_KIND,
            "policy_schema_role": arm6.POLICY_SCHEMA_ROLE,
            "authorized_runtime_role": arm6.AUTHORIZED_RUNTIME_ROLE,
            "limitations": arm6.LIMITATIONS,
            "token": arm6.CHILD_TOKEN_PREFIX + core_digest,
            "core_input_digest": core_digest,
            "input_digest": input_digest,
            "inputs": inputs,
            "run_commit": {"filename": arm6.RUN_COMMIT_FILENAME, "run_id": input_digest},
            "checkpoint": {"step": step, "files": files, "files_sha256": _digest(files)},
            "verification": {
                "training_returncode": 0,
                "inputs_unchanged_after_training": True,
                "postflight_input_digest": input_digest,
            },
        }
        child_path = model / arm6.CHILD_ATTESTATION_FILENAME
        _write_json(child_path, child)
        training_path = model.parent / "training_state/training_step.json"
        training_path.parent.mkdir()
        training_document = {"step": step, "num_processes": 1, "batch_size": 1}
        _write_json(training_path, training_document)
        child_data = child_path.read_bytes()
        training_data = training_path.read_bytes()
        children.append(
            {
                "step": step,
                "directory": directory,
                "attestation_path": f"checkpoints/{directory}/pretrained_model/{arm6.CHILD_ATTESTATION_FILENAME}",
                "attestation_size": len(child_data),
                "attestation_sha256": hashlib.sha256(child_data).hexdigest(),
                "checkpoint_files_sha256": _digest(files),
                "model_sha256": next(item["sha256"] for item in files if item["path"] == "model.safetensors"),
                "training_step": {
                    "path": "training_state/training_step.json",
                    "size": len(training_data),
                    "sha256": hashlib.sha256(training_data).hexdigest(),
                    "document": training_document,
                },
                "offline_load_smoke": _smoke(),
            }
        )
    (output / "checkpoints/last").symlink_to("020000")
    marker = {
        "schema_version": 1,
        "kind": arm6.RUN_COMMIT_KIND,
        "policy_schema_role": arm6.POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": arm6.AUTHORIZED_RUNTIME_ROLE,
        "limitations": arm6.LIMITATIONS,
        "token": arm6.CHILD_TOKEN_PREFIX + core_digest,
        "core_input_digest": core_digest,
        "input_digest": input_digest,
        "run_id": input_digest,
        "last_target": "020000",
        "children": children,
        "children_sha256": _digest(children),
    }
    marker["document_sha256"] = _digest(marker)
    marker_path = output / arm6.RUN_COMMIT_FILENAME
    _write_json(marker_path, marker)

    monkeypatch.setattr(arm6, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(arm6, "PRODUCER_SCRIPT", producer)
    monkeypatch.setattr(arm6, "MATERIALIZER_SCRIPT", materializer)
    monkeypatch.setattr(arm6, "EXPECTED_PRODUCER_SHA256", hashlib.sha256(producer.read_bytes()).hexdigest())
    monkeypatch.setattr(arm6, "EXPECTED_MATERIALIZER_SHA256", hashlib.sha256(materializer.read_bytes()).hexdigest())
    monkeypatch.setattr(arm6, "OUTPUT_DIR", output)
    monkeypatch.setattr(arm6, "SOURCE_DATASET_ROOT", source)
    monkeypatch.setattr(arm6, "DESTINATION_DATASET_ROOT", destination)
    monkeypatch.setattr(arm6, "CONTROLLER_CONTRACT", controller)
    monkeypatch.setattr(arm6, "EXPECTED_SOURCE_FILES_MANIFEST_SHA256", _digest(source_files))
    monkeypatch.setattr(arm6, "EXPECTED_SOURCE_DIRECTORIES_SHA256", _digest(source_directories))
    monkeypatch.setattr(arm6, "EXPECTED_RUN_COMMIT_SHA256", hashlib.sha256(marker_path.read_bytes()).hexdigest())

    fixture = {
        "repository": repository,
        "source": source,
        "destination": destination,
        "output": output,
        "producer": producer,
        "materializer": materializer,
        "manifest": manifest_path,
        "marker": marker_path,
    }

    def runner(command, **kwargs):
        fixture["calls"] = fixture.get("calls", 0) + 1
        assert command[-1] == "--verify-run"
        assert "lerobot.scripts.lerobot_train" not in command
        assert kwargs["check"] is False
        expected = arm6._capture_local().expected_producer_result
        return subprocess.CompletedProcess(command, 0, _bytes(expected).decode() + "\n", "")

    fixture["runner"] = runner
    return fixture


def test_capture_returns_canonical_arm6_binding_and_explicit_020000(fixed_run):
    capture = arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])
    assert capture.selected_step == 20_000
    assert capture.selected_checkpoint == fixed_run["output"] / "checkpoints/020000/pretrained_model"
    assert fixed_run["calls"] == 1
    binding = capture.source_binding()
    assert set(binding) == {"policy_attestation", "projection_binding"}
    policy = binding["policy_attestation"]
    assert policy["policy_schema_role"] == "arm_only_6d"
    assert policy["authorized_runtime_role"] == "arm6_seeking"
    assert policy["selected_checkpoint"]["directory"] == "020000"
    projection = binding["projection_binding"]
    assert projection["transform"]["removed_feature"]["source_index"] == 6
    assert projection["source"]["upstream_training_input_digest"] == arm6.EXPECTED_UPSTREAM_INPUT_DIGEST
    assert projection["video_byte_identity"]["file_count"] == 2
    roles = [item["role"] for item in capture.controlled_artifacts()]
    assert roles == sorted(
        [
            "run_commit",
            "child_attestation_010000",
            "child_attestation_020000",
            "model_010000",
            "model_020000",
            "projection_manifest",
            "producer_script",
            "materializer_script",
        ]
    )
    policy["selected_checkpoint"]["directory"] = "010000"
    assert capture.policy_attestation()["selected_checkpoint"]["directory"] == "020000"


def test_unfrozen_marker_pin_refuses_before_reading_run2(monkeypatch, tmp_path):
    monkeypatch.setattr(arm6, "EXPECTED_RUN_COMMIT_SHA256", None)
    monkeypatch.setattr(arm6, "OUTPUT_DIR", tmp_path / "does-not-exist")
    with pytest.raises(arm6.FireflyArm6AttestationError, match="has not been frozen"):
        arm6.capture_firefly_arm6_attestation(runner=lambda *_args, **_kwargs: pytest.fail())


def test_rejects_recomputed_marker_self_hash_against_frozen_pin(fixed_run):
    marker = json.loads(fixed_run["marker"].read_text())
    marker["authorized_runtime_role"] = "legacy_action7"
    unsigned = dict(marker)
    unsigned.pop("document_sha256")
    marker["document_sha256"] = _digest(unsigned)
    _write_json(fixed_run["marker"], marker)
    with pytest.raises(arm6.FireflyArm6AttestationError, match="marker file sha256"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_010000_selected_even_if_marker_pin_is_replaced(fixed_run):
    marker = json.loads(fixed_run["marker"].read_text())
    marker["last_target"] = "010000"
    unsigned = dict(marker)
    unsigned.pop("document_sha256")
    marker["document_sha256"] = _digest(unsigned)
    _write_json(fixed_run["marker"], marker)
    arm6.EXPECTED_RUN_COMMIT_SHA256 = hashlib.sha256(fixed_run["marker"].read_bytes()).hexdigest()
    with pytest.raises(arm6.FireflyArm6AttestationError, match="last_target"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_seven_dimensional_smolvla_after_full_self_recommit(fixed_run):
    config = fixed_run["output"] / "checkpoints/020000/pretrained_model/config.json"
    document = json.loads(config.read_text())
    document["input_features"]["observation.state"]["shape"] = [7]
    document["output_features"]["action"]["shape"] = [7]
    _write_json(config, document)
    _recommit(fixed_run)
    with pytest.raises(arm6.FireflyArm6AttestationError, match="state dimension"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_numeric_jaw_policy_input_after_full_self_recommit(fixed_run):
    config = fixed_run["output"] / "checkpoints/020000/pretrained_model/config.json"
    document = json.loads(config.read_text())
    document["input_features"]["observation.jaw"] = {
        "type": "STATE",
        "shape": [1],
    }
    _write_json(config, document)
    _recommit(fixed_run)
    with pytest.raises(arm6.FireflyArm6AttestationError, match="only J1-J6"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_child_a_b_mix(fixed_run):
    first = fixed_run["output"] / "checkpoints/010000/pretrained_model" / arm6.CHILD_ATTESTATION_FILENAME
    second = fixed_run["output"] / "checkpoints/020000/pretrained_model" / arm6.CHILD_ATTESTATION_FILENAME
    shutil.copyfile(first, second)
    with pytest.raises(arm6.FireflyArm6AttestationError):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_unattested_extra_checkpoint_file(fixed_run):
    extra = fixed_run["output"] / "checkpoints/010000/pretrained_model/extra.bin"
    extra.write_bytes(b"unbound")
    with pytest.raises(arm6.FireflyArm6AttestationError, match="exact regular artifact set"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_checkpoint_symlink_and_hardlink(fixed_run):
    first = fixed_run["output"] / "checkpoints/010000/pretrained_model/model.safetensors"
    first.unlink()
    first.symlink_to("train_config.json")
    with pytest.raises(arm6.FireflyArm6AttestationError, match="symbolic-link"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_cross_checkpoint_hardlink(fixed_run):
    first = fixed_run["output"] / "checkpoints/010000/pretrained_model/model.safetensors"
    second = fixed_run["output"] / "checkpoints/020000/pretrained_model/model.safetensors"
    second.unlink()
    os.link(first, second)
    with pytest.raises(arm6.FireflyArm6AttestationError, match="hard link"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_projection_jaw_mapping_and_video_drift(fixed_run):
    manifest = json.loads(fixed_run["manifest"].read_text())
    manifest["transform"]["removed_feature"]["source_index"] = 5
    unsigned = dict(manifest)
    unsigned.pop("document_sha256")
    manifest["document_sha256"] = _digest(unsigned)
    _write_json(fixed_run["manifest"], manifest)
    with pytest.raises(arm6.FireflyArm6AttestationError, match="7-to-6 transform"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_projection_video_byte_identity_drift(fixed_run):
    video = fixed_run["destination"] / "videos/observation.images.scene/chunk-000/episode.mp4"
    video.write_bytes(b"not-source-video")
    with pytest.raises(arm6.FireflyArm6AttestationError, match="complete destination files"):
        arm6.capture_firefly_arm6_attestation(runner=fixed_run["runner"])


def test_rejects_producer_result_mismatch(fixed_run):
    def runner(command, **_kwargs):
        expected = arm6._capture_local().expected_producer_result
        expected["last_target"] = "010000"
        return subprocess.CompletedProcess(command, 0, _bytes(expected).decode(), "")

    with pytest.raises(arm6.FireflyArm6AttestationError, match="producer verification result"):
        arm6.capture_firefly_arm6_attestation(runner=runner)


def test_rejects_toctou_change_restored_during_producer_call(fixed_run):
    target = fixed_run["source"] / "meta/info.json"

    def runner(command, **_kwargs):
        expected = arm6._capture_local().expected_producer_result
        original = target.read_bytes()
        target.write_bytes(b"temporary attacker bytes")
        target.write_bytes(original)
        return subprocess.CompletedProcess(command, 0, _bytes(expected).decode(), "")

    with pytest.raises(arm6.Arm6AttestationChangedError, match="changed across"):
        arm6.capture_firefly_arm6_attestation(runner=runner)
