from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from scripts import train_firefly_smolvla_arm_only_v1_finetune_20k_run2 as launcher


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _file_record(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _inventory(root: Path) -> list[dict]:
    return [
        _file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _policy_features(dimension: int) -> tuple[dict, dict]:
    return (
        {
            "observation.state": {"type": "STATE", "shape": [dimension]},
            "observation.images.scene": {
                "type": "VISUAL",
                "shape": [3, 480, 640],
            },
            "observation.images.wrist": {
                "type": "VISUAL",
                "shape": [3, 480, 640],
            },
        },
        {"action": {"type": "ACTION", "shape": [dimension]}},
    )


def _dataset_info() -> dict:
    visual_info = {
        "is_depth_map": False,
        "video.height": 480,
        "video.width": 640,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": False,
    }
    features = {
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": launcher.ARM_JOINT_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": launcher.ARM_JOINT_NAMES,
        },
        "observation.images.scene": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": visual_info,
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": visual_info,
        },
    }
    for key, dtype in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": features,
        "total_episodes": 50,
        "total_frames": 91836,
        "total_tasks": 1,
        "robot_type": "hex_zenoh_observer",
        "splits": {"train": "0:50"},
    }


def _dataset_statistics() -> dict:
    statistics = {}
    for feature_index, feature in enumerate(_dataset_info()["features"]):
        base = float(feature_index + 1)
        if feature in {"action", "observation.state"}:
            values = [base + joint / 100.0 for joint in range(6)]
        elif feature.startswith("observation.images."):
            values = [[[base + channel / 100.0]] for channel in range(3)]
        else:
            values = [base]
        statistics[feature] = {
            statistic: copy.deepcopy(values)
            for statistic in launcher._VECTOR_STATISTICS
        }
        statistics[feature]["count"] = [
            13962 if feature.startswith("observation.images.") else launcher.EXPECTED_FRAMES
        ]
    return statistics


def _processor_state(statistics: dict, *, preprocessor: bool) -> dict[str, np.ndarray]:
    tensors = {}
    for feature, feature_stats in statistics.items():
        for statistic, values in feature_stats.items():
            value = np.asarray(values, dtype=np.float32)
            if preprocessor and feature.startswith("observation.images.") and statistic == "count":
                value = value.reshape(1, 1, 1)
            tensors[f"{feature}.{statistic}"] = value
    return tensors


def test_effective_processor_statistics_override_only_non_depth_camera_mean_std():
    raw_statistics = _dataset_statistics()
    raw_before = copy.deepcopy(raw_statistics)
    features = _dataset_info()["features"]

    contract = launcher._derive_effective_processor_statistics(
        raw_statistics,
        features,
        use_imagenet_stats=True,
    )

    assert raw_statistics == raw_before
    assert contract["camera_keys"] == [
        "observation.images.scene",
        "observation.images.wrist",
    ]
    assert contract["depth_keys"] == []
    assert contract["non_depth_camera_keys"] == contract["camera_keys"]
    assert contract["raw_statistics_sha256"] == _digest(raw_before)
    assert contract["effective_statistics_sha256"] == _digest(
        contract["effective_statistics"]
    )
    assert len(contract["overrides"]) == 4
    effective = contract["effective_statistics"]
    expected_effective = copy.deepcopy(raw_before)
    for camera in contract["camera_keys"]:
        for statistic in ("mean", "std"):
            expected_effective[camera][statistic] = copy.deepcopy(
                launcher.PINNED_IMAGENET_PROCESSOR_STATS[statistic]
            )
    assert effective == expected_effective
    for camera in contract["camera_keys"]:
        for statistic in ("mean", "std"):
            assert (
                effective[camera][statistic]
                == launcher.PINNED_IMAGENET_PROCESSOR_STATS[statistic]
            )
        for statistic in (*launcher._VECTOR_STATISTICS, "count"):
            if statistic not in {"mean", "std"}:
                assert effective[camera][statistic] == raw_before[camera][statistic]
    assert effective["action"] == raw_before["action"]
    assert effective["observation.state"] == raw_before["observation.state"]

    depth_features = copy.deepcopy(features)
    depth_features["observation.images.scene"]["info"] = {
        **depth_features["observation.images.scene"]["info"],
        "is_depth_map": True,
    }
    depth_contract = launcher._derive_effective_processor_statistics(
        raw_statistics,
        depth_features,
        use_imagenet_stats=True,
    )
    assert depth_contract["depth_keys"] == ["observation.images.scene"]
    assert depth_contract["non_depth_camera_keys"] == ["observation.images.wrist"]
    assert (
        depth_contract["effective_statistics"]["observation.images.scene"]
        == raw_before["observation.images.scene"]
    )
    assert (
        depth_contract["effective_statistics"]["observation.images.wrist"]["mean"]
        == launcher.PINNED_IMAGENET_PROCESSOR_STATS["mean"]
    )


def test_effective_processor_statistics_false_is_raw_and_type_strict():
    raw_statistics = _dataset_statistics()
    features = _dataset_info()["features"]
    contract = launcher._derive_effective_processor_statistics(
        raw_statistics,
        features,
        use_imagenet_stats=False,
    )
    assert contract["overrides"] == []
    assert contract["effective_statistics"] == raw_statistics
    assert contract["effective_statistics_sha256"] == contract["raw_statistics_sha256"]

    for invalid in (0, 1, "true", None):
        with pytest.raises(launcher.ArmOnlyFinetuneError, match="strict boolean"):
            launcher._derive_effective_processor_statistics(
                raw_statistics,
                features,
                use_imagenet_stats=invalid,
            )


def _write_controller(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "joint_names": [*launcher.ARM_JOINT_NAMES, "jaw_joint1.pos"],
                "effective_step_limits": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.007],
            }
        ),
        encoding="utf-8",
    )


def _manifest_transform() -> dict:
    return {
        "projected_features": {
            key: {
                "source_dtype": "float32",
                "source_shape": [7],
                "destination_dtype": "float32",
                "destination_shape": [6],
                "operation": "destination_value[i] = source_value[i] for i in 0..5",
            }
            for key in ("action", "observation.state")
        },
        "retained_feature_names": launcher.ARM_JOINT_NAMES,
        "removed_feature": {
            "name": "jaw_joint1.pos",
            "source_index": 6,
            "removed_from": ["action", "observation.state"],
            "runtime_authority": (
                "jaw must be controlled independently; this dataset emits no jaw action"
            ),
        },
        "rewritten_metadata": [
            "meta/info.json feature shapes/names",
            "meta/stats.json action/state vector statistics",
            "meta/episodes/**/*.parquet action/state vector statistics",
            "data/**/*.parquet Arrow and Hugging Face schemas",
        ],
        "unchanged": [
            "episode/frame/task/index/timestamp columns",
            "all non-action/state parquet columns",
            "all videos byte-for-byte",
            "all unrelated metadata byte-for-byte",
        ],
    }


def _write_dataset(
    root: Path,
    controller: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict], dict]:
    files = {
        "meta/info.json": json.dumps(_dataset_info()).encode(),
        "meta/tasks.parquet": b"fixture task parquet",
        "meta/stats.json": json.dumps(_dataset_statistics()).encode(),
        "data/chunk-000/file-000.parquet": b"fixture six-axis rows",
        "videos/observation.images.scene/chunk-000/file-000.mp4": b"scene",
        "videos/observation.images.wrist/chunk-000/file-000.mp4": b"wrist",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    destination_files = _inventory(root)
    destination_directories = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )
    source_files = [
        {
            "path": f"source/file-{index:03d}.bin",
            "size": index,
            "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        }
        for index in range(197)
    ]
    source_digest = _digest(source_files)
    monkeypatch.setattr(launcher, "SOURCE_FILES_MANIFEST_SHA256", source_digest)
    directories_digest = _digest(destination_directories)
    monkeypatch.setattr(launcher, "SOURCE_DIRECTORIES_SHA256", directories_digest)
    controller_bytes = controller.read_bytes()
    limits = json.loads(controller_bytes)["effective_step_limits"]
    document = {
        "schema_version": 1,
        "kind": "firefly_y6_arm_only_dataset_projection",
        "projection_version": "exact_first_6_of_7_v1",
        "source": {
            "repo_id": launcher.SOURCE_DATASET_REPO_ID,
            "root": str(launcher.SOURCE_DATASET_ROOT),
            "episode_count": 50,
            "frame_count": 91836,
            "fps": 30,
            "task": launcher.EXPECTED_TASK,
            "upstream_training_input_digest": launcher.PARENT_INPUT_DIGEST,
            "files_manifest_sha256": source_digest,
            "files": source_files,
            "directories_sha256": directories_digest,
            "directories": destination_directories,
        },
        "destination": {
            "repo_id": launcher.DATASET_REPO_ID,
            "root": str(root),
            "episode_count": 50,
            "frame_count": 91836,
            "fps": 30,
            "task": launcher.EXPECTED_TASK,
            "files_manifest_sha256": _digest(destination_files),
            "files": destination_files,
            "directories_sha256": directories_digest,
            "directories": destination_directories,
        },
        "transform": _manifest_transform(),
        "controller_contract": {
            "path": str(controller),
            "size": len(controller_bytes),
            "sha256": hashlib.sha256(controller_bytes).hexdigest(),
            "retained_arm_effective_step_limits_rad": [float(value) for value in limits[:6]],
            "removed_jaw_effective_step_limit_rad": float(limits[6]),
        },
        "verification": {
            "exact_first_six_float32_bits": True,
            "one_to_one_file_paths": True,
            "copied_files_byte_identical": True,
            "destination_files_independent_regular_single_link": True,
            "source_unchanged_before_publish": True,
        },
    }
    document["document_sha256"] = _digest(document)
    manifest_path = root / launcher.DATASET_MANIFEST_RELATIVE
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return destination_files, document


def _write_config(path: Path, dataset_root: Path, output_dir: Path) -> None:
    raw = yaml.safe_load(launcher.CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dataset"]["root"] = str(dataset_root)
    raw["output_dir"] = str(output_dir)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _write_parent(parent: Path, vlm: Path) -> tuple[dict[str, tuple[int, str]], str]:
    parent.mkdir(parents=True)
    vlm.mkdir()
    (vlm / "tokenizer.json").write_text("{}", encoding="utf-8")
    inputs = {
        "dataset": {"repo_id": "local/firefly_y6_fullcycle_v4", "root": "/fixture/source"},
        "controller_contract": {"path": "/fixture/controller.json"},
        "expected_task": launcher.EXPECTED_TASK,
    }
    input_digest = _digest(inputs)
    token = f"{launcher.PARENT_ATTESTATION_TOKEN_PREFIX}{input_digest}"
    attestation = {
        "schema_version": 1,
        "kind": launcher.PARENT_ATTESTATION_KIND,
        "input_digest": input_digest,
        "token": token,
        "inputs": inputs,
        "verification": {
            "training_returncode": 0,
            "inputs_unchanged_after_training": True,
            "postflight_input_digest": input_digest,
        },
    }
    parent_inputs, parent_outputs = _policy_features(7)
    policy_config = {
        "type": "smolvla",
        "n_obs_steps": 1,
        "input_features": parent_inputs,
        "output_features": parent_outputs,
        "device": "cuda",
        "use_amp": False,
        "use_peft": False,
        "push_to_hub": False,
        "repo_id": None,
        "private": None,
        "tags": None,
        "license": None,
        "pretrained_path": "/fixture/base",
        "pretrained_revision": None,
        "chunk_size": 50,
        "n_action_steps": 50,
        "normalization_mapping": {
            "ACTION": "MEAN_STD",
            "STATE": "MEAN_STD",
            "VISUAL": "IDENTITY",
        },
        "max_state_dim": 32,
        "max_action_dim": 32,
        "resize_imgs_with_padding": [512, 512],
        "empty_cameras": 0,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "tokenizer_max_length": 48,
        "num_steps": 10,
        "use_cache": True,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "optimizer_lr": 1e-4,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-10,
        "optimizer_grad_clip_norm": 10.0,
        "scheduler_warmup_steps": 2000,
        "scheduler_decay_steps": 100000,
        "scheduler_decay_lr": 2.5e-6,
        "vlm_model_name": str(vlm),
        "load_vlm_weights": False,
        "add_image_special_tokens": False,
        "attention_mode": "cross_attn",
        "prefix_length": 0,
        "pad_language_to": "max_length",
        "num_expert_layers": 0,
        "num_vlm_layers": 16,
        "self_attn_every_n_layers": 2,
        "expert_width_multiplier": 0.75,
        "min_period": 0.004,
        "max_period": 4.0,
        "rtc_config": None,
        "compile_model": False,
        "compile_mode": "max-autotune",
    }
    train_config = {
        "dataset": {
            "repo_id": "local/firefly_y6_fullcycle_v4",
            "root": "/fixture/source",
            "episodes": None,
            "image_transforms": {"enable": False, "sentinel_nested_bool": True},
            "revision": None,
            "use_imagenet_stats": True,
            "video_backend": "torchcodec",
            "return_uint8": True,
            "depth_output_unit": "mm",
            "streaming": False,
            "eval_split": 0.0,
        },
        "env": None,
        "policy": copy.deepcopy(policy_config),
        "reward_model": None,
        "output_dir": "/fixture/parent-output",
        "job_name": "smolvla",
        "resume": False,
        "seed": 1000,
        "cudnn_deterministic": False,
        "num_workers": 4,
        "batch_size": 1,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "steps": 100000,
        "env_eval_freq": 0,
        "log_freq": 20,
        "eval_steps": 0,
        "max_eval_samples": 0,
        "tolerance_s": 0.0001,
        "save_checkpoint": True,
        "save_freq": 10000,
        "use_policy_training_preset": True,
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": 2000,
            "num_decay_steps": 100000,
            "peak_lr": 1e-4,
            "decay_lr": 2.5e-6,
        },
        "eval": {"n_episodes": 50},
        "wandb": {"enable": False, "notes": token},
        "peft": None,
        "job": {"target": "local"},
        "save_checkpoint_to_hub": False,
        "sample_weighting": None,
        "rename_map": {},
        "checkpoint_path": None,
    }
    save_file(
        {
            "model.bias": np.ones((2,), dtype=np.float32),
            "model.weight": np.ones((2, 2), dtype=np.float32),
        },
        parent / "model.safetensors",
    )
    contents = {
        "config.json": json.dumps(policy_config).encode(),
        launcher.PARENT_ATTESTATION_FILENAME: json.dumps(attestation).encode(),
        "policy_postprocessor.json": b"{}",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": b"post",
        "policy_preprocessor.json": b"{}",
        "policy_preprocessor_step_5_normalizer_processor.safetensors": b"pre",
        "train_config.json": json.dumps(train_config).encode(),
    }
    for name, content in contents.items():
        (parent / name).write_bytes(content)
    pins = {
        item["path"]: (item["size"], item["sha256"])
        for item in _inventory(parent)
    }
    return pins, input_digest


def _fixture_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    controller = tmp_path / "controller.json"
    _write_controller(controller)
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, controller, monkeypatch)
    parent = tmp_path / "parent"
    vlm = tmp_path / "vlm"
    parent_pins, parent_digest = _write_parent(parent, vlm)
    output = tmp_path / "output"
    config = tmp_path / "train.yaml"
    _write_config(config, dataset, output)
    training_python = tmp_path / "python3.12"
    training_python.write_bytes(b"fixture python")
    seen_parent = []

    def parent_verifier(path, document):
        seen_parent.append((path, document["input_digest"]))

    kwargs = {
        "dataset_root": dataset,
        "parent_checkpoint": parent,
        "controller_contract": controller,
        "config_path": config,
        "output_dir": output,
        "training_python": training_python,
        "task_loader": lambda _root: [
            {"task_index": 0, "task": launcher.EXPECTED_TASK}
        ],
        "projection_verifier": lambda _root, _controller: {
            "safe_for_arm_only_training": True,
            "action_dimensions": 6,
            "state_dimensions": 6,
        },
        "parent_verifier": parent_verifier,
        "vlm_validator": lambda path: None,
        "environment_loader": lambda _python: {
            "python_executable": str(training_python),
            "packages": {"lerobot": "fixture"},
            "offline_overrides": launcher.OFFLINE_ENVIRONMENT,
        },
        "verification_runtime_loader": lambda _python: {
            "python_executable": str(training_python),
            "fixture_isolated_runtime": True,
        },
        "parser_preflight": lambda _command, _python, expected: copy.deepcopy(expected),
        "expected_parent_files": parent_pins,
        "expected_parent_input_digest": parent_digest,
        "expected_vlm_bundle": vlm,
    }
    return {**kwargs, "seen_parent": seen_parent}


def _command_argument(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def _fake_smoke(_model: Path, _python: Path) -> dict:
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


def _write_child_checkpoint(
    output: Path, step: int, capture: launcher.FinetuneInputCapture
) -> Path:
    model = output / "checkpoints" / f"{step:06d}" / "pretrained_model"
    model.mkdir(parents=True)
    policy = launcher._expected_child_policy(capture)
    train_config = launcher._expected_child_train_config(capture, output_dir=output)
    (model / "config.json").write_text(json.dumps(policy), encoding="utf-8")
    (model / "train_config.json").write_text(json.dumps(train_config), encoding="utf-8")
    save_file(
        {
            "model.bias": np.full((2,), step, dtype=np.float32),
            "model.weight": np.full((2, 2), step, dtype=np.float32),
        },
        model / "model.safetensors",
    )
    preprocessor, postprocessor = launcher._expected_processor_documents(policy)
    (model / "policy_preprocessor.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )
    (model / "policy_postprocessor.json").write_text(
        json.dumps(postprocessor), encoding="utf-8"
    )
    statistics = capture.payload["processor_statistics_contract"][
        "effective_statistics"
    ]
    save_file(
        _processor_state(statistics, preprocessor=True),
        model / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_file(
        _processor_state(statistics, preprocessor=False),
        model / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    training_state = model.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(
        json.dumps({"step": step, "num_processes": 1, "batch_size": 1}),
        encoding="utf-8",
    )
    return model


def _fixture_capture(fixture: dict) -> launcher.FinetuneInputCapture:
    core = launcher.capture_core_inputs(**fixture)
    command = launcher.build_training_command(
        core,
        output_dir=fixture["output_dir"],
        training_python=fixture["training_python"],
    )
    return launcher.finalize_capture(core, command)


def _successful_fake_run(fixture: dict) -> tuple[list[Path], launcher.FinetuneInputCapture]:
    seen = {}

    def runner(command, *, check, env, cwd):
        assert check is False
        assert cwd == str(launcher.TRAINING_WORKING_DIRECTORY)
        assert env["HF_HUB_OFFLINE"] == "1"
        capture = launcher.finalize_capture(launcher.capture_core_inputs(**fixture), command)
        _write_child_checkpoint(fixture["output_dir"], 10000, capture)
        _write_child_checkpoint(fixture["output_dir"], 20000, capture)
        (fixture["output_dir"] / "checkpoints/last").symlink_to("020000")
        seen["capture"] = capture
        return subprocess.CompletedProcess(command, 0)

    written = launcher.run_attested_finetune(
        **fixture,
        runner=runner,
        child_smoke_validator=_fake_smoke,
    )
    return written, seen["capture"]


def test_checked_in_config_and_preview_are_exact(monkeypatch, capsys):
    launcher.validate_training_config()
    monkeypatch.setattr(launcher, "_bootstrap_or_reexec", lambda _argv: None)
    monkeypatch.setattr(
        launcher, "_bootstrap_runtime_snapshot", _clean_bootstrap_snapshot
    )
    monkeypatch.setattr(
        launcher,
        "run_attested_finetune",
        lambda: pytest.fail("dry-run must not capture inputs or train"),
    )
    assert launcher.main(["--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    command = preview["command"]
    assert _command_argument(command, "policy.path") == str(launcher.PARENT_CHECKPOINT)
    assert _command_argument(command, "policy.input_features") == "null"
    assert _command_argument(command, "policy.output_features") == "null"
    assert _command_argument(command, "policy.optimizer_lr") == "2e-05"
    assert _command_argument(command, "policy.scheduler_warmup_steps") == "500"
    assert _command_argument(command, "policy.scheduler_decay_steps") == "20000"
    assert _command_argument(command, "steps") == "20000"
    assert _command_argument(command, "save_freq") == "10000"
    assert _command_argument(command, "batch_size") == "1"
    assert _command_argument(command, "policy.use_amp") == "false"
    assert preview["environment"] == launcher.OFFLINE_ENVIRONMENT


def test_verify_run_cli_is_fixed_canonical_read_only(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_bootstrap_or_reexec", lambda _argv: None)
    expected = {
        "schema_version": 1,
        "kind": "fixture-verification",
        "policy_schema_role": launcher.POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": launcher.AUTHORIZED_RUNTIME_ROLE,
    }
    monkeypatch.setattr(
        launcher, "verify_fixed_attested_run", lambda: copy.deepcopy(expected)
    )
    monkeypatch.setattr(
        launcher,
        "run_attested_finetune",
        lambda: pytest.fail("verify-run must never start training"),
    )
    assert launcher.main(["--verify-run"]) == 0
    assert capsys.readouterr().out == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    with pytest.raises(SystemExit):
        launcher.parse_args(["--verify-run", "/tmp/arbitrary-output"])


def test_verify_run_cli_returns_nonzero_on_missing_or_tampered_run(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_bootstrap_or_reexec", lambda _argv: None)

    def refuse():
        raise launcher.ControlledInputChangedError("run commit marker missing or tampered")

    monkeypatch.setattr(launcher, "verify_fixed_attested_run", refuse)
    monkeypatch.setattr(
        launcher,
        "run_attested_finetune",
        lambda: pytest.fail("failed verify-run must never start training"),
    )
    assert launcher.main(["--verify-run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run commit marker missing or tampered" in captured.err


def _clean_bootstrap_snapshot() -> dict:
    return {
        "executable": launcher._BOOTSTRAP_TRAINING_PYTHON,
        "executable_sha256": launcher._BOOTSTRAP_TRAINING_PYTHON_SHA256,
        "isolated": 1,
        "no_user_site": 1,
        "ignore_environment": 1,
        "safe_path": True,
        "cwd": launcher._BOOTSTRAP_WORKING_DIRECTORY,
        "script": launcher._BOOTSTRAP_SCRIPT,
        "argv0": launcher._BOOTSTRAP_SCRIPT,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executable", "/usr/bin/python3"),
        ("executable_sha256", "0" * 64),
        ("isolated", 0),
        ("no_user_site", 0),
        ("ignore_environment", 0),
        ("safe_path", False),
        ("cwd", "/tmp"),
        ("sitecustomize_loaded", True),
        ("usercustomize_loaded", True),
    ],
)
def test_bootstrap_runtime_rejects_wrong_python_flags_cwd_and_customize(field, value):
    snapshot = _clean_bootstrap_snapshot()
    snapshot[field] = value
    environment = launcher._bootstrap_environment({})
    environment[launcher._VERIFICATION_REEXEC_MARKER] = (
        launcher._VERIFICATION_REEXEC_VALUE
    )
    assert field in launcher._bootstrap_runtime_violations(snapshot, environment)


def test_bootstrap_runtime_rejects_environment_injection_and_fake_marker(monkeypatch):
    snapshot = _clean_bootstrap_snapshot()
    environment = launcher._bootstrap_environment({})
    environment[launcher._VERIFICATION_REEXEC_MARKER] = (
        launcher._VERIFICATION_REEXEC_VALUE
    )
    environment["PYTHONPATH"] = "/tmp/attacker"
    assert "environment" in launcher._bootstrap_runtime_violations(snapshot, environment)

    monkeypatch.setattr(launcher, "_bootstrap_runtime_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        launcher.os,
        "environ",
        {
            **launcher._bootstrap_environment({}),
            launcher._VERIFICATION_REEXEC_MARKER: "spoofed",
        },
    )
    with pytest.raises(SystemExit, match="internal_marker"):
        launcher._bootstrap_or_reexec(["--dry-run"])


def test_bootstrap_reexec_scrubs_environment_and_uses_fixed_command(monkeypatch):
    class ExecObserved(Exception):
        pass

    calls = {}

    def fake_chdir(path):
        calls["cwd"] = path

    def fake_execve(path, argv, environment):
        calls.update(path=path, argv=argv, environment=environment)
        raise ExecObserved

    monkeypatch.setattr(
        launcher.os,
        "environ",
        {
            "PYTHONPATH": "/tmp/attacker",
            "PYTHONUSERBASE": "/tmp/userbase",
            "LD_LIBRARY_PATH": "/tmp/injected",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    with pytest.raises(ExecObserved):
        launcher._bootstrap_or_reexec(
            ["--dry-run"], execve=fake_execve, chdir=fake_chdir
        )
    assert calls["cwd"] == launcher._BOOTSTRAP_WORKING_DIRECTORY
    assert calls["path"] == launcher._BOOTSTRAP_TRAINING_PYTHON
    assert calls["argv"] == [
        launcher._BOOTSTRAP_TRAINING_PYTHON,
        "-I",
        "-s",
        launcher._BOOTSTRAP_SCRIPT,
        "--dry-run",
    ]
    assert calls["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls["environment"][launcher._VERIFICATION_REEXEC_MARKER] == (
        launcher._VERIFICATION_REEXEC_VALUE
    )
    for name in ("PYTHONPATH", "PYTHONUSERBASE", "LD_LIBRARY_PATH"):
        assert name not in calls["environment"]


def test_real_dry_run_auto_reexecs_from_wrong_python_cwd_and_dirty_env(tmp_path):
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "/tmp/attacker",
            "PYTHONUSERBASE": "/tmp/userbase",
            "LD_LIBRARY_PATH": "/tmp/injected",
        }
    )
    environment.pop(launcher._VERIFICATION_REEXEC_MARKER, None)
    result = subprocess.run(
        [sys.executable, launcher._BOOTSTRAP_SCRIPT, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    runtime = preview["verification_runtime"]
    assert runtime["python_executable"] == launcher._BOOTSTRAP_TRAINING_PYTHON
    assert runtime["working_directory"] == launcher._BOOTSTRAP_WORKING_DIRECTORY
    assert runtime["flags"] == {
        "isolated": 1,
        "no_user_site": 1,
        "ignore_environment": 1,
        "safe_path": True,
    }
    assert runtime["sitecustomize_loaded"] is False
    assert runtime["usercustomize_loaded"] is False
    for name in ("PYTHONPATH", "PYTHONUSERBASE", "LD_LIBRARY_PATH"):
        assert name not in runtime["environment_variable_names"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("steps",), 20001, "steps"),
        (("dataset", "image_transforms", "enable"), True, "image_transforms"),
        (("job_name",), "arbitrary", "job_name"),
    ],
)
def test_config_drift_is_rejected(tmp_path, path, value, message):
    config = tmp_path / "train.yaml"
    output = tmp_path / "output"
    _write_config(config, tmp_path / "dataset", output)
    raw = yaml.safe_load(config.read_text())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match=message):
        launcher.validate_training_config(
            config,
            dataset_root=tmp_path / "dataset",
            output_dir=output,
        )


def test_dataset_accepts_exact_6d_and_binds_projection_manifest(tmp_path, monkeypatch):
    controller = tmp_path / "controller.json"
    _write_controller(controller)
    dataset = tmp_path / "dataset"
    destination_files, document = _write_dataset(dataset, controller, monkeypatch)
    seen = []
    result = launcher.validate_dataset_metadata(
        dataset,
        task_loader=lambda _root: [{"task_index": 0, "task": launcher.EXPECTED_TASK}],
        controller_contract=controller,
        projection_verifier=lambda root, contract: seen.append((root, contract))
        or {"safe_for_arm_only_training": True},
    )
    assert seen == [(dataset, controller)]
    assert result["metadata"]["features"]["action"]["shape"] == [6]
    assert result["metadata"]["features"]["observation.state"]["shape"] == [6]
    assert result["arm_only_manifest"]["document"] == document
    declared = result["arm_only_manifest"]["document"]["destination"]["files"]
    assert declared == destination_files
    assert launcher.DATASET_MANIFEST_RELATIVE.as_posix() not in {
        item["path"] for item in declared
    }


def test_capture_rejects_verifier_runtime_before_controlled_input_validation(
    tmp_path, monkeypatch
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    touched = []

    def refuse_runtime(_python):
        raise launcher.ArmOnlyFinetuneError("injected verifier runtime refusal")

    fixture["verification_runtime_loader"] = refuse_runtime
    fixture["task_loader"] = lambda _root: touched.append("task") or []
    fixture["projection_verifier"] = (
        lambda _root, _controller: touched.append("projection") or {}
    )
    fixture["parent_verifier"] = (
        lambda _path, _document: touched.append("parent")
    )
    fixture["environment_loader"] = (
        lambda _python: touched.append("environment") or {}
    )
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="runtime refusal"):
        launcher.capture_core_inputs(**fixture)
    assert touched == []


def test_dataset_rejects_7d_jaw_camera_task_and_manifest_tamper(tmp_path, monkeypatch):
    controller = tmp_path / "controller.json"
    _write_controller(controller)
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, controller, monkeypatch)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [7]
    info["features"]["action"]["names"].append("jaw_joint1.pos")
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="shape"):
        launcher.validate_dataset_metadata(
            dataset,
            task_loader=lambda _root: [{"task_index": 0, "task": launcher.EXPECTED_TASK}],
            controller_contract=controller,
            projection_verifier=lambda *_args: {"safe_for_arm_only_training": True},
        )

    info_path.write_text(json.dumps(_dataset_info()), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="task row"):
        launcher.validate_dataset_metadata(
            dataset,
            task_loader=lambda _root: [{"task_index": 0, "task": "drift"}],
            controller_contract=controller,
            projection_verifier=lambda *_args: {"safe_for_arm_only_training": True},
        )

    manifest_path = dataset / launcher.DATASET_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text())
    manifest["transform"]["removed_feature"]["source_index"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="document_sha256"):
        launcher.validate_dataset_metadata(
            dataset,
            task_loader=lambda _root: [{"task_index": 0, "task": launcher.EXPECTED_TASK}],
            controller_contract=controller,
            projection_verifier=lambda *_args: {"safe_for_arm_only_training": True},
        )


def test_parent_requires_every_pinned_file_and_legacy_verification(tmp_path):
    parent = tmp_path / "parent"
    vlm = tmp_path / "vlm"
    pins, digest = _write_parent(parent, vlm)
    seen = []
    result = launcher.validate_parent_checkpoint(
        parent,
        expected_files=pins,
        expected_input_digest=digest,
        expected_vlm_bundle=vlm,
        parent_verifier=lambda path, document: seen.append((path, document["token"])),
        vlm_validator=lambda _path: None,
    )
    assert {item["path"] for item in result["files"]} == set(pins)
    assert result["training_attestation"]["verified_with_legacy_schema1_verifier"] is True
    assert seen and seen[0][0] == parent

    (parent / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="pinned 100k"):
        launcher.validate_parent_checkpoint(
            parent,
            expected_files=pins,
            expected_input_digest=digest,
            expected_vlm_bundle=vlm,
            parent_verifier=lambda *_args: None,
            vlm_validator=lambda _path: None,
        )


def test_capture_and_command_bind_all_inputs_without_override_surface(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    seen_parent = fixture.pop("seen_parent")
    core = launcher.capture_core_inputs(**fixture)
    command = launcher.build_training_command(
        core,
        output_dir=fixture["output_dir"],
        training_python=fixture["training_python"],
    )
    capture = launcher.finalize_capture(core, command)

    assert seen_parent
    assert core.payload["parent_checkpoint"]["files"]
    assert core.payload["dataset"]["arm_only_manifest"]["document"]["document_sha256"]
    assert core.payload["controller_contract"]["sha256"]
    assert core.payload["training_config"]["sha256"]
    assert core.payload["environment"]["offline_overrides"] == launcher.OFFLINE_ENVIRONMENT
    assert core.payload["cli_parser_preflight"]["dataset_features_derived_before_training"] is True
    assert core.payload["cli_parser_preflight"]["state_dimension"] == 6
    assert core.payload["cli_parser_preflight"]["action_dimension"] == 6
    assert {
        item["path"]
        for item in core.payload["verification_implementation"]["files"]
    } == {
        path.as_posix() for path in launcher._VERIFICATION_IMPLEMENTATION_FILES
    }
    assert capture.payload["command"] == command
    assert _command_argument(command, "policy.path") == str(fixture["parent_checkpoint"])
    assert _command_argument(command, "policy.input_features") == "null"
    assert _command_argument(command, "policy.output_features") == "null"
    assert _command_argument(command, "dataset.revision") == "null"
    assert _command_argument(command, "policy.push_to_hub") == "false"
    assert _command_argument(command, "wandb.notes") == core.token
    with pytest.raises(SystemExit):
        launcher.parse_args(["--policy.path=attacker/model"])


def test_parser_preflight_must_round_trip_exact_full_config(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")

    def tampered_parser(_command, _python, expected):
        parsed = copy.deepcopy(expected)
        parsed["policy"]["input_features"]["observation.state"]["shape"] = [7]
        return parsed

    fixture["parser_preflight"] = tampered_parser
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="parser preflight"):
        launcher.capture_core_inputs(**fixture)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("policy_semantic", "exact parent inheritance"),
        ("nested_bool_as_int", "exact parent inheritance"),
        ("processor_json_7d", "preprocessor exact 6-D"),
        ("processor_nested_bool", "preprocessor exact 6-D"),
        ("processor_state_7d", "float32"),
        ("processor_state_extra", "tensor keys"),
        ("model_nonfinite", "non-finite"),
    ],
)
def test_child_rejects_semantic_type_7d_state_and_nonfinite_tamper(
    tmp_path, monkeypatch, tamper, message
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    capture = _fixture_capture(fixture)
    _write_child_checkpoint(fixture["output_dir"], 10000, capture)
    model = fixture["output_dir"] / "checkpoints/010000/pretrained_model"
    if tamper == "policy_semantic":
        path = model / "config.json"
        document = json.loads(path.read_text())
        document["max_state_dim"] = 64
        path.write_text(json.dumps(document), encoding="utf-8")
    elif tamper == "nested_bool_as_int":
        path = model / "train_config.json"
        document = json.loads(path.read_text())
        document["dataset"]["image_transforms"]["sentinel_nested_bool"] = 1
        path.write_text(json.dumps(document), encoding="utf-8")
    elif tamper in {"processor_json_7d", "processor_nested_bool"}:
        path = model / "policy_preprocessor.json"
        document = json.loads(path.read_text())
        if tamper == "processor_json_7d":
            document["steps"][5]["config"]["features"]["observation.state"]["shape"] = [7]
        else:
            document["steps"][3]["config"]["truncation"] = 1
        path.write_text(json.dumps(document), encoding="utf-8")
    elif tamper in {"processor_state_7d", "processor_state_extra"}:
        statistics = capture.payload["dataset"]["metadata"]["statistics"]
        tensors = _processor_state(statistics, preprocessor=True)
        if tamper == "processor_state_7d":
            tensors["observation.state.mean"] = np.zeros((7,), dtype=np.float32)
        else:
            tensors["attacker.extra"] = np.zeros((1,), dtype=np.float32)
        save_file(
            tensors,
            model / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        )
    else:
        save_file(
            {
                "model.bias": np.asarray([np.nan, 1.0], dtype=np.float32),
                "model.weight": np.ones((2, 2), dtype=np.float32),
            },
            model / "model.safetensors",
        )
    with pytest.raises(launcher.ArmOnlyFinetuneError, match=message):
        launcher._validate_child_checkpoint(model, capture, output_dir=fixture["output_dir"])


@pytest.mark.parametrize(
    "tamper",
    ["raw_camera_stats", "wrong_imagenet_stats", "action_stats", "arm_state_stats"],
)
def test_child_rejects_raw_or_tampered_effective_processor_statistics(
    tmp_path, monkeypatch, tamper
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    capture = _fixture_capture(fixture)
    model = _write_child_checkpoint(fixture["output_dir"], 10000, capture)
    if tamper == "raw_camera_stats":
        statistics = copy.deepcopy(
            capture.payload["dataset"]["metadata"]["statistics"]
        )
    else:
        statistics = copy.deepcopy(
            capture.payload["processor_statistics_contract"][
                "effective_statistics"
            ]
        )
        if tamper == "wrong_imagenet_stats":
            statistics["observation.images.scene"]["mean"][0][0][0] += 0.125
        elif tamper == "action_stats":
            statistics["action"]["mean"][0] += 0.125
        else:
            statistics["observation.state"]["mean"][0] += 0.125
    save_file(
        _processor_state(statistics, preprocessor=True),
        model / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    with pytest.raises(
        launcher.ArmOnlyFinetuneError,
        match="bound effective processor statistics",
    ):
        launcher._validate_child_checkpoint(
            model,
            capture,
            output_dir=fixture["output_dir"],
        )


def test_existing_output_refuses_before_any_input_capture(tmp_path, monkeypatch):
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(
        launcher,
        "capture_core_inputs",
        lambda **_kwargs: pytest.fail("existing output must fail before capture"),
    )
    with pytest.raises(FileExistsError, match="new output"):
        launcher.run_attested_finetune(output_dir=output)


def test_training_environment_scrubs_python_and_endpoint_injection(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/attacker")
    monkeypatch.setenv("LD_PRELOAD", "/attacker/inject.so")
    monkeypatch.setenv("HF_ENDPOINT", "https://attacker.invalid")
    environment = launcher.build_training_environment()
    assert "PYTHONPATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert "HF_ENDPOINT" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_fake_training_attests_both_6d_checkpoints_and_offline_environment(
    tmp_path, monkeypatch
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    seen = {}

    def runner(command, *, check, env, cwd):
        assert check is False
        assert cwd == str(launcher.TRAINING_WORKING_DIRECTORY)
        seen["command"] = command
        seen["env"] = env
        core = launcher.capture_core_inputs(**fixture)
        capture = launcher.finalize_capture(core, command)
        output = fixture["output_dir"]
        _write_child_checkpoint(output, 10000, capture)
        _write_child_checkpoint(output, 20000, capture)
        (output / "checkpoints" / "last").symlink_to("020000")
        return subprocess.CompletedProcess(command, 0)

    written = launcher.run_attested_finetune(
        **fixture,
        runner=runner,
        child_smoke_validator=_fake_smoke,
    )

    assert len(written) == 2
    assert {path.parent.parent.name for path in written} == {"010000", "020000"}
    assert all(path.name == launcher.CHILD_ATTESTATION_FILENAME for path in written)
    assert (fixture["output_dir"] / launcher.RUN_COMMIT_FILENAME).is_file()
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"
    assert seen["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert seen["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert seen["env"]["WANDB_DISABLED"] == "true"
    for path in written:
        document = json.loads(path.read_text())
        assert document["kind"] == launcher.CHILD_ATTESTATION_KIND
        assert document["policy_schema_role"] == launcher.POLICY_SCHEMA_ROLE
        assert document["authorized_runtime_role"] == launcher.AUTHORIZED_RUNTIME_ROLE
        assert document["limitations"] == launcher.ARM_ONLY_OBSERVATION_LIMITATIONS
        processor_contract = document["inputs"]["processor_statistics_contract"]
        assert processor_contract["use_imagenet_stats"] is True
        assert processor_contract["raw_statistics_sha256"] == _digest(
            document["inputs"]["dataset"]["metadata"]["statistics"]
        )
        assert document["inputs"]["training_contract"]["use_imagenet_stats"] is True
        assert document["verification"]["inputs_unchanged_after_training"] is True
        assert document["inputs"]["command"] == seen["command"]
        assert launcher.verify_child_attestation(path.parent, launcher.finalize_capture(
            launcher.capture_core_inputs(**fixture), seen["command"]
        ))["checkpoint"]["step"] in {10000, 20000}


def test_failed_fake_training_never_attests(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")

    def runner(command, *, check, env, cwd):
        assert check is False and env["HF_HUB_OFFLINE"] == "1"
        assert cwd == str(launcher.TRAINING_WORKING_DIRECTORY)
        return subprocess.CompletedProcess(command, 9)

    with pytest.raises(launcher.ArmOnlyFinetuneError, match="status 9"):
        launcher.run_attested_finetune(
            **fixture,
            runner=runner,
            child_smoke_validator=_fake_smoke,
        )
    assert not list(fixture["output_dir"].rglob(launcher.CHILD_ATTESTATION_FILENAME))


def test_config_change_during_fake_training_refuses_attestation(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")

    def runner(command, *, check, env, cwd):
        assert check is False and env["HF_HUB_OFFLINE"] == "1"
        assert cwd == str(launcher.TRAINING_WORKING_DIRECTORY)
        config = fixture["config_path"]
        config.write_text(
            config.read_text(encoding="utf-8") + "\n# changed during training\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(launcher.ControlledInputChangedError, match="changed during training"):
        launcher.run_attested_finetune(
            **fixture,
            runner=runner,
            child_smoke_validator=_fake_smoke,
        )
    assert not list(fixture["output_dir"].rglob(launcher.CHILD_ATTESTATION_FILENAME))


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_atomic_commit_write_failure_leaves_no_verifiable_child(
    tmp_path, monkeypatch, failing_write
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    original = launcher._write_exclusive_json
    calls = 0

    def fail_write(path, document):
        nonlocal calls
        calls += 1
        if calls == failing_write:
            raise launcher.ArmOnlyFinetuneError("injected attestation write failure")
        return original(path, document)

    monkeypatch.setattr(launcher, "_write_exclusive_json", fail_write)
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="injected"):
        _successful_fake_run(fixture)
    output = fixture["output_dir"]
    assert not (output / launcher.RUN_COMMIT_FILENAME).exists()
    assert not list(output.rglob(launcher.CHILD_ATTESTATION_FILENAME))
    assert not list(output.rglob(f"*{launcher.CHILD_ATTESTATION_FILENAME}.staged"))


def test_precommit_validation_and_second_smoke_failure_leave_no_commit(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    original_validate = launcher._validate_run_commit

    def fail_staged_marker(output, capture, *, marker_path=None):
        if marker_path is not None:
            raise launcher.ArmOnlyFinetuneError("injected precommit validation failure")
        return original_validate(output, capture, marker_path=marker_path)

    monkeypatch.setattr(launcher, "_validate_run_commit", fail_staged_marker)
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="precommit"):
        _successful_fake_run(fixture)
    assert not (fixture["output_dir"] / launcher.RUN_COMMIT_FILENAME).exists()
    assert not list(fixture["output_dir"].rglob(launcher.CHILD_ATTESTATION_FILENAME))

    (tmp_path / "second").mkdir()
    second_fixture = _fixture_inputs(tmp_path / "second", monkeypatch)
    second_fixture.pop("seen_parent")
    calls = 0

    def second_smoke(model, python):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise launcher.ArmOnlyFinetuneError("injected second smoke failure")
        return _fake_smoke(model, python)

    def runner(command, *, check, env, cwd):
        capture = launcher.finalize_capture(
            launcher.capture_core_inputs(**second_fixture), command
        )
        _write_child_checkpoint(second_fixture["output_dir"], 10000, capture)
        _write_child_checkpoint(second_fixture["output_dir"], 20000, capture)
        (second_fixture["output_dir"] / "checkpoints/last").symlink_to("020000")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(launcher.ArmOnlyFinetuneError, match="second smoke"):
        launcher.run_attested_finetune(
            **second_fixture,
            runner=runner,
            child_smoke_validator=second_smoke,
        )
    assert not (second_fixture["output_dir"] / launcher.RUN_COMMIT_FILENAME).exists()
    assert not list(second_fixture["output_dir"].rglob(launcher.CHILD_ATTESTATION_FILENAME))


def test_attestation_exact_keys_and_step_match_directory(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    written, capture = _successful_fake_run(fixture)
    path = written[0]
    document = json.loads(path.read_text())
    files = document["checkpoint"]["files"]

    with_extra = copy.deepcopy(document)
    with_extra["unexpected"] = True
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="top-level keys"):
        launcher._validate_child_attestation_document(
            path.parent, capture, with_extra, expected_files=files
        )

    wrong_step = copy.deepcopy(document)
    wrong_step["checkpoint"]["step"] = 20000
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="checkpoint step"):
        launcher._validate_child_attestation_document(
            path.parent, capture, wrong_step, expected_files=files
        )

    wrong_role = copy.deepcopy(document)
    wrong_role["authorized_runtime_role"] = "arm_only_6d"
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="authorized_runtime_role"):
        launcher._validate_child_attestation_document(
            path.parent, capture, wrong_role, expected_files=files
        )


@pytest.mark.parametrize("last_state", ["missing", "wrong"])
def test_joint_verifier_rejects_missing_or_wrong_last(tmp_path, monkeypatch, last_state):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    written, capture = _successful_fake_run(fixture)
    last = fixture["output_dir"] / "checkpoints/last"
    last.unlink()
    if last_state == "wrong":
        last.symlink_to("010000")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="last"):
        launcher.verify_child_attestation(written[0].parent, capture)


@pytest.mark.parametrize("extra", ["staged", "checkpoint"])
def test_joint_verifier_rejects_extra_staged_or_checkpoint(tmp_path, monkeypatch, extra):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    written, capture = _successful_fake_run(fixture)
    if extra == "staged":
        (written[0].parent / launcher._STAGED_ATTESTATION_FILENAME).write_text(
            "{}", encoding="utf-8"
        )
    else:
        (fixture["output_dir"] / "checkpoints/015000").mkdir()
    with pytest.raises(launcher.ArmOnlyFinetuneError):
        launcher.verify_child_attestation(written[0].parent, capture)


def test_joint_verifier_rejects_swapped_marker_steps_and_training_process_drift(
    tmp_path, monkeypatch
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    written, capture = _successful_fake_run(fixture)
    marker_path = fixture["output_dir"] / launcher.RUN_COMMIT_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["children"] = list(reversed(marker["children"]))
    marker["children_sha256"] = _digest(marker["children"])
    marker["document_sha256"] = _digest(
        {key: value for key, value in marker.items() if key != "document_sha256"}
    )
    marker_path.chmod(0o644)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="run commit child"):
        launcher.verify_child_attestation(written[0].parent, capture)

    marker["children"] = list(reversed(marker["children"]))
    marker["authorized_runtime_role"] = "arm_only_6d"
    marker["children_sha256"] = _digest(marker["children"])
    marker["document_sha256"] = _digest(
        {key: value for key, value in marker.items() if key != "document_sha256"}
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="authorized_runtime_role"):
        launcher.verify_child_attestation(written[0].parent, capture)

    training_step = written[0].parent.parent / "training_state/training_step.json"
    training_document = json.loads(training_step.read_text())
    training_document["num_processes"] = 2
    training_step.write_text(json.dumps(training_document), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="num_processes"):
        launcher._training_step_record(written[0].parent.parent, expected_step=10000)


def test_joint_verifier_requires_commit_marker_and_rejects_nested_type_confusion(
    tmp_path, monkeypatch
):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    written, capture = _successful_fake_run(fixture)
    marker_path = fixture["output_dir"] / launcher.RUN_COMMIT_FILENAME
    marker = json.loads(marker_path.read_text())
    marker_path.unlink()
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="controlled input disappeared"):
        launcher.verify_child_attestation(written[0].parent, capture)

    marker["children"][0]["offline_load_smoke"]["strict_weights"] = 1
    marker["children_sha256"] = _digest(marker["children"])
    marker["document_sha256"] = _digest(
        {key: value for key, value in marker.items() if key != "document_sha256"}
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(launcher.ArmOnlyFinetuneError, match="strict_weights"):
        launcher.verify_child_attestation(written[0].parent, capture)


def test_atomic_commit_fsync_failure_leaves_no_verifiable_child(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")
    original = launcher._fsync_directory
    calls = 0

    def fail_second_fsync(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected fsync failure")
        return original(path)

    monkeypatch.setattr(launcher, "_fsync_directory", fail_second_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        _successful_fake_run(fixture)
    output = fixture["output_dir"]
    assert not (output / launcher.RUN_COMMIT_FILENAME).exists()
    assert not list(output.rglob(launcher.CHILD_ATTESTATION_FILENAME))


def test_atomic_commit_rejects_identical_checkpoint_models(tmp_path, monkeypatch):
    fixture = _fixture_inputs(tmp_path, monkeypatch)
    fixture.pop("seen_parent")

    def runner(command, *, check, env, cwd):
        capture = launcher.finalize_capture(launcher.capture_core_inputs(**fixture), command)
        first = _write_child_checkpoint(fixture["output_dir"], 10000, capture)
        second = _write_child_checkpoint(fixture["output_dir"], 20000, capture)
        (second / "model.safetensors").write_bytes(
            (first / "model.safetensors").read_bytes()
        )
        (fixture["output_dir"] / "checkpoints/last").symlink_to("020000")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(launcher.ArmOnlyFinetuneError, match="must be distinct"):
        launcher.run_attested_finetune(
            **fixture,
            runner=runner,
            child_smoke_validator=_fake_smoke,
        )
    output = fixture["output_dir"]
    assert not (output / launcher.RUN_COMMIT_FILENAME).exists()
    assert not list(output.rglob(launcher.CHILD_ATTESTATION_FILENAME))
