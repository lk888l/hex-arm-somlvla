"""Fail-closed dataset attestation for local Firefly SmolVLA training.

This module deliberately wraps the stock LeRobot CLI instead of patching its
training loop.  A deterministic token is injected through ``wandb.notes``;
LeRobot serializes that field into every checkpoint's ``train_config.json``.
After a successful training process, the complete input set is audited and
hashed again before a self-contained attestation is added to each numbered
``pretrained_model`` directory.

The attestation is an integrity/provenance guard for this local workflow, not a
cryptographic signature.  A consumer must recompute the current input capture
and call :func:`verify_training_attestation`; merely finding the JSON file is
not sufficient.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot_robot_hexbase.firefly_dataset_audit import (
    audit_firefly_dataset,
    load_controller_contract,
)
from lerobot_robot_hexbase.smolvla_base_bundle import (
    validate_materialized_smolvla_base,
)


ATTESTATION_FILENAME = "firefly_dataset_attestation.json"
ATTESTATION_KIND = "firefly_smolvla_training_dataset"
ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_TOKEN_PREFIX = "firefly-dataset-attestation-v1:"

_CHECKPOINT_NAME = re.compile(r"^[0-9]+$")
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_JSON_BYTES = 32 * 1024 * 1024


class TrainingAttestationError(RuntimeError):
    """The training inputs or outputs cannot be safely attested."""


class TrainingInputChangedError(TrainingAttestationError):
    """A controlled dataset or contract changed during the workflow."""


@dataclass(frozen=True)
class TrainingInputCapture:
    """Deterministic capture of every input covered by the training proof."""

    dataset_root: Path
    controller_contract: Path
    digest: str
    token: str
    payload: dict[str, Any]


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} path component does not exist: {current}") from exc
        if stat.S_ISLNK(mode):
            raise TrainingAttestationError(
                f"{label} must not contain a symbolic-link component: {current}"
            )
    return absolute


def _require_new_output_path(path: Path) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    missing_parent = False
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing_parent = True
            continue
        if missing_parent:
            raise TrainingAttestationError(
                f"training output has an inconsistent missing ancestor: {current}"
            )
        if stat.S_ISLNK(mode):
            raise TrainingAttestationError(
                f"training output must not contain a symbolic-link component: {current}"
            )
    if absolute.exists():
        raise FileExistsError(
            f"attested training requires a new output directory: {absolute}"
        )
    return absolute


def _identity(path: Path, *, directory: bool | None = None) -> tuple[int, ...]:
    try:
        item = path.lstat()
    except FileNotFoundError as exc:
        raise TrainingInputChangedError(f"controlled input disappeared: {path}") from exc
    if stat.S_ISLNK(item.st_mode):
        raise TrainingAttestationError(f"controlled input is a symbolic link: {path}")
    if directory is True and not stat.S_ISDIR(item.st_mode):
        raise TrainingAttestationError(f"controlled input is not a directory: {path}")
    if directory is False and not stat.S_ISREG(item.st_mode):
        raise TrainingAttestationError(f"controlled input is not a regular file: {path}")
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _walk_tree(
    root: Path, *, label: str
) -> tuple[list[Path], tuple[tuple[str, str, tuple[int, ...]], ...]]:
    root = _require_no_symlink_components(root, label=label)
    _identity(root, directory=True)
    files: list[Path] = []
    records: list[tuple[str, str, tuple[int, ...]]] = []

    def visit(directory: Path) -> None:
        relative_directory = directory.relative_to(root).as_posix()
        before = _identity(directory, directory=True)
        records.append((relative_directory, "directory", before))
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise TrainingAttestationError(
                f"cannot enumerate {label} directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise TrainingInputChangedError(
                    f"cannot stat {label} entry {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise TrainingAttestationError(f"{label} tree contains a symbolic link: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                files.append(path)
                records.append(
                    (path.relative_to(root).as_posix(), "file", _identity(path, directory=False))
                )
            else:
                raise TrainingAttestationError(
                    f"{label} tree contains a non-regular entry: {path}"
                )
        if _identity(directory, directory=True) != before:
            raise TrainingInputChangedError(f"{label} changed while being enumerated: {directory}")

    visit(root)
    records.sort(key=lambda item: (item[0], item[1]))
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return files, tuple(records)


def _hash_regular_file(path: Path) -> tuple[str, int]:
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrainingInputChangedError(f"cannot open controlled input {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before or not stat.S_ISREG(opened.st_mode):
            raise TrainingInputChangedError(f"controlled input changed before hashing: {path}")
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before:
            raise TrainingInputChangedError(f"controlled input changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise TrainingInputChangedError(f"controlled input was replaced while hashing: {path}")
    return digest.hexdigest(), before[3]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingAttestationError(f"attestation value is not strict JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _dataset_files_and_stability(
    dataset_root: Path,
) -> tuple[list[Path], tuple[tuple[str, str, str, tuple[int, ...]], ...]]:
    files: list[Path] = []
    stability: list[tuple[str, str, str, tuple[int, ...]]] = []
    for name in ("meta", "data", "videos"):
        tree_root = dataset_root / name
        tree_files, tree_stability = _walk_tree(
            tree_root, label=f"training dataset {name}"
        )
        files.extend(tree_files)
        stability.extend(
            (name, relative_path, kind, identity)
            for relative_path, kind, identity in tree_stability
        )
    files.sort(key=lambda item: item.relative_to(dataset_root).as_posix())
    stability.sort(key=lambda item: (item[0], item[1], item[2]))
    if not any(
        path.suffix == ".parquet"
        and path.relative_to(dataset_root).parts[0] == "data"
        for path in files
    ):
        raise TrainingAttestationError(f"training dataset contains no data parquet: {dataset_root}")
    if not any(
        path.suffix == ".mp4"
        and path.relative_to(dataset_root).parts[0] == "videos"
        for path in files
    ):
        raise TrainingAttestationError(f"training dataset contains no MP4 video: {dataset_root}")
    return files, tuple(stability)


def capture_training_inputs(
    dataset_root: Path,
    repo_id: str,
    *,
    controller_contract: Path,
    expected_task: str,
) -> TrainingInputCapture:
    """Audit and hash all local inputs used by one formal training run."""

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise TrainingAttestationError("dataset repo_id must be a non-empty string")
    if not isinstance(expected_task, str) or not expected_task.strip():
        raise TrainingAttestationError("expected task must be a non-empty string")
    dataset_root = _require_no_symlink_components(
        dataset_root, label="training dataset"
    )
    _identity(dataset_root, directory=True)
    controller_contract = _require_no_symlink_components(
        controller_contract, label="training controller contract"
    )
    contract_before = _identity(controller_contract, directory=False)
    files, stability_before = _dataset_files_and_stability(dataset_root)

    manifest: list[dict[str, int | str]] = []
    for path in files:
        digest, size = _hash_regular_file(path)
        manifest.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    contract_digest, contract_size = _hash_regular_file(controller_contract)

    contract = load_controller_contract(controller_contract)
    audit = audit_firefly_dataset(
        dataset_root,
        repo_id,
        controller_contract=contract,
        expected_task=expected_task,
    )
    summary = audit.get("summary")
    if not isinstance(summary, dict) or summary.get("safe_for_training") is not True:
        errors = summary.get("errors", []) if isinstance(summary, dict) else []
        codes = [
            item.get("code", "unknown")
            for item in errors
            if isinstance(item, dict)
        ]
        raise TrainingAttestationError(
            "dataset failed the formal pre-training audit: "
            + ", ".join(codes or ["unknown audit error"])
        )

    _current_files, stability_after = _dataset_files_and_stability(dataset_root)
    if stability_after != stability_before:
        raise TrainingInputChangedError(
            "training dataset changed during audit and manifest capture"
        )
    if _identity(controller_contract, directory=False) != contract_before:
        raise TrainingInputChangedError(
            "controller contract changed during audit and manifest capture"
        )

    manifest_digest = _sha256_json(manifest)
    audit_digest = _sha256_json(audit)
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "dataset": {
            "repo_id": repo_id,
            "root": str(dataset_root),
            "manifest_algorithm": "sha256",
            "manifest_sha256": manifest_digest,
            "files": manifest,
        },
        "expected_task": expected_task,
        "controller_contract": {
            "path": str(controller_contract),
            "size": contract_size,
            "sha256": contract_digest,
        },
        "audit": {
            "report_sha256": audit_digest,
            "report": audit,
        },
    }
    digest = _sha256_json(payload)
    return TrainingInputCapture(
        dataset_root=dataset_root,
        controller_contract=controller_contract,
        digest=digest,
        token=f"{ATTESTATION_TOKEN_PREFIX}{digest}",
        payload=payload,
    )


_FORCED_TRAIN_ARGUMENTS = (
    "--dataset.repo_id",
    "--dataset.root",
    "--dataset.episodes",
    "--dataset.revision",
    "--dataset.streaming",
    "--dataset.eval_split",
    "--output_dir",
    "--output-dir",
    "--resume",
    "--save_checkpoint",
    "--save_checkpoint_to_hub",
    "--wandb.enable",
    "--wandb.notes",
    "--job.target",
    "--policy.path",
    "--policy.pretrained_path",
    "--policy.input_features",
    "--policy.output_features",
    "--policy.load_vlm_weights",
    "--policy.vlm_model_name",
    "--policy.use_peft",
    "--policy.push_to_hub",
    "--peft",
    "--rename_map",
)


def _reject_forced_overrides(training_args: Sequence[str]) -> None:
    for argument in training_args:
        for forced in _FORCED_TRAIN_ARGUMENTS:
            if (
                argument == forced
                or argument.startswith(f"{forced}=")
                or argument.startswith(f"{forced}.")
            ):
                raise TrainingAttestationError(
                    f"training argument {forced} is controlled by the attestation wrapper"
                )


def _validated_smolvla_base(path: Path) -> tuple[Path, Path]:
    try:
        base, provenance = validate_materialized_smolvla_base(Path(path))
    except (OSError, ValueError) as exc:
        raise TrainingAttestationError(
            f"invalid pinned local SmolVLA base {path}: {exc}"
        ) from exc
    binding = provenance.get("smolvlm_bundle")
    if not isinstance(binding, dict):
        raise TrainingAttestationError("pinned SmolVLA base lacks a SmolVLM binding")
    vlm_value = binding.get("canonical_path")
    if not isinstance(vlm_value, str) or not Path(vlm_value).is_absolute():
        raise TrainingAttestationError(
            "pinned SmolVLA base has an invalid SmolVLM canonical path"
        )
    return base, Path(vlm_value)


def build_training_command(
    capture: TrainingInputCapture,
    output_dir: Path,
    training_args: Sequence[str],
    *,
    smolvla_base: Path,
    python_executable: Path | str = sys.executable,
) -> list[str]:
    """Build a stock LeRobot command with all provenance-sensitive flags forced."""

    args = list(training_args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise TrainingAttestationError("no LeRobot training arguments were provided after '--'")
    _reject_forced_overrides(args)
    base, vlm_bundle = _validated_smolvla_base(smolvla_base)
    output_dir = _absolute_without_resolving(output_dir)
    dataset = capture.payload["dataset"]
    forced = [
        f"--dataset.repo_id={dataset['repo_id']}",
        f"--dataset.root={capture.dataset_root}",
        "--dataset.episodes=null",
        "--dataset.revision=null",
        "--dataset.streaming=false",
        "--dataset.eval_split=0.0",
        f"--output_dir={output_dir}",
        "--resume=false",
        "--save_checkpoint=true",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=false",
        f"--wandb.notes={capture.token}",
        "--job.target=local",
        f"--policy.path={base}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.load_vlm_weights=false",
        f"--policy.vlm_model_name={vlm_bundle}",
        "--policy.use_peft=false",
        "--policy.push_to_hub=false",
        "--peft=null",
        "--rename_map={}",
    ]
    return [
        os.fspath(python_executable),
        "-m",
        "lerobot.scripts.lerobot_train",
        *args,
        *forced,
    ]


def _read_regular_bytes(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    before = _identity(path, directory=False)
    if before[3] > max_bytes:
        raise TrainingAttestationError(f"JSON file is unexpectedly large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrainingAttestationError(f"cannot open JSON file {path}: {exc}") from exc
    try:
        data = bytearray()
        while chunk := os.read(descriptor, min(_HASH_CHUNK_SIZE, max_bytes + 1 - len(data))):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise TrainingAttestationError(f"JSON file is unexpectedly large: {path}")
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before:
            raise TrainingAttestationError(f"JSON file changed while being read: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise TrainingAttestationError(f"JSON file was replaced while being read: {path}")
    return bytes(data)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingAttestationError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingAttestationError(f"{label} must contain a JSON object: {path}")
    return value


def _discover_pretrained_models(output_dir: Path) -> list[Path]:
    output_dir = _require_no_symlink_components(output_dir, label="training output")
    checkpoints = output_dir / "checkpoints"
    _identity(checkpoints, directory=True)
    models: list[Path] = []
    try:
        entries = sorted(os.scandir(checkpoints), key=lambda entry: entry.name)
    except OSError as exc:
        raise TrainingAttestationError(f"cannot enumerate checkpoints: {exc}") from exc
    for entry in entries:
        path = Path(entry.path)
        mode = entry.stat(follow_symlinks=False).st_mode
        if entry.name == "last":
            if not stat.S_ISLNK(mode):
                raise TrainingAttestationError("checkpoints/last must be LeRobot's symbolic link")
            continue
        if stat.S_ISLNK(mode):
            raise TrainingAttestationError(f"numbered checkpoint must not be a symlink: {path}")
        if not _CHECKPOINT_NAME.fullmatch(entry.name) or not stat.S_ISDIR(mode):
            raise TrainingAttestationError(f"unexpected entry in checkpoints directory: {path}")
        model = path / "pretrained_model"
        _identity(model, directory=True)
        models.append(model)
    if not models:
        raise TrainingAttestationError(f"training produced no numbered checkpoints: {checkpoints}")
    return models


def _validate_checkpoint_train_config(
    pretrained_model: Path, capture: TrainingInputCapture, output_dir: Path
) -> None:
    config = _read_json_object(
        pretrained_model / "train_config.json", label="checkpoint train_config"
    )
    dataset = config.get("dataset")
    policy = config.get("policy")
    wandb = config.get("wandb")
    job = config.get("job")
    if not all(isinstance(value, dict) for value in (dataset, policy, wandb, job)):
        raise TrainingAttestationError(
            f"checkpoint train_config lacks dataset/policy/wandb/job objects: {pretrained_model}"
        )
    pretrained_path = policy.get("pretrained_path")
    if not isinstance(pretrained_path, str) or not Path(pretrained_path).is_absolute():
        raise TrainingAttestationError(
            f"checkpoint policy.pretrained_path must identify the pinned local base: "
            f"{pretrained_model}"
        )
    base, vlm_bundle = _validated_smolvla_base(Path(pretrained_path))
    expected_repo_id = capture.payload["dataset"]["repo_id"]
    checks = {
        "dataset.repo_id": (dataset.get("repo_id"), expected_repo_id),
        "dataset.root": (dataset.get("root"), str(capture.dataset_root)),
        "dataset.episodes": (dataset.get("episodes"), None),
        "dataset.revision": (dataset.get("revision"), None),
        "dataset.streaming": (dataset.get("streaming"), False),
        "dataset.eval_split": (dataset.get("eval_split"), 0.0),
        "policy.type": (policy.get("type"), "smolvla"),
        "policy.pretrained_path": (pretrained_path, str(base)),
        "policy.load_vlm_weights": (policy.get("load_vlm_weights"), False),
        "policy.vlm_model_name": (policy.get("vlm_model_name"), str(vlm_bundle)),
        "policy.use_peft": (policy.get("use_peft"), False),
        "policy.push_to_hub": (policy.get("push_to_hub"), False),
        "peft": (config.get("peft"), None),
        "wandb.enable": (wandb.get("enable"), False),
        "wandb.notes": (wandb.get("notes"), capture.token),
        "job.target": (job.get("target"), "local"),
        "output_dir": (config.get("output_dir"), str(output_dir)),
        "resume": (config.get("resume"), False),
        "save_checkpoint": (config.get("save_checkpoint"), True),
        "save_checkpoint_to_hub": (config.get("save_checkpoint_to_hub"), False),
        "rename_map": (config.get("rename_map"), {}),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise TrainingAttestationError(
            f"checkpoint does not preserve the attested training contract ({pretrained_model}): "
            + "; ".join(mismatches)
        )


def _attestation_document(capture: TrainingInputCapture) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "token": capture.token,
        "input_digest": capture.digest,
        "inputs": capture.payload,
        "verification": {
            "training_returncode": 0,
            "inputs_unchanged_after_training": True,
            "postflight_input_digest": capture.digest,
        },
    }


def _write_new_json(path: Path, value: Any) -> None:
    data = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise TrainingAttestationError(f"cannot create attestation {path}: {exc}") from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TrainingAttestationError(f"short write while creating attestation: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_training_attestation(
    pretrained_model: Path, expected_capture: TrainingInputCapture
) -> dict[str, Any]:
    """Require a valid attestation matching a freshly recomputed input capture."""

    pretrained_model = _require_no_symlink_components(
        pretrained_model, label="attested pretrained model"
    )
    _identity(pretrained_model, directory=True)
    attestation_path = pretrained_model / ATTESTATION_FILENAME
    try:
        attestation_mode = attestation_path.lstat().st_mode
    except FileNotFoundError as exc:
        raise TrainingAttestationError(
            f"checkpoint has no formal Firefly training attestation: {attestation_path}"
        ) from exc
    if stat.S_ISLNK(attestation_mode) or not stat.S_ISREG(attestation_mode):
        raise TrainingAttestationError(
            f"training attestation must be a regular non-symlink file: {attestation_path}"
        )
    step_dir = pretrained_model.parent
    checkpoints_dir = step_dir.parent
    output_dir = checkpoints_dir.parent
    if (
        pretrained_model.name != "pretrained_model"
        or checkpoints_dir.name != "checkpoints"
        or not _CHECKPOINT_NAME.fullmatch(step_dir.name)
    ):
        raise TrainingAttestationError(
            f"attested model must be a numbered checkpoints/<step>/pretrained_model directory: "
            f"{pretrained_model}"
        )
    _validate_checkpoint_train_config(pretrained_model, expected_capture, output_dir)
    document = _read_json_object(
        attestation_path, label="training attestation"
    )
    if document.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise TrainingAttestationError("unsupported training attestation schema")
    if document.get("kind") != ATTESTATION_KIND:
        raise TrainingAttestationError("unexpected training attestation kind")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise TrainingAttestationError("training attestation inputs must be an object")
    embedded_digest = _sha256_json(inputs)
    expected_token = f"{ATTESTATION_TOKEN_PREFIX}{embedded_digest}"
    if document.get("input_digest") != embedded_digest or document.get("token") != expected_token:
        raise TrainingAttestationError("training attestation has an invalid embedded digest")
    verification = document.get("verification")
    if not isinstance(verification, dict) or verification != {
        "training_returncode": 0,
        "inputs_unchanged_after_training": True,
        "postflight_input_digest": embedded_digest,
    }:
        raise TrainingAttestationError("training attestation was not successfully finalized")
    if inputs != expected_capture.payload or embedded_digest != expected_capture.digest:
        raise TrainingInputChangedError(
            "training attestation does not match the currently audited dataset and contract"
        )
    if document.get("token") != expected_capture.token:
        raise TrainingInputChangedError("training attestation token does not match current inputs")
    return document


def finalize_training_attestations(
    output_dir: Path,
    preflight: TrainingInputCapture,
) -> list[Path]:
    """Re-audit unchanged inputs and attest every numbered local checkpoint."""

    output_dir = _absolute_without_resolving(output_dir)
    postflight = capture_training_inputs(
        preflight.dataset_root,
        preflight.payload["dataset"]["repo_id"],
        controller_contract=preflight.controller_contract,
        expected_task=preflight.payload["expected_task"],
    )
    if postflight.payload != preflight.payload or postflight.digest != preflight.digest:
        raise TrainingInputChangedError(
            "dataset, task audit, or controller contract changed during training; "
            "no checkpoint was attested"
        )
    models = _discover_pretrained_models(output_dir)
    for model in models:
        _validate_checkpoint_train_config(model, preflight, output_dir)
        attestation = model / ATTESTATION_FILENAME
        if attestation.exists() or attestation.is_symlink():
            raise TrainingAttestationError(
                f"refusing to overwrite an existing checkpoint attestation: {attestation}"
            )
    document = _attestation_document(preflight)
    written: list[Path] = []
    for model in models:
        path = model / ATTESTATION_FILENAME
        _write_new_json(path, document)
        verify_training_attestation(model, postflight)
        written.append(path)
    return written


def run_attested_training(
    *,
    dataset_root: Path,
    repo_id: str,
    controller_contract: Path,
    expected_task: str,
    output_dir: Path,
    smolvla_base: Path,
    training_args: Sequence[str],
    python_executable: Path | str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> list[Path]:
    """Run stock LeRobot locally and finalize only safely matching checkpoints."""

    output_dir = _require_new_output_path(output_dir)
    preflight = capture_training_inputs(
        dataset_root,
        repo_id,
        controller_contract=controller_contract,
        expected_task=expected_task,
    )
    command = build_training_command(
        preflight,
        output_dir,
        training_args,
        smolvla_base=smolvla_base,
        python_executable=python_executable,
    )
    completed = runner(command, check=False)
    if completed.returncode != 0:
        raise TrainingAttestationError(
            f"LeRobot training exited with status {completed.returncode}; "
            "partial checkpoints remain unattested"
        )
    return finalize_training_attestations(output_dir, preflight)
