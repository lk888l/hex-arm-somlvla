"""Strict consumer for the fixed Firefly arm-only SmolVLA run-2 attestation.

This module is deliberately independent from the training producer.  It never
imports training code and its public capture entry point has no caller supplied
artifact paths.  A successful capture binds the fixed 10k/20k jointly committed
run, the exact 7-D to 6-D dataset projection, and the producer's read-only
``--verify-run`` result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts/train_firefly_smolvla_arm_only_v1_finetune_20k_run2.py"
)
MATERIALIZER_SCRIPT = REPOSITORY_ROOT / "scripts/materialize_firefly_y6_arm_only_v1.py"
EXPECTED_PRODUCER_SHA256 = (
    "6fc98ce48362d03db82747059ff2b989907618c30517c2a340a45c11213e32c1"
)
EXPECTED_MATERIALIZER_SHA256 = (
    "a6037746dd4087c11a2998104b7b5df00da3889cf5de157183c613a040ada2e7"
)
# Frozen out of band after the finished run-2 marker passed independent verification.
# A self-declared marker digest is not an attestation root: without this pin an
# attacker could rewrite a model, both children, and all of their self-hashes.
EXPECTED_RUN_COMMIT_SHA256: str | None = (
    "b514f9815f40e1dc1c831b154e8f346b479194afe386fd94210ba06b596757b1"
)

OUTPUT_DIR = Path(
    "/home/kk/kk_data/lerobot/outputs/train/"
    "firefly_y6_smolvla_fullcycle_v4_arm_only_v1_finetune_20k_attested_run2"
)
SOURCE_DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
DESTINATION_DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4_arm_only_v1"
)
PROJECTION_MANIFEST_RELATIVE = Path("meta/firefly_arm_only_v1_manifest.json")
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/controller_contract_validated_v2_arm0435_ee0165.json"
)

SOURCE_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4"
DESTINATION_DATASET_REPO_ID = "local/firefly_y6_fullcycle_v4_arm_only_v1"
EXPECTED_SOURCE_FILES_MANIFEST_SHA256 = (
    "0d7fa29d13e9826b293c8f397fc4627a64dc5199e82e2bf09dedde8b9a5cfcf4"
)
EXPECTED_SOURCE_DIRECTORIES_SHA256 = (
    "4dd16b2b62ef312e9a7542513f1ff43f8e587f70d4f71ea9fbee0fe9ea736e74"
)
EXPECTED_SOURCE_FILE_COUNT = 197
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 91836
EXPECTED_FPS = 30
EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
EXPECTED_UPSTREAM_INPUT_DIGEST = (
    "d07991b194092acb608ed1fbad7f73c1c4f6afc9b9a627bf5e23812f67921ca9"
)

POLICY_SCHEMA_ROLE = "arm_only_6d"
AUTHORIZED_RUNTIME_ROLE = "arm6_seeking"
LIMITATIONS = {
    "numeric_state_and_action_exclude_jaw": True,
    "source_videos_are_unchanged": True,
    "jaw_may_remain_visible_in_video_pixels": True,
    "not_a_claim_of_jaw_free_visual_observation": True,
}
CHILD_ATTESTATION_FILENAME = "firefly_arm_only_v1_finetune_attestation.json"
CHILD_ATTESTATION_KIND = "firefly_smolvla_arm_only_v1_finetune"
CHILD_TOKEN_PREFIX = "firefly-arm-only-v1-finetune-inputs-v1:"
RUN_COMMIT_FILENAME = "firefly_arm_only_v1_finetune_run_commit.json"
RUN_COMMIT_KIND = "firefly_smolvla_arm_only_v1_finetune_run_commit"
PROJECTION_KIND = "firefly_y6_arm_only_dataset_projection"
PROJECTION_VERSION = "exact_first_6_of_7_v1"
EXPECTED_CHECKPOINTS = ((10_000, "010000"), (20_000, "020000"))
SELECTED_STEP = 20_000
SELECTED_DIRECTORY = "020000"
REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "train_config.json",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024


class FireflyArm6AttestationError(RuntimeError):
    """The fixed arm-only policy/projection attestation failed closed."""


class Arm6AttestationChangedError(FireflyArm6AttestationError):
    """A controlled artifact changed during or between stable captures."""


@dataclass(frozen=True)
class ControlledArtifact:
    """One deployment-stability artifact captured by path and content."""

    role: str
    path: Path
    size: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FireflyArm6AttestationCapture:
    """Immutable, canonical binding suitable for deployment and audit capture."""

    selected_checkpoint: Path
    selected_step: int
    run_id: str
    core_input_digest: str
    input_digest: str
    run_commit_path: Path
    projection_manifest_path: Path
    producer_script_path: Path
    materializer_script_path: Path
    source_dataset_root: Path
    destination_dataset_root: Path
    _policy_json: bytes
    _projection_json: bytes
    _artifacts: tuple[ControlledArtifact, ...]

    def policy_attestation(self) -> dict[str, Any]:
        return json.loads(self._policy_json)

    def projection_binding(self) -> dict[str, Any]:
        return json.loads(self._projection_json)

    def source_binding(self) -> dict[str, Any]:
        return {
            "policy_attestation": self.policy_attestation(),
            "projection_binding": self.projection_binding(),
        }

    def controlled_artifacts(self) -> list[dict[str, Any]]:
        """Return canonical role-sorted records for deployment stability checks."""

        return [artifact.as_dict() for artifact in self._artifacts]


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _FileCapture:
    path: Path
    size: int
    sha256: str
    identity: _Identity
    data: bytes | None


@dataclass(frozen=True)
class _Inventory:
    files: tuple[dict[str, Any], ...]
    directories: tuple[str, ...]


@dataclass(frozen=True)
class _LocalSnapshot:
    policy: dict[str, Any]
    projection: dict[str, Any]
    expected_producer_result: dict[str, Any]
    artifacts: tuple[ControlledArtifact, ...]
    selected_checkpoint: Path
    run_id: str
    core_input_digest: str
    input_digest: str
    identities: tuple[tuple[str, _Identity], ...]
    directories: tuple[tuple[str, _Identity], ...]
    last_link_identity: _Identity

    def comparison_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "policy": self.policy,
                "projection": self.projection,
                "producer_result": self.expected_producer_result,
                "artifacts": [item.as_dict() for item in self.artifacts],
                "selected_checkpoint": str(self.selected_checkpoint),
                "run_id": self.run_id,
                "core_input_digest": self.core_input_digest,
                "input_digest": self.input_digest,
                "identities": [
                    [path, list(identity.__dict__.values())]
                    for path, identity in self.identities
                ],
                "directories": [
                    [path, list(identity.__dict__.values())]
                    for path, identity in self.directories
                ],
                "last_link_identity": list(self.last_link_identity.__dict__.values()),
            }
        )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FireflyArm6AttestationError(f"value is not strict JSON: {exc}") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _identity(path: Path, *, directory: bool | None = None) -> _Identity:
    try:
        item = path.lstat()
    except FileNotFoundError as exc:
        raise Arm6AttestationChangedError(f"controlled path disappeared: {path}") from exc
    if stat.S_ISLNK(item.st_mode):
        raise FireflyArm6AttestationError(f"controlled path is a symbolic link: {path}")
    if directory is True and not stat.S_ISDIR(item.st_mode):
        raise FireflyArm6AttestationError(f"controlled path is not a directory: {path}")
    if directory is False and not stat.S_ISREG(item.st_mode):
        raise FireflyArm6AttestationError(f"controlled path is not a regular file: {path}")
    if directory is False and item.st_nlink != 1:
        raise FireflyArm6AttestationError(
            f"controlled file must have exactly one hard link: {path}"
        )
    return _Identity(
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise Arm6AttestationChangedError(
                f"{label} path component is missing: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise FireflyArm6AttestationError(
                f"{label} contains a symbolic-link component: {current}"
            )
    return absolute


class _Registry:
    def __init__(self) -> None:
        self.files: dict[Path, _FileCapture] = {}
        self.directories: dict[Path, _Identity] = {}
        self.inodes: dict[tuple[int, int], Path] = {}

    def directory(self, path: Path, *, label: str) -> Path:
        absolute = _no_symlink_components(path, label=label)
        captured = _identity(absolute, directory=True)
        previous = self.directories.get(absolute)
        if previous is not None and previous != captured:
            raise Arm6AttestationChangedError(f"directory changed during capture: {absolute}")
        self.directories[absolute] = captured
        return absolute

    def file(
        self,
        path: Path,
        *,
        label: str,
        keep_data: bool = False,
        max_bytes: int | None = None,
    ) -> _FileCapture:
        absolute = _no_symlink_components(path, label=label)
        cached = self.files.get(absolute)
        if cached is not None and (not keep_data or cached.data is not None):
            return cached
        before = _identity(absolute, directory=False)
        inode_key = (before.device, before.inode)
        alias = self.inodes.get(inode_key)
        if alias is not None and alias != absolute:
            raise FireflyArm6AttestationError(
                f"controlled files must not be hard-linked aliases: {alias} and {absolute}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute, flags)
        except OSError as exc:
            raise Arm6AttestationChangedError(
                f"cannot open controlled file {absolute}: {exc}"
            ) from exc
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if keep_data else None
        total = 0
        try:
            opened = os.fstat(descriptor)
            opened_identity = _Identity(
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if opened_identity != before or not stat.S_ISREG(opened.st_mode):
                raise Arm6AttestationChangedError(f"file changed before read: {absolute}")
            while True:
                chunk = os.read(descriptor, _HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise FireflyArm6AttestationError(
                        f"controlled JSON file is unexpectedly large: {absolute}"
                    )
                if chunks is not None:
                    chunks.append(chunk)
            closed_identity_stat = os.fstat(descriptor)
            after = _Identity(
                closed_identity_stat.st_dev,
                closed_identity_stat.st_ino,
                closed_identity_stat.st_mode,
                closed_identity_stat.st_nlink,
                closed_identity_stat.st_size,
                closed_identity_stat.st_mtime_ns,
                closed_identity_stat.st_ctime_ns,
            )
            if after != before or total != before.size:
                raise Arm6AttestationChangedError(f"file changed while read: {absolute}")
        finally:
            os.close(descriptor)
        if _identity(absolute, directory=False) != before:
            raise Arm6AttestationChangedError(f"file was replaced while read: {absolute}")
        capture = _FileCapture(
            absolute,
            before.size,
            digest.hexdigest(),
            before,
            b"".join(chunks) if chunks is not None else None,
        )
        if cached is not None and (
            cached.identity != capture.identity
            or cached.size != capture.size
            or cached.sha256 != capture.sha256
        ):
            raise Arm6AttestationChangedError(
                f"controlled file changed between reads: {absolute}"
            )
        self.files[absolute] = capture
        self.inodes[inode_key] = absolute
        return capture

    def json(self, path: Path, *, label: str) -> tuple[dict[str, Any], _FileCapture]:
        captured = self.file(
            path, label=label, keep_data=True, max_bytes=_MAX_JSON_BYTES
        )
        assert captured.data is not None

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number {value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        try:
            value = json.loads(
                captured.data.decode("utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise FireflyArm6AttestationError(f"invalid strict JSON in {label}: {exc}") from exc
        if not isinstance(value, dict):
            raise FireflyArm6AttestationError(f"{label} must contain one JSON object")
        _canonical_json_bytes(value)
        return value, captured

    def finish(self) -> tuple[tuple[tuple[str, _Identity], ...], tuple[tuple[str, _Identity], ...]]:
        for path, capture in self.files.items():
            if _identity(path, directory=False) != capture.identity:
                raise Arm6AttestationChangedError(f"controlled file changed: {path}")
        for path, capture in self.directories.items():
            if _identity(path, directory=True) != capture:
                raise Arm6AttestationChangedError(f"controlled directory changed: {path}")
        return (
            tuple(sorted((str(path), item.identity) for path, item in self.files.items())),
            tuple(sorted((str(path), item) for path, item in self.directories.items())),
        )


def _exact(actual: Any, expected: Any, *, label: str) -> None:
    if _canonical_json_bytes(actual) != _canonical_json_bytes(expected):
        raise FireflyArm6AttestationError(f"{label} does not match the fixed contract")


def _keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FireflyArm6AttestationError(f"{label} keys drifted")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FireflyArm6AttestationError(f"{label} must be a lowercase SHA-256")
    return value


def _frozen_run_commit_sha256() -> str:
    if EXPECTED_RUN_COMMIT_SHA256 is None:
        raise FireflyArm6AttestationError(
            "fixed run2 marker SHA-256 has not been frozen; refusing before reading run2 artifacts"
        )
    return _sha(EXPECTED_RUN_COMMIT_SHA256, label="fixed run2 marker pin")


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FireflyArm6AttestationError(f"{label} must be an integer")
    return value


def _safe_manifest_files(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FireflyArm6AttestationError(f"{label} must be a list")
    result: list[dict[str, Any]] = []
    previous = ""
    for index, item in enumerate(value):
        record = _keys(item, {"path", "size", "sha256"}, label=f"{label}[{index}]")
        path = record.get("path")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or path <= previous
        ):
            raise FireflyArm6AttestationError(
                f"{label} paths must be safe, unique, and sorted"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise FireflyArm6AttestationError(f"{label}[{index}].size is invalid")
        result.append({"path": path, "size": size, "sha256": _sha(digest, label=label)})
        previous = path
    return result


def _safe_directories(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise FireflyArm6AttestationError(f"{label} must be a list")
    result: list[str] = []
    previous = ""
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item.startswith("/")
            or ".." in Path(item).parts
            or item <= previous
        ):
            raise FireflyArm6AttestationError(
                f"{label} paths must be safe, unique, and sorted"
            )
        result.append(item)
        previous = item
    return result


def _inventory(
    registry: _Registry,
    root: Path,
    *,
    label: str,
    excluded: frozenset[str] = frozenset(),
) -> _Inventory:
    root = registry.directory(root, label=label)
    try:
        root_entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        raise FireflyArm6AttestationError(f"cannot enumerate {label}: {exc}") from exc
    if [entry.name for entry in root_entries] != ["data", "meta", "videos"]:
        raise FireflyArm6AttestationError(
            f"{label} root must contain exactly data, meta, and videos"
        )
    records: list[dict[str, Any]] = []
    directories: list[str] = []

    def visit(directory: Path) -> None:
        directory = registry.directory(directory, label=label)
        relative_directory = directory.relative_to(root).as_posix()
        directories.append(relative_directory)
        before = registry.directories[directory]
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise FireflyArm6AttestationError(f"cannot enumerate {label}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise FireflyArm6AttestationError(f"{label} contains a symbolic link: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                relative = path.relative_to(root).as_posix()
                capture = registry.file(path, label=label)
                if relative not in excluded:
                    records.append(
                        {"path": relative, "size": capture.size, "sha256": capture.sha256}
                    )
            else:
                raise FireflyArm6AttestationError(f"{label} contains a non-regular entry: {path}")
        if _identity(directory, directory=True) != before:
            raise Arm6AttestationChangedError(f"{label} changed while enumerated: {directory}")

    for name in ("meta", "data", "videos"):
        visit(root / name)
    records.sort(key=lambda item: item["path"])
    directories.sort()
    return _Inventory(tuple(records), tuple(directories))


def _expected_transform() -> dict[str, Any]:
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
        "retained_feature_names": [f"joint_{index}.pos" for index in range(1, 7)],
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


def _expected_smoke() -> dict[str, Any]:
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


def _validate_policy_config(document: Mapping[str, Any]) -> None:
    if document.get("type") != "smolvla":
        raise FireflyArm6AttestationError("child policy must be SmolVLA")
    inputs = document.get("input_features")
    outputs = document.get("output_features")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise FireflyArm6AttestationError("child policy feature schemas are missing")
    expected_inputs = {
        "observation.state",
        "observation.images.scene",
        "observation.images.wrist",
    }
    if set(inputs) != expected_inputs:
        raise FireflyArm6AttestationError(
            "child policy must consume only J1-J6 state plus scene/wrist video"
        )
    if set(outputs) != {"action"}:
        raise FireflyArm6AttestationError("child policy must emit only action")
    state = inputs["observation.state"]
    action = outputs["action"]
    if not isinstance(state, Mapping) or not isinstance(action, Mapping):
        raise FireflyArm6AttestationError("child policy numeric feature schemas are invalid")
    _exact(state.get("type"), "STATE", label="child policy state role")
    _exact(state.get("shape"), [6], label="child policy state dimension")
    _exact(action.get("type"), "ACTION", label="child policy action role")
    _exact(action.get("shape"), [6], label="child policy action dimension")
    for camera in ("observation.images.scene", "observation.images.wrist"):
        feature = inputs[camera]
        if not isinstance(feature, Mapping):
            raise FireflyArm6AttestationError(
                f"child policy {camera} schema is invalid"
            )
        _exact(feature.get("type"), "VISUAL", label=f"child policy {camera} role")
        _exact(
            feature.get("shape"),
            [3, 480, 640],
            label=f"child policy {camera} shape",
        )


def _validate_projection(
    registry: _Registry,
) -> tuple[dict[str, Any], _FileCapture, _Inventory, _Inventory]:
    source_root = registry.directory(SOURCE_DATASET_ROOT, label="source dataset")
    destination_root = registry.directory(DESTINATION_DATASET_ROOT, label="arm-only dataset")
    if (
        source_root == destination_root
        or source_root in destination_root.parents
        or destination_root in source_root.parents
    ):
        raise FireflyArm6AttestationError("source and destination dataset roots must be disjoint")
    source_inventory = _inventory(registry, source_root, label="source dataset")
    destination_inventory = _inventory(
        registry,
        destination_root,
        label="arm-only dataset",
        excluded=frozenset({PROJECTION_MANIFEST_RELATIVE.as_posix()}),
    )
    manifest, manifest_file = registry.json(
        destination_root / PROJECTION_MANIFEST_RELATIVE,
        label="arm-only projection manifest",
    )
    _keys(
        manifest,
        {
            "schema_version",
            "kind",
            "projection_version",
            "source",
            "destination",
            "transform",
            "controller_contract",
            "verification",
            "document_sha256",
        },
        label="projection manifest",
    )
    _exact(manifest.get("schema_version"), 1, label="projection schema_version")
    _exact(manifest.get("kind"), PROJECTION_KIND, label="projection kind")
    _exact(manifest.get("projection_version"), PROJECTION_VERSION, label="projection version")
    unsigned = dict(manifest)
    claimed_document_sha = _sha(unsigned.pop("document_sha256", None), label="projection document_sha256")
    _exact(claimed_document_sha, _sha256_json(unsigned), label="projection self digest")

    common = {
        "repo_id",
        "root",
        "episode_count",
        "frame_count",
        "fps",
        "task",
        "files_manifest_sha256",
        "files",
        "directories_sha256",
        "directories",
    }
    source = _keys(manifest.get("source"), common | {"upstream_training_input_digest"}, label="projection source")
    destination = _keys(manifest.get("destination"), common, label="projection destination")
    expected_common = {
        "episode_count": EXPECTED_EPISODES,
        "frame_count": EXPECTED_FRAMES,
        "fps": EXPECTED_FPS,
        "task": EXPECTED_TASK,
    }
    for field, expected in {
        **expected_common,
        "repo_id": SOURCE_DATASET_REPO_ID,
        "root": str(source_root),
        "upstream_training_input_digest": EXPECTED_UPSTREAM_INPUT_DIGEST,
        "files_manifest_sha256": EXPECTED_SOURCE_FILES_MANIFEST_SHA256,
        "directories_sha256": EXPECTED_SOURCE_DIRECTORIES_SHA256,
    }.items():
        _exact(source.get(field), expected, label=f"projection source.{field}")
    for field, expected in {
        **expected_common,
        "repo_id": DESTINATION_DATASET_REPO_ID,
        "root": str(destination_root),
    }.items():
        _exact(destination.get(field), expected, label=f"projection destination.{field}")

    source_files = _safe_manifest_files(source.get("files"), label="projection source.files")
    destination_files = _safe_manifest_files(destination.get("files"), label="projection destination.files")
    source_directories = _safe_directories(source.get("directories"), label="projection source.directories")
    destination_directories = _safe_directories(destination.get("directories"), label="projection destination.directories")
    if len(source_files) != EXPECTED_SOURCE_FILE_COUNT:
        raise FireflyArm6AttestationError("projection source file count drifted")
    _exact(source_files, list(source_inventory.files), label="projection complete source files")
    _exact(destination_files, list(destination_inventory.files), label="projection complete destination files")
    _exact(source_directories, list(source_inventory.directories), label="projection complete source directories")
    _exact(destination_directories, list(destination_inventory.directories), label="projection complete destination directories")
    _exact(destination_directories, source_directories, label="projection one-to-one directories")
    _exact([item["path"] for item in destination_files], [item["path"] for item in source_files], label="projection one-to-one file paths")
    _exact(source.get("files_manifest_sha256"), _sha256_json(source_files), label="projection source files digest")
    _exact(destination.get("files_manifest_sha256"), _sha256_json(destination_files), label="projection destination files digest")
    _exact(source.get("directories_sha256"), _sha256_json(source_directories), label="projection source directories digest")
    _exact(destination.get("directories_sha256"), _sha256_json(destination_directories), label="projection destination directories digest")

    _exact(manifest.get("transform"), _expected_transform(), label="projection exact 7-to-6 transform")
    _exact(
        manifest.get("verification"),
        {
            "exact_first_six_float32_bits": True,
            "one_to_one_file_paths": True,
            "copied_files_byte_identical": True,
            "destination_files_independent_regular_single_link": True,
            "source_unchanged_before_publish": True,
        },
        label="projection verification",
    )

    controller = _keys(
        manifest.get("controller_contract"),
        {
            "path",
            "size",
            "sha256",
            "retained_arm_effective_step_limits_rad",
            "removed_jaw_effective_step_limit_rad",
        },
        label="projection controller contract",
    )
    _exact(controller.get("path"), str(_absolute(CONTROLLER_CONTRACT)), label="controller path")
    controller_document, controller_file = registry.json(CONTROLLER_CONTRACT, label="controller contract")
    _exact(controller.get("size"), controller_file.size, label="controller size")
    _exact(controller.get("sha256"), controller_file.sha256, label="controller sha256")
    limits = controller_document.get("effective_step_limits")
    if (
        not isinstance(limits, list)
        or len(limits) != 7
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
            for item in limits
        )
    ):
        raise FireflyArm6AttestationError("controller effective step limits are invalid")
    _exact(controller.get("retained_arm_effective_step_limits_rad"), [float(item) for item in limits[:6]], label="controller retained arm limits")
    _exact(controller.get("removed_jaw_effective_step_limit_rad"), float(limits[6]), label="controller removed jaw limit")

    source_by_path = {item["path"]: item for item in source_files}
    destination_by_path = {item["path"]: item for item in destination_files}
    videos = [item for item in source_files if item["path"].startswith("videos/")]
    if not videos or not any("observation.images.scene" in item["path"] for item in videos) or not any("observation.images.wrist" in item["path"] for item in videos):
        raise FireflyArm6AttestationError("projection must bind both scene and wrist videos")
    for source_record in videos:
        _exact(destination_by_path[source_record["path"]], source_record, label="projection video byte identity")
    video_digest = _sha256_json(videos)

    materializer = registry.file(MATERIALIZER_SCRIPT, label="frozen materializer")
    _exact(materializer.sha256, EXPECTED_MATERIALIZER_SHA256, label="frozen materializer sha256")
    projection_binding = {
        "schema_version": 1,
        "kind": PROJECTION_KIND,
        "projection_version": PROJECTION_VERSION,
        "manifest": {
            "path": str(manifest_file.path),
            "size": manifest_file.size,
            "sha256": manifest_file.sha256,
            "document_sha256": claimed_document_sha,
        },
        "materializer": {
            "path": str(materializer.path),
            "size": materializer.size,
            "sha256": materializer.sha256,
        },
        "source": {
            "repo_id": source["repo_id"],
            "root": source["root"],
            "upstream_training_input_digest": source[
                "upstream_training_input_digest"
            ],
            "files_manifest_sha256": source["files_manifest_sha256"],
            "directories_sha256": source["directories_sha256"],
            "file_count": len(source_files),
            "directory_count": len(source_directories),
        },
        "destination": {
            "repo_id": destination["repo_id"],
            "root": destination["root"],
            "files_manifest_sha256": destination["files_manifest_sha256"],
            "directories_sha256": destination["directories_sha256"],
            "file_count": len(destination_files),
            "directory_count": len(destination_directories),
        },
        "transform": copy.deepcopy(manifest["transform"]),
        "controller_contract": copy.deepcopy(controller),
        "verification": copy.deepcopy(manifest["verification"]),
        "video_byte_identity": {
            "verified": True,
            "file_count": len(videos),
            "files_sha256": video_digest,
        },
    }
    return projection_binding, manifest_file, source_inventory, destination_inventory


def _validate_child_inputs(
    inputs: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_file: _FileCapture,
) -> tuple[str, str]:
    input_digest = _sha256_json(inputs)
    if "command" not in inputs or not isinstance(inputs.get("command"), list):
        raise FireflyArm6AttestationError("child inputs must bind the training command")
    core = dict(inputs)
    core.pop("command")
    core_digest = _sha256_json(core)
    _exact(inputs.get("schema_version"), 1, label="child inputs schema_version")
    _exact(inputs.get("kind"), CHILD_ATTESTATION_KIND, label="child inputs kind")
    contract = inputs.get("training_contract")
    if not isinstance(contract, Mapping):
        raise FireflyArm6AttestationError("child inputs training_contract is missing")
    for field, expected in {
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "observation_limitations": LIMITATIONS,
        "output_dir": str(_absolute(OUTPUT_DIR)),
        "job_name": OUTPUT_DIR.name,
        "steps": 20_000,
        "save_freq": 10_000,
    }.items():
        _exact(contract.get(field), expected, label=f"child inputs training_contract.{field}")
    dataset = inputs.get("dataset")
    if not isinstance(dataset, Mapping):
        raise FireflyArm6AttestationError("child inputs dataset binding is missing")
    _exact(dataset.get("repo_id"), DESTINATION_DATASET_REPO_ID, label="child input dataset repo")
    _exact(dataset.get("root"), str(_absolute(DESTINATION_DATASET_ROOT)), label="child input dataset root")
    arm_manifest = dataset.get("arm_only_manifest")
    if not isinstance(arm_manifest, Mapping):
        raise FireflyArm6AttestationError("child inputs projection binding is missing")
    _exact(
        arm_manifest,
        {
            "path": PROJECTION_MANIFEST_RELATIVE.as_posix(),
            "size": manifest_file.size,
            "sha256": manifest_file.sha256,
            "document": manifest,
        },
        label="child inputs complete projection manifest binding",
    )
    return core_digest, input_digest


def _validate_run(registry: _Registry, projection: dict[str, Any], manifest_file: _FileCapture) -> tuple[dict[str, Any], tuple[ControlledArtifact, ...], Path, dict[str, Any]]:
    output_dir = registry.directory(OUTPUT_DIR, label="fixed run2 output")
    checkpoints = registry.directory(output_dir / "checkpoints", label="fixed run2 checkpoints")
    try:
        entries = sorted(os.scandir(checkpoints), key=lambda item: item.name)
    except OSError as exc:
        raise FireflyArm6AttestationError(f"cannot enumerate fixed checkpoints: {exc}") from exc
    if [entry.name for entry in entries] != ["010000", "020000", "last"]:
        raise FireflyArm6AttestationError("fixed run2 must contain exactly 010000, 020000, and last")
    for entry in entries[:2]:
        if stat.S_ISLNK(entry.stat(follow_symlinks=False).st_mode) or not entry.is_dir(follow_symlinks=False):
            raise FireflyArm6AttestationError(f"checkpoint is not a direct directory: {entry.path}")
    last = Path(entries[2].path)
    last_stat = last.lstat()
    if not stat.S_ISLNK(last_stat.st_mode) or os.readlink(last) != SELECTED_DIRECTORY:
        raise FireflyArm6AttestationError("checkpoints/last must be the canonical 020000 pointer")
    last_identity = _Identity(last_stat.st_dev, last_stat.st_ino, last_stat.st_mode, last_stat.st_nlink, last_stat.st_size, last_stat.st_mtime_ns, last_stat.st_ctime_ns)

    marker, marker_file = registry.json(output_dir / RUN_COMMIT_FILENAME, label="fixed run2 marker")
    _exact(
        marker_file.sha256,
        _frozen_run_commit_sha256(),
        label="fixed run2 marker file sha256",
    )
    _keys(marker, {"schema_version", "kind", "policy_schema_role", "authorized_runtime_role", "limitations", "token", "core_input_digest", "input_digest", "run_id", "last_target", "children", "children_sha256", "document_sha256"}, label="fixed run2 marker")
    unsigned = dict(marker)
    marker_document_sha = _sha(unsigned.pop("document_sha256", None), label="run marker document_sha256")
    _exact(marker_document_sha, _sha256_json(unsigned), label="run marker self digest")
    for field, expected in {
        "schema_version": 1,
        "kind": RUN_COMMIT_KIND,
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": LIMITATIONS,
        "last_target": SELECTED_DIRECTORY,
    }.items():
        _exact(marker.get(field), expected, label=f"run marker {field}")
    core_digest = _sha(marker.get("core_input_digest"), label="run marker core_input_digest")
    input_digest = _sha(marker.get("input_digest"), label="run marker input_digest")
    _exact(marker.get("run_id"), input_digest, label="run marker run_id")
    _exact(marker.get("token"), f"{CHILD_TOKEN_PREFIX}{core_digest}", label="run marker token")
    children = marker.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise FireflyArm6AttestationError("run marker must bind exactly two children")
    _exact(marker.get("children_sha256"), _sha256_json(children), label="run marker children digest")

    policy_children: list[dict[str, Any]] = []
    producer_children: list[dict[str, Any]] = []
    artifacts: list[ControlledArtifact] = []
    common_inputs: Mapping[str, Any] | None = None
    model_digests: set[str] = set()
    for index, (step, directory) in enumerate(EXPECTED_CHECKPOINTS):
        record = _keys(children[index], {"step", "directory", "attestation_path", "attestation_size", "attestation_sha256", "checkpoint_files_sha256", "model_sha256", "training_step", "offline_load_smoke"}, label=f"run marker child {step}")
        _exact(record.get("step"), step, label=f"run marker child {step} step")
        _exact(record.get("directory"), directory, label=f"run marker child {step} directory")
        model_root = registry.directory(checkpoints / directory / "pretrained_model", label=f"child {step} model")
        registry.directory(checkpoints / directory / "training_state", label=f"child {step} training state")
        attestation_relative = f"checkpoints/{directory}/pretrained_model/{CHILD_ATTESTATION_FILENAME}"
        _exact(record.get("attestation_path"), attestation_relative, label=f"child {step} attestation path")
        child, child_file = registry.json(model_root / CHILD_ATTESTATION_FILENAME, label=f"child {step} attestation")
        _exact(record.get("attestation_size"), child_file.size, label=f"child {step} attestation size")
        _exact(record.get("attestation_sha256"), child_file.sha256, label=f"child {step} attestation sha256")
        _keys(child, {"schema_version", "kind", "policy_schema_role", "authorized_runtime_role", "limitations", "token", "core_input_digest", "input_digest", "inputs", "run_commit", "checkpoint", "verification"}, label=f"child {step} attestation")
        for field, expected in {
            "schema_version": 1,
            "kind": CHILD_ATTESTATION_KIND,
            "policy_schema_role": POLICY_SCHEMA_ROLE,
            "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
            "limitations": LIMITATIONS,
            "token": marker["token"],
            "core_input_digest": core_digest,
            "input_digest": input_digest,
        }.items():
            _exact(child.get(field), expected, label=f"child {step} {field}")
        _exact(child.get("run_commit"), {"filename": RUN_COMMIT_FILENAME, "run_id": input_digest}, label=f"child {step} run commit reference")
        _exact(child.get("verification"), {"training_returncode": 0, "inputs_unchanged_after_training": True, "postflight_input_digest": input_digest}, label=f"child {step} verification")
        inputs = child.get("inputs")
        if not isinstance(inputs, Mapping):
            raise FireflyArm6AttestationError(f"child {step} inputs are missing")
        recomputed_core, recomputed_input = _validate_child_inputs(inputs, manifest=projection["_manifest"], manifest_file=manifest_file)
        _exact(recomputed_core, core_digest, label=f"child {step} recomputed core digest")
        _exact(recomputed_input, input_digest, label=f"child {step} recomputed input digest")
        if common_inputs is None:
            common_inputs = inputs
        else:
            _exact(inputs, common_inputs, label="10k/20k child input binding")

        checkpoint = _keys(child.get("checkpoint"), {"step", "files", "files_sha256"}, label=f"child {step} checkpoint")
        _exact(checkpoint.get("step"), step, label=f"child {step} checkpoint step")
        files = _safe_manifest_files(checkpoint.get("files"), label=f"child {step} checkpoint.files")
        if {item["path"] for item in files} != REQUIRED_CHECKPOINT_FILES or any("/" in item["path"] for item in files):
            raise FireflyArm6AttestationError(f"child {step} checkpoint artifact set drifted")
        try:
            model_entries = sorted(os.scandir(model_root), key=lambda item: item.name)
        except OSError as exc:
            raise FireflyArm6AttestationError(
                f"cannot enumerate child {step} checkpoint: {exc}"
            ) from exc
        expected_model_entries = sorted(
            {*REQUIRED_CHECKPOINT_FILES, CHILD_ATTESTATION_FILENAME}
        )
        if [entry.name for entry in model_entries] != expected_model_entries:
            raise FireflyArm6AttestationError(
                f"child {step} checkpoint tree must be the exact regular artifact set"
            )
        for entry in model_entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise FireflyArm6AttestationError(
                    f"child {step} checkpoint contains a symbolic-link artifact: {entry.path}"
                )
            if not stat.S_ISREG(mode):
                raise FireflyArm6AttestationError(
                    f"child {step} checkpoint contains a non-regular artifact: {entry.path}"
                )
        actual_files: list[dict[str, Any]] = []
        for item in files:
            captured = registry.file(model_root / item["path"], label=f"child {step} checkpoint file")
            actual_files.append({"path": item["path"], "size": captured.size, "sha256": captured.sha256})
        _exact(files, actual_files, label=f"child {step} complete checkpoint files")
        checkpoint_digest = _sha256_json(files)
        _exact(checkpoint.get("files_sha256"), checkpoint_digest, label=f"child {step} checkpoint files digest")
        _exact(record.get("checkpoint_files_sha256"), checkpoint_digest, label=f"run marker child {step} checkpoint files digest")
        model_record = next(item for item in files if item["path"] == "model.safetensors")
        _exact(record.get("model_sha256"), model_record["sha256"], label=f"run marker child {step} model sha256")
        model_digests.add(model_record["sha256"])
        config_document, _config_file = registry.json(model_root / "config.json", label=f"child {step} policy config")
        _validate_policy_config(config_document)

        training_record = _keys(record.get("training_step"), {"path", "size", "sha256", "document"}, label=f"child {step} training step")
        _exact(training_record.get("path"), "training_state/training_step.json", label=f"child {step} training step path")
        training_document, training_file = registry.json(checkpoints / directory / "training_state/training_step.json", label=f"child {step} training step")
        expected_training_document = {"step": step, "num_processes": 1, "batch_size": 1}
        _exact(training_document, expected_training_document, label=f"child {step} training step document")
        _exact(training_record, {"path": "training_state/training_step.json", "size": training_file.size, "sha256": training_file.sha256, "document": expected_training_document}, label=f"child {step} training step binding")
        _exact(record.get("offline_load_smoke"), _expected_smoke(), label=f"child {step} SmolVLA 6-D smoke")

        artifacts.extend([
            ControlledArtifact(f"child_attestation_{directory}", child_file.path, child_file.size, child_file.sha256),
            ControlledArtifact(f"model_{directory}", model_root / "model.safetensors", model_record["size"], model_record["sha256"]),
        ])
        policy_child = {
            "step": step,
            "directory": directory,
            "pretrained_model": str(model_root),
            "attestation": {"path": str(child_file.path), "size": child_file.size, "sha256": child_file.sha256},
            "checkpoint_files_sha256": checkpoint_digest,
            "model": {"path": str(model_root / "model.safetensors"), "size": model_record["size"], "sha256": model_record["sha256"]},
        }
        policy_children.append(policy_child)
        producer_children.append({
            "step": step,
            "checkpoint_directory": directory,
            "pretrained_model": str(model_root),
            "attestation_size": child_file.size,
            "attestation_sha256": child_file.sha256,
            "checkpoint_files_sha256": checkpoint_digest,
            "model_sha256": model_record["sha256"],
        })
    if len(model_digests) != 2:
        raise FireflyArm6AttestationError("010000 and 020000 model weights must be distinct")

    producer = registry.file(PRODUCER_SCRIPT, label="frozen run2 producer")
    _exact(producer.sha256, EXPECTED_PRODUCER_SHA256, label="frozen producer sha256")
    materializer = registry.file(MATERIALIZER_SCRIPT, label="frozen materializer")
    selected = policy_children[1]
    run_commit_binding = {
        "path": str(marker_file.path),
        "size": marker_file.size,
        "sha256": marker_file.sha256,
        "document_sha256": marker_document_sha,
        "children_sha256": marker["children_sha256"],
    }
    expected_producer = {
        "schema_version": 1,
        "kind": "firefly_smolvla_arm_only_v1_finetune_verification_result",
        "status": "verified-read-only-no-training-started",
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": LIMITATIONS,
        "output_dir": str(output_dir),
        "core_input_digest": core_digest,
        "input_digest": input_digest,
        "run_commit": {"path": str(marker_file.path), "size": marker_file.size, "sha256": marker_file.sha256},
        "last_target": SELECTED_DIRECTORY,
        "children": producer_children,
    }
    policy = {
        "schema_version": 1,
        "kind": "firefly_smolvla_arm_only_v1_attestation_capture",
        "policy_schema_role": POLICY_SCHEMA_ROLE,
        "authorized_runtime_role": AUTHORIZED_RUNTIME_ROLE,
        "limitations": copy.deepcopy(LIMITATIONS),
        "run_id": input_digest,
        "core_input_digest": core_digest,
        "input_digest": input_digest,
        "run_commit": run_commit_binding,
        "children": policy_children,
        "selected_checkpoint": {
            "step": selected["step"],
            "directory": selected["directory"],
            "pretrained_model": selected["pretrained_model"],
            "attestation_sha256": selected["attestation"]["sha256"],
            "checkpoint_files_sha256": selected["checkpoint_files_sha256"],
            "model_sha256": selected["model"]["sha256"],
        },
        "producer_verify_run": {
            "script": {"path": str(producer.path), "size": producer.size, "sha256": producer.sha256},
            "status": "verified-read-only-no-training-started",
            "result_sha256": "<BOUND_AFTER_VERIFY>",
        },
    }
    artifacts.extend([
        ControlledArtifact("materializer_script", materializer.path, materializer.size, materializer.sha256),
        ControlledArtifact("producer_script", producer.path, producer.size, producer.sha256),
        ControlledArtifact("projection_manifest", manifest_file.path, manifest_file.size, manifest_file.sha256),
        ControlledArtifact("run_commit", marker_file.path, marker_file.size, marker_file.sha256),
    ])
    artifacts.sort(key=lambda item: (item.role, str(item.path)))
    return policy, tuple(artifacts), Path(selected["pretrained_model"]), {
        "result": expected_producer,
        "last_identity": last_identity,
        "last_path": last,
    }


def _capture_local() -> _LocalSnapshot:
    # Check the out-of-band root of trust before touching unfinished run2 or
    # projection artifacts.
    _frozen_run_commit_sha256()
    registry = _Registry()
    projection, manifest_file, _source_inventory, _destination_inventory = _validate_projection(registry)
    manifest, _ = registry.json(manifest_file.path, label="arm-only projection manifest")
    projection_with_private = {**projection, "_manifest": manifest}
    policy, artifacts, selected, extra = _validate_run(registry, projection_with_private, manifest_file)
    identities, directories = registry.finish()
    current_last = extra["last_path"].lstat()
    current_last_identity = _Identity(
        current_last.st_dev,
        current_last.st_ino,
        current_last.st_mode,
        current_last.st_nlink,
        current_last.st_size,
        current_last.st_mtime_ns,
        current_last.st_ctime_ns,
    )
    if (
        not stat.S_ISLNK(current_last.st_mode)
        or os.readlink(extra["last_path"]) != SELECTED_DIRECTORY
        or current_last_identity != extra["last_identity"]
    ):
        raise Arm6AttestationChangedError(
            "canonical checkpoints/last pointer changed during capture"
        )
    return _LocalSnapshot(
        policy=policy,
        projection=projection,
        expected_producer_result=extra["result"],
        artifacts=artifacts,
        selected_checkpoint=selected,
        run_id=policy["run_id"],
        core_input_digest=policy["core_input_digest"],
        input_digest=policy["input_digest"],
        identities=identities,
        directories=directories,
        last_link_identity=extra["last_identity"],
    )


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_frozen_producer(runner: Runner) -> dict[str, Any]:
    command = [sys.executable, "-I", "-s", str(_absolute(PRODUCER_SCRIPT)), "--verify-run"]
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    for name in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_absolute(REPOSITORY_ROOT)),
            env=environment,
            timeout=1200,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FireflyArm6AttestationError(f"frozen producer --verify-run could not run: {exc}") from exc
    if result.returncode != 0:
        raise FireflyArm6AttestationError(
            f"frozen producer --verify-run refused with status {result.returncode}: {result.stderr[-2000:]}"
        )
    def reject_constant(item: str) -> None:
        raise ValueError(item)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            result.stdout,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise FireflyArm6AttestationError("frozen producer returned invalid strict JSON") from exc
    if not isinstance(value, dict):
        raise FireflyArm6AttestationError("frozen producer returned a non-object")
    _canonical_json_bytes(value)
    return value


def capture_firefly_arm6_attestation(
    *, runner: Runner = subprocess.run
) -> FireflyArm6AttestationCapture:
    """Capture the one fixed run-2 arm-only policy and projection binding.

    The producer verification is mandatory and read-only.  Artifact paths are
    intentionally not parameters: callers cannot redirect this trust decision
    to another run or select the 10k child.
    """

    before = _capture_local()
    producer_result = _run_frozen_producer(runner)
    _exact(producer_result, before.expected_producer_result, label="frozen producer verification result")
    after = _capture_local()
    if before.comparison_bytes() != after.comparison_bytes():
        raise Arm6AttestationChangedError(
            "controlled run/projection artifacts changed across producer verification"
        )
    _exact(producer_result, after.expected_producer_result, label="post-verification producer result")
    result_sha = _sha256_json(producer_result)
    policy = copy.deepcopy(after.policy)
    policy["producer_verify_run"]["result_sha256"] = result_sha
    policy_json = _canonical_json_bytes(policy)
    projection_json = _canonical_json_bytes(after.projection)
    return FireflyArm6AttestationCapture(
        selected_checkpoint=after.selected_checkpoint,
        selected_step=SELECTED_STEP,
        run_id=after.run_id,
        core_input_digest=after.core_input_digest,
        input_digest=after.input_digest,
        run_commit_path=_absolute(OUTPUT_DIR) / RUN_COMMIT_FILENAME,
        projection_manifest_path=_absolute(DESTINATION_DATASET_ROOT) / PROJECTION_MANIFEST_RELATIVE,
        producer_script_path=_absolute(PRODUCER_SCRIPT),
        materializer_script_path=_absolute(MATERIALIZER_SCRIPT),
        source_dataset_root=_absolute(SOURCE_DATASET_ROOT),
        destination_dataset_root=_absolute(DESTINATION_DATASET_ROOT),
        _policy_json=policy_json,
        _projection_json=projection_json,
        _artifacts=after.artifacts,
    )


__all__ = [
    "Arm6AttestationChangedError",
    "ControlledArtifact",
    "FireflyArm6AttestationCapture",
    "FireflyArm6AttestationError",
    "capture_firefly_arm6_attestation",
]
