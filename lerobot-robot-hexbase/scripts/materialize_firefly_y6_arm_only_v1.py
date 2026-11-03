#!/usr/bin/env python3
"""Materialize and validate the pinned Firefly fullcycle-v4 arm-only dataset.

This is an offline, lossless projection of the already attested 7-DoF LeRobot
v3 dataset.  ``action`` and ``observation.state`` are projected from seven to
six float32 values by removing only index 6 (``jaw_joint1.pos``).  Frames,
episodes, tasks, timestamps, images, and the first six values are unchanged.

The default command is read-only and audits the fixed source.  Publishing the
fixed destination requires the explicit ``--materialize`` flag.  A real run
writes a temporary sibling, validates the complete source-to-target mapping,
and publishes it with Linux ``renameat2(RENAME_NOREPLACE)``.  It never rewrites
the source and never overwrites an existing destination.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO_ID = "local/firefly_y6_fullcycle_v4"
SOURCE_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_fullcycle_v4"
)
DESTINATION_REPO_ID = "local/firefly_y6_fullcycle_v4_arm_only_v1"
DESTINATION_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/"
    "firefly_y6_fullcycle_v4_arm_only_v1"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
MANIFEST_RELATIVE_PATH = Path("meta/firefly_arm_only_v1_manifest.json")

EXPECTED_TASK = (
    "Find the red block, pick it up, place it in the blue box, release it, "
    "and return the arm to its initial pose."
)
JOINT_NAMES_7 = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "jaw_joint1.pos",
)
ARM_JOINT_NAMES = JOINT_NAMES_7[:6]
PROJECTED_FEATURES = ("action", "observation.state")
VECTOR_STATISTICS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")

MANIFEST_KIND = "firefly_y6_arm_only_dataset_projection"
MANIFEST_SCHEMA_VERSION = 1
PROJECTION_VERSION = "exact_first_6_of_7_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "0d7fa29d13e9826b293c8f397fc4627a64dc5199e82e2bf09dedde8b9a5cfcf4"
)
EXPECTED_SOURCE_DIRECTORIES_SHA256 = (
    "4dd16b2b62ef312e9a7542513f1ff43f8e587f70d4f71ea9fbee0fe9ea736e74"
)
EXPECTED_CONTROLLER_CONTRACT_SHA256 = (
    "8a5526cea85fc7d6e9e542564683521b0599f80b3fd752712a105d011e1b2d19"
)
UPSTREAM_TRAINING_INPUT_DIGEST = (
    "d07991b194092acb608ed1fbad7f73c1c4f6afc9b9a627bf5e23812f67921ca9"
)
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 91836
EXPECTED_FPS = 30
EXPECTED_SOURCE_FILES = 197
EXPECTED_DATA_FILES = 50
EXPECTED_EPISODE_METADATA_FILES = 44
EXPECTED_VIDEO_FILES = 100

_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_JSON_BYTES = 32 * 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class ArmOnlyMaterializationError(RuntimeError):
    """A source, derivative, or requested operation violates the fixed contract."""


class SourceChangedError(ArmOnlyMaterializationError):
    """The pinned source changed while it was being inspected or copied."""


@dataclass(frozen=True)
class ProjectionContract:
    source_repo_id: str
    destination_repo_id: str
    expected_task: str
    expected_episodes: int
    expected_frames: int
    expected_fps: int
    expected_source_manifest_sha256: str
    expected_source_directories_sha256: str
    expected_source_files: int
    expected_data_files: int
    expected_episode_metadata_files: int
    expected_video_files: int
    expected_controller_contract_sha256: str
    upstream_training_input_digest: str


PRODUCTION_CONTRACT = ProjectionContract(
    source_repo_id=SOURCE_REPO_ID,
    destination_repo_id=DESTINATION_REPO_ID,
    expected_task=EXPECTED_TASK,
    expected_episodes=EXPECTED_EPISODES,
    expected_frames=EXPECTED_FRAMES,
    expected_fps=EXPECTED_FPS,
    expected_source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_source_directories_sha256=EXPECTED_SOURCE_DIRECTORIES_SHA256,
    expected_source_files=EXPECTED_SOURCE_FILES,
    expected_data_files=EXPECTED_DATA_FILES,
    expected_episode_metadata_files=EXPECTED_EPISODE_METADATA_FILES,
    expected_video_files=EXPECTED_VIDEO_FILES,
    expected_controller_contract_sha256=EXPECTED_CONTROLLER_CONTRACT_SHA256,
    upstream_training_input_digest=UPSTREAM_TRAINING_INPUT_DIGEST,
)


@dataclass(frozen=True)
class InventoryCapture:
    files: tuple[dict[str, Any], ...]
    identities: tuple[tuple[str, tuple[int, ...]], ...]
    directories: tuple[str, ...]
    directory_identities: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def digest(self) -> str:
        return _sha256_json(list(self.files))

    @property
    def directories_digest(self) -> str:
        return _sha256_json(list(self.directories))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArmOnlyMaterializationError(
            f"value is not strict JSON: {exc}"
        ) from exc
    return text.encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArmOnlyMaterializationError(
            f"value is not strict JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _identity(path: Path, *, directory: bool | None = None) -> tuple[int, ...]:
    try:
        item = path.lstat()
    except FileNotFoundError as exc:
        raise SourceChangedError(f"controlled path disappeared: {path}") from exc
    if stat.S_ISLNK(item.st_mode):
        raise ArmOnlyMaterializationError(f"symbolic links are forbidden: {path}")
    if directory is True and not stat.S_ISDIR(item.st_mode):
        raise ArmOnlyMaterializationError(f"expected directory: {path}")
    if directory is False and not stat.S_ISREG(item.st_mode):
        raise ArmOnlyMaterializationError(f"expected regular file: {path}")
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ArmOnlyMaterializationError(
                f"{label} path component does not exist: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ArmOnlyMaterializationError(
                f"{label} contains a symbolic-link component: {current}"
            )
    return absolute


def _validate_roots(
    source_root: Path,
    destination_root: Path,
    *,
    destination_must_be_absent: bool,
) -> tuple[Path, Path]:
    source = _require_no_symlink_components(source_root, label="source dataset")
    _identity(source, directory=True)
    destination = _absolute_without_resolving(destination_root)
    parent = _require_no_symlink_components(
        destination.parent, label="destination parent"
    )
    _identity(parent, directory=True)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ArmOnlyMaterializationError(
            "source and destination must be disjoint sibling-capable trees"
        )
    if destination_must_be_absent:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"destination already exists and will not be overwritten: {destination}"
            )
    else:
        destination = _require_no_symlink_components(
            destination, label="arm-only dataset"
        )
        _identity(destination, directory=True)
    return source, destination


def _hash_regular_file(path: Path) -> tuple[str, int, tuple[int, ...]]:
    before = _identity(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceChangedError(f"cannot open controlled file {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before or not stat.S_ISREG(opened.st_mode):
            raise SourceChangedError(f"file changed before hashing: {path}")
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before:
            raise SourceChangedError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise SourceChangedError(f"file was replaced while hashing: {path}")
    return digest.hexdigest(), before[4], before


def _walk_regular_files(
    root: Path,
) -> tuple[list[Path], list[tuple[Path, tuple[int, ...]]]]:
    result: list[Path] = []
    directories: list[tuple[Path, tuple[int, ...]]] = []

    def visit(directory: Path) -> None:
        before = _identity(directory, directory=True)
        directories.append((directory, before))
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ArmOnlyMaterializationError(
                f"cannot enumerate dataset directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ArmOnlyMaterializationError(
                    f"dataset contains a symbolic link: {path}"
                )
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                result.append(path)
            else:
                raise ArmOnlyMaterializationError(
                    f"dataset contains a non-regular entry: {path}"
                )
        if _identity(directory, directory=True) != before:
            raise SourceChangedError(
                f"dataset directory changed while being enumerated: {directory}"
            )

    visit(root)
    return result, directories


def _inventory(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset(),
) -> InventoryCapture:
    files: list[dict[str, Any]] = []
    identities: list[tuple[str, tuple[int, ...]]] = []
    directories: list[str] = []
    directory_identities: list[tuple[str, tuple[int, ...]]] = []
    try:
        root_entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        raise ArmOnlyMaterializationError(
            f"cannot enumerate dataset root {root}: {exc}"
        ) from exc
    if [entry.name for entry in root_entries] != ["data", "meta", "videos"]:
        raise ArmOnlyMaterializationError(
            "dataset root must contain exactly the data, meta, and videos directories"
        )
    if any(
        not stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode)
        or stat.S_ISLNK(entry.stat(follow_symlinks=False).st_mode)
        for entry in root_entries
    ):
        raise ArmOnlyMaterializationError(
            "dataset root entries must be non-symlink directories"
        )
    for tree in ("meta", "data", "videos"):
        tree_root = root / tree
        if tree_root.is_symlink() or not tree_root.is_dir():
            raise ArmOnlyMaterializationError(
                f"dataset {tree} must be a non-symlink directory: {tree_root}"
            )
        tree_files, tree_directories = _walk_regular_files(tree_root)
        for directory, identity in tree_directories:
            relative_directory = directory.relative_to(root).as_posix()
            directories.append(relative_directory)
            directory_identities.append((relative_directory, identity))
        for path in tree_files:
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            digest, size, identity = _hash_regular_file(path)
            files.append({"path": relative, "size": size, "sha256": digest})
            identities.append((relative, identity))
    files.sort(key=lambda item: item["path"])
    identities.sort(key=lambda item: item[0])
    directories.sort()
    directory_identities.sort(key=lambda item: item[0])
    return InventoryCapture(
        tuple(files),
        tuple(identities),
        tuple(directories),
        tuple(directory_identities),
    )


def _verify_capture_unchanged(
    root: Path,
    capture: InventoryCapture,
    *,
    excluded: frozenset[str] = frozenset(),
) -> None:
    current = _inventory(root, excluded=excluded)
    if current != capture:
        raise SourceChangedError("captured dataset changed during materialization")


def _captured_sha256(
    capture: InventoryCapture, root: Path, path: Path
) -> str:
    relative = path.relative_to(root).as_posix()
    entries = {item["path"]: item["sha256"] for item in capture.files}
    try:
        return str(entries[relative])
    except KeyError as exc:
        raise ArmOnlyMaterializationError(
            f"file is absent from the captured inventory: {relative}"
        ) from exc


def _read_regular_bytes(
    path: Path,
    *,
    max_bytes: int = _MAX_JSON_BYTES,
    expected_sha256: str | None = None,
) -> bytes:
    digest, size, before = _hash_regular_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SourceChangedError(
            f"file digest does not match its captured inventory: {path}"
        )
    if size > max_bytes:
        raise ArmOnlyMaterializationError(f"file is unexpectedly large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        data = bytearray()
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ArmOnlyMaterializationError(
                    f"file is unexpectedly large: {path}"
                )
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before:
            raise SourceChangedError(f"file changed while reading: {path}")
    finally:
        os.close(descriptor)
    if _identity(path, directory=False) != before:
        raise SourceChangedError(f"file was replaced while reading: {path}")
    payload = bytes(data)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise SourceChangedError(f"file bytes changed between hash and read: {path}")
    return payload


def _read_json_object(
    path: Path, *, label: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_bytes(
                path, expected_sha256=expected_sha256
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArmOnlyMaterializationError(
            f"{label} is not valid UTF-8 JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ArmOnlyMaterializationError(f"{label} must be a JSON object: {path}")
    return value


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_regular_file_independent(source: Path, destination: Path) -> None:
    before = _identity(source, directory=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    target_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, source_flags)
    try:
        opened = os.fstat(source_fd)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != before:
            raise SourceChangedError(f"source changed before copy: {source}")
        target_fd = os.open(destination, target_flags, 0o644)
        try:
            while chunk := os.read(source_fd, _HASH_CHUNK_SIZE):
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)
    if _identity(source, directory=False) != before:
        raise SourceChangedError(f"source changed while copying: {source}")
    target = _identity(destination, directory=False)
    if target[3] != 1 or target[:2] == before[:2]:
        raise ArmOnlyMaterializationError(
            f"copied file is not an independent regular file: {destination}"
        )


def _require_exact_int(value: Any, expected: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ArmOnlyMaterializationError(
            f"{label} must be exactly {expected}, got {value!r}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArmOnlyMaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_feature_contract(
    info: Mapping[str, Any], *, dimensions: int, contract: ProjectionContract
) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ArmOnlyMaterializationError("dataset codebase_version must be v3.0")
    _require_exact_int(info.get("fps"), contract.expected_fps, label="info.fps")
    _require_exact_int(
        info.get("total_episodes"),
        contract.expected_episodes,
        label="info.total_episodes",
    )
    _require_exact_int(
        info.get("total_frames"), contract.expected_frames, label="info.total_frames"
    )
    _require_exact_int(info.get("total_tasks"), 1, label="info.total_tasks")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ArmOnlyMaterializationError("info.features must be an object")
    expected_names = list(JOINT_NAMES_7[:dimensions])
    for key in PROJECTED_FEATURES:
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            raise ArmOnlyMaterializationError(f"info.features.{key} is missing")
        if feature.get("dtype") != "float32":
            raise ArmOnlyMaterializationError(f"{key} dtype must be float32")
        if feature.get("shape") != [dimensions]:
            raise ArmOnlyMaterializationError(
                f"{key} shape must be [{dimensions}]"
            )
        if feature.get("names") != expected_names:
            raise ArmOnlyMaterializationError(
                f"{key} names do not match the {dimensions}-DoF contract"
            )
    splits = info.get("splits")
    if not isinstance(splits, Mapping) or splits.get("train") != f"0:{contract.expected_episodes}":
        raise ArmOnlyMaterializationError("info.splits.train is not the full episode set")


def _project_info(source: Mapping[str, Any], contract: ProjectionContract) -> dict[str, Any]:
    _validate_feature_contract(source, dimensions=7, contract=contract)
    result = copy.deepcopy(dict(source))
    features = result["features"]
    for key in PROJECTED_FEATURES:
        features[key]["shape"] = [6]
        features[key]["names"] = list(ARM_JOINT_NAMES)
    _validate_feature_contract(result, dimensions=6, contract=contract)
    return result


def _slice_stat_vector(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != 7:
        raise ArmOnlyMaterializationError(f"{label} must contain seven values")
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ArmOnlyMaterializationError(f"{label} contains a non-finite value")
    return copy.deepcopy(value[:6])


def _validate_stats_feature(
    feature: Any,
    *,
    dimensions: int,
    expected_frames: int,
    label: str,
) -> None:
    if not isinstance(feature, Mapping):
        raise ArmOnlyMaterializationError(f"{label} is missing")
    for statistic in VECTOR_STATISTICS:
        values = feature.get(statistic)
        if not isinstance(values, list) or len(values) != dimensions:
            raise ArmOnlyMaterializationError(
                f"{label}.{statistic} must contain {dimensions} values"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ArmOnlyMaterializationError(
                f"{label}.{statistic} contains a non-finite value"
            )
    if feature.get("count") != [expected_frames]:
        raise ArmOnlyMaterializationError(
            f"{label}.count must be exactly [{expected_frames}]"
        )


def _project_stats(
    source: Mapping[str, Any], *, expected_frames: int
) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    for key in PROJECTED_FEATURES:
        feature = result.get(key)
        _validate_stats_feature(
            feature,
            dimensions=7,
            expected_frames=expected_frames,
            label=f"stats.{key}",
        )
        assert isinstance(feature, dict)
        for statistic in VECTOR_STATISTICS:
            feature[statistic] = _slice_stat_vector(
                feature[statistic], label=f"stats.{key}.{statistic}"
            )
        _validate_stats_feature(
            feature,
            dimensions=6,
            expected_frames=expected_frames,
            label=f"projected stats.{key}",
        )
    return result


def _fixed_list_matrix(table: pa.Table, key: str, dimensions: int) -> np.ndarray:
    if key not in table.column_names:
        raise ArmOnlyMaterializationError(f"parquet is missing {key}")
    field = table.schema.field(key)
    expected = pa.list_(pa.float32(), dimensions)
    if field.type != expected:
        raise ArmOnlyMaterializationError(
            f"{key} must be fixed_size_list<float32>[{dimensions}], got {field.type}"
        )
    array = table[key].combine_chunks()
    if array.null_count or array.values.null_count:
        raise ArmOnlyMaterializationError(f"{key} contains null values")
    values = array.values.to_numpy(zero_copy_only=False)
    if values.size != len(table) * dimensions:
        raise ArmOnlyMaterializationError(f"{key} has an invalid flattened length")
    matrix = values.reshape(len(table), dimensions)
    if not np.isfinite(matrix).all():
        raise ArmOnlyMaterializationError(f"{key} contains NaN or infinity")
    return matrix


def _huggingface_features(
    metadata: Mapping[bytes, bytes] | None,
) -> tuple[dict[bytes, bytes], dict[str, Any]]:
    if not metadata or b"huggingface" not in metadata:
        raise ArmOnlyMaterializationError(
            "data parquet lacks Hugging Face schema metadata"
        )
    result = dict(metadata)
    try:
        payload = json.loads(result[b"huggingface"].decode("utf-8"))
        features = payload["info"]["features"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArmOnlyMaterializationError(
            "data parquet has invalid Hugging Face schema metadata"
        ) from exc
    if not isinstance(features, dict):
        raise ArmOnlyMaterializationError(
            "data parquet Hugging Face features must be an object"
        )
    return result, features


def _validate_huggingface_metadata(
    metadata: Mapping[bytes, bytes] | None, *, dimensions: int
) -> None:
    _metadata, features = _huggingface_features(metadata)
    for key in PROJECTED_FEATURES:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("length") != dimensions:
            raise ArmOnlyMaterializationError(
                f"Hugging Face metadata {key}.length must be {dimensions}"
            )
        if feature.get("_type") != "List":
            raise ArmOnlyMaterializationError(
                f"Hugging Face metadata {key} must be a List"
            )


def _project_huggingface_metadata(
    metadata: Mapping[bytes, bytes] | None,
) -> dict[bytes, bytes]:
    _validate_huggingface_metadata(metadata, dimensions=7)
    result, features = _huggingface_features(metadata)
    for key in PROJECTED_FEATURES:
        feature = features[key]
        assert isinstance(feature, dict)
        feature["length"] = 6
    payload = json.loads(result[b"huggingface"].decode("utf-8"))
    payload["info"]["features"] = features
    result[b"huggingface"] = json.dumps(
        payload, allow_nan=False, ensure_ascii=False
    ).encode("utf-8")
    return result


def _project_data_table(source: pa.Table) -> pa.Table:
    result = source
    for key in PROJECTED_FEATURES:
        matrix = _fixed_list_matrix(source, key, 7)
        flat = np.ascontiguousarray(matrix[:, :6]).reshape(-1)
        projected = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), 6
        )
        index = result.schema.get_field_index(key)
        old_field = result.schema.field(index)
        field = pa.field(
            key,
            pa.list_(pa.float32(), 6),
            nullable=old_field.nullable,
            metadata=old_field.metadata,
        )
        result = result.set_column(index, field, projected)
    return result.replace_schema_metadata(
        _project_huggingface_metadata(source.schema.metadata)
    )


def _episode_vector_fields() -> tuple[str, ...]:
    return tuple(
        f"stats/{feature}/{statistic}"
        for feature in PROJECTED_FEATURES
        for statistic in VECTOR_STATISTICS
    )


def _project_episode_table(source: pa.Table) -> pa.Table:
    result = source
    for key in _episode_vector_fields():
        if key not in source.column_names:
            raise ArmOnlyMaterializationError(
                f"episode metadata is missing {key}"
            )
        values = source[key].to_pylist()
        projected = [
            _slice_stat_vector(value, label=f"episode metadata {key}")
            for value in values
        ]
        index = result.schema.get_field_index(key)
        field = source.schema.field(key)
        result = result.set_column(
            index,
            field,
            pa.array(projected, type=field.type),
        )
    for feature in PROJECTED_FEATURES:
        count_key = f"stats/{feature}/count"
        if count_key not in source.column_names:
            raise ArmOnlyMaterializationError(
                f"episode metadata is missing {count_key}"
            )
        for value in source[count_key].to_pylist():
            if not isinstance(value, list) or len(value) != 1 or value[0] <= 0:
                raise ArmOnlyMaterializationError(
                    f"episode metadata {count_key} is invalid"
                )
    return result


def _read_parquet(
    path: Path, *, label: str, expected_sha256: str | None = None
) -> pa.Table:
    try:
        payload = _read_regular_bytes(
            path,
            max_bytes=512 * 1024 * 1024,
            expected_sha256=expected_sha256,
        )
        return pq.read_table(pa.BufferReader(payload))
    except ArmOnlyMaterializationError:
        raise
    except Exception as exc:
        raise ArmOnlyMaterializationError(
            f"cannot read {label} parquet {path}: {exc}"
        ) from exc


def _write_parquet_exclusive(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"staging file already exists: {path}")
    try:
        pq.write_table(table, path)
    except Exception as exc:
        raise ArmOnlyMaterializationError(
            f"cannot write projected parquet {path}: {exc}"
        ) from exc
    identity = _identity(path, directory=False)
    if identity[3] != 1:
        raise ArmOnlyMaterializationError(
            f"projected parquet is hard-linked: {path}"
        )


def _file_transform(relative: str) -> str:
    path = Path(relative)
    if path.parts[:1] == ("data",):
        if path.suffix != ".parquet":
            raise ArmOnlyMaterializationError(
                f"unexpected non-parquet data file: {relative}"
            )
        return "project_data_parquet"
    if path.parts[:2] == ("meta", "episodes"):
        if path.suffix != ".parquet":
            raise ArmOnlyMaterializationError(
                f"unexpected non-parquet episode metadata: {relative}"
            )
        return "project_episode_stats"
    if relative == "meta/info.json":
        return "project_info"
    if relative == "meta/stats.json":
        return "project_global_stats"
    return "byte_copy"


def _validate_source_inventory(
    capture: InventoryCapture, contract: ProjectionContract
) -> None:
    _require_exact_int(
        len(capture.files),
        contract.expected_source_files,
        label="source file count",
    )
    if capture.digest != contract.expected_source_manifest_sha256:
        raise ArmOnlyMaterializationError(
            "source file manifest SHA-256 does not match the pinned fullcycle-v4 input"
        )
    if capture.directories_digest != contract.expected_source_directories_sha256:
        raise ArmOnlyMaterializationError(
            "source directory topology SHA-256 does not match the pinned "
            "fullcycle-v4 input"
        )
    paths = [item["path"] for item in capture.files]
    data_count = sum(path.startswith("data/") and path.endswith(".parquet") for path in paths)
    episode_count = sum(
        path.startswith("meta/episodes/") and path.endswith(".parquet")
        for path in paths
    )
    video_count = sum(path.startswith("videos/") and path.endswith(".mp4") for path in paths)
    _require_exact_int(data_count, contract.expected_data_files, label="data parquet count")
    _require_exact_int(
        episode_count,
        contract.expected_episode_metadata_files,
        label="episode metadata parquet count",
    )
    _require_exact_int(video_count, contract.expected_video_files, label="video file count")


def _validate_tasks(
    root: Path, contract: ProjectionContract, capture: InventoryCapture
) -> None:
    path = root / "meta/tasks.parquet"
    table = _read_parquet(
        path,
        label="task metadata",
        expected_sha256=_captured_sha256(capture, root, path),
    )
    if table.column_names != ["task_index", "task"]:
        raise ArmOnlyMaterializationError("tasks.parquet has unexpected columns")
    rows = table.to_pylist()
    if rows != [{"task_index": 0, "task": contract.expected_task}]:
        raise ArmOnlyMaterializationError("tasks.parquet does not contain the pinned task")


def _validate_episode_stats_table(table: pa.Table, *, dimensions: int) -> None:
    lengths = table["length"].to_pylist() if "length" in table.column_names else None
    if lengths is None:
        raise ArmOnlyMaterializationError("episode metadata is missing length")
    for key in _episode_vector_fields():
        if key not in table.column_names:
            raise ArmOnlyMaterializationError(f"episode metadata is missing {key}")
        if table.schema.field(key).type != pa.list_(pa.float64()):
            raise ArmOnlyMaterializationError(
                f"episode metadata {key} must be list<float64>"
            )
        for value in table[key].to_pylist():
            if not isinstance(value, list) or len(value) != dimensions:
                raise ArmOnlyMaterializationError(
                    f"episode metadata {key} must contain {dimensions} values"
                )
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            ):
                raise ArmOnlyMaterializationError(
                    f"episode metadata {key} contains a non-finite value"
                )
    for feature in PROJECTED_FEATURES:
        key = f"stats/{feature}/count"
        if key not in table.column_names or table.schema.field(key).type != pa.list_(pa.int64()):
            raise ArmOnlyMaterializationError(
                f"episode metadata {key} must be list<int64>"
            )
        if table[key].to_pylist() != [[length] for length in lengths]:
            raise ArmOnlyMaterializationError(
                f"episode metadata {key} must equal each episode length"
            )


def _episode_rows(
    root: Path,
    contract: ProjectionContract,
    capture: InventoryCapture,
    *,
    dimensions: int,
) -> list[dict[str, Any]]:
    paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if len(paths) != contract.expected_episode_metadata_files:
        raise ArmOnlyMaterializationError("episode metadata file count mismatch")
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = _read_parquet(
            path,
            label="episode metadata",
            expected_sha256=_captured_sha256(capture, root, path),
        )
        _validate_episode_stats_table(table, dimensions=dimensions)
        rows.extend(table.select(["episode_index", "tasks", "length"]).to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    if len(rows) != contract.expected_episodes:
        raise ArmOnlyMaterializationError("episode metadata row count mismatch")
    for index, row in enumerate(rows):
        if row.get("episode_index") != index:
            raise ArmOnlyMaterializationError("episode indexes are not contiguous")
        if row.get("tasks") != [contract.expected_task]:
            raise ArmOnlyMaterializationError(
                f"episode {index} does not bind the pinned task"
            )
        length = row.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ArmOnlyMaterializationError(
                f"episode {index} has an invalid length"
            )
    if sum(row["length"] for row in rows) != contract.expected_frames:
        raise ArmOnlyMaterializationError("episode lengths do not sum to total_frames")
    return rows


def _validate_frame_tables(
    root: Path,
    contract: ProjectionContract,
    capture: InventoryCapture,
    *,
    dimensions: int,
    episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if len(paths) != contract.expected_data_files:
        raise ArmOnlyMaterializationError("data parquet file count mismatch")
    seen_frames = [0] * contract.expected_episodes
    expected_global_index = 0
    for path in paths:
        table = _read_parquet(
            path,
            label="frame data",
            expected_sha256=_captured_sha256(capture, root, path),
        )
        _validate_huggingface_metadata(table.schema.metadata, dimensions=dimensions)
        for key in PROJECTED_FEATURES:
            _fixed_list_matrix(table, key, dimensions)
        required = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
        if not required.issubset(table.column_names):
            raise ArmOnlyMaterializationError(f"frame parquet lacks indexes: {path}")
        indexes = table.select(sorted(required)).to_pylist()
        for row in indexes:
            episode_index = row["episode_index"]
            if (
                isinstance(episode_index, bool)
                or not isinstance(episode_index, int)
                or not 0 <= episode_index < contract.expected_episodes
            ):
                raise ArmOnlyMaterializationError("frame has an invalid episode_index")
            if row["frame_index"] != seen_frames[episode_index]:
                raise ArmOnlyMaterializationError(
                    f"episode {episode_index} frame_index is not contiguous"
                )
            if row["index"] != expected_global_index:
                raise ArmOnlyMaterializationError("global frame index is not contiguous")
            if row["task_index"] != 0:
                raise ArmOnlyMaterializationError("frame task_index must be zero")
            timestamp = row["timestamp"]
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
            ):
                raise ArmOnlyMaterializationError("frame timestamp is non-finite")
            seen_frames[episode_index] += 1
            expected_global_index += 1
    if expected_global_index != contract.expected_frames:
        raise ArmOnlyMaterializationError("data rows do not match total_frames")
    expected_lengths = [int(row["length"]) for row in episode_rows]
    if seen_frames != expected_lengths:
        raise ArmOnlyMaterializationError(
            "data frame counts do not match episode metadata lengths"
        )


def _validate_dataset_structure(
    root: Path,
    contract: ProjectionContract,
    capture: InventoryCapture,
    *,
    dimensions: int,
) -> dict[str, Any]:
    info_path = root / "meta/info.json"
    info = _read_json_object(
        info_path,
        label="dataset info",
        expected_sha256=_captured_sha256(capture, root, info_path),
    )
    _validate_feature_contract(info, dimensions=dimensions, contract=contract)
    _validate_tasks(root, contract, capture)
    episodes = _episode_rows(
        root, contract, capture, dimensions=dimensions
    )
    _validate_frame_tables(
        root,
        contract,
        capture,
        dimensions=dimensions,
        episode_rows=episodes,
    )
    stats_path = root / "meta/stats.json"
    stats = _read_json_object(
        stats_path,
        label="dataset stats",
        expected_sha256=_captured_sha256(capture, root, stats_path),
    )
    if dimensions == 7:
        _project_stats(stats, expected_frames=contract.expected_frames)
    else:
        for key in PROJECTED_FEATURES:
            _validate_stats_feature(
                stats.get(key),
                dimensions=6,
                expected_frames=contract.expected_frames,
                label=f"stats.{key}",
            )
    return {"info": info, "stats": stats, "episodes": episodes}


def _validate_controller_contract(
    path: Path, contract: ProjectionContract
) -> dict[str, Any]:
    absolute = _require_no_symlink_components(path, label="controller contract")
    digest, size, _identity_value = _hash_regular_file(absolute)
    if digest != contract.expected_controller_contract_sha256:
        raise ArmOnlyMaterializationError(
            "controller contract SHA-256 does not match the pinned recording contract"
        )
    payload = _read_json_object(
        absolute,
        label="controller contract",
        expected_sha256=digest,
    )
    if payload.get("joint_names") != list(JOINT_NAMES_7):
        raise ArmOnlyMaterializationError("controller contract joint names mismatch")
    limits = payload.get("effective_step_limits")
    if (
        not isinstance(limits, list)
        or len(limits) != 7
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in limits
        )
    ):
        raise ArmOnlyMaterializationError(
            "controller contract effective_step_limits are invalid"
        )
    return {
        "path": str(absolute),
        "size": size,
        "sha256": digest,
        "retained_arm_effective_step_limits_rad": [float(value) for value in limits[:6]],
        "removed_jaw_effective_step_limit_rad": float(limits[6]),
    }


def _tables_equal_except_projection(source: pa.Table, destination: pa.Table) -> None:
    if source.column_names != destination.column_names or len(source) != len(destination):
        raise ArmOnlyMaterializationError("projected data parquet changed rows or columns")
    for key in source.column_names:
        if key in PROJECTED_FEATURES:
            source_values = _fixed_list_matrix(source, key, 7)
            destination_values = _fixed_list_matrix(destination, key, 6)
            expected_bits = np.ascontiguousarray(source_values[:, :6]).view(np.uint32)
            actual_bits = np.ascontiguousarray(destination_values).view(np.uint32)
            if not np.array_equal(expected_bits, actual_bits):
                raise ArmOnlyMaterializationError(
                    f"{key} is not an exact float32 first-six projection"
                )
        elif not source[key].equals(destination[key]):
            raise ArmOnlyMaterializationError(
                f"non-projected data column changed: {key}"
            )
    expected_metadata = _project_huggingface_metadata(source.schema.metadata)
    if destination.schema.metadata != expected_metadata:
        raise ArmOnlyMaterializationError(
            "projected data Hugging Face schema metadata mismatch"
        )


def _episode_tables_equal_except_projection(
    source: pa.Table, destination: pa.Table
) -> None:
    if source.column_names != destination.column_names or len(source) != len(destination):
        raise ArmOnlyMaterializationError(
            "projected episode metadata changed rows or columns"
        )
    projected_fields = set(_episode_vector_fields())
    for key in source.column_names:
        if key in projected_fields:
            source_values = source[key].to_pylist()
            destination_values = destination[key].to_pylist()
            expected = [
                _slice_stat_vector(value, label=f"episode metadata {key}")
                for value in source_values
            ]
            if destination_values != expected:
                raise ArmOnlyMaterializationError(
                    f"episode vector statistic was not projected exactly: {key}"
                )
        elif not source[key].equals(destination[key]):
            raise ArmOnlyMaterializationError(
                f"non-projected episode metadata changed: {key}"
            )
    if source.schema.metadata != destination.schema.metadata:
        raise ArmOnlyMaterializationError(
            "episode parquet schema metadata changed unexpectedly"
        )


def _validate_exact_transform(
    source_root: Path,
    destination_root: Path,
    source_capture: InventoryCapture,
    destination_capture: InventoryCapture,
    contract: ProjectionContract,
) -> None:
    source_paths = [item["path"] for item in source_capture.files]
    destination_paths = [item["path"] for item in destination_capture.files]
    if destination_paths != source_paths:
        raise ArmOnlyMaterializationError(
            "destination content paths are not a one-to-one source mapping"
        )
    if destination_capture.directories != source_capture.directories:
        raise ArmOnlyMaterializationError(
            "destination directory topology is not a one-to-one source mapping"
        )
    source_identities = dict(source_capture.identities)
    destination_identities = dict(destination_capture.identities)
    source_entries = {item["path"]: item for item in source_capture.files}
    destination_entries = {item["path"]: item for item in destination_capture.files}
    for relative in source_paths:
        target_identity = destination_identities[relative]
        if target_identity[3] != 1:
            raise ArmOnlyMaterializationError(
                f"destination file has hard links: {relative}"
            )
        if target_identity[:2] == source_identities[relative][:2]:
            raise ArmOnlyMaterializationError(
                f"destination shares a source inode: {relative}"
            )
        source_path = source_root / relative
        destination_path = destination_root / relative
        transform = _file_transform(relative)
        if transform == "byte_copy":
            if source_entries[relative] != destination_entries[relative]:
                raise ArmOnlyMaterializationError(
                    f"byte-copied file digest differs: {relative}"
                )
        elif transform == "project_info":
            expected = _project_info(
                _read_json_object(
                    source_path,
                    label="source info",
                    expected_sha256=source_entries[relative]["sha256"],
                ),
                contract,
            )
            actual = _read_json_object(
                destination_path,
                label="destination info",
                expected_sha256=destination_entries[relative]["sha256"],
            )
            if actual != expected:
                raise ArmOnlyMaterializationError("destination info.json mismatch")
        elif transform == "project_global_stats":
            expected = _project_stats(
                _read_json_object(
                    source_path,
                    label="source stats",
                    expected_sha256=source_entries[relative]["sha256"],
                ),
                expected_frames=contract.expected_frames,
            )
            actual = _read_json_object(
                destination_path,
                label="destination stats",
                expected_sha256=destination_entries[relative]["sha256"],
            )
            if actual != expected:
                raise ArmOnlyMaterializationError("destination stats.json mismatch")
        elif transform == "project_data_parquet":
            _tables_equal_except_projection(
                _read_parquet(
                    source_path,
                    label="source data",
                    expected_sha256=source_entries[relative]["sha256"],
                ),
                _read_parquet(
                    destination_path,
                    label="destination data",
                    expected_sha256=destination_entries[relative]["sha256"],
                ),
            )
        elif transform == "project_episode_stats":
            _episode_tables_equal_except_projection(
                _read_parquet(
                    source_path,
                    label="source episode metadata",
                    expected_sha256=source_entries[relative]["sha256"],
                ),
                _read_parquet(
                    destination_path,
                    label="destination episode metadata",
                    expected_sha256=destination_entries[relative]["sha256"],
                ),
            )
        else:  # pragma: no cover - exhaustive guard
            raise AssertionError(transform)


def _manifest_document(
    *,
    source_root: Path,
    destination_root: Path,
    source_capture: InventoryCapture,
    destination_capture: InventoryCapture,
    controller_binding: Mapping[str, Any],
    contract: ProjectionContract,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "projection_version": PROJECTION_VERSION,
        "source": {
            "repo_id": contract.source_repo_id,
            "root": str(source_root),
            "episode_count": contract.expected_episodes,
            "frame_count": contract.expected_frames,
            "fps": contract.expected_fps,
            "task": contract.expected_task,
            "upstream_training_input_digest": contract.upstream_training_input_digest,
            "files_manifest_sha256": source_capture.digest,
            "files": list(source_capture.files),
            "directories_sha256": source_capture.directories_digest,
            "directories": list(source_capture.directories),
        },
        "destination": {
            "repo_id": contract.destination_repo_id,
            "root": str(destination_root),
            "episode_count": contract.expected_episodes,
            "frame_count": contract.expected_frames,
            "fps": contract.expected_fps,
            "task": contract.expected_task,
            "files_manifest_sha256": destination_capture.digest,
            "files": list(destination_capture.files),
            "directories_sha256": destination_capture.directories_digest,
            "directories": list(destination_capture.directories),
        },
        "transform": {
            "projected_features": {
                key: {
                    "source_dtype": "float32",
                    "source_shape": [7],
                    "destination_dtype": "float32",
                    "destination_shape": [6],
                    "operation": "destination_value[i] = source_value[i] for i in 0..5",
                }
                for key in PROJECTED_FEATURES
            },
            "retained_feature_names": list(ARM_JOINT_NAMES),
            "removed_feature": {
                "name": "jaw_joint1.pos",
                "source_index": 6,
                "removed_from": list(PROJECTED_FEATURES),
                "runtime_authority": (
                    "jaw must be controlled independently; this dataset emits no "
                    "jaw action"
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
        },
        "controller_contract": dict(controller_binding),
        "verification": {
            "exact_first_six_float32_bits": True,
            "one_to_one_file_paths": True,
            "copied_files_byte_identical": True,
            "destination_files_independent_regular_single_link": True,
            "source_unchanged_before_publish": True,
        },
    }
    document["document_sha256"] = _sha256_json(document)
    return document


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    source_root: Path,
    declared_destination_root: Path,
    source_capture: InventoryCapture,
    destination_capture: InventoryCapture,
    controller_binding: Mapping[str, Any],
    contract: ProjectionContract,
) -> None:
    expected = _manifest_document(
        source_root=source_root,
        destination_root=declared_destination_root,
        source_capture=source_capture,
        destination_capture=destination_capture,
        controller_binding=controller_binding,
        contract=contract,
    )
    if dict(manifest) != expected:
        raise ArmOnlyMaterializationError(
            "arm-only manifest does not match the recomputed source/target contract"
        )
    _require_sha256(manifest.get("document_sha256"), label="manifest document_sha256")


def validate_arm_only_dataset(
    source_root: Path,
    destination_root: Path,
    *,
    controller_contract_path: Path,
    contract: ProjectionContract = PRODUCTION_CONTRACT,
    declared_destination_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly validate a published derivative or an unpublished staging tree."""

    source, destination = _validate_roots(
        source_root, destination_root, destination_must_be_absent=False
    )
    declared = _absolute_without_resolving(
        declared_destination_root if declared_destination_root is not None else destination
    )
    source_capture = _inventory(source)
    _validate_source_inventory(source_capture, contract)
    destination_capture = _inventory(
        destination,
        excluded=frozenset({MANIFEST_RELATIVE_PATH.as_posix()}),
    )
    controller_binding = _validate_controller_contract(
        controller_contract_path, contract
    )
    _validate_dataset_structure(
        source, contract, source_capture, dimensions=7
    )
    _validate_dataset_structure(
        destination, contract, destination_capture, dimensions=6
    )
    _validate_exact_transform(
        source, destination, source_capture, destination_capture, contract
    )
    manifest = _read_json_object(
        destination / MANIFEST_RELATIVE_PATH, label="arm-only manifest"
    )
    _validate_manifest(
        manifest,
        source_root=source,
        declared_destination_root=declared,
        source_capture=source_capture,
        destination_capture=destination_capture,
        controller_binding=controller_binding,
        contract=contract,
    )
    manifest_identity = _identity(
        destination / MANIFEST_RELATIVE_PATH, directory=False
    )
    if manifest_identity[3] != 1:
        raise ArmOnlyMaterializationError("arm-only manifest must not be hard-linked")
    _verify_capture_unchanged(source, source_capture)
    _verify_capture_unchanged(
        destination,
        destination_capture,
        excluded=frozenset({MANIFEST_RELATIVE_PATH.as_posix()}),
    )
    return {
        "safe_for_arm_only_training": True,
        "source_manifest_sha256": source_capture.digest,
        "destination_manifest_sha256": destination_capture.digest,
        "document_sha256": manifest["document_sha256"],
        "episode_count": contract.expected_episodes,
        "frame_count": contract.expected_frames,
        "action_dimensions": 6,
        "state_dimensions": 6,
        "removed_feature": "jaw_joint1.pos",
    }


def _materialize_file(
    source_root: Path,
    staging_root: Path,
    relative: str,
    contract: ProjectionContract,
    expected_source_sha256: str,
) -> None:
    source = source_root / relative
    destination = staging_root / relative
    transform = _file_transform(relative)
    if transform == "byte_copy":
        _copy_regular_file_independent(source, destination)
    elif transform == "project_info":
        payload = _project_info(
            _read_json_object(
                source,
                label="source info",
                expected_sha256=expected_source_sha256,
            ),
            contract,
        )
        _write_bytes_exclusive(destination, _pretty_json_bytes(payload))
    elif transform == "project_global_stats":
        payload = _project_stats(
            _read_json_object(
                source,
                label="source stats",
                expected_sha256=expected_source_sha256,
            ),
            expected_frames=contract.expected_frames,
        )
        _write_bytes_exclusive(destination, _pretty_json_bytes(payload))
    elif transform == "project_data_parquet":
        _write_parquet_exclusive(
            _project_data_table(
                _read_parquet(
                    source,
                    label="source data",
                    expected_sha256=expected_source_sha256,
                )
            ),
            destination,
        )
    elif transform == "project_episode_stats":
        _write_parquet_exclusive(
            _project_episode_table(
                _read_parquet(
                    source,
                    label="source episode metadata",
                    expected_sha256=expected_source_sha256,
                )
            ),
            destination,
        )
    else:  # pragma: no cover - exhaustive guard
        raise AssertionError(transform)


def _publish_noreplace(staging_root: Path, destination_root: Path) -> None:
    """Atomically publish one directory without ever replacing a path."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArmOnlyMaterializationError(
            "Linux renameat2 is required for no-replace publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(staging_root),
        _AT_FDCWD,
        os.fsencode(destination_root),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            f"destination appeared and was not overwritten: {destination_root}"
        )
    raise OSError(error, os.strerror(error), os.fspath(destination_root))


def audit_pinned_source(
    source_root: Path = SOURCE_ROOT,
    *,
    controller_contract_path: Path = CONTROLLER_CONTRACT,
    contract: ProjectionContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    source = _require_no_symlink_components(source_root, label="source dataset")
    _identity(source, directory=True)
    capture = _inventory(source)
    _validate_source_inventory(capture, contract)
    _validate_dataset_structure(source, contract, capture, dimensions=7)
    controller = _validate_controller_contract(controller_contract_path, contract)
    _verify_capture_unchanged(source, capture)
    return {
        "status": "source_audited_read_only",
        "source": {
            "repo_id": contract.source_repo_id,
            "root": str(source),
            "files": len(capture.files),
            "files_manifest_sha256": capture.digest,
            "directories": len(capture.directories),
            "directories_sha256": capture.directories_digest,
            "episode_count": contract.expected_episodes,
            "frame_count": contract.expected_frames,
            "fps": contract.expected_fps,
            "task": contract.expected_task,
            "source_dimensions": 7,
        },
        "destination": {
            "repo_id": contract.destination_repo_id,
            "root": str(DESTINATION_ROOT),
            "destination_dimensions": 6,
            "must_not_exist_before_materialization": True,
        },
        "removed_feature": {"name": "jaw_joint1.pos", "source_index": 6},
        "controller_contract": controller,
    }


def materialize_arm_only_dataset(
    source_root: Path = SOURCE_ROOT,
    destination_root: Path = DESTINATION_ROOT,
    *,
    controller_contract_path: Path = CONTROLLER_CONTRACT,
    contract: ProjectionContract = PRODUCTION_CONTRACT,
) -> Path:
    source, destination = _validate_roots(
        source_root, destination_root, destination_must_be_absent=True
    )
    source_capture = _inventory(source)
    _validate_source_inventory(source_capture, contract)
    _validate_dataset_structure(source, contract, source_capture, dimensions=7)
    controller_binding = _validate_controller_contract(
        controller_contract_path, contract
    )

    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-", dir=destination.parent
        )
    )
    staging_root = staging_parent / "dataset"
    staging_root.mkdir(mode=0o755)
    published = False
    try:
        for relative_directory in source_capture.directories:
            (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)
        for entry in source_capture.files:
            _materialize_file(
                source,
                staging_root,
                entry["path"],
                contract,
                entry["sha256"],
            )

        destination_capture = _inventory(staging_root)
        _validate_dataset_structure(
            staging_root, contract, destination_capture, dimensions=6
        )
        _validate_exact_transform(
            source,
            staging_root,
            source_capture,
            destination_capture,
            contract,
        )
        manifest = _manifest_document(
            source_root=source,
            destination_root=destination,
            source_capture=source_capture,
            destination_capture=destination_capture,
            controller_binding=controller_binding,
            contract=contract,
        )
        _write_bytes_exclusive(
            staging_root / MANIFEST_RELATIVE_PATH,
            _pretty_json_bytes(manifest),
        )
        publish_capture = _inventory(staging_root)
        validate_arm_only_dataset(
            source,
            staging_root,
            controller_contract_path=controller_contract_path,
            contract=contract,
            declared_destination_root=destination,
        )
        _verify_capture_unchanged(source, source_capture)
        _verify_capture_unchanged(staging_root, publish_capture)
        _publish_noreplace(staging_root, destination)
        published = True
    finally:
        if published:
            try:
                staging_parent.rmdir()
            except OSError:
                pass
        else:
            try:
                shutil.rmtree(staging_parent)
            except OSError as exc:
                raise ArmOnlyMaterializationError(
                    "materialization failed and the owned staging directory could "
                    f"not be removed: {staging_parent}: {exc}"
                ) from exc
            if staging_parent.exists() or staging_parent.is_symlink():
                raise ArmOnlyMaterializationError(
                    "materialization failed and left an owned staging directory: "
                    f"{staging_parent}"
                )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--materialize",
        action="store_true",
        help="Create the fixed destination after full staging validation.",
    )
    action.add_argument(
        "--validate",
        action="store_true",
        help="Validate the already materialized fixed destination read-only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.materialize:
            destination = materialize_arm_only_dataset()
            result = validate_arm_only_dataset(
                SOURCE_ROOT,
                destination,
                controller_contract_path=CONTROLLER_CONTRACT,
            )
            result["status"] = "materialized"
            result["destination"] = str(destination)
        elif args.validate:
            result = validate_arm_only_dataset(
                SOURCE_ROOT,
                DESTINATION_ROOT,
                controller_contract_path=CONTROLLER_CONTRACT,
            )
            result["status"] = "validated"
            result["destination"] = str(DESTINATION_ROOT)
        else:
            result = audit_pinned_source()
        print(_pretty_json_bytes(result).decode("utf-8"), end="")
        return 0
    except (
        ArmOnlyMaterializationError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"arm-only materialization refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
