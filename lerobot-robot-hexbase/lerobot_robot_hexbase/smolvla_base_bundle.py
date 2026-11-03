"""Materialize a pinned, fully offline ``lerobot/smolvla_base`` policy.

The upstream policy is copied from a Hugging Face cache snapshot after every
required artifact is verified.  Two JSON fields are then patched to point at a
separately verified local SmolVLM bundle, so stock LeRobot can construct both
the policy and tokenizer without consulting the Hub.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .smolvlm_bundle import (
    SMOLVLM_REPO_ID,
    validate_materialized_smolvlm_bundle,
)


SMOLVLA_BASE_REPO_ID: Final = "lerobot/smolvla_base"
SMOLVLA_BASE_REVISION: Final = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SMOLVLA_BASE_PROVENANCE_FILENAME: Final = "firefly-smolvla-base-provenance.lock.json"
SMOLVLA_BASE_SCHEMA_VERSION: Final = 1
SMOLVLA_BASE_BUNDLE_TYPE: Final = "firefly_pinned_smolvla_base"


@dataclass(frozen=True)
class SmolVLABaseAssetFile:
    path: str
    size: int
    sha256: str


SMOLVLA_BASE_ASSET_FILES: Final[tuple[SmolVLABaseAssetFile, ...]] = (
    SmolVLABaseAssetFile(
        "config.json",
        2_299,
        "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d",
    ),
    SmolVLABaseAssetFile(
        "model.safetensors",
        906_712_520,
        "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
    ),
    SmolVLABaseAssetFile(
        "policy_postprocessor.json",
        660,
        "2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b",
    ),
    SmolVLABaseAssetFile(
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        640,
        "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8",
    ),
    SmolVLABaseAssetFile(
        "policy_preprocessor.json",
        1_872,
        "7683d648280a72f0d51e869fb8dd1596beed90b7f84849330cf1bc26634a4bd6",
    ),
    SmolVLABaseAssetFile(
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        640,
        "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8",
    ),
)

_UPSTREAM_CONFIG_CANONICAL_SHA256: Final = (
    "85718d26fae7aa167615fbbc37bea3ec716ce3ab052140336222faef9e4ae30c"
)
_UPSTREAM_PREPROCESSOR_CANONICAL_SHA256: Final = (
    "519ffb6609d816f3537d1a2ab3db9269ba12aa8775de4f36761c692d2b6ed99b"
)
_PATCHED_JSON_FILES: Final = frozenset({"config.json", "policy_preprocessor.json"})
_COPY_CHUNK_SIZE: Final = 1024 * 1024
_MAX_JSON_BYTES: Final = 1024 * 1024
_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1


class SmolVLABaseBundleError(ValueError):
    """Raised when the pinned base policy, patch, or provenance is invalid."""


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
            raise SmolVLABaseBundleError(f"{label} must not contain symbolic links: {current}")
    return absolute


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _require_directory(path: Path, *, label: str) -> tuple[int, ...]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return _identity(info)


def _open_source_regular(path: Path) -> int:
    try:
        entry_mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"pinned SmolVLA base source file is missing: {path.name}") from exc
    if not (stat.S_ISREG(entry_mode) or stat.S_ISLNK(entry_mode)):
        raise SmolVLABaseBundleError(
            f"base source entry must be a file or Hugging Face cache symlink: {path.name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SmolVLABaseBundleError(
            f"base source entry does not resolve to a regular file: {path.name}"
        )
    return descriptor


def _new_output_descriptor(path: Path) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path, flags, 0o600)


def _verify_observed_asset(
    expected: SmolVLABaseAssetFile, observed_size: int, observed_digest: str
) -> None:
    if observed_size != expected.size or observed_digest != expected.sha256:
        raise SmolVLABaseBundleError(
            f"source file does not match pinned {SMOLVLA_BASE_REPO_ID}@"
            f"{SMOLVLA_BASE_REVISION}: {expected.path} "
            f"(size={observed_size}, sha256={observed_digest})"
        )


def _copy_source_asset(
    source: Path, destination: Path, expected: SmolVLABaseAssetFile
) -> dict[str, int | str]:
    source_fd = _open_source_regular(source)
    try:
        destination_fd = _new_output_descriptor(destination)
    except BaseException:
        os.close(source_fd)
        raise
    digest = hashlib.sha256()
    observed_size = 0
    try:
        before = _identity(os.fstat(source_fd))
        with os.fdopen(source_fd, "rb", closefd=False) as source_handle, os.fdopen(
            destination_fd, "wb", closefd=False
        ) as destination_handle:
            while chunk := source_handle.read(_COPY_CHUNK_SIZE):
                destination_handle.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            destination_handle.flush()
            os.fsync(destination_fd)
        after = _identity(os.fstat(source_fd))
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    if before != after:
        raise SmolVLABaseBundleError(f"source file changed while copying: {expected.path}")
    observed_digest = digest.hexdigest()
    _verify_observed_asset(expected, observed_size, observed_digest)
    os.chmod(destination, 0o444, follow_symlinks=False)
    return {"path": expected.path, "size": observed_size, "sha256": observed_digest}


def _read_source_asset(source: Path, expected: SmolVLABaseAssetFile) -> bytes:
    descriptor = _open_source_regular(source)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed_size = 0
    try:
        before = _identity(os.fstat(descriptor))
        if before[3] > _MAX_JSON_BYTES:
            raise SmolVLABaseBundleError(f"base JSON source is unexpectedly large: {expected.path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(_COPY_CHUNK_SIZE):
                chunks.append(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
        after = _identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if before != after:
        raise SmolVLABaseBundleError(f"source file changed while reading: {expected.path}")
    observed_digest = digest.hexdigest()
    _verify_observed_asset(expected, observed_size, observed_digest)
    return b"".join(chunks)


def _write_output_file(path: Path, payload: bytes) -> dict[str, int | str]:
    descriptor = _new_output_descriptor(path)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444, follow_symlinks=False)
    return {
        "path": path.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SmolVLABaseBundleError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmolVLABaseBundleError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SmolVLABaseBundleError(f"{label} must contain a JSON object")
    return value


def _pretty_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_step(processor: dict[str, object]) -> dict[str, object]:
    steps = processor.get("steps")
    if not isinstance(steps, list):
        raise SmolVLABaseBundleError("policy_preprocessor.json steps must be a list")
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor"
    ]
    if len(matches) != 1:
        raise SmolVLABaseBundleError(
            "policy_preprocessor.json must have exactly one tokenizer_processor step"
        )
    config = matches[0].get("config")
    if not isinstance(config, dict):
        raise SmolVLABaseBundleError("tokenizer_processor config must be an object")
    return config


def _patch_config(payload: bytes, bundle_path: Path) -> bytes:
    config = _load_json(payload, label="upstream config.json")
    if _canonical_json_digest(config) != _UPSTREAM_CONFIG_CANONICAL_SHA256:
        raise SmolVLABaseBundleError("upstream config.json semantic digest is not pinned")
    if config.get("vlm_model_name") != SMOLVLM_REPO_ID:
        raise SmolVLABaseBundleError("upstream config.json has an unexpected vlm_model_name")
    if config.get("load_vlm_weights") is not True:
        raise SmolVLABaseBundleError("upstream config.json load_vlm_weights must be true")
    config["vlm_model_name"] = str(bundle_path)
    config["load_vlm_weights"] = False
    return _pretty_json(config)


def _patch_preprocessor(payload: bytes, bundle_path: Path) -> bytes:
    processor = _load_json(payload, label="upstream policy_preprocessor.json")
    if _canonical_json_digest(processor) != _UPSTREAM_PREPROCESSOR_CANONICAL_SHA256:
        raise SmolVLABaseBundleError(
            "upstream policy_preprocessor.json semantic digest is not pinned"
        )
    tokenizer = _tokenizer_step(processor)
    if tokenizer.get("tokenizer_name") != SMOLVLM_REPO_ID:
        raise SmolVLABaseBundleError("upstream tokenizer source is not the pinned SmolVLM")
    tokenizer["tokenizer_name"] = str(bundle_path)
    return _pretty_json(processor)


def _upstream_manifest() -> dict[str, object]:
    return {
        "repo_id": SMOLVLA_BASE_REPO_ID,
        "revision": SMOLVLA_BASE_REVISION,
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in SMOLVLA_BASE_ASSET_FILES
        ],
    }


def _patch_manifest(bundle_path: Path) -> list[dict[str, object]]:
    local_path = str(bundle_path)
    return [
        {
            "path": "config.json",
            "json_pointer": "/vlm_model_name",
            "from": SMOLVLM_REPO_ID,
            "to": local_path,
        },
        {
            "path": "config.json",
            "json_pointer": "/load_vlm_weights",
            "from": True,
            "to": False,
        },
        {
            "path": "policy_preprocessor.json",
            "selector": {"registry_name": "tokenizer_processor"},
            "json_pointer": "/config/tokenizer_name",
            "from": SMOLVLM_REPO_ID,
            "to": local_path,
        },
    ]


def _bundle_binding(bundle_path: Path, manifest: dict[str, object]) -> dict[str, object]:
    return {
        "canonical_path": str(bundle_path),
        "manifest_sha256": _canonical_json_digest(manifest),
        "manifest": copy.deepcopy(manifest),
    }


def _provenance_manifest(
    bundle_path: Path,
    bundle_manifest: dict[str, object],
    materialized_files: list[dict[str, int | str]],
) -> dict[str, object]:
    return {
        "schema_version": SMOLVLA_BASE_SCHEMA_VERSION,
        "bundle_type": SMOLVLA_BASE_BUNDLE_TYPE,
        "upstream": _upstream_manifest(),
        "patches": _patch_manifest(bundle_path),
        "smolvlm_bundle": _bundle_binding(bundle_path, bundle_manifest),
        "materialized_files": materialized_files,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_noreplace(staging: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic no-replace publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(staging),
        _AT_FDCWD,
        os.fsencode(output),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, "refusing to overwrite existing bundle", output)
    raise OSError(error_number, os.strerror(error_number), output)


def _read_output_regular(path: Path, *, max_bytes: int | None = None) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmolVLABaseBundleError(f"base bundle asset must be regular: {path.name}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise SmolVLABaseBundleError(f"base bundle JSON is unexpectedly large: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(_COPY_CHUNK_SIZE):
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise SmolVLABaseBundleError(f"base bundle asset changed while reading: {path.name}")
    return b"".join(chunks)


def _hash_output_regular(path: Path) -> dict[str, int | str]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed_size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmolVLABaseBundleError(f"base bundle asset must be regular: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(_COPY_CHUNK_SIZE):
                digest.update(chunk)
                observed_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise SmolVLABaseBundleError(f"base bundle asset changed while hashing: {path.name}")
    return {"path": path.name, "size": observed_size, "sha256": digest.hexdigest()}


def _validate_patched_config(path: Path, bundle_path: Path) -> None:
    payload = _read_output_regular(path, max_bytes=_MAX_JSON_BYTES)
    config = _load_json(payload, label="materialized config.json")
    if payload != _pretty_json(config):
        raise SmolVLABaseBundleError("materialized config.json is not canonical JSON")
    if config.get("vlm_model_name") != str(bundle_path):
        raise SmolVLABaseBundleError("materialized config.json has the wrong local VLM path")
    if config.get("load_vlm_weights") is not False:
        raise SmolVLABaseBundleError("materialized config.json must disable external VLM weights")
    upstream = copy.deepcopy(config)
    upstream["vlm_model_name"] = SMOLVLM_REPO_ID
    upstream["load_vlm_weights"] = True
    if _canonical_json_digest(upstream) != _UPSTREAM_CONFIG_CANONICAL_SHA256:
        raise SmolVLABaseBundleError("materialized config.json contains an unapproved patch")


def _validate_patched_preprocessor(path: Path, bundle_path: Path) -> None:
    payload = _read_output_regular(path, max_bytes=_MAX_JSON_BYTES)
    processor = _load_json(payload, label="materialized policy_preprocessor.json")
    if payload != _pretty_json(processor):
        raise SmolVLABaseBundleError(
            "materialized policy_preprocessor.json is not canonical JSON"
        )
    tokenizer = _tokenizer_step(processor)
    if tokenizer.get("tokenizer_name") != str(bundle_path):
        raise SmolVLABaseBundleError("materialized tokenizer has the wrong local VLM path")
    upstream = copy.deepcopy(processor)
    _tokenizer_step(upstream)["tokenizer_name"] = SMOLVLM_REPO_ID
    if _canonical_json_digest(upstream) != _UPSTREAM_PREPROCESSOR_CANONICAL_SHA256:
        raise SmolVLABaseBundleError(
            "materialized policy_preprocessor.json contains an unapproved patch"
        )


def validate_materialized_smolvla_base(
    path: Path,
) -> tuple[Path, dict[str, object]]:
    """Validate an offline base policy and its exact bound SmolVLM bundle."""
    base = _require_no_symlink_components(Path(path), label="SmolVLA base bundle")
    directory_before = _require_directory(base, label="SmolVLA base bundle")
    expected_names = {item.path for item in SMOLVLA_BASE_ASSET_FILES}
    expected_names.add(SMOLVLA_BASE_PROVENANCE_FILENAME)
    with os.scandir(base) as entries:
        actual_entries = {
            entry.name: (entry.is_symlink(), entry.is_file(follow_symlinks=False))
            for entry in entries
        }
    if set(actual_entries) != expected_names:
        missing = sorted(expected_names - set(actual_entries))
        unexpected = sorted(set(actual_entries) - expected_names)
        raise SmolVLABaseBundleError(
            f"SmolVLA base file set mismatch; missing={missing!r}, unexpected={unexpected!r}"
        )
    for name, (is_symlink, is_regular) in actual_entries.items():
        if is_symlink or not is_regular:
            raise SmolVLABaseBundleError(f"SmolVLA base entry must be regular: {name}")

    lock_payload = _read_output_regular(
        base / SMOLVLA_BASE_PROVENANCE_FILENAME, max_bytes=_MAX_JSON_BYTES
    )
    lock = _load_json(lock_payload, label="SmolVLA base provenance lock")
    if lock_payload != _pretty_json(lock):
        raise SmolVLABaseBundleError("SmolVLA base provenance lock is not canonical JSON")
    binding = lock.get("smolvlm_bundle")
    if not isinstance(binding, dict):
        raise SmolVLABaseBundleError("SmolVLA base provenance lacks a VLM bundle binding")
    bundle_value = binding.get("canonical_path")
    if not isinstance(bundle_value, str) or not Path(bundle_value).is_absolute():
        raise SmolVLABaseBundleError("bound SmolVLM bundle path must be absolute")
    bundle_path, bundle_manifest = validate_materialized_smolvlm_bundle(Path(bundle_value))
    expected_binding = _bundle_binding(bundle_path, bundle_manifest)
    if binding != expected_binding:
        raise SmolVLABaseBundleError("bound SmolVLM bundle manifest or digest does not match")

    _validate_patched_config(base / "config.json", bundle_path)
    _validate_patched_preprocessor(base / "policy_preprocessor.json", bundle_path)
    materialized_files = [
        _hash_output_regular(base / item.path) for item in SMOLVLA_BASE_ASSET_FILES
    ]
    upstream_by_name = {item.path: item for item in SMOLVLA_BASE_ASSET_FILES}
    for entry in materialized_files:
        if entry["path"] in _PATCHED_JSON_FILES:
            continue
        upstream = upstream_by_name[str(entry["path"])]
        if entry["size"] != upstream.size or entry["sha256"] != upstream.sha256:
            raise SmolVLABaseBundleError(
                f"unpatched base asset does not match upstream: {entry['path']}"
            )
    expected_lock = _provenance_manifest(bundle_path, bundle_manifest, materialized_files)
    if lock != expected_lock:
        raise SmolVLABaseBundleError("SmolVLA base provenance lock does not match its files")
    directory_after = _require_directory(base, label="SmolVLA base bundle")
    if directory_before != directory_after:
        raise SmolVLABaseBundleError("SmolVLA base directory changed during validation")
    return base.resolve(strict=True), copy.deepcopy(lock)


def materialize_smolvla_base(
    source_snapshot: Path,
    smolvlm_bundle: Path,
    output: Path,
) -> tuple[Path, dict[str, object]]:
    """Patch and atomically publish the pinned base policy for offline training."""
    bundle_path, bundle_manifest = validate_materialized_smolvlm_bundle(
        Path(smolvlm_bundle)
    )
    source = _require_no_symlink_components(
        Path(source_snapshot), label="SmolVLA base source snapshot"
    )
    _require_directory(source, label="SmolVLA base source snapshot")
    target = _absolute_without_resolving(Path(output))
    parent = _require_no_symlink_components(target.parent, label="SmolVLA base output parent")
    _require_directory(parent, label="SmolVLA base output parent")
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite existing bundle", target)
    if source in target.parents:
        raise SmolVLABaseBundleError("SmolVLA base output must not be inside its source snapshot")
    if bundle_path in target.parents:
        raise SmolVLABaseBundleError("SmolVLA base output must not be inside its SmolVLM bundle")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    published = False
    try:
        materialized_by_name: dict[str, dict[str, int | str]] = {}
        for expected in SMOLVLA_BASE_ASSET_FILES:
            source_path = source / expected.path
            destination = staging / expected.path
            if expected.path == "config.json":
                payload = _patch_config(
                    _read_source_asset(source_path, expected), bundle_path
                )
                entry = _write_output_file(destination, payload)
            elif expected.path == "policy_preprocessor.json":
                payload = _patch_preprocessor(
                    _read_source_asset(source_path, expected), bundle_path
                )
                entry = _write_output_file(destination, payload)
            else:
                entry = _copy_source_asset(source_path, destination, expected)
            materialized_by_name[expected.path] = entry
        materialized_files = [
            materialized_by_name[item.path] for item in SMOLVLA_BASE_ASSET_FILES
        ]
        provenance = _provenance_manifest(
            bundle_path, bundle_manifest, materialized_files
        )
        _write_output_file(
            staging / SMOLVLA_BASE_PROVENANCE_FILENAME, _pretty_json(provenance)
        )
        os.chmod(staging, 0o755)
        _fsync_directory(staging)
        validate_materialized_smolvla_base(staging)
        _atomic_publish_noreplace(staging, target)
        published = True
        _fsync_directory(parent)
        return target.resolve(strict=True), provenance
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)
