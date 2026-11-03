"""Pinned, offline SmolVLM assets used by Firefly SmolVLA policies.

The Hugging Face cache snapshot is a symlink farm into a mutable blob cache.  This
module verifies the exact upstream files and materializes a flat, symlink-free
bundle that deployment provenance can hash and monitor.
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


SMOLVLM_REPO_ID: Final = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM_REVISION: Final = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SMOLVLM_PROVENANCE_FILENAME: Final = "firefly-smolvlm-provenance.lock.json"
SMOLVLM_BUNDLE_SCHEMA_VERSION: Final = 1
SMOLVLM_BUNDLE_TYPE: Final = "firefly_smolvlm_assets"


@dataclass(frozen=True)
class SmolVLMAssetFile:
    path: str
    size: int
    sha256: str


SMOLVLM_ASSET_FILES: Final[tuple[SmolVLMAssetFile, ...]] = (
    SmolVLMAssetFile(
        "added_tokens.json",
        4_739,
        "74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a",
    ),
    SmolVLMAssetFile(
        "chat_template.json",
        430,
        "b585e3598909a5687f9f9d738d35223724dedef256b9b274e1cbfb32b13c74bf",
    ),
    SmolVLMAssetFile(
        "config.json",
        3_767,
        "ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17",
    ),
    SmolVLMAssetFile(
        "generation_config.json",
        136,
        "34835060c9f0f74d1acb456cc72ca32746d3843d9eb5f578f9cbffac1d2eb840",
    ),
    SmolVLMAssetFile(
        "merges.txt",
        466_391,
        "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    ),
    SmolVLMAssetFile(
        "preprocessor_config.json",
        599,
        "149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374",
    ),
    SmolVLMAssetFile(
        "processor_config.json",
        67,
        "f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c",
    ),
    SmolVLMAssetFile(
        "special_tokens_map.json",
        868,
        "2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60",
    ),
    SmolVLMAssetFile(
        "tokenizer.json",
        3_548_256,
        "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    ),
    SmolVLMAssetFile(
        "tokenizer_config.json",
        28_626,
        "dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913",
    ),
    SmolVLMAssetFile(
        "vocab.json",
        800_662,
        "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    ),
)

_COPY_CHUNK_SIZE: Final = 1024 * 1024
_MAX_LOCK_BYTES: Final = 1024 * 1024
_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1


class SmolVLMBundleError(ValueError):
    """Raised when pinned SmolVLM assets or provenance are invalid."""


def expected_smolvlm_manifest() -> dict[str, object]:
    """Return a fresh copy of the one accepted Firefly SmolVLM manifest."""
    return {
        "schema_version": SMOLVLM_BUNDLE_SCHEMA_VERSION,
        "bundle_type": SMOLVLM_BUNDLE_TYPE,
        "repo_id": SMOLVLM_REPO_ID,
        "revision": SMOLVLM_REVISION,
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in SMOLVLM_ASSET_FILES
        ],
    }


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
            raise SmolVLMBundleError(f"{label} must not contain symbolic links: {current}")
    return absolute


def _directory_identity(path: Path, *, label: str) -> tuple[int, ...]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_source_regular(path: Path) -> int:
    try:
        entry_mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"pinned SmolVLM source file is missing: {path.name}") from exc
    if not (stat.S_ISREG(entry_mode) or stat.S_ISLNK(entry_mode)):
        raise SmolVLMBundleError(
            f"pinned SmolVLM source entry must be a file or cache symlink: {path.name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SmolVLMBundleError(
            f"pinned SmolVLM source entry does not resolve to a regular file: {path.name}"
        )
    return descriptor


def _copy_and_verify(source: Path, destination: Path, expected: SmolVLMAssetFile) -> None:
    source_fd = _open_source_regular(source)
    output_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        destination_fd = os.open(destination, output_flags, 0o600)
    except BaseException:
        os.close(source_fd)
        raise
    digest = hashlib.sha256()
    observed_size = 0
    try:
        source_before = _file_identity(os.fstat(source_fd))
        with os.fdopen(source_fd, "rb", closefd=False) as source_handle, os.fdopen(
            destination_fd, "wb", closefd=False
        ) as destination_handle:
            while chunk := source_handle.read(_COPY_CHUNK_SIZE):
                destination_handle.write(chunk)
                digest.update(chunk)
                observed_size += len(chunk)
            destination_handle.flush()
            os.fsync(destination_fd)
        source_after = _file_identity(os.fstat(source_fd))
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    if source_before != source_after:
        raise SmolVLMBundleError(f"source file changed while copying: {expected.path}")
    actual_digest = digest.hexdigest()
    if observed_size != expected.size or actual_digest != expected.sha256:
        raise SmolVLMBundleError(
            f"source file does not match pinned {SMOLVLM_REPO_ID}@{SMOLVLM_REVISION}: "
            f"{expected.path} (size={observed_size}, sha256={actual_digest})"
        )
    os.chmod(destination, 0o444, follow_symlinks=False)


def _write_lock(path: Path, manifest: dict[str, object]) -> None:
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444, follow_symlinks=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_noreplace(staging: Path, output: Path) -> None:
    """Atomically rename a directory while refusing even an empty existing target."""
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SmolVLMBundleError(f"provenance lock contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_lock(path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SmolVLMBundleError("SmolVLM provenance lock must be a regular file")
        if before.st_size > _MAX_LOCK_BYTES:
            raise SmolVLMBundleError("SmolVLM provenance lock is unexpectedly large")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_LOCK_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise SmolVLMBundleError("SmolVLM provenance lock changed while reading")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmolVLMBundleError("SmolVLM provenance lock is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SmolVLMBundleError("SmolVLM provenance lock must contain a JSON object")
    return value


def _hash_regular_file(path: Path) -> tuple[int, str]:
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
            raise SmolVLMBundleError(f"bundle asset must be a regular file: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(_COPY_CHUNK_SIZE):
                digest.update(chunk)
                observed_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise SmolVLMBundleError(f"bundle asset changed while hashing: {path.name}")
    return observed_size, digest.hexdigest()


def validate_materialized_smolvlm_bundle(
    path: Path,
) -> tuple[Path, dict[str, object]]:
    """Validate a pinned, symlink-free bundle and return its canonical path and manifest."""
    bundle = _require_no_symlink_components(Path(path), label="SmolVLM bundle")
    directory_before = _directory_identity(bundle, label="SmolVLM bundle")
    expected_manifest = expected_smolvlm_manifest()
    expected_names = {item.path for item in SMOLVLM_ASSET_FILES}
    expected_names.add(SMOLVLM_PROVENANCE_FILENAME)
    with os.scandir(bundle) as entries:
        actual_entries = {
            entry.name: (entry.is_symlink(), entry.is_file(follow_symlinks=False))
            for entry in entries
        }
    if set(actual_entries) != expected_names:
        missing = sorted(expected_names - set(actual_entries))
        unexpected = sorted(set(actual_entries) - expected_names)
        raise SmolVLMBundleError(
            f"SmolVLM bundle file set mismatch; missing={missing!r}, unexpected={unexpected!r}"
        )
    for name, (is_symlink, is_regular) in actual_entries.items():
        if is_symlink or not is_regular:
            raise SmolVLMBundleError(f"SmolVLM bundle entry must be a regular file: {name}")

    lock = _read_lock(bundle / SMOLVLM_PROVENANCE_FILENAME)
    if lock != expected_manifest:
        raise SmolVLMBundleError(
            "SmolVLM provenance lock does not match the fixed Firefly manifest"
        )
    for expected in SMOLVLM_ASSET_FILES:
        actual_size, actual_digest = _hash_regular_file(bundle / expected.path)
        if actual_size != expected.size or actual_digest != expected.sha256:
            raise SmolVLMBundleError(
                f"SmolVLM bundle asset does not match its lock: {expected.path} "
                f"(size={actual_size}, sha256={actual_digest})"
            )
    directory_after = _directory_identity(bundle, label="SmolVLM bundle")
    if directory_before != directory_after:
        raise SmolVLMBundleError("SmolVLM bundle directory changed during validation")
    return bundle.resolve(strict=True), copy.deepcopy(lock)


def materialize_smolvlm_bundle(
    source_snapshot: Path,
    output: Path,
) -> tuple[Path, dict[str, object]]:
    """Verify a Hugging Face snapshot and atomically publish a new local bundle."""
    source = _require_no_symlink_components(
        Path(source_snapshot), label="SmolVLM source snapshot"
    )
    _directory_identity(source, label="SmolVLM source snapshot")
    target = _absolute_without_resolving(Path(output))
    parent = _require_no_symlink_components(target.parent, label="SmolVLM output parent")
    _directory_identity(parent, label="SmolVLM output parent")
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite existing bundle", target)
    if source == parent or source in target.parents:
        raise SmolVLMBundleError("SmolVLM output must not be inside the source snapshot")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    published = False
    try:
        for expected in SMOLVLM_ASSET_FILES:
            _copy_and_verify(source / expected.path, staging / expected.path, expected)
        manifest = expected_smolvlm_manifest()
        _write_lock(staging / SMOLVLM_PROVENANCE_FILENAME, manifest)
        os.chmod(staging, 0o755)
        _fsync_directory(staging)
        validate_materialized_smolvlm_bundle(staging)
        _atomic_publish_noreplace(staging, target)
        published = True
        _fsync_directory(parent)
        return target.resolve(strict=True), manifest
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)
