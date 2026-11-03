from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from lerobot_robot_hexbase import smolvlm_bundle
from scripts import materialize_firefly_smolvlm


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def tiny_snapshot(tmp_path: Path, monkeypatch):
    payloads = {
        "config.json": b'{"model_type":"smolvlm"}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "vocab.json": b'{"hello":0}\n',
    }
    specs = tuple(
        smolvlm_bundle.SmolVLMAssetFile(name, len(payload), _digest(payload))
        for name, payload in sorted(payloads.items())
    )
    monkeypatch.setattr(smolvlm_bundle, "SMOLVLM_ASSET_FILES", specs)

    cache = tmp_path / "cache"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / smolvlm_bundle.SMOLVLM_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob_paths = {}
    for index, (name, payload) in enumerate(sorted(payloads.items())):
        blob = blobs / f"blob-{index}"
        blob.write_bytes(payload)
        blob_paths[name] = blob
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
    (snapshot / "model.safetensors").write_bytes(b"deliberately not materialized")
    return snapshot, payloads, blob_paths


def test_fixed_manifest_pins_expected_upstream_assets():
    manifest = smolvlm_bundle.expected_smolvlm_manifest()

    assert manifest["repo_id"] == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    assert manifest["revision"] == "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    assert manifest["schema_version"] == 1
    files = manifest["files"]
    assert [item["path"] for item in files] == [
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    assert sum(item["size"] for item in files) == 4_854_541
    assert all(len(item["sha256"]) == 64 for item in files)


def test_materialize_accepts_cache_symlinks_and_publishes_regular_files(
    tmp_path: Path, tiny_snapshot
):
    snapshot, payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"

    canonical, manifest = smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)

    assert canonical == output.resolve(strict=True)
    assert manifest == smolvlm_bundle.expected_smolvlm_manifest()
    expected_names = set(payloads) | {smolvlm_bundle.SMOLVLM_PROVENANCE_FILENAME}
    assert {entry.name for entry in output.iterdir()} == expected_names
    assert "model.safetensors" not in expected_names
    for name, payload in payloads.items():
        asset = output / name
        assert stat.S_ISREG(asset.lstat().st_mode)
        assert not asset.is_symlink()
        assert asset.read_bytes() == payload
        assert asset.stat().st_mode & stat.S_IWUSR == 0
    lock = json.loads(
        (output / smolvlm_bundle.SMOLVLM_PROVENANCE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert lock == manifest
    assert smolvlm_bundle.validate_materialized_smolvlm_bundle(output) == (
        canonical,
        manifest,
    )
    assert not list(tmp_path.glob(".materialized.staging-*"))


def test_materialize_rejects_wrong_source_hash_without_publishing(
    tmp_path: Path, tiny_snapshot
):
    snapshot, _payloads, blobs = tiny_snapshot
    blobs["tokenizer.json"].write_bytes(b"tampered")
    output = tmp_path / "materialized"

    with pytest.raises(smolvlm_bundle.SmolVLMBundleError, match="does not match pinned"):
        smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".materialized.staging-*"))


def test_materialize_never_overwrites_existing_output(tmp_path: Path, tiny_snapshot):
    snapshot, _payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"
    output.mkdir()
    sentinel = output / "user-data"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_failed_atomic_publish_removes_only_staging(
    tmp_path: Path, tiny_snapshot, monkeypatch
):
    snapshot, _payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"

    def fail_publish(_staging, _output):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(smolvlm_bundle, "_atomic_publish_noreplace", fail_publish)
    with pytest.raises(OSError, match="simulated publication failure"):
        smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".materialized.staging-*"))


def test_validator_rejects_tampered_asset(tmp_path: Path, tiny_snapshot):
    snapshot, _payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"
    smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)
    asset = output / "config.json"
    asset.chmod(0o644)
    asset.write_bytes(b"tampered")

    with pytest.raises(smolvlm_bundle.SmolVLMBundleError, match="does not match its lock"):
        smolvlm_bundle.validate_materialized_smolvlm_bundle(output)


def test_validator_rejects_extra_file_and_symlink(tmp_path: Path, tiny_snapshot):
    snapshot, _payloads, blobs = tiny_snapshot
    output = tmp_path / "materialized"
    smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)
    (output / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(smolvlm_bundle.SmolVLMBundleError, match="file set mismatch"):
        smolvlm_bundle.validate_materialized_smolvlm_bundle(output)

    (output / "unexpected.json").unlink()
    asset = output / "config.json"
    asset.unlink()
    asset.symlink_to(blobs["config.json"])
    with pytest.raises(smolvlm_bundle.SmolVLMBundleError, match="regular file"):
        smolvlm_bundle.validate_materialized_smolvlm_bundle(output)


def test_validator_rejects_modified_provenance_lock(tmp_path: Path, tiny_snapshot):
    snapshot, _payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"
    smolvlm_bundle.materialize_smolvlm_bundle(snapshot, output)
    lock_path = output / smolvlm_bundle.SMOLVLM_PROVENANCE_FILENAME
    manifest = smolvlm_bundle.expected_smolvlm_manifest()
    manifest["revision"] = "0" * 40
    lock_path.chmod(0o644)
    lock_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(smolvlm_bundle.SmolVLMBundleError, match="fixed Firefly manifest"):
        smolvlm_bundle.validate_materialized_smolvlm_bundle(output)


def test_cli_prints_validated_manifest_and_refuses_second_run(
    tmp_path: Path, tiny_snapshot, capsys
):
    snapshot, _payloads, _blobs = tiny_snapshot
    output = tmp_path / "materialized"
    argv = ["--source-snapshot", str(snapshot), "--output", str(output)]

    assert materialize_firefly_smolvlm.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "bundle_path": str(output.resolve(strict=True)),
        "manifest": smolvlm_bundle.expected_smolvlm_manifest(),
    }

    assert materialize_firefly_smolvlm.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "refusing to overwrite existing bundle" in captured.err
