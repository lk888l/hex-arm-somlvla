from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from lerobot_robot_hexbase import smolvla_base_bundle
from scripts import materialize_firefly_smolvla_base


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def tiny_base_snapshot(tmp_path: Path, monkeypatch):
    config = {
        "type": "smolvla",
        "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "chunk_size": 50,
        "vlm_model_name": smolvla_base_bundle.SMOLVLM_REPO_ID,
        "load_vlm_weights": True,
    }
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "tokenizer_name": smolvla_base_bundle.SMOLVLM_REPO_ID,
                    "padding": "max_length",
                },
            },
            {
                "registry_name": "normalizer_processor",
                "config": {"features": config["input_features"]},
                "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
            },
        ],
    }
    payloads = {
        "config.json": json.dumps(config, indent=4).encode(),
        "model.safetensors": b"fake-complete-policy-weights",
        "policy_postprocessor.json": b'{"name":"policy_postprocessor","steps":[]}\n',
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": b"post-stats",
        "policy_preprocessor.json": json.dumps(preprocessor, indent=2).encode(),
        "policy_preprocessor_step_5_normalizer_processor.safetensors": b"pre-stats",
    }
    specs = tuple(
        smolvla_base_bundle.SmolVLABaseAssetFile(name, len(payload), _digest(payload))
        for name, payload in sorted(payloads.items())
    )
    monkeypatch.setattr(smolvla_base_bundle, "SMOLVLA_BASE_ASSET_FILES", specs)
    monkeypatch.setattr(
        smolvla_base_bundle,
        "_UPSTREAM_CONFIG_CANONICAL_SHA256",
        smolvla_base_bundle._canonical_json_digest(config),
    )
    monkeypatch.setattr(
        smolvla_base_bundle,
        "_UPSTREAM_PREPROCESSOR_CANONICAL_SHA256",
        smolvla_base_bundle._canonical_json_digest(preprocessor),
    )

    cache = tmp_path / "cache"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / smolvla_base_bundle.SMOLVLA_BASE_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob_paths = {}
    for index, (name, payload) in enumerate(sorted(payloads.items())):
        blob = blobs / f"blob-{index}"
        blob.write_bytes(payload)
        blob_paths[name] = blob
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
    (snapshot / "README.md").write_text("ignored upstream documentation", encoding="utf-8")

    vlm_bundle = tmp_path / "pinned-vlm"
    vlm_bundle.mkdir()
    vlm_manifest = {
        "schema_version": 1,
        "bundle_type": "test_smolvlm_assets",
        "repo_id": smolvla_base_bundle.SMOLVLM_REPO_ID,
        "revision": "f" * 40,
        "files": [{"path": "config.json", "size": 2, "sha256": "a" * 64}],
    }

    def validate_fake_bundle(path):
        assert Path(path).resolve(strict=True) == vlm_bundle.resolve(strict=True)
        return vlm_bundle.resolve(strict=True), copy.deepcopy(vlm_manifest)

    monkeypatch.setattr(
        smolvla_base_bundle,
        "validate_materialized_smolvlm_bundle",
        validate_fake_bundle,
    )
    return snapshot, payloads, blob_paths, vlm_bundle, vlm_manifest


def test_fixed_base_manifest_pins_complete_policy_artifacts():
    files = smolvla_base_bundle.SMOLVLA_BASE_ASSET_FILES

    assert smolvla_base_bundle.SMOLVLA_BASE_REPO_ID == "lerobot/smolvla_base"
    assert (
        smolvla_base_bundle.SMOLVLA_BASE_REVISION
        == "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    )
    assert [item.path for item in files] == [
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    ]
    assert sum(item.size for item in files) == 906_718_631
    model = next(item for item in files if item.path == "model.safetensors")
    assert model.size == 906_712_520
    assert model.sha256 == "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"


def test_materialize_patches_only_offline_asset_references(tmp_path: Path, tiny_base_snapshot):
    snapshot, payloads, _blobs, vlm_bundle, vlm_manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"

    canonical, provenance = smolvla_base_bundle.materialize_smolvla_base(
        snapshot, vlm_bundle, output
    )

    assert canonical == output.resolve(strict=True)
    expected_names = set(payloads) | {
        smolvla_base_bundle.SMOLVLA_BASE_PROVENANCE_FILENAME
    }
    assert {entry.name for entry in output.iterdir()} == expected_names
    assert "README.md" not in expected_names
    for entry in output.iterdir():
        assert stat.S_ISREG(entry.lstat().st_mode)
        assert not entry.is_symlink()
        assert entry.stat().st_mode & stat.S_IWUSR == 0

    local_vlm = str(vlm_bundle.resolve(strict=True))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["vlm_model_name"] == local_vlm
    assert config["load_vlm_weights"] is False
    assert config["input_features"] == {
        "observation.state": {"type": "STATE", "shape": [6]}
    }
    processor = json.loads(
        (output / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    tokenizer = next(
        step for step in processor["steps"] if step["registry_name"] == "tokenizer_processor"
    )
    assert tokenizer["config"]["tokenizer_name"] == local_vlm
    assert smolvla_base_bundle.SMOLVLM_REPO_ID not in (
        output / "config.json"
    ).read_text(encoding="utf-8")
    assert smolvla_base_bundle.SMOLVLM_REPO_ID not in (
        output / "policy_preprocessor.json"
    ).read_text(encoding="utf-8")
    assert (output / "model.safetensors").read_bytes() == payloads["model.safetensors"]

    assert provenance["upstream"]["repo_id"] == "lerobot/smolvla_base"
    assert provenance["smolvlm_bundle"]["manifest"] == vlm_manifest
    assert provenance["smolvlm_bundle"]["manifest_sha256"] == (
        smolvla_base_bundle._canonical_json_digest(vlm_manifest)
    )
    assert len(provenance["patches"]) == 3
    lock = json.loads(
        (output / smolvla_base_bundle.SMOLVLA_BASE_PROVENANCE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert lock == provenance
    assert smolvla_base_bundle.validate_materialized_smolvla_base(output) == (
        canonical,
        provenance,
    )


def test_materialize_rejects_source_hash_mismatch_without_output(
    tmp_path: Path, tiny_base_snapshot
):
    snapshot, _payloads, blobs, vlm_bundle, _manifest = tiny_base_snapshot
    blobs["model.safetensors"].write_bytes(b"wrong weights")
    output = tmp_path / "pinned-base"

    with pytest.raises(smolvla_base_bundle.SmolVLABaseBundleError, match="does not match pinned"):
        smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".pinned-base.staging-*"))


def test_materialize_refuses_existing_output(tmp_path: Path, tiny_base_snapshot):
    snapshot, _payloads, _blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)

    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_failed_atomic_publish_cleans_staging(tmp_path: Path, tiny_base_snapshot, monkeypatch):
    snapshot, _payloads, _blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"

    def fail_publish(_staging, _output):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(smolvla_base_bundle, "_atomic_publish_noreplace", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".pinned-base.staging-*"))


def test_validator_rejects_unapproved_config_change(tmp_path: Path, tiny_base_snapshot):
    snapshot, _payloads, _blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"
    smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chunk_size"] = 1
    config_path.chmod(0o644)
    config_path.write_bytes(smolvla_base_bundle._pretty_json(config))

    with pytest.raises(smolvla_base_bundle.SmolVLABaseBundleError, match="unapproved patch"):
        smolvla_base_bundle.validate_materialized_smolvla_base(output)


def test_validator_rejects_wrong_bound_bundle_digest(tmp_path: Path, tiny_base_snapshot):
    snapshot, _payloads, _blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"
    smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)
    lock_path = output / smolvla_base_bundle.SMOLVLA_BASE_PROVENANCE_FILENAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["smolvlm_bundle"]["manifest_sha256"] = "0" * 64
    lock_path.chmod(0o644)
    lock_path.write_bytes(smolvla_base_bundle._pretty_json(lock))

    with pytest.raises(smolvla_base_bundle.SmolVLABaseBundleError, match="manifest or digest"):
        smolvla_base_bundle.validate_materialized_smolvla_base(output)


def test_validator_rejects_extra_entry_and_policy_symlink(tmp_path: Path, tiny_base_snapshot):
    snapshot, _payloads, blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"
    smolvla_base_bundle.materialize_smolvla_base(snapshot, vlm_bundle, output)
    (output / "README.md").write_text("unexpected", encoding="utf-8")
    with pytest.raises(smolvla_base_bundle.SmolVLABaseBundleError, match="file set mismatch"):
        smolvla_base_bundle.validate_materialized_smolvla_base(output)

    (output / "README.md").unlink()
    model = output / "model.safetensors"
    model.unlink()
    model.symlink_to(blobs["model.safetensors"])
    with pytest.raises(smolvla_base_bundle.SmolVLABaseBundleError, match="must be regular"):
        smolvla_base_bundle.validate_materialized_smolvla_base(output)


def test_cli_reports_provenance_and_refuses_second_run(
    tmp_path: Path, tiny_base_snapshot, capsys
):
    snapshot, _payloads, _blobs, vlm_bundle, _manifest = tiny_base_snapshot
    output = tmp_path / "pinned-base"
    argv = [
        "--source-snapshot",
        str(snapshot),
        "--smolvlm-bundle",
        str(vlm_bundle),
        "--output",
        str(output),
    ]

    assert materialize_firefly_smolvla_base.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["base_path"] == str(output.resolve(strict=True))
    assert report["provenance"]["upstream"]["revision"] == (
        smolvla_base_bundle.SMOLVLA_BASE_REVISION
    )

    assert materialize_firefly_smolvla_base.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "refusing to overwrite existing bundle" in captured.err
