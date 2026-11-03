from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import materialize_firefly_y6_arm_only_v1 as arm_only


TEST_TASK = "fixture red block task"


def _vector(seed: float) -> list[float]:
    return [seed + index * 0.125 for index in range(7)]


def _stats_feature(seed: float, frame_count: int) -> dict:
    return {
        statistic: _vector(seed + offset)
        for offset, statistic in enumerate(arm_only.VECTOR_STATISTICS)
    } | {"count": [frame_count]}


def _huggingface_metadata() -> dict[bytes, bytes]:
    features = {
        key: {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": 7,
            "_type": "List",
        }
        for key in arm_only.PROJECTED_FEATURES
    }
    features.update(
        {
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
    )
    return {
        b"huggingface": json.dumps(
            {"info": {"features": features}}, ensure_ascii=False
        ).encode("utf-8")
    }


def _write_fixture_source(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    for camera in ("scene", "wrist"):
        path = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"independent-{camera}-video".encode())

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": list(arm_only.JOINT_NAMES_7),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": list(arm_only.JOINT_NAMES_7),
            },
            "observation.images.scene": {
                "dtype": "video",
                "shape": [2, 3, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [2, 3, 3],
                "names": ["height", "width", "channels"],
            },
        },
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/"
            "file-{file_index:03d}.mp4"
        ),
        "robot_type": "hex_zenoh_observer",
        "splits": {"train": "0:2"},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    stats = {
        "action": _stats_feature(1.0, 4),
        "observation.state": _stats_feature(11.0, 4),
        "timestamp": {"min": [0.0], "count": [4]},
    }
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (root / "meta/extra.json").write_bytes(b'{"must":"remain byte exact"}\n')

    pq.write_table(
        pa.table({"task_index": pa.array([0], pa.int64()), "task": [TEST_TASK]}),
        root / "meta/tasks.parquet",
    )

    episode_columns: dict[str, pa.Array] = {
        "episode_index": pa.array([0, 1], pa.int64()),
        "tasks": pa.array([[TEST_TASK], [TEST_TASK]], pa.list_(pa.string())),
        "length": pa.array([2, 2], pa.int64()),
        "data/chunk_index": pa.array([0, 0], pa.int64()),
        "data/file_index": pa.array([0, 0], pa.int64()),
    }
    for feature_offset, feature in enumerate(arm_only.PROJECTED_FEATURES):
        for statistic_offset, statistic in enumerate(arm_only.VECTOR_STATISTICS):
            episode_columns[f"stats/{feature}/{statistic}"] = pa.array(
                [
                    _vector(20.0 + feature_offset * 20 + statistic_offset),
                    _vector(30.0 + feature_offset * 20 + statistic_offset),
                ],
                type=pa.list_(pa.float64()),
            )
        episode_columns[f"stats/{feature}/count"] = pa.array(
            [[2], [2]], type=pa.list_(pa.int64())
        )
    pq.write_table(
        pa.table(episode_columns),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )

    action = [
        [0.1, -0.2, 1.1, 0.4, 0.5, -0.6, 0.301],
        [0.2, -0.1, 1.2, 0.5, 0.6, -0.5, 0.401],
        [0.3, 0.0, 1.3, 0.6, 0.7, -0.4, 0.302],
        [0.4, 0.1, 1.4, 0.7, 0.8, -0.3, 0.402],
    ]
    state = [
        [value - 0.001 for value in row[:-1]] + [row[-1] + 0.002]
        for row in action
    ]
    data = pa.table(
        {
            "action": pa.array(action, type=pa.list_(pa.float32(), 7)),
            "observation.state": pa.array(
                state, type=pa.list_(pa.float32(), 7)
            ),
            "timestamp": pa.array([0.0, 1 / 30, 0.0, 1 / 30], pa.float32()),
            "frame_index": pa.array([0, 1, 0, 1], pa.int64()),
            "episode_index": pa.array([0, 0, 1, 1], pa.int64()),
            "index": pa.array([0, 1, 2, 3], pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], pa.int64()),
        }
    ).replace_schema_metadata(_huggingface_metadata())
    pq.write_table(data, root / "data/chunk-000/file-000.parquet")


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "firefly_y6_fullcycle_v4_arm_only_v1"
    source.mkdir()
    _write_fixture_source(source)
    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps(
            {
                "schema": "test",
                "joint_names": list(arm_only.JOINT_NAMES_7),
                "effective_step_limits": [0.0435] * 6 + [0.0165],
            }
        ),
        encoding="utf-8",
    )
    source_capture = arm_only._inventory(source)
    controller_sha, _size, _identity = arm_only._hash_regular_file(controller)
    contract = replace(
        arm_only.PRODUCTION_CONTRACT,
        expected_task=TEST_TASK,
        expected_episodes=2,
        expected_frames=4,
        expected_source_manifest_sha256=source_capture.digest,
        expected_source_directories_sha256=source_capture.directories_digest,
        expected_source_files=len(source_capture.files),
        expected_data_files=1,
        expected_episode_metadata_files=1,
        expected_video_files=2,
        expected_controller_contract_sha256=controller_sha,
        upstream_training_input_digest="a" * 64,
    )
    return source, destination, controller, contract


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, tuple[int, ...]]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[path.relative_to(root).as_posix()] = (
            path.read_bytes(),
            arm_only._identity(path, directory=False),
        )
    return result


def test_materializes_exact_7_to_6_self_contained_dataset(tmp_path: Path):
    source, destination, controller, contract = _fixture(tmp_path)
    source_before = _tree_snapshot(source)

    result = arm_only.materialize_arm_only_dataset(
        source,
        destination,
        controller_contract_path=controller,
        contract=contract,
    )

    assert result == destination
    assert _tree_snapshot(source) == source_before
    validation = arm_only.validate_arm_only_dataset(
        source,
        destination,
        controller_contract_path=controller,
        contract=contract,
    )
    assert validation["safe_for_arm_only_training"] is True
    assert validation["action_dimensions"] == 6
    assert validation["state_dimensions"] == 6
    assert validation["removed_feature"] == "jaw_joint1.pos"

    info = json.loads((destination / "meta/info.json").read_text())
    for key in arm_only.PROJECTED_FEATURES:
        assert info["features"][key]["shape"] == [6]
        assert info["features"][key]["names"] == list(arm_only.ARM_JOINT_NAMES)
    stats = json.loads((destination / "meta/stats.json").read_text())
    for key in arm_only.PROJECTED_FEATURES:
        assert stats[key]["count"] == [4]
        for statistic in arm_only.VECTOR_STATISTICS:
            assert len(stats[key][statistic]) == 6

    source_data = pq.read_table(source / "data/chunk-000/file-000.parquet")
    target_data = pq.read_table(destination / "data/chunk-000/file-000.parquet")
    for key in arm_only.PROJECTED_FEATURES:
        assert target_data.schema.field(key).type == pa.list_(pa.float32(), 6)
        assert target_data[key].to_pylist() == [row[:6] for row in source_data[key].to_pylist()]
    for key in set(source_data.column_names) - set(arm_only.PROJECTED_FEATURES):
        assert source_data[key].equals(target_data[key])
    hf_metadata = json.loads(target_data.schema.metadata[b"huggingface"])
    for key in arm_only.PROJECTED_FEATURES:
        assert hf_metadata["info"]["features"][key]["length"] == 6

    source_episode = pq.read_table(
        source / "meta/episodes/chunk-000/file-000.parquet"
    )
    target_episode = pq.read_table(
        destination / "meta/episodes/chunk-000/file-000.parquet"
    )
    for key in arm_only._episode_vector_fields():
        assert target_episode[key].to_pylist() == [
            row[:6] for row in source_episode[key].to_pylist()
        ]
    for feature in arm_only.PROJECTED_FEATURES:
        count = f"stats/{feature}/count"
        assert target_episode[count].equals(source_episode[count])

    for relative in (
        "meta/extra.json",
        "meta/tasks.parquet",
        "videos/observation.images.scene/chunk-000/file-000.mp4",
        "videos/observation.images.wrist/chunk-000/file-000.mp4",
    ):
        source_file = source / relative
        target_file = destination / relative
        assert target_file.read_bytes() == source_file.read_bytes()
        assert target_file.stat().st_ino != source_file.stat().st_ino
        assert target_file.stat().st_nlink == 1


def test_manifest_excludes_itself_and_binds_source_target_files(tmp_path: Path):
    source, destination, controller, contract = _fixture(tmp_path)
    arm_only.materialize_arm_only_dataset(
        source,
        destination,
        controller_contract_path=controller,
        contract=contract,
    )
    manifest = json.loads((destination / arm_only.MANIFEST_RELATIVE_PATH).read_text())

    assert manifest["kind"] == "firefly_y6_arm_only_dataset_projection"
    assert manifest["schema_version"] == 1
    source_paths = [item["path"] for item in manifest["source"]["files"]]
    destination_paths = [item["path"] for item in manifest["destination"]["files"]]
    assert destination_paths == source_paths
    assert arm_only.MANIFEST_RELATIVE_PATH.as_posix() not in destination_paths
    assert manifest["source"]["files_manifest_sha256"] == contract.expected_source_manifest_sha256
    assert manifest["source"]["directories_sha256"] == (
        contract.expected_source_directories_sha256
    )
    assert manifest["destination"]["directories"] == manifest["source"]["directories"]
    assert manifest["destination"]["directories_sha256"] == arm_only._sha256_json(
        manifest["destination"]["directories"]
    )
    assert manifest["destination"]["files_manifest_sha256"] == arm_only._sha256_json(
        manifest["destination"]["files"]
    )
    unsigned = dict(manifest)
    claimed = unsigned.pop("document_sha256")
    assert claimed == arm_only._sha256_json(unsigned)
    assert manifest["transform"]["removed_feature"] == {
        "name": "jaw_joint1.pos",
        "source_index": 6,
        "removed_from": ["action", "observation.state"],
        "runtime_authority": (
            "jaw must be controlled independently; this dataset emits no jaw action"
        ),
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "copied_video",
        "projected_data",
        "manifest",
        "rogue_root_file",
        "rogue_empty_directory",
        "manifest_hardlink",
    ],
)
def test_validator_rejects_tampering(tmp_path: Path, tamper: str):
    source, destination, controller, contract = _fixture(tmp_path)
    arm_only.materialize_arm_only_dataset(
        source,
        destination,
        controller_contract_path=controller,
        contract=contract,
    )

    if tamper == "copied_video":
        path = destination / "videos/observation.images.scene/chunk-000/file-000.mp4"
        path.write_bytes(path.read_bytes() + b"tamper")
    elif tamper == "projected_data":
        path = destination / "data/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        rows = table["action"].to_pylist()
        rows[0][0] += 0.25
        table = table.set_column(
            table.schema.get_field_index("action"),
            table.schema.field("action"),
            pa.array(rows, type=pa.list_(pa.float32(), 6)),
        )
        pq.write_table(table, path)
    elif tamper == "manifest":
        path = destination / arm_only.MANIFEST_RELATIVE_PATH
        manifest = json.loads(path.read_text())
        manifest["transform"]["removed_feature"]["source_index"] = 5
        path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "rogue_root_file":
        (destination / "rogue.bin").write_bytes(b"not in the target manifest")
    elif tamper == "rogue_empty_directory":
        (destination / "meta/rogue-empty").mkdir()
    else:
        path = destination / arm_only.MANIFEST_RELATIVE_PATH
        os.link(path, tmp_path / "external-manifest-hardlink.json")

    with pytest.raises(arm_only.ArmOnlyMaterializationError):
        arm_only.validate_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )


def test_failure_removes_staging_and_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, destination, controller, contract = _fixture(tmp_path)
    source_before = _tree_snapshot(source)
    original = arm_only._materialize_file
    calls = 0

    def fail_after_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected transform failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(arm_only, "_materialize_file", fail_after_one)
    with pytest.raises(RuntimeError, match="injected transform failure"):
        arm_only.materialize_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.partial-*")) == []
    assert _tree_snapshot(source) == source_before


def test_source_empty_directory_is_rejected_by_pinned_topology(tmp_path: Path):
    source, destination, controller, contract = _fixture(tmp_path)
    (source / "meta/unpinned-empty-directory").mkdir()

    with pytest.raises(
        arm_only.ArmOnlyMaterializationError,
        match="source directory topology SHA-256",
    ):
        arm_only.materialize_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.partial-*")) == []


def test_late_staging_tamper_is_rejected_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, destination, controller, contract = _fixture(tmp_path)
    original = arm_only._verify_capture_unchanged
    source_verifications = 0

    def tamper_after_final_source_verification(root, capture, **kwargs):
        nonlocal source_verifications
        result = original(root, capture, **kwargs)
        if root == source:
            source_verifications += 1
            if source_verifications == 2:
                partial = next(
                    tmp_path.glob(f".{destination.name}.partial-*")
                )
                video = (
                    partial
                    / "dataset/videos/observation.images.scene/"
                    "chunk-000/file-000.mp4"
                )
                video.write_bytes(video.read_bytes() + b"late-tamper")
        return result

    monkeypatch.setattr(
        arm_only,
        "_verify_capture_unchanged",
        tamper_after_final_source_verification,
    )
    with pytest.raises(
        arm_only.SourceChangedError,
        match="captured dataset changed",
    ):
        arm_only.materialize_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )

    assert source_verifications == 2
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.partial-*")) == []


def test_publish_race_never_replaces_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, destination, controller, contract = _fixture(tmp_path)

    def race(_staging: Path, target: Path):
        target.mkdir()
        (target / "owned-by-user").write_text("keep", encoding="utf-8")
        raise FileExistsError("injected publish race")

    monkeypatch.setattr(arm_only, "_publish_noreplace", race)
    with pytest.raises(FileExistsError, match="injected publish race"):
        arm_only.materialize_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )

    assert (destination / "owned-by-user").read_text() == "keep"
    assert list(tmp_path.glob(f".{destination.name}.partial-*")) == []


def test_existing_destination_is_rejected_before_any_write(tmp_path: Path):
    source, destination, controller, contract = _fixture(tmp_path)
    destination.mkdir()
    marker = destination / "owned-by-user"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        arm_only.materialize_arm_only_dataset(
            source,
            destination,
            controller_contract_path=controller,
            contract=contract,
        )

    assert marker.read_text() == "keep"
    assert list(tmp_path.glob(f".{destination.name}.partial-*")) == []


def test_cli_requires_explicit_materialize_flag():
    assert arm_only.parse_args([]).materialize is False
    assert arm_only.parse_args(["--materialize"]).materialize is True
    with pytest.raises(SystemExit):
        arm_only.parse_args(["--materialize", "--validate"])


def test_parquet_reader_requires_the_captured_sha(tmp_path: Path):
    source, _destination, _controller, _contract = _fixture(tmp_path)
    path = source / "data/chunk-000/file-000.parquet"

    with pytest.raises(arm_only.SourceChangedError, match="captured inventory"):
        arm_only._read_parquet(
            path,
            label="test data",
            expected_sha256="0" * 64,
        )
