from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_robot_hexbase.firefly_dataset_audit import (
    DatasetAuditInputError,
    EXPECTED_JOINT_NAMES,
    audit_firefly_dataset,
    firefly_contract_from_mapping,
    strict_json_dumps,
)


TASK = "Pick up the red block and place it in the blue tray."


def _contract_mapping() -> dict:
    return {
        "source": "saved live controller description for test",
        "joint_names": list(EXPECTED_JOINT_NAMES),
        "pos_min": [-2.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.0],
        "pos_max": [2.0, 2.0, 3.2, 2.0, 2.0, 2.0, 0.6],
        "effective_step_limits": [0.0435] * 6 + [0.0165],
    }


def _rows() -> list[dict]:
    rows = []
    global_index = 0
    for episode_index, offset in ((0, 0.0), (1, 0.2)):
        for frame_index in range(3):
            action = [
                offset + 0.01 * frame_index,
                0.1,
                1.0,
                0.2,
                -0.2,
                0.05,
                0.3 + 0.005 * frame_index,
            ]
            state = [value - 0.003 for value in action]
            rows.append(
                {
                    "action": action,
                    "observation.state": state,
                    "timestamp": frame_index / 30,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
    return rows


def _write_dataset(
    cache_root: Path,
    *,
    rows: list[dict] | None = None,
    action_names: list[str] | None = None,
    task: str = TASK,
) -> Path:
    dataset_root = cache_root / "local" / "firefly_test"
    data_dir = dataset_root / "data" / "chunk-000"
    episode_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    materialized = _rows() if rows is None else rows

    table = pa.table(
        {
            "action": pa.array(
                [row["action"] for row in materialized], type=pa.list_(pa.float32(), 7)
            ),
            "observation.state": pa.array(
                [row["observation.state"] for row in materialized],
                type=pa.list_(pa.float32(), 7),
            ),
            "timestamp": pa.array(
                [row["timestamp"] for row in materialized], type=pa.float32()
            ),
            "frame_index": pa.array(
                [row["frame_index"] for row in materialized], type=pa.int64()
            ),
            "episode_index": pa.array(
                [row["episode_index"] for row in materialized], type=pa.int64()
            ),
            "index": pa.array([row["index"] for row in materialized], type=pa.int64()),
            "task_index": pa.array(
                [row["task_index"] for row in materialized], type=pa.int64()
            ),
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")

    episode_indexes = sorted({row["episode_index"] for row in materialized})
    episode_lengths = [
        sum(row["episode_index"] == episode_index for row in materialized)
        for episode_index in episode_indexes
    ]
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(episode_indexes, type=pa.int64()),
                "tasks": pa.array([[task] for _ in episode_indexes], type=pa.list_(pa.string())),
                "length": pa.array(episode_lengths, type=pa.int64()),
            }
        ),
        episode_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([task], type=pa.string()),
            }
        ),
        dataset_root / "meta" / "tasks.parquet",
    )

    names = list(EXPECTED_JOINT_NAMES) if action_names is None else action_names
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": names},
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": list(EXPECTED_JOINT_NAMES),
        },
    }
    for image_key in ("observation.images.scene", "observation.images.wrist"):
        features[image_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False, "video.fps": 30},
        }
        video = dataset_root / "videos" / image_key / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"test-only-placeholder")
    for scalar, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[scalar] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": features,
        "total_episodes": len(episode_indexes),
        "total_frames": len(materialized),
        "total_tasks": 1,
        "robot_type": "hex_zenoh_observer",
    }
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    return dataset_root


def _audit(dataset_root: Path, *, expected_task: str | None = TASK):
    return audit_firefly_dataset(
        dataset_root,
        "local/firefly_test",
        controller_contract=firefly_contract_from_mapping(_contract_mapping()),
        expected_task=expected_task,
    )


def _error_codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["summary"]["errors"]}


def _warning_codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["summary"]["warnings"]}


def test_valid_dataset_passes_and_reports_action_state_error(tmp_path):
    dataset_root = _write_dataset(tmp_path)

    report = _audit(dataset_root)

    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["safe_for_training"] is True
    assert report["checks"]["features"]["action"]["passed"] is True
    assert report["checks"]["episode_30hz_sequence"]["passed"] is True
    assert report["checks"]["position_bounds"]["action_margin_violation_values"] == 0
    assert report["checks"]["action_step"]["consecutive_action_violation_values"] == 0
    relationship = report["checks"]["action_state_relationship"]
    assert relationship["all_frames_exactly_equal"] is False
    assert relationship["error_action_minus_state"]["overall"]["mean_abs"] > 0
    assert json.loads(strict_json_dumps(report))["summary"]["verdict"] == "pass"


def test_first_action_state_tracking_error_is_diagnostic_only(tmp_path):
    rows = _rows()
    rows[0]["observation.state"][0] = -0.20
    dataset_root = _write_dataset(tmp_path, rows=rows)

    report = _audit(dataset_root)

    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["safe_for_training"] is True
    assert "first_action_step_limit_exceeded" not in _error_codes(report)
    assert (
        "first_action_state_tracking_error_diagnostic" in _warning_codes(report)
    )
    step = report["checks"]["action_step"]
    assert step["passed"] is True
    assert step["consecutive_action_violation_values"] == 0
    tracking = step["first_action_vs_state_tracking_error"]
    assert tracking["diagnostic_only"] is True
    assert tracking["step_reference_exceedance_values"] == 1
    assert tracking["step_reference_exceedance_episodes"] == 1
    assert tracking["max_abs"] > 0.035
    # Schema-v1 consumers can continue reading the old keys; they are now
    # explicitly diagnostic rather than part of the pass/fail calculation.
    assert step["first_action_vs_state_violation_values"] == 1
    assert step["first_action_vs_state_violation_episodes"] == 1


def test_safety_violations_and_nonfinite_values_fail(tmp_path):
    rows = _rows()
    rows[1]["action"][0] = 0.20  # > 0.035 from the previous accepted action
    rows[1]["action"][1] = -1.99  # raw-safe, but below the -1.98 margin bound
    rows[2]["action"][2] = float("nan")
    rows[4]["observation.state"][3] = 2.10  # outside raw controller bound
    dataset_root = _write_dataset(tmp_path, rows=rows)

    report = _audit(dataset_root)

    assert report["summary"]["verdict"] == "fail"
    assert {
        "nonfinite_data",
        "action_position_out_of_margin_bounds",
        "state_position_out_of_raw_bounds",
        "action_step_limit_exceeded",
    }.issubset(_error_codes(report))
    bounds = report["checks"]["position_bounds"]
    assert bounds["action_raw_violation_values"] == 0
    assert bounds["action_margin_violation_values"] >= 1
    assert bounds["state_raw_violation_values"] == 1
    assert report["dataset"]["data"]["nonfinite_values"] == 1
    assert report["checks"]["action_step"]["passed"] is False
    strict_json_dumps(report)


def test_action_copied_from_state_is_rejected(tmp_path):
    rows = _rows()
    for row in rows:
        row["observation.state"] = list(row["action"])
    dataset_root = _write_dataset(tmp_path, rows=rows)

    report = _audit(dataset_root)

    relationship = report["checks"]["action_state_relationship"]
    assert relationship["all_frames_exactly_equal"] is True
    assert relationship["exactly_equal_frame_fraction"] == 1.0
    assert "action_state_identical_dataset" in _error_codes(report)


def test_missing_contract_and_wrong_task_fail_closed(tmp_path):
    dataset_root = _write_dataset(tmp_path)

    report = audit_firefly_dataset(
        dataset_root,
        "local/firefly_test",
        controller_contract=None,
        expected_task="A different instruction.",
    )

    assert report["checks"]["position_bounds"]["skipped"] is True
    assert report["checks"]["action_step"]["skipped"] is True
    assert report["summary"]["safe_for_training"] is False
    assert {"controller_contract_missing", "expected_task_mismatch"}.issubset(
        _error_codes(report)
    )


def test_follower_event_contract_computes_effective_steps():
    raw = {
        "event": "controller_contract",
        "source": "captured execute log",
        "arm_joint_names": [f"joint_{index}" for index in range(1, 7)],
        "arm_pos_min": [-2.0] * 6,
        "arm_pos_max": [2.0] * 6,
        "arm_vel_max": [0.6, 2.0, 2.0, 2.0, 2.0, 2.0],
        "ee_joint_names": ["jaw_joint1"],
        "ee_pos_min": [0.0],
        "ee_pos_max": [0.6],
        "ee_vel_max": [0.3],
    }

    contract = firefly_contract_from_mapping(raw)

    assert contract.input_format == "follower_event"
    assert contract.envelope.joint_names == EXPECTED_JOINT_NAMES
    assert contract.envelope.step_limits[0] == 0.02
    assert contract.envelope.step_limits[1] == 0.0435
    assert contract.envelope.step_limits[-1] == 0.01


def test_current_and_legacy_stricter_recording_step_contracts_are_accepted():
    current = _contract_mapping()
    parsed = firefly_contract_from_mapping(current)
    assert parsed.envelope.step_limits == pytest.approx([0.0435] * 6 + [0.0165])

    previous = {**current, "effective_step_limits": [0.0385] * 6 + [0.0165]}
    parsed_previous = firefly_contract_from_mapping(previous)
    assert parsed_previous.envelope.step_limits == pytest.approx(
        [0.0385] * 6 + [0.0165]
    )

    legacy = {**current, "effective_step_limits": [0.035] * 6 + [0.015]}
    parsed_legacy = firefly_contract_from_mapping(legacy)
    assert parsed_legacy.envelope.step_limits == pytest.approx([0.035] * 6 + [0.015])


@pytest.mark.parametrize(
    "effective_step_limits",
    ([0.0435001] + [0.0435] * 5 + [0.0165], [0.0435] * 6 + [0.0165001]),
)
def test_recording_step_contract_rejects_limits_above_current_approval(
    effective_step_limits,
):
    weakened = {**_contract_mapping(), "effective_step_limits": effective_step_limits}
    with pytest.raises(DatasetAuditInputError, match="weakens current limit"):
        firefly_contract_from_mapping(weakened)


def test_cli_is_read_only_and_exit_status_matches_verdict(tmp_path):
    cache_root = tmp_path / "cache"
    dataset_root = _write_dataset(cache_root)
    contract_path = tmp_path / "controller_contract.json"
    contract_path.write_text(json.dumps(_contract_mapping()), encoding="utf-8")
    before = {
        path.relative_to(dataset_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_firefly_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(cache_root),
            "--repo-id",
            "local/firefly_test",
            "--controller-contract",
            str(contract_path),
            "--expected-task",
            TASK,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["summary"]["safe_for_training"] is True
    after = {
        path.relative_to(dataset_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_feature_order_mismatch_is_reported(tmp_path):
    names = list(EXPECTED_JOINT_NAMES)
    names[0], names[1] = names[1], names[0]
    dataset_root = _write_dataset(tmp_path, action_names=names)

    report = _audit(dataset_root)

    assert report["checks"]["features"]["action"]["passed"] is False
    assert "joint_feature_schema_mismatch" in _error_codes(report)
