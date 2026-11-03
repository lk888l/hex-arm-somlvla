"""Read-only safety audit for Firefly LeRobot v3 datasets.

The auditor deliberately reads metadata and parquet files directly.  It never
instantiates a robot, opens Zenoh, decodes video, repairs metadata, or writes to
the dataset.  A controller contract is required for a positive safety verdict;
without one, the structural checks still run but the result is fail-closed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .shadow_envelope import EnvelopeContract


EXPECTED_FPS = 30.0
EXPECTED_JOINT_NAMES = tuple(
    [f"joint_{index}.pos" for index in range(1, 7)] + ["jaw_joint1.pos"]
)
EXPECTED_IMAGE_FEATURES = {
    "observation.images.scene": (480, 640, 3),
    "observation.images.wrist": (480, 640, 3),
}
CURRENT_POSITION_MARGINS = (0.020,) * 6 + (0.005,)
CURRENT_CONFIGURED_STEP_LIMITS = (0.0435,) * 6 + (0.0165,)
REQUIRED_DATA_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
MAX_EXAMPLES = 12


class DatasetAuditInputError(ValueError):
    """The requested dataset or contract could not be read safely."""


@dataclass(frozen=True)
class FireflyAuditContract:
    envelope: EnvelopeContract
    position_margins: tuple[float, ...]
    input_format: str

    @property
    def margin_min(self) -> tuple[float, ...]:
        return tuple(
            lower + margin
            for lower, margin in zip(
                self.envelope.pos_min, self.position_margins, strict=True
            )
        )

    @property
    def margin_max(self) -> tuple[float, ...]:
        return tuple(
            upper - margin
            for upper, margin in zip(
                self.envelope.pos_max, self.position_margins, strict=True
            )
        )


@dataclass(frozen=True)
class _Frame:
    action: tuple[float, ...]
    state: tuple[float, ...]
    timestamp: float
    frame_index: int
    episode_index: int
    index: int
    task_index: int


class _Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def error(self, code: str, message: str, **details: Any) -> None:
        finding = {"code": code, "message": message}
        if details:
            finding["details"] = details
        self.errors.append(finding)

    def warning(self, code: str, message: str, **details: Any) -> None:
        finding = {"code": code, "message": message}
        if details:
            finding["details"] = details
        self.warnings.append(finding)


def _normalise_joint_name(value: Any) -> str:
    name = str(value)
    return name if name.endswith(".pos") else f"{name}.pos"


def _numeric_sequence(raw: Any, label: str) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DatasetAuditInputError(f"{label} must be a numeric sequence")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetAuditInputError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise DatasetAuditInputError(f"{label} contains NaN or infinite values")
    return values


def _contract_margins(raw: Mapping[str, Any], size: int) -> tuple[float, ...]:
    if "position_margins" in raw:
        margins = _numeric_sequence(raw["position_margins"], "position_margins")
    elif "arm_position_margin_rad" in raw or "ee_position_margin_rad" in raw:
        arm_count = size - 1
        margins = (
            float(raw.get("arm_position_margin_rad", CURRENT_POSITION_MARGINS[0])),
        ) * arm_count + (
            float(raw.get("ee_position_margin_rad", CURRENT_POSITION_MARGINS[-1])),
        )
    else:
        margins = CURRENT_POSITION_MARGINS
    if len(margins) != size:
        raise DatasetAuditInputError("position margins must match contract joint count")
    if not all(math.isfinite(value) and value >= 0 for value in margins):
        raise DatasetAuditInputError("position margins must be finite and non-negative")
    for index, (actual, required) in enumerate(
        zip(margins, CURRENT_POSITION_MARGINS, strict=True)
    ):
        if actual + 1e-12 < required:
            raise DatasetAuditInputError(
                f"position margin {index} weakens the current Firefly minimum "
                f"({actual} < {required})"
            )
    return margins


def firefly_contract_from_mapping(raw: Mapping[str, Any]) -> FireflyAuditContract:
    """Load the existing flat envelope format or a follower contract log event."""

    if not isinstance(raw, Mapping):
        raise DatasetAuditInputError("controller contract JSON must be an object")
    if {"joint_names", "pos_min", "pos_max"}.issubset(raw):
        names = tuple(_normalise_joint_name(value) for value in raw["joint_names"])
        steps = raw.get("effective_step_limits")
        if steps is None:
            raise DatasetAuditInputError(
                "flat controller contract is missing effective_step_limits"
            )
        envelope_raw = {
            "source": raw.get("source", "controller contract JSON"),
            "joint_names": names,
            "pos_min": raw["pos_min"],
            "pos_max": raw["pos_max"],
            "effective_step_limits": steps,
        }
        input_format = "flat_envelope"
    elif {
        "arm_joint_names",
        "arm_pos_min",
        "arm_pos_max",
        "ee_joint_names",
        "ee_pos_min",
        "ee_pos_max",
    }.issubset(raw):
        arm_names = tuple(_normalise_joint_name(value) for value in raw["arm_joint_names"])
        ee_names = tuple(_normalise_joint_name(value) for value in raw["ee_joint_names"])
        names = arm_names + ee_names
        steps = raw.get("effective_step_limits")
        if steps is None:
            try:
                velocities = _numeric_sequence(
                    tuple(raw["arm_vel_max"]) + tuple(raw["ee_vel_max"]),
                    "controller velocity limits",
                )
            except KeyError as exc:
                raise DatasetAuditInputError(
                    "follower controller contract needs arm_vel_max and ee_vel_max "
                    "when effective_step_limits is absent"
                ) from exc
            if len(velocities) != len(names):
                raise DatasetAuditInputError(
                    "controller velocity limits must match joint count"
                )
            configured = CURRENT_CONFIGURED_STEP_LIMITS[: len(names)]
            steps = tuple(
                min(limit, velocity / EXPECTED_FPS)
                for limit, velocity in zip(configured, velocities, strict=True)
            )
        envelope_raw = {
            "source": raw.get("source", "follower controller_contract event"),
            "joint_names": names,
            "pos_min": tuple(raw["arm_pos_min"]) + tuple(raw["ee_pos_min"]),
            "pos_max": tuple(raw["arm_pos_max"]) + tuple(raw["ee_pos_max"]),
            "effective_step_limits": steps,
        }
        input_format = "follower_event"
    else:
        raise DatasetAuditInputError(
            "controller contract must use joint_names/pos_min/pos_max/"
            "effective_step_limits or the follower arm_*/ee_* event fields"
        )

    try:
        envelope = EnvelopeContract.from_mapping(envelope_raw)
    except ValueError as exc:
        raise DatasetAuditInputError(str(exc)) from exc
    if len(envelope.joint_names) != len(CURRENT_CONFIGURED_STEP_LIMITS):
        raise DatasetAuditInputError(
            f"Firefly controller contract has {len(envelope.joint_names)} joints, expected 7"
        )
    for name, actual, maximum in zip(
        envelope.joint_names,
        envelope.step_limits,
        CURRENT_CONFIGURED_STEP_LIMITS,
        strict=True,
    ):
        if actual > maximum + 1e-12:
            raise DatasetAuditInputError(
                f"{name} effective step limit {actual} weakens current limit {maximum}"
            )
    margins = _contract_margins(raw, len(envelope.joint_names))
    for name, lower, upper, margin in zip(
        envelope.joint_names,
        envelope.pos_min,
        envelope.pos_max,
        margins,
        strict=True,
    ):
        if lower + margin >= upper - margin:
            raise DatasetAuditInputError(
                f"{name} controller range is too narrow for margin {margin}"
            )
    return FireflyAuditContract(envelope, margins, input_format)


def resolve_dataset_root(root: Path, repo_id: str) -> Path:
    """Accept either LeRobot's dataset root or a cache root plus repo_id."""

    root = root.expanduser().resolve()
    direct = root / "meta" / "info.json"
    nested_root = root / Path(repo_id)
    nested = nested_root / "meta" / "info.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return nested_root
    raise DatasetAuditInputError(
        "could not find meta/info.json at either "
        f"{direct} or {nested}"
    )


def load_controller_contract(path: Path) -> FireflyAuditContract:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAuditInputError(f"cannot read controller contract {path}: {exc}") from exc
    return firefly_contract_from_mapping(raw)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAuditInputError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DatasetAuditInputError(f"{label} must contain a JSON object")
    return value


def _feature_summary(info: Mapping[str, Any], findings: _Findings) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        findings.error("features_missing", "meta/info.json has no features object")
        features = {}

    result: dict[str, Any] = {}
    for key in ("action", "observation.state"):
        raw = features.get(key)
        actual = raw if isinstance(raw, Mapping) else {}
        names = actual.get("names")
        shape = actual.get("shape")
        dtype = actual.get("dtype")
        passed = (
            names == list(EXPECTED_JOINT_NAMES)
            and shape == [len(EXPECTED_JOINT_NAMES)]
            and dtype == "float32"
        )
        result[key] = {
            "passed": passed,
            "expected_names": list(EXPECTED_JOINT_NAMES),
            "actual_names": names,
            "expected_shape": [len(EXPECTED_JOINT_NAMES)],
            "actual_shape": shape,
            "expected_dtype": "float32",
            "actual_dtype": dtype,
        }
        if not passed:
            findings.error(
                "joint_feature_schema_mismatch",
                f"{key} must be float32[7] in the Firefly joint order",
                feature=key,
            )

    image_result: dict[str, Any] = {}
    for key, expected_shape in EXPECTED_IMAGE_FEATURES.items():
        raw = features.get(key)
        actual = raw if isinstance(raw, Mapping) else {}
        image_info = actual.get("info") if isinstance(actual.get("info"), Mapping) else {}
        actual_video_fps = image_info.get("video.fps")
        try:
            video_fps_ok = actual_video_fps is not None and math.isclose(
                float(actual_video_fps), EXPECTED_FPS, rel_tol=0.0, abs_tol=1e-9
            )
        except (TypeError, ValueError, OverflowError):
            video_fps_ok = False
        passed = (
            actual.get("dtype") == "video"
            and actual.get("shape") == list(expected_shape)
            and actual.get("names") == ["height", "width", "channels"]
            and video_fps_ok
        )
        image_result[key] = {
            "passed": passed,
            "expected_shape": list(expected_shape),
            "actual_shape": actual.get("shape"),
            "dtype": actual.get("dtype"),
            "video_fps": actual_video_fps,
        }
        if not passed:
            findings.error(
                "image_feature_schema_mismatch",
                f"{key} must be 480x640 RGB video at 30 FPS",
                feature=key,
            )
    result["images"] = image_result

    fps = info.get("fps")
    try:
        fps_ok = math.isclose(float(fps), EXPECTED_FPS, rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError, OverflowError):
        fps_ok = False
    if not fps_ok:
        findings.error("dataset_fps_mismatch", "dataset metadata FPS must be exactly 30")
    result["fps"] = {"passed": fps_ok, "expected": EXPECTED_FPS, "actual": fps}

    scalar_types = {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }
    scalar_result: dict[str, Any] = {}
    for key, expected_dtype in scalar_types.items():
        raw = features.get(key)
        actual = raw if isinstance(raw, Mapping) else {}
        passed = actual.get("dtype") == expected_dtype and actual.get("shape") == [1]
        scalar_result[key] = {
            "passed": passed,
            "expected_dtype": expected_dtype,
            "actual_dtype": actual.get("dtype"),
            "expected_shape": [1],
            "actual_shape": actual.get("shape"),
        }
        if not passed:
            findings.error(
                "scalar_feature_schema_mismatch",
                f"{key} metadata has the wrong dtype or shape",
                feature=key,
            )
    result["scalars"] = scalar_result

    robot_type = info.get("robot_type")
    robot_ok = robot_type == "hex_zenoh_observer"
    if not robot_ok:
        findings.error(
            "robot_type_mismatch",
            "dataset robot_type must be hex_zenoh_observer",
            actual=robot_type,
        )
    result["robot_type"] = {
        "passed": robot_ok,
        "expected": "hex_zenoh_observer",
        "actual": robot_type,
    }
    return result


def _read_tasks(dataset_root: Path, findings: _Findings) -> dict[int, str]:
    parquet_path = dataset_root / "meta" / "tasks.parquet"
    jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    rows: list[Mapping[str, Any]] = []
    if parquet_path.is_file():
        try:
            import pyarrow.parquet as pq

            rows = pq.read_table(parquet_path).to_pylist()
        except Exception as exc:
            findings.error("tasks_unreadable", f"cannot read {parquet_path}: {exc}")
            return {}
    elif jsonl_path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            findings.error("tasks_unreadable", f"cannot read {jsonl_path}: {exc}")
            return {}
    else:
        findings.error("tasks_missing", "meta/tasks.parquet or meta/tasks.jsonl is required")
        return {}

    tasks: dict[int, str] = {}
    text_to_index: dict[str, int] = {}
    for row_number, row in enumerate(rows):
        try:
            index = int(row["task_index"])
            text = row["task"]
        except (KeyError, TypeError, ValueError, OverflowError):
            findings.error(
                "task_row_invalid", "task row needs integer task_index and string task"
            )
            continue
        if not isinstance(text, str) or not text.strip():
            findings.error(
                "task_text_empty", "SmolVLA task text must be a non-empty string", row=row_number
            )
            continue
        if text != text.strip():
            findings.warning(
                "task_text_whitespace",
                "task text has leading or trailing whitespace",
                task_index=index,
            )
        if index in tasks:
            findings.error("task_index_duplicate", "task_index occurs more than once", index=index)
            continue
        if text in text_to_index:
            findings.error(
                "task_text_duplicate",
                "the same task text is assigned to multiple task indexes",
                task=text,
            )
            continue
        tasks[index] = text
        text_to_index[text] = index
    return tasks


def _read_episode_metadata(dataset_root: Path, findings: _Findings) -> dict[int, dict[str, Any]]:
    files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        findings.error("episode_metadata_missing", "meta/episodes parquet files are required")
        return {}
    episodes: dict[int, dict[str, Any]] = {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetAuditInputError("pyarrow is required to audit LeRobot parquet") from exc
    for path in files:
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            findings.error("episode_metadata_unreadable", f"cannot read {path}: {exc}")
            continue
        for row in rows:
            try:
                index = int(row["episode_index"])
            except (KeyError, TypeError, ValueError, OverflowError):
                findings.error("episode_metadata_invalid", "episode_index is missing or invalid")
                continue
            if index in episodes:
                findings.error(
                    "episode_metadata_duplicate", "episode metadata index is duplicated", index=index
                )
                continue
            episodes[index] = dict(row)
    return episodes


def _coerce_vector(raw: Any) -> tuple[tuple[float, ...] | None, str | None]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None, "not_sequence"
    if len(raw) != len(EXPECTED_JOINT_NAMES):
        return None, "wrong_shape"
    try:
        return tuple(float(value) for value in raw), None
    except (TypeError, ValueError, OverflowError):
        return None, "not_numeric"


def _read_frames(dataset_root: Path, findings: _Findings) -> tuple[list[_Frame], dict[str, Any]]:
    files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not files:
        findings.error("data_files_missing", "dataset contains no data parquet files")
        return [], {"files": [], "rows": 0, "invalid_rows": 0, "nonfinite_values": 0}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetAuditInputError("pyarrow is required to audit LeRobot parquet") from exc

    frames: list[_Frame] = []
    rows_seen = 0
    invalid_rows = 0
    nonfinite_values = 0
    bad_shape_rows = 0
    bad_scalar_rows = 0
    column_schema_mismatches: list[dict[str, str]] = []
    expected_types = {
        "action": pa.list_(pa.float32(), len(EXPECTED_JOINT_NAMES)),
        "observation.state": pa.list_(pa.float32(), len(EXPECTED_JOINT_NAMES)),
        "timestamp": pa.float32(),
        "frame_index": pa.int64(),
        "episode_index": pa.int64(),
        "index": pa.int64(),
        "task_index": pa.int64(),
    }
    for path in files:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            findings.error("data_file_unreadable", f"cannot read {path}: {exc}")
            continue
        missing = [name for name in REQUIRED_DATA_COLUMNS if name not in table.column_names]
        if missing:
            findings.error(
                "data_columns_missing",
                "data parquet is missing required columns",
                file=str(path),
                columns=missing,
            )
            rows_seen += table.num_rows
            invalid_rows += table.num_rows
            continue
        for column_name, expected_type in expected_types.items():
            actual_type = table.schema.field(column_name).type
            if actual_type != expected_type:
                column_schema_mismatches.append(
                    {
                        "file": str(path.relative_to(dataset_root)),
                        "column": column_name,
                        "expected": str(expected_type),
                        "actual": str(actual_type),
                    }
                )
        for row in table.select(REQUIRED_DATA_COLUMNS).to_pylist():
            rows_seen += 1
            action, action_error = _coerce_vector(row["action"])
            state, state_error = _coerce_vector(row["observation.state"])
            if action_error or state_error:
                bad_shape_rows += 1
                invalid_rows += 1
                continue
            assert action is not None and state is not None
            try:
                timestamp = float(row["timestamp"])
                frame_index = int(row["frame_index"])
                episode_index = int(row["episode_index"])
                global_index = int(row["index"])
                task_index = int(row["task_index"])
            except (TypeError, ValueError, OverflowError):
                bad_scalar_rows += 1
                invalid_rows += 1
                continue
            values = action + state + (timestamp,)
            nonfinite_values += sum(not math.isfinite(value) for value in values)
            frames.append(
                _Frame(
                    action,
                    state,
                    timestamp,
                    frame_index,
                    episode_index,
                    global_index,
                    task_index,
                )
            )
    if bad_shape_rows:
        findings.error(
            "vector_shape_or_type_invalid",
            "action/state rows must each contain exactly seven numeric values",
            rows=bad_shape_rows,
        )
    if bad_scalar_rows:
        findings.error(
            "scalar_columns_invalid",
            "timestamp and index columns must be numeric",
            rows=bad_scalar_rows,
        )
    if nonfinite_values:
        findings.error(
            "nonfinite_data",
            "action/state/timestamp contain NaN or infinite values",
            values=nonfinite_values,
        )
    if column_schema_mismatches:
        findings.error(
            "parquet_column_schema_mismatch",
            "parquet column dtypes/shapes do not match the Firefly v3 schema",
            mismatches=column_schema_mismatches[:MAX_EXAMPLES],
            total=len(column_schema_mismatches),
        )
    return frames, {
        "files": [str(path.relative_to(dataset_root)) for path in files],
        "rows": rows_seen,
        "valid_shape_rows": len(frames),
        "invalid_rows": invalid_rows,
        "nonfinite_values": nonfinite_values,
        "column_schema_mismatches": len(column_schema_mismatches),
    }


def _metadata_consistency(
    info: Mapping[str, Any],
    frames: list[_Frame],
    rows_seen: int,
    tasks: Mapping[int, str],
    episode_metadata: Mapping[int, Mapping[str, Any]],
    findings: _Findings,
) -> dict[str, Any]:
    actual_episodes = sorted({frame.episode_index for frame in frames})
    comparisons = {
        "total_frames": (info.get("total_frames"), rows_seen),
        "total_episodes": (info.get("total_episodes"), len(actual_episodes)),
        "total_tasks": (info.get("total_tasks"), len(tasks)),
    }
    passed = True
    for label, (metadata_value, actual) in comparisons.items():
        if metadata_value != actual:
            passed = False
            findings.error(
                "metadata_count_mismatch",
                f"{label} does not match parquet content",
                field=label,
                metadata=metadata_value,
                actual=actual,
            )
    if rows_seen == 0:
        passed = False
        findings.error("dataset_empty", "dataset has no frames")
    episode_keys = sorted(episode_metadata)
    if episode_keys != actual_episodes:
        passed = False
        findings.error(
            "episode_metadata_mismatch",
            "meta/episodes indexes do not match data episode indexes",
            metadata_indexes=episode_keys,
            data_indexes=actual_episodes,
        )
    return {
        "passed": passed,
        **{
            label: {"metadata": metadata, "actual": actual}
            for label, (metadata, actual) in comparisons.items()
        },
    }


def _sequence_checks(
    frames: list[_Frame],
    episode_metadata: Mapping[int, Mapping[str, Any]],
    findings: _Findings,
) -> dict[str, Any]:
    by_episode: dict[int, list[_Frame]] = {}
    for frame in frames:
        by_episode.setdefault(frame.episode_index, []).append(frame)
    frame_index_violations = 0
    timestamp_violations = 0
    metadata_length_violations = 0
    duplicate_global_indexes = len(frames) - len({frame.index for frame in frames})
    max_timestamp_error = 0.0
    timestamp_tolerance_s = 1e-4
    for episode_index, episode_frames in by_episode.items():
        ordered = sorted(episode_frames, key=lambda frame: frame.frame_index)
        actual_indexes = [frame.frame_index for frame in ordered]
        expected_indexes = list(range(len(ordered)))
        frame_index_violations += sum(
            actual != expected
            for actual, expected in zip(actual_indexes, expected_indexes, strict=True)
        )
        metadata = episode_metadata.get(episode_index)
        if metadata is not None and metadata.get("length") != len(ordered):
            metadata_length_violations += 1
        for frame in ordered:
            if not math.isfinite(frame.timestamp):
                continue
            expected_timestamp = frame.frame_index / EXPECTED_FPS
            error = abs(frame.timestamp - expected_timestamp)
            max_timestamp_error = max(max_timestamp_error, error)
            timestamp_violations += error > timestamp_tolerance_s
    if frame_index_violations:
        findings.error(
            "frame_index_not_contiguous",
            "frame_index must start at zero and be contiguous inside every episode",
            rows=frame_index_violations,
        )
    if timestamp_violations:
        findings.error(
            "timestamp_not_30hz",
            "timestamps must equal frame_index / 30 within 0.1 ms",
            rows=timestamp_violations,
        )
    if metadata_length_violations:
        findings.error(
            "episode_length_mismatch",
            "episode metadata length does not match data rows",
            episodes=metadata_length_violations,
        )
    if duplicate_global_indexes:
        findings.error(
            "global_index_duplicate",
            "global data index contains duplicates",
            rows=duplicate_global_indexes,
        )
    return {
        "passed": not any(
            (
                frame_index_violations,
                timestamp_violations,
                metadata_length_violations,
                duplicate_global_indexes,
            )
        ),
        "episodes": len(by_episode),
        "frame_index_violation_rows": frame_index_violations,
        "timestamp_30hz_violation_rows": timestamp_violations,
        "timestamp_tolerance_s": timestamp_tolerance_s,
        "max_timestamp_error_s": max_timestamp_error,
        "episode_length_mismatches": metadata_length_violations,
        "duplicate_global_indexes": duplicate_global_indexes,
    }


def _task_checks(
    frames: list[_Frame],
    tasks: Mapping[int, str],
    episode_metadata: Mapping[int, Mapping[str, Any]],
    expected_task: str | None,
    findings: _Findings,
) -> dict[str, Any]:
    used_indexes = sorted({frame.task_index for frame in frames})
    missing_indexes = [index for index in used_indexes if index not in tasks]
    if missing_indexes:
        findings.error(
            "task_index_unmapped",
            "data rows reference task indexes absent from task metadata",
            indexes=missing_indexes,
        )
    episode_task_mismatches = 0
    by_episode: dict[int, set[int]] = {}
    for frame in frames:
        by_episode.setdefault(frame.episode_index, set()).add(frame.task_index)
    for episode_index, indexes in by_episode.items():
        metadata_tasks = episode_metadata.get(episode_index, {}).get("tasks")
        if metadata_tasks is None:
            episode_task_mismatches += 1
            continue
        expected_texts = {tasks[index] for index in indexes if index in tasks}
        if set(metadata_tasks) != expected_texts:
            episode_task_mismatches += 1
    if episode_task_mismatches:
        findings.error(
            "episode_task_mismatch",
            "episode task text does not match frame task indexes",
            episodes=episode_task_mismatches,
        )
    expected_task_mismatches = 0
    if expected_task is not None:
        used_texts = {tasks[index] for index in used_indexes if index in tasks}
        expected_task_mismatches = sum(text != expected_task for text in used_texts)
        if not expected_task.strip():
            findings.error("expected_task_empty", "--expected-task must not be empty")
        elif expected_task_mismatches or used_texts != {expected_task}:
            findings.error(
                "expected_task_mismatch",
                "dataset task text does not exactly match --expected-task",
                expected=expected_task,
                actual=sorted(used_texts),
            )
    return {
        "passed": not missing_indexes
        and not episode_task_mismatches
        and not expected_task_mismatches
        and bool(tasks),
        "task_texts": [
            {"task_index": index, "task": text} for index, text in sorted(tasks.items())
        ],
        "used_task_indexes": used_indexes,
        "unmapped_task_indexes": missing_indexes,
        "episode_task_mismatches": episode_task_mismatches,
        "expected_task": expected_task,
    }


def _contract_summary(contract: FireflyAuditContract | None) -> dict[str, Any]:
    if contract is None:
        return {
            "provided": False,
            "passed": False,
            "source": None,
            "note": "No static fallback is used; provide a saved live controller contract.",
        }
    envelope = contract.envelope
    return {
        "provided": True,
        "passed": envelope.joint_names == EXPECTED_JOINT_NAMES,
        "input_format": contract.input_format,
        "source": envelope.source,
        "joint_names": list(envelope.joint_names),
        "raw_position_min": list(envelope.pos_min),
        "raw_position_max": list(envelope.pos_max),
        "position_margins": list(contract.position_margins),
        "margin_position_min": list(contract.margin_min),
        "margin_position_max": list(contract.margin_max),
        "effective_step_limits": list(envelope.step_limits),
    }


def _position_checks(
    frames: list[_Frame], contract: FireflyAuditContract | None, findings: _Findings
) -> dict[str, Any]:
    if contract is None:
        return {"passed": False, "skipped": True, "reason": "controller contract missing"}
    envelope = contract.envelope
    if envelope.joint_names != EXPECTED_JOINT_NAMES:
        findings.error(
            "contract_joint_order_mismatch",
            "controller contract joint order does not match Firefly dataset order",
            actual=list(envelope.joint_names),
            expected=list(EXPECTED_JOINT_NAMES),
        )
        return {"passed": False, "skipped": True, "reason": "contract joint mismatch"}

    fields = (
        "raw_below_values",
        "raw_above_values",
        "margin_below_values",
        "margin_above_values",
    )
    per_feature: dict[str, dict[str, dict[str, Any]]] = {
        feature: {
            name: {"min": None, "max": None, **dict.fromkeys(fields, 0)}
            for name in EXPECTED_JOINT_NAMES
        }
        for feature in ("action", "observation.state")
    }
    aggregate = {
        "action_raw_violation_values": 0,
        "action_margin_violation_values": 0,
        "state_raw_violation_values": 0,
        "state_margin_violation_values_diagnostic": 0,
    }
    action_raw_frames: set[tuple[int, int]] = set()
    action_margin_frames: set[tuple[int, int]] = set()
    state_raw_frames: set[tuple[int, int]] = set()
    examples: list[dict[str, Any]] = []
    for frame in frames:
        for feature, values in (("action", frame.action), ("observation.state", frame.state)):
            for joint_index, (name, value, raw_min, raw_max, margin_min, margin_max) in enumerate(
                zip(
                    EXPECTED_JOINT_NAMES,
                    values,
                    envelope.pos_min,
                    envelope.pos_max,
                    contract.margin_min,
                    contract.margin_max,
                    strict=True,
                )
            ):
                if not math.isfinite(value):
                    continue
                stats = per_feature[feature][name]
                stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                stats["max"] = value if stats["max"] is None else max(stats["max"], value)
                raw_below = value < raw_min
                raw_above = value > raw_max
                margin_below = value < margin_min
                margin_above = value > margin_max
                stats["raw_below_values"] += raw_below
                stats["raw_above_values"] += raw_above
                stats["margin_below_values"] += margin_below
                stats["margin_above_values"] += margin_above
                raw_bad = raw_below or raw_above
                margin_bad = margin_below or margin_above
                key = (frame.episode_index, frame.frame_index)
                if feature == "action":
                    aggregate["action_raw_violation_values"] += raw_bad
                    aggregate["action_margin_violation_values"] += margin_bad
                    if raw_bad:
                        action_raw_frames.add(key)
                    if margin_bad:
                        action_margin_frames.add(key)
                else:
                    aggregate["state_raw_violation_values"] += raw_bad
                    aggregate["state_margin_violation_values_diagnostic"] += margin_bad
                    if raw_bad:
                        state_raw_frames.add(key)
                if (raw_bad or (feature == "action" and margin_bad)) and len(examples) < MAX_EXAMPLES:
                    examples.append(
                        {
                            "episode_index": frame.episode_index,
                            "frame_index": frame.frame_index,
                            "feature": feature,
                            "joint_index": joint_index,
                            "joint_name": name,
                            "value": value,
                            "raw_range": [raw_min, raw_max],
                            "margin_range": [margin_min, margin_max],
                        }
                    )
    aggregate.update(
        {
            "action_raw_violation_frames": len(action_raw_frames),
            "action_margin_violation_frames": len(action_margin_frames),
            "state_raw_violation_frames": len(state_raw_frames),
        }
    )
    hard_violations = (
        aggregate["action_margin_violation_values"]
        + aggregate["state_raw_violation_values"]
    )
    if aggregate["action_margin_violation_values"]:
        findings.error(
            "action_position_out_of_margin_bounds",
            "recorded action is outside the current inward-margin controller range",
            values=aggregate["action_margin_violation_values"],
            frames=aggregate["action_margin_violation_frames"],
        )
    if aggregate["state_raw_violation_values"]:
        findings.error(
            "state_position_out_of_raw_bounds",
            "measured state is outside the raw controller range",
            values=aggregate["state_raw_violation_values"],
            frames=aggregate["state_raw_violation_frames"],
        )
    return {
        "passed": hard_violations == 0,
        "skipped": False,
        **aggregate,
        "note": "state margin counts are diagnostic; measured state is required only inside raw bounds",
        "per_feature_per_joint": per_feature,
        "examples": examples,
    }


def _step_checks(
    frames: list[_Frame], contract: FireflyAuditContract | None, findings: _Findings
) -> dict[str, Any]:
    if contract is None:
        return {"passed": False, "skipped": True, "reason": "controller contract missing"}
    if contract.envelope.joint_names != EXPECTED_JOINT_NAMES:
        return {"passed": False, "skipped": True, "reason": "contract joint mismatch"}
    limits = contract.envelope.step_limits
    per_joint = {
        name: {
            "limit_rad_per_30hz_step": limit,
            "consecutive_action_violations": 0,
            "first_action_vs_state_diagnostic_exceedances": 0,
            "first_action_vs_state_violations": 0,
            "max_first_action_vs_state_tracking_error": 0.0,
            "max_consecutive_action_delta": 0.0,
        }
        for name, limit in zip(EXPECTED_JOINT_NAMES, limits, strict=True)
    }
    by_episode: dict[int, list[_Frame]] = {}
    for frame in frames:
        by_episode.setdefault(frame.episode_index, []).append(frame)
    consecutive_values = 0
    consecutive_frames: set[tuple[int, int]] = set()
    first_values = 0
    first_episodes: set[int] = set()
    first_comparisons = 0
    first_total_abs = 0.0
    first_max_abs = 0.0
    comparisons = 0
    examples: list[dict[str, Any]] = []
    for episode_index, episode_frames in by_episode.items():
        ordered = sorted(episode_frames, key=lambda frame: frame.frame_index)
        if not ordered:
            continue
        first = ordered[0]
        for joint_index, (name, action, state, limit) in enumerate(
            zip(EXPECTED_JOINT_NAMES, first.action, first.state, limits, strict=True)
        ):
            if not math.isfinite(action) or not math.isfinite(state):
                continue
            delta = abs(action - state)
            first_comparisons += 1
            first_total_abs += delta
            first_max_abs = max(first_max_abs, delta)
            per_joint[name]["max_first_action_vs_state_tracking_error"] = max(
                per_joint[name]["max_first_action_vs_state_tracking_error"], delta
            )
            if delta > limit + 1e-9:
                first_values += 1
                first_episodes.add(episode_index)
                per_joint[name]["first_action_vs_state_diagnostic_exceedances"] += 1
                # Compatibility alias retained for schema-v1 report consumers. This
                # count is diagnostic and is not a hard step-limit violation.
                per_joint[name]["first_action_vs_state_violations"] += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append(
                        {
                            "kind": "first_action_vs_state",
                            "episode_index": episode_index,
                            "frame_index": first.frame_index,
                            "joint_index": joint_index,
                            "joint_name": name,
                            "delta": delta,
                            "limit": limit,
                        }
                    )
        for previous, current in zip(ordered, ordered[1:]):
            if current.frame_index != previous.frame_index + 1:
                continue
            comparisons += 1
            bad_frame = False
            for joint_index, (name, before, after, limit) in enumerate(
                zip(
                    EXPECTED_JOINT_NAMES,
                    previous.action,
                    current.action,
                    limits,
                    strict=True,
                )
            ):
                if not math.isfinite(before) or not math.isfinite(after):
                    continue
                delta = abs(after - before)
                stats = per_joint[name]
                stats["max_consecutive_action_delta"] = max(
                    stats["max_consecutive_action_delta"], delta
                )
                if delta > limit + 1e-9:
                    bad_frame = True
                    consecutive_values += 1
                    stats["consecutive_action_violations"] += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append(
                            {
                                "kind": "consecutive_action",
                                "episode_index": episode_index,
                                "frame_index": current.frame_index,
                                "joint_index": joint_index,
                                "joint_name": name,
                                "delta": delta,
                                "limit": limit,
                            }
                        )
            if bad_frame:
                consecutive_frames.add((episode_index, current.frame_index))
    if consecutive_values:
        findings.error(
            "action_step_limit_exceeded",
            "consecutive episode actions exceed the effective 30 Hz step contract",
            values=consecutive_values,
            frames=len(consecutive_frames),
        )
    if first_values:
        findings.warning(
            "first_action_state_tracking_error_diagnostic",
            "episode-first action/state tracking error exceeds the 30 Hz action-step "
            "reference; this diagnostic does not fail the dataset",
            values=first_values,
            episodes=len(first_episodes),
        )
    return {
        "passed": not consecutive_values,
        "skipped": False,
        "consecutive_comparisons": comparisons,
        "consecutive_action_violation_values": consecutive_values,
        "consecutive_action_violation_frames": len(consecutive_frames),
        "first_action_vs_state_tracking_error": {
            "diagnostic_only": True,
            "comparison_values": first_comparisons,
            "mean_abs": (
                first_total_abs / first_comparisons if first_comparisons else None
            ),
            "max_abs": first_max_abs if first_comparisons else None,
            "step_reference_exceedance_values": first_values,
            "step_reference_exceedance_episodes": len(first_episodes),
        },
        # Compatibility aliases: schema-v1 reports exposed these names before the
        # first-frame tracking/step semantics were separated.
        "first_action_vs_state_violation_values": first_values,
        "first_action_vs_state_violation_episodes": len(first_episodes),
        "note": (
            "only consecutive actions within an episode are hard step-limit checks; "
            "episode-first action/state differences are tracking diagnostics"
        ),
        "per_joint": per_joint,
        "examples": examples,
    }


def _error_metric(count: int, total_abs: float, total_signed: float, total_squared: float, maximum: float) -> dict[str, Any]:
    if count == 0:
        return {
            "count": 0,
            "mean_abs": None,
            "mean_signed": None,
            "rmse": None,
            "max_abs": None,
        }
    return {
        "count": count,
        "mean_abs": total_abs / count,
        "mean_signed": total_signed / count,
        "rmse": math.sqrt(total_squared / count),
        "max_abs": maximum,
    }


def _action_state_checks(frames: list[_Frame], findings: _Findings) -> dict[str, Any]:
    accumulators = [
        {"count": 0, "abs": 0.0, "signed": 0.0, "squared": 0.0, "max": 0.0}
        for _ in EXPECTED_JOINT_NAMES
    ]
    exact_values = 0
    compared_values = 0
    exact_frames = 0
    comparable_frames = 0
    for frame in frames:
        frame_comparable = True
        frame_exact = True
        for index, (action, state) in enumerate(zip(frame.action, frame.state, strict=True)):
            if not math.isfinite(action) or not math.isfinite(state):
                frame_comparable = False
                frame_exact = False
                continue
            difference = action - state
            absolute = abs(difference)
            stats = accumulators[index]
            stats["count"] += 1
            stats["abs"] += absolute
            stats["signed"] += difference
            stats["squared"] += difference * difference
            stats["max"] = max(stats["max"], absolute)
            compared_values += 1
            equal = action == state
            exact_values += equal
            frame_exact = frame_exact and equal
        if frame_comparable:
            comparable_frames += 1
            exact_frames += frame_exact
    all_frames_identical = comparable_frames > 0 and exact_frames == comparable_frames
    if all_frames_identical:
        findings.error(
            "action_state_identical_dataset",
            "action equals observation.state exactly in every comparable frame; "
            "this does not demonstrate recorded requested targets",
            frames=comparable_frames,
        )
    per_joint = {}
    for name, stats in zip(EXPECTED_JOINT_NAMES, accumulators, strict=True):
        per_joint[name] = _error_metric(
            stats["count"], stats["abs"], stats["signed"], stats["squared"], stats["max"]
        )
    totals = {
        key: sum(stats[key] for stats in accumulators)
        for key in ("count", "abs", "signed", "squared")
    }
    overall = _error_metric(
        totals["count"], totals["abs"], totals["signed"], totals["squared"],
        max((stats["max"] for stats in accumulators), default=0.0),
    )
    return {
        "passed": not all_frames_identical and comparable_frames > 0,
        "comparable_frames": comparable_frames,
        "exactly_equal_frames": exact_frames,
        "exactly_equal_frame_fraction": (
            exact_frames / comparable_frames if comparable_frames else None
        ),
        "all_frames_exactly_equal": all_frames_identical,
        "compared_values": compared_values,
        "exactly_equal_values": exact_values,
        "exactly_equal_value_fraction": (
            exact_values / compared_values if compared_values else None
        ),
        "error_action_minus_state": {"overall": overall, "per_joint": per_joint},
    }


def _media_checks(dataset_root: Path, info: Mapping[str, Any], findings: _Findings) -> dict[str, Any]:
    total_frames = info.get("total_frames", 0)
    result: dict[str, Any] = {}
    passed = True
    for key in EXPECTED_IMAGE_FEATURES:
        files = sorted((dataset_root / "videos" / key).glob("**/*.mp4"))
        key_passed = bool(files) if total_frames else True
        if not key_passed:
            passed = False
            findings.error(
                "video_files_missing",
                "non-empty dataset has no MP4 files for required image feature",
                feature=key,
            )
        result[key] = {
            "passed": key_passed,
            "file_count": len(files),
            "files": [str(path.relative_to(dataset_root)) for path in files],
        }
    return {"passed": passed, "features": result, "note": "MP4 presence only; video is not decoded"}


def audit_firefly_dataset(
    dataset_root: Path,
    repo_id: str,
    *,
    controller_contract: FireflyAuditContract | None,
    expected_task: str | None = None,
) -> dict[str, Any]:
    """Audit one resolved dataset root and return a strict-JSON-compatible report."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetAuditInputError(f"dataset metadata does not exist: {info_path}")
    info = _read_json(info_path, "dataset metadata")
    findings = _Findings()
    if controller_contract is None:
        findings.error(
            "controller_contract_missing",
            "a saved live controller contract is required for a training-safe verdict",
        )

    feature_check = _feature_summary(info, findings)
    tasks = _read_tasks(dataset_root, findings)
    episode_metadata = _read_episode_metadata(dataset_root, findings)
    frames, data_summary = _read_frames(dataset_root, findings)
    metadata_check = _metadata_consistency(
        info,
        frames,
        data_summary["rows"],
        tasks,
        episode_metadata,
        findings,
    )
    sequence_check = _sequence_checks(frames, episode_metadata, findings)
    task_check = _task_checks(frames, tasks, episode_metadata, expected_task, findings)
    contract_check = _contract_summary(controller_contract)
    position_check = _position_checks(frames, controller_contract, findings)
    step_check = _step_checks(frames, controller_contract, findings)
    action_state_check = _action_state_checks(frames, findings)
    media_check = _media_checks(dataset_root, info, findings)

    if controller_contract is not None and not contract_check["passed"]:
        findings.error(
            "contract_joint_order_mismatch",
            "controller contract joint order does not match Firefly dataset order",
        )

    verdict = "pass" if not findings.errors else "fail"
    return {
        "schema_version": 1,
        "audit": "firefly_lerobot_dataset",
        "read_only": True,
        "dataset": {
            "root": str(dataset_root),
            "repo_id": repo_id,
            "codebase_version": info.get("codebase_version"),
            "data": data_summary,
        },
        "checks": {
            "features": feature_check,
            "metadata_consistency": metadata_check,
            "episode_30hz_sequence": sequence_check,
            "task_text": task_check,
            "controller_contract": contract_check,
            "position_bounds": position_check,
            "action_step": step_check,
            "action_state_relationship": action_state_check,
            "media_presence": media_check,
        },
        "summary": {
            "verdict": verdict,
            "safe_for_training": verdict == "pass",
            "error_count": len(findings.errors),
            "warning_count": len(findings.warnings),
            "errors": findings.errors,
            "warnings": findings.warnings,
        },
    }


def strict_json_dumps(report: Mapping[str, Any]) -> str:
    """Render reports without JavaScript-only NaN/Infinity literals."""

    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
