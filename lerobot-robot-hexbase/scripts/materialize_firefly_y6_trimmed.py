#!/usr/bin/env python3
"""Materialize the fixed, action-onset-trimmed Firefly v3 dataset offline.

The source dataset is never modified.  ``--dry-run`` performs the complete
read-only source audit and trim planning pass.  A real run writes through the
LeRobot v3 writer into a temporary sibling directory, validates the result,
and only then renames it to the fixed destination.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lerobot_robot_hexbase.firefly_dataset_audit import (  # noqa: E402
    DatasetAuditInputError,
    FireflyAuditContract,
    audit_firefly_dataset,
    load_controller_contract,
)


SOURCE_REPO_ID = "local/firefly_y6_manual_validated_v2"
SOURCE_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v2"
)
DESTINATION_REPO_ID = "local/firefly_y6_manual_validated_v3_trimmed"
DESTINATION_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/firefly_y6_manual_validated_v3_trimmed"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
EXPECTED_TASK = "Pick up the red block and place it in the blue box."
EXPECTED_EPISODES = 20
EXPECTED_FPS = 30
ARM_DIMENSIONS = 6
ONSET_THRESHOLD_RAD = 0.005
PRE_ROLL_FRAMES = 2
VALIDATION_WINDOW_FRAMES = 10
MANIFEST_FILENAME = "firefly_trim_manifest.json"
MANIFEST_KIND = "firefly_action_onset_trim"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DECODE_BATCH_FRAMES = 60
OFFLINE_IMAGE_WRITER_THREADS = 4

_REQUIRED_FRAME_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
_VIDEO_KEYS = (
    "observation.images.scene",
    "observation.images.wrist",
)
_HASH_CHUNK_SIZE = 1024 * 1024


class TrimMaterializationError(RuntimeError):
    """The source or proposed output violates the pinned trim contract."""


@dataclass(frozen=True)
class TrimEpisodePlan:
    episode_index: int
    source_length: int
    onset_frame: int
    trim_start_frame: int
    trimmed_length: int
    source_initial_action: tuple[float, ...]
    trimmed_first_action: tuple[float, ...]
    first_10_max_abs_arm_delta_from_source_initial_rad: float
    first_10_max_abs_arm_delta_from_trimmed_first_rad: float
    trim_adjusted_for_first_10_motion: bool
    first_10_motion_gate_frame: int
    first_10_has_motion: bool
    first_action_state_abs_error_rad: tuple[float, ...]
    first_action_state_within_contract: bool


@dataclass(frozen=True)
class TrimPlan:
    source_total_frames: int
    destination_total_frames: int
    episodes: tuple[TrimEpisodePlan, ...]


AuditFunction = Callable[..., Mapping[str, Any]]


def _strict_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise TrimMaterializationError(
                f"{label} path component does not exist: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise TrimMaterializationError(
                f"{label} must not contain a symbolic-link component: {current}"
            )
    return absolute


def _validate_fixed_paths(source_root: Path, destination_root: Path) -> tuple[Path, Path]:
    source = _require_no_symlink_components(source_root, label="source dataset")
    if not source.is_dir():
        raise TrimMaterializationError(f"source dataset is not a directory: {source}")

    destination = _absolute_without_resolving(destination_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"trimmed destination already exists and will not be overwritten: {destination}"
        )
    if source == destination:
        raise TrimMaterializationError("source and destination roots must differ")
    if source in destination.parents:
        raise TrimMaterializationError("destination must not be inside the source dataset")
    if destination in source.parents:
        raise TrimMaterializationError("source must not be inside the destination dataset")
    return source, destination


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrimMaterializationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrimMaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def _read_episode_metadata(root: Path) -> dict[int, dict[str, Any]]:
    files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise TrimMaterializationError("source has no episode metadata parquet")
    result: dict[int, dict[str, Any]] = {}
    for path in files:
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            raise TrimMaterializationError(
                f"cannot read source episode metadata {path}: {exc}"
            ) from exc
        for raw in rows:
            try:
                episode_index = int(raw["episode_index"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise TrimMaterializationError(
                    f"invalid episode_index in source metadata {path}"
                ) from exc
            if episode_index in result:
                raise TrimMaterializationError(
                    f"duplicate source episode metadata index: {episode_index}"
                )
            result[episode_index] = dict(raw)
    return result


def _read_frames_by_episode(root: Path) -> dict[int, list[dict[str, Any]]]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise TrimMaterializationError("source has no data parquet")
    frames: dict[int, list[dict[str, Any]]] = {}
    for path in files:
        try:
            table = pq.read_table(path, columns=list(_REQUIRED_FRAME_COLUMNS))
        except Exception as exc:
            raise TrimMaterializationError(
                f"cannot read source data parquet {path}: {exc}"
            ) from exc
        for raw in table.to_pylist():
            try:
                episode_index = int(raw["episode_index"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise TrimMaterializationError(
                    f"invalid episode index in {path}"
                ) from exc
            frames.setdefault(episode_index, []).append(dict(raw))
    for episode_frames in frames.values():
        episode_frames.sort(key=lambda row: int(row["frame_index"]))
    return frames


def _require_safe_audit(report: Mapping[str, Any]) -> None:
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or summary.get("safe_for_training") is not True:
        errors = summary.get("errors", []) if isinstance(summary, Mapping) else []
        codes = [
            str(item.get("code", "unknown"))
            for item in errors
            if isinstance(item, Mapping)
        ]
        raise TrimMaterializationError(
            "source dataset failed the formal audit: "
            + ", ".join(codes or ["unknown audit failure"])
        )


def plan_trimmed_episodes(
    frames_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    episode_metadata: Mapping[int, Mapping[str, Any]],
    *,
    effective_step_limits: Sequence[float],
    expected_episodes: int = EXPECTED_EPISODES,
    threshold_rad: float = ONSET_THRESHOLD_RAD,
    pre_roll_frames: int = PRE_ROLL_FRAMES,
) -> TrimPlan:
    """Compute the fixed arm-action onset plan without writing anything."""

    if (
        not math.isfinite(threshold_rad)
        or threshold_rad <= 0
        or isinstance(pre_roll_frames, bool)
        or pre_roll_frames < 0
    ):
        raise TrimMaterializationError("trim threshold/pre-roll parameters are invalid")
    if sorted(frames_by_episode) != list(range(expected_episodes)):
        raise TrimMaterializationError(
            f"source must contain exactly contiguous episodes 0:{expected_episodes}"
        )
    if sorted(episode_metadata) != list(range(expected_episodes)):
        raise TrimMaterializationError(
            f"episode metadata must contain exactly contiguous episodes 0:{expected_episodes}"
        )

    limits = np.asarray(effective_step_limits, dtype=np.float64)
    if limits.shape != (7,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise TrimMaterializationError(
            "controller effective_step_limits must contain seven positive finite values"
        )

    plans: list[TrimEpisodePlan] = []
    source_total = 0
    destination_total = 0
    for episode_index in range(expected_episodes):
        rows = list(frames_by_episode[episode_index])
        if not rows:
            raise TrimMaterializationError(f"episode {episode_index} is empty")
        actual_indexes = [int(row["frame_index"]) for row in rows]
        if actual_indexes != list(range(len(rows))):
            raise TrimMaterializationError(
                f"episode {episode_index} frame_index is not contiguous from zero"
            )
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise TrimMaterializationError(
                f"episode {episode_index} contains a mismatched episode_index"
            )
        metadata_length = episode_metadata[episode_index].get("length")
        if metadata_length != len(rows):
            raise TrimMaterializationError(
                f"episode {episode_index} metadata length {metadata_length!r} "
                f"does not match {len(rows)} data rows"
            )

        try:
            actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
            states = np.asarray(
                [row["observation.state"] for row in rows], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TrimMaterializationError(
                f"episode {episode_index} action/state cannot be converted"
            ) from exc
        if actions.shape != (len(rows), 7) or states.shape != (len(rows), 7):
            raise TrimMaterializationError(
                f"episode {episode_index} action/state must have shape (frames, 7)"
            )
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise TrimMaterializationError(
                f"episode {episode_index} action/state contains NaN or infinity"
            )

        initial = actions[0]
        arm_delta_from_source_initial = np.max(
            np.abs(actions[:, :ARM_DIMENSIONS] - initial[:ARM_DIMENSIONS]), axis=1
        )
        onset_candidates = np.flatnonzero(arm_delta_from_source_initial > threshold_rad)
        if onset_candidates.size == 0:
            raise TrimMaterializationError(
                f"episode {episode_index} has no arm-action onset above {threshold_rad} rad"
            )
        onset = int(onset_candidates[0])
        trim_start = max(0, onset - pre_roll_frames)

        def first_window_motion(start: int) -> float:
            end = min(len(rows), start + VALIDATION_WINDOW_FRAMES)
            arm_window = actions[start:end, :ARM_DIMENSIONS]
            return float(
                np.max(
                    np.abs(
                        arm_window - actions[start, :ARM_DIMENSIONS]
                    )
                )
            )

        trimmed_first_motion = first_window_motion(trim_start)
        trim_adjusted = False
        motion_gate_frame = onset
        if trimmed_first_motion <= threshold_rad:
            fallback: tuple[int, int, float] | None = None
            for candidate_gate in range(onset, len(rows)):
                candidate_start = max(0, candidate_gate - pre_roll_frames)
                local_gate_delta = float(
                    np.max(
                        np.abs(
                            actions[candidate_gate, :ARM_DIMENSIONS]
                            - actions[candidate_start, :ARM_DIMENSIONS]
                        )
                    )
                )
                candidate_motion = first_window_motion(candidate_start)
                if (
                    local_gate_delta > threshold_rad
                    and candidate_motion > threshold_rad
                ):
                    fallback = (candidate_start, candidate_gate, candidate_motion)
                    break
            if fallback is None:
                raise TrimMaterializationError(
                    f"episode {episode_index} cannot place at most {pre_roll_frames} "
                    f"pre-roll frames before a > {threshold_rad} rad arm motion that "
                    f"appears in the first {VALIDATION_WINDOW_FRAMES} output frames"
                )
            trim_start, motion_gate_frame, trimmed_first_motion = fallback
            trim_adjusted = True

        trimmed_length = len(rows) - trim_start
        window_end = min(len(rows), trim_start + VALIDATION_WINDOW_FRAMES)
        first_window = actions[trim_start:window_end, :ARM_DIMENSIONS]
        source_initial_motion = float(
            np.max(np.abs(first_window - initial[:ARM_DIMENSIONS]))
        )
        first_10_has_motion = trimmed_first_motion > threshold_rad
        if not first_10_has_motion:
            raise TrimMaterializationError(
                f"episode {episode_index} first {VALIDATION_WINDOW_FRAMES} trimmed "
                f"frames do not move > {threshold_rad} rad from the trimmed first action"
            )

        first_tracking_error = np.abs(actions[trim_start] - states[trim_start])
        first_tracking_ok = bool(np.all(first_tracking_error <= limits + 1e-12))
        if not first_tracking_ok:
            bad = np.flatnonzero(first_tracking_error > limits + 1e-12).tolist()
            raise TrimMaterializationError(
                f"episode {episode_index} trimmed first action-state error exceeds "
                f"the controller contract at joints {bad}"
            )

        plans.append(
            TrimEpisodePlan(
                episode_index=episode_index,
                source_length=len(rows),
                onset_frame=onset,
                trim_start_frame=trim_start,
                trimmed_length=trimmed_length,
                source_initial_action=tuple(float(value) for value in initial),
                trimmed_first_action=tuple(
                    float(value) for value in actions[trim_start]
                ),
                first_10_max_abs_arm_delta_from_source_initial_rad=source_initial_motion,
                first_10_max_abs_arm_delta_from_trimmed_first_rad=trimmed_first_motion,
                trim_adjusted_for_first_10_motion=trim_adjusted,
                first_10_motion_gate_frame=motion_gate_frame,
                first_10_has_motion=first_10_has_motion,
                first_action_state_abs_error_rad=tuple(
                    float(value) for value in first_tracking_error
                ),
                first_action_state_within_contract=first_tracking_ok,
            )
        )
        source_total += len(rows)
        destination_total += trimmed_length

    return TrimPlan(source_total, destination_total, tuple(plans))


def prepare_plan(
    source_root: Path = SOURCE_ROOT,
    *,
    source_repo_id: str = SOURCE_REPO_ID,
    controller_contract_path: Path = CONTROLLER_CONTRACT,
    expected_task: str = EXPECTED_TASK,
    audit: AuditFunction = audit_firefly_dataset,
) -> tuple[TrimPlan, dict[str, Any], dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]], FireflyAuditContract]:
    source = _require_no_symlink_components(source_root, label="source dataset")
    info = _read_json_object(source / "meta" / "info.json", label="source info")
    if info.get("fps") != EXPECTED_FPS:
        raise TrimMaterializationError(
            f"source FPS must be exactly {EXPECTED_FPS}, got {info.get('fps')!r}"
        )
    if info.get("total_episodes") != EXPECTED_EPISODES:
        raise TrimMaterializationError(
            f"source must contain exactly {EXPECTED_EPISODES} episodes"
        )
    try:
        contract = load_controller_contract(controller_contract_path)
    except DatasetAuditInputError as exc:
        raise TrimMaterializationError(
            f"controller contract cannot be loaded: {exc}"
        ) from exc
    try:
        report = audit(
            source,
            source_repo_id,
            controller_contract=contract,
            expected_task=expected_task,
        )
    except DatasetAuditInputError as exc:
        raise TrimMaterializationError(f"source audit could not complete: {exc}") from exc
    _require_safe_audit(report)
    episode_metadata = _read_episode_metadata(source)
    frames = _read_frames_by_episode(source)
    plan = plan_trimmed_episodes(
        frames,
        episode_metadata,
        effective_step_limits=contract.envelope.step_limits,
    )
    if info.get("total_frames") != plan.source_total_frames:
        raise TrimMaterializationError(
            "source info total_frames does not match the planned source frame count"
        )
    return plan, info, frames, episode_metadata, contract


def plan_summary(plan: TrimPlan) -> dict[str, Any]:
    return {
        "status": "dry_run_plan",
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "root": str(SOURCE_ROOT),
            "total_frames": plan.source_total_frames,
        },
        "destination": {
            "repo_id": DESTINATION_REPO_ID,
            "root": str(DESTINATION_ROOT),
            "total_frames": plan.destination_total_frames,
        },
        "algorithm": {
            "motion_dimensions": "action[0:6] (arm only)",
            "threshold_rad": ONSET_THRESHOLD_RAD,
            "comparison": "max_abs(action_t - source_action_0) > threshold",
            "first_10_motion_comparison": "max_abs(action_t - trimmed_action_0) > threshold",
            "fallback": "advance to earliest 2-frame local arm delta above threshold",
            "pre_roll_frames": PRE_ROLL_FRAMES,
            "validation_window_frames": VALIDATION_WINDOW_FRAMES,
        },
        "summary": {
            "episode_count": len(plan.episodes),
            "removed_frames": plan.source_total_frames
            - plan.destination_total_frames,
            "all_first10_has_motion": all(
                episode.first_10_has_motion for episode in plan.episodes
            ),
            "all_first_action_state_within_contract": all(
                episode.first_action_state_within_contract
                for episode in plan.episodes
            ),
        },
        "episodes": [
            {
                "episode_index": episode.episode_index,
                "source_length": episode.source_length,
                "onset_frame": episode.onset_frame,
                "trim_start_frame": episode.trim_start_frame,
                "trimmed_length": episode.trimmed_length,
                "first_10_max_abs_arm_delta_from_source_initial_rad": (
                    episode.first_10_max_abs_arm_delta_from_source_initial_rad
                ),
                "first_10_max_abs_arm_delta_from_trimmed_first_rad": (
                    episode.first_10_max_abs_arm_delta_from_trimmed_first_rad
                ),
                "first_10_has_motion": episode.first_10_has_motion,
                "first_action_state_within_contract": (
                    episode.first_action_state_within_contract
                ),
            }
            for episode in plan.episodes
        ],
    }


def _source_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    manifest: list[dict[str, Any]] = []
    identities: dict[str, tuple[int, ...]] = {}
    for tree in ("meta", "data", "videos"):
        tree_root = root / tree
        if not tree_root.is_dir() or tree_root.is_symlink():
            raise TrimMaterializationError(
                f"source {tree} must be a regular, non-symlink directory"
            )
        for path in sorted(tree_root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            item = path.lstat()
            if stat.S_ISLNK(item.st_mode):
                raise TrimMaterializationError(
                    f"source dataset contains a symbolic link: {path}"
                )
            if stat.S_ISDIR(item.st_mode):
                continue
            if not stat.S_ISREG(item.st_mode):
                raise TrimMaterializationError(
                    f"source dataset contains a non-regular entry: {path}"
                )
            identity = (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            identities[relative] = identity
            manifest.append(
                {
                    "path": relative,
                    "size": item.st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return manifest, identities


def _verify_source_identities(
    root: Path,
    expected_manifest: Sequence[Mapping[str, Any]],
    expected_identities: Mapping[str, tuple[int, ...]],
) -> None:
    actual_manifest, actual_identities = _source_inventory(root)
    if (
        actual_manifest != list(expected_manifest)
        or actual_identities != dict(expected_identities)
    ):
        raise TrimMaterializationError("source dataset changed during materialization")


def _require_offline_video_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise TrimMaterializationError(
            "missing offline video tools in PATH: "
            + ", ".join(missing)
            + "; run `source activate-io` first"
        )


def _video_path(
    source_root: Path,
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
    video_key: str,
) -> Path:
    template = info.get("video_path")
    if not isinstance(template, str):
        raise TrimMaterializationError("source info.video_path is missing")
    try:
        relative = template.format(
            video_key=video_key,
            chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode[f"videos/{video_key}/file_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrimMaterializationError(
            f"source episode lacks video mapping for {video_key}"
        ) from exc
    path = source_root / relative
    if path.is_symlink() or not path.is_file():
        raise TrimMaterializationError(
            f"source video must be a regular, non-symlink file: {path}"
        )
    return path


def _source_video_timestamps(
    episode: Mapping[str, Any],
    video_key: str,
    source_frame_indexes: Sequence[int],
) -> list[float]:
    try:
        start = float(episode[f"videos/{video_key}/from_timestamp"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TrimMaterializationError(
            f"source episode has invalid video start for {video_key}"
        ) from exc
    if not math.isfinite(start):
        raise TrimMaterializationError(
            f"source episode has non-finite video start for {video_key}"
        )
    return [start + frame_index / EXPECTED_FPS for frame_index in source_frame_indexes]


def _writer_features(info: Mapping[str, Any]) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise TrimMaterializationError("source info.features is missing")
    if set(_VIDEO_KEYS) - set(features):
        raise TrimMaterializationError("source is missing one or more pinned video features")
    return copy.deepcopy(dict(features))


def _materialize_to_staging(
    *,
    source_root: Path,
    staging_root: Path,
    info: Mapping[str, Any],
    frames_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    episode_metadata: Mapping[int, Mapping[str, Any]],
    plan: TrimPlan,
    decode_batch_frames: int,
) -> None:
    try:
        from lerobot.configs import encoder_config_from_video_info
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.video_utils import decode_video_frames
    except ImportError as exc:
        raise TrimMaterializationError(
            "the active Python environment does not provide LeRobot dataset I/O"
        ) from exc

    if decode_batch_frames <= 0:
        raise TrimMaterializationError("decode_batch_frames must be positive")
    features = _writer_features(info)
    encoders = []
    for video_key in _VIDEO_KEYS:
        feature = features[video_key]
        video_info = feature.get("info") if isinstance(feature, Mapping) else None
        if not isinstance(video_info, Mapping):
            raise TrimMaterializationError(
                f"source video feature {video_key} has no encoder metadata"
            )
        encoders.append(encoder_config_from_video_info(dict(video_info)))
    if encoders[0] != encoders[1]:
        raise TrimMaterializationError(
            "scene and wrist source videos use different encoder contracts"
        )

    writer = LeRobotDataset.create(
        repo_id=DESTINATION_REPO_ID,
        fps=EXPECTED_FPS,
        features=features,
        root=staging_root,
        robot_type=str(info.get("robot_type", "hex_zenoh_observer")),
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=encoders[0],
        streaming_encoding=False,
        image_writer_processes=0,
        image_writer_threads=OFFLINE_IMAGE_WRITER_THREADS,
        metadata_buffer_size=10,
        data_files_size_in_mb=int(info.get("data_files_size_in_mb", 100)),
        video_files_size_in_mb=int(info.get("video_files_size_in_mb", 200)),
    )
    try:
        for episode_plan in plan.episodes:
            episode_index = episode_plan.episode_index
            source_rows = frames_by_episode[episode_index]
            source_episode = episode_metadata[episode_index]
            tasks = source_episode.get("tasks")
            if tasks != [EXPECTED_TASK]:
                raise TrimMaterializationError(
                    f"episode {episode_index} must have exactly the pinned task"
                )

            for batch_start in range(
                episode_plan.trim_start_frame,
                episode_plan.source_length,
                decode_batch_frames,
            ):
                batch_end = min(
                    episode_plan.source_length, batch_start + decode_batch_frames
                )
                source_indexes = list(range(batch_start, batch_end))
                decoded: dict[str, Any] = {}
                for video_key in _VIDEO_KEYS:
                    decoded[video_key] = decode_video_frames(
                        _video_path(source_root, info, source_episode, video_key),
                        _source_video_timestamps(
                            source_episode, video_key, source_indexes
                        ),
                        tolerance_s=1e-4,
                        backend="pyav",
                        return_uint8=True,
                    )
                    if len(decoded[video_key]) != len(source_indexes):
                        raise TrimMaterializationError(
                            f"decoder returned the wrong frame count for episode "
                            f"{episode_index} {video_key}"
                        )

                for batch_offset, source_frame_index in enumerate(source_indexes):
                    source_row = source_rows[source_frame_index]
                    writer.add_frame(
                        {
                            "action": np.asarray(
                                source_row["action"], dtype=np.float32
                            ),
                            "observation.state": np.asarray(
                                source_row["observation.state"], dtype=np.float32
                            ),
                            _VIDEO_KEYS[0]: decoded[_VIDEO_KEYS[0]][
                                batch_offset
                            ].permute(1, 2, 0).contiguous().numpy(),
                            _VIDEO_KEYS[1]: decoded[_VIDEO_KEYS[1]][
                                batch_offset
                            ].permute(1, 2, 0).contiguous().numpy(),
                            "task": EXPECTED_TASK,
                        }
                    )
            writer.save_episode(parallel_encoding=False)
    finally:
        writer.finalize()


def _manifest_document(
    *,
    plan: TrimPlan,
    source_manifest: list[dict[str, Any]],
    controller_contract_path: Path,
    contract: FireflyAuditContract,
) -> dict[str, Any]:
    episodes = []
    for episode in plan.episodes:
        raw = asdict(episode)
        # Source action vectors are useful for post-write verification but are
        # not needed by the training gate and would add ambiguity to the public
        # manifest contract.
        raw.pop("source_initial_action")
        raw.pop("trimmed_first_action")
        episodes.append(raw)
    document = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "root": str(SOURCE_ROOT),
            "episode_count": EXPECTED_EPISODES,
            "total_frames": plan.source_total_frames,
            "files": source_manifest,
            "files_manifest_sha256": _sha256_json(source_manifest),
        },
        "destination": {
            "repo_id": DESTINATION_REPO_ID,
            "root": str(DESTINATION_ROOT),
            "episode_count": EXPECTED_EPISODES,
            "total_frames": plan.destination_total_frames,
        },
        "algorithm": {
            "motion_dimensions": "action[0:6] (arm only)",
            "threshold_rad": ONSET_THRESHOLD_RAD,
            "comparison": "max_abs(action_t - source_action_0) > threshold",
            "first_10_motion_comparison": "max_abs(action_t - trimmed_action_0) > threshold",
            "fallback": "advance to earliest 2-frame local arm delta above threshold",
            "pre_roll_frames": PRE_ROLL_FRAMES,
            "validation_window_frames": VALIDATION_WINDOW_FRAMES,
        },
        "controller_contract": {
            "path": str(controller_contract_path),
            "sha256": _sha256_file(controller_contract_path),
            "effective_step_limits": list(contract.envelope.step_limits),
        },
        "summary": {
            "episode_count": EXPECTED_EPISODES,
            "all_first10_has_motion": all(
                episode.first_10_has_motion for episode in plan.episodes
            ),
            "all_first_action_state_within_contract": all(
                episode.first_action_state_within_contract
                for episode in plan.episodes
            ),
        },
        "episodes": episodes,
    }
    document["document_sha256"] = _sha256_json(document)
    return document


def validate_trim_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_root: Path | None = None,
) -> None:
    """Validate the fields consumed by the fixed v3 trimmed training entry."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TrimMaterializationError("trim manifest schema_version mismatch")
    if manifest.get("kind") != MANIFEST_KIND:
        raise TrimMaterializationError("trim manifest kind mismatch")
    source = manifest.get("source")
    destination = manifest.get("destination")
    algorithm = manifest.get("algorithm")
    summary = manifest.get("summary")
    episodes = manifest.get("episodes")
    if not all(
        isinstance(value, Mapping)
        for value in (source, destination, algorithm, summary)
    ) or not isinstance(episodes, list):
        raise TrimMaterializationError("trim manifest has missing structured fields")
    assert isinstance(source, Mapping)
    assert isinstance(destination, Mapping)
    assert isinstance(algorithm, Mapping)
    assert isinstance(summary, Mapping)
    if source.get("repo_id") != SOURCE_REPO_ID:
        raise TrimMaterializationError("trim manifest source repo_id is not validated_v2")
    if destination.get("repo_id") != DESTINATION_REPO_ID:
        raise TrimMaterializationError("trim manifest destination repo_id mismatch")
    if expected_root is not None and destination.get("root") != str(expected_root):
        raise TrimMaterializationError("trim manifest destination root mismatch")
    if source.get("episode_count") != EXPECTED_EPISODES:
        raise TrimMaterializationError("trim manifest source episode count mismatch")
    if summary.get("episode_count") != EXPECTED_EPISODES or len(episodes) != EXPECTED_EPISODES:
        raise TrimMaterializationError("trim manifest episode count mismatch")
    if algorithm.get("threshold_rad") != ONSET_THRESHOLD_RAD:
        raise TrimMaterializationError("trim manifest threshold mismatch")
    if algorithm.get("pre_roll_frames") != PRE_ROLL_FRAMES:
        raise TrimMaterializationError("trim manifest pre-roll mismatch")
    if summary.get("all_first10_has_motion") is not True:
        raise TrimMaterializationError("trim manifest first-10 motion gate failed")
    if summary.get("all_first_action_state_within_contract") is not True:
        raise TrimMaterializationError("trim manifest first action-state gate failed")
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TrimMaterializationError("trim manifest episode entry is invalid")
        if episode.get("episode_index") != expected_index:
            raise TrimMaterializationError("trim manifest episode indexes are not contiguous")
        if episode.get("first_10_has_motion") is not True:
            raise TrimMaterializationError(
                f"trim manifest episode {expected_index} lacks first-10 motion"
            )
        if episode.get("first_action_state_within_contract") is not True:
            raise TrimMaterializationError(
                f"trim manifest episode {expected_index} first action-state gate failed"
            )
    claimed_digest = manifest.get("document_sha256")
    unsigned = dict(manifest)
    unsigned.pop("document_sha256", None)
    if claimed_digest != _sha256_json(unsigned):
        raise TrimMaterializationError("trim manifest document digest mismatch")


def _validate_materialized_output(
    *,
    staging_root: Path,
    plan: TrimPlan,
    contract: FireflyAuditContract,
    audit: AuditFunction,
) -> None:
    output_frames = _read_frames_by_episode(staging_root)
    output_episodes = _read_episode_metadata(staging_root)
    if sorted(output_frames) != list(range(EXPECTED_EPISODES)):
        raise TrimMaterializationError("materialized output episode indexes are invalid")
    limits = np.asarray(contract.envelope.step_limits, dtype=np.float64)
    for episode_plan in plan.episodes:
        output_rows = output_frames[episode_plan.episode_index]
        if len(output_rows) != episode_plan.trimmed_length:
            raise TrimMaterializationError(
                f"materialized episode {episode_plan.episode_index} has wrong length"
            )
        if output_episodes[episode_plan.episode_index].get("length") != len(output_rows):
            raise TrimMaterializationError(
                f"materialized episode {episode_plan.episode_index} metadata has wrong length"
            )
        actions = np.asarray([row["action"] for row in output_rows], dtype=np.float64)
        states = np.asarray(
            [row["observation.state"] for row in output_rows], dtype=np.float64
        )
        if not np.array_equal(
            actions[0].astype(np.float32),
            np.asarray(episode_plan.trimmed_first_action, dtype=np.float32),
        ):
            raise TrimMaterializationError(
                f"materialized episode {episode_plan.episode_index} starts at the wrong action"
            )
        first_window = actions[:VALIDATION_WINDOW_FRAMES, :ARM_DIMENSIONS]
        motion = float(np.max(np.abs(first_window - first_window[0])))
        if motion <= ONSET_THRESHOLD_RAD:
            raise TrimMaterializationError(
                f"materialized episode {episode_plan.episode_index} lost its onset"
            )
        if np.any(np.abs(actions[0] - states[0]) > limits + 1e-12):
            raise TrimMaterializationError(
                f"materialized episode {episode_plan.episode_index} first action-state "
                "error exceeds the controller contract"
            )

    try:
        report = audit(
            staging_root,
            DESTINATION_REPO_ID,
            controller_contract=contract,
            expected_task=EXPECTED_TASK,
        )
    except DatasetAuditInputError as exc:
        raise TrimMaterializationError(
            f"materialized output audit could not complete: {exc}"
        ) from exc
    _require_safe_audit(report)

    manifest_path = staging_root / "meta" / MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path, label="trim manifest")
    validate_trim_manifest(manifest, expected_root=DESTINATION_ROOT)


def materialize_trimmed_dataset(
    *,
    source_root: Path = SOURCE_ROOT,
    destination_root: Path = DESTINATION_ROOT,
    controller_contract_path: Path = CONTROLLER_CONTRACT,
    decode_batch_frames: int = DEFAULT_DECODE_BATCH_FRAMES,
    audit: AuditFunction = audit_firefly_dataset,
) -> Path:
    source, destination = _validate_fixed_paths(source_root, destination_root)
    _require_offline_video_tools()
    plan, info, frames, episodes, contract = prepare_plan(
        source,
        controller_contract_path=controller_contract_path,
        audit=audit,
    )
    source_manifest, source_identities = _source_inventory(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-", dir=destination.parent
        )
    )
    staging = staging_parent / "dataset"
    try:
        _materialize_to_staging(
            source_root=source,
            staging_root=staging,
            info=info,
            frames_by_episode=frames,
            episode_metadata=episodes,
            plan=plan,
            decode_batch_frames=decode_batch_frames,
        )
        manifest = _manifest_document(
            plan=plan,
            source_manifest=source_manifest,
            controller_contract_path=controller_contract_path,
            contract=contract,
        )
        manifest_path = staging / "meta" / MANIFEST_FILENAME
        manifest_path.write_text(_strict_json(manifest) + "\n", encoding="utf-8")
        _validate_materialized_output(
            staging_root=staging,
            plan=plan,
            contract=contract,
            audit=audit,
        )
        _verify_source_identities(source, source_manifest, source_identities)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"destination appeared during materialization and will not be overwritten: "
                f"{destination}"
            )
        staging.rename(destination)
        staging_parent.rmdir()
    except Exception as exc:
        raise TrimMaterializationError(
            f"materialization failed; source was not modified and partial output was "
            f"left at {staging_parent}: {exc}"
        ) from exc
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and print the exact trim plan without writing any file.",
    )
    parser.add_argument(
        "--decode-batch-frames",
        type=int,
        default=DEFAULT_DECODE_BATCH_FRAMES,
        help="Official PyAV decoder batch size for a real run (default: 60).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_fixed_paths(SOURCE_ROOT, DESTINATION_ROOT)
        _require_offline_video_tools()
        plan, _info, _frames, _episodes, _contract = prepare_plan()
        summary = plan_summary(plan)
        if args.dry_run:
            print(_strict_json(summary))
            return 0
        destination = materialize_trimmed_dataset(
            decode_batch_frames=args.decode_batch_frames
        )
        print(
            _strict_json(
                {
                    "status": "materialized",
                    "destination": str(destination),
                    "manifest": str(destination / "meta" / MANIFEST_FILENAME),
                    "episode_count": len(plan.episodes),
                    "total_frames": plan.destination_total_frames,
                }
            )
        )
        return 0
    except (
        TrimMaterializationError,
        DatasetAuditInputError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"trim materialization refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
