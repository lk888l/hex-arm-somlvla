"""Fixed data-quality gates for the Firefly Y6 full-cycle v4 dataset."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FPS = 30
PREFERRED_START_MOTION_S = 1.5
MAX_START_MOTION_S = 2.0
START_MOTION_THRESHOLD_RAD = 0.005
START_MOTION_CONFIRM_FRAMES = 2


class FullcycleV4ProtocolError(RuntimeError):
    """The dataset does not satisfy the fixed full-cycle v4 protocol."""


def _load_action_rows(dataset_root: Path) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise FullcycleV4ProtocolError(
            "pyarrow is required to audit full-cycle start motion"
        ) from exc

    paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FullcycleV4ProtocolError(
            f"dataset contains no data parquet files: {dataset_root}"
        )
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            table = parquet.read_table(
                path,
                columns=["episode_index", "frame_index", "action"],
            )
        except Exception as exc:
            raise FullcycleV4ProtocolError(
                f"cannot read action rows from {path}: {exc}"
            ) from exc
        rows.extend(table.to_pylist())
    return rows


def audit_start_motion_windows(
    dataset_root: Path,
    *,
    expected_episodes: int,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit purposeful arm-action onset without trimming or changing data."""

    if type(expected_episodes) is not int or expected_episodes <= 0:
        raise FullcycleV4ProtocolError("expected_episodes must be a positive integer")
    source_rows = _load_action_rows(dataset_root) if rows is None else list(rows)
    by_episode: dict[int, list[tuple[int, tuple[float, ...]]]] = {}
    for row_number, row in enumerate(source_rows):
        try:
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            raw_action = row["action"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FullcycleV4ProtocolError(
                f"invalid action row {row_number}: missing or invalid indexes/action"
            ) from exc
        if (
            not isinstance(raw_action, Sequence)
            or isinstance(raw_action, (str, bytes))
            or len(raw_action) != 7
        ):
            raise FullcycleV4ProtocolError(
                f"invalid action row {row_number}: expected seven action values"
            )
        try:
            action = tuple(float(value) for value in raw_action)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FullcycleV4ProtocolError(
                f"invalid action row {row_number}: action is not numeric"
            ) from exc
        if not all(math.isfinite(value) for value in action):
            raise FullcycleV4ProtocolError(
                f"invalid action row {row_number}: action is not finite"
            )
        by_episode.setdefault(episode_index, []).append((frame_index, action))

    expected_indexes = set(range(expected_episodes))
    actual_indexes = set(by_episode)
    if actual_indexes != expected_indexes:
        raise FullcycleV4ProtocolError(
            "start-motion audit episode indexes mismatch: "
            f"expected={sorted(expected_indexes)}, actual={sorted(actual_indexes)}"
        )

    preferred_frame = round(PREFERRED_START_MOTION_S * FPS)
    hard_frame = round(MAX_START_MOTION_S * FPS)
    episodes: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for episode_index in range(expected_episodes):
        episode_rows = sorted(by_episode[episode_index], key=lambda item: item[0])
        frame_indexes = [item[0] for item in episode_rows]
        if not frame_indexes or frame_indexes != list(range(len(frame_indexes))):
            raise FullcycleV4ProtocolError(
                f"episode {episode_index} frame indexes are not contiguous from zero"
            )
        baseline = episode_rows[0][1][:6]
        motion_flags = [
            max(abs(action[joint] - baseline[joint]) for joint in range(6))
            > START_MOTION_THRESHOLD_RAD
            for _, action in episode_rows
        ]
        onset_frame: int | None = None
        for candidate in range(
            0,
            len(motion_flags) - START_MOTION_CONFIRM_FRAMES + 1,
        ):
            if all(
                motion_flags[candidate : candidate + START_MOTION_CONFIRM_FRAMES]
            ):
                onset_frame = episode_rows[candidate][0]
                break

        onset_s = None if onset_frame is None else onset_frame / FPS
        within_preferred = onset_frame is not None and onset_frame <= preferred_frame
        within_hard = onset_frame is not None and onset_frame <= hard_frame
        status = "passed"
        if onset_frame is None:
            status = "error"
            errors.append(
                {
                    "episode_index": episode_index,
                    "code": "initial_arm_motion_missing",
                    "message": "no confirmed arm action onset; jaw motion is ignored",
                }
            )
        elif not within_hard:
            status = "error"
            errors.append(
                {
                    "episode_index": episode_index,
                    "code": "initial_arm_motion_too_late",
                    "onset_frame": onset_frame,
                    "onset_s": onset_s,
                    "max_s": MAX_START_MOTION_S,
                }
            )
        elif not within_preferred:
            status = "warning"
            warnings.append(
                {
                    "episode_index": episode_index,
                    "code": "initial_arm_motion_after_preferred_window",
                    "onset_frame": onset_frame,
                    "onset_s": onset_s,
                    "preferred_s": PREFERRED_START_MOTION_S,
                }
            )
        episodes.append(
            {
                "episode_index": episode_index,
                "onset_frame": onset_frame,
                "onset_s": onset_s,
                "within_preferred": within_preferred,
                "within_hard_limit": within_hard,
                "status": status,
            }
        )

    return {
        "schema_version": 1,
        "kind": "firefly_fullcycle_v4_start_motion",
        "safe_for_training": not errors,
        "profile": {
            "fps": FPS,
            "preferred_start_motion_s": PREFERRED_START_MOTION_S,
            "max_start_motion_s": MAX_START_MOTION_S,
            "motion_threshold_rad": START_MOTION_THRESHOLD_RAD,
            "confirmation_frames": START_MOTION_CONFIRM_FRAMES,
            "motion_dimensions": "action[0:6] (jaw excluded)",
        },
        "episodes": episodes,
        "warnings": warnings,
        "errors": errors,
    }


def validate_start_motion_windows(
    dataset_root: Path,
    *,
    expected_episodes: int,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the report, raising when any episode exceeds the hard limit."""

    report = audit_start_motion_windows(
        dataset_root,
        expected_episodes=expected_episodes,
        rows=rows,
    )
    if report["safe_for_training"] is not True:
        first = report["errors"][0]
        raise FullcycleV4ProtocolError(
            f"episode {first['episode_index']} failed start-motion gate: "
            f"{first['code']}"
        )
    return report
