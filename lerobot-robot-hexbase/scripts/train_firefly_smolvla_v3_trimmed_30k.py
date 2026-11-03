#!/usr/bin/env python3
"""Launch the fixed 20-episode Firefly v3-trimmed SmolVLA 30k job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "local/firefly_y6_manual_validated_v3_trimmed"
DATASET_ROOT = Path(
    "/home/kk/kk_data/lerobot/datasets/local/"
    "firefly_y6_manual_validated_v3_trimmed"
)
CONTROLLER_CONTRACT = (
    REPOSITORY_ROOT
    / "outputs/firefly_diagnostics/"
    "controller_contract_validated_v2_arm0435_ee0165.json"
)
OUTPUT_DIR = Path(
    "/home/kk/kk_data/lerobot/outputs/train/"
    "firefly_y6_smolvla_validated_v3_trimmed_attested_run1"
)
SMOLVLA_BASE = Path(
    "/home/kk/kk_data/lerobot/assets/firefly/"
    "smolvla_base-c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)
CONFIG = REPOSITORY_ROOT / "configs/train_smolvla_firefly_y6_v3_trimmed.yaml"
EXPECTED_TASK = "Pick up the red block and place it in the blue box."
EXPECTED_EPISODES = 20
MAX_TRAINING_FRAMES = 30_000
SOURCE_REPO_ID = "local/firefly_y6_manual_validated_v2"
TRIM_MANIFEST_FILENAME = "firefly_trim_manifest.json"
TRIM_MANIFEST_SCHEMA_VERSION = 1
TRIM_MANIFEST_KIND = "firefly_action_onset_trim"
TRIM_THRESHOLD_RAD = 0.005
TRIM_PRE_ROLL_FRAMES = 2
TRIM_VALIDATION_WINDOW_FRAMES = 10


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"invalid JSON constant {token}")


def _read_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _is_exact_int(value: object, expected: int | None = None) -> bool:
    if type(value) is not int:
        return False
    return expected is None or value == expected


def _validate_trim_manifest(
    manifest: dict,
    *,
    dataset_root: Path,
    total_frames: int,
) -> None:
    if not _is_exact_int(
        manifest.get("schema_version"), TRIM_MANIFEST_SCHEMA_VERSION
    ):
        raise RuntimeError("trim manifest schema_version mismatch")
    if manifest.get("kind") != TRIM_MANIFEST_KIND:
        raise RuntimeError("trim manifest kind mismatch")

    source = manifest.get("source")
    destination = manifest.get("destination")
    algorithm = manifest.get("algorithm")
    summary = manifest.get("summary")
    episodes = manifest.get("episodes")
    if not all(
        isinstance(value, dict)
        for value in (source, destination, algorithm, summary)
    ) or not isinstance(episodes, list):
        raise RuntimeError("trim manifest has missing structured fields")

    if source.get("repo_id") != SOURCE_REPO_ID:
        raise RuntimeError("trim manifest source repo_id mismatch")
    if not _is_exact_int(source.get("episode_count"), EXPECTED_EPISODES):
        raise RuntimeError("trim manifest source episode count mismatch")
    if destination.get("repo_id") != REPO_ID:
        raise RuntimeError("trim manifest destination repo_id mismatch")
    if destination.get("root") != str(dataset_root):
        raise RuntimeError("trim manifest destination root mismatch")
    if not _is_exact_int(destination.get("episode_count"), EXPECTED_EPISODES):
        raise RuntimeError("trim manifest destination episode count mismatch")
    if not _is_exact_int(destination.get("total_frames"), total_frames):
        raise RuntimeError("trim manifest destination total_frames mismatch")

    threshold = algorithm.get("threshold_rad")
    if type(threshold) not in (int, float) or threshold != TRIM_THRESHOLD_RAD:
        raise RuntimeError("trim manifest threshold mismatch")
    if not _is_exact_int(
        algorithm.get("pre_roll_frames"), TRIM_PRE_ROLL_FRAMES
    ):
        raise RuntimeError("trim manifest pre-roll mismatch")
    if not _is_exact_int(
        algorithm.get("validation_window_frames"),
        TRIM_VALIDATION_WINDOW_FRAMES,
    ):
        raise RuntimeError("trim manifest validation window mismatch")

    if not _is_exact_int(summary.get("episode_count"), EXPECTED_EPISODES):
        raise RuntimeError("trim manifest summary episode count mismatch")
    if summary.get("all_first10_has_motion") is not True:
        raise RuntimeError("trim manifest first-10 motion gate failed")
    if summary.get("all_first_action_state_within_contract") is not True:
        raise RuntimeError("trim manifest first action-state gate failed")
    if len(episodes) != EXPECTED_EPISODES:
        raise RuntimeError("trim manifest episode list must contain exactly 20 entries")
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise RuntimeError("trim manifest episode entry is invalid")
        if not _is_exact_int(episode.get("episode_index"), expected_index):
            raise RuntimeError("trim manifest episode indexes are not contiguous")
        if episode.get("first_10_has_motion") is not True:
            raise RuntimeError(
                f"trim manifest episode {expected_index} lacks first-10 motion"
            )
        if episode.get("first_action_state_within_contract") is not True:
            raise RuntimeError(
                f"trim manifest episode {expected_index} first action-state gate failed"
            )

    claimed_digest = manifest.get("document_sha256")
    unsigned = dict(manifest)
    unsigned.pop("document_sha256", None)
    if claimed_digest != _sha256_json(unsigned):
        raise RuntimeError("trim manifest document digest mismatch")


def validate_dataset_metadata(dataset_root: Path = DATASET_ROOT) -> None:
    """Require the fixed, proved 20-episode trimmed dataset before training."""

    info_path = dataset_root / "meta/info.json"
    info = _read_json_object(info_path, label="trimmed dataset metadata")
    episode_count = info.get("total_episodes")
    if not _is_exact_int(episode_count, EXPECTED_EPISODES):
        raise RuntimeError(
            "trimmed dataset must contain exactly "
            f"{EXPECTED_EPISODES} episodes, got {episode_count!r}"
        )
    splits = info.get("splits")
    expected_split = f"0:{EXPECTED_EPISODES}"
    if not isinstance(splits, dict) or splits.get("train") != expected_split:
        raise RuntimeError(
            f"trimmed dataset train split must be {expected_split!r}, got {splits!r}"
        )
    total_frames = info.get("total_frames")
    if (
        not _is_exact_int(total_frames)
        or total_frames <= 0
        or total_frames > MAX_TRAINING_FRAMES
    ):
        raise RuntimeError(
            "trimmed dataset total_frames must be an integer in "
            f"[1, {MAX_TRAINING_FRAMES}], got {total_frames!r}"
        )

    manifest_path = dataset_root / "meta" / TRIM_MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path, label="trim manifest")
    _validate_trim_manifest(
        manifest,
        dataset_root=dataset_root,
        total_frames=total_frames,
    )


def build_command() -> list[str]:
    wrapper = REPOSITORY_ROOT / "scripts/train_firefly_smolvla_attested.py"
    return [
        sys.executable,
        str(wrapper),
        f"--dataset-root={DATASET_ROOT}",
        f"--repo-id={REPO_ID}",
        f"--controller-contract={CONTROLLER_CONTRACT}",
        f"--expected-task={EXPECTED_TASK}",
        f"--output-dir={OUTPUT_DIR}",
        f"--smolvla-base={SMOLVLA_BASE}",
        "--",
        f"--config_path={CONFIG}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--steps=30000",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved wrapper command without validating data or training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command()
    if args.dry_run:
        print("\n".join(command))
        return 0

    try:
        validate_dataset_metadata()
    except RuntimeError as exc:
        print(f"v3 trimmed training refused: {exc}", file=sys.stderr)
        return 2

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
