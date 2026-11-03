from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np
import torch

from lerobot_robot_hexbase.firefly_dataset_audit import (
    EXPECTED_JOINT_NAMES,
    firefly_contract_from_mapping,
)
from scripts import materialize_firefly_y6_trimmed as trimming


def _contract():
    return firefly_contract_from_mapping(
        {
            "source": "test controller contract",
            "joint_names": list(EXPECTED_JOINT_NAMES),
            "pos_min": [-2.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.0],
            "pos_max": [2.0, 2.0, 3.2, 2.0, 2.0, 2.0, 0.6],
            "effective_step_limits": [0.0435] * 6 + [0.0165],
        }
    )


def _source_tables(*, episodes: int = trimming.EXPECTED_EPISODES):
    frames: dict[int, list[dict]] = {}
    metadata: dict[int, dict] = {}
    global_index = 0
    for episode_index in range(episodes):
        rows = []
        for frame_index in range(15):
            action = [0.0] * 7
            # A gripper change is deliberately earlier than the arm motion and
            # must not be selected as the onset.
            if frame_index >= 1:
                action[-1] = 0.02
            # Equality at frame 4 must not satisfy the strict > 0.005 contract.
            if frame_index == 4:
                action[0] = trimming.ONSET_THRESHOLD_RAD
            elif frame_index >= 5:
                action[0] = 0.006 + (frame_index - 5) * 0.001
            state = [value - 0.001 for value in action]
            rows.append(
                {
                    "action": action,
                    "observation.state": state,
                    "timestamp": frame_index / trimming.EXPECTED_FPS,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
        frames[episode_index] = rows
        metadata[episode_index] = {
            "episode_index": episode_index,
            "length": len(rows),
            "tasks": [trimming.EXPECTED_TASK],
        }
    return frames, metadata


def _plan():
    frames, metadata = _source_tables()
    return trimming.plan_trimmed_episodes(
        frames,
        metadata,
        effective_step_limits=_contract().envelope.step_limits,
    )


def test_plan_uses_arm_only_strict_threshold_and_two_frame_preroll():
    plan = _plan()

    assert len(plan.episodes) == 20
    assert plan.source_total_frames == 300
    assert plan.destination_total_frames == 240
    for episode in plan.episodes:
        assert episode.onset_frame == 5
        assert episode.trim_start_frame == 3
        assert episode.trimmed_length == 12
        assert episode.first_10_has_motion is True
        assert episode.first_action_state_within_contract is True


def test_no_arm_onset_fails_closed_even_when_gripper_moves():
    frames, metadata = _source_tables()
    for row in frames[7]:
        row["action"][:6] = [0.0] * 6

    with pytest.raises(
        trimming.TrimMaterializationError,
        match="episode 7 has no arm-action onset",
    ):
        trimming.plan_trimmed_episodes(
            frames,
            metadata,
            effective_step_limits=_contract().envelope.step_limits,
        )


def test_trimmed_first_action_state_must_fit_controller_contract():
    frames, metadata = _source_tables()
    # Onset is frame 5, so the pinned two-frame pre-roll starts at frame 3.
    frames[4][3]["observation.state"][0] = -0.1

    with pytest.raises(
        trimming.TrimMaterializationError,
        match="episode 4 trimmed first action-state error exceeds",
    ):
        trimming.plan_trimmed_episodes(
            frames,
            metadata,
            effective_step_limits=_contract().envelope.step_limits,
        )


def test_noncontiguous_or_wrong_episode_count_is_rejected():
    frames, metadata = _source_tables(episodes=19)

    with pytest.raises(
        trimming.TrimMaterializationError,
        match="exactly contiguous episodes",
    ):
        trimming.plan_trimmed_episodes(
            frames,
            metadata,
            effective_step_limits=_contract().envelope.step_limits,
        )


def test_plan_summary_exposes_training_gate_fields():
    summary = trimming.plan_summary(_plan())

    assert summary["source"]["repo_id"] == trimming.SOURCE_REPO_ID
    assert summary["destination"]["repo_id"] == trimming.DESTINATION_REPO_ID
    assert summary["algorithm"]["threshold_rad"] == 0.005
    assert summary["algorithm"]["pre_roll_frames"] == 2
    assert summary["summary"] == {
        "episode_count": 20,
        "removed_frames": 60,
        "all_first10_has_motion": True,
        "all_first_action_state_within_contract": True,
    }


def test_fixed_manifest_is_valid_and_tampering_is_detected(tmp_path: Path):
    contract_path = tmp_path / "controller_contract.json"
    contract_path.write_text('{"test":true}', encoding="utf-8")
    document = trimming._manifest_document(
        plan=_plan(),
        source_manifest=[
            {"path": "meta/info.json", "size": 2, "sha256": "ab"}
        ],
        controller_contract_path=contract_path,
        contract=_contract(),
    )

    trimming.validate_trim_manifest(document, expected_root=trimming.DESTINATION_ROOT)
    assert document["source"]["repo_id"] == "local/firefly_y6_manual_validated_v2"
    assert document["source"]["episode_count"] == 20
    assert document["algorithm"]["threshold_rad"] == 0.005
    assert document["algorithm"]["pre_roll_frames"] == 2
    assert all(episode["first_10_has_motion"] for episode in document["episodes"])

    document["episodes"][3]["first_10_has_motion"] = False
    with pytest.raises(trimming.TrimMaterializationError, match="lacks first-10 motion"):
        trimming.validate_trim_manifest(document)


def test_existing_destination_is_never_overwritten(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    marker = destination / "owned-by-user"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        trimming._validate_fixed_paths(source, destination)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_does_not_allow_threshold_or_preroll_overrides():
    with pytest.raises(SystemExit) as threshold:
        trimming.parse_args(["--threshold-rad=0.02"])
    with pytest.raises(SystemExit) as preroll:
        trimming.parse_args(["--pre-roll-frames=0"])

    assert threshold.value.code == 2
    assert preroll.value.code == 2


def test_video_timestamp_mapping_preserves_source_file_offset():
    episode = {
        "videos/observation.images.scene/from_timestamp": 54.9,
    }

    timestamps = trimming._source_video_timestamps(
        episode,
        "observation.images.scene",
        [166, 167, 168],
    )

    assert timestamps == pytest.approx(
        [54.9 + 166 / 30, 54.9 + 167 / 30, 54.9 + 168 / 30]
    )
def test_ep17_style_fallback_moves_forward_and_gates_from_trimmed_first():
    frames, metadata = _source_tables()
    for frame_index, row in enumerate(frames[17]):
        if frame_index == 0:
            value = 0.0
        elif frame_index <= 4:
            value = 0.004
        elif frame_index <= 12:
            value = 0.0051
        else:
            value = 0.0051 + (frame_index - 12) * 0.003
        row["action"][0] = value
        row["observation.state"][0] = value - 0.001

    plan = trimming.plan_trimmed_episodes(
        frames,
        metadata,
        effective_step_limits=_contract().envelope.step_limits,
    )
    episode = plan.episodes[17]

    assert episode.onset_frame == 5
    assert episode.trim_adjusted_for_first_10_motion is True
    assert episode.first_10_motion_gate_frame == 14
    assert episode.trim_start_frame == 12
    assert episode.first_10_has_motion is True
    assert episode.first_10_max_abs_arm_delta_from_trimmed_first_rad > 0.005
    assert episode.first_10_motion_gate_frame - episode.trim_start_frame <= 2


def test_offline_writer_is_nonstreaming_hwc_uint8_and_root_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.configs as video_configs
    import lerobot.datasets.video_utils as video_utils

    frames, metadata = _source_tables(episodes=1)
    plan = trimming.plan_trimmed_episodes(
        frames,
        metadata,
        effective_step_limits=_contract().envelope.step_limits,
        expected_episodes=1,
    )
    source = tmp_path / "source"
    for video_key in trimming._VIDEO_KEYS:
        path = source / "videos" / video_key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        metadata[0][f"videos/{video_key}/chunk_index"] = 0
        metadata[0][f"videos/{video_key}/file_index"] = 0
        metadata[0][f"videos/{video_key}/from_timestamp"] = 0.0

    features = {
        "action": {"dtype": "float32", "shape": [7], "names": None},
        "observation.state": {"dtype": "float32", "shape": [7], "names": None},
    }
    for video_key in trimming._VIDEO_KEYS:
        features[video_key] = {
            "dtype": "video",
            "shape": [4, 5, 3],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
    info = {
        "features": features,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "test",
    }
    seen: dict = {"frames": []}

    class FakeWriter:
        def add_frame(self, frame):
            for video_key in trimming._VIDEO_KEYS:
                image = frame[video_key]
                assert isinstance(image, np.ndarray)
                assert image.dtype == np.uint8
                assert image.shape == (4, 5, 3)
            seen["frames"].append(frame)

        def save_episode(self, *, parallel_encoding):
            assert parallel_encoding is False

        def finalize(self):
            seen["finalized"] = True

    def fake_create(**kwargs):
        seen["create"] = kwargs
        assert not Path(kwargs["root"]).exists()
        return FakeWriter()

    def fake_decode(_path, timestamps, **kwargs):
        assert kwargs == {
            "tolerance_s": 1e-4,
            "backend": "pyav",
            "return_uint8": True,
        }
        return torch.zeros((len(timestamps), 3, 4, 5), dtype=torch.uint8)

    monkeypatch.setattr(video_configs, "encoder_config_from_video_info", lambda _info: "encoder")
    monkeypatch.setattr(LeRobotDataset, "create", fake_create)
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)

    staging = tmp_path / "partial" / "dataset"
    trimming._materialize_to_staging(
        source_root=source,
        staging_root=staging,
        info=info,
        frames_by_episode=frames,
        episode_metadata=metadata,
        plan=plan,
        decode_batch_frames=4,
    )

    assert seen["create"]["streaming_encoding"] is False
    assert seen["create"]["image_writer_processes"] == 0
    assert seen["create"]["image_writer_threads"] == trimming.OFFLINE_IMAGE_WRITER_THREADS
    assert "encoder_queue_maxsize" not in seen["create"]
    assert len(seen["frames"]) == plan.destination_total_frames
    assert seen["finalized"] is True


def test_output_motion_validation_is_relative_to_trimmed_first(monkeypatch, tmp_path):
    plan = _plan()
    first = replace(
        plan.episodes[0],
        trimmed_first_action=(0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02),
    )
    plan = replace(plan, episodes=(first, *plan.episodes[1:]))
    output_frames = {episode: [] for episode in range(trimming.EXPECTED_EPISODES)}
    for frame_index in range(first.trimmed_length):
        action = [0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
        if frame_index >= trimming.VALIDATION_WINDOW_FRAMES:
            action[0] = 0.02
        output_frames[0].append(
            {"action": action, "observation.state": [value - 0.001 for value in action]}
        )
    output_metadata = {
        episode.episode_index: {"length": episode.trimmed_length}
        for episode in plan.episodes
    }
    monkeypatch.setattr(trimming, "_read_frames_by_episode", lambda _root: output_frames)
    monkeypatch.setattr(trimming, "_read_episode_metadata", lambda _root: output_metadata)

    with pytest.raises(trimming.TrimMaterializationError, match="lost its onset"):
        trimming._validate_materialized_output(
            staging_root=tmp_path,
            plan=plan,
            contract=_contract(),
            audit=lambda *_args, **_kwargs: pytest.fail("audit must not run"),
        )
