from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_robot_hexbase.firefly_fullcycle_v4 import (
    FullcycleV4ProtocolError,
    audit_start_motion_windows,
    validate_start_motion_windows,
)


def _rows(
    *,
    onset_frame: int | None,
    jaw_onset_frame: int | None = None,
    spike_frame: int | None = None,
    frames: int = 75,
) -> list[dict]:
    rows: list[dict] = []
    for frame_index in range(frames):
        arm = 0.0
        if onset_frame is not None and frame_index >= onset_frame:
            arm = 0.006
        if spike_frame is not None and frame_index == spike_frame:
            arm = 0.006
        jaw = 0.38
        if jaw_onset_frame is not None and frame_index >= jaw_onset_frame:
            jaw = 0.20
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame_index,
                "action": [arm, 0.0, 0.0, 0.0, 0.0, 0.0, jaw],
            }
        )
    return rows


@pytest.mark.parametrize(
    ("onset_frame", "safe", "status"),
    [
        (45, True, "passed"),
        (46, True, "warning"),
        (60, True, "warning"),
        (61, False, "error"),
    ],
)
def test_start_motion_boundaries(onset_frame, safe, status):
    report = audit_start_motion_windows(
        Path("/unused"),
        expected_episodes=1,
        rows=_rows(onset_frame=onset_frame),
    )

    assert report["safe_for_training"] is safe
    assert report["episodes"][0]["onset_frame"] == onset_frame
    assert report["episodes"][0]["status"] == status


def test_jaw_motion_never_counts_as_arm_onset():
    report = audit_start_motion_windows(
        Path("/unused"),
        expected_episodes=1,
        rows=_rows(onset_frame=None, jaw_onset_frame=1),
    )

    assert report["safe_for_training"] is False
    assert report["errors"][0]["code"] == "initial_arm_motion_missing"


def test_single_frame_arm_spike_does_not_replace_confirmed_onset():
    report = audit_start_motion_windows(
        Path("/unused"),
        expected_episodes=1,
        rows=_rows(onset_frame=45, spike_frame=10),
    )

    assert report["episodes"][0]["onset_frame"] == 45


def test_validator_raises_only_after_hard_limit():
    accepted = validate_start_motion_windows(
        Path("/unused"),
        expected_episodes=1,
        rows=_rows(onset_frame=60),
    )
    assert accepted["safe_for_training"] is True

    with pytest.raises(FullcycleV4ProtocolError, match="too_late"):
        validate_start_motion_windows(
            Path("/unused"),
            expected_episodes=1,
            rows=_rows(onset_frame=61),
        )
