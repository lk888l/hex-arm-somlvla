from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lerobot_robot_hexbase.firefly_visual_latch import (
    BlobEvidence,
    D405VisualDetector,
    FrameStatus,
    VisualEvidence,
    VisualGraspReleaseLatch,
    VisualLatchPhase,
    VisualPhaseLatchConfig,
    compose_visual_action,
)


_STEP_NS = 33_333_333


def _config(**overrides) -> VisualPhaseLatchConfig:
    defaults = dict(
        enabled=True,
        close_confirm_frames=2,
        grasp_confirm_frames=2,
        lift_confirm_frames=2,
        transport_confirm_frames=2,
        box_confirm_frames=2,
        release_confirm_cycles=2,
    )
    defaults.update(overrides)
    return replace(VisualPhaseLatchConfig(), **defaults)


def _evidence(
    index: int,
    *,
    red: bool = False,
    held_red: bool | None = None,
    blue: bool = False,
    status: FrameStatus = FrameStatus.FRESH_UNIQUE,
    evaluated_at_ns: int | None = None,
) -> VisualEvidence:
    captured = 1_000_000_000 + index * _STEP_NS
    evaluated = captured + 1_000_000 if evaluated_at_ns is None else evaluated_at_ns
    return VisualEvidence(
        frame_id=index,
        captured_at_ns=captured,
        evaluated_at_ns=evaluated,
        frame_age_ms=(evaluated - captured) / 1_000_000,
        frame_status=status,
        red=BlobEvidence(
            accepted=red,
            held_accepted=red if held_red is None else held_red,
        ),
        blue=BlobEvidence(accepted=blue),
    )


def _q(ee: float) -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ee]


def _prime_and_latch(
    latch: VisualGraspReleaseLatch,
    *,
    start_index: int = 0,
    z_m: float = 0.20,
) -> int:
    latch.step(
        _evidence(start_index), _q(0.314), _q(0.314), tcp_z_m=z_m, tcp_speed_m_s=0
    )
    latch.step(
        _evidence(start_index + 1),
        _q(0.39),
        _q(0.39),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
    )
    first = latch.step(
        _evidence(start_index + 2, red=True),
        _q(0.39),
        _q(0.41),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
        policy_close_intent=True,
    )
    assert first.phase_after is VisualLatchPhase.CLOSE_PENDING
    latched = latch.step(
        _evidence(start_index + 3, red=True),
        _q(0.39),
        _q(0.41),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
        policy_close_intent=True,
    )
    assert latched.phase_after is VisualLatchPhase.CLOSING
    return start_index + 4


def _confirm_grasp(
    latch: VisualGraspReleaseLatch, index: int, *, z_m: float = 0.20
) -> int:
    # First motion into contact is deliberately not a settled confirmation.
    latch.step(
        _evidence(index, red=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
    )
    latch.step(
        _evidence(index + 1, red=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
    )
    confirmed = latch.step(
        _evidence(index + 2, red=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=z_m,
        tcp_speed_m_s=0,
    )
    assert confirmed.phase_after is VisualLatchPhase.GRASP_CONFIRMED
    return index + 3


def test_detector_uses_rgb_without_second_rotation():
    config = _config()
    detector = D405VisualDetector(config)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # This square passes only in the already-rotated image. A second 180-degree
    # rotation would move its center outside the calibrated grasp ROI.
    rgb[207:312, 230:335] = (255, 0, 0)

    evidence = detector.detect(
        rgb,
        frame_id=1,
        captured_at_ns=1_000_000_000,
        now_ns=1_001_000_000,
    )

    assert evidence.usable
    assert evidence.red.accepted
    assert evidence.red.held_accepted
    assert evidence.red.centroid_norm == pytest.approx((0.440625, 0.5395833), abs=0.003)


def test_ambiguity_threshold_must_count_every_held_target():
    with pytest.raises(ValueError, match="ambiguity threshold"):
        _config(
            red_ambiguity_area_fraction_min=0.015,
            held_red_contour_area_fraction_min=0.015,
        )


def test_red_gate_rejects_ambiguous_candidates_and_bad_shape():
    detector = D405VisualDetector(_config())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[207:312, 230:335] = (255, 0, 0)
    rgb[20:90, 20:90] = (255, 0, 0)

    ambiguous = detector.detect(
        rgb, frame_id=1, captured_at_ns=1_000_000_000, now_ns=1_001_000_000
    )
    assert not ambiguous.red.accepted
    assert "red_candidates_ambiguous" in ambiguous.red.rejection_reasons
    assert "red_candidates_ambiguous" in ambiguous.red.held_rejection_reasons

    detector.reset()
    rgb[:] = 0
    rgb[245:270, 80:510] = (255, 0, 0)
    elongated = detector.detect(
        rgb, frame_id=2, captured_at_ns=2_000_000_000, now_ns=2_001_000_000
    )
    assert not elongated.red.accepted
    assert "red_aspect_ratio_outside_gate" in elongated.red.rejection_reasons


def test_blue_gate_requires_global_roi_and_shape():
    detector = D405VisualDetector(_config())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:] = (0, 0, 255)
    accepted = detector.detect(
        rgb, frame_id=1, captured_at_ns=1_000_000_000, now_ns=1_001_000_000
    )
    assert accepted.blue.accepted
    assert accepted.blue.mask_area_fraction == pytest.approx(1.0)
    assert accepted.blue.roi_mask_fraction == pytest.approx(1.0)

    detector.reset()
    rgb[:] = 0
    rgb[160:480, 120:400] = (0, 0, 255)
    rejected = detector.detect(
        rgb, frame_id=2, captured_at_ns=2_000_000_000, now_ns=2_001_000_000
    )
    assert not rejected.blue.accepted
    assert "blue_global_fraction_below_gate" in rejected.blue.rejection_reasons


def test_duplicate_stale_and_out_of_order_frames_are_not_usable():
    detector = D405VisualDetector(_config())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    first = detector.detect(
        rgb, frame_id=1, captured_at_ns=1_000_000_000, now_ns=1_001_000_000
    )
    duplicate = detector.detect(
        rgb, frame_id=1, captured_at_ns=1_000_000_000, now_ns=1_020_000_000
    )
    out_of_order = detector.detect(
        rgb, frame_id=2, captured_at_ns=999_000_000, now_ns=1_020_000_000
    )
    stale = detector.detect(
        rgb, frame_id=3, captured_at_ns=1_010_000_000, now_ns=1_200_000_000
    )

    assert first.frame_status is FrameStatus.FRESH_UNIQUE
    assert duplicate.frame_status is FrameStatus.DUPLICATE
    assert out_of_order.frame_status is FrameStatus.OUT_OF_ORDER
    assert stale.frame_status is FrameStatus.STALE
    assert not duplicate.usable and not out_of_order.usable and not stale.usable


def test_initial_closed_jaw_passes_policy_until_natural_preopen():
    config = _config()
    latch = VisualGraspReleaseLatch(config)
    initial = latch.step(
        _evidence(0, red=True),
        _q(0.314),
        _q(0.314),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    composed = compose_visual_action(
        _q(0.35), _q(0.314), initial, ee_step_limit_rad=config.ee_step_limit_rad
    )
    assert initial.ee_target_rad is None
    assert composed.action[6] == pytest.approx(0.35)
    assert initial.phase_after is VisualLatchPhase.SEEKING

    opened = latch.step(
        _evidence(1), _q(0.39), _q(0.39), tcp_z_m=0.2, tcp_speed_m_s=0
    )
    assert opened.pregrasp_open_seen
    assert opened.ee_target_rad == pytest.approx(0.41)

    vision_only = latch.step(
        _evidence(2, red=True),
        _q(0.39),
        _q(0.41),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    assert vision_only.phase_after is VisualLatchPhase.SEEKING
    assert vision_only.close_streak == 0
    assert not vision_only.policy_close_intent

    policy_and_vision = latch.step(
        _evidence(3, red=True),
        _q(0.39),
        _q(0.41),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
        policy_close_intent=True,
    )
    assert policy_and_vision.phase_after is VisualLatchPhase.CLOSE_PENDING
    assert policy_and_vision.policy_close_intent


def test_duplicate_frame_pauses_close_debounce_until_new_evidence():
    latch = VisualGraspReleaseLatch(_config(close_confirm_frames=2))
    latch.step(_evidence(0), _q(0.39), _q(0.39), tcp_z_m=0.2, tcp_speed_m_s=0)
    pending = latch.step(
        _evidence(1, red=True),
        _q(0.39),
        _q(0.41),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
        policy_close_intent=True,
    )
    duplicate = latch.step(
        _evidence(1, red=True, status=FrameStatus.DUPLICATE, evaluated_at_ns=1_080_000_000),
        _q(0.39),
        _q(0.41),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
        policy_close_intent=True,
    )
    assert pending.phase_after is VisualLatchPhase.CLOSE_PENDING
    assert duplicate.phase_after is VisualLatchPhase.CLOSE_PENDING
    assert duplicate.close_streak == 1
    stale = latch.step(
        _evidence(2, red=True, status=FrameStatus.STALE),
        _q(0.39),
        _q(0.41),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    assert stale.phase_after is VisualLatchPhase.SEEKING
    assert stale.close_streak == 0


def test_frozen_camera_in_seeking_eventually_fails_closed():
    latch = VisualGraspReleaseLatch(_config(vision_loss_fault_ms=50.0))
    first_evidence = _evidence(0)
    first = latch.step(
        first_evidence,
        _q(0.39),
        _q(0.39),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    frozen = latch.step(
        _evidence(
            0,
            status=FrameStatus.DUPLICATE,
            evaluated_at_ns=first_evidence.evaluated_at_ns + 60_000_000,
        ),
        _q(0.39),
        _q(0.39),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )

    assert first.phase_after is VisualLatchPhase.SEEKING
    assert frozen.phase_after is VisualLatchPhase.FAULT_HOLD
    assert frozen.fault_reason == "vision_unavailable"
    assert frozen.arm_hold

def test_contact_gap_confirms_grasp_and_empty_close_does_not():
    latch = VisualGraspReleaseLatch(_config(grasp_confirm_frames=2))
    index = _prime_and_latch(latch)
    for offset in range(4):
        empty = latch.step(
            _evidence(index + offset, red=True),
            _q(0.321),
            _q(0.32),
            tcp_z_m=0.2,
            tcp_speed_m_s=0,
        )
    assert empty.phase_after is VisualLatchPhase.CLOSING
    assert empty.grasp_streak == 0

    # A stable, blocked jaw has the demonstrated positive contact gap.
    index += 4
    latch.step(
        _evidence(index, red=True), _q(0.35), _q(0.32), tcp_z_m=0.2, tcp_speed_m_s=0
    )
    latch.step(
        _evidence(index + 1, red=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    held = latch.step(
        _evidence(index + 2, red=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.2,
        tcp_speed_m_s=0,
    )
    assert held.phase_after is VisualLatchPhase.GRASP_CONFIRMED


def test_release_requires_lift_transport_blue_and_low_speed():
    config = _config()
    latch = VisualGraspReleaseLatch(config)
    index = _confirm_grasp(latch, _prime_and_latch(latch))

    # Blue is visible, but neither lift nor transport has been proven.
    early = latch.step(
        _evidence(index, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.20,
        tcp_speed_m_s=0,
    )
    assert early.phase_after is VisualLatchPhase.GRASP_CONFIRMED

    for offset in range(1, 3):
        lifted = latch.step(
            _evidence(index + offset, held_red=True),
            _q(0.35),
            _q(0.32),
            tcp_z_m=0.225,
            tcp_speed_m_s=0,
        )
    assert lifted.phase_after is VisualLatchPhase.LIFT_CONFIRMED
    index += 3
    for offset in range(2):
        transported = latch.step(
            _evidence(index + offset, held_red=True),
            _q(0.35),
            _q(0.32),
            tcp_z_m=0.305,
            tcp_speed_m_s=0,
        )
    assert transported.phase_after is VisualLatchPhase.TRANSPORT_CONFIRMED
    index += 2

    moving = latch.step(
        _evidence(index, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0.1,
    )
    assert moving.box_streak == 1
    duplicate_box = latch.step(
        _evidence(
            index,
            held_red=True,
            blue=True,
            status=FrameStatus.DUPLICATE,
            evaluated_at_ns=_evidence(index).evaluated_at_ns + 10_000_000,
        ),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0,
    )
    assert duplicate_box.phase_after is VisualLatchPhase.TRANSPORT_CONFIRMED
    assert duplicate_box.box_streak == 1
    aligned = latch.step(
        _evidence(index + 1, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0.1,
    )
    assert aligned.phase_after is VisualLatchPhase.BOX_ALIGNED
    assert aligned.arm_hold
    assert aligned.settle_streak == 0

    # Box evidence is already stable, but high TCP speed cannot advance the
    # independent final-settle gate.
    unsettled = latch.step(
        _evidence(index + 2, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0.1,
    )
    assert unsettled.phase_after is VisualLatchPhase.BOX_ALIGNED
    assert unsettled.settle_streak == 0
    first_settled = latch.step(
        _evidence(index + 3, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0,
    )
    assert first_settled.settle_streak == 1
    duplicate_settle = latch.step(
        _evidence(
            index + 3,
            held_red=True,
            blue=True,
            status=FrameStatus.DUPLICATE,
            evaluated_at_ns=_evidence(index + 3).evaluated_at_ns + 10_000_000,
        ),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.305,
        tcp_speed_m_s=0,
    )
    assert duplicate_settle.phase_after is VisualLatchPhase.BOX_ALIGNED
    assert duplicate_settle.settle_streak == 1
    for offset in range(4, 6):
        releasing = latch.step(
            _evidence(index + offset, held_red=True, blue=True),
            _q(0.35),
            _q(0.32),
            tcp_z_m=0.305,
            tcp_speed_m_s=0,
        )
    assert releasing.phase_after is VisualLatchPhase.RELEASING
    assert releasing.arm_hold
    assert releasing.ee_target_rad == pytest.approx(0.42)

    latch.step(
        _evidence(index + 6, status=FrameStatus.INVALID),
        _q(0.40),
        _q(0.42),
        tcp_z_m=None,
        tcp_speed_m_s=None,
    )
    released = latch.step(
        _evidence(index + 7, status=FrameStatus.INVALID),
        _q(0.40),
        _q(0.42),
        tcp_z_m=None,
        tcp_speed_m_s=None,
    )
    assert released.phase_after is VisualLatchPhase.RELEASED
    assert released.ee_target_rad == pytest.approx(0.42)

    # An early policy re-close cannot affect EE while the arm returns home.
    still_returning = latch.step(
        _evidence(index + 8, status=FrameStatus.INVALID),
        _q(0.40),
        _q(0.42),
        tcp_z_m=None,
        tcp_speed_m_s=None,
        home_ready=False,
    )
    blocked_reclose = compose_visual_action(
        _q(0.10),
        _q(0.42),
        still_returning,
        ee_step_limit_rad=config.ee_step_limit_rad,
    )
    assert still_returning.phase_after is VisualLatchPhase.RELEASED
    assert blocked_reclose.action[6] == pytest.approx(0.42)

    # Only five consecutive externally verified home cycles arm empty-jaw close.
    for offset in range(9, 14):
        home_closing = latch.step(
            _evidence(index + offset, status=FrameStatus.INVALID),
            _q(0.40),
            _q(0.42),
            tcp_z_m=None,
            tcp_speed_m_s=None,
            home_ready=True,
        )
    assert home_closing.phase_after is VisualLatchPhase.HOME_CLOSING
    assert home_closing.home_ready_streak == 5
    assert home_closing.arm_hold
    assert home_closing.ee_target_rad == pytest.approx(0.314)
    first_home_close = compose_visual_action(
        _q(0.10),
        _q(0.42),
        home_closing,
        ee_step_limit_rad=config.ee_step_limit_rad,
    )
    assert first_home_close.action[6] == pytest.approx(0.4035)

    # The first large measured change is not settled; three subsequent stable
    # measured/command cycles are required for the terminal task event.
    latch.step(
        _evidence(index + 14, status=FrameStatus.INVALID),
        _q(0.314),
        _q(0.314),
        tcp_z_m=None,
        tcp_speed_m_s=None,
        home_ready=True,
    )
    for offset in range(15, 18):
        done = latch.step(
            _evidence(index + offset, status=FrameStatus.INVALID),
            _q(0.314),
            _q(0.314),
            tcp_z_m=None,
            tcp_speed_m_s=None,
            home_ready=True,
        )
    assert done.phase_after is VisualLatchPhase.DONE
    assert done.event == "task_done"
    assert done.arm_hold
    assert done.ee_target_rad == pytest.approx(0.314)


def test_release_box_gate_requires_safe_tcp_delta_z_window():
    latch = VisualGraspReleaseLatch(_config(box_confirm_frames=1))
    index = _confirm_grasp(latch, _prime_and_latch(latch))
    for offset in range(2):
        latch.step(
            _evidence(index + offset, held_red=True),
            _q(0.35),
            _q(0.32),
            tcp_z_m=0.225,
            tcp_speed_m_s=0,
        )
    index += 2
    for offset in range(2):
        transported = latch.step(
            _evidence(index + offset, held_red=True),
            _q(0.35),
            _q(0.32),
            tcp_z_m=0.305,
            tcp_speed_m_s=0,
        )
    assert transported.phase_after is VisualLatchPhase.TRANSPORT_CONFIRMED

    # The transport proof remains latched, but the current pose is above the
    # reviewed release-height window and must not align or release.
    unsafe = latch.step(
        _evidence(index + 2, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.345,
        tcp_speed_m_s=0,
    )
    assert unsafe.phase_after is VisualLatchPhase.TRANSPORT_CONFIRMED
    assert unsafe.box_streak == 0

    safe = latch.step(
        _evidence(index + 3, held_red=True, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.300,
        tcp_speed_m_s=0,
    )
    assert safe.phase_after is VisualLatchPhase.BOX_ALIGNED
    assert safe.release_started is False


def test_camera_loss_after_latch_fails_closed():
    config = _config(vision_loss_fault_ms=50.0)
    latch = VisualGraspReleaseLatch(config)
    index = _prime_and_latch(latch)
    last_good_eval = _evidence(index - 1, red=True).evaluated_at_ns
    stale = _evidence(
        index,
        red=True,
        status=FrameStatus.DUPLICATE,
        evaluated_at_ns=last_good_eval + 60_000_000,
    )
    fault = latch.step(
        stale, _q(0.35), _q(0.32), tcp_z_m=0.2, tcp_speed_m_s=0
    )
    assert fault.phase_after is VisualLatchPhase.FAULT_HOLD
    assert fault.fault_reason == "vision_unavailable"
    assert fault.ee_target_rad == pytest.approx(0.32)
    assert fault.arm_hold


def test_lost_contact_fails_closed_even_when_red_remains_visible():
    config = _config(held_contact_loss_fault_ms=50.0)
    latch = VisualGraspReleaseLatch(config)
    index = _confirm_grasp(latch, _prime_and_latch(latch))

    suspected_drop = latch.step(
        _evidence(index, red=True),
        _q(0.321),
        _q(0.32),
        tcp_z_m=0.20,
        tcp_speed_m_s=0,
    )
    latch.step(
        _evidence(index + 1, red=True),
        _q(0.321),
        _q(0.32),
        tcp_z_m=0.20,
        tcp_speed_m_s=0,
    )
    fault = latch.step(
        _evidence(index + 2, red=True),
        _q(0.321),
        _q(0.32),
        tcp_z_m=0.20,
        tcp_speed_m_s=0,
    )

    assert suspected_drop.phase_after is VisualLatchPhase.GRASP_CONFIRMED
    assert suspected_drop.arm_hold
    assert suspected_drop.held_contact_ok is False
    assert fault.phase_after is VisualLatchPhase.FAULT_HOLD
    assert fault.fault_reason == "held_contact_lost"
    assert fault.ee_target_rad == pytest.approx(config.close_target_rad)
    assert fault.arm_hold


def test_short_held_red_occlusion_keeps_contact_latch_but_never_advances():
    config = _config(held_red_loss_fault_ms=100.0)
    latch = VisualGraspReleaseLatch(config)
    index = _confirm_grasp(latch, _prime_and_latch(latch))

    occluded = latch.step(
        _evidence(index, held_red=False, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.35,
        tcp_speed_m_s=0,
    )
    still_occluded = latch.step(
        _evidence(index + 2, held_red=False, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.35,
        tcp_speed_m_s=0,
    )
    assert occluded.phase_after is VisualLatchPhase.GRASP_CONFIRMED
    assert still_occluded.phase_after is VisualLatchPhase.GRASP_CONFIRMED
    assert occluded.held_contact_ok is True
    assert not occluded.arm_hold
    assert not still_occluded.arm_hold
    assert not still_occluded.release_started

    prolonged = latch.step(
        _evidence(index + 4, held_red=False, blue=True),
        _q(0.35),
        _q(0.32),
        tcp_z_m=0.35,
        tcp_speed_m_s=0,
    )
    assert prolonged.phase_after is VisualLatchPhase.FAULT_HOLD
    assert prolonged.arm_hold
    assert prolonged.ee_target_rad == pytest.approx(config.close_target_rad)
    assert prolonged.fault_reason == "held_red_lost"


def test_composer_changes_only_ee_and_slews_from_command_target():
    config = _config()
    latch = VisualGraspReleaseLatch(config)
    decision = latch.step(
        _evidence(0), _q(0.39), _q(0.39), tcp_z_m=0.2, tcp_speed_m_s=0
    )
    policy = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.1]
    composed = compose_visual_action(
        policy,
        _q(0.39),
        decision,
        ee_step_limit_rad=config.ee_step_limit_rad,
    )
    assert composed.action[:6] == pytest.approx(policy[:6])
    assert composed.action[6] == pytest.approx(0.4065)
    assert composed.ee_step_limited

    with pytest.raises(ValueError, match=r"shaped \(7,\)"):
        compose_visual_action(
            policy[:6], _q(0.39), decision, ee_step_limit_rad=config.ee_step_limit_rad
        )
