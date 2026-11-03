"""Fail-closed visual grasp/release latch for the Firefly Y6.

This module deliberately has no robot or camera-driver dependencies.  The
caller supplies an already-oriented RGB D405 image and monotonically stamped
frame metadata.  The detector produces immutable evidence and the state
machine only owns the seventh (end-effector) action dimension.

The full-cycle demonstrations begin with a closed jaw.  Consequently SEEKING
does not force the jaw open from the first frame: policy EE actions pass
through until a measured or accepted-command pregrasp opening has been seen.
Only then is the open target clamped and visual closing armed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math
from numbers import Integral

import numpy as np

from .firefly_tip_servo import TipServoConfig, TipServoMode


class VisualLatchPhase(StrEnum):
    DISABLED = "disabled"
    SEEKING = "seeking"
    CLOSE_PENDING = "close_pending"
    CLOSING = "closing"
    GRASP_CONFIRMED = "grasp_confirmed"
    LIFT_CONFIRMED = "lift_confirmed"
    TRANSPORT_CONFIRMED = "transport_confirmed"
    BOX_ALIGNED = "box_aligned"
    RELEASING = "releasing"
    # RELEASED is the return-home phase: the arm may move, but EE remains open.
    RELEASED = "released"
    HOME_CLOSING = "home_closing"
    DONE = "done"
    FAULT_HOLD = "fault_hold"


class FrameStatus(StrEnum):
    FRESH_UNIQUE = "fresh_unique"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    STALE = "stale"
    FUTURE = "future"
    INVALID = "invalid"


@dataclass(frozen=True)
class VisualPhaseLatchConfig:
    """Reviewed detector and state-machine constants.

    Pixel ROIs use ``(left, top, right, bottom)`` with an exclusive right and
    bottom edge.  Normalized ROIs use the same ordering.
    """

    enabled: bool = False

    tip_servo: TipServoConfig = field(default_factory=TipServoConfig)

    d405_serial: str = "323622270296"
    d405_rotation_degrees: int = 180
    expected_width: int = 640
    expected_height: int = 480
    max_frame_age_ms: float = 100.0
    future_frame_tolerance_ms: float = 5.0

    red_hsv_ranges: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ] = (
        ((0, 100, 60), (6, 255, 255)),
        ((174, 100, 60), (179, 255, 255)),
    )
    red_center_roi_norm: tuple[float, float, float, float] = (
        0.40,
        0.49,
        0.48,
        # Episode 2 crosses the policy close-intent gate with the cube center
        # at y=0.5900--0.59045. Keep a narrow 0.60 lower edge so the latch is
        # anchored at the actual pregrasp rather than 90 frames into lift.
        0.60,
    )
    red_contour_area_fraction_min: float = 0.025
    red_contour_area_fraction_max: float = 0.045
    red_aspect_ratio_min: float = 0.55
    red_aspect_ratio_max: float = 1.80
    red_solidity_min: float = 0.72
    # The orange GR80 fingers contribute two stable HSV-red contours around
    # 0.005--0.008 of the image (audited maximum 0.013787). A second graspable
    # block is at least 0.015 in the held gate, so 0.014 ignores self while
    # preserving ambiguity for every object large enough to pass that gate.
    red_ambiguity_area_fraction_min: float = 0.014
    red_max_candidate_count: int = 1
    held_red_center_roi_norm: tuple[float, float, float, float] = (
        0.38,
        0.48,
        0.52,
        0.65,
    )
    held_red_contour_area_fraction_min: float = 0.015
    held_red_contour_area_fraction_max: float = 0.050

    blue_hsv_range: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (90, 80, 40),
        (125, 255, 255),
    )
    blue_global_fraction_min: float = 0.70
    blue_release_roi_px: tuple[int, int, int, int] = (120, 160, 400, 480)
    blue_release_roi_fraction_min: float = 0.58
    blue_largest_contour_fraction_min: float = 0.40
    blue_aspect_ratio_min: float = 0.40
    blue_aspect_ratio_max: float = 3.0
    blue_solidity_min: float = 0.40

    morphology_kernel_px: int = 5
    excluded_rois_px: tuple[tuple[int, int, int, int], ...] = ()

    pregrasp_open_ready_rad: float = 0.38
    pregrasp_open_target_rad: float = 0.41
    close_target_rad: float = 0.32
    pickup_policy_close_intent_max_rad: float = 0.35
    pickup_policy_close_intent_hold_ms: float = 500.0
    release_target_rad: float = 0.42
    release_confirm_min_rad: float = 0.39
    home_target_rad: float = 0.314
    # Rounded-up maxima of the 50 demonstrations' measured start-to-final
    # per-axis errors: [0.1849, 0.1364, 0.0362, 0.0966, 0.1270, 0.1077].
    home_arm_tolerance_rad: tuple[float, ...] = (
        0.20,
        0.15,
        0.05,
        0.11,
        0.14,
        0.12,
    )
    home_max_arm_speed_rad_s: float = 0.05
    home_policy_close_intent_max_rad: float = 0.35
    ee_step_limit_rad: float = 0.0165

    close_confirm_frames: int = 9
    grasp_confirm_frames: int = 5
    lift_confirm_frames: int = 3
    transport_confirm_frames: int = 3
    box_confirm_frames: int = 10
    release_settle_frames: int = 3
    release_confirm_cycles: int = 3
    home_ready_confirm_cycles: int = 5
    home_settle_confirm_cycles: int = 3

    close_max_arm_speed_rad_s: float = 0.15
    close_command_ready_max_rad: float = 0.335
    contact_gap_min_rad: float = 0.012
    contact_gap_max_rad: float = 0.055
    held_contact_gap_min_rad: float = 0.008
    jaw_settled_speed_max_rad_s: float = 0.05
    home_command_tolerance_rad: float = 0.005
    home_measured_tolerance_rad: float = 0.015

    lift_height_m: float = 0.020
    transport_height_m: float = 0.100
    release_tcp_delta_z_min_m: float = -0.010
    release_tcp_delta_z_max_m: float = 0.140
    release_max_tcp_speed_m_s: float = 0.030
    require_tcp_speed_for_release: bool = True

    vision_loss_fault_ms: float = 250.0
    # The held cube is partially occluded by the blue-box rim for as many as
    # 71 consecutive frames in the audited demonstrations. During that short
    # interval the independent measured contact-gap latch remains mandatory;
    # red loss still blocks every phase confirmation and all release gates.
    held_red_loss_fault_ms: float = 3000.0
    held_contact_loss_fault_ms: float = 250.0
    fk_loss_fault_ms: float = 250.0
    closing_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.tip_servo, TipServoConfig):
            raise ValueError("tip_servo must be a TipServoConfig")
        if self.tip_servo.mode is not TipServoMode.DISABLED and not self.enabled:
            raise ValueError("tip servo requires the visual phase latch to be enabled")
        if not math.isclose(
            self.tip_servo.pregrasp_open_target_rad,
            self.pregrasp_open_target_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.tip_servo.pregrasp_open_ready_rad,
            self.pregrasp_open_ready_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("tip servo and visual latch pregrasp EE targets must match")
        if (
            self.tip_servo.mode is not TipServoMode.DISABLED
            and self.tip_servo.vision_loss_timeout_ms
            > self.vision_loss_fault_ms
        ):
            raise ValueError(
                "tip servo visual-loss timeout must not exceed the latch timeout"
            )
        if not self.d405_serial:
            raise ValueError("d405_serial must not be empty")
        if self.d405_rotation_degrees != 180:
            raise ValueError("D405 frames must already be rotated exactly 180 degrees")
        if self.expected_width <= 0 or self.expected_height <= 0:
            raise ValueError("expected D405 image dimensions must be positive")
        if self.morphology_kernel_px <= 0 or self.morphology_kernel_px % 2 == 0:
            raise ValueError("morphology_kernel_px must be a positive odd integer")

        _validate_norm_roi(self.red_center_roi_norm, "red_center_roi_norm")
        _validate_norm_roi(
            self.held_red_center_roi_norm, "held_red_center_roi_norm"
        )
        _validate_pixel_roi(
            self.blue_release_roi_px,
            self.expected_width,
            self.expected_height,
            "blue_release_roi_px",
        )
        for index, roi in enumerate(self.excluded_rois_px):
            _validate_pixel_roi(
                roi,
                self.expected_width,
                self.expected_height,
                f"excluded_rois_px[{index}]",
            )
        for index, hsv_range in enumerate(self.red_hsv_ranges):
            _validate_hsv_range(hsv_range, f"red_hsv_ranges[{index}]")
        if not self.red_hsv_ranges:
            raise ValueError("at least one red HSV range is required")
        _validate_hsv_range(self.blue_hsv_range, "blue_hsv_range")

        positive_values = {
            "max_frame_age_ms": self.max_frame_age_ms,
            "ee_step_limit_rad": self.ee_step_limit_rad,
            "close_max_arm_speed_rad_s": self.close_max_arm_speed_rad_s,
            "pickup_policy_close_intent_hold_ms":
                self.pickup_policy_close_intent_hold_ms,
            "jaw_settled_speed_max_rad_s": self.jaw_settled_speed_max_rad_s,
            "lift_height_m": self.lift_height_m,
            "transport_height_m": self.transport_height_m,
            "release_max_tcp_speed_m_s": self.release_max_tcp_speed_m_s,
            "home_max_arm_speed_rad_s": self.home_max_arm_speed_rad_s,
            "vision_loss_fault_ms": self.vision_loss_fault_ms,
            "held_red_loss_fault_ms": self.held_red_loss_fault_ms,
            "held_contact_loss_fault_ms": self.held_contact_loss_fault_ms,
            "fk_loss_fault_ms": self.fk_loss_fault_ms,
            "closing_timeout_s": self.closing_timeout_s,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.future_frame_tolerance_ms)
            or self.future_frame_tolerance_ms < 0
        ):
            raise ValueError("future_frame_tolerance_ms must be finite and nonnegative")

        unit_intervals = {
            "red_contour_area_fraction_min": self.red_contour_area_fraction_min,
            "red_contour_area_fraction_max": self.red_contour_area_fraction_max,
            "red_solidity_min": self.red_solidity_min,
            "red_ambiguity_area_fraction_min": self.red_ambiguity_area_fraction_min,
            "held_red_contour_area_fraction_min": (
                self.held_red_contour_area_fraction_min
            ),
            "held_red_contour_area_fraction_max": (
                self.held_red_contour_area_fraction_max
            ),
            "blue_global_fraction_min": self.blue_global_fraction_min,
            "blue_release_roi_fraction_min": self.blue_release_roi_fraction_min,
            "blue_largest_contour_fraction_min": self.blue_largest_contour_fraction_min,
            "blue_solidity_min": self.blue_solidity_min,
        }
        for name, value in unit_intervals.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.red_contour_area_fraction_min >= self.red_contour_area_fraction_max:
            raise ValueError("red contour area minimum must be below its maximum")
        if (
            self.held_red_contour_area_fraction_min
            >= self.held_red_contour_area_fraction_max
        ):
            raise ValueError("held-red contour area minimum must be below its maximum")
        if (
            self.red_ambiguity_area_fraction_min
            >= self.held_red_contour_area_fraction_min
        ):
            raise ValueError(
                "red ambiguity threshold must remain below the held-red target minimum"
            )
        if self.red_aspect_ratio_min <= 0 or (
            self.red_aspect_ratio_min > self.red_aspect_ratio_max
        ):
            raise ValueError("invalid red aspect-ratio interval")
        if self.blue_aspect_ratio_min <= 0 or (
            self.blue_aspect_ratio_min > self.blue_aspect_ratio_max
        ):
            raise ValueError("invalid blue aspect-ratio interval")
        if self.red_max_candidate_count < 1:
            raise ValueError("red_max_candidate_count must be positive")

        count_fields = (
            "close_confirm_frames",
            "grasp_confirm_frames",
            "lift_confirm_frames",
            "transport_confirm_frames",
            "box_confirm_frames",
            "release_settle_frames",
            "release_confirm_cycles",
            "home_ready_confirm_cycles",
            "home_settle_confirm_cycles",
        )
        for name in count_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        targets = (
            self.pregrasp_open_ready_rad,
            self.pregrasp_open_target_rad,
            self.close_target_rad,
            self.pickup_policy_close_intent_max_rad,
            self.release_target_rad,
            self.release_confirm_min_rad,
            self.home_target_rad,
            self.home_policy_close_intent_max_rad,
            self.close_command_ready_max_rad,
            self.contact_gap_min_rad,
            self.contact_gap_max_rad,
            self.held_contact_gap_min_rad,
        )
        if not all(math.isfinite(value) for value in targets):
            raise ValueError("all EE targets and contact gates must be finite")
        if not (
            self.close_target_rad
            < self.pregrasp_open_ready_rad
            <= self.pregrasp_open_target_rad
        ):
            raise ValueError("close/pregrasp EE targets are inconsistent")
        if not (
            self.close_target_rad
            < self.pickup_policy_close_intent_max_rad
            < self.pregrasp_open_ready_rad
        ):
            raise ValueError(
                "pickup policy close intent must be between close and pregrasp targets"
            )
        if self.release_target_rad < self.release_confirm_min_rad:
            raise ValueError("release target must reach the release confirmation gate")
        if self.home_target_rad >= self.pregrasp_open_ready_rad:
            raise ValueError("home EE target must remain below the pregrasp-open gate")
        if not (
            self.home_target_rad
            < self.home_policy_close_intent_max_rad
            < self.pregrasp_open_ready_rad
        ):
            raise ValueError(
                "home policy close intent must be between home and pregrasp targets"
            )
        if (
            len(self.home_arm_tolerance_rad) != 6
            or not all(
                math.isfinite(float(value)) and float(value) > 0
                for value in self.home_arm_tolerance_rad
            )
        ):
            raise ValueError(
                "home_arm_tolerance_rad must contain six positive finite values"
            )
        if self.contact_gap_min_rad < 0 or (
            self.contact_gap_min_rad >= self.contact_gap_max_rad
        ):
            raise ValueError("invalid contact-gap interval")
        if not (
            0
            <= self.held_contact_gap_min_rad
            < self.contact_gap_min_rad
        ):
            raise ValueError(
                "held contact gap must be nonnegative and below the grasp-confirm gap"
            )
        if self.lift_height_m >= self.transport_height_m:
            raise ValueError("lift_height_m must be below transport_height_m")
        for name, value in {
            "home_command_tolerance_rad": self.home_command_tolerance_rad,
            "home_measured_tolerance_rad": self.home_measured_tolerance_rad,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            math.isfinite(self.release_tcp_delta_z_min_m)
            and math.isfinite(self.release_tcp_delta_z_max_m)
            and self.release_tcp_delta_z_min_m < self.release_tcp_delta_z_max_m
        ):
            raise ValueError("invalid release TCP delta-z safety window")


@dataclass(frozen=True)
class BlobEvidence:
    accepted: bool = False
    held_accepted: bool = False
    mask_area_fraction: float = 0.0
    roi_mask_fraction: float | None = None
    contour_area_fraction: float = 0.0
    centroid_norm: tuple[float, float] | None = None
    bbox_norm: tuple[float, float, float, float] | None = None
    aspect_ratio: float | None = None
    solidity: float | None = None
    candidate_count: int = 0
    rejection_reasons: tuple[str, ...] = ()
    held_rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualEvidence:
    frame_id: int
    captured_at_ns: int
    evaluated_at_ns: int
    frame_age_ms: float
    frame_status: FrameStatus
    red: BlobEvidence
    blue: BlobEvidence

    @property
    def usable(self) -> bool:
        return self.frame_status is FrameStatus.FRESH_UNIQUE


@dataclass(frozen=True)
class LatchDecision:
    phase_before: VisualLatchPhase
    phase_after: VisualLatchPhase
    event: str | None
    ee_target_rad: float | None
    arm_hold: bool
    pregrasp_open_seen: bool
    policy_close_intent: bool
    grasp_latched: bool
    release_started: bool
    fault_reason: str | None
    close_streak: int
    grasp_streak: int
    lift_streak: int
    transport_streak: int
    box_streak: int
    settle_streak: int
    release_streak: int
    home_ready_streak: int
    home_settle_streak: int
    contact_gap_rad: float | None
    held_contact_ok: bool | None
    arm_speed_rad_s: float | None
    jaw_speed_rad_s: float | None
    tcp_delta_z_m: float | None
    visual_usable: bool


@dataclass(frozen=True)
class ComposedVisualAction:
    action: tuple[float, ...]
    arm_hold: bool
    ee_step_limited: bool


def _validate_norm_roi(roi: Sequence[float], name: str) -> None:
    if len(roi) != 4 or not all(math.isfinite(float(value)) for value in roi):
        raise ValueError(f"{name} must contain four finite values")
    left, top, right, bottom = (float(value) for value in roi)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"{name} must be an ordered ROI inside [0, 1]")


def _validate_pixel_roi(
    roi: Sequence[int], width: int, height: int, name: str
) -> None:
    if len(roi) != 4 or not all(isinstance(value, Integral) for value in roi):
        raise ValueError(f"{name} must contain four integers")
    left, top, right, bottom = (int(value) for value in roi)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"{name} is outside the configured image")


def _validate_hsv_range(hsv_range, name: str) -> None:
    if len(hsv_range) != 2 or any(len(bound) != 3 for bound in hsv_range):
        raise ValueError(f"{name} must contain lower and upper HSV triplets")
    lower, upper = hsv_range
    maxima = (179, 255, 255)
    for index, maximum in enumerate(maxima):
        if not (
            isinstance(lower[index], Integral)
            and isinstance(upper[index], Integral)
            and 0 <= int(lower[index]) <= int(upper[index]) <= maximum
        ):
            raise ValueError(f"{name} has an invalid channel {index}")


class D405VisualDetector:
    """Extract calibrated red-grasp and blue-release evidence from D405 RGB."""

    def __init__(self, config: VisualPhaseLatchConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_frame_id: int | None = None
        self._last_captured_at_ns: int | None = None

    def detect(
        self,
        rgb: np.ndarray,
        *,
        frame_id: int,
        captured_at_ns: int,
        now_ns: int,
    ) -> VisualEvidence:
        status, age_ms, metadata_valid = self._classify_frame(
            frame_id=frame_id,
            captured_at_ns=captured_at_ns,
            now_ns=now_ns,
        )
        invalid_reason = self._validate_image(rgb)
        if invalid_reason is not None:
            return VisualEvidence(
                frame_id=int(frame_id) if isinstance(frame_id, Integral) else -1,
                captured_at_ns=(
                    int(captured_at_ns) if isinstance(captured_at_ns, Integral) else -1
                ),
                evaluated_at_ns=int(now_ns) if isinstance(now_ns, Integral) else -1,
                frame_age_ms=age_ms,
                frame_status=FrameStatus.INVALID,
                red=BlobEvidence(rejection_reasons=(invalid_reason,)),
                blue=BlobEvidence(rejection_reasons=(invalid_reason,)),
            )

        if metadata_valid and status in {FrameStatus.FRESH_UNIQUE, FrameStatus.STALE}:
            self._last_frame_id = int(frame_id)
            self._last_captured_at_ns = int(captured_at_ns)

        try:
            import cv2

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            red_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for lower, upper in self.config.red_hsv_ranges:
                red_mask = cv2.bitwise_or(
                    red_mask,
                    cv2.inRange(
                        hsv,
                        np.asarray(lower, dtype=np.uint8),
                        np.asarray(upper, dtype=np.uint8),
                    ),
                )
            blue_lower, blue_upper = self.config.blue_hsv_range
            blue_mask = cv2.inRange(
                hsv,
                np.asarray(blue_lower, dtype=np.uint8),
                np.asarray(blue_upper, dtype=np.uint8),
            )
            kernel = np.ones(
                (self.config.morphology_kernel_px, self.config.morphology_kernel_px),
                dtype=np.uint8,
            )
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            for left, top, right, bottom in self.config.excluded_rois_px:
                red_mask[top:bottom, left:right] = 0
                blue_mask[top:bottom, left:right] = 0
            red = self._red_evidence(red_mask, cv2)
            blue = self._blue_evidence(blue_mask, cv2)
        except Exception as exc:
            reason = f"detector_error:{type(exc).__name__}"
            return VisualEvidence(
                frame_id=int(frame_id),
                captured_at_ns=int(captured_at_ns),
                evaluated_at_ns=int(now_ns),
                frame_age_ms=age_ms,
                frame_status=FrameStatus.INVALID,
                red=BlobEvidence(rejection_reasons=(reason,)),
                blue=BlobEvidence(rejection_reasons=(reason,)),
            )

        return VisualEvidence(
            frame_id=int(frame_id),
            captured_at_ns=int(captured_at_ns),
            evaluated_at_ns=int(now_ns),
            frame_age_ms=age_ms,
            frame_status=status,
            red=red,
            blue=blue,
        )

    def _classify_frame(
        self, *, frame_id: int, captured_at_ns: int, now_ns: int
    ) -> tuple[FrameStatus, float, bool]:
        metadata = (frame_id, captured_at_ns, now_ns)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in metadata
        ):
            return FrameStatus.INVALID, math.nan, False
        frame_id = int(frame_id)
        captured_at_ns = int(captured_at_ns)
        now_ns = int(now_ns)
        if frame_id < 0 or captured_at_ns <= 0 or now_ns <= 0:
            return FrameStatus.INVALID, math.nan, False
        age_ms = (now_ns - captured_at_ns) / 1_000_000.0
        if age_ms < -self.config.future_frame_tolerance_ms:
            return FrameStatus.FUTURE, age_ms, True
        if self._last_captured_at_ns is not None:
            if captured_at_ns < self._last_captured_at_ns:
                return FrameStatus.OUT_OF_ORDER, age_ms, True
            if (
                captured_at_ns == self._last_captured_at_ns
                or frame_id == self._last_frame_id
            ):
                return FrameStatus.DUPLICATE, age_ms, True
        if age_ms > self.config.max_frame_age_ms:
            return FrameStatus.STALE, age_ms, True
        return FrameStatus.FRESH_UNIQUE, age_ms, True

    def _validate_image(self, rgb: np.ndarray) -> str | None:
        if not isinstance(rgb, np.ndarray):
            return "image_not_ndarray"
        expected = (
            self.config.expected_height,
            self.config.expected_width,
            3,
        )
        if rgb.shape != expected:
            return f"image_shape:{rgb.shape!r}"
        if rgb.dtype != np.uint8:
            return f"image_dtype:{rgb.dtype}"
        return None

    def _contour_metrics(self, contour, cv2) -> dict[str, object]:
        image_area = float(self.config.expected_width * self.config.expected_height)
        contour_area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            centroid = (
                float(moments["m10"] / moments["m00"] / self.config.expected_width),
                float(moments["m01"] / moments["m00"] / self.config.expected_height),
            )
        else:
            centroid = None
        x, y, width, height = cv2.boundingRect(contour)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        return {
            "contour_area_fraction": contour_area / image_area,
            "centroid_norm": centroid,
            "bbox_norm": (
                x / self.config.expected_width,
                y / self.config.expected_height,
                width / self.config.expected_width,
                height / self.config.expected_height,
            ),
            "aspect_ratio": width / height if height else math.inf,
            "solidity": contour_area / hull_area if hull_area > 0 else 0.0,
        }

    def _red_evidence(self, mask: np.ndarray, cv2) -> BlobEvidence:
        image_area = float(mask.size)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        candidate_count = sum(
            cv2.contourArea(contour) / image_area
            >= self.config.red_ambiguity_area_fraction_min
            for contour in contours
        )
        mask_fraction = float(np.count_nonzero(mask) / mask.size)
        if not contours:
            return BlobEvidence(
                mask_area_fraction=mask_fraction,
                rejection_reasons=("red_not_detected",),
            )
        metrics = self._contour_metrics(max(contours, key=cv2.contourArea), cv2)
        close_reasons: list[str] = []
        held_reasons: list[str] = []
        centroid = metrics["centroid_norm"]
        left, top, right, bottom = self.config.red_center_roi_norm
        if centroid is None or not (
            left <= centroid[0] <= right and top <= centroid[1] <= bottom
        ):
            close_reasons.append("red_centroid_outside_grasp_roi")
        held_left, held_top, held_right, held_bottom = (
            self.config.held_red_center_roi_norm
        )
        if centroid is None or not (
            held_left <= centroid[0] <= held_right
            and held_top <= centroid[1] <= held_bottom
        ):
            held_reasons.append("red_centroid_outside_held_roi")
        area = float(metrics["contour_area_fraction"])
        if not (
            self.config.red_contour_area_fraction_min
            <= area
            <= self.config.red_contour_area_fraction_max
        ):
            close_reasons.append("red_contour_area_outside_gate")
        if not (
            self.config.held_red_contour_area_fraction_min
            <= area
            <= self.config.held_red_contour_area_fraction_max
        ):
            held_reasons.append("red_contour_area_outside_held_gate")
        aspect = float(metrics["aspect_ratio"])
        if not self.config.red_aspect_ratio_min <= aspect <= self.config.red_aspect_ratio_max:
            close_reasons.append("red_aspect_ratio_outside_gate")
            held_reasons.append("red_aspect_ratio_outside_gate")
        solidity = float(metrics["solidity"])
        if solidity < self.config.red_solidity_min:
            close_reasons.append("red_solidity_below_gate")
            held_reasons.append("red_solidity_below_gate")
        if candidate_count > self.config.red_max_candidate_count:
            close_reasons.append("red_candidates_ambiguous")
            held_reasons.append("red_candidates_ambiguous")
        return BlobEvidence(
            accepted=not close_reasons,
            held_accepted=not held_reasons,
            mask_area_fraction=mask_fraction,
            contour_area_fraction=area,
            centroid_norm=centroid,
            bbox_norm=metrics["bbox_norm"],
            aspect_ratio=aspect,
            solidity=solidity,
            candidate_count=candidate_count,
            rejection_reasons=tuple(close_reasons),
            held_rejection_reasons=tuple(held_reasons),
        )

    def _blue_evidence(self, mask: np.ndarray, cv2) -> BlobEvidence:
        image_area = float(mask.size)
        global_fraction = float(np.count_nonzero(mask) / mask.size)
        left, top, right, bottom = self.config.blue_release_roi_px
        roi = mask[top:bottom, left:right]
        roi_fraction = float(np.count_nonzero(roi) / roi.size)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        if not contours:
            return BlobEvidence(
                mask_area_fraction=global_fraction,
                roi_mask_fraction=roi_fraction,
                rejection_reasons=("blue_not_detected",),
            )
        metrics = self._contour_metrics(max(contours, key=cv2.contourArea), cv2)
        reasons: list[str] = []
        if global_fraction < self.config.blue_global_fraction_min:
            reasons.append("blue_global_fraction_below_gate")
        if roi_fraction < self.config.blue_release_roi_fraction_min:
            reasons.append("blue_release_roi_fraction_below_gate")
        area = float(metrics["contour_area_fraction"])
        if area < self.config.blue_largest_contour_fraction_min:
            reasons.append("blue_contour_area_below_gate")
        aspect = float(metrics["aspect_ratio"])
        if not self.config.blue_aspect_ratio_min <= aspect <= self.config.blue_aspect_ratio_max:
            reasons.append("blue_aspect_ratio_outside_gate")
        solidity = float(metrics["solidity"])
        if solidity < self.config.blue_solidity_min:
            reasons.append("blue_solidity_below_gate")
        return BlobEvidence(
            accepted=not reasons,
            mask_area_fraction=global_fraction,
            roi_mask_fraction=roi_fraction,
            contour_area_fraction=area,
            centroid_norm=metrics["centroid_norm"],
            bbox_norm=metrics["bbox_norm"],
            aspect_ratio=aspect,
            solidity=solidity,
            candidate_count=len(contours),
            rejection_reasons=tuple(reasons),
        )


_CLOSED_PRE_RELEASE_PHASES = {
    VisualLatchPhase.CLOSING,
    VisualLatchPhase.GRASP_CONFIRMED,
    VisualLatchPhase.LIFT_CONFIRMED,
    VisualLatchPhase.TRANSPORT_CONFIRMED,
    VisualLatchPhase.BOX_ALIGNED,
}
_HELD_RED_PHASES = {
    VisualLatchPhase.GRASP_CONFIRMED,
    VisualLatchPhase.LIFT_CONFIRMED,
    VisualLatchPhase.TRANSPORT_CONFIRMED,
    VisualLatchPhase.BOX_ALIGNED,
}
_VISUAL_REQUIRED_PHASES = {
    VisualLatchPhase.SEEKING,
    VisualLatchPhase.CLOSE_PENDING,
    *_CLOSED_PRE_RELEASE_PHASES,
}


class VisualGraspReleaseLatch:
    """Monotonic visual phase latch which never releases on missing evidence."""

    def __init__(self, config: VisualPhaseLatchConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = (
            VisualLatchPhase.SEEKING
            if self.config.enabled
            else VisualLatchPhase.DISABLED
        )
        self.pregrasp_open_seen = False
        self.policy_close_intent = False
        self.grasp_latched = False
        self.release_started = False
        self.fault_reason: str | None = None
        self.close_streak = 0
        self.grasp_streak = 0
        self.lift_streak = 0
        self.transport_streak = 0
        self.box_streak = 0
        self.settle_streak = 0
        self.release_streak = 0
        self.home_ready_streak = 0
        self.home_settle_streak = 0
        self._last_q: np.ndarray | None = None
        self._last_sample_ns: int | None = None
        self._last_usable_visual_ns: int | None = None
        self._visual_watch_started_ns: int | None = None
        self._red_missing_since_ns: int | None = None
        self._contact_missing_since_ns: int | None = None
        self._held_contact_ok: bool | None = None
        self._fk_missing_since_ns: int | None = None
        self._latch_started_ns: int | None = None
        self._grasp_z_m: float | None = None

    def step(
        self,
        evidence: VisualEvidence,
        measured: Sequence[float],
        command_target: Sequence[float],
        *,
        tcp_z_m: float | None,
        tcp_speed_m_s: float | None,
        home_ready: bool = False,
        policy_close_intent: bool = False,
        external_fault_reason: str | None = None,
    ) -> LatchDecision:
        if not isinstance(evidence, VisualEvidence):
            raise TypeError("evidence must be VisualEvidence")
        if not isinstance(home_ready, (bool, np.bool_)):
            raise TypeError("home_ready must be a boolean")
        if not isinstance(policy_close_intent, (bool, np.bool_)):
            raise TypeError("policy_close_intent must be a boolean")
        self.policy_close_intent = bool(policy_close_intent)
        q = _finite_vector(measured, "measured")
        command = _finite_vector(command_target, "command_target")
        now_ns = evidence.evaluated_at_ns
        if not isinstance(now_ns, Integral) or now_ns <= 0:
            raise ValueError("evidence.evaluated_at_ns must be positive")
        now_ns = int(now_ns)
        if self._visual_watch_started_ns is None:
            self._visual_watch_started_ns = now_ns
        phase_before = self.phase
        event: str | None = None

        arm_speed, jaw_speed = self._sample_speeds(q, now_ns)
        tcp_z = _optional_finite(tcp_z_m)
        tcp_speed = _optional_finite(tcp_speed_m_s)
        if tcp_speed is not None:
            tcp_speed = abs(tcp_speed)
        if evidence.usable:
            self._last_usable_visual_ns = now_ns

        if self.phase is VisualLatchPhase.DISABLED:
            return self._finish(
                phase_before, event, q, command, now_ns, arm_speed, jaw_speed, tcp_z
            )

        if (
            float(q[6]) >= self.config.pregrasp_open_ready_rad
            or float(command[6]) >= self.config.pregrasp_open_ready_rad
        ):
            self.pregrasp_open_seen = True

        if external_fault_reason:
            event = self._enter_fault(f"external:{external_fault_reason}")

        if self.phase in _VISUAL_REQUIRED_PHASES:
            visual_reference_ns = (
                self._last_usable_visual_ns
                if self._last_usable_visual_ns is not None
                else self._visual_watch_started_ns
            )
            if (
                visual_reference_ns is not None
                and (now_ns - visual_reference_ns) / 1_000_000.0
                > self.config.vision_loss_fault_ms
            ):
                event = self._enter_fault("vision_unavailable")

        if self.phase in _HELD_RED_PHASES and evidence.usable:
            if evidence.red.held_accepted:
                self._red_missing_since_ns = None
            elif self._red_missing_since_ns is None:
                self._red_missing_since_ns = now_ns
            elif (
                (now_ns - self._red_missing_since_ns) / 1_000_000.0
                > self.config.held_red_loss_fault_ms
            ):
                event = self._enter_fault("held_red_lost")

        if self.phase in _HELD_RED_PHASES:
            held_gap = float(q[6] - command[6])
            self._held_contact_ok = bool(
                self.config.held_contact_gap_min_rad
                <= held_gap
                <= self.config.contact_gap_max_rad
            )
            if self._held_contact_ok:
                self._contact_missing_since_ns = None
            elif self._contact_missing_since_ns is None:
                self._contact_missing_since_ns = now_ns
            elif (
                (now_ns - self._contact_missing_since_ns) / 1_000_000.0
                > self.config.held_contact_loss_fault_ms
            ):
                event = self._enter_fault("held_contact_lost")

        if self.phase in _HELD_RED_PHASES:
            if tcp_z is not None:
                self._fk_missing_since_ns = None
            elif self._fk_missing_since_ns is None:
                self._fk_missing_since_ns = now_ns
            elif (
                (now_ns - self._fk_missing_since_ns) / 1_000_000.0
                > self.config.fk_loss_fault_ms
            ):
                event = self._enter_fault("fk_unavailable")

        if self.phase is not VisualLatchPhase.FAULT_HOLD:
            duplicate_frame = evidence.frame_status is FrameStatus.DUPLICATE
            if self.phase is VisualLatchPhase.SEEKING:
                if self._close_candidate(
                    evidence, q, command, arm_speed, self.policy_close_intent
                ):
                    self.close_streak = 1
                    if self.close_streak >= self.config.close_confirm_frames:
                        event = self._latch_close(now_ns, tcp_z)
                    else:
                        self.phase = VisualLatchPhase.CLOSE_PENDING
                        event = "close_candidate_started"
                elif not duplicate_frame:
                    self.close_streak = 0

            elif self.phase is VisualLatchPhase.CLOSE_PENDING:
                if self._close_candidate(
                    evidence, q, command, arm_speed, self.policy_close_intent
                ):
                    self.close_streak += 1
                    if self.close_streak >= self.config.close_confirm_frames:
                        event = self._latch_close(now_ns, tcp_z)
                elif not duplicate_frame:
                    self.close_streak = 0
                    self.phase = VisualLatchPhase.SEEKING
                    event = "close_candidate_cancelled"

            elif self.phase is VisualLatchPhase.CLOSING:
                contact_gap = float(q[6] - command[6])
                contact_candidate = bool(
                    evidence.usable
                    and evidence.red.accepted
                    and command[6] <= self.config.close_command_ready_max_rad
                    and self.config.contact_gap_min_rad
                    <= contact_gap
                    <= self.config.contact_gap_max_rad
                    and jaw_speed is not None
                    and jaw_speed <= self.config.jaw_settled_speed_max_rad_s
                )
                if contact_candidate:
                    self.grasp_streak += 1
                elif not duplicate_frame:
                    self.grasp_streak = 0
                if self.grasp_streak >= self.config.grasp_confirm_frames:
                    self.phase = VisualLatchPhase.GRASP_CONFIRMED
                    if self._grasp_z_m is None:
                        self._grasp_z_m = tcp_z
                    event = "grasp_confirmed"
                elif (
                    self._latch_started_ns is not None
                    and (now_ns - self._latch_started_ns) / 1_000_000_000.0
                    > self.config.closing_timeout_s
                ):
                    event = self._enter_fault("grasp_confirmation_timeout")

            elif self.phase is VisualLatchPhase.GRASP_CONFIRMED:
                lift_candidate = bool(
                    evidence.usable
                    and evidence.red.held_accepted
                    and self._held_contact_ok is True
                    and self._tcp_delta(tcp_z) is not None
                    and self._tcp_delta(tcp_z) >= self.config.lift_height_m
                )
                if lift_candidate:
                    self.lift_streak += 1
                elif not duplicate_frame:
                    self.lift_streak = 0
                if self.lift_streak >= self.config.lift_confirm_frames:
                    self.phase = VisualLatchPhase.LIFT_CONFIRMED
                    event = "lift_confirmed"

            elif self.phase is VisualLatchPhase.LIFT_CONFIRMED:
                transport_candidate = bool(
                    evidence.usable
                    and evidence.red.held_accepted
                    and self._held_contact_ok is True
                    and self._tcp_delta(tcp_z) is not None
                    and self._tcp_delta(tcp_z) >= self.config.transport_height_m
                )
                if transport_candidate:
                    self.transport_streak += 1
                elif not duplicate_frame:
                    self.transport_streak = 0
                if self.transport_streak >= self.config.transport_confirm_frames:
                    self.phase = VisualLatchPhase.TRANSPORT_CONFIRMED
                    event = "transport_confirmed"

            elif self.phase is VisualLatchPhase.TRANSPORT_CONFIRMED:
                box_candidate = self._box_visual_candidate(evidence, tcp_z)
                if box_candidate:
                    self.box_streak += 1
                elif not duplicate_frame:
                    self.box_streak = 0
                if self.box_streak >= self.config.box_confirm_frames:
                    self.phase = VisualLatchPhase.BOX_ALIGNED
                    self.settle_streak = 0
                    event = "box_aligned"

            elif self.phase is VisualLatchPhase.BOX_ALIGNED:
                settle_candidate = self._release_settle_candidate(
                    evidence, tcp_z, tcp_speed
                )
                if settle_candidate:
                    self.settle_streak += 1
                elif not duplicate_frame:
                    self.settle_streak = 0
                if self.settle_streak >= self.config.release_settle_frames:
                    self.phase = VisualLatchPhase.RELEASING
                    self.release_started = True
                    event = "release_unlatched"

            elif self.phase is VisualLatchPhase.RELEASING:
                self.release_streak = (
                    self.release_streak + 1
                    if q[6] >= self.config.release_confirm_min_rad
                    else 0
                )
                if self.release_streak >= self.config.release_confirm_cycles:
                    self.phase = VisualLatchPhase.RELEASED
                    self.home_ready_streak = 0
                    event = "release_confirmed"

            elif self.phase is VisualLatchPhase.RELEASED:
                # The external home gate owns arm geometry and low-speed
                # debounce. Requiring it for several control cycles here makes
                # one noisy home observation insufficient to close the empty jaw.
                self.home_ready_streak = (
                    self.home_ready_streak + 1 if bool(home_ready) else 0
                )
                if (
                    self.home_ready_streak
                    >= self.config.home_ready_confirm_cycles
                ):
                    self.phase = VisualLatchPhase.HOME_CLOSING
                    self.home_settle_streak = 0
                    event = "home_closing_started"

            elif self.phase is VisualLatchPhase.HOME_CLOSING:
                home_settled = bool(
                    abs(float(command[6]) - self.config.home_target_rad)
                    <= self.config.home_command_tolerance_rad
                    and abs(float(q[6]) - self.config.home_target_rad)
                    <= self.config.home_measured_tolerance_rad
                    and jaw_speed is not None
                    and jaw_speed <= self.config.jaw_settled_speed_max_rad_s
                )
                self.home_settle_streak = (
                    self.home_settle_streak + 1 if home_settled else 0
                )
                if (
                    self.home_settle_streak
                    >= self.config.home_settle_confirm_cycles
                ):
                    self.phase = VisualLatchPhase.DONE
                    event = "task_done"

            # DONE is terminal until reset. Both arm and empty jaw remain held.

        return self._finish(
            phase_before, event, q, command, now_ns, arm_speed, jaw_speed, tcp_z
        )

    def _close_candidate(
        self,
        evidence: VisualEvidence,
        q: np.ndarray,
        command: np.ndarray,
        arm_speed: float | None,
        policy_close_intent: bool,
    ) -> bool:
        jaw_is_open = bool(
            q[6] >= self.config.pregrasp_open_ready_rad
            or command[6] >= self.config.pregrasp_open_ready_rad
        )
        return bool(
            self.pregrasp_open_seen
            and policy_close_intent
            and jaw_is_open
            and evidence.usable
            and evidence.red.accepted
            and arm_speed is not None
            and arm_speed <= self.config.close_max_arm_speed_rad_s
        )

    def _box_visual_candidate(
        self, evidence: VisualEvidence, tcp_z: float | None
    ) -> bool:
        delta_z = self._tcp_delta(tcp_z)
        return bool(
            evidence.usable
            and evidence.red.held_accepted
            and self._held_contact_ok is True
            and evidence.blue.accepted
            and delta_z is not None
            and self.config.release_tcp_delta_z_min_m
            <= delta_z
            <= self.config.release_tcp_delta_z_max_m
        )

    def _release_settle_candidate(
        self,
        evidence: VisualEvidence,
        tcp_z: float | None,
        tcp_speed: float | None,
    ) -> bool:
        speed_ok = (
            tcp_speed is not None
            and tcp_speed <= self.config.release_max_tcp_speed_m_s
        )
        if not self.config.require_tcp_speed_for_release and tcp_speed is None:
            speed_ok = True
        return self._box_visual_candidate(evidence, tcp_z) and speed_ok

    def _latch_close(self, now_ns: int, tcp_z: float | None) -> str:
        self.phase = VisualLatchPhase.CLOSING
        self.grasp_latched = True
        self._latch_started_ns = now_ns
        self._grasp_z_m = tcp_z
        return "grasp_latched"

    def _enter_fault(self, reason: str) -> str:
        if self.phase is not VisualLatchPhase.FAULT_HOLD:
            self.phase = VisualLatchPhase.FAULT_HOLD
            self.fault_reason = reason
            return "fault_hold"
        return "fault_hold"

    def _sample_speeds(
        self, q: np.ndarray, now_ns: int
    ) -> tuple[float | None, float | None]:
        if self._last_q is None or self._last_sample_ns is None:
            return None, None
        elapsed_s = (now_ns - self._last_sample_ns) / 1_000_000_000.0
        if elapsed_s <= 0:
            return None, None
        delta = np.abs(q - self._last_q) / elapsed_s
        return float(np.max(delta[:6])), float(delta[6])

    def _tcp_delta(self, tcp_z: float | None) -> float | None:
        if tcp_z is None or self._grasp_z_m is None:
            return None
        return float(tcp_z - self._grasp_z_m)

    def _finish(
        self,
        phase_before: VisualLatchPhase,
        event: str | None,
        q: np.ndarray,
        command: np.ndarray,
        now_ns: int,
        arm_speed: float | None,
        jaw_speed: float | None,
        tcp_z: float | None,
    ) -> LatchDecision:
        contact_gap = float(q[6] - command[6])
        target: float | None
        if self.phase is VisualLatchPhase.DISABLED:
            target = None
        elif self.phase in {VisualLatchPhase.SEEKING, VisualLatchPhase.CLOSE_PENDING}:
            target = (
                self.config.pregrasp_open_target_rad
                if self.pregrasp_open_seen
                else None
            )
        elif self.phase in _CLOSED_PRE_RELEASE_PHASES:
            target = self.config.close_target_rad
        elif self.phase in {VisualLatchPhase.RELEASING, VisualLatchPhase.RELEASED}:
            target = self.config.release_target_rad
        elif self.phase in {VisualLatchPhase.HOME_CLOSING, VisualLatchPhase.DONE}:
            target = self.config.home_target_rad
        else:
            target = (
                self.config.release_target_rad
                if self.release_started
                else (
                    self.config.close_target_rad
                    if self.grasp_latched
                    else float(command[6])
                )
            )
        arm_hold = self.phase in {
            VisualLatchPhase.CLOSING,
            VisualLatchPhase.BOX_ALIGNED,
            VisualLatchPhase.RELEASING,
            VisualLatchPhase.HOME_CLOSING,
            VisualLatchPhase.DONE,
            VisualLatchPhase.FAULT_HOLD,
        }
        if self.phase in _HELD_RED_PHASES and (
            self._contact_missing_since_ns is not None
            or self._fk_missing_since_ns is not None
        ):
            arm_hold = True
        decision = LatchDecision(
            phase_before=phase_before,
            phase_after=self.phase,
            event=event,
            ee_target_rad=target,
            arm_hold=arm_hold,
            pregrasp_open_seen=self.pregrasp_open_seen,
            policy_close_intent=self.policy_close_intent,
            grasp_latched=self.grasp_latched,
            release_started=self.release_started,
            fault_reason=self.fault_reason,
            close_streak=self.close_streak,
            grasp_streak=self.grasp_streak,
            lift_streak=self.lift_streak,
            transport_streak=self.transport_streak,
            box_streak=self.box_streak,
            settle_streak=self.settle_streak,
            release_streak=self.release_streak,
            home_ready_streak=self.home_ready_streak,
            home_settle_streak=self.home_settle_streak,
            contact_gap_rad=contact_gap,
            held_contact_ok=self._held_contact_ok,
            arm_speed_rad_s=arm_speed,
            jaw_speed_rad_s=jaw_speed,
            tcp_delta_z_m=self._tcp_delta(tcp_z),
            visual_usable=(self._last_usable_visual_ns == now_ns),
        )
        self._last_q = q.copy()
        self._last_sample_ns = now_ns
        return decision


def compose_visual_action(
    policy_action: Sequence[float],
    command_target: Sequence[float],
    decision: LatchDecision,
    *,
    ee_step_limit_rad: float,
) -> ComposedVisualAction:
    """Apply the visual EE target without ever modifying policy arm axes."""

    policy = _finite_vector(policy_action, "policy_action")
    reference = _finite_vector(command_target, "command_target")
    if not math.isfinite(ee_step_limit_rad) or ee_step_limit_rad <= 0:
        raise ValueError("ee_step_limit_rad must be finite and positive")
    action = policy.copy()
    limited = False
    if decision.ee_target_rad is not None:
        desired = float(decision.ee_target_rad)
        if not math.isfinite(desired):
            raise ValueError("decision EE target must be finite")
        delta = desired - float(reference[6])
        limited_delta = float(np.clip(delta, -ee_step_limit_rad, ee_step_limit_rad))
        action[6] = float(reference[6] + limited_delta)
        limited = not math.isclose(delta, limited_delta, rel_tol=0.0, abs_tol=1e-12)
    return ComposedVisualAction(
        action=tuple(float(value) for value in action),
        arm_hold=decision.arm_hold,
        ee_step_limited=limited,
    )


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.shape != (7,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector shaped (7,)")
    return vector


def _optional_finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None
