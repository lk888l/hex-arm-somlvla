#!/usr/bin/env python3
"""Read-only, fail-closed Firefly scene/wrist camera alignment check.

This tool opens only the two RGB RealSense cameras named by the reviewed local
deployment YAML.  It never constructs a LeRobot robot, imports the Hex/Zenoh
stack, opens CAN, or publishes an action.  The wrist camera's configured
``rotation: 180`` is passed to the camera driver exactly once; frames are never
rotated again in this module.

After camera-driver warmup and an additional bounded settling interval, the
tool computes a rolling median of the largest red/blue HSV blobs and compares
it with the reviewed v1 HOLD images.  A missing blob, incomplete window, camera
contract mismatch, or tolerance failure produces ``alignment_ok=false`` and a
non-zero exit status.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/deploy_firefly_y6_hold_fullcycle_v4_100k.local.yaml"
)

EXPECTED_CAMERA_NAMES = frozenset({"scene", "wrist"})
EXPECTED_CAMERA_SERIALS = {
    "scene": "030522071867",
    "wrist": "323622270296",
}
EXPECTED_FPS = 30
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
EXPECTED_ROTATIONS = {"scene": 0, "wrist": 180}

RED_HSV_RANGES = (
    ((0, 100, 60), (6, 255, 255)),
    ((174, 100, 60), (179, 255, 255)),
)
BLUE_HSV_RANGES = (((90, 80, 40), (125, 255, 255)),)
MORPHOLOGY_KERNEL_PX = 5

# These constants and the mask/contour operations below intentionally match
# VisualPhaseLatchConfig/D405VisualDetector.  Importing that module through the
# robot plugin package would execute robot/Zenoh registration imports, so this
# camera-only process keeps the small, immutable vision primitive local.

REFERENCE_PROVENANCE = {
    "source": "tip_probe_v1 HOLD, median of settled zero-based frames 15..29",
    "reference_window_frames_zero_based": [15, 29],
    "scene_video": (
        "outputs/firefly_deployment_hold_fullcycle_v4_100k_tip_probe_v1/"
        "hold_scene.mp4"
    ),
    "scene_video_sha256": (
        "55c7e0f2a729bc4a5681b9a7e382c4411ad17906cf46664c9603824a6e196e85"
    ),
    "wrist_video": (
        "outputs/firefly_deployment_hold_fullcycle_v4_100k_tip_probe_v1/"
        "hold_wrist.mp4"
    ),
    "wrist_video_sha256": (
        "1f37d6f60688d91037a84f1167b01bc7854d5ca34e4f7cba58308bfe0aa84392"
    ),
    "camera_serials": EXPECTED_CAMERA_SERIALS,
    "camera_rotations_degrees": EXPECTED_ROTATIONS,
    "color_mode": "rgb",
    "frame_shape_hwc": [EXPECTED_HEIGHT, EXPECTED_WIDTH, 3],
}


class AlignmentError(RuntimeError):
    """A camera contract or alignment invariant failed closed."""


class CameraLike(Protocol):
    def connect(self) -> None: ...

    def async_read(self, timeout_ms: float = 200) -> np.ndarray: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class CameraSpec:
    name: str
    serial: str
    fps: int
    width: int
    height: int
    rotation_degrees: int
    warmup_s: int


@dataclass(frozen=True)
class BlobMeasurement:
    centroid_px: tuple[float, float]
    area_fraction: float
    bbox_px: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameMeasurements:
    scene_red: BlobMeasurement | None
    scene_blue: BlobMeasurement | None
    wrist_red: BlobMeasurement | None


@dataclass(frozen=True)
class BlobReference:
    centroid_px: tuple[float, float]
    area_fraction: float
    bbox_px: tuple[float, float, float, float]
    centroid_tolerance_px: tuple[float, float]
    area_tolerance_fraction: float
    bbox_tolerance_px: tuple[float, float, float, float]


V1_HOLD_REFERENCES: dict[str, BlobReference] = {
    "scene_red": BlobReference(
        centroid_px=(493.4897038, 249.4880952),
        area_fraction=0.0066455,
        bbox_px=(466.0, 222.0, 56.0, 56.0),
        centroid_tolerance_px=(1.0, 1.0),
        area_tolerance_fraction=0.00020,
        bbox_tolerance_px=(2.0, 2.0, 2.0, 2.0),
    ),
    "scene_blue": BlobReference(
        centroid_px=(180.9638430, 260.9303444),
        area_fraction=0.38222819,
        bbox_px=(0.0, 40.0, 354.0, 440.0),
        centroid_tolerance_px=(0.5, 0.5),
        area_tolerance_fraction=0.005,
        bbox_tolerance_px=(1.0, 2.0, 2.0, 2.0),
    ),
    "wrist_red": BlobReference(
        centroid_px=(106.2341339, 334.8415128),
        area_fraction=0.0195524,
        bbox_px=(63.0, 288.0, 91.0, 93.0),
        centroid_tolerance_px=(1.5, 1.5),
        area_tolerance_fraction=0.00060,
        bbox_tolerance_px=(3.0, 3.0, 3.0, 3.0),
    ),
}


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlignmentError(f"{label} must be a mapping")
    return value


def _required_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlignmentError(f"{label} must be an integer")
    return value


def load_camera_specs(path: Path) -> dict[str, CameraSpec]:
    """Load only ``robot.cameras`` without constructing deployment/robot config."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AlignmentError(f"cannot read camera YAML {path}: {exc}") from exc
    root = _required_mapping(raw, "configuration")
    robot = _required_mapping(root.get("robot"), "robot")
    cameras = _required_mapping(robot.get("cameras"), "robot.cameras")
    if frozenset(cameras) != EXPECTED_CAMERA_NAMES:
        raise AlignmentError(
            "robot.cameras must contain exactly scene and wrist; "
            f"actual={sorted(str(name) for name in cameras)}"
        )

    specs: dict[str, CameraSpec] = {}
    for name in sorted(EXPECTED_CAMERA_NAMES):
        camera = _required_mapping(cameras.get(name), f"robot.cameras.{name}")
        if camera.get("type") != "intelrealsense":
            raise AlignmentError(f"{name} must use type=intelrealsense")
        serial = camera.get("serial_number_or_name")
        if serial != EXPECTED_CAMERA_SERIALS[name]:
            raise AlignmentError(
                f"{name} serial must remain {EXPECTED_CAMERA_SERIALS[name]!r}; "
                f"actual={serial!r}"
            )
        fps = _required_int(camera.get("fps"), f"{name}.fps")
        width = _required_int(camera.get("width"), f"{name}.width")
        height = _required_int(camera.get("height"), f"{name}.height")
        rotation = _required_int(camera.get("rotation", 0), f"{name}.rotation")
        warmup_s = _required_int(camera.get("warmup_s", 1), f"{name}.warmup_s")
        if (fps, width, height) != (
            EXPECTED_FPS,
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
        ):
            raise AlignmentError(
                f"{name} must remain 640x480@30; actual={width}x{height}@{fps}"
            )
        if rotation != EXPECTED_ROTATIONS[name]:
            raise AlignmentError(
                f"{name} rotation must be {EXPECTED_ROTATIONS[name]} degrees; "
                f"actual={rotation}. Do not rotate D405 a second time."
            )
        if camera.get("use_rgb", True) is not True:
            raise AlignmentError(f"{name}.use_rgb must be true")
        if camera.get("use_depth", False) is not False:
            raise AlignmentError(f"{name}.use_depth must be false")
        if camera.get("color_mode", "rgb") != "rgb":
            raise AlignmentError(f"{name}.color_mode must be rgb")
        if warmup_s < 1 or warmup_s > 10:
            raise AlignmentError(f"{name}.warmup_s must be inside [1, 10]")
        specs[name] = CameraSpec(
            name=name,
            serial=str(serial),
            fps=fps,
            width=width,
            height=height,
            rotation_degrees=rotation,
            warmup_s=warmup_s,
        )
    return specs


_FORBIDDEN_RUNTIME_IMPORT_PREFIXES = (
    "can",
    "canopen",
    "zenoh",
    "lerobot.robots",
    "lerobot_robot_hexbase",
)


def _matches_module_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def create_realsense_cameras(specs: Mapping[str, CameraSpec]) -> dict[str, CameraLike]:
    """Construct only LeRobot camera drivers; never import a robot plugin."""

    if frozenset(specs) != EXPECTED_CAMERA_NAMES:
        raise AlignmentError("camera specs must contain exactly scene and wrist")
    before = frozenset(sys.modules)
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    newly_loaded = frozenset(sys.modules).difference(before)
    forbidden = sorted(
        module
        for module in newly_loaded
        if any(
            _matches_module_prefix(module, prefix)
            for prefix in _FORBIDDEN_RUNTIME_IMPORT_PREFIXES
        )
    )
    if forbidden:
        raise AlignmentError(
            "camera-only import unexpectedly loaded a robot/transport module: "
            + ", ".join(forbidden)
        )

    cameras: dict[str, CameraLike] = {}
    for name in sorted(EXPECTED_CAMERA_NAMES):
        spec = specs[name]
        config = RealSenseCameraConfig(
            serial_number_or_name=spec.serial,
            fps=spec.fps,
            width=spec.width,
            height=spec.height,
            color_mode=ColorMode.RGB,
            use_rgb=True,
            use_depth=False,
            rotation=Cv2Rotation(spec.rotation_degrees),
            warmup_s=spec.warmup_s,
        )
        cameras[name] = RealSenseCamera(config)
    return cameras


def _validate_rgb(rgb: np.ndarray, label: str) -> None:
    if not isinstance(rgb, np.ndarray):
        raise AlignmentError(f"{label} frame is not an ndarray")
    if rgb.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise AlignmentError(f"{label} frame shape is {rgb.shape!r}, expected 480x640x3")
    if rgb.dtype != np.uint8:
        raise AlignmentError(f"{label} frame dtype is {rgb.dtype}, expected uint8")


def _largest_blob(
    rgb: np.ndarray,
    hsv_ranges: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> BlobMeasurement | None:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            ),
        )
    kernel = np.ones((MORPHOLOGY_KERNEL_PX, MORPHOLOGY_KERNEL_PX), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    x, y, width, height = cv2.boundingRect(contour)
    return BlobMeasurement(
        centroid_px=(
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ),
        area_fraction=float(cv2.contourArea(contour) / mask.size),
        bbox_px=(float(x), float(y), float(width), float(height)),
    )


def measure_frames(scene_rgb: np.ndarray, wrist_rgb: np.ndarray) -> FrameMeasurements:
    """Measure already-oriented RGB frames without applying another rotation."""

    _validate_rgb(scene_rgb, "scene")
    _validate_rgb(wrist_rgb, "wrist")
    return FrameMeasurements(
        scene_red=_largest_blob(scene_rgb, RED_HSV_RANGES),
        scene_blue=_largest_blob(scene_rgb, BLUE_HSV_RANGES),
        wrist_red=_largest_blob(wrist_rgb, RED_HSV_RANGES),
    )


def _median_blob(values: Sequence[BlobMeasurement]) -> BlobMeasurement:
    centroids = np.asarray([value.centroid_px for value in values], dtype=np.float64)
    areas = np.asarray([value.area_fraction for value in values], dtype=np.float64)
    boxes = np.asarray([value.bbox_px for value in values], dtype=np.float64)
    return BlobMeasurement(
        centroid_px=tuple(float(value) for value in np.median(centroids, axis=0)),
        area_fraction=float(np.median(areas)),
        bbox_px=tuple(float(value) for value in np.median(boxes, axis=0)),
    )


def _blob_check(
    name: str,
    measured: BlobMeasurement | None,
    reference: BlobReference,
) -> dict[str, Any]:
    target = asdict(reference)
    if measured is None:
        return {
            "ok": False,
            "target": target,
            "measured": None,
            "errors": [f"{name}:blob_missing_in_sliding_window"],
        }
    centroid_delta = tuple(
        actual - expected
        for actual, expected in zip(measured.centroid_px, reference.centroid_px, strict=True)
    )
    area_delta = measured.area_fraction - reference.area_fraction
    bbox_delta = tuple(
        actual - expected
        for actual, expected in zip(measured.bbox_px, reference.bbox_px, strict=True)
    )
    errors: list[str] = []
    axes = ("x", "y")
    for axis, delta, tolerance in zip(
        axes,
        centroid_delta,
        reference.centroid_tolerance_px,
        strict=True,
    ):
        if abs(delta) > tolerance:
            errors.append(
                f"{name}:centroid_{axis}_delta={delta:+.3f}px exceeds {tolerance:.3f}px"
            )
    if abs(area_delta) > reference.area_tolerance_fraction:
        errors.append(
            f"{name}:area_delta={area_delta:+.7f} exceeds "
            f"{reference.area_tolerance_fraction:.7f}"
        )
    bbox_names = ("left", "top", "width", "height")
    for component, delta, tolerance in zip(
        bbox_names,
        bbox_delta,
        reference.bbox_tolerance_px,
        strict=True,
    ):
        if abs(delta) > tolerance:
            errors.append(
                f"{name}:bbox_{component}_delta={delta:+.3f}px exceeds "
                f"{tolerance:.3f}px"
            )
    return {
        "ok": not errors,
        "target": target,
        "measured": asdict(measured),
        "delta": {
            "centroid_px": centroid_delta,
            "area_fraction": area_delta,
            "bbox_px": bbox_delta,
        },
        "errors": errors,
    }


class SlidingAlignmentWindow:
    def __init__(self, window_frames: int):
        if window_frames < 3:
            raise ValueError("window_frames must be at least three")
        self.window_frames = window_frames
        self._samples: deque[FrameMeasurements] = deque(maxlen=window_frames)

    def add(self, sample: FrameMeasurements) -> None:
        self._samples.append(sample)

    def report(self) -> dict[str, Any]:
        ready = len(self._samples) == self.window_frames
        medians: dict[str, BlobMeasurement | None] = {}
        missing: list[str] = []
        for name in V1_HOLD_REFERENCES:
            values = [getattr(sample, name) for sample in self._samples]
            if not ready or any(value is None for value in values):
                medians[name] = None
                if ready:
                    missing.append(f"{name}:one_or_more_window_frames_missing_blob")
                continue
            medians[name] = _median_blob(
                [value for value in values if value is not None]
            )
        checks = {
            name: _blob_check(name, medians[name], reference)
            for name, reference in V1_HOLD_REFERENCES.items()
        }
        errors = list(missing)
        for check in checks.values():
            errors.extend(check["errors"])
        if not ready:
            errors.insert(
                0,
                f"sliding_window_incomplete:{len(self._samples)}/{self.window_frames}",
            )
        return {
            "ready": ready,
            "window_frames": self.window_frames,
            "samples_in_window": len(self._samples),
            "alignment_ok": bool(ready and not errors),
            "checks": checks,
            "errors": errors,
        }


def _camera_contract(specs: Mapping[str, CameraSpec]) -> dict[str, Any]:
    return {name: asdict(specs[name]) for name in sorted(specs)}


def _annotate(
    rgb: np.ndarray,
    camera_name: str,
    current: FrameMeasurements,
    report: Mapping[str, Any],
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blob_names = ("scene_red", "scene_blue") if camera_name == "scene" else ("wrist_red",)
    for name in blob_names:
        reference = V1_HOLD_REFERENCES[name]
        left, top, width, height = (int(round(value)) for value in reference.bbox_px)
        cv2.rectangle(bgr, (left, top), (left + width, top + height), (255, 255, 0), 2)
        cx, cy = (int(round(value)) for value in reference.centroid_px)
        cv2.drawMarker(bgr, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        measured = getattr(current, name)
        if measured is not None:
            left, top, width, height = (int(round(value)) for value in measured.bbox_px)
            cv2.rectangle(bgr, (left, top), (left + width, top + height), (0, 255, 255), 2)
            cx, cy = (int(round(value)) for value in measured.centroid_px)
            cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    status = "PASS" if report.get("alignment_ok") else "FAIL / WAIT"
    color = (0, 200, 0) if report.get("alignment_ok") else (0, 0, 255)
    cv2.putText(
        bgr,
        f"{camera_name} {status}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    for index, name in enumerate(blob_names):
        measured = getattr(current, name)
        if measured is None:
            message = f"{name}: MISSING"
        else:
            cx, cy = measured.centroid_px
            left, top, width, height = measured.bbox_px
            message = (
                f"{name}: c=({cx:.1f},{cy:.1f}) a={measured.area_fraction:.5f} "
                f"b=({left:.0f},{top:.0f},{width:.0f},{height:.0f})"
            )
        cv2.putText(
            bgr,
            message,
            (12, 56 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return bgr


def _show_frames(
    scene_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    current: FrameMeasurements,
    report: Mapping[str, Any],
) -> bool:
    import cv2

    scene = _annotate(scene_rgb, "scene", current, report)
    wrist = _annotate(wrist_rgb, "wrist", current, report)
    cv2.imshow("Firefly read-only static camera alignment", np.hstack((scene, wrist)))
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def capture_alignment(
    cameras: Mapping[str, CameraLike],
    specs: Mapping[str, CameraSpec],
    *,
    duration_s: float,
    settle_s: float,
    window_frames: int,
    timeout_ms: float,
    headless: bool,
) -> dict[str, Any]:
    if frozenset(cameras) != EXPECTED_CAMERA_NAMES:
        raise AlignmentError("camera factory must return exactly scene and wrist")
    fps = specs["scene"].fps
    capture_frames = int(math.ceil(duration_s * fps))
    settle_frames = int(math.ceil(settle_s * fps))
    if capture_frames < window_frames:
        raise AlignmentError(
            f"duration yields {capture_frames} frames, fewer than window={window_frames}"
        )

    connected: list[CameraLike] = []
    disconnect_errors: list[str] = []
    window = SlidingAlignmentWindow(window_frames)
    captured = 0
    quit_early = False
    try:
        for name in ("scene", "wrist"):
            camera = cameras[name]
            connected.append(camera)
            camera.connect()
        for _ in range(settle_frames):
            cameras["scene"].async_read(timeout_ms=timeout_ms)
            cameras["wrist"].async_read(timeout_ms=timeout_ms)
        for _ in range(capture_frames):
            scene_rgb = np.asarray(cameras["scene"].async_read(timeout_ms=timeout_ms))
            wrist_rgb = np.asarray(cameras["wrist"].async_read(timeout_ms=timeout_ms))
            current = measure_frames(scene_rgb, wrist_rgb)
            window.add(current)
            captured += 1
            report = window.report()
            if not headless and not _show_frames(scene_rgb, wrist_rgb, current, report):
                quit_early = True
                break
    finally:
        if not headless:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        for camera in reversed(connected):
            try:
                camera.disconnect()
            except Exception as exc:
                disconnect_errors.append(f"{type(exc).__name__}:{exc}")

    report = window.report()
    errors = list(report["errors"])
    if quit_early:
        errors.append("operator_ended_display_before_finite_duration")
    if disconnect_errors:
        errors.extend(f"camera_disconnect_failed:{error}" for error in disconnect_errors)
    report.update(
        {
            "schema_version": 1,
            "tool": "firefly_static_camera_alignment",
            "read_only": True,
            "robot_or_transport_constructed": False,
            "reference_provenance": REFERENCE_PROVENANCE,
            "camera_contract": _camera_contract(specs),
            "requested_duration_s": duration_s,
            "post_driver_settle_s": settle_s,
            "captured_frame_pairs": captured,
            "headless": headless,
            "quit_early": quit_early,
            "errors": errors,
            "alignment_ok": bool(report["ready"] and not errors),
        }
    )
    return report


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def _bounded_window(raw: str) -> int:
    value = int(raw)
    if not 3 <= value <= 90:
        raise argparse.ArgumentTypeError("must be inside [3, 90]")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print the camera-only contract; do not construct cameras.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-s", type=_positive_float, default=10.0)
    parser.add_argument(
        "--settle-s",
        type=_nonnegative_float,
        default=1.0,
        help="Additional RGB-only settling after each driver's configured warmup.",
    )
    parser.add_argument("--window-frames", type=_bounded_window, default=15)
    parser.add_argument("--timeout-ms", type=_positive_float, default=500.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of --output-json; never affects camera or robot state.",
    )
    args = parser.parse_args(argv)
    if args.duration_s > 60.0:
        parser.error("--duration-s must not exceed 60 seconds")
    if args.settle_s > 10.0:
        parser.error("--settle-s must not exceed 10 seconds")
    if args.overwrite and args.output_json is None:
        parser.error("--overwrite requires --output-json")
    return args


def _emit_report(report: Mapping[str, Any], path: Path | None, *, overwrite: bool) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(payload)
    if path is None:
        return
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8") as stream:
            stream.write(payload)
    except OSError as exc:
        raise AlignmentError(f"cannot write report {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = load_camera_specs(args.config)
        if args.check_config:
            report = {
                "schema_version": 1,
                "tool": "firefly_static_camera_alignment",
                "read_only": True,
                "check_config_only": True,
                "camera_constructed": False,
                "robot_or_transport_constructed": False,
                "camera_contract": _camera_contract(specs),
                "reference_provenance": REFERENCE_PROVENANCE,
                "references": {
                    name: asdict(reference)
                    for name, reference in V1_HOLD_REFERENCES.items()
                },
            }
            _emit_report(report, args.output_json, overwrite=args.overwrite)
            return 0
        cameras = create_realsense_cameras(specs)
        report = capture_alignment(
            cameras,
            specs,
            duration_s=args.duration_s,
            settle_s=args.settle_s,
            window_frames=args.window_frames,
            timeout_ms=args.timeout_ms,
            headless=args.headless,
        )
        _emit_report(report, args.output_json, overwrite=args.overwrite)
        return 0 if report["alignment_ok"] else 2
    except KeyboardInterrupt:
        error = {
            "schema_version": 1,
            "tool": "firefly_static_camera_alignment",
            "read_only": True,
            "alignment_ok": False,
            "error": "KeyboardInterrupt:operator_interrupted_finite_capture",
        }
        sys.stderr.write(json.dumps(error, sort_keys=True) + "\n")
        return 130
    except Exception as exc:
        error = {
            "schema_version": 1,
            "tool": "firefly_static_camera_alignment",
            "read_only": True,
            "alignment_ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        sys.stderr.write(json.dumps(error, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
