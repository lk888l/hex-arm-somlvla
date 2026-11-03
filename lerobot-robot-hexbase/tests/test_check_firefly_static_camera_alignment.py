from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import yaml

from scripts import check_firefly_static_camera_alignment as alignment


def _camera_config() -> dict[str, Any]:
    return {
        "robot": {
            "cameras": {
                "scene": {
                    "type": "intelrealsense",
                    "serial_number_or_name": "030522071867",
                    "fps": 30,
                    "width": 640,
                    "height": 480,
                    "use_rgb": True,
                    "use_depth": False,
                    "warmup_s": 1,
                },
                "wrist": {
                    "type": "intelrealsense",
                    "serial_number_or_name": "323622270296",
                    "fps": 30,
                    "width": 640,
                    "height": 480,
                    "rotation": 180,
                    "use_rgb": True,
                    "use_depth": False,
                    "warmup_s": 1,
                },
            }
        }
    }


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "camera_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _specs(tmp_path: Path) -> dict[str, alignment.CameraSpec]:
    return alignment.load_camera_specs(_write_config(tmp_path, _camera_config()))


def _reference_sample() -> alignment.FrameMeasurements:
    values = {
        name: alignment.BlobMeasurement(
            centroid_px=reference.centroid_px,
            area_fraction=reference.area_fraction,
            bbox_px=reference.bbox_px,
        )
        for name, reference in alignment.V1_HOLD_REFERENCES.items()
    }
    return alignment.FrameMeasurements(**values)


def _synthetic_frames() -> tuple[np.ndarray, np.ndarray]:
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist = np.zeros_like(scene)
    scene[60:180, 80:220] = (0, 0, 255)  # RGB blue.
    scene[230:290, 470:530] = (255, 0, 0)  # RGB red.
    wrist[330:390, 30:100] = (255, 0, 0)  # Stays bottom-left after driver.
    return scene, wrist


class FakeCamera:
    def __init__(
        self,
        frame: np.ndarray,
        *,
        connect_error: BaseException | None = None,
        read_error: BaseException | None = None,
    ) -> None:
        self.frame = frame
        self.connect_error = connect_error
        self.read_error = read_error
        self.connect_calls = 0
        self.read_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        del timeout_ms
        self.read_calls += 1
        if self.read_error is not None:
            raise self.read_error
        return self.frame.copy()

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def test_module_import_has_no_robot_zenoh_or_can_side_effects() -> None:
    repository = Path(alignment.__file__).resolve().parents[1]
    code = """
import json
import sys

before = set(sys.modules)
import scripts.check_firefly_static_camera_alignment  # noqa: F401
prefixes = ('can', 'canopen', 'zenoh', 'lerobot.robots', 'lerobot_robot_hexbase')
new = set(sys.modules) - before
bad = sorted(
    name for name in new
    if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes)
)
print(json.dumps(bad))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_source_contains_no_robot_transport_import_or_action_call() -> None:
    source = Path(alignment.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    called: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)

    forbidden_prefixes = (
        "can",
        "canopen",
        "zenoh",
        "lerobot.robots",
        "lerobot_robot_hexbase",
    )
    assert not {
        module
        for module in imported
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    }
    assert not set(called).intersection(
        {
            "send_action",
            "send_command",
            "publish",
            "open_zenoh_session",
            "write_joint_command",
        }
    )
    assert "cv2.rotate" not in source
    assert "np.rot90" not in source


def test_current_camera_contract_loads_with_d405_rotation_exactly_once() -> None:
    specs = alignment.load_camera_specs(alignment.DEFAULT_CONFIG_PATH)

    assert specs["scene"].serial == "030522071867"
    assert specs["scene"].rotation_degrees == 0
    assert specs["wrist"].serial == "323622270296"
    assert specs["wrist"].rotation_degrees == 180
    assert all((spec.width, spec.height, spec.fps) == (640, 480, 30) for spec in specs.values())


@pytest.mark.parametrize(
    ("camera_name", "field", "value"),
    [
        ("wrist", "rotation", 0),
        ("wrist", "rotation", 90),
        ("scene", "rotation", 180),
        ("wrist", "use_depth", True),
        ("scene", "use_rgb", False),
        ("scene", "width", 1280),
        ("wrist", "serial_number_or_name", "wrong"),
        ("scene", "warmup_s", 0),
    ],
)
def test_camera_contract_rejects_any_unreviewed_change(
    tmp_path: Path,
    camera_name: str,
    field: str,
    value: object,
) -> None:
    raw = _camera_config()
    raw["robot"]["cameras"][camera_name][field] = value

    with pytest.raises(alignment.AlignmentError):
        alignment.load_camera_specs(_write_config(tmp_path, raw))


def test_camera_contract_rejects_extra_camera(tmp_path: Path) -> None:
    raw = _camera_config()
    raw["robot"]["cameras"]["extra"] = dict(raw["robot"]["cameras"]["scene"])

    with pytest.raises(alignment.AlignmentError, match="exactly scene and wrist"):
        alignment.load_camera_specs(_write_config(tmp_path, raw))


def test_check_config_never_constructs_a_camera(tmp_path: Path, monkeypatch, capsys) -> None:
    config = _write_config(tmp_path, _camera_config())
    monkeypatch.setattr(
        alignment,
        "create_realsense_cameras",
        lambda _specs: pytest.fail("--check-config must not construct a camera"),
    )

    assert alignment.main(["--config", str(config), "--check-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["camera_constructed"] is False
    assert report["robot_or_transport_constructed"] is False
    assert report["camera_contract"]["wrist"]["rotation_degrees"] == 180


def test_second_d405_rotation_is_rejected_before_factory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    raw = _camera_config()
    raw["robot"]["cameras"]["wrist"]["rotation"] = 0
    config = _write_config(tmp_path, raw)
    monkeypatch.setattr(
        alignment,
        "create_realsense_cameras",
        lambda _specs: pytest.fail("invalid rotation reached camera factory"),
    )

    assert alignment.main(["--config", str(config), "--check-config"]) == 2
    assert "Do not rotate D405 a second time" in capsys.readouterr().err


def test_measurement_uses_rgb_and_never_rotates_driver_frames_again() -> None:
    scene, wrist = _synthetic_frames()

    measured = alignment.measure_frames(scene, wrist)

    assert measured.scene_red is not None
    assert measured.scene_blue is not None
    assert measured.wrist_red is not None
    assert measured.scene_red.bbox_px == (470.0, 230.0, 60.0, 60.0)
    assert measured.scene_blue.bbox_px == (80.0, 60.0, 140.0, 120.0)
    assert measured.wrist_red.bbox_px == (30.0, 330.0, 70.0, 60.0)
    assert measured.wrist_red.centroid_px[0] < 100
    assert measured.wrist_red.centroid_px[1] > 300


def test_measurement_fails_closed_on_bad_shape_or_missing_blob() -> None:
    scene, wrist = _synthetic_frames()
    with pytest.raises(alignment.AlignmentError, match="shape"):
        alignment.measure_frames(scene[:400], wrist)

    blank = np.zeros_like(scene)
    measured = alignment.measure_frames(blank, blank)
    assert measured == alignment.FrameMeasurements(None, None, None)


def test_sliding_window_median_passes_only_with_every_required_blob() -> None:
    window = alignment.SlidingAlignmentWindow(3)
    for _ in range(3):
        window.add(_reference_sample())

    report = window.report()
    assert report["ready"] is True
    assert report["alignment_ok"] is True
    assert report["errors"] == []

    missing = alignment.SlidingAlignmentWindow(3)
    sample = _reference_sample()
    missing.add(sample)
    missing.add(alignment.FrameMeasurements(None, sample.scene_blue, sample.wrist_red))
    missing.add(sample)
    failed = missing.report()
    assert failed["alignment_ok"] is False
    assert any("scene_red" in error and "missing" in error for error in failed["errors"])


def test_sliding_window_reports_geometric_delta_outside_tolerance() -> None:
    sample = _reference_sample()
    ref = alignment.V1_HOLD_REFERENCES["wrist_red"]
    shifted = alignment.BlobMeasurement(
        centroid_px=(ref.centroid_px[0] + 2.0, ref.centroid_px[1]),
        area_fraction=ref.area_fraction,
        bbox_px=ref.bbox_px,
    )
    window = alignment.SlidingAlignmentWindow(3)
    for _ in range(3):
        window.add(
            alignment.FrameMeasurements(sample.scene_red, sample.scene_blue, shifted)
        )

    report = window.report()
    assert report["alignment_ok"] is False
    assert any("wrist_red:centroid_x_delta" in error for error in report["errors"])


def test_headless_capture_is_finite_and_disconnects_both_cameras(tmp_path: Path) -> None:
    scene_frame, wrist_frame = _synthetic_frames()
    scene = FakeCamera(scene_frame)
    wrist = FakeCamera(wrist_frame)

    report = alignment.capture_alignment(
        {"scene": scene, "wrist": wrist},
        _specs(tmp_path),
        duration_s=0.1,
        settle_s=0,
        window_frames=3,
        timeout_ms=100,
        headless=True,
    )

    assert report["captured_frame_pairs"] == 3
    assert report["headless"] is True
    assert report["read_only"] is True
    assert report["robot_or_transport_constructed"] is False
    assert scene.connect_calls == wrist.connect_calls == 1
    assert scene.read_calls == wrist.read_calls == 3
    assert scene.disconnect_calls == wrist.disconnect_calls == 1


def test_partial_connect_failure_disconnects_the_first_camera(tmp_path: Path) -> None:
    scene_frame, wrist_frame = _synthetic_frames()
    scene = FakeCamera(scene_frame)
    wrist = FakeCamera(wrist_frame, connect_error=ConnectionError("busy"))

    with pytest.raises(ConnectionError, match="busy"):
        alignment.capture_alignment(
            {"scene": scene, "wrist": wrist},
            _specs(tmp_path),
            duration_s=0.1,
            settle_s=0,
            window_frames=3,
            timeout_ms=100,
            headless=True,
        )

    assert scene.disconnect_calls == 1
    assert wrist.disconnect_calls == 1


def test_keyboard_interrupt_disconnects_every_connected_camera(tmp_path: Path) -> None:
    scene_frame, wrist_frame = _synthetic_frames()
    scene = FakeCamera(scene_frame)
    wrist = FakeCamera(wrist_frame, read_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        alignment.capture_alignment(
            {"scene": scene, "wrist": wrist},
            _specs(tmp_path),
            duration_s=0.1,
            settle_s=0,
            window_frames=3,
            timeout_ms=100,
            headless=True,
        )

    assert scene.disconnect_calls == wrist.disconnect_calls == 1


def test_main_converts_keyboard_interrupt_to_fail_closed_exit(monkeypatch, capsys) -> None:
    specs = {
        name: alignment.CameraSpec(
            name=name,
            serial=alignment.EXPECTED_CAMERA_SERIALS[name],
            fps=30,
            width=640,
            height=480,
            rotation_degrees=alignment.EXPECTED_ROTATIONS[name],
            warmup_s=1,
        )
        for name in alignment.EXPECTED_CAMERA_NAMES
    }
    monkeypatch.setattr(alignment, "load_camera_specs", lambda _path: specs)
    monkeypatch.setattr(alignment, "create_realsense_cameras", lambda _specs: {})
    monkeypatch.setattr(
        alignment,
        "capture_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert alignment.main(["--headless"]) == 130
    error = json.loads(capsys.readouterr().err)
    assert error["alignment_ok"] is False
    assert error["read_only"] is True
