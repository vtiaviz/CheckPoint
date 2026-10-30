import numpy as np

from checkpoint.camera import CameraReport, mean_brightness


def report(**overrides) -> CameraReport:
    values = {"width": 640, "height": 480, "requested_fps": 30, "measured_fps": 30.0,
              "mean_brightness": 120.0, "frames": 90, "backend": "TEST"}
    values.update(overrides)
    return CameraReport(**values)


def test_mean_brightness_color_and_gray():
    assert mean_brightness(np.full((4, 4, 3), 100, dtype=np.uint8)) == 100.0
    assert mean_brightness(np.full((4, 4), 50, dtype=np.uint8)) == 50.0


def test_good_capture_has_no_warnings():
    assert report().warnings == []


def test_dark_slow_capture_warns():
    warnings = report(measured_fps=8.0, mean_brightness=40.0).warnings
    assert len(warnings) == 2
    assert any("FPS baixo" in w for w in warnings)
    assert any("escura" in w for w in warnings)


def test_overexposed_capture_warns():
    assert any("estourada" in w for w in report(mean_brightness=230.0).warnings)
