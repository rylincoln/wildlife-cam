"""Tests for the pure MotionDetector (cv2/numpy; skipped when cv2 is absent)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from wildlife.motion import MotionDetector, MotionResult  # noqa: E402


def _blank(h: int = 180, w: int = 320, value: int = 0) -> np.ndarray:
    """A solid-gray BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _with_blob(frac_w: float = 0.4, frac_h: float = 0.4) -> np.ndarray:
    """A dark frame with a bright rectangle covering ~frac_w*frac_h of the area."""
    frame = _blank()
    h, w, _ = frame.shape
    bw, bh = int(w * frac_w), int(h * frac_h)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    frame[y0 : y0 + bh, x0 : x0 + bw] = 255
    return frame


def test_moving_blob_after_warmin_is_motion():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01)
    # Warm the background model on a static scene.
    for _ in range(25):
        det.update(_blank())
    result = det.update(_with_blob())
    assert isinstance(result, MotionResult)
    assert result.motion is True
    assert result.area_frac >= 0.01


def test_speckle_noise_is_not_motion():
    rng = np.random.default_rng(0)
    det = MotionDetector(downscale_width=160, min_area_frac=0.02)
    for _ in range(25):
        det.update(_blank())
    # A frame with sparse single-pixel speckle: morphology + area gate reject it.
    noisy = _blank()
    ys = rng.integers(0, noisy.shape[0], size=40)
    xs = rng.integers(0, noisy.shape[1], size=40)
    noisy[ys, xs] = 255
    result = det.update(noisy)
    assert result.motion is False


def test_blob_inside_ignore_mask_is_not_motion():
    # Mask out the entire frame as an ignore region.
    full = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, mask_polys=full)
    for _ in range(25):
        det.update(_blank())
    result = det.update(_with_blob())
    assert result.motion is False


def test_whole_frame_brightness_jump_is_scene_change():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, scene_change_thresh=40.0)
    det.update(_blank(value=0))
    result = det.update(_blank(value=200))  # IR-cut-style flip
    assert result.scene_change is True


def test_frame_diff_algorithm_detects_blob():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, algorithm="frame_diff")
    for _ in range(5):
        det.update(_blank())
    result = det.update(_with_blob())
    assert result.motion is True


def test_reset_rebuilds_state_without_error():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01)
    det.update(_blank())
    det.reset()
    # After reset the detector still works.
    result = det.update(_with_blob())
    assert isinstance(result, MotionResult)
