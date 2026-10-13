"""Frame features shared by the worker's night-insect gate and wildlife-replay."""

from __future__ import annotations

import numpy as np

from wildlife.frame_features import box_brightness_ratio, ir_glow_reject, is_ir_frame


def _ir(h=200, w=300, level=20) -> np.ndarray:
    """A monochrome (IR night-mode) BGR frame: all three channels equal."""
    return np.full((h, w, 3), level, np.uint8)


def _colour(h=200, w=300) -> np.ndarray:
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 40, 110, 90  # B, G, R: sagebrush-ish
    return frame


def test_is_ir_frame_tells_monochrome_from_colour():
    assert is_ir_frame(_ir()) is True
    assert is_ir_frame(_colour()) is False
    assert is_ir_frame(np.zeros((50, 50), np.uint8)) is True  # single channel


def test_box_brightness_ratio_glowing_and_dark_boxes():
    frame = _ir(level=20)
    frame[50:100, 50:100] = 200  # moth-bright patch
    assert box_brightness_ratio(frame, (50, 50, 100, 100)) > 5
    dark = _ir(level=80)
    dark[50:100, 50:100] = 30  # animal darker than the lit ground
    assert box_brightness_ratio(dark, (50, 50, 100, 100)) < 0.5
    assert box_brightness_ratio(frame, (400, 400, 500, 500)) == 1.0  # off-frame box


def test_ir_glow_reject_only_judges_ir_frames_and_honours_disable():
    glowing = _ir(level=20)
    glowing[50:100, 50:100] = 200
    box = (50, 50, 100, 100)
    assert ir_glow_reject(glowing, box, 1.2) is True
    assert ir_glow_reject(glowing, box, 0.0) is False  # 0 disables
    bright_day = _colour()
    bright_day[50:100, 50:100] = 250  # a pale animal by day is never judged
    assert ir_glow_reject(bright_day, box, 1.2) is False
    dark_animal = _ir(level=80)
    dark_animal[50:100, 50:100] = 30
    assert ir_glow_reject(dark_animal, box, 1.2) is False


def test_worker_reject_reason_applies_glow_gate_first():
    from wildlife.worker import _reject_reason

    frame = _ir(level=20)
    frame[50:100, 50:100] = 200
    reason = _reject_reason(frame, (50, 50, 100, 100), None, 1.0, max_ir_glow=1.2)
    assert reason is not None and "glows under IR" in reason
    # Disabled (the default): the glow gate never fires.
    off = _reject_reason(frame, (50, 50, 100, 100), None, 1.0)
    assert off is None or "glows under IR" not in off


def test_detection_config_defaults_glow_gate_off(det_config):
    assert det_config.max_ir_box_brightness == 0.0
