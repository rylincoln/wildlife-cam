"""Validation tests for ContinuousConfig and CameraConfig.motion_mask."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildlife.config import CameraConfig, ContinuousConfig


def _camera(**overrides):
    base = dict(
        id="north_field",
        host="192.168.1.101",
        username="admin",
        password="x",
        rtsp_main="rtsp://{username}:{password}@{host}:554/main",
        rtsp_sub="rtsp://{username}:{password}@{host}:554/sub",
    )
    base.update(overrides)
    return base


def test_continuous_defaults_are_inert():
    cc = ContinuousConfig()
    assert cc.enabled is False
    assert cc.sample_fps == 4
    assert cc.downscale_width == 480
    assert cc.min_area_frac == pytest.approx(0.003)
    assert cc.refractory_s == 8.0
    assert cc.warmup_s == 10.0
    assert cc.algorithm == "mog2"
    assert cc.active_hours == ""


def test_continuous_rejects_bad_values():
    with pytest.raises(ValidationError):
        ContinuousConfig(sample_fps=0)
    with pytest.raises(ValidationError):
        ContinuousConfig(downscale_width=32)
    with pytest.raises(ValidationError):
        ContinuousConfig(min_area_frac=0.0)
    with pytest.raises(ValidationError):
        ContinuousConfig(min_area_frac=1.0)
    with pytest.raises(ValidationError):
        ContinuousConfig(algorithm="optical_flow")


def test_active_hours_accepts_empty_and_valid_windows():
    assert ContinuousConfig(active_hours="").active_hours == ""
    assert ContinuousConfig(active_hours="20:00-06:00").active_hours == "20:00-06:00"
    assert ContinuousConfig(active_hours="06:30-18:45").active_hours == "06:30-18:45"


def test_active_hours_rejects_malformed():
    for bad in ("20:00", "25:00-06:00", "20:61-06:00", "8:00-9:00", "abc"):
        with pytest.raises(ValidationError):
            ContinuousConfig(active_hours=bad)


def test_motion_mask_defaults_none_and_accepts_valid_polygons():
    assert CameraConfig(**_camera()).motion_mask is None
    cam = CameraConfig(
        **_camera(motion_mask=[[[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]])
    )
    assert cam.motion_mask == [[(0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)]]


def test_motion_mask_rejects_short_polygon_and_out_of_range_points():
    with pytest.raises(ValidationError):
        CameraConfig(**_camera(motion_mask=[[[0.0, 0.0], [1.0, 1.0]]]))  # only 2 points
    with pytest.raises(ValidationError):
        CameraConfig(**_camera(motion_mask=[[[0.0, 0.0], [1.0, 0.0], [1.5, 1.0]]]))  # x>1
