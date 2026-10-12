"""go2rtc MJPEG restream generation (``livestream.mjpeg``).

Hardware-free: builds a :class:`~wildlife.config.Config` inline and inspects the
dict produced by :func:`wildlife.stream.build_go2rtc_config`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wildlife.config import (
    CameraConfig,
    CaptureConfig,
    Config,
    DedupeConfig,
    DetectionConfig,
    GalleryConfig,
    LivestreamConfig,
    MjpegConfig,
    ResourceGuardConfig,
    RetentionConfig,
    StorageConfig,
)
from wildlife.stream import build_go2rtc_config

CAMERA_IDS = ("north_field", "south_trail")


def _config(tmp_path: Path, livestream: LivestreamConfig) -> Config:
    cameras = [
        CameraConfig(
            id=cid,
            host=f"192.168.1.10{i}",
            username="admin",
            password="secret",
            rtsp_main="rtsp://{username}:{password}@{host}:554/Preview_01_main",
            rtsp_sub="rtsp://{username}:{password}@{host}:554/Preview_01_sub",
        )
        for i, cid in enumerate(CAMERA_IDS)
    ]
    return Config(
        cameras=cameras,
        event_source="reolink_native",
        capture=CaptureConfig(
            burst_frames=5, burst_interval_ms=200, stream="main", rtsp_timeout_s=10,
            max_concurrent=1,
        ),
        detection=DetectionConfig(
            model_path="models/yolov8s.pt", device="cpu", animal_classes=["bird"],
            confidence_threshold=0.5, min_box_area_frac=0.01, save_best_only=True,
        ),
        dedupe=DedupeConfig(cooldown_s=30),
        storage=StorageConfig(captures_dir=tmp_path / "captures", db_path=tmp_path / "c.db"),
        retention=RetentionConfig(max_age_days=30),
        gallery=GalleryConfig(host="0.0.0.0", port=8080, page_size=60),
        resource_guard=ResourceGuardConfig(),
        livestream=livestream,
    )


def test_no_mjpeg_streams_by_default(tmp_path: Path) -> None:
    streams = build_go2rtc_config(_config(tmp_path, LivestreamConfig()))["streams"]
    assert len(streams) == 2 * len(CAMERA_IDS)
    assert not any("mjpeg" in name for name in streams)


def test_mjpeg_adds_two_derived_streams_per_camera(tmp_path: Path) -> None:
    ls = LivestreamConfig(mjpeg=MjpegConfig())
    streams = build_go2rtc_config(_config(tmp_path, ls))["streams"]
    assert len(streams) == 4 * len(CAMERA_IDS)
    assert streams["north_field_mjpeg"] == [
        "ffmpeg:north_field_sub#video=mjpeg#raw=-r 4 -q:v 7"
    ]
    assert streams["north_field_mjpeg_hd"] == [
        "ffmpeg:north_field_main#video=mjpeg#width=1280#raw=-r 4 -q:v 14"
    ]


def test_mjpeg_settings_flow_into_ffmpeg_args(tmp_path: Path) -> None:
    ls = LivestreamConfig(mjpeg=MjpegConfig(fps=2, quality=9, hd_width=960, hd_quality=18))
    streams = build_go2rtc_config(_config(tmp_path, ls))["streams"]
    assert streams["south_trail_mjpeg"] == ["ffmpeg:south_trail_sub#video=mjpeg#raw=-r 2 -q:v 9"]
    assert streams["south_trail_mjpeg_hd"] == [
        "ffmpeg:south_trail_main#video=mjpeg#width=960#raw=-r 2 -q:v 18"
    ]


def test_mjpeg_sources_reference_go2rtc_streams_not_camera_credentials(tmp_path: Path) -> None:
    streams = build_go2rtc_config(_config(tmp_path, LivestreamConfig(mjpeg=MjpegConfig())))
    for name, sources in streams["streams"].items():
        if "mjpeg" in name:
            assert "secret" not in sources[0] and "rtsp://" not in sources[0]


@pytest.mark.parametrize(
    "field,value", [("fps", 0), ("fps", 16), ("quality", 1), ("quality", 32), ("hd_width", 100)]
)
def test_mjpeg_rejects_out_of_range(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        MjpegConfig(**{field: value})


def test_mjpeg_block_parses_from_mapping() -> None:
    # pydantic silently drops unknown keys, so prove the field is really wired.
    ls = LivestreamConfig.model_validate({"mjpeg": {"fps": 3}})
    assert ls.mjpeg is not None
    assert (ls.mjpeg.fps, ls.mjpeg.quality, ls.mjpeg.hd_width, ls.mjpeg.hd_quality) == (3, 7, 1280, 14)
