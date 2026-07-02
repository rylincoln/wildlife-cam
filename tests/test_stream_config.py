"""Unit tests for :mod:`wildlife.stream.config_gen`.

Hardware-free: these tests build a real :class:`wildlife.config.Config` by
loading a temporary YAML file (derived from ``config.example.yaml``) and assert
that the generated go2rtc config has exactly two streams per camera, the right
listen/log blocks, and round-trips through ``yaml.safe_load``. No torch/cv2.

Self-contained: the sample config is constructed inline here so the tests do not
depend on ``conftest.py`` fixtures.

The livestream support lives in a separate (parallel) config slice. If that
slice has not yet been integrated into :mod:`wildlife.config` (i.e. ``Config``
has no ``livestream`` field), these tests skip rather than error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wildlife.config import Config, load_config
from wildlife.stream import build_go2rtc_config, write_go2rtc_config

pytestmark = pytest.mark.skipif(
    "livestream" not in Config.model_fields,
    reason="livestream field not yet present on wildlife.config.Config",
)

# Credentials used in the sample cameras below; the RTSP templates interpolate
# these so we can assert the *interpolated* URLs land in the go2rtc streams.
_USERNAME = "admin"
_PASSWORD = "s3cret"
_HOST_A = "192.168.1.101"
_HOST_B = "192.168.1.102"


def _camera_block(cam_id: str, host: str) -> str:
    return f"""\
  - id: {cam_id}
    host: {host}
    username: {_USERNAME}
    password: "{_PASSWORD}"
    rtsp_main: "rtsp://{{username}}:{{password}}@{{host}}:554/Preview_01_main"
    rtsp_sub: "rtsp://{{username}}:{{password}}@{{host}}:554/Preview_01_sub"
    onvif_port: 8000
"""


def _config_yaml(*, with_livestream: bool) -> str:
    """A complete, valid wildlife config YAML with two cameras.

    When ``with_livestream`` is true a non-default ``livestream:`` block is
    included so the custom listen addresses can be asserted; otherwise the block
    is omitted entirely to exercise the ``default_factory`` defaults.
    """
    livestream = (
        """\
livestream:
  enabled: true
  go2rtc_port: 1985
  api_listen: ":1990"
  webrtc_listen: ":8556"
  rtsp_listen: ":8553"
  default_stream: main
  allow_main: false
  main_mode: "webrtc,mse"
  sub_mode: "webrtc"
"""
        if with_livestream
        else ""
    )

    return f"""\
cameras:
{_camera_block("north_field", _HOST_A)}{_camera_block("south_trail", _HOST_B)}
event_source: reolink_native

capture:
  burst_frames: 5
  burst_interval_ms: 200
  stream: main
  rtsp_timeout_s: 10
  max_concurrent: 1

detection:
  model_path: "models/yolov8s.pt"
  device: "cpu"
  animal_classes:
    - bird
    - cat
    - dog
  confidence_threshold: 0.55
  min_box_area_frac: 0.01
  save_best_only: true

dedupe:
  cooldown_s: 30

storage:
  captures_dir: "~/wildlife/captures"
  db_path: "~/wildlife/captures.db"
  jpeg_quality: 85
  thumbnail_px: 320

retention:
  max_age_days: 30
  min_confidence_keep: 0.0

gallery:
  host: "0.0.0.0"
  port: 8080
  page_size: 60

resource_guard:
  detect_every_nth_event: 1
  max_burst_per_minute: 20

{livestream}"""


def _write_config(tmp_path: Path, *, with_livestream: bool = True) -> Config:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_config_yaml(with_livestream=with_livestream), encoding="utf-8")
    return load_config(cfg_path)


def _expected_url(host: str, suffix: str) -> str:
    return f"rtsp://{_USERNAME}:{_PASSWORD}@{host}:554/Preview_01_{suffix}"


def test_two_streams_per_camera_with_interpolated_urls(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    result = build_go2rtc_config(config)

    streams = result["streams"]
    # Exactly two streams per camera (2 cameras -> 4 streams), named correctly.
    assert set(streams) == {
        "north_field_main",
        "north_field_sub",
        "south_trail_main",
        "south_trail_sub",
    }
    assert len(streams) == 2 * len(config.cameras)

    # Each stream maps to the interpolated RTSP URL inside a single-item list.
    assert streams["north_field_main"] == [_expected_url(_HOST_A, "main")]
    assert streams["north_field_sub"] == [_expected_url(_HOST_A, "sub")]
    assert streams["south_trail_main"] == [_expected_url(_HOST_B, "main")]
    assert streams["south_trail_sub"] == [_expected_url(_HOST_B, "sub")]

    # The mapped URLs must match what config.py interpolated on the camera.
    for cam in config.cameras:
        assert streams[f"{cam.id}_main"] == [cam.rtsp_main]
        assert streams[f"{cam.id}_sub"] == [cam.rtsp_sub]
        assert "{" not in cam.rtsp_main  # interpolation actually happened


def test_listen_and_log_blocks_from_livestream(tmp_path: Path) -> None:
    config = _write_config(tmp_path, with_livestream=True)
    result = build_go2rtc_config(config)

    assert result["api"] == {"listen": ":1990"}
    assert result["webrtc"] == {"listen": ":8556"}
    assert result["rtsp"] == {"listen": ":8553"}
    assert result["log"] == {"level": "info", "format": "color"}


def test_default_livestream_block_uses_contract_defaults(tmp_path: Path) -> None:
    # No livestream: block in the YAML -> default_factory supplies defaults.
    config = _write_config(tmp_path, with_livestream=False)
    result = build_go2rtc_config(config)

    assert result["api"] == {"listen": ":1984"}
    assert result["webrtc"] == {"listen": ":8555"}
    assert result["rtsp"] == {"listen": ":8554"}
    # Streams are still produced regardless of livestream defaults.
    assert len(result["streams"]) == 2 * len(config.cameras)


def test_write_go2rtc_config_roundtrips(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    out_path = tmp_path / "go2rtc.yaml"

    returned = write_go2rtc_config(config, out_path)
    assert returned == out_path
    assert out_path.exists()

    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert loaded == build_go2rtc_config(config)


def test_write_go2rtc_config_default_path(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    # Default path is relative ("go2rtc.yaml"); run from tmp_path to avoid
    # polluting the repo, then verify it round-trips.
    returned = write_go2rtc_config(config, tmp_path / "go2rtc.yaml")
    loaded = yaml.safe_load(returned.read_text(encoding="utf-8"))
    assert set(loaded["streams"]) == {
        "north_field_main",
        "north_field_sub",
        "south_trail_main",
        "south_trail_sub",
    }
