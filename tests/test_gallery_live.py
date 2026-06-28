"""Flask-level tests for the optional livestream (go2rtc) gallery routes.

Hardware-free: a self-contained :class:`~wildlife.config.Config` (with an inline
:class:`~wildlife.config.LivestreamConfig`) is built directly and handed to the
gallery app factory. Storage paths point under ``tmp_path`` so the
:class:`wildlife.store.Store` boot step initialises a throwaway SQLite DB. No
torch / cv2 / go2rtc binary is required -- the live routes only render iframe
embed URLs, they never contact go2rtc.
"""

from __future__ import annotations

from pathlib import Path

from wildlife.config import (
    CameraConfig,
    CaptureConfig,
    Config,
    DedupeConfig,
    DetectionConfig,
    GalleryConfig,
    LivestreamConfig,
    ResourceGuardConfig,
    RetentionConfig,
    StorageConfig,
)
from wildlife.gallery.app import create_app

CAMERA_IDS = ("north_field", "south_trail")


def _build_config(tmp_path: Path, *, livestream: LivestreamConfig) -> Config:
    """Assemble a minimal but fully valid :class:`Config` for the gallery app.

    Only the fields the gallery touches matter here; the rest are populated with
    plausible values so pydantic validation passes. ``storage`` is rooted under
    ``tmp_path`` so the per-app SQLite DB and captures tree are throwaway.
    """
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
            burst_frames=5,
            burst_interval_ms=200,
            stream="main",
            rtsp_timeout_s=10,
            max_concurrent=1,
        ),
        detection=DetectionConfig(
            model_path="models/yolov8s.pt",
            device="cpu",
            animal_classes=["bird", "cat", "dog"],
            confidence_threshold=0.55,
            min_box_area_frac=0.01,
            save_best_only=True,
        ),
        dedupe=DedupeConfig(cooldown_s=30),
        storage=StorageConfig(
            captures_dir=tmp_path / "captures",
            db_path=tmp_path / "captures.db",
        ),
        retention=RetentionConfig(max_age_days=30),
        gallery=GalleryConfig(host="0.0.0.0", port=8080, page_size=60),
        resource_guard=ResourceGuardConfig(),
        livestream=livestream,
    )


def _client(config: Config):
    """Return a Flask test client for the gallery app built from ``config``."""
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------------- #
# Enabled
# --------------------------------------------------------------------------- #
def test_live_grid_renders_every_camera(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True))
    client = _client(config)

    resp = client.get("/live")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)

    # go2rtc player embed present; both stream variants exposed per camera.
    assert "stream.html" in body
    for cid in CAMERA_IDS:
        assert cid in body
        assert f"{cid}_sub" in body  # default (sub) stream
        assert f"{cid}_main" in body  # available for the Main toggle


def test_live_default_iframe_src_uses_sub_stream(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True))
    client = _client(config)

    body = client.get("/live").get_data(as_text=True)
    # The initial iframe src is derived from the request host + go2rtc_port and
    # points at the camera's default (sub) stream.
    for cid in CAMERA_IDS:
        assert f'src="http://localhost:1984/stream.html?src={cid}_sub' in body


def test_live_single_camera_ok_and_isolated(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True))
    client = _client(config)

    resp = client.get("/live/north_field")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "north_field_sub" in body
    assert "stream.html" in body
    # Single view shows only the requested camera.
    assert "south_trail_sub" not in body


def test_live_single_camera_unknown_404(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True))
    client = _client(config)
    assert client.get("/live/does_not_exist").status_code == 404


def test_index_shows_live_link_when_enabled(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True))
    client = _client(config)

    body = client.get("/").get_data(as_text=True)
    assert 'href="/live"' in body
    assert ">Live<" in body


def test_main_toggle_hidden_when_allow_main_false(tmp_path: Path) -> None:
    config = _build_config(
        tmp_path,
        livestream=LivestreamConfig(enabled=True, allow_main=False),
    )
    client = _client(config)

    body = client.get("/live").get_data(as_text=True)
    assert "seg-toggle" not in body


# --------------------------------------------------------------------------- #
# Disabled
# --------------------------------------------------------------------------- #
def test_live_routes_404_when_disabled(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=False))
    client = _client(config)

    assert client.get("/live").status_code == 404
    assert client.get("/live/north_field").status_code == 404


def test_index_hides_live_link_when_disabled(tmp_path: Path) -> None:
    config = _build_config(tmp_path, livestream=LivestreamConfig(enabled=False))
    client = _client(config)

    body = client.get("/").get_data(as_text=True)
    assert 'href="/live"' not in body
