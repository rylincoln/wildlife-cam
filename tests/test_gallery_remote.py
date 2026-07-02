"""Tests for the shared-secret remote-access gate on the gallery."""

from __future__ import annotations

from pathlib import Path

from werkzeug.security import generate_password_hash

from wildlife.config import (
    CameraConfig, CaptureConfig, Config, DedupeConfig, DetectionConfig,
    GalleryConfig, RemoteConfig, ResourceGuardConfig, RetentionConfig, StorageConfig,
)
from wildlife.gallery.app import create_app

_SECRET = "opensesame-abc123"
_LAN = {"REMOTE_ADDR": "192.168.1.50"}   # simulate a LAN client
# (test client default REMOTE_ADDR is 127.0.0.1 == "via tunnel")


def _config(tmp_path: Path, *, remote: RemoteConfig) -> Config:
    return Config(
        cameras=[CameraConfig(
            id="north_field", host="192.168.1.101", username="admin", password="secret",
            rtsp_main="rtsp://{username}:{password}@{host}:554/Preview_01_main",
            rtsp_sub="rtsp://{username}:{password}@{host}:554/Preview_01_sub",
        )],
        event_source="reolink_native",
        capture=CaptureConfig(burst_frames=5, burst_interval_ms=200, stream="main",
                              rtsp_timeout_s=10, max_concurrent=1),
        detection=DetectionConfig(model_path="models/yolov8s.pt", device="cpu",
                                  animal_classes=["bird"], confidence_threshold=0.5,
                                  min_box_area_frac=0.01, save_best_only=True),
        dedupe=DedupeConfig(cooldown_s=30),
        storage=StorageConfig(captures_dir=tmp_path / "captures", db_path=tmp_path / "captures.db"),
        retention=RetentionConfig(max_age_days=30),
        gallery=GalleryConfig(host="0.0.0.0", port=8080, page_size=60),
        resource_guard=ResourceGuardConfig(),
        remote=remote,
    )


def _client(config: Config):
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def _enabled(hash_secret: str = _SECRET) -> RemoteConfig:
    return RemoteConfig(enabled=True, base_url="https://cam.rlblais.org",
                        share_secret_hash=generate_password_hash(hash_secret))


def test_disabled_is_open_over_tunnel(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=RemoteConfig(enabled=False)))
    assert client.get("/").status_code == 200  # remote off => unchanged, open


def test_lan_is_open_even_when_enabled(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/", environ_base=_LAN).status_code == 200


def test_tunnel_without_secret_404(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/").status_code == 404  # default client == tunnel, no key/cookie


def test_tunnel_valid_key_sets_cookie_and_serves(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    resp = client.get(f"/?key={_SECRET}")
    assert resp.status_code == 200
    set_cookie = resp.headers.get("Set-Cookie", "")
    assert "wl_key=" in set_cookie and "HttpOnly" in set_cookie and "Secure" in set_cookie


def test_tunnel_cookie_grants_access(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    client.get(f"/?key={_SECRET}")            # sets the cookie on the client jar
    assert client.get("/").status_code == 200  # subsequent request carries the cookie


def test_tunnel_wrong_key_404(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/?key=nope").status_code == 404


def test_enabled_without_hash_fails_closed_over_tunnel(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=RemoteConfig(enabled=True, share_secret_hash=None)))
    assert client.get("/").status_code == 404
    assert client.get("/", environ_base=_LAN).status_code == 200  # LAN still open


def test_referrer_policy_header_when_enabled(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    resp = client.get(f"/?key={_SECRET}")
    assert resp.headers.get("Referrer-Policy") == "no-referrer"


def test_admin_404_over_tunnel_but_auth_on_lan(tmp_path: Path) -> None:
    # /admin only mounts with a config_path; write a real config file for this one.
    from tests.test_admin_config_io import render_base
    from wildlife.admin import config_io as cio
    from wildlife.config import load_config

    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    cio.set_admin_password(str(cfgp), generate_password_hash("supersecret"))
    cio.set_remote_secret(str(cfgp), generate_password_hash(_SECRET))
    app = create_app(load_config(cfgp), config_path=str(cfgp))
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/admin/").status_code == 404             # via tunnel -> blocked
    assert client.get("/admin/", environ_base=_LAN).status_code == 401  # LAN -> auth challenge
