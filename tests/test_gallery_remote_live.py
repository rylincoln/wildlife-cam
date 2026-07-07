"""Live-embed URL tests: LAN (direct + configured modes) vs tunnel (base_path + MSE)."""

from __future__ import annotations

from pathlib import Path

from werkzeug.security import generate_password_hash

from wildlife.config import LivestreamConfig, RemoteConfig
from wildlife.gallery.app import create_app
from tests.test_gallery_live import _build_config

_SECRET = "livesecret-xyz789"
_LAN = {"REMOTE_ADDR": "192.168.1.50"}


def _remote_config(tmp_path: Path) -> object:
    cfg = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True, base_path="/go2rtc"))
    cfg.remote = RemoteConfig(enabled=True, base_url="https://cam.example.com",
                              share_secret_hash=generate_password_hash(_SECRET))
    return cfg


def _client(config):
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def test_tunnel_embed_uses_base_path_and_mse(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    body = client.get(f"/live?key={_SECRET}").get_data(as_text=True)
    # Same-origin sub-path, forced MSE (WebRTC can't cross the tunnel). Jinja
    # autoescapes "&" to "&amp;" inside the iframe's src="..." attribute (same
    # as the pre-existing embeds in test_gallery_live.py -- those just never
    # assert far enough into the query string to see it).
    assert 'src="/go2rtc/stream.html?src=north_field_sub&amp;mode=mse' in body
    assert "http://localhost:1984" not in body  # never the raw go2rtc origin over the tunnel


def test_lan_embed_uses_direct_origin_with_base_path(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    body = client.get("/live", environ_base=_LAN).get_data(as_text=True)
    # LAN goes straight to go2rtc:1984 (under base_path) with the configured mode.
    assert 'src="http://localhost:1984/go2rtc/stream.html?src=north_field_sub&amp;mode=webrtc' in body


def test_remote_hint_shown_only_over_tunnel(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    assert "live-note" in client.get(f"/live?key={_SECRET}").get_data(as_text=True)
    assert "live-note" not in client.get("/live", environ_base=_LAN).get_data(as_text=True)
