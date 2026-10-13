"""Flask tests for the LAN-only /frame kiosk routes."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

import wildlife.frame.ha as ha_mod
import wildlife.frame.routes as frame_routes
from wildlife.config import (
    CameraConfig,
    CaptureConfig,
    Config,
    DedupeConfig,
    DetectionConfig,
    FrameConfig,
    GalleryConfig,
    HAEntityConfig,
    HomeAssistantConfig,
    LivestreamConfig,
    MjpegConfig,
    RemoteConfig,
    ResourceGuardConfig,
    RetentionConfig,
    StorageConfig,
)
from wildlife.gallery.app import create_app

CAMERA_IDS = ("north_field", "south_trail")
LAN = {"REMOTE_ADDR": "192.168.1.66"}
HOST = "http://192.168.1.193:8080"
TOKEN = "super-secret-token-value"
SECRET = "opensesame-abc123"


def _config(
    tmp_path: Path,
    *,
    frame: FrameConfig | None = None,
    livestream: LivestreamConfig | None = None,
    remote: RemoteConfig | None = None,
) -> Config:
    cameras = [
        CameraConfig(
            id=cid, host=f"192.168.1.10{i}", username="admin", password="secret",
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
        livestream=livestream
        or LivestreamConfig(enabled=True, base_path="/go2rtc", mjpeg=MjpegConfig()),
        remote=remote or RemoteConfig(),
        frame=frame or FrameConfig(enabled=True),
        homeassistant=HomeAssistantConfig(
            url="http://192.168.1.33:8123",
            token=TOKEN,
            entities=[HAEntityConfig(id="sensor.outdoor_temp", slot="temps", label="Outside")],
        ),
    )


def _client(config: Config):
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def _get(client, path: str, **kwargs):
    return client.get(path, base_url=HOST, environ_base=LAN, **kwargs)


@pytest.fixture
def fake_ha(monkeypatch):
    calls: list[str] = []

    def fetch(method, url, headers, body, timeout):
        calls.append(url)
        return {
            "entity_id": "sensor.outdoor_temp",
            "state": "68",
            "attributes": {"unit_of_measurement": "°F"},
        }

    monkeypatch.setattr(ha_mod, "urllib_fetch", fetch)
    return calls


@pytest.fixture
def at(monkeypatch):
    def set_time(hour: int, minute: int) -> datetime:
        moment = datetime(2026, 10, 12, hour, minute)
        monkeypatch.setattr(frame_routes, "_now", lambda: moment)
        return moment

    return set_time


def test_all_frame_routes_404_when_disabled(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, frame=FrameConfig(enabled=False)))
    for path in ("/frame", "/frame/state", "/frame/info"):
        assert _get(client, path).status_code == 404


def test_page_renders_mjpeg_urls_for_each_camera(tmp_path: Path) -> None:
    resp = _get(_client(_config(tmp_path)), "/frame")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    for cid in CAMERA_IDS:
        assert f"http://192.168.1.193:1984/go2rtc/api/stream.mjpeg?src={cid}_mjpeg" in body
        assert f"http://192.168.1.193:1984/go2rtc/api/stream.mjpeg?src={cid}_mjpeg_hd" in body
    assert 'data-refresh="15"' in body
    assert TOKEN not in body


def test_go2rtc_url_override_is_honoured(tmp_path: Path) -> None:
    ls = LivestreamConfig(
        enabled=True, go2rtc_url="http://cams.lan:1984/", base_path="/go2rtc", mjpeg=MjpegConfig()
    )
    body = _get(_client(_config(tmp_path, livestream=ls)), "/frame").get_data(as_text=True)
    assert "http://cams.lan:1984/go2rtc/api/stream.mjpeg?src=north_field_mjpeg" in body


def test_camera_order_and_unknown_ids(tmp_path: Path) -> None:  # Review Focus 4
    only_south = _config(tmp_path, frame=FrameConfig(enabled=True, cameras=["south_trail", "ghost"]))
    body = _get(_client(only_south), "/frame").get_data(as_text=True)
    assert "src=south_trail_mjpeg" in body and "src=north_field_mjpeg" not in body
    all_unknown = _config(tmp_path, frame=FrameConfig(enabled=True, cameras=["ghost"]))
    body = _get(_client(all_unknown), "/frame").get_data(as_text=True)
    assert "src=south_trail_mjpeg" in body and "src=north_field_mjpeg" in body


def test_page_without_mjpeg_shows_notice(tmp_path: Path) -> None:
    ls = LivestreamConfig(enabled=True, base_path="/go2rtc")
    body = _get(_client(_config(tmp_path, livestream=ls)), "/frame").get_data(as_text=True)
    assert "livestream.mjpeg" in body and "/api/stream.mjpeg" not in body


def test_state_reports_dim_window_and_ha(tmp_path: Path, fake_ha, at) -> None:
    moment = at(23, 0)
    resp = _get(_client(_config(tmp_path)), "/frame/state")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"
    data = resp.get_json()
    assert data["now"] == moment.timestamp()
    assert data["dim"] == {
        "active": True, "level": 0.12, "wake_seconds": 60, "start": "22:00", "end": "06:30"
    }
    assert data["ha"]["ok"] is True
    assert data["ha"]["items"][0]["display"] == "68°"
    assert fake_ha == ["http://192.168.1.33:8123/api/states/sensor.outdoor_temp"]


def test_state_dim_inactive_at_midday(tmp_path: Path, fake_ha, at) -> None:
    at(12, 0)
    data = _get(_client(_config(tmp_path)), "/frame/state").get_json()
    assert data["dim"]["active"] is False


def test_state_with_dimming_disabled(tmp_path: Path, fake_ha, at) -> None:
    at(23, 0)
    cfg = _config(tmp_path, frame=FrameConfig(enabled=True, dim=None))
    data = _get(_client(cfg), "/frame/state").get_json()
    assert data["dim"] == {
        "active": False, "level": 1.0, "wake_seconds": 0, "start": None, "end": None
    }


def test_state_never_exposes_token(tmp_path: Path, fake_ha, at) -> None:
    at(12, 0)
    assert TOKEN not in _get(_client(_config(tmp_path)), "/frame/state").get_data(as_text=True)


def test_frame_routes_are_lan_only_even_with_valid_share_key(tmp_path: Path, fake_ha) -> None:
    remote = RemoteConfig(enabled=True, share_secret_hash=generate_password_hash(SECRET))
    client = _client(_config(tmp_path, remote=remote))
    for path in ("/frame", "/frame/state", "/frame/info"):
        # Default test REMOTE_ADDR is 127.0.0.1 == arrived via the cloudflared tunnel.
        assert client.get(f"{path}?key={SECRET}").status_code == 404
    assert _get(client, "/frame").status_code == 200  # LAN still works


def test_info_records_last_poller(tmp_path: Path, fake_ha, at) -> None:
    at(12, 0)
    client = _client(_config(tmp_path))
    assert _get(client, "/frame/info").get_data(as_text=True) == "no polls yet\n"
    _get(client, "/frame/state", headers={"User-Agent": "Mozilla/5.0 FrameKiosk/1"})
    text = _get(client, "/frame/info").get_data(as_text=True)
    assert "192.168.1.66" in text and "FrameKiosk/1" in text and "2026-10-12T12:00:00" in text


def test_config_reload_changes_entities_without_restart(tmp_path: Path, fake_ha, at) -> None:
    # Review Focus 5: create_app(config_path=...) reloads config.yaml on mtime change.
    import os

    import yaml

    at(12, 0)
    cfg = _config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    app = create_app(cfg, config_path=path)
    app.config.update(TESTING=True)
    client = app.test_client()
    _get(client, "/frame/state")
    data = cfg.model_dump(mode="json")
    data["homeassistant"]["entities"] = [{"id": "sensor.other_temp", "slot": "temps", "label": "Shed"}]
    path.write_text(yaml.safe_dump(data))
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))
    items = _get(client, "/frame/state").get_json()["ha"]["items"]
    assert [i["label"] for i in items] == ["Shed"]
    assert fake_ha[-1].endswith("/api/states/sensor.other_temp")


GALLERY_DIR = Path(__file__).resolve().parents[1] / "src" / "wildlife" / "gallery"
FORBIDDEN_CSS = ("oklch(", "color-mix(", ":has(", "dvh", "dvw", "svh", "lvh", "@container", "@layer")


def test_page_has_strip_overlays_and_stylesheet(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    for element_id in ("time", "date", "weather", "values", "offline", "dimmer", "full"):
        assert f'id="{element_id}"' in body
    assert "/static/frame.css" in body
    assert "FrameNative" in body
    assert "innerHTML" not in body  # HA strings must be rendered with textContent


def test_frame_css_is_chrome_106_safe() -> None:
    css = (GALLERY_DIR / "static" / "frame.css").read_text(encoding="utf-8")
    for token in FORBIDDEN_CSS:
        assert token not in css, token
    assert "&" not in css  # no CSS nesting


def test_frame_stylesheet_is_served_on_lan(tmp_path: Path) -> None:
    resp = _get(_client(_config(tmp_path)), "/static/frame.css")
    assert resp.status_code == 200 and "1280px" in resp.get_data(as_text=True)


def test_camera_images_have_empty_alt_so_a_down_stream_paints_nothing(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    alts = re.findall(r'<img alt="([^"]*)"', body)
    assert len(alts) >= len(CAMERA_IDS) + 1  # tiles + full-screen overlay (+ wildlife thumbs)
    assert all(alt == "" for alt in alts)


def test_night_mode_turns_text_red_and_dims_cameras_more_than_text(tmp_path: Path) -> None:
    # Owner request: at night the strip text goes red and stays brighter than the camera images
    # (the backlight dims everything equally, so the images get an extra CSS fade).
    css = (GALLERY_DIR / "static" / "frame.css").read_text(encoding="utf-8")
    assert "#frame.night .strip" in css and "#ff3b30" in css
    assert "#frame.night .tile img" in css and "opacity: 0.3" in css
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'root.className = want ? "night" : ""' in body


def test_state_survives_an_unexpected_ha_failure(tmp_path: Path, monkeypatch, at) -> None:
    # final review #1: an HA fault must never take dimming and the clock down with it.
    at(23, 0)

    def boom(self, ha, refresh_seconds):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ha_mod.HACache, "snapshot", boom)
    resp = _get(_client(_config(tmp_path)), "/frame/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["dim"]["active"] is True
    assert data["ha"] == {
        "ok": False, "stale": True, "age_seconds": None, "items": [], "forecast": None
    }


def test_page_renders_initial_dim_state_so_a_reload_does_not_flash(tmp_path: Path, at) -> None:
    # final review #3: the nightly 04:00 reload sits inside the dim window.
    at(4, 0)
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'id="frame" class="night"' in body
    assert 'data-dim-active="true"' in body and 'data-dim-level="0.12"' in body
    at(12, 0)
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'data-dim-active="false"' in body and 'class="night"' not in body


def test_page_has_rain_rows_container(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'id="rain"' in body and 'item.slot === "rain"' in body


def test_cloudflare_requests_get_404_even_with_remote_disabled(tmp_path: Path) -> None:
    # minor #13: a cloudflared left running after remote.enabled is switched off must not
    # expose the HA proxy; tunnel traffic always carries Cloudflare headers.
    client = _client(_config(tmp_path))
    for path in ("/frame", "/frame/state", "/frame/info", "/frame/wildlife"):
        assert _get(client, path, headers={"Cf-Connecting-IP": "203.0.113.9"}).status_code == 404
        assert _get(client, path, headers={"Cf-Ray": "8a1b2c3d4e5f-SJC"}).status_code == 404
    assert _get(client, "/frame").status_code == 200


def test_cameras_hidden_when_livestream_disabled(tmp_path: Path) -> None:  # minor #14
    ls = LivestreamConfig(enabled=False, base_path="/go2rtc", mjpeg=MjpegConfig())
    body = _get(_client(_config(tmp_path, livestream=ls)), "/frame").get_data(as_text=True)
    assert "/api/stream.mjpeg" not in body and "livestream" in body


def test_page_and_state_carry_the_dim_window(tmp_path: Path, fake_ha, at) -> None:  # minor #11
    at(12, 0)
    client = _client(_config(tmp_path))
    dim = _get(client, "/frame/state").get_json()["dim"]
    assert (dim["start"], dim["end"]) == ("22:00", "06:30")
    body = _get(client, "/frame").get_data(as_text=True)
    assert 'data-dim-start="22:00"' in body and 'data-dim-end="06:30"' in body


def test_page_script_behaviour_pins(tmp_path: Path) -> None:
    # Page JS has no unit harness here; these pin the fixes, behaviour is checked in a browser.
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert "FULL_TIMEOUT_MS" not in body  # owner: full screen stays until tapped
    assert "failures === 0" in body  # minor #12: no 04:00 reload while the server is unreachable
    assert "just now" in body  # minor #16
    assert "inDimWindow(" in body  # minor #11: the page keeps the dim schedule offline
    assert "fullRetry" in body  # minor #5: the full-screen retry timer is tracked and cleared
    # A stream that ends mid-flight leaves the <img> complete with zero width and no error
    # event (seen in Chromium); a watchdog must treat that as an error and reconnect.
    assert "naturalWidth === 0" in body


def test_state_carries_battery_flow(tmp_path: Path, monkeypatch, at) -> None:
    at(12, 0)
    states = {
        "sensor.shop_battery_soc": {"state": "75.3", "attributes": {"unit_of_measurement": "%"}},
        "sensor.shop_shunt_power": {"state": "-420", "attributes": {"unit_of_measurement": "W"}},
    }
    monkeypatch.setattr(ha_mod, "urllib_fetch", lambda m, url, h, b, t: states[url.rsplit("/", 1)[1]])
    cfg = _config(tmp_path)
    cfg.homeassistant.entities = [
        HAEntityConfig(
            id="sensor.shop_battery_soc", slot="energy", label="Shop battery", flow="sensor.shop_shunt_power"
        )
    ]
    item = _get(_client(cfg), "/frame/state").get_json()["ha"]["items"][0]
    assert (item["label"], item["display"], item["flow"]) == ("Shop battery", "75%", "−420 W")


def test_page_renders_battery_flow_inside_the_value(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'el("span", "flow", item.flow)' in body


def test_thumbnail_captions_stack_name_over_time_without_wrapping(tmp_path: Path) -> None:
    # Owner request: "Person · Oct 9, 4:50 PM" wrapped mid-time next to the thumbnail. The name
    # and time are now two single-line, ellipsized rows, and the time row goes red at night.
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    assert 'el("div", "cap-name", label)' in body and 'el("div", "cap-time", time)' in body
    css = (GALLERY_DIR / "static" / "frame.css").read_text(encoding="utf-8")
    assert ".cap-name, .cap-time { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }" in css
    assert "#frame.night .cap-time" in css


def test_bird_sits_under_the_camera_sighting_leaving_the_weather_column_free(tmp_path: Path) -> None:
    # Owner request: the latest bird moves to the clock column under the latest camera sighting,
    # so the weather column has room for more data.
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    clock, mid = body.index('<div class="clock">'), body.index('<div class="mid">')
    assert clock < body.index('id="sighting"') < body.index('id="bird"') < mid
