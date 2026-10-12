"""Latest wildlife sighting + recent bird IDs for the /frame kiosk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from werkzeug.security import generate_password_hash

import wildlife.frame.routes as frame_routes
from tests.test_frame_routes import SECRET, _client, _config, _get
from wildlife.config import Config, FrameConfig, RemoteConfig
from wildlife.frame.wildlife import build_wildlife, pretty_label, time_text
from wildlife.models import Detection
from wildlife.store import Store

NOW = datetime(2026, 10, 12, 17, 0)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("red_fox", "Red fox"),
        ("white_tailed_deer", "White tailed deer"),
        ("deer", "Deer"),
        ("Steller's Jay", "Steller's Jay"),
        ("", ""),
        (None, ""),
    ],
)
def test_pretty_label(raw, expected) -> None:
    assert pretty_label(raw) == expected


@pytest.mark.parametrize(
    "ts,expected",
    [
        ("2026-10-12T16:47:10.057425", "4:47 PM"),
        ("2026-10-12T05:55:09", "5:55 AM"),
        ("2026-10-12T00:05:00", "12:05 AM"),
        ("2026-10-12T12:30:00", "12:30 PM"),
        ("2026-10-11T05:55:09", "Oct 11, 5:55 AM"),
        ("not-a-date", ""),
        (None, ""),
    ],
)
def test_time_text(ts, expected) -> None:
    assert time_text(ts, NOW) == expected


def _cap(cid, kind, label, ts, conf=0.9, camera="water_tower"):
    audio = kind == "audio"
    return {
        "id": cid, "source_kind": kind, "label": label, "capture_ts": ts, "confidence": conf,
        "camera_id": camera, "thumb_url": f"/thumb/{cid}", "image_url": f"/image/{cid}",
        "audio_url": f"/audio/{cid}" if audio else None,
        "species_photo_url": f"/species_photo/x{cid}" if audio else None,
    }


def test_build_wildlife_with_nothing_recorded() -> None:
    assert build_wildlife([], [], NOW) == {"sighting": None, "birds": []}


def test_sighting_is_the_newest_camera_capture_across_kinds() -> None:
    cams = [
        _cap(1, "reolink", "deer", "2026-10-12T05:00:00"),
        _cap(2, "continuous", "red_fox", "2026-10-12T05:55:09", camera="fenceline_behind_shop"),
    ]
    assert build_wildlife(cams, [], NOW)["sighting"] == {
        "label": "Red fox", "camera": "fenceline behind shop", "time": "5:55 AM",
        "thumb_url": "/thumb/2", "image_url": "/image/2",
    }


def test_birds_are_distinct_species_newest_first_and_capped() -> None:
    birds = [
        _cap(10, "audio", "Steller's Jay", "2026-10-12T16:47:00", 0.97),
        _cap(9, "audio", "Common Raven", "2026-10-12T16:00:00", 0.83),
        _cap(8, "audio", "Steller's Jay", "2026-10-12T15:00:00", 0.5),
    ] + [_cap(100 + i, "audio", f"Bird {i}", "2026-10-12T10:00:00") for i in range(20)]
    out = build_wildlife([], birds, NOW, max_birds=5)["birds"]
    assert [b["label"] for b in out] == ["Steller's Jay", "Common Raven", "Bird 0", "Bird 1", "Bird 2"]
    assert out[0] == {
        "label": "Steller's Jay", "time": "4:47 PM", "confidence": "97%",
        "photo_url": "/species_photo/x10", "audio_url": "/audio/10",
    }


def test_bird_without_clip_photo_or_confidence_is_still_listed() -> None:
    bird = _cap(5, "audio", "Rock Wren", "2026-10-12T16:00:00", conf=None)
    bird["audio_url"] = bird["species_photo_url"] = None
    out = build_wildlife([], [bird], NOW)["birds"][0]
    assert (out["audio_url"], out["photo_url"], out["confidence"]) == (None, None, "")


def _seed(config: Config) -> None:
    store = Store(config.storage.db_path, config.storage.captures_dir)
    store.init_schema()
    store.save_capture(
        camera_id="south_trail", event_ts=datetime(2026, 10, 12, 5, 55),
        capture_ts=datetime(2026, 10, 12, 5, 55, 9), frame=np.zeros((32, 64, 3), np.uint8),
        det=Detection(label="red_fox", confidence=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0),
                      box_area_frac=0.1),
        source_kind="continuous",
    )
    store.save_audio_capture(
        camera_id="north_field", event_ts=datetime(2026, 10, 12, 16, 47),
        capture_ts=datetime(2026, 10, 12, 16, 47, 10), species="American Robin", confidence=0.97,
        spectrogram_rgb=np.zeros((32, 64, 3), np.uint8), clip_bytes=b"\x00\x01",
    )
    store.close()


def test_wildlife_route_returns_latest_sighting_and_birds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frame_routes, "_now", lambda: NOW)
    cfg = _config(tmp_path)
    _seed(cfg)
    resp = _get(_client(cfg), "/frame/wildlife")
    assert resp.status_code == 200 and resp.headers["Cache-Control"] == "no-store"
    data = resp.get_json()
    sighting = data["sighting"]
    assert (sighting["label"], sighting["camera"], sighting["time"]) == ("Red fox", "south trail", "5:55 AM")
    assert sighting["thumb_url"].startswith("/thumb/")
    bird = data["birds"][0]
    assert (bird["label"], bird["time"], bird["confidence"]) == ("American Robin", "4:47 PM", "97%")
    assert bird["audio_url"].startswith("/audio/") and bird["photo_url"].startswith("/species_photo/")


def test_wildlife_route_with_empty_database(tmp_path: Path) -> None:
    data = _get(_client(_config(tmp_path)), "/frame/wildlife").get_json()
    assert data == {"sighting": None, "birds": []}


def test_wildlife_route_404_when_frame_disabled(tmp_path: Path) -> None:
    cfg = _config(tmp_path, frame=FrameConfig(enabled=False))
    assert _get(_client(cfg), "/frame/wildlife").status_code == 404


def test_wildlife_route_is_lan_only(tmp_path: Path) -> None:
    remote = RemoteConfig(enabled=True, share_secret_hash=generate_password_hash(SECRET))
    client = _client(_config(tmp_path, remote=remote))
    assert client.get(f"/frame/wildlife?key={SECRET}").status_code == 404


def test_page_has_wildlife_panel_bird_list_and_player(tmp_path: Path) -> None:
    body = _get(_client(_config(tmp_path)), "/frame").get_data(as_text=True)
    for element_id in ("sighting", "bird", "birds", "bird-list", "birds-close", "player"):
        assert f'id="{element_id}"' in body
    assert "/frame/wildlife" in body
    assert "innerHTML" not in body
