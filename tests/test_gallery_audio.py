"""Gallery audio endpoint + source_kind serialization/filter tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask")

from datetime import datetime  # noqa: E402

from wildlife.config import load_config  # noqa: E402
from wildlife.gallery.app import create_app  # noqa: E402
from wildlife.models import Detection  # noqa: E402
from wildlife.store import Store  # noqa: E402


def _app(tmp_path):
    # minimal config on disk
    cfg_text = f"""
cameras:
  - id: cam1
    host: 1.2.3.4
    username: u
    password: p
    rtsp_main: "rtsp://{{username}}:{{password}}@{{host}}/main"
    rtsp_sub: "rtsp://{{username}}:{{password}}@{{host}}/sub"
event_source: reolink_native
capture: {{burst_frames: 3, burst_interval_ms: 100, stream: main, rtsp_timeout_s: 5, max_concurrent: 1}}
detection: {{model_path: m.pt, device: cpu, animal_classes: [bird], confidence_threshold: 0.5, min_box_area_frac: 0.01, save_best_only: true}}
dedupe: {{cooldown_s: 0}}
storage: {{captures_dir: "{tmp_path/'caps'}", db_path: "{tmp_path/'c.db'}"}}
retention: {{max_age_days: 30}}
gallery: {{host: 0.0.0.0, port: 8080, page_size: 60}}
resource_guard: {{detect_every_nth_event: 1, max_burst_per_minute: 20}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)
    config = load_config(cfg_path)
    store = Store(config.storage.db_path, config.storage.captures_dir)
    store.init_schema()
    cid = store.save_audio_capture(
        camera_id="cam1", event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, 0, 1), species="American Robin",
        confidence=0.8, spectrogram_rgb=np.zeros((32, 64, 3), np.uint8), clip_bytes=b"\x00\x01",
    )
    store.close()
    return create_app(config), cid


def test_audio_route_serves_clip(tmp_path):
    app, cid = _app(tmp_path)
    client = app.test_client()
    resp = client.get(f"/audio/{cid}")
    assert resp.status_code == 200
    assert resp.mimetype == "audio/mp4"


def test_audio_route_404_when_no_clip(tmp_path):
    app, _ = _app(tmp_path)
    client = app.test_client()
    assert client.get("/audio/99999").status_code == 404


def test_index_payload_marks_audio_rows(tmp_path):
    app, cid = _app(tmp_path)
    client = app.test_client()
    data = client.get("/api/captures").get_json()
    row = next(c for c in data["captures"] if c["id"] == cid)
    assert row["source_kind"] == "audio"
    assert row["audio_url"].endswith(f"/audio/{cid}")
    # Bird species get a Merlin deep-link (American Robin -> eBird code 'amerob').
    assert row["species_url"] == "https://merlinbirds.org/species/amerob"


def test_photo_capture_has_no_species_link(tmp_path):
    """Only audio (bird) captures get a species link; photos (elk/deer/...) don't."""
    from wildlife.gallery.app import _species_url
    assert _species_url("elk", "reolink") is None
    assert _species_url("bird", "audio") is None  # generic, not an eBird species
    assert _species_url("Lesser Goldfinch", "audio") == "https://merlinbirds.org/species/lesgol"


def test_audio_payload_has_species_photo_url(tmp_path):
    app, cid = _app(tmp_path)
    data = app.test_client().get("/api/captures").get_json()
    row = next(c for c in data["captures"] if c["id"] == cid)
    # American Robin -> eBird code 'amerob' -> cached-photo route
    assert row["species_photo_url"].endswith("/species_photo/amerob")


def test_species_photo_route_caches_hits_and_misses(tmp_path, monkeypatch):
    """The route fetches once, caches hits on disk, and negative-caches misses."""
    import wildlife.gallery.app as gallery_app

    calls = []

    def _fake_fetch(common, timeout=6.0):
        calls.append(common)
        return b"\xff\xd8\xffFAKEJPEG" if common == "American Robin" else None

    monkeypatch.setattr(gallery_app, "_fetch_species_photo", _fake_fetch)
    client = _app(tmp_path)[0].test_client()

    # success -> 200 + cached on disk (fetched once even across two requests)
    assert client.get("/species_photo/amerob").status_code == 200
    assert client.get("/species_photo/amerob").mimetype == "image/jpeg"
    # miss -> 404, negative-cached (fetched once, second request served from marker)
    assert client.get("/species_photo/lesgol").status_code == 404
    assert client.get("/species_photo/lesgol").status_code == 404
    assert calls == ["American Robin", "Lesser Goldfinch"]
    # unknown code -> 404 without any network fetch
    assert client.get("/species_photo/notacode").status_code == 404
    assert calls == ["American Robin", "Lesser Goldfinch"]


def test_audio_route_404_for_photo_capture(tmp_path):
    """A normal photo capture has source_kind='reolink' and audio_path=NULL;
    /audio/<id> must 404 for it (not just for a nonexistent id)."""
    app, _ = _app(tmp_path)
    config = load_config(tmp_path / "config.yaml")
    store = Store(config.storage.db_path, config.storage.captures_dir)
    store.init_schema()
    photo_id = store.save_capture(
        camera_id="cam1",
        event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, 0, 1),
        frame=np.zeros((32, 64, 3), dtype=np.uint8),
        det=Detection(label="bird", confidence=0.9, box_xyxy=(0.0, 0.0, 10.0, 10.0), box_area_frac=0.1),
    )
    store.close()

    client = app.test_client()
    assert client.get(f"/audio/{photo_id}").status_code == 404


def test_index_html_renders_audio_card(tmp_path):
    app, cid = _app(tmp_path)
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'data-kind="audio"' in html
    assert f'data-audio="/audio/{cid}"' in html
    assert 'name="source_kind"' in html  # the Kind filter control is present
