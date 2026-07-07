"""Gallery audio endpoint + source_kind serialization/filter tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask")

from datetime import datetime  # noqa: E402

from wildlife.config import load_config  # noqa: E402
from wildlife.gallery.app import create_app  # noqa: E402
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
