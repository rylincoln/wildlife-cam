"""Tests for the gallery's Server-Sent Events live-update stream (/api/stream).

The endpoint is an intentionally-infinite generator (it polls the DB forever),
so these tests drive it with ``run_wsgi_app(..., buffered=False)`` and pull a
bounded number of chunks rather than consuming ``response.data`` (which would
hang). Each ``yield`` in the view becomes one WSGI chunk, so a capture that
already exists at connect time is delivered on the first poll -- no real-time
sleep is needed to observe it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("flask")

from datetime import datetime  # noqa: E402

from werkzeug.test import EnvironBuilder, run_wsgi_app  # noqa: E402

from wildlife.config import load_config  # noqa: E402
from wildlife.gallery.app import create_app  # noqa: E402
from wildlife.models import Detection  # noqa: E402
from wildlife.store import Store  # noqa: E402


def _write_config(tmp_path, *, live_refresh_seconds: float = 0.05, max_live_streams: int = 12) -> str:
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
gallery: {{host: 0.0.0.0, port: 8080, page_size: 60, live_refresh_seconds: {live_refresh_seconds}, max_live_streams: {max_live_streams}}}
resource_guard: {{detect_every_nth_event: 1, max_burst_per_minute: 20}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)
    return str(cfg_path)


def _store(cfg_path) -> Store:
    config = load_config(cfg_path)
    store = Store(config.storage.db_path, config.storage.captures_dir)
    store.init_schema()
    return store


def _save_photo(store: Store, *, camera="cam1", label="bird", conf=0.9, minute=0) -> int:
    return store.save_capture(
        camera_id=camera,
        event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, minute, 0),
        frame=np.zeros((32, 64, 3), dtype=np.uint8),
        det=Detection(label=label, confidence=conf, box_xyxy=(0.0, 0.0, 10.0, 10.0), box_area_frac=0.1),
    )


def _read_stream(app, query_string, *, headers=None, stop=b"event: capture", max_chunks=40):
    """Open /api/stream and accumulate chunks until ``stop`` appears (or a cap).

    Returns ``(status, headers_dict, body_bytes)``. Always closes the underlying
    generator so its DB handle is released (exercising the finally: store.close()).
    """
    builder = EnvironBuilder(path="/api/stream", query_string=query_string, headers=headers or {})
    app_iter, status, resp_headers = run_wsgi_app(app, builder.get_environ(), buffered=False)
    acc = b""
    try:
        for i, chunk in enumerate(app_iter):
            acc += chunk
            if (stop is not None and stop in acc) or i + 1 >= max_chunks:
                break
    finally:
        close = getattr(app_iter, "close", None)
        if close:
            close()
    return status, dict(resp_headers), acc


def _capture_events(body: bytes) -> list[dict]:
    """Parse the ``data:`` JSON payloads out of an SSE body."""
    out = []
    for line in body.decode().splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:"):].strip()))
    return out


def test_stream_pushes_existing_capture_with_headers(tmp_path):
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    cid = _save_photo(store)
    store.close()
    app = create_app(load_config(cfg))

    status, headers, body = _read_stream(app, "since=0")
    assert status.startswith("200")
    assert headers.get("Content-Type", "").startswith("text/event-stream")
    assert headers.get("Cache-Control") == "no-cache"
    assert headers.get("X-Accel-Buffering") == "no"
    assert f"id: {cid}".encode() in body
    events = _capture_events(body)
    assert any(e["id"] == cid for e in events)


def test_stream_disabled_returns_404(tmp_path):
    cfg = _write_config(tmp_path, live_refresh_seconds=0)
    _store(cfg).close()
    app = create_app(load_config(cfg))
    # Disabled -> plain 404, no streaming, so the normal test client is fine.
    assert app.test_client().get("/api/stream?since=0").status_code == 404


def test_stream_since_watermark_skips_older(tmp_path):
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    older = _save_photo(store, minute=0)
    newer = _save_photo(store, minute=1)
    store.close()
    app = create_app(load_config(cfg))

    _status, _headers, body = _read_stream(app, f"since={older}")
    events = _capture_events(body)
    ids = [e["id"] for e in events]
    assert newer in ids
    assert older not in ids


def test_stream_respects_filters(tmp_path):
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    _save_photo(store, label="dog", camera="south", minute=0)
    bird = _save_photo(store, label="bird", camera="north", minute=1)
    store.close()
    app = create_app(load_config(cfg))

    _status, _headers, body = _read_stream(app, "since=0&label=bird")
    events = _capture_events(body)
    assert [e["id"] for e in events] == [bird]
    assert all(e["label"] == "bird" for e in events)


def test_stream_resumes_from_last_event_id_header(tmp_path):
    """Last-Event-ID (sent by the browser on reconnect) overrides ?since=."""
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    first = _save_photo(store, minute=0)
    second = _save_photo(store, minute=1)
    store.close()
    app = create_app(load_config(cfg))

    # since=0 would replay `first`, but the header watermark supersedes it.
    _status, _headers, body = _read_stream(
        app, "since=0", headers={"Last-Event-ID": str(first)}
    )
    ids = [e["id"] for e in _capture_events(body)]
    assert second in ids
    assert first not in ids


def test_stream_out_of_range_since_does_not_crash(tmp_path):
    """A watermark past SQLite's 64-bit range must clamp to 'from now', not raise
    OverflowError inside the generator (which would crash-loop on reconnect)."""
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    _save_photo(store)  # one existing row
    store.close()
    app = create_app(load_config(cfg))

    # Huge since -> clamp to max_id; loop reaches the heartbeat without raising.
    _status, _headers, body = _read_stream(app, "since=99999999999999999999999999", stop=b": ping")
    assert b": ping" in body
    assert b"event: capture" not in body  # nothing replayed


def test_stream_negative_since_does_not_replay_table(tmp_path):
    """?since=-1 would mean id > -1 (the whole table); it must clamp to 'from now'."""
    cfg = _write_config(tmp_path)
    store = _store(cfg)
    _save_photo(store)
    _save_photo(store, minute=1)
    store.close()
    app = create_app(load_config(cfg))

    _status, _headers, body = _read_stream(app, "since=-1", stop=b": ping")
    assert b": ping" in body
    assert b"event: capture" not in body  # existing rows are NOT replayed


def test_stream_cap_returns_503_when_slots_exhausted(tmp_path):
    """Beyond max_live_streams, /api/stream refuses with 503 instead of piling on
    another parked thread + connection."""
    cfg = _write_config(tmp_path, max_live_streams=1)
    _store(cfg).close()
    app = create_app(load_config(cfg))

    # Hold the single slot by opening (and starting) one stream, without closing.
    b1 = EnvironBuilder(path="/api/stream", query_string="since=0")
    it1, _status1, _h1 = run_wsgi_app(app, b1.get_environ(), buffered=False)
    next(iter(it1))  # run the generator body so the slot is actually held
    try:
        b2 = EnvironBuilder(path="/api/stream", query_string="since=0")
        _it2, status2, _h2 = run_wsgi_app(app, b2.get_environ(), buffered=False)
        assert status2.startswith("503")
    finally:
        close = getattr(it1, "close", None)
        if close:
            close()

    # Slot was released on close -> a fresh stream is accepted again.
    _status3, headers3, _body3 = _read_stream(app, "since=0", stop=b": ping")
    assert headers3.get("Content-Type", "").startswith("text/event-stream")


def test_stream_terminates_at_lifetime_cap(tmp_path, monkeypatch):
    """The stream ends itself after _STREAM_MAX_SECONDS so the client reconnects
    (re-auth + frees the HTTP/1.1 slot). Without the cap this iterator is infinite."""
    import wildlife.gallery.app as appmod

    monkeypatch.setattr(appmod, "_STREAM_MAX_SECONDS", 0.12)  # ~2-3 polls at 0.05s
    cfg = _write_config(tmp_path)
    _store(cfg).close()
    app = create_app(load_config(cfg))

    builder = EnvironBuilder(path="/api/stream", query_string="since=0")
    app_iter, status, _headers = run_wsgi_app(app, builder.get_environ(), buffered=False)
    chunks = list(app_iter)  # would hang forever if the loop never terminated
    assert status.startswith("200")
    assert any(b"retry:" in c or b"ping" in c for c in chunks)
