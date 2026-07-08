"""Worker dual-producer wiring (guarded: worker pulls in torch/cv2)."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("wildlife.worker")

from wildlife.config import Config  # noqa: E402
from wildlife.worker import _Worker  # noqa: E402


def _config_dict(continuous_enabled: bool):
    return {
        "cameras": [
            {
                "id": "cam1",
                "host": "192.168.1.101",
                "username": "admin",
                "password": "x",
                "rtsp_main": "rtsp://{username}:{password}@{host}:554/main",
                "rtsp_sub": "rtsp://{username}:{password}@{host}:554/sub",
            }
        ],
        "event_source": "reolink_native",
        "capture": {
            "burst_frames": 3,
            "burst_interval_ms": 100,
            "stream": "main",
            "rtsp_timeout_s": 5,
            "max_concurrent": 1,
        },
        "detection": {
            "model_path": "models/yolov8s.pt",
            "device": "cpu",
            "animal_classes": ["bird"],
            "confidence_threshold": 0.5,
            "min_box_area_frac": 0.01,
            "save_best_only": True,
        },
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": "/tmp/wc_caps", "db_path": "/tmp/wc.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
        "continuous": {"enabled": continuous_enabled},
    }


def _make_worker(continuous_enabled: bool) -> _Worker:
    return _Worker(Config.model_validate(_config_dict(continuous_enabled)))


class _FakeSource:
    """Stops the producer loop after one iteration; records close()."""

    def __init__(self, shutdown: threading.Event) -> None:
        self._sd = shutdown
        self.closed = False

    def stream(self):
        self._sd.set()  # make _produce exit after this pass
        return iter(())

    def close(self):
        self.closed = True


def test_produce_uses_composite_source_key(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    camera = worker._cameras["cam1"]
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(worker._shutdown),
    )
    worker._produce(camera, "reolink_native")
    worker._shutdown.clear()
    worker._produce(camera, "continuous_motion")
    assert set(worker._sources) == {"cam1:reolink_native", "cam1:continuous_motion"}


def test_teardown_closes_both_sources():
    worker = _make_worker(continuous_enabled=True)
    a, b = _FakeSource(worker._shutdown), _FakeSource(worker._shutdown)
    worker._sources = {"cam1:reolink_native": a, "cam1:continuous_motion": b}
    worker._teardown()
    assert a.closed and b.closed


def _stop_and_join(worker: _Worker) -> None:
    """Let the (daemon) producer threads exit and join them, so nothing leaks."""
    worker._shutdown.set()
    for t in worker._producer_threads:
        t.join(timeout=1.0)
        assert not t.is_alive()


def test_second_producer_starts_only_when_enabled(monkeypatch):
    # Each fake is bound to ITS worker's _shutdown so _produce exits after one
    # pass — otherwise the daemon threads spin and, once monkeypatch reverts,
    # would call the real make_event_source and open real RTSP sessions.
    on = _make_worker(continuous_enabled=True)
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(on._shutdown),
    )
    on._start_producers()
    assert len(on._producer_threads) == 2  # reolink + continuous
    _stop_and_join(on)

    off = _make_worker(continuous_enabled=False)
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(off._shutdown),
    )
    off._start_producers()
    assert len(off._producer_threads) == 1  # reolink only
    _stop_and_join(off)


from datetime import datetime  # noqa: E402

from wildlife.events.continuous_motion import EVENT_KIND  # noqa: E402
from wildlife.gate import Deduper  # noqa: E402
from wildlife.models import CameraEvent  # noqa: E402


def _prime_consumer(worker: _Worker) -> None:
    """Give _handle_event the non-None collaborators it asserts on."""
    worker._detector = object()
    worker._store = object()
    worker._deduper = Deduper(0, 10_000)  # always processes


def test_continuous_event_routes_burst_through_go2rtc(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    _prime_consumer(worker)
    captured = {}

    def _fake_grab(camera, n, interval, stream, timeout, rtsp_url=None):
        captured["url"] = rtsp_url
        return []  # empty -> _handle_event returns before touching detector/store

    monkeypatch.setattr("wildlife.worker.grab_burst", _fake_grab)
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind=EVENT_KIND)
    )
    assert captured["url"] == "rtsp://127.0.0.1:8554/cam1_main"


def test_reolink_event_uses_direct_burst(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    _prime_consumer(worker)
    captured = {}

    def _fake_grab(camera, n, interval, stream, timeout, rtsp_url=None):
        captured["url"] = rtsp_url
        return []

    monkeypatch.setattr("wildlife.worker.grab_burst", _fake_grab)
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind="animal")
    )
    assert captured["url"] is None


def test_continuous_capture_persists_source_kind(monkeypatch, tmp_path):
    import numpy as np

    from wildlife.models import Detection
    from wildlife.store import Store

    class _FakeDetector:
        def infer(self, _frame):
            return [Detection("bird", 0.95, (0.0, 0.0, 40.0, 40.0), 0.3)]

    worker = _make_worker(continuous_enabled=True)
    worker._detector = _FakeDetector()
    worker._store = Store(tmp_path / "c.db", tmp_path / "caps")
    worker._store.init_schema()
    worker._deduper = Deduper(0, 10_000)

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[20:60, 20:60] = 255  # non-blank content (a flat/zeros frame is now dropped as a hiccup)
    monkeypatch.setattr(
        "wildlife.worker.grab_burst",
        lambda *a, **k: [frame],
    )
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind=EVENT_KIND)
    )
    rows = worker._store.query(camera_id="cam1")
    assert rows and rows[0]["source_kind"] == "continuous"
    worker._store.close()
