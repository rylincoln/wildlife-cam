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
