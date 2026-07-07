"""Worker audio wiring (guarded: worker pulls in torch/cv2)."""

from __future__ import annotations

import pytest

pytest.importorskip("wildlife.worker")

from wildlife.config import Config  # noqa: E402
from wildlife import worker as W  # noqa: E402


def _config_dict(audio_enabled: bool):
    d = {
        "cameras": [{
            "id": "cam1", "host": "192.168.1.101", "username": "admin", "password": "x",
            "rtsp_main": "rtsp://{username}:{password}@{host}:554/main",
            "rtsp_sub": "rtsp://{username}:{password}@{host}:554/sub",
        }],
        "event_source": "reolink_native",
        "capture": {"burst_frames": 3, "burst_interval_ms": 100, "stream": "main",
                    "rtsp_timeout_s": 5, "max_concurrent": 1},
        "detection": {"model_path": "models/yolov8s.pt", "device": "cpu",
                      "animal_classes": ["bird"], "confidence_threshold": 0.5,
                      "min_box_area_frac": 0.01, "save_best_only": True},
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": "/tmp/wa_caps", "db_path": "/tmp/wa.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
    }
    if audio_enabled:
        d["audio"] = {"enabled": True, "use_geo_filter": False}
    return d


class _FakeSource:
    def __init__(self, *a, **k):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self, timeout=5.0):
        self.stopped = True


def test_audio_sources_start_only_when_enabled(monkeypatch):
    made = []
    monkeypatch.setattr(W, "AudioDetectionSource",
                        lambda *a, **k: made.append(_FakeSource()) or made[-1])
    monkeypatch.setattr(W, "AudioAnalyzer", lambda cfg: object())

    on = W._Worker(Config.model_validate(_config_dict(True)))
    on._audio_analyzer = object()  # pretend _setup built it
    # Set shutdown first so the primary reolink producer threads _start_producers
    # also launches exit immediately (their _produce loop is `while not shutdown`) —
    # otherwise they'd attempt a real RTSP/reolink connection. The audio-source loop
    # is not shutdown-gated, so the faked sources are still created + started.
    on._shutdown.set()
    on._start_producers()
    assert len(on._audio_sources) == 1 and on._audio_sources[0].started is True

    off = W._Worker(Config.model_validate(_config_dict(False)))
    off._shutdown.set()
    off._start_producers()
    assert off._audio_sources == []


def test_teardown_stops_audio_sources():
    w = W._Worker(Config.model_validate(_config_dict(True)))
    s = _FakeSource()
    w._audio_sources = [s]
    w._teardown()
    assert s.stopped is True


def test_audio_load_failure_degrades_gracefully(monkeypatch):
    def _boom(cfg):
        raise RuntimeError("no [audio] extra")

    monkeypatch.setattr(W, "AudioAnalyzer", _boom)
    monkeypatch.setattr(
        W, "Detector", lambda *a, **k: type("D", (), {"device_in_use": lambda self: "cpu"})()
    )

    class _FakeStore:
        def init_schema(self):
            pass

    monkeypatch.setattr(W, "Store", lambda *a, **k: _FakeStore())

    w = W._Worker(Config.model_validate(_config_dict(True)))
    w._setup()  # must NOT raise
    assert w._audio_analyzer is None
