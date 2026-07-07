"""MegaDetector second-pass wiring in the worker: the rescue throttle + person
override path (guarded: worker pulls in torch/cv2). The pure decision logic is
covered separately in test_megadetector.py; here we exercise _megadetector_pass
with a fake MD so no model loads."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pytest.importorskip("wildlife.worker")

from wildlife.config import Config  # noqa: E402
from wildlife.models import Detection  # noqa: E402
from wildlife.worker import _Worker  # noqa: E402


class _FakeMD:
    """Stand-in MegaDetector: returns canned detections and counts infer calls."""

    def __init__(self, dets):
        self._dets = dets
        self.calls = 0

    def infer(self, _frame):
        self.calls += 1
        return list(self._dets)

    def device_in_use(self):
        return "cpu"


def _config_dict(md: dict, *, save_best_only: bool = False):
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
            "burst_frames": 3, "burst_interval_ms": 100, "stream": "main",
            "rtsp_timeout_s": 5, "max_concurrent": 1,
        },
        "detection": {
            "model_path": "models/yolov8s.pt", "device": "cpu",
            "animal_classes": ["bird"], "confidence_threshold": 0.5,
            "min_box_area_frac": 0.01, "save_best_only": save_best_only,
        },
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": "/tmp/wmd_caps", "db_path": "/tmp/wmd.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
        "megadetector": md,
    }


def _worker(md: dict, fake: _FakeMD, *, save_best_only: bool = False) -> _Worker:
    w = _Worker(Config.model_validate(_config_dict(md, save_best_only=save_best_only)))
    w._megadetector = fake  # bypass _setup / model load
    return w


def _det(label, conf, box=(100.0, 100.0, 200.0, 500.0), area=0.05) -> Detection:
    return Detection(label=label, confidence=conf, box_xyxy=box, box_area_frac=area)


def test_rescue_cooldown_throttles_back_to_back_empty_events():
    """Two consecutive nothing-kept events within the cooldown run MD only once."""
    fake = _FakeMD([_det("animal", 0.6, area=0.08)])
    w = _worker(
        {"enabled": True, "rescue_misses": True, "person_override": False,
         "suppress_false_positives": False, "rescue_cooldown_s": 60, "classes": ["animal"]},
        fake,
    )
    # First empty event -> rescue runs (MD inference #1), records the run time.
    _bd, _bf, positives, _r = w._megadetector_pass([object()], None, None, [], False, "cam1")
    assert fake.calls == 1 and len(positives) == 1 and positives[0][1].label == "animal"
    # Second empty event immediately after -> throttled, no second inference.
    _bd, _bf, positives, _r = w._megadetector_pass([object()], None, None, [], False, "cam1")
    assert fake.calls == 1 and positives == []


def test_rescue_runs_again_after_cooldown_elapses():
    fake = _FakeMD([_det("animal", 0.6, area=0.08)])
    w = _worker(
        {"enabled": True, "rescue_misses": True, "rescue_cooldown_s": 30, "classes": ["animal"]},
        fake,
    )
    # Pretend the last rescue was 120s ago -> cooldown elapsed -> MD runs.
    w._md_last_rescue["cam1"] = datetime.now() - timedelta(seconds=120)
    _bd, _bf, positives, _r = w._megadetector_pass([object()], None, None, [], False, "cam1")
    assert fake.calls == 1 and len(positives) == 1


def test_person_override_on_kept_event_is_not_throttled():
    """Override runs on every kept event regardless of the rescue cooldown."""
    fake = _FakeMD([_det("person", 0.95)])
    w = _worker(
        {"enabled": True, "person_override": True, "rescue_misses": True,
         "rescue_cooldown_s": 9999, "classes": ["person", "animal"]},
        fake,
    )
    # Even with a very-recent rescue timestamp, a KEPT event still runs MD.
    w._md_last_rescue["cam1"] = datetime.now()
    bird = _det("bird", 0.7)
    for _ in range(2):
        _bd, _bf, positives, reason = w._megadetector_pass(
            [object()], None, None, [("f", bird)], False, "cam1"
        )
        assert positives[0][1].label == "person" and "person" in reason
    assert fake.calls == 2  # not throttled


def test_md_rescue_ready_helper():
    fake = _FakeMD([])
    w = _worker({"enabled": True, "rescue_cooldown_s": 30}, fake)
    now = datetime.now()
    assert w._md_rescue_ready("cam1", now, 0) is True  # 0 = no throttle
    assert w._md_rescue_ready("cam1", now, 30) is True  # never run before
    w._md_last_rescue["cam1"] = now
    assert w._md_rescue_ready("cam1", now, 30) is False  # just ran
    assert w._md_rescue_ready("cam1", now + timedelta(seconds=31), 30) is True  # elapsed
