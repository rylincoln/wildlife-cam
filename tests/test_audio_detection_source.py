"""AudioDetectionSource orchestration tests (no ffmpeg/birdnet)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from wildlife.config import AudioConfig
from wildlife.events.audio_detection import AudioDetectionSource, HOP_SAMPLES, WIN_SAMPLES


def _source(**audio_overrides):
    cfg = AudioConfig(use_geo_filter=False, **audio_overrides)
    config = SimpleNamespace(audio=cfg, livestream=SimpleNamespace(rtsp_listen=":8554"))
    camera = SimpleNamespace(id="cam1")
    analyzer = SimpleNamespace(analyze=lambda pcm: [("American Robin", 0.9)])
    store = SimpleNamespace(saved=[])
    store.save_audio_capture = lambda **kw: store.saved.append(kw) or 1
    return AudioDetectionSource(camera, config, analyzer, store), store


def test_window_hop_constants():
    assert WIN_SAMPLES == 144000
    assert HOP_SAMPLES == 72000


def test_confirmed_detection_saves_audio_capture(monkeypatch):
    # render_spectrogram is patched to avoid heavy work; assert a save happens on
    # the SECOND window (min_confirmations defaults to 2).
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(min_confirmations=2, cooldown_s=0)
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")  # skip ffmpeg
    win = np.zeros(WIN_SAMPLES, dtype=np.float32)
    src._process_window(win, datetime(2026, 7, 6, 6, 0, 0))
    assert store.saved == []                      # 1st hit: not confirmed
    src._process_window(win, datetime(2026, 7, 6, 6, 0, 2))
    assert len(store.saved) == 1                  # 2nd hit -> saved
    row = store.saved[0]
    assert row["species"] == "American Robin"
    assert row["source_kind"] == "audio"
    assert row["clip_bytes"] == b"clip"


def test_active_hours_gate_blocks_processing():
    src, store = _source(active_hours="20:00-06:00", min_confirmations=1, cooldown_s=0)
    # 12:00 is outside the window -> nothing analyzed/saved
    src._process_window(np.zeros(WIN_SAMPLES, np.float32), datetime(2026, 7, 6, 12, 0, 0))
    assert store.saved == []


def test_within_active_hours_wraps_midnight():
    src, _ = _source(active_hours="20:00-06:00")
    assert src._within_active_hours(datetime(2026, 7, 6, 23, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 3, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 12, 0)) is False
