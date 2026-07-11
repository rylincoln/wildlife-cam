"""AudioDetectionSource orchestration tests (no ffmpeg/birdnet)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from wildlife.config import AudioConfig
from wildlife.events.audio_detection import AudioDetectionSource, HOP_SAMPLES, WIN_SAMPLES


def _source(analyze=None, **audio_overrides):
    cfg = AudioConfig(use_geo_filter=False, **audio_overrides)
    config = SimpleNamespace(audio=cfg, livestream=SimpleNamespace(rtsp_listen=":8554"))
    camera = SimpleNamespace(id="cam1")
    if analyze is None:
        analyze = lambda pcm: [("American Robin", 0.9)]
    analyzer = SimpleNamespace(analyze=analyze)
    store = SimpleNamespace(saved=[], deleted=[], _next_id=0)

    def _save(**kw):
        store._next_id += 1
        store.saved.append({**kw, "id": store._next_id})
        return store._next_id

    store.save_audio_capture = _save
    store.delete = lambda cid: store.deleted.append(cid) or True
    return AudioDetectionSource(camera, config, analyzer, store), store


def test_window_hop_constants():
    assert WIN_SAMPLES == 144000
    assert HOP_SAMPLES == 144000  # 3 s hop (no overlap) so the warm session keeps up


def test_confirmed_detection_saves_audio_capture(monkeypatch):
    # render_spectrogram is patched to avoid heavy work; assert a save happens on
    # the SECOND window (min_confirmations defaults to 2).
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(min_confirmations=2, dedup_window_s=0)
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
    src, store = _source(active_hours="20:00-06:00", min_confirmations=1, dedup_window_s=0)
    # 12:00 is outside the window -> nothing analyzed/saved
    src._process_window(np.zeros(WIN_SAMPLES, np.float32), datetime(2026, 7, 6, 12, 0, 0))
    assert store.saved == []


def test_within_active_hours_wraps_midnight():
    src, _ = _source(active_hours="20:00-06:00")
    assert src._within_active_hours(datetime(2026, 7, 6, 23, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 3, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 12, 0)) is False


def test_species_hours_drops_out_of_window(monkeypatch):
    # A diurnal species (default analyzer = American Robin) gated to daytime must
    # NEVER save from a 4 a.m. window — the drop happens before the confirmer, so
    # even many repeats (as insect noise would produce) can't clear min_confirmations.
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(species_hours={"American Robin": "05:00-21:00"}, min_confirmations=2, dedup_window_s=0)
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")
    win = np.zeros(WIN_SAMPLES, np.float32)
    for s in range(6):
        src._process_window(win, datetime(2026, 7, 6, 4, 0, s))
    assert store.saved == []


def test_species_hours_allows_in_window(monkeypatch):
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(species_hours={"American Robin": "05:00-21:00"}, min_confirmations=2, dedup_window_s=0)
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")
    win = np.zeros(WIN_SAMPLES, np.float32)
    src._process_window(win, datetime(2026, 7, 6, 9, 0, 0))
    src._process_window(win, datetime(2026, 7, 6, 9, 0, 2))
    assert len(store.saved) == 1  # inside the window -> normal confirm+save


def test_unlisted_species_runs_24_7(monkeypatch):
    # No species_hours -> a genuinely nocturnal caller (owl/nighthawk) still saves at
    # 4 a.m. This is why the gate is per-species, not a global night-mute.
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(min_confirmations=1, dedup_window_s=0)
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")
    src._process_window(np.zeros(WIN_SAMPLES, np.float32), datetime(2026, 7, 6, 4, 0, 0))
    assert len(store.saved) == 1


def test_species_within_hours_wrap_and_default():
    src, _ = _source(species_hours={"Great Horned Owl": "20:00-06:00"})
    assert src._species_within_hours("Great Horned Owl", datetime(2026, 7, 6, 23, 0)) is True
    assert src._species_within_hours("Great Horned Owl", datetime(2026, 7, 6, 12, 0)) is False
    # A species with no configured window is always allowed.
    assert src._species_within_hours("American Robin", datetime(2026, 7, 6, 3, 0)) is True


def test_keeps_best_sample_and_replaces_within_window(monkeypatch):
    # A calling bout collapses to ONE capture: the clearest (highest-SNR) sample.
    # Ranking is by clip SNR, so drive signal_quality directly; confidence is fixed.
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    snr = [5.0]  # dB
    monkeypatch.setattr(mod, "signal_quality", lambda pcm, sr=48000: snr[0])
    src, store = _source(
        analyze=lambda pcm: [("American Robin", 0.7)],
        min_confirmations=1, dedup_window_s=900,
    )
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")
    win = np.zeros(WIN_SAMPLES, np.float32)

    # First detection opens the bout and saves (SNR 5) -> id 1.
    src._process_window(win, datetime(2026, 7, 6, 9, 0, 0))
    assert len(store.saved) == 1 and store.deleted == []
    assert store.saved[0]["event_ts"] == datetime(2026, 7, 6, 9, 0, 0)

    # A clearer clip (SNR 12) replaces it: earlier id deleted, better one saved -> id 2.
    snr[0] = 12.0
    src._process_window(win, datetime(2026, 7, 6, 9, 2, 0))
    assert store.deleted == [1] and len(store.saved) == 2
    # event_ts stays anchored at the bout start (9:00), not the replacement time (9:02).
    assert store.saved[-1]["event_ts"] == datetime(2026, 7, 6, 9, 0, 0)

    # A noisier clip (SNR 8 < 12) within the window is discarded: no new save/delete.
    snr[0] = 8.0
    src._process_window(win, datetime(2026, 7, 6, 9, 4, 0))
    assert store.deleted == [1] and len(store.saved) == 2

    # Past the 15-min window (anchored at the first save), a fresh bout saves anew
    # and anchors event_ts at the new bout's start.
    snr[0] = 6.0
    src._process_window(win, datetime(2026, 7, 6, 9, 20, 0))
    assert len(store.saved) == 3 and store.deleted == [1]  # nothing replaced (fresh window)
    assert store.saved[-1]["event_ts"] == datetime(2026, 7, 6, 9, 20, 0)
