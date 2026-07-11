"""Tests for the pure SpeciesDeduper (no hardware)."""

from __future__ import annotations

from datetime import datetime, timedelta

from wildlife.audio_gate import SpeciesDeduper


def _t(sec: float) -> datetime:
    return datetime(2026, 7, 6, 12, 0, 0) + timedelta(seconds=sec)


# --- noise gate: min_confirmations to open a bout -------------------------

def test_fires_only_after_min_confirmations_within_window():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=0)
    assert d.offer("robin", 0.9, _t(0)).save is False   # 1st hit
    dec = d.offer("robin", 0.9, _t(2))                    # 2nd within window -> fire
    assert dec.save is True and dec.replace_id is None


def test_single_hit_does_not_fire():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=0)
    assert d.offer("robin", 0.9, _t(0)).save is False


def test_hits_outside_window_do_not_accumulate():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=10, dedup_window_s=0)
    assert d.offer("robin", 0.9, _t(0)).save is False
    # 20s later the first hit has aged out of the 10s window -> still only 1 in-window
    assert d.offer("robin", 0.9, _t(20)).save is False
    assert d.offer("robin", 0.9, _t(21)).save is True    # now 2 within the window


def test_species_tracked_independently():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=0)
    assert d.offer("robin", 0.9, _t(0)).save is False
    assert d.offer("jay", 0.9, _t(1)).save is False       # different species, own count
    assert d.offer("robin", 0.9, _t(2)).save is True      # robin reaches 2
    assert d.offer("jay", 0.9, _t(3)).save is True         # jay reaches 2 independently


# --- keep-best consolidation window ---------------------------------------

def _open_bout(d, species="wren", conf=0.5, t0=0.0):
    """Confirm a species (2 quick hits) so its dedup window opens; return the save t."""
    assert d.offer(species, conf, _t(t0)).save is False
    dec = d.offer(species, conf, _t(t0 + 1))
    assert dec.save is True and dec.replace_id is None
    assert dec.bout_start == _t(t0 + 1)                    # fresh bout anchors at the opening save
    d.commit(species, capture_id=100, score=conf, now=_t(t0 + 1))
    return t0 + 1


def test_better_detection_replaces_within_window():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=900)
    _open_bout(d, conf=0.5)                                # saved id=100 @ ~t1 (anchor t1)
    dec = d.offer("wren", 0.8, _t(120))                    # higher conf, still in window
    assert dec.save is True and dec.replace_id == 100      # replace the earlier sample
    assert dec.bout_start == _t(1)                         # anchor preserved, not _t(120)
    d.commit("wren", capture_id=101, score=0.8, now=_t(120))
    # A further, even-better call replaces the new best (id 101).
    dec2 = d.offer("wren", 0.95, _t(240))
    assert dec2.save is True and dec2.replace_id == 101
    assert dec2.bout_start == _t(1)                        # still the original anchor


def test_worse_or_equal_detection_discarded_within_window():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=900)
    _open_bout(d, conf=0.7)                                # best = 0.7
    assert d.offer("wren", 0.6, _t(120)).save is False     # lower -> discard
    assert d.offer("wren", 0.7, _t(180)).save is False     # equal -> discard (strictly greater wins)


def test_new_window_after_expiry_reconfirms_and_saves_fresh():
    d = SpeciesDeduper(min_confirmations=2, confirm_window_s=15, dedup_window_s=900)
    _open_bout(d, conf=0.6)                                # window opens ~t1
    # Past the 900s window a single hit must NOT save (bout must re-confirm).
    assert d.offer("wren", 0.9, _t(1000)).save is False
    dec = d.offer("wren", 0.9, _t(1002))                   # 2nd hit -> new bout
    assert dec.save is True and dec.replace_id is None      # fresh capture, nothing replaced


def test_window_anchored_at_first_save_not_extended_by_replace():
    # Replacing keeps the ORIGINAL anchor, so the window is a fixed dedup_window_s
    # from the first save — not rolling.
    d = SpeciesDeduper(min_confirmations=1, confirm_window_s=15, dedup_window_s=100)
    assert d.offer("wren", 0.4, _t(0)).save is True         # min_confirmations=1 -> opens at t0
    d.commit("wren", 100, 0.4, _t(0))
    dec = d.offer("wren", 0.5, _t(60))                       # replace within window
    assert dec.replace_id == 100
    d.commit("wren", 101, 0.5, _t(60))                       # anchor stays t0, not t60
    # t120 is 120s past the t0 anchor (>100) -> window expired, needs a fresh save.
    dec2 = d.offer("wren", 0.9, _t(120))
    assert dec2.save is True and dec2.replace_id is None


def test_dedup_window_zero_saves_every_confirmed_detection():
    d = SpeciesDeduper(min_confirmations=1, confirm_window_s=15, dedup_window_s=0)
    dec1 = d.offer("wren", 0.4, _t(0))
    assert dec1.save is True and dec1.replace_id is None
    d.commit("wren", 100, 0.4, _t(0))
    dec2 = d.offer("wren", 0.3, _t(1))                       # even a LOWER conf saves fresh
    assert dec2.save is True and dec2.replace_id is None
