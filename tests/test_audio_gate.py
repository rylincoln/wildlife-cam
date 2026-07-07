"""Tests for the pure RepeatConfirmer (no hardware)."""

from __future__ import annotations

from datetime import datetime, timedelta

from wildlife.audio_gate import RepeatConfirmer


def _t(sec: float) -> datetime:
    return datetime(2026, 7, 6, 12, 0, 0) + timedelta(seconds=sec)


def test_fires_only_after_min_confirmations_within_window():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False   # 1st hit
    assert rc.offer("robin", 0.9, _t(2)) is True     # 2nd within window -> fire


def test_single_hit_does_not_fire():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False


def test_hits_outside_window_do_not_accumulate():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=10, cooldown_s=0)
    assert rc.offer("robin", 0.9, _t(0)) is False
    # 20s later the first hit has aged out of the 10s window -> still only 1 in-window
    assert rc.offer("robin", 0.9, _t(20)) is False
    assert rc.offer("robin", 0.9, _t(21)) is True    # now 2 within the window


def test_cooldown_suppresses_refire():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False
    assert rc.offer("robin", 0.9, _t(1)) is True     # fire, arm 30s cooldown
    assert rc.offer("robin", 0.9, _t(5)) is False    # within cooldown
    assert rc.offer("robin", 0.9, _t(10)) is False   # still cooling down
    # after cooldown, a fresh pair fires again
    assert rc.offer("robin", 0.9, _t(40)) is False
    assert rc.offer("robin", 0.9, _t(41)) is True


def test_species_tracked_independently():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False
    assert rc.offer("jay", 0.9, _t(1)) is False       # different species, own count
    assert rc.offer("robin", 0.9, _t(2)) is True      # robin reaches 2
    assert rc.offer("jay", 0.9, _t(3)) is True         # jay reaches 2 independently
