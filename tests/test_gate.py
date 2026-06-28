"""Unit tests for :mod:`wildlife.gate` -- pure logic, no hardware.

All time-dependent behaviour is driven by an injected ``now`` built from
:class:`datetime.datetime` / :class:`datetime.timedelta`; the wall clock is
never consulted.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from wildlife.config import DetectionConfig
from wildlife.gate import Deduper, pick_best, select_keepers

# A fixed reference instant for deterministic time math.
T0 = datetime(2026, 6, 28, 12, 0, 0)


def _cfg(**overrides) -> DetectionConfig:
    """Build a DetectionConfig with sensible defaults, overridable per test."""
    base = dict(
        model_path="models/yolov8s.pt",
        device="cpu",
        animal_classes=["bird", "cat", "dog", "horse", "bear"],
        confidence_threshold=0.55,
        min_box_area_frac=0.01,
        save_best_only=True,
    )
    base.update(overrides)
    return DetectionConfig(**base)


# ---------------------------------------------------------------------------
# select_keepers
# ---------------------------------------------------------------------------


def test_select_keepers_empty_input(det_config):
    assert select_keepers([], det_config) == []


def test_select_keepers_class_filter(make_detection, det_config):
    """A non-animal label is dropped even with high confidence and big box."""
    person = make_detection("person", conf=0.99, area_frac=0.9)
    cat = make_detection("cat", conf=0.99, area_frac=0.9)
    assert select_keepers([person, cat], det_config) == [cat]


def test_select_keepers_confidence_threshold(make_detection):
    cfg = _cfg(confidence_threshold=0.55, min_box_area_frac=0.0)
    below = make_detection("cat", conf=0.5499, area_frac=0.2)
    at = make_detection("dog", conf=0.55, area_frac=0.2)  # >= keeps
    above = make_detection("bird", conf=0.9, area_frac=0.2)
    keepers = select_keepers([below, at, above], cfg)
    assert keepers == [at, above]


def test_select_keepers_box_area_threshold(make_detection):
    cfg = _cfg(min_box_area_frac=0.01, confidence_threshold=0.0)
    speck = make_detection("cat", conf=0.9, area_frac=0.009)
    at = make_detection("dog", conf=0.9, area_frac=0.01)  # >= keeps
    big = make_detection("bird", conf=0.9, area_frac=0.5)
    keepers = select_keepers([speck, at, big], cfg)
    assert keepers == [at, big]


def test_select_keepers_combined_and_order_preserved(make_detection, det_config):
    """Only detections passing all three gates survive, in input order."""
    d_ok1 = make_detection("bird", conf=0.8, area_frac=0.2)
    d_bad_class = make_detection("car", conf=0.95, area_frac=0.5)
    d_low_conf = make_detection("dog", conf=0.3, area_frac=0.5)
    d_small = make_detection("bear", conf=0.95, area_frac=0.005)
    d_ok2 = make_detection("horse", conf=0.6, area_frac=0.05)
    dets = [d_ok1, d_bad_class, d_low_conf, d_small, d_ok2]
    assert select_keepers(dets, det_config) == [d_ok1, d_ok2]


# ---------------------------------------------------------------------------
# pick_best
# ---------------------------------------------------------------------------


def test_pick_best_empty_returns_none():
    assert pick_best([]) is None


def test_pick_best_single(make_detection):
    only = make_detection("cat", conf=0.7)
    assert pick_best([only]) is only


def test_pick_best_picks_max_confidence(make_detection):
    a = make_detection("cat", conf=0.6)
    b = make_detection("dog", conf=0.92)
    c = make_detection("bird", conf=0.81)
    assert pick_best([a, b, c]) is b


# ---------------------------------------------------------------------------
# Deduper -- cooldown
# ---------------------------------------------------------------------------


def test_deduper_allows_first_event():
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    assert d.should_process("north_field", T0) is True


def test_deduper_should_process_has_no_side_effects():
    """Calling should_process repeatedly (without mark) keeps returning True."""
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    assert d.should_process("north_field", T0) is True
    assert d.should_process("north_field", T0) is True
    assert d.should_process("north_field", T0 + timedelta(seconds=1)) is True


def test_deduper_cooldown_suppresses_within_window():
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    d.mark_saved("north_field", T0)
    # Anything strictly inside the 30s window is suppressed.
    assert d.should_process("north_field", T0 + timedelta(seconds=1)) is False
    assert d.should_process("north_field", T0 + timedelta(seconds=29)) is False


def test_deduper_cooldown_allows_at_and_after_window():
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    d.mark_saved("north_field", T0)
    # Exactly at the boundary the cooldown no longer suppresses (uses <).
    assert d.should_process("north_field", T0 + timedelta(seconds=30)) is True
    assert d.should_process("north_field", T0 + timedelta(seconds=31)) is True


def test_deduper_cooldown_is_per_camera():
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    d.mark_saved("north_field", T0)
    # A different camera is unaffected by north_field's cooldown.
    assert d.should_process("south_trail", T0 + timedelta(seconds=5)) is True
    # ...but north_field itself is still in cooldown.
    assert d.should_process("north_field", T0 + timedelta(seconds=5)) is False


def test_deduper_resaving_after_cooldown_restarts_window():
    d = Deduper(cooldown_s=30, max_burst_per_minute=20)
    d.mark_saved("north_field", T0)
    t1 = T0 + timedelta(seconds=30)
    assert d.should_process("north_field", t1) is True
    d.mark_saved("north_field", t1)
    # New cooldown window starts at t1.
    assert d.should_process("north_field", t1 + timedelta(seconds=10)) is False
    assert d.should_process("north_field", t1 + timedelta(seconds=30)) is True


# ---------------------------------------------------------------------------
# Deduper -- global burst cap
# ---------------------------------------------------------------------------


def test_deduper_burst_cap_blocks_across_cameras():
    """The burst cap is global: once the cap is hit, other cameras are blocked too."""
    d = Deduper(cooldown_s=0, max_burst_per_minute=3)
    # Three saves across distinct cameras at the same instant.
    d.mark_saved("cam0", T0)
    d.mark_saved("cam1", T0)
    d.mark_saved("cam2", T0)
    # A fourth, never-saved camera (no cooldown) is blocked by the global cap.
    assert d.should_process("cam3", T0) is False


def test_deduper_burst_cap_counts_marks_in_trailing_60s():
    d = Deduper(cooldown_s=0, max_burst_per_minute=3)
    d.mark_saved("cam0", T0)
    d.mark_saved("cam1", T0)
    d.mark_saved("cam2", T0)
    # Just under 60s: all three marks still count -> blocked.
    assert d.should_process("cam3", T0 + timedelta(seconds=59)) is False


def test_deduper_burst_marks_expire_after_60s():
    d = Deduper(cooldown_s=0, max_burst_per_minute=3)
    d.mark_saved("cam0", T0)
    d.mark_saved("cam1", T0)
    d.mark_saved("cam2", T0)
    assert d.should_process("cam3", T0) is False
    # At exactly +60s the three marks at T0 have expired (cutoff uses <=).
    assert d.should_process("cam3", T0 + timedelta(seconds=60)) is True


def test_deduper_burst_partial_expiry():
    """Only marks older than 60s drop off; newer ones still count."""
    d = Deduper(cooldown_s=0, max_burst_per_minute=2)
    d.mark_saved("cam0", T0)
    d.mark_saved("cam1", T0 + timedelta(seconds=10))
    # Two marks within the window -> blocked.
    assert d.should_process("cam2", T0 + timedelta(seconds=20)) is False
    # Advance so the T0 mark expires but the +10s mark remains: 1 < 2 -> allowed.
    now = T0 + timedelta(seconds=61)
    assert d.should_process("cam2", now) is True


def test_deduper_burst_cap_one_allows_single_then_blocks():
    d = Deduper(cooldown_s=0, max_burst_per_minute=1)
    assert d.should_process("cam0", T0) is True
    d.mark_saved("cam0", T0)
    # Cap of one is now exhausted for the whole minute.
    assert d.should_process("cam1", T0 + timedelta(seconds=30)) is False
    # After the mark ages out, room opens again.
    assert d.should_process("cam1", T0 + timedelta(seconds=60)) is True
