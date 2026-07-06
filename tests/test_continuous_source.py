"""Temporal-logic tests for ContinuousMotionEventSource (hardware-free)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from wildlife.config import ContinuousConfig
from wildlife.events.continuous_motion import EVENT_KIND, ContinuousMotionEventSource


def _result(motion=False, scene_change=False, area_frac=0.0):
    return SimpleNamespace(motion=motion, scene_change=scene_change, area_frac=area_frac)


def _source(**cc_overrides):
    cc = ContinuousConfig(**cc_overrides)
    config = SimpleNamespace(
        continuous=cc, livestream=SimpleNamespace(rtsp_listen=":8554")
    )
    camera = SimpleNamespace(id="cam1", motion_mask=None)
    return ContinuousMotionEventSource(camera, config)


def test_event_kind_constant():
    assert EVENT_KIND == "motion_continuous"


def test_rising_edge_emits_once():
    src = _source(warmup_s=0, refractory_s=0)
    assert src._consider(_result(motion=False), 0.0, datetime(2026, 7, 6, 12)) is None
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) == "emit"
    # sustained motion (no new rising edge) does not re-emit
    assert src._consider(_result(motion=True), 2.0, datetime(2026, 7, 6, 12)) is None


def test_warmup_suppresses_emit():
    src = _source(warmup_s=10, refractory_s=0)
    src._warmup_until_mono = 100.0
    assert src._consider(_result(motion=True), 50.0, datetime(2026, 7, 6, 12)) is None
    # after warmup, sustained motion is not a rising edge -> a fresh edge is needed
    src._consider(_result(motion=False), 101.0, datetime(2026, 7, 6, 12))
    assert src._consider(_result(motion=True), 102.0, datetime(2026, 7, 6, 12)) == "emit"


def test_refractory_suppresses_second_emit():
    src = _source(warmup_s=0, refractory_s=8)
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) == "emit"
    src._consider(_result(motion=False), 2.0, datetime(2026, 7, 6, 12))
    # rising edge within refractory window -> suppressed
    assert src._consider(_result(motion=True), 5.0, datetime(2026, 7, 6, 12)) is None
    src._consider(_result(motion=False), 6.0, datetime(2026, 7, 6, 12))
    # rising edge after refractory window -> emits
    assert src._consider(_result(motion=True), 12.0, datetime(2026, 7, 6, 12)) == "emit"


def test_scene_change_requests_reset_and_suppresses():
    src = _source(warmup_s=0, refractory_s=8)
    action = src._consider(_result(motion=True, scene_change=True), 1.0, datetime(2026, 7, 6, 12))
    assert action == "reset"
    # the reset also arms refractory, so an immediate rising edge is suppressed
    assert src._consider(_result(motion=True), 2.0, datetime(2026, 7, 6, 12)) is None


def test_active_hours_gate_blocks_emit_outside_window():
    src = _source(warmup_s=0, refractory_s=0, active_hours="20:00-06:00")
    # 12:00 is outside the 20:00-06:00 window -> no emit even on a rising edge
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) is None
    # 22:00 is inside the window -> emits
    src2 = _source(warmup_s=0, refractory_s=0, active_hours="20:00-06:00")
    assert src2._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 22)) == "emit"


def test_within_active_hours_wraps_midnight():
    src = _source(active_hours="20:00-06:00")
    assert src._within_active_hours(datetime(2026, 7, 6, 23)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 3)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 12)) is False


def test_make_event_source_dispatches_continuous():
    from wildlife.events.base import make_event_source

    config = SimpleNamespace(
        continuous=ContinuousConfig(), livestream=SimpleNamespace(rtsp_listen=":8554")
    )
    camera = SimpleNamespace(id="cam1", motion_mask=None)
    src = make_event_source("continuous_motion", camera, config)
    assert isinstance(src, ContinuousMotionEventSource)
