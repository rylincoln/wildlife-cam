"""The worker's latest-event-per-camera inbox."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

from wildlife.event_mailbox import EventMailbox
from wildlife.models import CameraEvent

T0 = datetime(2026, 10, 11, 5, 35)


def _ev(cam: str, seconds: int, kind: str = "motion") -> CameraEvent:
    return CameraEvent(camera_id=cam, event_ts=T0 + timedelta(seconds=seconds), kind=kind)


def test_a_newer_event_replaces_the_one_still_waiting():
    box = EventMailbox()
    assert box.put(_ev("a", 0)) is None
    assert box.put(_ev("a", 5)) == _ev("a", 0)
    assert len(box) == 1
    assert box.get(timeout=0) == _ev("a", 5)  # the burst is taken for the newest moment
    assert box.get(timeout=0) is None


def test_camera_ai_animal_goes_first_and_survives_later_motion():
    box = EventMailbox()
    box.put(_ev("a", 0))
    box.put(_ev("b", 1, kind="animal"))
    box.put(_ev("b", 2))  # plain motion right after the AI flag
    got = box.get(timeout=0)
    assert (got.camera_id, got.kind, got.event_ts) == ("b", "animal", T0 + timedelta(seconds=2))
    assert box.get(timeout=0).camera_id == "a"


def test_cameras_take_turns_by_waiting_time_not_event_count():
    box = EventMailbox()
    box.put(_ev("a", 0))
    box.put(_ev("b", 1))
    for s in range(2, 50):  # a's insect storm keeps replacing its pending event
        box.put(_ev("a", s))
    assert box.get(timeout=0).camera_id == "a"  # waited longest
    assert box.get(timeout=0).camera_id == "b"


def test_get_wakes_when_an_event_arrives_and_clear_discards():
    box = EventMailbox()
    got = []
    t = threading.Thread(target=lambda: got.append(box.get(timeout=5)))
    t.start()
    box.put(_ev("a", 0))
    t.join(timeout=5)
    assert got == [_ev("a", 0)]
    box.put(_ev("a", 1))
    box.put(_ev("b", 1))
    assert box.clear() == 2 and len(box) == 0
