"""The worker's event inbox: at most one pending event per camera, newest wins.

A camera event is only useful while the scene still shows what triggered it: the
worker grabs its burst *after* taking the event off the inbox. The old FIFO queue
(256 deep, one consumer at ~5 s per event) filled every night with insect-
triggered motion, so each burst was taken ~25 minutes after its event and the
animals the cameras' own AI had flagged were long gone. Here a newer event for a
camera replaces the older one still waiting, so a burst is never more than one
service round behind the scene. Camera-AI ``animal`` events go first; otherwise
the camera that has waited longest does, so one busy camera can't starve another.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from wildlife.models import CameraEvent

__all__ = ["PRIORITY_KINDS", "EventMailbox"]

# Event kinds served ahead of plain motion: the camera's own AI saw an animal.
PRIORITY_KINDS = frozenset({"animal"})


@dataclass
class _Slot:
    event: CameraEvent
    waiting_since: float  # monotonic time the camera's oldest unserved event arrived


class EventMailbox:
    """Thread-safe latest-event-per-camera inbox (many producers, one consumer)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._slots: dict[str, _Slot] = {}

    def put(self, event: CameraEvent) -> CameraEvent | None:
        """Make ``event`` its camera's pending event; return the event it replaced.

        A replacement keeps the camera's place in line, and keeps an AI ``animal``
        flag already pending: the newer moment is grabbed, but at that priority.
        """
        with self._cond:
            slot = self._slots.get(event.camera_id)
            if slot is None:
                self._slots[event.camera_id] = _Slot(event, time.monotonic())
                self._cond.notify()
                return None
            old = slot.event
            if old.kind in PRIORITY_KINDS and event.kind not in PRIORITY_KINDS:
                event = replace(event, kind=old.kind)
            slot.event = event
            return old

    def get(self, timeout: float) -> CameraEvent | None:
        """Take the next event to handle, waiting up to ``timeout`` s; None if none came."""
        with self._cond:
            if not self._slots and not self._cond.wait_for(lambda: bool(self._slots), timeout):
                return None
            camera_id = min(
                self._slots,
                key=lambda c: (self._slots[c].event.kind not in PRIORITY_KINDS,
                               self._slots[c].waiting_since),
            )
            return self._slots.pop(camera_id).event

    def clear(self) -> int:
        """Discard every pending event; return how many there were."""
        with self._cond:
            n = len(self._slots)
            self._slots.clear()
            return n

    def __len__(self) -> int:
        with self._cond:
            return len(self._slots)
