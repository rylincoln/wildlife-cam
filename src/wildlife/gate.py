"""Pure gating logic for the wildlife detector.

This module decides which raw detections are worth keeping and enforces
save-rate discipline (per-camera cooldown plus a global burst cap). It contains
**no hardware dependencies** (no torch/cv2/ultralytics/numpy) and performs no
I/O, which makes every decision here fully unit-testable.

Time is always supplied by the caller via an injected ``now`` argument; this
module never reads the wall clock, so tests can drive it deterministically with
:class:`datetime.datetime` / :class:`datetime.timedelta`.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from wildlife.config import DetectionConfig
from wildlife.models import Detection

__all__ = ["select_keepers", "pick_best", "Deduper"]

# Width of the rolling window used for the global burst cap.
_BURST_WINDOW = timedelta(seconds=60)


def select_keepers(dets: list[Detection], cfg: DetectionConfig) -> list[Detection]:
    """Filter raw detections down to the ones worth saving.

    A detection is kept when *all* of the following hold:

    * its ``label`` is one of the configured ``animal_classes``;
    * its ``confidence`` is at least ``cfg.confidence_threshold``;
    * its ``box_area_frac`` is at least ``cfg.min_box_area_frac``.

    Input order is preserved among the kept detections.
    """
    animal_classes = set(cfg.animal_classes)
    threshold = cfg.confidence_threshold
    min_area = cfg.min_box_area_frac
    return [
        d
        for d in dets
        if d.label in animal_classes
        and d.confidence >= threshold
        and d.box_area_frac >= min_area
    ]


def pick_best(keepers: list[Detection]) -> Detection | None:
    """Return the highest-confidence detection, or ``None`` if there are none."""
    return max(keepers, key=lambda d: d.confidence) if keepers else None


class Deduper:
    """Suppress redundant saves via per-camera cooldown and a global burst cap.

    The deduper tracks, per camera, the timestamp of the last accepted save, and
    keeps a rolling deque of *all* recent save timestamps (across every camera)
    to enforce a hard ``max_burst_per_minute`` ceiling.

    All timestamps are passed in by the caller; the deduper never calls
    :func:`datetime.datetime.now` itself.
    """

    def __init__(self, cooldown_s: int, max_burst_per_minute: int) -> None:
        self._cooldown = timedelta(seconds=cooldown_s)
        self._max_burst_per_minute = max_burst_per_minute
        # Last accepted-save time, keyed by camera id.
        self._last_saved: dict[str, datetime] = {}
        # Timestamps of every recent save, across all cameras (oldest first).
        self._recent_marks: deque[datetime] = deque()

    def _prune(self, now: datetime) -> None:
        """Drop marks that have aged out of the trailing 60s window.

        A mark expires after exactly 60 seconds: a mark at ``now - 60s`` is no
        longer counted, while a mark strictly newer than that still is.
        """
        cutoff = now - _BURST_WINDOW
        marks = self._recent_marks
        while marks and marks[0] <= cutoff:
            marks.popleft()

    def should_process(self, camera_id: str, now: datetime) -> bool:
        """Return ``True`` when a save for ``camera_id`` is allowed at ``now``.

        Returns ``False`` if either:

        * this camera saved less than ``cooldown_s`` seconds ago, or
        * the number of saves in the trailing 60 seconds (across all cameras)
          has reached ``max_burst_per_minute``.

        This method is side-effect free with respect to the cooldown/last-save
        state: it only prunes expired burst marks, so repeated calls without an
        intervening :meth:`mark_saved` return a stable answer.
        """
        last = self._last_saved.get(camera_id)
        if last is not None and (now - last) < self._cooldown:
            return False
        self._prune(now)
        if len(self._recent_marks) >= self._max_burst_per_minute:
            return False
        return True

    def mark_saved(self, camera_id: str, now: datetime) -> None:
        """Record that a save happened for ``camera_id`` at ``now``.

        Updates the per-camera cooldown clock and appends to the global rolling
        window used for the burst cap.
        """
        self._last_saved[camera_id] = now
        self._prune(now)
        self._recent_marks.append(now)
