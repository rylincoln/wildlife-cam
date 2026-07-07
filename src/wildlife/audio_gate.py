"""Repeat-confirmation + per-species cooldown for audio detections.

Pure, hardware-free logic (stdlib only), injected clock — the audio analyzer's
false-positive gate. Chaotic/broadband noise (wind) rarely reproduces the *same*
species across windows, so requiring N same-species hits within a short window is
the dominant defense; a per-species cooldown then throttles a persistently-calling
bird to one saved detection per cooldown period.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

__all__ = ["RepeatConfirmer"]


class RepeatConfirmer:
    """Decide when a species is *confirmed* (fire a save) and not in cooldown.

    Parameters
    ----------
    min_confirmations:
        Number of same-species hits required within ``confirm_window_s`` to fire.
    confirm_window_s:
        Trailing window (seconds) over which hits are counted.
    cooldown_s:
        After a fire, suppress that species for this many seconds.
    """

    def __init__(
        self, min_confirmations: int, confirm_window_s: float, cooldown_s: float
    ) -> None:
        self._min = int(min_confirmations)
        self._window_s = float(confirm_window_s)
        self._cooldown_s = float(cooldown_s)
        self._hits: dict[str, deque[datetime]] = defaultdict(deque)
        self._last_fired: dict[str, datetime] = {}

    def offer(self, species: str, confidence: float, now: datetime) -> bool:
        """Record a hit for ``species``; return True when it fires a save."""
        last = self._last_fired.get(species)
        if last is not None and (now - last).total_seconds() < self._cooldown_s:
            return False  # cooling down; don't even accumulate

        hits = self._hits[species]
        hits.append(now)
        # Evict hits older than the trailing window.
        cutoff = now
        while hits and (cutoff - hits[0]).total_seconds() > self._window_s:
            hits.popleft()

        if len(hits) >= self._min:
            self._last_fired[species] = now
            hits.clear()
            return True
        return False
