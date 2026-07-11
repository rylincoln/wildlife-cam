"""Per-species dedup for audio detections: confirm the bird, then keep only the
single best-quality sample within a consolidation window.

Pure, hardware-free logic (stdlib only), injected clock. Two jobs:

1. **Reject one-off noise.** Require ``min_confirmations`` same-species hits within
   ``confirm_window_s`` before a species is saved at all. Chaotic/broadband noise
   (wind) rarely reproduces the *same* species across windows; real birds repeat.

2. **Collapse a calling bout to ONE capture.** The first confirmed save opens a
   ``dedup_window_s`` window. Within it, a *higher-quality* detection REPLACES the
   saved one (the best sample wins) and lower/equal ones are dropped — so a bird
   calling for minutes yields one good recording, not dozens. After the window
   expires, the next detection re-confirms and opens a fresh window.

The ranking ``score`` is supplied by the caller (a band-limited SNR of the clip: the
clearest-sounding sample wins), decoupled from the BirdNET confidence that gates
*which* detections count as the bird upstream. The window is fixed from the first
save (``dedup_window_s`` long); a bout that outlasts it simply starts a new sample.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

__all__ = ["SpeciesDeduper", "SaveDecision"]


@dataclass(frozen=True)
class SaveDecision:
    """Outcome for one offered detection.

    ``save`` is True when the caller should render/encode/save this window. When
    ``replace_id`` is set, the caller must delete that superseded capture (a
    lower-quality earlier sample of the same bout) after the new save succeeds.
    ``bout_start`` is the anchor time of the bout this sample belongs to (the same
    value across every replacement in a window), so the saved row's event_ts stays
    stable at when the bird *started* calling rather than drifting to the best
    sample's time. It is None for a discard.
    """

    save: bool
    replace_id: int | None = None
    bout_start: datetime | None = None


_DISCARD = SaveDecision(save=False)


@dataclass
class _Window:
    start: datetime      # anchor: when this bout's first sample was saved
    best_score: float    # quality score (SNR) of the currently-saved best sample
    best_id: int         # capture id of the currently-saved best sample


class SpeciesDeduper:
    """Decide, per offered detection, whether to save it (and what it replaces).

    Args:
        min_confirmations: Same-species hits required within ``confirm_window_s``
            to open a new bout (the noise gate).
        confirm_window_s: Trailing window (seconds) over which hits are counted.
        dedup_window_s: Consolidation window (seconds). Within it, only a strictly
            higher-``score`` detection is kept, replacing the prior best. ``0``
            disables consolidation (every confirmed detection saves).
    """

    def __init__(
        self, min_confirmations: int, confirm_window_s: float, dedup_window_s: float
    ) -> None:
        self._min = int(min_confirmations)
        self._confirm_window_s = float(confirm_window_s)
        self._dedup_window_s = float(dedup_window_s)
        self._hits: dict[str, deque[datetime]] = defaultdict(deque)
        self._windows: dict[str, _Window] = {}

    def _open_window(self, species: str, now: datetime) -> _Window | None:
        """Return the species' still-open window, evicting it if it has expired."""
        w = self._windows.get(species)
        if w is None:
            return None
        if (now - w.start).total_seconds() >= self._dedup_window_s:
            del self._windows[species]  # expired: a fresh bout must re-confirm
            return None
        return w

    def offer(self, species: str, score: float, now: datetime) -> SaveDecision:
        """Decide what to do with a detection given its quality ``score``.

        Mutates only the confirmation counter. The saved-window state changes in
        :meth:`commit`, which the caller invokes *after* a successful save — so a
        render/encode failure leaves the prior best untouched.
        """
        w = self._open_window(species, now)
        if w is not None:
            # Bout already confirmed and saved: compete on quality only. The bout
            # keeps its original anchor so event_ts doesn't drift to this sample.
            if score > w.best_score:
                return SaveDecision(save=True, replace_id=w.best_id, bout_start=w.start)
            return _DISCARD

        # No open window: require repeat-confirmation to open one.
        hits = self._hits[species]
        hits.append(now)
        while hits and (now - hits[0]).total_seconds() > self._confirm_window_s:
            hits.popleft()
        if len(hits) >= self._min:
            hits.clear()
            return SaveDecision(save=True, replace_id=None, bout_start=now)
        return _DISCARD

    def commit(self, species: str, capture_id: int, score: float, now: datetime) -> None:
        """Record a successful save as this species' current best.

        A replace within an open window keeps the original anchor (so the window
        stays fixed at ``dedup_window_s``); otherwise a fresh window anchors here.
        """
        w = self._windows.get(species)
        if w is not None and (now - w.start).total_seconds() < self._dedup_window_s:
            w.best_score = score
            w.best_id = capture_id
        else:
            self._windows[species] = _Window(start=now, best_score=score, best_id=capture_id)
