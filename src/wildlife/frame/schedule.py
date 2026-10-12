"""Night-dimming window for the /frame kiosk.

Pure functions over :class:`datetime.time` so the wrap-past-midnight logic is
trivially testable. Times are validated ``HH:MM`` strings from :class:`DimConfig`.
"""

from __future__ import annotations

from datetime import time

__all__ = ["in_window"]


def _parse(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def in_window(now: time, start: str, end: str) -> bool:
    """True when ``now`` is inside ``[start, end)``; wraps midnight when start > end."""
    begin, finish = _parse(start), _parse(end)
    if begin == finish:
        return False
    if begin < finish:
        return begin <= now < finish
    return now >= begin or now < finish
