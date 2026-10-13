"""Latest camera sighting + recent bird IDs for the /frame kiosk.

Pure shaping over capture dicts as produced by the gallery's serializer (newest
first, the order :meth:`wildlife.store.Store.query` returns): collapses camera
captures to one per burst event (its most confident frame; a burst can save
several), de-duplicates bird detections by species (keeping each species'
newest clip), and formats labels, times and confidences for display so the page
JS only places text.
"""

from __future__ import annotations

from datetime import datetime

__all__ = ["CAMERA_KINDS", "build_wildlife", "pretty_label", "sightings_page", "time_text"]

CAMERA_KINDS = ("reolink", "continuous")


def pretty_label(label: str | None) -> str:
    """'red_fox' -> 'Red fox'; already-formatted names ("Steller's Jay") pass through."""
    if not label:
        return ""
    if "_" in label or label.islower():
        text = label.replace("_", " ").strip()
        return text[:1].upper() + text[1:]
    return label


def _clock(moment: datetime) -> str:
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def time_text(ts: str | None, now: datetime) -> str:
    """'4:47 PM' for today, 'Oct 11, 5:55 AM' for earlier days, '' if unparseable."""
    if not isinstance(ts, str):
        return ""
    try:
        moment = datetime.fromisoformat(ts)
    except ValueError:
        return ""
    if moment.date() == now.date():
        return _clock(moment)
    return f"{moment.strftime('%b')} {moment.day}, {_clock(moment)}"


def _confidence(value: object) -> str:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return ""


def _events(camera_caps: list[dict]) -> list[dict]:
    """One capture per burst event (camera + event time), newest event first.

    Each event is represented by its most confident frame: the best photo of it.
    """
    best: dict[tuple, dict] = {}
    latest: dict[tuple, str] = {}
    for cap in camera_caps:
        ts = cap.get("capture_ts")
        if not isinstance(ts, str):
            continue
        key = (cap.get("camera_id"), cap.get("event_ts") or ts)
        latest[key] = max(latest.get(key, ""), ts)
        held = best.get(key)
        if held is None or (cap.get("confidence") or 0.0) > (held.get("confidence") or 0.0):
            best[key] = cap
    return [
        {**best[k], "event_latest": latest[k]} for k in sorted(best, key=lambda k: latest[k], reverse=True)
    ]


def _sighting(cap: dict, now: datetime) -> dict:
    return {
        "label": pretty_label(cap.get("label")),
        "camera": (cap.get("camera_id") or "").replace("_", " "),
        "time": time_text(cap["capture_ts"], now),
        "confidence": _confidence(cap.get("confidence")),
        "thumb_url": cap.get("thumb_url"),
        "image_url": cap.get("image_url"),
        # The event's newest capture: the panel's cursor for paging further back.
        "ts": cap.get("event_latest") or cap["capture_ts"],
    }


def sightings_page(event_caps: list[dict], now: datetime) -> list[dict]:
    """Shape one page of :meth:`wildlife.store.Store.query_events` rows for the panel."""
    return [_sighting(c, now) for c in event_caps if isinstance(c.get("capture_ts"), str)]


def build_wildlife(
    camera_caps: list[dict],
    bird_caps: list[dict],
    now: datetime,
    max_birds: int = 12,
    max_sightings: int = 12,
) -> dict:
    """Shape ``{"sighting", "sightings", "birds"}`` for ``/frame/wildlife``.

    ``sighting`` is the newest event (the strip thumbnail, or None); ``sightings``
    lists up to ``max_sightings`` recent events for the panel it opens.
    """
    sightings = [_sighting(c, now) for c in _events(camera_caps)[:max_sightings]]
    birds: list[dict] = []
    seen: set[str] = set()
    for cap in bird_caps:
        name = (cap.get("label") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        birds.append(
            {
                "label": name,
                "time": time_text(cap.get("capture_ts"), now),
                "confidence": _confidence(cap.get("confidence")),
                "photo_url": cap.get("species_photo_url"),
                "audio_url": cap.get("audio_url"),
            }
        )
        if len(birds) >= max_birds:
            break
    return {"sighting": sightings[0] if sightings else None, "sightings": sightings, "birds": birds}
