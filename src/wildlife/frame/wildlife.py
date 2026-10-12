"""Latest camera sighting + recent bird IDs for the /frame kiosk.

Pure shaping over capture dicts as produced by the gallery's serializer (newest
first, the order :meth:`wildlife.store.Store.query` returns): picks the newest
camera capture, de-duplicates bird detections by species (keeping each species'
newest clip), and formats labels, times and confidences for display so the page
JS only places text.
"""

from __future__ import annotations

from datetime import datetime

__all__ = ["CAMERA_KINDS", "build_wildlife", "pretty_label", "time_text"]

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


def build_wildlife(
    camera_caps: list[dict], bird_caps: list[dict], now: datetime, max_birds: int = 12
) -> dict:
    """Shape ``{"sighting": {...} | None, "birds": [...]}`` for ``/frame/wildlife``."""
    sighting = None
    dated = [c for c in camera_caps if isinstance(c.get("capture_ts"), str)]
    if dated:
        newest = max(dated, key=lambda c: c["capture_ts"])
        sighting = {
            "label": pretty_label(newest.get("label")),
            "camera": (newest.get("camera_id") or "").replace("_", " "),
            "time": time_text(newest["capture_ts"], now),
            "thumb_url": newest.get("thumb_url"),
            "image_url": newest.get("image_url"),
        }
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
    return {"sighting": sighting, "birds": birds}
