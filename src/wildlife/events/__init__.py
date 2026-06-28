"""Camera event-source package.

Turns each camera's motion / animal detection feed into a simple synchronous
iterator of :class:`~wildlife.models.CameraEvent` values consumed by the worker.

Public surface::

    from wildlife.events import make_event_source, EventSource
    src = make_event_source(config.event_source, camera)
    for event in src.stream():
        ...

Two engines are available and selected via ``Config.event_source``:

* :class:`ReolinkEventSource` (``"reolink_native"``) -- preferred, ``reolink-aio``.
* :class:`OnvifEventSource` (``"onvif_bridge"``) -- fallback, ``onvif-zeep`` PullPoint.

The concrete source classes are re-exported for convenience and typing. Their
third-party dependencies are imported lazily inside the classes, so importing
this package does not require the optional ``cameras`` dependency group to be
installed.
"""

from __future__ import annotations

from wildlife.events.base import CameraEvent, EventSource, make_event_source
from wildlife.events.onvif_bridge import OnvifEventSource
from wildlife.events.reolink_native import ReolinkEventSource

__all__ = [
    "CameraEvent",
    "EventSource",
    "make_event_source",
    "ReolinkEventSource",
    "OnvifEventSource",
]
