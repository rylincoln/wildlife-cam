"""Core domain dataclasses shared across the wildlife detection system.

This module is the source-of-truth for the lightweight value objects that flow
between components (event sources, detector, gate, store, worker, gallery).

It MUST remain dependency-free: standard library only. No ``torch``, ``cv2``,
``numpy``, ``pydantic`` or any other heavy/optional import at module scope. This
keeps the contract types importable in hardware-free unit tests and in light
modules (``config``, ``gate``, ``store``) without dragging in the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["CameraEvent", "Detection"]


@dataclass(frozen=True, slots=True)
class CameraEvent:
    """A single motion/animal trigger emitted by a camera's event source.

    Attributes:
        camera_id: Stable identifier of the originating camera (``CameraConfig.id``).
        event_ts: Timestamp at which the camera-side event fired.
        kind: Event category as reported by the source (e.g. ``"animal"``,
            ``"motion"``). Used for logging/tuning; the Mac-side YOLO pass is the
            precise gate.
    """

    camera_id: str
    event_ts: datetime
    kind: str


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detected by the YOLO model on one frame.

    Attributes:
        label: COCO class name for the detection (e.g. ``"deer"`` is not COCO;
            stock classes are ``"bird"``, ``"cat"``, ``"dog"``, ...).
        confidence: Model confidence score in the range ``0.0`` to ``1.0``.
        box_xyxy: Bounding box in pixel coordinates as ``(x1, y1, x2, y2)`` where
            ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` the bottom-right.
        box_area_frac: Box area divided by full frame area, in ``0.0`` to ``1.0``;
            used to discard specks far below the configured minimum.
    """

    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    box_area_frac: float
