"""YOLO object detection on Apple Silicon (MPS) for the wildlife detector.

This module wraps Ultralytics YOLO and exposes a small, stable :class:`Detector`
facade used by the worker. It loads the model **once** (model load is the
expensive step; per-frame inference is cheap on the M1 GPU) and maps raw YOLO
results into the lightweight :class:`~wildlife.models.Detection` dataclass.

Design notes
------------
* The detector is *ungated*: :meth:`Detector.infer` returns **every** detection
  the model produces. Class/confidence/box-area filtering lives in
  :mod:`wildlife.gate` so that policy stays unit-testable and hardware-free.
* ``box_area_frac`` is computed here from the frame shape (box area divided by
  ``height * width``) and clamped to ``[0, 1]``.
* Device fallback: if ``device="mps"`` is requested but Metal Performance
  Shaders are unavailable (e.g. running on a non-Apple-Silicon host or in CI),
  we log a warning and transparently fall back to ``"cpu"``. Use
  :meth:`Detector.device_in_use` to confirm what is actually running -- this is
  what ``scripts/test_detect.py`` checks to ensure inference is not silently on
  CPU.

Optional Core ML / Apple Neural Engine export path
--------------------------------------------------
For lower power draw and better co-tenancy with the media server (so inference
does not fight a transcode for the GPU shaders), the model can be exported to
Core ML and run on the Apple Neural Engine instead of MPS::

    from ultralytics import YOLO
    model = YOLO("models/yolov8s.pt")
    model.export(format="coreml")        # writes models/yolov8s.mlpackage

Point ``detection.model_path`` at the resulting ``.mlpackage`` and Ultralytics
will load and run it through Core ML automatically; no code change is needed
here. MPS remains the default for simplicity -- prefer the Core ML path only if
you observe GPU contention with the media server.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from ultralytics import YOLO

# Re-export Detection so callers may simply ``from wildlife.detect import Detection``.
from wildlife.models import Detection

__all__ = ["Detector", "Detection"]

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Return the device to actually use, falling back ``mps`` -> ``cpu`` if needed.

    If ``mps`` is requested but ``torch.backends.mps.is_available()`` is False, a
    warning is logged and ``"cpu"`` is returned so inference still works (slower)
    rather than crashing.
    """
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning(
            "Device 'mps' requested but Metal Performance Shaders are not "
            "available on this host; falling back to 'cpu'."
        )
        return "cpu"
    return requested


class Detector:
    """Load a YOLO model once and run ungated inference, returning ``Detection``s.

    Parameters
    ----------
    model_path:
        Path to Ultralytics weights (e.g. ``"models/yolov8s.pt"``) or an exported
        Core ML ``.mlpackage`` (see module docstring for the export path).
    device:
        Requested inference device, ``"mps"`` (M1 GPU, default) or ``"cpu"``. If
        ``"mps"`` is unavailable it is downgraded to ``"cpu"`` with a warning.
    """

    def __init__(self, model_path: str, device: str = "mps") -> None:
        self._model_path = model_path
        self._requested_device = device
        self._device = _resolve_device(device)
        logger.info(
            "Loading YOLO model from %s (requested device=%s, using device=%s)",
            model_path,
            device,
            self._device,
        )
        self._model = YOLO(model_path)
        # ``names`` maps class index -> human-readable COCO label.
        self._names: dict[int, str] = dict(self._model.names)

    def device_in_use(self) -> str:
        """Return the device inference actually runs on (``"mps"`` or ``"cpu"``)."""
        return self._device

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO on a single BGR/RGB frame and return ALL detections (ungated).

        ``box_area_frac`` is the detection box area divided by the full frame area
        (``height * width``), clamped to ``[0, 1]``. Class/confidence/area gating
        is intentionally left to :mod:`wildlife.gate`.
        """
        height, width = frame.shape[:2]
        frame_area = float(height) * float(width)

        results = self._model(frame, device=self._device, verbose=False)

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            # Move tensors to plain Python lists (works for MPS/CPU/Core ML).
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()

            for cls_idx, confidence, (x1, y1, x2, y2) in zip(
                classes, confidences, xyxy
            ):
                label = self._names.get(int(cls_idx), str(int(cls_idx)))
                box_w = max(0.0, float(x2) - float(x1))
                box_h = max(0.0, float(y2) - float(y1))
                box_area = box_w * box_h
                area_frac = box_area / frame_area if frame_area > 0 else 0.0
                area_frac = min(1.0, max(0.0, area_frac))

                detections.append(
                    Detection(
                        label=label,
                        confidence=float(confidence),
                        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        box_area_frac=area_frac,
                    )
                )

        return detections
