"""MegaDetector v6 second pass over the species detector's output.

The species detector (:mod:`wildlife.detect`) is a 21-class fine-grained model.
On out-of-distribution or hard subjects it can (a) mislabel a coarse *type* --
famously a distant person tagged ``bird`` -- (b) miss an animal entirely, or (c)
fire on nothing (waving sagebrush, shadow). MegaDetector v6 is a robust,
class-agnostic camera-trap detector that only distinguishes ``animal`` /
``person`` / ``vehicle`` but does so *far* better than a species model at
distance. Running it as a cheap **per-event** second pass lets it play three
roles (each independently toggleable in config):

* **person arbiter** -- relabel a kept detection to ``person`` when MD sees a
  person on the same box (fixes person->bird);
* **recall net** -- when the species model kept nothing but MD sees an actionable
  object, save it with the coarse MD label so the event isn't lost;
* **false-positive filter** -- drop a kept detection MD sees nothing behind
  (opt-in, off by default since MD can miss too).

Design mirrors :mod:`wildlife.detect`:

* The model is loaded **once** (in the worker's ``_setup``) and reused; the heavy
  ``ultralytics``/``torch`` imports are deferred to :class:`MegaDetector`
  construction so this module -- and its *pure* decision logic
  (:func:`second_pass_decision`) -- stays importable in hardware-free tests.
* :meth:`MegaDetector.infer` returns ungated :class:`~wildlife.models.Detection`
  objects (labels ``animal``/``person``/``vehicle``, pixel ``xyxy``), matching
  the :class:`~wildlife.detect.Detector` contract so the two are symmetric.
* Categories are resolved from the checkpoint **by name**, never by hard-coded
  index (the raw MDv6 ``.pt`` is ``{0:animal,1:person,2:vehicle}`` but that must
  not be assumed).
* Like the species detector, the BGR frame is handed to ultralytics **as-is**
  (ultralytics converts BGR->RGB internally for the model); do NOT pre-convert.
"""

from __future__ import annotations

import logging

from wildlife.models import Detection

__all__ = ["MegaDetector", "second_pass_decision"]

logger = logging.getLogger(__name__)

# MegaDetector v6 was trained at 1280px; camera-trap inference uses the same.
_DEFAULT_IMGSZ = 1280


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` pixel boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def second_pass_decision(
    *,
    md_dets: list[Detection],
    best_det: Detection | None,
    best_frame: object | None,
    positives: list[tuple],
    representative_frame: object,
    save_best_only: bool,
    cfg,
    min_area_frac: float = 0.0,
) -> tuple[Detection | None, object | None, list[tuple], str]:
    """Fold MegaDetector detections into the species detector's per-event result.

    Pure (no torch): all inputs are plain data, so this is unit-tested directly.

    Args:
        md_dets: MegaDetector detections on ``representative_frame`` (labels
            ``animal``/``person``/``vehicle``), ungated.
        best_det/best_frame: the ``save_best_only`` winner (or ``None``).
        positives: the all-positives list of ``(frame, Detection)`` pairs.
        representative_frame: the frame MD ran on; used as the frame for a rescue.
        save_best_only: which of the two result shapes is live.
        cfg: the ``megadetector`` config block (``classes``, ``confidence``,
            ``iou_match``, ``person_override``, ``rescue_misses``,
            ``suppress_false_positives``).
        min_area_frac: area floor applied to a rescued detection (mirrors the
            species gate's ``min_box_area_frac`` so MD can't rescue a speck).

    Returns:
        Updated ``(best_det, best_frame, positives, reason)``. ``reason`` is a
        human log string, empty when nothing changed.

    Note:
        In all-positives mode the per-positive boxes come from their own burst
        frames while MD ran on one representative frame; because burst frames are
        ~100ms apart the subject barely moves, so IoU matching across them is a
        deliberate, good-enough approximation (keeps MD to one inference/event).
    """
    classes = set(cfg.classes)
    actionable = [d for d in md_dets if d.label in classes and d.confidence >= cfg.confidence]
    persons = [d for d in actionable if d.label == "person"]
    kept = (best_det is not None) if save_best_only else bool(positives)

    # --- Recall net: species kept nothing, MD sees an actionable object. ------
    if not kept:
        if cfg.rescue_misses:
            # Rescue bypasses the species gate and saves a generic label, so it
            # needs a higher confidence bar than person-override (a low-conf MD box
            # on a hiccup/flat frame would otherwise be saved as "animal").
            rescue_min = max(cfg.confidence, getattr(cfg, "rescue_min_confidence", 0.0))
            candidates = [
                d for d in actionable
                if d.box_area_frac >= min_area_frac and d.confidence >= rescue_min
            ]
            if candidates:
                top = max(candidates, key=lambda d: d.confidence)
                rescued = Detection(
                    label=top.label,
                    confidence=top.confidence,
                    box_xyxy=top.box_xyxy,
                    box_area_frac=top.box_area_frac,
                )
                reason = f"rescued a missed {top.label} (conf={top.confidence:.2f})"
                if save_best_only:
                    return rescued, representative_frame, positives, reason
                return best_det, best_frame, [(representative_frame, rescued)], reason
        return best_det, best_frame, positives, ""

    # --- Kept: person-override and/or false-positive suppression. -------------
    def _best_person_over(box) -> Detection | None:
        matches = [p for p in persons if _iou(box, p.box_xyxy) >= cfg.iou_match]
        return max(matches, key=lambda p: p.confidence) if matches else None

    def _has_actionable_over(box) -> bool:
        return any(_iou(box, d.box_xyxy) >= cfg.iou_match for d in actionable)

    def _relabel_person(det: Detection) -> tuple[Detection, bool]:
        if not (cfg.person_override and det.label != "person"):
            return det, False
        p = _best_person_over(det.box_xyxy)
        if p is None:
            return det, False
        return (
            Detection(
                label="person",
                confidence=p.confidence,
                box_xyxy=det.box_xyxy,
                box_area_frac=det.box_area_frac,
            ),
            True,
        )

    overrides = 0
    drops = 0

    if save_best_only:
        det, changed = _relabel_person(best_det)
        overrides += int(changed)
        if cfg.suppress_false_positives and not _has_actionable_over(det.box_xyxy):
            return None, None, positives, f"suppressed a false positive ({best_det.label})"
        reason = f"relabeled {best_det.label} -> person" if overrides else ""
        return det, best_frame, positives, reason

    new_positives: list[tuple] = []
    for frame, det in positives:
        det, changed = _relabel_person(det)
        overrides += int(changed)
        if cfg.suppress_false_positives and not _has_actionable_over(det.box_xyxy):
            drops += 1
            continue
        new_positives.append((frame, det))
    parts = []
    if overrides:
        parts.append(f"relabeled {overrides} -> person")
    if drops:
        parts.append(f"suppressed {drops} false positive(s)")
    return best_det, best_frame, new_positives, "; ".join(parts)


class MegaDetector:
    """Load MegaDetector v6 once and run ungated per-frame inference.

    Parameters
    ----------
    weights_path:
        Path to the MDv6 ultralytics ``.pt`` (e.g. ``weights/MDV6-yolov10e.pt``).
        Resolved relative to the process CWD if not absolute (the worker runs from
        the repo root under launchd).
    device:
        ``"mps"`` (default) or ``"cpu"``; ``"mps"`` downgrades to ``"cpu"`` with a
        warning if Metal is unavailable (shares :func:`wildlife.detect._resolve_device`).
    confidence:
        Detection score floor passed to ultralytics ``predict`` (MDv6 camera-trap
        default ~0.2), so :meth:`infer` never returns sub-threshold boxes.
    imgsz:
        Inference size; MDv6 was trained at 1280.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "mps",
        confidence: float = 0.2,
        imgsz: int = _DEFAULT_IMGSZ,
    ) -> None:
        # Deferred so importing this module (and second_pass_decision) needs no
        # torch/ultralytics -- construction is the only heavy step.
        from ultralytics import YOLO

        from wildlife.detect import _resolve_device

        self._weights_path = weights_path
        self._requested_device = device
        self._device = _resolve_device(device)
        self._confidence = float(confidence)
        self._imgsz = int(imgsz)
        logger.info(
            "Loading MegaDetector from %s (requested device=%s, using device=%s, conf=%.2f, imgsz=%d)",
            weights_path,
            device,
            self._device,
            self._confidence,
            self._imgsz,
        )
        self._model = YOLO(weights_path)
        # Resolve animal/person/vehicle BY NAME (never a hard-coded index; the raw
        # MDv6 .pt is 0-indexed animal/person/vehicle but that must not be assumed).
        self._names: dict[int, str] = {int(k): str(v) for k, v in self._model.names.items()}
        by_name = {str(v).strip().lower(): int(k) for k, v in self._names.items()}
        if "animal" not in by_name or "person" not in by_name:
            raise ValueError(
                f"MegaDetector weights {weights_path!r} lack animal/person classes; got {self._names}"
            )

    def device_in_use(self) -> str:
        """Return the device inference actually runs on (``"mps"`` or ``"cpu"``)."""
        return self._device

    def infer(self, frame) -> list[Detection]:
        """Run MegaDetector on one BGR frame and return ALL detections (ungated).

        The BGR ``frame`` is handed to ultralytics as-is (it converts BGR->RGB for
        the model internally, exactly like :meth:`wildlife.detect.Detector.infer`).
        Labels are the MD class names (``animal``/``person``/``vehicle``); boxes are
        absolute pixel ``xyxy``; ``box_area_frac`` is box area / frame area.
        """
        height, width = frame.shape[:2]
        frame_area = float(height) * float(width)

        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            for cls_idx, confidence, (x1, y1, x2, y2) in zip(classes, confidences, xyxy):
                label = self._names.get(int(cls_idx), str(int(cls_idx)))
                box_w = max(0.0, float(x2) - float(x1))
                box_h = max(0.0, float(y2) - float(y1))
                area_frac = (box_w * box_h) / frame_area if frame_area > 0 else 0.0
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
