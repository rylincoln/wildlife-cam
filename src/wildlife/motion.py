"""Pure per-frame motion decision for continuous, motion-gated detection.

This module mirrors ``capture.py``'s hardware confinement: only ``cv2``/``numpy``
(plus stdlib) are imported, so the heavyweight decode/vision dependency stays out
of the light, hardware-free library modules. It holds no I/O and no temporal
logic — a :class:`MotionDetector` answers one question per frame ("does *this*
frame show motion, and did the scene just change wholesale?"). The rising-edge /
refractory / warmup orchestration lives in the event source that drives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["MotionDetector", "MotionResult"]

# Morphology kernel used to erode-then-dilate the foreground mask, killing
# single-pixel speckle before contour analysis.
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Foreground binarisation threshold for the frame_diff algorithm.
_FRAME_DIFF_THRESH = 25
# Exponential-moving-average weight for the frame_diff reference frame.
_FRAME_DIFF_REF_ALPHA = 0.1


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Outcome of one :meth:`MotionDetector.update` call.

    Attributes:
        motion: True if the largest motion contour meets the area threshold.
        scene_change: True on a whole-frame change (PTZ / IR-cut flip / exposure
            jump) — the caller should ``reset()`` and swallow the transient.
        area_frac: Largest motion contour area as a fraction of the (downscaled)
            frame area; surfaced for tuning/logging.
    """

    motion: bool
    scene_change: bool
    area_frac: float


class MotionDetector:
    """Decide per-frame motion via MOG2 background subtraction (or frame diff).

    Parameters
    ----------
    downscale_width:
        Width (px) the motion computation runs at; taller frames are shrunk to
        this width (aspect preserved) so the vision work stays cheap.
    min_area_frac:
        The largest motion contour must cover at least this fraction of the
        downscaled frame to count as motion.
    algorithm:
        ``"mog2"`` (default; adapts to swaying vegetation / gradual light) or
        ``"frame_diff"`` (lighter absdiff-vs-rolling-reference fallback).
    mask_polys:
        Optional ignore regions as normalised ``0..1`` polygons; motion inside
        them is discarded (roads / canopy / flags / water).
    scene_change_thresh:
        Mean absolute (0..255) whole-frame diff above which ``scene_change`` trips.
    history, var_threshold:
        MOG2 tuning (``detectShadows`` is always False).
    """

    def __init__(
        self,
        downscale_width: int,
        min_area_frac: float,
        algorithm: str = "mog2",
        mask_polys: list[list[tuple[float, float]]] | None = None,
        scene_change_thresh: float = 40.0,
        history: int = 500,
        var_threshold: int = 16,
    ) -> None:
        self._downscale_width = int(downscale_width)
        self._min_area_frac = float(min_area_frac)
        self._algorithm = algorithm
        self._scene_change_thresh = float(scene_change_thresh)
        self._history = int(history)
        self._var_threshold = int(var_threshold)
        self._mask_polys = mask_polys or []

        self._prev_gray: np.ndarray | None = None  # for scene-change diff
        self._ref_gray: np.ndarray | None = None  # frame_diff rolling reference
        self._mask: np.ndarray | None = None  # rasterised keep-mask at work size
        self._subtractor = self._make_subtractor()

    def _make_subtractor(self):
        """Build a fresh MOG2 subtractor (None for the frame_diff algorithm)."""
        if self._algorithm == "mog2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._var_threshold,
                detectShadows=False,
            )
        return None

    def reset(self) -> None:
        """Rebuild the background model and clear cached frames.

        Called by the driving source on a scene change or reconnect so a wholesale
        scene shift does not read as a lasting field of motion.
        """
        self._subtractor = self._make_subtractor()
        self._prev_gray = None
        self._ref_gray = None

    def update(self, frame_bgr: np.ndarray) -> MotionResult:
        """Return the :class:`MotionResult` for one BGR frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self._downscale_width:
            new_h = max(1, round(h * self._downscale_width / w))
            gray = cv2.resize(
                gray, (self._downscale_width, new_h), interpolation=cv2.INTER_AREA
            )

        if self._mask is None and self._mask_polys:
            self._mask = self._rasterize_mask(gray.shape)

        # Whole-frame scene change (catches PTZ / IR-cut flip / exposure jump).
        scene_change = False
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            scene_change = (
                float(cv2.absdiff(gray, self._prev_gray).mean())
                > self._scene_change_thresh
            )
        self._prev_gray = gray

        fg = self._foreground(gray)
        if fg is None:  # frame_diff seeding its reference on the first frame
            return MotionResult(motion=False, scene_change=scene_change, area_frac=0.0)

        if self._mask is not None:
            fg = cv2.bitwise_and(fg, self._mask)
        # Erode-then-dilate: remove speckle without inflating real blobs.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _KERNEL)

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = fg.shape[0] * fg.shape[1]
        largest = max((cv2.contourArea(c) for c in contours), default=0.0)
        area_frac = (largest / frame_area) if frame_area else 0.0
        return MotionResult(
            motion=area_frac >= self._min_area_frac,
            scene_change=scene_change,
            area_frac=area_frac,
        )

    def _foreground(self, gray: np.ndarray) -> np.ndarray | None:
        """Binary foreground mask for ``gray``; None while frame_diff seeds itself."""
        if self._algorithm == "mog2":
            # detectShadows=False -> apply() already yields a 0/255 mask.
            return self._subtractor.apply(gray)
        # frame_diff: absdiff vs a slowly-updated rolling reference.
        if self._ref_gray is None:
            self._ref_gray = gray.copy()
            return None
        diff = cv2.absdiff(gray, self._ref_gray)
        _, fg = cv2.threshold(diff, _FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        self._ref_gray = cv2.addWeighted(
            self._ref_gray, 1.0 - _FRAME_DIFF_REF_ALPHA, gray, _FRAME_DIFF_REF_ALPHA, 0.0
        )
        return fg

    def _rasterize_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Rasterise ignore polygons into a keep-mask (255=keep, 0=ignore)."""
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        for poly in self._mask_polys:
            pts = np.array(
                [[int(round(x * w)), int(round(y * h))] for (x, y) in poly],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 0)
        return mask
