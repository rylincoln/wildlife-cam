"""Cheap per-frame / per-box image features shared by the worker and the replay harness.

numpy-only (no cv2) so the offline harness (``wildlife-replay``) measures exactly
the code the live worker runs, and both stay unit-testable without OpenCV.

All frames are BGR ``uint8`` arrays (OpenCV's channel order), boxes are
``(x1, y1, x2, y2)`` in pixel coordinates.
"""

from __future__ import annotations

import numpy as np

# An IR (night-mode) Reolink frame is monochrome: the three channels carry the
# same value to within JPEG noise. Daylight colour frames -- dawn and dusk
# included -- differ by far more than this.
_IR_MAX_CHANNEL_SPREAD = 2.0
# Sub-sample stride for the IR test; the decision is global, so 1/64 of the
# pixels is plenty and keeps it ~free on a 4K frame.
_IR_SAMPLE_STRIDE = 8
# Luma weights for BGR (ITU-R BT.601, what cv2.COLOR_BGR2GRAY uses).
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def _luma(img: np.ndarray) -> np.ndarray:
    """Float luma of a BGR image (or pass-through for a single-channel one)."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return img[..., :3].astype(np.float32) @ _LUMA_BGR


def is_ir_frame(frame: np.ndarray) -> bool:
    """True if ``frame`` was shot in IR night mode (monochrome), False if colour."""
    if frame.ndim == 2 or frame.shape[2] < 3:
        return True
    s = frame[::_IR_SAMPLE_STRIDE, ::_IR_SAMPLE_STRIDE, :3].astype(np.int16)
    spread = max(
        float(np.abs(s[..., 0] - s[..., 1]).mean()),
        float(np.abs(s[..., 1] - s[..., 2]).mean()),
        float(np.abs(s[..., 0] - s[..., 2]).mean()),
    )
    return spread < _IR_MAX_CHANNEL_SPREAD


def box_brightness_ratio(frame: np.ndarray, box) -> float:
    """Mean luma inside ``box`` divided by the whole frame's mean luma.

    Under the IR illuminator an insect a few centimetres from the lens is lit far
    brighter than anything at scene distance, so its box glows: on the labelled
    archive moths ran 0.77-5.8x (median 1.7) while real night animals -- darker
    than the IR-lit ground -- ran 0.39-0.58x. Returns 1.0 (neutral) for a box
    that falls outside the frame.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 1.0
    frame_mean = float(_luma(frame[::_IR_SAMPLE_STRIDE, ::_IR_SAMPLE_STRIDE]).mean())
    box_mean = float(_luma(frame[y1:y2, x1:x2]).mean())
    return box_mean / max(frame_mean, 1.0)


def ir_glow_reject(frame: np.ndarray, box, max_ratio: float) -> bool:
    """True if an IR-mode detection's box glows brighter than ``max_ratio`` x the frame.

    The night-insect gate. ``max_ratio <= 0`` disables it, and colour (daylight)
    frames are never judged: a pale animal against dark daytime ground can be
    legitimately bright, and the insects only light up under the illuminator.
    """
    if max_ratio <= 0 or not is_ir_frame(frame):
        return False
    return box_brightness_ratio(frame, box) > max_ratio
