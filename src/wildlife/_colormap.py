"""A small magma colormap LUT for spectrograms (no matplotlib dependency).

The viridis-family colormaps (viridis/magma/inferno/plasma) by Stefan van der Walt
and Nathaniel Smith are released into the public domain (CC0); matplotlib merely
embeds the same arrays. We store a handful of anchor stops and interpolate them up
to a 256-entry uint8 lookup table at import.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAGMA_LUT"]

# (position 0..1, R, G, B) anchor stops approximating magma.
_ANCHORS = np.array(
    [
        [0.00, 0, 0, 4],
        [0.14, 28, 16, 68],
        [0.29, 79, 18, 123],
        [0.43, 129, 37, 129],
        [0.57, 181, 54, 122],
        [0.71, 229, 80, 100],
        [0.86, 251, 135, 97],
        [1.00, 252, 253, 191],
    ],
    dtype=np.float64,
)


def _build_lut() -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256)
    pos = _ANCHORS[:, 0]
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, pos, _ANCHORS[:, ch + 1]).round().astype(np.uint8)
    return lut


MAGMA_LUT: np.ndarray = _build_lut()
