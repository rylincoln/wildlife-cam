"""Audio bird-ID: BirdNET analysis + spectrogram rendering.

Hardware/heavy imports are confined here (mirrors ``capture.py``/``motion.py``):
``birdnet`` is imported LAZILY inside :class:`AudioAnalyzer`, so ``render_spectrogram``
(numpy + Pillow only) and this module import fine without the ``[audio]`` extra —
keeping the pure spectrogram tests hardware-free.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from wildlife._colormap import MAGMA_LUT

__all__ = ["render_spectrogram"]

# STFT params for a 3s / 48kHz window: ~560 columns, 46.9 Hz/bin.
_N_FFT = 1024
_HOP = 256
_FMAX_HZ = 12000  # crop above this (bird band); avoids an empty top third
_DB_FLOOR = 80.0
_OUT_HEIGHT = 256


def render_spectrogram(pcm: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Render a 48kHz mono PCM window to an RGB ``uint8`` spectrogram image.

    Hand-rolled STFT (numpy) → log-magnitude (dB, 80 dB floor) → magma LUT → a
    fixed-height (256 px) RGB array, low frequencies at the bottom. Deterministic,
    no matplotlib/scipy.
    """
    x = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:  # int16-scaled input -> normalize to [-1, 1]
        x = x / 32768.0
    if x.size < _N_FFT:
        x = np.pad(x, (0, _N_FFT - x.size))

    window = np.hanning(_N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, _N_FFT)[::_HOP]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1))  # (frames, n_fft/2+1)

    db = 20.0 * np.log10(spec + 1e-6)
    db = np.maximum(db, db.max() - _DB_FLOOR)

    keep = int(_FMAX_HZ / (sr / _N_FFT)) + 1  # bins up to _FMAX_HZ
    db = db[:, :keep]

    lo, hi = float(db.min()), float(db.max())
    norm = ((db - lo) / (hi - lo + 1e-9) * 255.0).astype(np.uint8)  # (frames, freq)
    norm = np.flipud(norm.T)  # (freq, frames), low freq at bottom
    rgb = MAGMA_LUT[norm]  # (freq, frames, 3)

    pil = Image.fromarray(rgb, mode="RGB")  # size = (frames, freq)
    new_w = max(1, round(pil.width * _OUT_HEIGHT / pil.height))
    pil = pil.resize((new_w, _OUT_HEIGHT), Image.LANCZOS)
    return np.asarray(pil)
