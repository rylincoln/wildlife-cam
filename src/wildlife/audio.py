"""Audio bird-ID: BirdNET analysis + spectrogram rendering.

Hardware/heavy imports are confined here (mirrors ``capture.py``/``motion.py``):
``birdnet`` is imported LAZILY inside :class:`AudioAnalyzer`, so ``render_spectrogram``
(numpy + Pillow only) and this module import fine without the ``[audio]`` extra —
keeping the pure spectrogram tests hardware-free.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime

import numpy as np
from PIL import Image

from wildlife._colormap import MAGMA_LUT

__all__ = ["render_spectrogram", "AudioAnalyzer"]

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


logger = logging.getLogger(__name__)

_SAMPLE_RATE = 48000
_TOP_K = 1  # take the single best species per 3s window


def _week_of_year(now: datetime) -> int:
    """BirdNET's 1-48 week convention (4 weeks per month)."""
    return (now.month - 1) * 4 + min(4, (now.day - 1) // 7 + 1)


class AudioAnalyzer:
    """Shared, thread-safe BirdNET wrapper. Loads the models once; ``analyze`` a window.

    ``birdnet`` (and thus TensorFlow) is imported lazily here so the rest of the
    package imports without the ``[audio]`` extra.
    """

    def __init__(self, cfg: object) -> None:  # cfg: AudioConfig (typed loosely to avoid import)
        import birdnet  # lazy: pulls TensorFlow

        self._cfg = cfg
        self._lock = threading.Lock()
        self._acoustic = birdnet.load("acoustic", "2.4", "tf")
        self._species_list: list[str] | None = None
        if getattr(cfg, "use_geo_filter", False):
            self._species_list = self._build_geo_shortlist(birdnet, cfg)
        logger.info(
            "AudioAnalyzer ready (geo_filter=%s, species_shortlist=%s).",
            getattr(cfg, "use_geo_filter", False),
            "-" if self._species_list is None else len(self._species_list),
        )

    def _build_geo_shortlist(self, birdnet, cfg) -> list[str] | None:
        """Occurrence shortlist for lat/lon + current week; None on any failure."""
        try:
            geo = birdnet.load("geo", "2.4", "tf")
            week = _week_of_year(datetime.now())
            result = geo.predict(cfg.latitude, cfg.longitude, week=week)
            arr = result.to_structured_array()  # verify column on prod: 'species_name'
            return [str(s) for s in arr["species_name"]]
        except Exception:  # noqa: BLE001 - geo is best-effort; fall back to no filter
            logger.warning("Geo shortlist unavailable; running without it.", exc_info=True)
            return None

    def analyze(self, pcm: np.ndarray) -> list[tuple[str, float]]:
        """Return ``[(common_name, confidence), …]`` for a 48kHz mono float32 window."""
        cfg = self._cfg
        with self._lock:
            result = self._acoustic.predict_arrays(
                (np.asarray(pcm, dtype=np.float32), _SAMPLE_RATE),
                top_k=_TOP_K,
                default_confidence_threshold=cfg.confidence_threshold,
                bandpass_fmin=cfg.bandpass_fmin,
                custom_species_list=self._species_list,
            )
        arr = result.to_structured_array()
        out: list[tuple[str, float]] = []
        for row in arr:
            species_name = str(row["species_name"])
            common = species_name.split("_", 1)[-1]  # "Sci_Common" -> "Common"
            out.append((common, float(row["confidence"])))
        return out
