"""Audio bird-ID: BirdNET analysis + spectrogram rendering.

Hardware/heavy imports are confined here (mirrors ``capture.py``/``motion.py``):
``birdnet`` is imported LAZILY inside :class:`AudioAnalyzer`, so ``render_spectrogram``
(numpy + Pillow only) and this module import fine without the ``[audio]`` extra —
keeping the pure spectrogram tests hardware-free.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
from PIL import Image

from wildlife._colormap import MAGMA_LUT

__all__ = ["render_spectrogram", "signal_quality", "AudioAnalyzer"]

# STFT params for a 3s / 48kHz window: ~560 columns, 46.9 Hz/bin.
_N_FFT = 1024
_HOP = 256
_FMAX_HZ = 12000  # crop above this (bird band); avoids an empty top third
_DB_FLOOR = 80.0
_OUT_HEIGHT = 256

# Band for the SNR quality metric: skip sub-1 kHz wind/rumble; the 16 kHz mic holds
# no bird energy above ~8 kHz, so 1–8 kHz brackets the call band.
_SNR_FMIN_HZ = 1000.0
_SNR_FMAX_HZ = 8000.0


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


def signal_quality(pcm: np.ndarray, sr: int = 48000) -> float:
    """Estimate a mono window's call clarity as a band-limited SNR in dB.

    Frames the signal, sums magnitude energy in the bird band (1–8 kHz) per frame,
    then returns ``10*log10(peak-frame energy / noise-floor energy)`` where the noise
    floor is the median frame energy (the between-call background, which dominates the
    frame count) and the peak is the 95th-percentile frame (the loudest call energy).

    A clear call standing out from a quiet background scores high; steady broadband
    noise (wind, rain) has peak ≈ floor and scores near 0. Deterministic, numpy-only —
    used to pick the *clearest-sounding* sample among a species' repeats, whereas
    BirdNET confidence gates *which* detections count as the bird at all.
    """
    x = np.asarray(pcm, dtype=np.float32)
    if x.size < _N_FFT:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:  # int16-scaled input -> normalize (scale cancels in the ratio anyway)
        x = x / 32768.0

    window = np.hanning(_N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, _N_FFT)[::_HOP]
    mag = np.abs(np.fft.rfft(frames * window, axis=-1))  # (frames, n_fft/2+1)

    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    band = (freqs >= _SNR_FMIN_HZ) & (freqs <= _SNR_FMAX_HZ)
    if not band.any():
        return 0.0
    energy = (mag[:, band] ** 2).sum(axis=1)  # per-frame in-band energy
    if energy.size == 0:
        return 0.0

    eps = 1e-12
    floor = float(np.median(energy)) + eps
    signal = float(np.percentile(energy, 95)) + eps
    return 10.0 * float(np.log10(signal / floor))


logger = logging.getLogger(__name__)

_SAMPLE_RATE = 48000
_TOP_K = 1  # take the single best species per 3s window


def _week_of_year(now: datetime) -> int:
    """BirdNET's 1-48 week convention (4 weeks per month)."""
    return (now.month - 1) * 4 + min(4, (now.day - 1) // 7 + 1)


# birdnet 0.2.16 leaks one mp.Queue (2 pipe FDs + 3 semaphores) + a feeder thread +
# ~12 loggers on EVERY predict_arrays call: it attaches a QueueHandler to a uniquely
# named "birdnet.session_<id>" logger and never removes it in the main process (macOS
# spawn), so logging's registry pins it forever. Unreversed, the worker climbs to
# "OSError: Too many open files" within hours. We undo it after each analyze().
_TMP_SWEEP_INTERVAL_S = 60.0
_last_tmp_sweep = [0.0]  # module-level throttle for the tempdir sweep (monotonic s)


def _purge_birdnet_session_loggers() -> None:
    """Drop birdnet's per-call session loggers/queues; sweep its per-call temp logs."""
    mgr = logging.Logger.manager
    stale = [n for n in list(mgr.loggerDict)
             if n.startswith(("birdnet.session_", "birdnet_file_writer.session_"))]
    for name in stale:
        lg = mgr.loggerDict.pop(name, None)
        if not isinstance(lg, logging.Logger):
            continue
        for h in list(lg.handlers):
            lg.removeHandler(h)
            q = getattr(h, "queue", None)
            if q is not None and hasattr(q, "close"):
                try:
                    q.close()
                    q.join_thread()  # stop + reap the queue's feeder thread
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
    # birdnet also writes one <tempdir>/birdnet_session_<id>.log per call and never
    # removes it (a disk-inode leak). Sweep age-gated (>30s, never an in-flight file)
    # and throttled (a tempdir glob isn't worth doing every 1.5s).
    now = time.monotonic()
    if now - _last_tmp_sweep[0] < _TMP_SWEEP_INTERVAL_S:
        return
    _last_tmp_sweep[0] = now
    cutoff = time.time() - 30.0
    for f in glob.glob(os.path.join(tempfile.gettempdir(), "birdnet_session_*.log")):
        try:
            if os.path.getmtime(f) < cutoff:
                os.unlink(f)
        except OSError:
            pass


class AudioAnalyzer:
    """Shared, thread-safe BirdNET wrapper. Loads the models once; ``analyze`` a window.

    ``birdnet`` (and thus TensorFlow) is imported lazily here so the rest of the
    package imports without the ``[audio]`` extra.
    """

    def __init__(self, cfg: object) -> None:  # cfg: AudioConfig (typed loosely to avoid import)
        import birdnet  # lazy: pulls TensorFlow

        self._cfg = cfg
        self._birdnet = birdnet
        self._lock = threading.Lock()
        self._acoustic = birdnet.load("acoustic", "2.4", "tf")
        self._species_list: list[str] | None = None
        self._shortlist_week: int | None = None
        self._session = None  # warm, held-open prediction session (see _open_session)
        if getattr(cfg, "use_geo_filter", False):
            self._refresh_geo(datetime.now())
        self._open_session()
        logger.info(
            "AudioAnalyzer ready (geo_filter=%s, species_shortlist=%s, warm session).",
            getattr(cfg, "use_geo_filter", False),
            "-" if self._species_list is None else len(self._species_list),
        )

    def _open_session(self) -> None:
        """(Re)open a warm birdnet prediction session held across windows.

        ``predict_arrays`` opens AND tears down the whole producer/worker pipeline
        on EVERY call (~5 s), which can't keep up with the real-time stream -- the
        analyzer falls behind, the ffmpeg pipe fills, go2rtc drops the audio, and
        nothing is ever detected. A held-open session keeps one warm worker so each
        3 s window runs in ~1 s. Recreated when the weekly geo shortlist changes.
        This also supersedes the per-call multiprocessing that caused the earlier
        CPU-saturation and FD churn (now one persistent worker, not a per-call spawn).
        """
        self._close_session()
        session = self._acoustic.predict_session(
            top_k=_TOP_K,
            n_workers=1,
            n_producers=1,
            default_confidence_threshold=self._cfg.confidence_threshold,
            bandpass_fmin=self._cfg.bandpass_fmin,
            custom_species_list=self._species_list,
        )
        session.__enter__()  # spawns the warm worker (one-time cost)
        self._session = session

    def _close_session(self) -> None:
        """Tear down the warm session + reap its worker (best-effort)."""
        session = getattr(self, "_session", None)
        if session is not None:
            try:
                session.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.exception("Error closing birdnet session.")
            self._session = None

    def close(self) -> None:
        """Reap the warm session; call on worker shutdown."""
        with self._lock:
            self._close_session()

    def _refresh_geo(self, now: datetime) -> None:
        """(Re)build the geo shortlist when the BirdNET week rolls over.

        The occurrence shortlist is season-dependent, so a worker left running for
        weeks/months would otherwise keep its startup week's species forever and
        silently filter out newly-arrived migrants. Cheap: the week check runs per
        window, but a rebuild happens ~once a week. The (possibly failed) week is
        recorded either way so a transient geo-model error doesn't retry every 1.5s.
        """
        week = _week_of_year(now)
        if week == self._shortlist_week:
            return
        self._species_list = self._build_geo_shortlist(self._birdnet, self._cfg)
        self._shortlist_week = week
        # The warm session bakes in the species shortlist at creation, so rebuild it
        # with the new week's list (only when one is already open -- during __init__
        # the session is opened once after this returns).
        if getattr(self, "_session", None) is not None:
            self._open_session()
        logger.info(
            "Geo shortlist refreshed for week %d (%s species).",
            week, "-" if self._species_list is None else len(self._species_list),
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
            if getattr(cfg, "use_geo_filter", False):
                self._refresh_geo(datetime.now())  # rebuild ~weekly so the shortlist tracks the season
            # Reuse the warm session (~1 s/window) instead of predict_arrays, which
            # would spawn+tear-down the worker pipeline every call (~5 s) and never
            # keep up with the live stream.
            result = self._session.run_arrays(
                (np.asarray(pcm, dtype=np.float32), _SAMPLE_RATE)
            )
        arr = result.to_structured_array()
        out: list[tuple[str, float]] = []
        for row in arr:
            species_name = str(row["species_name"])
            common = species_name.split("_", 1)[-1]  # "Sci_Common" -> "Common"
            out.append((common, float(row["confidence"])))
        return out
