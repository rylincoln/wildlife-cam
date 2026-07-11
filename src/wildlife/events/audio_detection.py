"""Per-camera BirdNET audio analyzer thread.

Manages its own daemon thread (``start()``/``stop()``), mirroring the notifier's
lifecycle rather than the queue-backed event sources — it saves detections
DIRECTLY (BirdNET is the model; there is no YOLO consumer to feed). ffmpeg decodes
the go2rtc restream audio to 48 kHz mono PCM on a subprocess pipe; the reader frames
it into 3 s windows (1.5 s hop) and, on a confirmed detection, renders a spectrogram,
encodes an AAC clip from the same PCM, and calls ``save_audio_capture``.

Heavy work is confined: ``render_spectrogram`` (numpy/Pillow) is module-level, but
``subprocess``/ffmpeg run only inside the thread. ``AudioAnalyzer`` (birdnet) is
injected, already loaded.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from datetime import datetime

import numpy as np

from wildlife.audio import render_spectrogram
from wildlife.audio_gate import RepeatConfirmer

__all__ = ["AudioDetectionSource", "WIN_SAMPLES", "HOP_SAMPLES"]

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 48000
WIN_SAMPLES = _SAMPLE_RATE * 3          # 144000  (3.0 s)
# 3.0 s hop (no overlap). BirdNET inference is ~1 s/window via the warm session, so
# two cameras (one window each per hop) at a 1.5 s hop couldn't keep up (2 windows /
# 1.5 s > 1 / 1 s) and the stream backed up. A 3 s hop keeps up with headroom.
HOP_SAMPLES = _SAMPLE_RATE * 3          # 144000  (3.0 s hop)
_HOP_BYTES = HOP_SAMPLES * 2           # int16
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def _parse_port(listen: str) -> int:
    return int(listen.rsplit(":", 1)[-1])


def _parse_active_hours(spec: str) -> tuple[int, int] | None:
    spec = spec.strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", spec)
    if not m:
        return None
    sh, sm, eh, em = (int(g) for g in m.groups())
    return (sh * 60 + sm, eh * 60 + em)


def _in_window(cur_min: int, window: tuple[int, int]) -> bool:
    """True if minute-of-day ``cur_min`` falls in ``window`` (wraps if start > end)."""
    start, end = window
    if start <= end:
        return start <= cur_min < end
    return cur_min >= start or cur_min < end


class AudioDetectionSource:
    """Analyze one camera's audio and save confirmed bird detections directly."""

    def __init__(self, camera: object, config: object, analyzer: object, store: object) -> None:
        cc = config.audio
        self._camera = camera
        self._analyzer = analyzer
        self._store = store
        self._stream = cc.stream
        self._rtsp_port = _parse_port(config.livestream.rtsp_listen)
        self._active_window = _parse_active_hours(cc.active_hours)
        # Per-species local windows: a species is dropped outside its own window.
        self._species_windows = {
            name: w
            for name, spec in (getattr(cc, "species_hours", None) or {}).items()
            if (w := _parse_active_hours(spec)) is not None
        }
        if self._species_windows:
            # Log the raw config so a mistyped key (must equal BirdNET's exact common
            # name, e.g. "Canyon Wren") is visible, and so it's clear which species
            # are gated — noise that re-labels onto an UNlisted diurnal species at
            # night would still slip through until added here.
            logger.info(
                "Camera %s: audio species-hour windows active: %s",
                self._camera.id, dict(cc.species_hours),
            )
        self._confirmer = RepeatConfirmer(cc.min_confirmations, cc.confirm_window_s, cc.cooldown_s)
        self._clip_filter = (getattr(cc, "clip_audio_filter", "") or "").strip()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name=f"audio-{self._camera.id}", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._cleanup_proc()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def _cleanup_proc(self) -> None:
        """Kill AND fully reap the ffmpeg child.

        Just ``kill()`` leaves the stdout pipe FD open and the child a zombie until
        GC; across many reconnects that leaks file descriptors (the worker has hit
        ``OSError: Too many open files``). Closing stdout + ``wait()`` releases both.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - best-effort
            pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - don't hang shutdown on a stuck child
            pass
        self._proc = None

    # -- pure decision helpers (unit-tested) ------------------------------
    def _within_active_hours(self, now_wall: datetime) -> bool:
        if self._active_window is None:
            return True
        cur = now_wall.hour * 60 + now_wall.minute
        return _in_window(cur, self._active_window)

    def _species_within_hours(self, species: str, now_wall: datetime) -> bool:
        """False when ``species`` has a configured window and ``now`` is outside it.

        Diurnal birds get labelled onto nocturnal insect stridulation (Canyon Wren
        at 4 a.m. = crickets). A per-species window drops those without muting
        genuinely nocturnal callers (owls, nighthawks), which carry no window.
        """
        window = self._species_windows.get(species)
        if window is None:
            return True
        cur = now_wall.hour * 60 + now_wall.minute
        return _in_window(cur, window)

    def _process_window(self, pcm: np.ndarray, now: datetime) -> None:
        """Analyze one window; on a confirmed species, render+encode+save."""
        if not self._within_active_hours(now):
            return
        for species, conf in self._analyzer.analyze(pcm):
            # Drop out-of-hours species BEFORE the confirmer so nocturnal insect
            # noise never even accumulates toward a same-species confirmation.
            if not self._species_within_hours(species, now):
                continue
            if not self._confirmer.offer(species, conf, now):
                continue
            try:
                spec_rgb = render_spectrogram(pcm, _SAMPLE_RATE)
            except Exception:  # noqa: BLE001 - a bad render skips the detection
                logger.exception("Camera %s: spectrogram render failed.", self._camera.id)
                continue
            clip = None
            try:
                clip = self._encode_clip(pcm)
            except Exception:  # noqa: BLE001 - save w/o clip rather than lose the detection
                logger.warning("Camera %s: clip encode failed; saving without audio.", self._camera.id)
            self._store.save_audio_capture(
                camera_id=self._camera.id,
                event_ts=now,
                capture_ts=datetime.now(),
                species=species,
                confidence=conf,
                spectrogram_rgb=spec_rgb,
                clip_bytes=clip,
                source_kind="audio",
            )
            logger.info("Camera %s: audio detection %s (%.2f) saved.", self._camera.id, species, conf)

    # -- ffmpeg (hardware; not unit-tested) -------------------------------
    def _decode_cmd(self) -> list[str]:
        url = f"rtsp://127.0.0.1:{self._rtsp_port}/{self._camera.id}_{self._stream}"
        return [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", url, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(_SAMPLE_RATE),
            "-f", "s16le", "-",
        ]

    def _encode_clip(self, pcm: np.ndarray) -> bytes:
        """Encode a float32 window to AAC/.m4a bytes via ffmpeg, from stdin PCM.

        MP4/.m4a with ``+faststart`` needs a *seekable* output (piping to stdout
        fails: "muxer does not support non seekable output"), so we encode to a
        temp file and read the bytes back. The clip is tiny (~38 KB), so this is cheap.
        """
        import os
        import tempfile

        pcm16 = np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()
        fd, tmp = tempfile.mkstemp(suffix=".m4a")
        os.close(fd)
        try:
            # De-wind the playback clip (detection already ran on the raw window, so
            # this only affects what a human hears): high-pass rumble + adaptive
            # denoise + boost the faint call. Blank filter -> encode raw.
            af = ["-af", self._clip_filter] if self._clip_filter else []
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-f", "s16le", "-ar", str(_SAMPLE_RATE), "-ac", "1", "-i", "-",
                 *af,
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", tmp],
                input=pcm16, check=True, timeout=15,
            )
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run(self) -> None:
        """Open ffmpeg, frame the PCM into windows, process each; reconnect on drop."""
        backoff = _INITIAL_BACKOFF_S
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._decode_cmd(), stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=WIN_SAMPLES * 2,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Camera %s: failed to start audio ffmpeg.", self._camera.id)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue

            ring = bytearray()
            delivered = False
            try:
                while not self._stop.is_set():
                    hop = self._read_exact(self._proc.stdout, _HOP_BYTES)
                    if hop is None:
                        break  # stream dropped -> reopen
                    delivered = True
                    ring += hop
                    if len(ring) >= WIN_SAMPLES * 2:
                        window = np.frombuffer(ring[-WIN_SAMPLES * 2:], dtype="<i2")
                        pcm = window.astype(np.float32) / 32768.0
                        try:
                            self._process_window(pcm, datetime.now())
                        except Exception:  # noqa: BLE001 - one bad window mustn't kill the thread
                            logger.exception("Camera %s: audio window error.", self._camera.id)
                        del ring[:_HOP_BYTES]
            finally:
                self._cleanup_proc()  # close stdout + reap even if the read loop raised
            if self._stop.is_set():
                break
            if delivered:
                backoff = _INITIAL_BACKOFF_S
            logger.warning("Camera %s: audio stream ended; reconnecting in %.1fs.", self._camera.id, backoff)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

    def _read_exact(self, pipe, n: int) -> bytes | None:
        """Read exactly ``n`` bytes (pipe reads are short); None on EOF/stop."""
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            chunk = pipe.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)
