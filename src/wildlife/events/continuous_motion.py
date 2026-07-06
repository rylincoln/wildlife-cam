"""Continuous, motion-gated event source.

A second per-camera producer that turns the fine-tuned Mac-side YOLO into the
always-on arbiter. It reads the go2rtc **sub** restream, runs a cheap
:class:`~wildlife.motion.MotionDetector` on sampled frames, and emits a
``CameraEvent(kind="motion_continuous")`` on a motion rising edge — which the
worker's existing consumer turns into a capture->YOLO->gate->save pass.

Threading/queue/backoff machinery is inherited verbatim from
:class:`~wildlife.events.base._QueueBackedEventSource` (exactly as
``ReolinkEventSource`` does); only :meth:`_run` is implemented here. The temporal
decision (rising edge, refractory, warmup, active-hours, scene-change reset) is
factored into the pure :meth:`_consider` so it is unit-testable without a camera.

``cv2``, :class:`~wildlife.motion.MotionDetector`, and ``capture._apply_timeout``
are imported lazily inside :meth:`_run` so importing this module never requires
the ``detect`` extra — keeping the temporal tests hardware-free.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime

from wildlife.events.base import CameraEvent, _QueueBackedEventSource

__all__ = ["ContinuousMotionEventSource", "EVENT_KIND"]

logger = logging.getLogger(__name__)

# CameraEvent.kind emitted by this source; the worker routes bursts by matching
# this exact value. (Distinct from the *source* kind "continuous_motion".)
EVENT_KIND = "motion_continuous"

# RTSP open/read timeout (ms) for the persistent motion reader; a wedged read
# fails within this window so the outer loop reopens (acts as the watchdog).
_RTSP_TIMEOUT_MS = 10_000
# Reconnect backoff bounds when the restream will not open / ends.
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def _parse_port(listen: str) -> int:
    """Extract the port from a go2rtc bind string like ``":8554"`` / ``"0.0.0.0:8554"``."""
    return int(listen.rsplit(":", 1)[-1])


def _parse_active_hours(spec: str) -> tuple[int, int] | None:
    """Parse ``"HH:MM-HH:MM"`` into (start_minute, end_minute); ``""`` -> None (24/7)."""
    spec = spec.strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", spec)
    if not m:  # config validation should already guarantee the format
        return None
    sh, sm, eh, em = (int(g) for g in m.groups())
    return (sh * 60 + sm, eh * 60 + em)


class ContinuousMotionEventSource(_QueueBackedEventSource):
    """Emit ``motion_continuous`` events from a go2rtc sub restream's motion.

    Parameters
    ----------
    camera:
        The camera's :class:`wildlife.config.CameraConfig` (needs ``id`` and,
        optionally, ``motion_mask``).
    config:
        The full validated :class:`wildlife.config.Config` — this source reads
        ``config.continuous`` (its knobs), ``config.livestream.rtsp_listen`` (the
        restream port), and ``camera.motion_mask`` (ignore polygons).
    """

    def __init__(self, camera: object, config: object) -> None:
        super().__init__(camera)
        cc = config.continuous
        self._sample_interval_s = 1.0 / max(1, cc.sample_fps)
        self._downscale_width = cc.downscale_width
        self._min_area_frac = cc.min_area_frac
        self._algorithm = cc.algorithm
        self._refractory_s = cc.refractory_s
        self._warmup_s = cc.warmup_s
        self._mask_polys = getattr(camera, "motion_mask", None) or []
        self._rtsp_port = _parse_port(config.livestream.rtsp_listen)
        self._active_window = _parse_active_hours(cc.active_hours)

        # Temporal state (monotonic clock for warmup/refractory).
        self._active = False  # previous frame's motion state (for rising edge)
        self._last_emit_mono: float | None = None
        self._warmup_until_mono = 0.0
        self._detector = None  # built in _run() (needs cv2)

    # -- pure decision logic (unit-tested without a camera) ---------------

    def _within_active_hours(self, now_wall: datetime) -> bool:
        """True if ``now_wall`` falls in the configured window (or none is set)."""
        if self._active_window is None:
            return True
        start, end = self._active_window
        cur = now_wall.hour * 60 + now_wall.minute
        if start <= end:
            return start <= cur < end
        return cur >= start or cur < end  # window wraps past midnight

    def _consider(self, result, now_mono: float, now_wall: datetime) -> str | None:
        """Fold one motion result into the temporal state; return the action.

        Returns ``"emit"`` (fire a CameraEvent), ``"reset"`` (rebuild the motion
        model after a scene change), or ``None`` (do nothing). ``result`` needs
        ``.motion`` and ``.scene_change`` attributes.
        """
        if not self._within_active_hours(now_wall):
            self._active = bool(result.motion)  # track edge; never emit off-hours
            return None

        if result.scene_change:
            # Whole-scene shift: rebuild the model and start a fresh refractory so
            # the transient does not read as a lasting field of motion.
            self._active = False
            self._last_emit_mono = now_mono
            return "reset"

        motion = bool(result.motion)
        rising = motion and not self._active
        self._active = motion
        if not rising:
            return None
        if now_mono < self._warmup_until_mono:
            return None  # model still stabilising after (re)connect
        if (
            self._last_emit_mono is not None
            and (now_mono - self._last_emit_mono) < self._refractory_s
        ):
            return None  # one animal -> one event
        self._last_emit_mono = now_mono
        return "emit"

    def _arm_warmup(self) -> None:
        """Suppress emits for ``warmup_s`` and clear the edge after (re)connect."""
        self._warmup_until_mono = time.monotonic() + self._warmup_s
        self._active = False

    # -- worker-thread run loop (hardware; lazy heavy imports) ------------

    def _run(self) -> None:
        """Open the sub restream, sample motion, and emit rising-edge events."""
        import cv2  # lazy: confine the heavy decode dependency to runtime

        from wildlife.motion import MotionDetector

        url = f"rtsp://127.0.0.1:{self._rtsp_port}/{self.camera.id}_sub"
        backoff = _INITIAL_BACKOFF_S
        while not self._stopping:
            cap = self._open_capture(cv2, url)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                logger.warning(
                    "Camera %s: continuous restream will not open (%s); retry in %.1fs.",
                    self.camera.id,
                    url,
                    backoff,
                )
                self._sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue

            self._detector = MotionDetector(
                self._downscale_width,
                self._min_area_frac,
                self._algorithm,
                self._mask_polys,
            )
            self._arm_warmup()
            logger.info("Camera %s: continuous motion source reading %s.", self.camera.id, url)
            delivered = False
            try:
                delivered = self._read_loop(cap)
            except Exception:  # noqa: BLE001 - reopen rather than die
                logger.exception("Camera %s: continuous read loop error.", self.camera.id)
            finally:
                cap.release()

            if self._stopping:
                break
            # Only a stream that actually delivered frames earns a fast reconnect;
            # an open-then-immediate-EOF keeps growing backoff so we don't churn
            # RTSP sessions when go2rtc is up but the sub stream is unavailable.
            if delivered:
                backoff = _INITIAL_BACKOFF_S
            logger.warning(
                "Camera %s: continuous stream ended; reconnecting in %.1fs.",
                self.camera.id,
                backoff,
            )
            self._sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

    def _open_capture(self, cv2, url: str):
        """Open a TCP-forced, single-buffered FFmpeg capture with open/read timeouts."""
        from wildlife.capture import _apply_timeout

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        _apply_timeout(cap, _RTSP_TIMEOUT_MS // 1000)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - backend dependent
            pass
        return cap

    # `_signal_stop` is intentionally NOT overridden: a `cv2.VideoCapture.grab()`
    # cannot be safely cancelled cross-thread, so on shutdown a wedged read may
    # delay this daemon thread's exit up to the ~10s RTSP read timeout
    # (`CAP_PROP_READ_TIMEOUT_MSEC`). The process is never blocked by this since
    # the thread is a daemon and the queue consumer is unblocked immediately by
    # the sentinel.
    def _read_loop(self, cap) -> bool:
        """Grab frames (paced by the stream), decode at sample_fps, emit on motion.

        Returns True if at least one frame was successfully retrieved (a healthy
        stream), so the caller can distinguish a real stream that later ended
        from an open-then-immediate-EOF that should back off.
        """
        last_sample = 0.0
        delivered = False
        while not self._stopping:
            # grab() blocks on the socket -> the loop is paced by the stream, not a
            # busy spin; a wedged read fails the RTSP timeout and ends the loop.
            if not cap.grab():
                return delivered  # stream ended -> outer loop reopens
            now = time.monotonic()
            if (now - last_sample) < self._sample_interval_s:
                continue  # drain to the newest frame without decoding
            last_sample = now
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            delivered = True
            result = self._detector.update(frame)
            action = self._consider(result, time.monotonic(), datetime.now())
            if action == "reset":
                self._detector.reset()
                self._arm_warmup()
            elif action == "emit":
                self._emit(
                    CameraEvent(
                        camera_id=self.camera.id,
                        event_ts=datetime.now(),
                        kind=EVENT_KIND,
                    )
                )
                logger.info(
                    "Camera %s: continuous motion event (area_frac=%.4f).",
                    self.camera.id,
                    result.area_frac,
                )
        return delivered
