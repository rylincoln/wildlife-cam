"""RTSP burst-capture for the wildlife detector.

This module owns the *stream lifecycle* for grabbing a short burst of frames
from a Reolink camera on a motion/animal event. The single rule that matters
for Reolink hardware: **open -> grab -> close**, every time. Reolink cameras
drop connections when hit by multiple concurrent RTSP sessions from the same
host IP, so we open a fresh capture, pull the burst, and *always* release it in
a ``finally`` block.

The public surface is a single function, :func:`grab_burst`, which returns a
list of BGR ``numpy`` frames (OpenCV's native channel order). The list may be
shorter than the requested count if reads fail transiently or the overall
timeout is exceeded — callers should treat the result as best-effort and cope
with an empty list.

Only ``cv2``/``numpy`` (plus stdlib) are imported here; the heavyweight import
is deliberately confined to this hardware module so the pure-logic modules stay
test-friendly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # avoid importing pydantic config at runtime for typing only
    from wildlife.config import CameraConfig

logger = logging.getLogger(__name__)

# Per-frame read attempts before we give up on a single burst slot. RTSP/HEVC
# decode can hiccup on the first reads after a connection opens; a couple of
# quick retries smooths over transient ``cap.read()`` failures without stalling
# the whole burst.
_READ_RETRIES_PER_FRAME = 3
# Short pause between failed reads within a slot so we don't spin the CPU.
_READ_RETRY_SLEEP_S = 0.05
# FFmpeg capture options applied (if the user hasn't set their own) to force
# TCP RTSP transport, which is markedly more reliable than UDP for Reolink
# H.265 streams over a LAN.
_FFMPEG_CAPTURE_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_DEFAULT_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"


def _select_url(camera: "CameraConfig", stream: str) -> str:
    """Pick the main or sub RTSP URL off the camera config.

    ``camera.rtsp_main`` / ``camera.rtsp_sub`` are already credential-interpolated
    by :class:`wildlife.config.CameraConfig`, so this is a straight selection.
    Anything other than ``"sub"`` falls back to the main stream.
    """
    if stream == "sub":
        return camera.rtsp_sub
    if stream != "main":
        logger.warning(
            "Unknown stream %r for camera %s; defaulting to 'main'.",
            stream,
            getattr(camera, "id", "?"),
        )
    return camera.rtsp_main


def _apply_timeout(cap: "cv2.VideoCapture", timeout_s: int) -> None:
    """Best-effort open/read timeouts on the FFmpeg backend.

    ``CAP_PROP_OPEN_TIMEOUT_MSEC`` / ``CAP_PROP_READ_TIMEOUT_MSEC`` exist on
    modern OpenCV builds but aren't guaranteed, so we set them defensively via
    ``getattr`` and ignore any that are missing.
    """
    timeout_ms = max(1, int(timeout_s * 1000))
    for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            try:
                cap.set(prop, timeout_ms)
            except cv2.error:  # pragma: no cover - backend dependent
                logger.debug("Could not set %s on capture.", prop_name)


def grab_burst(
    camera: "CameraConfig",
    n: int,
    interval_ms: int,
    stream: str,
    timeout_s: int,
    rtsp_url: str | None = None,
) -> list[np.ndarray]:
    """Open the camera's RTSP stream, grab ``n`` frames, and close it.

    Opens a fresh ``cv2.VideoCapture`` against ``rtsp_main`` or ``rtsp_sub``
    (selected by ``stream``) using the FFmpeg backend, with a 1-frame buffer so
    we get *current* frames rather than a stale queue. It then pulls up to ``n``
    frames spaced ``interval_ms`` apart, retrying transient read failures within
    each slot, and **always** releases the capture before returning.

    Args:
        camera: Camera config with interpolated ``rtsp_main``/``rtsp_sub`` URLs.
        n: Number of frames to attempt to grab.
        interval_ms: Target spacing between successive frame grabs.
        stream: ``"main"`` or ``"sub"`` — which RTSP URL to open.
        timeout_s: Overall wall-clock budget for the burst (also applied as the
            FFmpeg open/read timeout where supported).
        rtsp_url: Optional explicit RTSP URL to open instead of the camera's
            own main/sub URL — used to route continuous bursts through go2rtc.

    Returns:
        A list of BGR ``numpy.ndarray`` frames, newest-last. May be shorter than
        ``n`` (including empty) if the stream won't open, reads fail, or the
        timeout is hit.
    """
    if n <= 0:
        logger.warning("grab_burst called with n=%d; nothing to do.", n)
        return []

    camera_id = getattr(camera, "id", "?")
    url = rtsp_url if rtsp_url else _select_url(camera, stream)
    interval_s = max(0.0, interval_ms / 1000.0)
    deadline = time.monotonic() + max(0.0, float(timeout_s))

    # Force TCP RTSP transport for Reolink reliability unless the operator has
    # explicitly configured their own FFmpeg capture options.
    os.environ.setdefault(_FFMPEG_CAPTURE_OPTIONS_ENV, _DEFAULT_FFMPEG_CAPTURE_OPTIONS)

    frames: list[np.ndarray] = []
    cap: cv2.VideoCapture | None = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        _apply_timeout(cap, timeout_s)
        # Keep only the freshest frame buffered so motion events grab "now".
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - backend dependent
            logger.debug("Camera %s: backend rejected BUFFERSIZE=1.", camera_id)

        if not cap.isOpened():
            logger.error(
                "Camera %s: failed to open RTSP stream (%s).", camera_id, stream
            )
            return frames

        logger.info(
            "Camera %s: opened %s stream, grabbing %d frame(s) every %dms.",
            camera_id,
            stream,
            n,
            interval_ms,
        )

        next_grab = time.monotonic()
        for i in range(n):
            # Pace the burst: wait until this slot's scheduled time, but never
            # past the overall deadline.
            now = time.monotonic()
            if now >= deadline:
                logger.warning(
                    "Camera %s: burst timed out after %d/%d frame(s).",
                    camera_id,
                    len(frames),
                    n,
                )
                break
            if next_grab > now:
                time.sleep(min(next_grab - now, deadline - now))

            frame = _read_one(cap, camera_id, slot=i, deadline=deadline)
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(
                    "Camera %s: dropping frame %d/%d after retries.",
                    camera_id,
                    i + 1,
                    n,
                )

            next_grab += interval_s

        logger.info(
            "Camera %s: burst complete, captured %d/%d frame(s).",
            camera_id,
            len(frames),
            n,
        )
        return frames
    except cv2.error as exc:
        logger.error("Camera %s: OpenCV error during burst: %s", camera_id, exc)
        return frames
    finally:
        # The whole point of this module: never leak the RTSP session.
        if cap is not None:
            cap.release()
            logger.debug("Camera %s: released RTSP capture.", camera_id)


def _read_one(
    cap: "cv2.VideoCapture",
    camera_id: str,
    slot: int,
    deadline: float,
) -> np.ndarray | None:
    """Read a single frame, retrying transient failures up to a small cap.

    Returns the BGR frame, or ``None`` if every attempt failed or the deadline
    passed. A copy is returned so the frame stays valid independent of OpenCV's
    internal buffer reuse.
    """
    for attempt in range(_READ_RETRIES_PER_FRAME):
        if time.monotonic() >= deadline:
            return None
        try:
            ok, frame = cap.read()
        except cv2.error as exc:  # pragma: no cover - backend dependent
            logger.debug(
                "Camera %s: read error on slot %d attempt %d: %s",
                camera_id,
                slot,
                attempt + 1,
                exc,
            )
            ok, frame = False, None

        if ok and frame is not None and frame.size > 0:
            # Copy out of OpenCV's reused capture buffer.
            return frame.copy()

        logger.debug(
            "Camera %s: empty read on slot %d (attempt %d/%d), retrying.",
            camera_id,
            slot,
            attempt + 1,
            _READ_RETRIES_PER_FRAME,
        )
        time.sleep(_READ_RETRY_SLEEP_S)

    return None
