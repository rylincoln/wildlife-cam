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

# --- Decoder sync -----------------------------------------------------------
# A fresh RTSP connection lands mid-GOP: FFmpeg starts handing us inter-frames
# whose reference frames it never saw ("Could not find ref with POC N"), so it
# emits a neutral-grey 128 plate carrying only the motion-vector residue -- and
# on a chroma failure, a green or magenta plate. On this rig's 4K HEVC Reolink
# main stream the first keyframe can be over a second out, which swallowed a
# whole 5-frame/200ms burst: measured 19 of 20 frames grey before this fix.
#
# So after opening we read and DISCARD frames until the decoder produces a real
# one, then start the burst. Measured sync cost: 0.2-1.5s (4-43 frames).
_DECODER_SYNC_BUDGET_S = 5.0
# A pre-keyframe grey plate is judged on a coarse grid of cell means, not on
# pixels. Averaging a cell washes out the motion-vector residue a plate carries
# and leaves its flat DC level near 128. A per-pixel test (within +/-4 of 128)
# missed partly decoded frames whose residue had textured the plate and drifted
# it to ~133: only 0.34-0.42 of their pixels qualified, so a coyote, a person and
# a raccoon were saved on plates. Fraction of cells within +/- tolerance of 128,
# measured 2026-10-18 over the 250 captures on disk: real scenes 0.00-0.23 (day,
# IR night, sky-heavy fenceline views), decode artifacts 0.50-1.00.
_GREY_PLATE_GRID = (64, 36)  # (w, h) cells; ~60 px squares on the 4K main stream
_GREY_PLATE_LEVEL = 128
_GREY_PLATE_TOLERANCE = 8
_MAX_GREY_PLATE_FRAC = 0.35
# Ratio of the brightest to the dimmest channel mean above which the frame is a
# chroma decode failure (the green/magenta plates). Measured: real frames top out
# at 1.19 (IR night frames are ~1.00), corrupt ones start at 2.02.
_MAX_CHANNEL_IMBALANCE = 1.5
# Floor for a totally featureless plate that somehow dodges both tests above.
_MIN_FRAME_VARIANCE = 10.0


def is_corrupt_frame(frame: np.ndarray) -> bool:
    """True if ``frame`` is a decode artifact rather than a picture of the scene.

    Catches the three shapes a mid-GOP RTSP open produces, in order of how often
    they showed up in the saved captures:

    1. a neutral-grey 128 plate (no reference frame -- residue only), including a
       partly decoded one where only some regions have been refreshed,
    2. a green/magenta plate (chroma planes decoded without luma, or vice versa),
    3. a completely flat plate of any colour.

    These matter because the species model happily hallucinates on them: every
    one of the 215 corrupt frames found in the capture archive had been saved
    with a confident animal label on it. Any error is treated as "not corrupt"
    so a guard can never take the pipeline down.
    """
    try:
        if frame is None or frame.size == 0:
            return True
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cells = cv2.resize(grey, _GREY_PLATE_GRID, interpolation=cv2.INTER_AREA).astype(np.int16)
        near_neutral = np.abs(cells - _GREY_PLATE_LEVEL) <= _GREY_PLATE_TOLERANCE
        if float(near_neutral.mean()) > _MAX_GREY_PLATE_FRAC:
            return True
        means = [float(frame[:, :, i].mean()) for i in range(frame.shape[2])]
        if max(means) / max(min(means), 1e-6) > _MAX_CHANNEL_IMBALANCE:
            return True
        return float(frame.var()) < _MIN_FRAME_VARIANCE
    except Exception:  # noqa: BLE001 - a guard must never crash the pipeline
        return False


def _sync_decoder(
    cap: "cv2.VideoCapture", camera_id: str, deadline: float
) -> tuple[np.ndarray | None, int]:
    """Discard pre-keyframe garbage; return the first real frame and the drop count.

    Returns ``(None, n)`` if the sync budget or the overall burst deadline expires
    without a clean frame -- the caller then treats the burst as a failed capture
    rather than feeding grey plates to the detector.
    """
    budget_end = min(deadline, time.monotonic() + _DECODER_SYNC_BUDGET_S)
    discarded = 0
    while time.monotonic() < budget_end:
        try:
            ok, frame = cap.read()
        except cv2.error as exc:  # pragma: no cover - backend dependent
            logger.debug("Camera %s: read error during decoder sync: %s", camera_id, exc)
            ok, frame = False, None
        if not ok or frame is None or frame.size == 0:
            time.sleep(_READ_RETRY_SLEEP_S)
            continue
        if not is_corrupt_frame(frame):
            return frame.copy(), discarded
        discarded += 1
    return None, discarded


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
    we get *current* frames rather than a stale queue. It waits for the decoder
    to sync (see :func:`_sync_decoder` -- a fresh RTSP connection lands mid-GOP
    and hands back grey/green plates until the first keyframe), then pulls up to
    ``n`` frames spaced ``interval_ms`` apart, retrying transient read failures
    within each slot, and **always** releases the capture before returning.

    Every returned frame has passed :func:`is_corrupt_frame`, so callers get real
    pictures of the scene or nothing at all.

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
        ``n`` (including empty) if the stream won't open, the decoder never
        syncs, reads fail, or the timeout is hit.
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

        # Wait out the mid-GOP garbage before the burst starts, otherwise the
        # whole burst can be grey plates (see _DECODER_SYNC_BUDGET_S). The frame
        # that ends the sync is a real one, so it becomes the burst's first slot
        # rather than being thrown away.
        first, discarded = _sync_decoder(cap, camera_id, deadline)
        if first is None:
            logger.warning(
                "Camera %s: decoder never synced (%d corrupt frame(s) discarded in "
                "%.1fs); no usable frames.",
                camera_id,
                discarded,
                _DECODER_SYNC_BUDGET_S,
            )
            return frames
        if discarded:
            logger.debug(
                "Camera %s: decoder synced after discarding %d pre-keyframe frame(s).",
                camera_id,
                discarded,
            )
        frames.append(first)

        next_grab = time.monotonic() + interval_s
        for i in range(1, n):
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
            if frame is not None and is_corrupt_frame(frame):
                logger.debug(
                    "Camera %s: dropping corrupt frame %d/%d (decode artifact).",
                    camera_id,
                    i + 1,
                    n,
                )
                frame = None
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
