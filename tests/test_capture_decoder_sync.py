"""grab_burst waits out mid-GOP decode garbage instead of returning it.

A fresh RTSP connection lands mid-GOP, so FFmpeg hands back neutral-grey 128
plates (and green/magenta chroma failures) until the first keyframe arrives. The
species model hallucinates confident animals on those plates, so grab_burst must
discard them rather than pass them downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from wildlife import capture  # noqa: E402


def _scene(h: int = 120, w: int = 160) -> np.ndarray:
    """A structured, colour-balanced frame that must never read as corrupt."""
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return np.repeat(ramp[:, :, None], 3, axis=2)


def _grey_plate(h: int = 120, w: int = 160) -> np.ndarray:
    """A pre-keyframe plate: neutral 128 with sparse motion-vector residue."""
    frame = np.full((h, w, 3), 128, np.uint8)
    frame[10:20, 10:30] = 240  # the ghost blob the model latches onto
    return frame


class _FakeCapture:
    """cv2.VideoCapture stand-in that replays a scripted list of frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.reads = 0

    def __call__(self, *_args):  # stands in for the constructor
        return self

    def set(self, *_args):
        return True

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if not self._frames:
            return False, None
        frame = self._frames.pop(0)
        return frame is not None, frame

    def release(self):
        pass


class _Camera:
    id = "water_tower"
    rtsp_main = "rtsp://cam/main"
    rtsp_sub = "rtsp://cam/sub"


def test_grey_plate_is_corrupt():
    """The residue-carrying grey plate that the old variance test let through."""
    plate = _grey_plate()
    assert plate.var() > 10.0, "this plate must survive a bare variance>10 test"
    assert capture.is_corrupt_frame(plate) is True


def test_chroma_failure_is_corrupt():
    """Green and magenta plates (chroma decoded without luma) are corrupt."""
    green = np.zeros((120, 160, 3), np.uint8)
    green[:, :, 1] = 136
    magenta = np.full((120, 160, 3), 30, np.uint8)
    magenta[:, :, 0] = 150
    magenta[:, :, 2] = 150
    assert capture.is_corrupt_frame(green) is True
    assert capture.is_corrupt_frame(magenta) is True


def test_real_scene_is_not_corrupt():
    """A structured frame, and a near-monochrome IR night frame, both pass."""
    assert capture.is_corrupt_frame(_scene()) is False
    ir = _scene()  # equal channels == IR; imbalance 1.0
    assert capture.is_corrupt_frame(ir) is False


def test_burst_discards_pre_keyframe_plates(monkeypatch):
    """Grey plates ahead of the first real frame are dropped, not returned."""
    fake = _FakeCapture([_grey_plate()] * 6 + [_scene()] * 3)
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    frames = capture.grab_burst(_Camera(), 3, 0, "main", 10)
    assert len(frames) == 3
    assert all(not capture.is_corrupt_frame(f) for f in frames)
    assert fake.reads == 9  # 6 discarded during sync + 3 real


def test_burst_returns_empty_when_decoder_never_syncs(monkeypatch):
    """An all-garbage stream yields nothing rather than plates to hallucinate on."""
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture([_grey_plate()] * 400))
    monkeypatch.setattr(capture, "_DECODER_SYNC_BUDGET_S", 0.3)
    assert capture.grab_burst(_Camera(), 5, 0, "main", 10) == []
