"""grab_burst honours an explicit rtsp_url override (cv2 monkeypatched)."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from wildlife import capture  # noqa: E402


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in that records the URL and opens 'closed'."""

    opened_urls: list[str] = []

    def __init__(self, url, _backend):
        _FakeCapture.opened_urls.append(url)

    def set(self, *_args):
        return True

    def isOpened(self):
        return False  # short-circuits grab_burst -> returns [] fast

    def release(self):
        pass


class _Camera:
    id = "north_field"
    rtsp_main = "rtsp://cam/main"
    rtsp_sub = "rtsp://cam/sub"


def test_explicit_rtsp_url_is_used(monkeypatch):
    _FakeCapture.opened_urls = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    capture.grab_burst(_Camera(), 3, 100, "main", 5, rtsp_url="rtsp://127.0.0.1:8554/x_main")
    assert _FakeCapture.opened_urls == ["rtsp://127.0.0.1:8554/x_main"]


def test_no_override_falls_back_to_select_url(monkeypatch):
    _FakeCapture.opened_urls = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    capture.grab_burst(_Camera(), 3, 100, "sub", 5)
    assert _FakeCapture.opened_urls == ["rtsp://cam/sub"]
