"""Unit tests for the camera-probe URL redaction (hardware-free).

Importing :mod:`wildlife.admin.probe` must not require cv2/reolink -- those are
imported lazily inside the probe functions, so only :func:`_redact` is exercised
here.
"""

from __future__ import annotations

import pytest

from wildlife.admin.probe import _redact


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://admin:simple@192.168.1.100:554/Preview_01_sub",
        "rtsp://admin:pa/ss@192.168.1.100:554/Preview_01_sub",  # '/' in password
        "rtsp://admin:pa?ss@192.168.1.100:554/Preview_01_sub",  # '?' in password
        "rtsp://admin:pa#ss@192.168.1.100:554/Preview_01_sub",  # '#' in password
        "rtsp://admin:pa:ss@192.168.1.100:554/Preview_01_sub",  # ':' in password
    ],
)
def test_password_is_masked_even_with_delimiters(url: str) -> None:
    out = _redact(url)
    assert "****" in out
    # None of the password variants leak.
    for secret in ("simple", "pa/ss", "pa?ss", "pa#ss", "pa:ss"):
        assert secret not in out
    assert out.startswith("rtsp://admin:****@192.168.1.100:554/")


def test_credentialless_url_untouched() -> None:
    url = "rtsp://192.168.1.100:554/Preview_01_sub"
    assert _redact(url) == url
