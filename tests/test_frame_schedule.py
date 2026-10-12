"""Night-dimming window maths for the /frame kiosk."""

from __future__ import annotations

from datetime import time

import pytest

from wildlife.frame.schedule import in_window


@pytest.mark.parametrize(
    "now,expected",
    [
        (time(21, 59), False),
        (time(22, 0), True),
        (time(23, 30), True),
        (time(0, 0), True),
        (time(6, 29), True),
        (time(6, 30), False),
        (time(12, 0), False),
    ],
)
def test_overnight_window_wraps_midnight(now: time, expected: bool) -> None:
    assert in_window(now, "22:00", "06:30") is expected


@pytest.mark.parametrize(
    "now,expected",
    [(time(8, 59), False), (time(9, 0), True), (time(16, 59), True), (time(17, 0), False)],
)
def test_same_day_window(now: time, expected: bool) -> None:
    assert in_window(now, "09:00", "17:00") is expected


def test_equal_start_and_end_is_never_active() -> None:
    assert in_window(time(10, 0), "10:00", "10:00") is False
