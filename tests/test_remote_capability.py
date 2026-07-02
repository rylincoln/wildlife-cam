"""Unit tests for the pure shared-secret gate helpers."""

from __future__ import annotations

from werkzeug.security import generate_password_hash

from wildlife.remote import capability as cap


def test_cookie_name_constant() -> None:
    assert cap.COOKIE_NAME == "wl_key"


def test_is_loopback() -> None:
    assert cap.is_loopback("127.0.0.1") is True
    assert cap.is_loopback("::1") is True
    assert cap.is_loopback("192.168.1.50") is False
    assert cap.is_loopback(None) is False


def test_secret_ok() -> None:
    h = generate_password_hash("s3cr3t")
    assert cap.secret_ok(h, "s3cr3t") is True
    assert cap.secret_ok(h, "wrong") is False
    assert cap.secret_ok(h, None) is False
    assert cap.secret_ok(None, "s3cr3t") is False


def test_rate_limiter_blocks_after_max() -> None:
    rl = cap.RateLimiter(max_fails=3)
    ip = "203.0.113.7"
    assert rl.blocked(ip) is False
    for _ in range(3):
        rl.record_fail(ip)
    assert rl.blocked(ip) is True
    rl.reset(ip)
    assert rl.blocked(ip) is False


def test_rate_limiter_is_per_instance() -> None:
    a, b = cap.RateLimiter(max_fails=1), cap.RateLimiter(max_fails=1)
    a.record_fail("x")
    assert a.blocked("x") is True
    assert b.blocked("x") is False  # no shared/global state
