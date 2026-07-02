"""Pure helpers for the shared-secret 'anyone with the link' remote-access gate.

Hardware-free logic used by the gallery's request hooks: telling tunnel traffic
from LAN, verifying the secret, and a tiny per-instance failure limiter. The Flask
wiring lives in :mod:`wildlife.gallery.app`.
"""

from __future__ import annotations

from werkzeug.security import check_password_hash

#: Cookie carrying the validated share secret. The Cloudflare WAF rule guarding the
#: /go2rtc path matches this exact name/value, so do not rename without updating it.
COOKIE_NAME = "wl_key"

#: The local cloudflared connector proxies the gallery from loopback, so a loopback
#: remote_addr means "this request arrived via the tunnel".
_LOOPBACK = frozenset({"127.0.0.1", "::1"})


def is_loopback(remote_addr: str | None) -> bool:
    """True if ``remote_addr`` is loopback (request arrived via cloudflared)."""
    return remote_addr in _LOOPBACK


def secret_ok(secret_hash: str | None, provided: str | None) -> bool:
    """Constant-time check that ``provided`` matches the stored Werkzeug hash."""
    if not secret_hash or not provided:
        return False
    return check_password_hash(secret_hash, provided)


class RateLimiter:
    """Cap failed attempts per IP. In-memory, per gallery process, per instance.

    A 256-bit secret already makes brute force infeasible; this only bounds log
    noise / abuse. State is per instance so tests don't leak across each other.
    A blocked IP is treated exactly like a bad key (404) so the gate stays a
    non-oracle.
    """

    def __init__(self, max_fails: int = 20) -> None:
        self.max_fails = max_fails
        self._fails: dict[str, int] = {}

    def blocked(self, ip: str | None) -> bool:
        """True once ``ip`` has exceeded the failure cap."""
        return self._fails.get(ip or "", 0) >= self.max_fails

    def record_fail(self, ip: str | None) -> None:
        """Count one failed attempt for ``ip``."""
        key = ip or ""
        self._fails[key] = self._fails.get(key, 0) + 1

    def reset(self, ip: str | None) -> None:
        """Clear an IP's failure count (call after a successful auth)."""
        self._fails.pop(ip or "", None)
