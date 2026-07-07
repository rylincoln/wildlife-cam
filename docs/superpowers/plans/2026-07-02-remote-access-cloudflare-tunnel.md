# Remote Access via Cloudflare Tunnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the wildlife-cam gallery + live view at one Cloudflare-Tunnel hostname, protected by a single shared-secret link ("anyone with the link"), with `/admin` unreachable remotely and LAN behavior unchanged.

**Architecture:** A named Cloudflare Tunnel points `cam.example.com/*` at the existing Flask gallery (`localhost:8080`) and `cam.example.com/go2rtc/*` at go2rtc (`localhost:1984`, served under `api.base_path=/go2rtc`). The Flask app gates its own routes on **tunnel traffic** (detected by a loopback `remote_addr`) using a shared secret passed as `?key=` then carried in a cookie; a Cloudflare WAF rule gates the `/go2rtc` path with the same cookie. Live view uses go2rtc's MSE player (WebRTC can't cross the tunnel). All new gating is inert unless `remote.enabled` is set, so existing behavior is untouched.

**Tech Stack:** Python 3.12, Flask/Werkzeug (existing), pydantic v2 (existing), PyYAML (existing), `secrets` (stdlib). Config mutations reuse `wildlife.admin.config_io` (ruamel round-trip). go2rtc + cloudflared are external and configured via a runbook.

## Global Constraints

- **Python 3.12** (3.11–3.13 supported). Keep `wildlife.config`, `wildlife.stream.config_gen`, and new pure helpers free of torch/cv2/numpy so hardware-free tests keep importing them.
- **No new runtime dependencies.** Uses stdlib `secrets` and the already-present `werkzeug`/`pyyaml`. The new `wildlife-share-secret` CLI writes config through `wildlife.admin.config_io`, which needs the `admin` extra's `ruamel.yaml` — exactly like the existing `wildlife-admin-password` CLI. Do not add packages.
- **Follow existing patterns:** pydantic v2 models with `Field`/validators in `config.py`; config writes via `config_io.write`/`set_*`; the CLI mirrors `src/wildlife/admin/passwd.py`; tests are hardware-free under `tests/`, using the Flask **test client with `environ_base={"REMOTE_ADDR": ...}`** to simulate LAN vs tunnel.
- **"Arrived via the tunnel" == loopback `remote_addr`** (`127.0.0.1`/`::1`), because the local cloudflared connector proxies from loopback and a remote attacker cannot reach `:8080` directly. Assumes cloudflared runs on the same Mac as the gallery.
- **Secret:** 256-bit `secrets.token_urlsafe(32)`, stored **only as a Werkzeug hash** in `config.yaml`. Cookie name is exactly `wl_key` (the Cloudflare WAF rule matches this name/value).
- **The remote gate applies ONLY to tunnel (loopback) traffic when `remote.enabled` is true.** LAN traffic is never gated. `/admin` returns **404** over the tunnel.
- **Camera codec (operator step, documented, no code):** each Reolink **sub-stream must be H.264 Main/Baseline** so MSE plays cross-browser incl. Safari.
- **This dev box ≠ production.** The system runs on a separate Mac mini. All Python is hardware-free and unit-tested here; the prod-only shell script (`scripts/setup_remote.sh`, Task 7) cannot be fully executed here — it is verified with `bash -n` + `shellcheck` only, and run for real by the user on prod.

---

### Task 1: `RemoteConfig` model + `livestream.base_path` (inert config)

**Files:**
- Modify: `src/wildlife/config.py` (add `base_path` to `LivestreamConfig`; add `RemoteConfig`; add `remote` to `Config`; extend `__all__`)
- Test: `tests/test_config_remote.py` (create)

**Interfaces:**
- Produces: `wildlife.config.RemoteConfig(enabled: bool = False, base_url: str = "", share_secret_hash: str | None = None, block_admin: bool = True)`; `Config.remote: RemoteConfig`; `LivestreamConfig.base_path: str = ""` (normalized: `""` or a leading-slash, no-trailing-slash path).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_remote.py`:

```python
"""Validation tests for the RemoteConfig block and livestream.base_path."""

from __future__ import annotations

from wildlife.config import Config, LivestreamConfig, RemoteConfig


def _minimal_config_dict() -> dict:
    """A minimal mapping that Config.model_validate accepts (no cameras needed)."""
    return {
        "cameras": [],
        "event_source": "reolink_native",
        "capture": {
            "burst_frames": 5, "burst_interval_ms": 200, "stream": "main",
            "rtsp_timeout_s": 10, "max_concurrent": 1,
        },
        "detection": {
            "model_path": "models/yolov8s.pt", "device": "cpu",
            "animal_classes": ["bird"], "confidence_threshold": 0.5,
            "min_box_area_frac": 0.01, "save_best_only": True,
        },
        "dedupe": {"cooldown_s": 30},
        "storage": {"captures_dir": "/tmp/wl-caps", "db_path": "/tmp/wl.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {},
    }


def test_remote_defaults_disabled() -> None:
    cfg = Config.model_validate(_minimal_config_dict())
    assert cfg.remote.enabled is False
    assert cfg.remote.share_secret_hash is None
    assert cfg.remote.block_admin is True
    assert cfg.remote.base_url == ""


def test_remote_block_parses() -> None:
    data = _minimal_config_dict()
    data["remote"] = {
        "enabled": True, "base_url": "https://cam.example.com",
        "share_secret_hash": "pbkdf2:sha256:xxx", "block_admin": True,
    }
    cfg = Config.model_validate(data)
    assert cfg.remote.enabled is True
    assert cfg.remote.base_url == "https://cam.example.com"


def test_base_path_defaults_empty() -> None:
    assert LivestreamConfig().base_path == ""


def test_base_path_normalized() -> None:
    assert LivestreamConfig(base_path="go2rtc").base_path == "/go2rtc"
    assert LivestreamConfig(base_path="/go2rtc/").base_path == "/go2rtc"
    assert LivestreamConfig(base_path="  ").base_path == ""


def test_remote_model_direct() -> None:
    r = RemoteConfig(enabled=True, base_url="https://x")
    assert r.enabled and r.base_url == "https://x" and r.block_admin is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_remote.py -v`
Expected: FAIL — `ImportError: cannot import name 'RemoteConfig'` (and `base_path` AttributeError).

- [ ] **Step 3: Write minimal implementation**

In `src/wildlife/config.py`, add `base_path` to `LivestreamConfig` (after `sub_mode`):

```python
    base_path: str = ""  # serve go2rtc under a sub-path (e.g. "/go2rtc") for tunnel exposure

    @field_validator("base_path")
    @classmethod
    def _normalize_base_path(cls, value: str) -> str:
        """Normalize a go2rtc sub-path: '' stays '', else leading slash, no trailing slash."""
        value = value.strip()
        if not value:
            return ""
        if not value.startswith("/"):
            value = "/" + value
        return value.rstrip("/")
```

Add a new model after `AdminConfig`:

```python
class RemoteConfig(BaseModel):
    """Optional Cloudflare-Tunnel remote access gated by a shared-secret link.

    When ``enabled``, requests arriving via the local cloudflared connector (a
    loopback ``remote_addr``) must carry the shared secret -- as ``?key=`` on the
    first hit, then a cookie -- or receive a 404. LAN traffic is unaffected, and
    ``/admin`` is refused over the tunnel regardless of the secret. Only a Werkzeug
    *hash* of the secret is stored (set via ``wildlife-share-secret``).
    """

    enabled: bool = False
    base_url: str = ""  # canonical public URL e.g. "https://cam.example.com" (share links / logs)
    share_secret_hash: str | None = None
    block_admin: bool = True
```

Add to `Config` (after the `admin` field):

```python
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
```

Add `"RemoteConfig"` to `__all__` (after `"AdminConfig"`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_remote.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS (existing tests unaffected — `remote` and `base_path` default to inert values).

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/config.py tests/test_config_remote.py
git commit -m "feat(config): add RemoteConfig + livestream.base_path (inert defaults)"
```

---

### Task 2: `set_remote_secret` + `wildlife-share-secret` CLI

**Files:**
- Modify: `src/wildlife/admin/config_io.py` (add `set_remote_secret`; extend `__all__`)
- Create: `src/wildlife/remote/__init__.py`
- Create: `src/wildlife/remote/share_secret.py`
- Modify: `pyproject.toml` (`[project.scripts]` — add `wildlife-share-secret`)
- Test: `tests/test_remote_share_secret.py` (create)

**Interfaces:**
- Consumes: `Config.remote` (Task 1); `config_io.write`, `config_io._ensure_map`, `config_io.read_raw`, `config_io.ConfigError`.
- Produces: `config_io.set_remote_secret(config_path, secret_hash: str) -> Config` (sets `remote.enabled=True`, `remote.share_secret_hash`, no reload); `wildlife.remote.share_secret.main() -> int` (generates a 256-bit secret, stores its hash, prints the share URL + raw secret).

- [ ] **Step 1: Write the failing test**

Create `tests/test_remote_share_secret.py`:

```python
"""Tests for set_remote_secret and the wildlife-share-secret CLI."""

from __future__ import annotations

from pathlib import Path

from werkzeug.security import check_password_hash

from wildlife.admin import config_io as cio
from wildlife.config import load_config
from wildlife.remote import share_secret
from tests.test_admin_config_io import render_base


def test_set_remote_secret_persists_and_enables(tmp_path: Path) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")

    cio.set_remote_secret(str(cfgp), "pbkdf2:sha256:deadbeef")

    cfg = load_config(cfgp)
    assert cfg.remote.enabled is True
    assert cfg.remote.share_secret_hash == "pbkdf2:sha256:deadbeef"


def test_cli_mints_verifiable_secret_and_prints_link(tmp_path, monkeypatch, capsys) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    # Give it a base_url so the printed link is concrete.
    cio.update_section(str(cfgp), "remote", {"base_url": "https://cam.example.com"})

    monkeypatch.setattr("sys.argv", ["wildlife-share-secret", str(cfgp)])
    assert share_secret.main() == 0

    out = capsys.readouterr().out
    # Extract the printed raw secret and confirm it verifies against the stored hash.
    line = next(ln for ln in out.splitlines() if "Raw secret:" in ln)
    secret = line.split("Raw secret:")[1].strip()
    assert len(secret) >= 40  # token_urlsafe(32) is ~43 chars
    cfg = load_config(cfgp)
    assert check_password_hash(cfg.remote.share_secret_hash, secret)
    assert f"https://cam.example.com/?key={secret}" in out


def test_cli_rotation_invalidates_old(tmp_path, monkeypatch) -> None:
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    monkeypatch.setattr("sys.argv", ["wildlife-share-secret", str(cfgp)])

    share_secret.main()
    first = load_config(cfgp).remote.share_secret_hash
    share_secret.main()
    second = load_config(cfgp).remote.share_secret_hash
    assert first != second  # a fresh secret each run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_remote_share_secret.py -v`
Expected: FAIL — `ImportError` (`set_remote_secret` / `wildlife.remote` don't exist).

- [ ] **Step 3: Write minimal implementation**

In `src/wildlife/admin/config_io.py`, add to `__all__` (after `"set_admin_password"`): `"set_remote_secret"`. Append this function after `set_admin_password`:

```python
def set_remote_secret(config_path: str | Path, secret_hash: str) -> Config:
    """Persist a Werkzeug hash of the remote share secret under ``remote.share_secret_hash``.

    Also enables remote access (``remote.enabled = True``). Does not trigger a
    reload -- the gallery reads ``remote`` live per request; the worker/go2rtc
    don't consume it.
    """

    def _mutate(doc) -> None:
        remote = _ensure_map(doc, "remote")
        remote["enabled"] = True
        remote["share_secret_hash"] = secret_hash

    return write(config_path, _mutate, trigger_reload=False)
```

Create `src/wildlife/remote/__init__.py`:

```python
"""Remote-access (Cloudflare Tunnel) support: shared-secret gate + CLI."""
```

Create `src/wildlife/remote/share_secret.py`:

```python
"""``wildlife-share-secret`` -- mint (or rotate) the remote share-link secret.

Generates a 256-bit URL-safe secret, stores only its Werkzeug *hash* in
``config.yaml`` (enabling ``remote``), and prints -- once -- the shareable URL, the
raw secret (to paste into the Cloudflare WAF rule guarding ``/go2rtc``), and a
note. Re-running rotates the secret and invalidates every previously-shared link.

Usage::

    wildlife-share-secret                 # edits ./config.yaml
    wildlife-share-secret /path/config.yaml
"""

from __future__ import annotations

import secrets
import sys

from werkzeug.security import generate_password_hash

from wildlife.admin.config_io import ConfigError, read_raw, set_remote_secret

_SECRET_NBYTES = 32  # 256 bits


def main() -> int:
    """Generate a new share secret, persist its hash, and print the share URL."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    secret = secrets.token_urlsafe(_SECRET_NBYTES)
    try:
        set_remote_secret(config_path, generate_password_hash(secret))
    except (ConfigError, OSError) as exc:
        print(f"Failed to write {config_path}: {exc}", file=sys.stderr)
        return 1

    base_url = (read_raw(config_path).get("remote") or {}).get("base_url") or "https://<your-host>"
    share_url = f"{base_url.rstrip('/')}/?key={secret}"
    print("Remote share secret set (rotated). Previously-shared links no longer work.\n")
    print(f"  Shareable link:  {share_url}")
    print(f"  Raw secret:      {secret}")
    print("\nPaste the raw secret into the Cloudflare WAF rule guarding /go2rtc/*")
    print("(cookie 'wl_key' must equal this value). Store the link safely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

In `pyproject.toml`, under `[project.scripts]`, add after the `wildlife-admin-password` line:

```toml
# Mint/rotate the remote (Cloudflare Tunnel) share-link secret.
wildlife-share-secret = "wildlife.remote.share_secret:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_remote_share_secret.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wildlife/admin/config_io.py src/wildlife/remote/ pyproject.toml tests/test_remote_share_secret.py
git commit -m "feat(remote): set_remote_secret + wildlife-share-secret CLI"
```

---

### Task 3: capability gate helpers (pure logic)

**Files:**
- Create: `src/wildlife/remote/capability.py`
- Test: `tests/test_remote_capability.py` (create)

**Interfaces:**
- Produces: `capability.COOKIE_NAME = "wl_key"`; `capability.is_loopback(remote_addr: str | None) -> bool`; `capability.secret_ok(secret_hash: str | None, provided: str | None) -> bool`; `capability.RateLimiter(max_fails=20)` with `.blocked(ip) -> bool`, `.record_fail(ip)`, `.reset(ip)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_remote_capability.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_remote_capability.py -v`
Expected: FAIL — `ModuleNotFoundError: wildlife.remote.capability`.

- [ ] **Step 3: Write minimal implementation**

Create `src/wildlife/remote/capability.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_remote_capability.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wildlife/remote/capability.py tests/test_remote_capability.py
git commit -m "feat(remote): pure shared-secret gate helpers"
```

---

### Task 4: wire the shared-secret gate into the gallery

**Files:**
- Modify: `src/wildlife/gallery/app.py` (module import; `_via_tunnel` helper; `before_request` gate; `after_request` cookie + `Referrer-Policy`)
- Test: `tests/test_gallery_remote.py` (create)

**Interfaces:**
- Consumes: `Config.remote` (Task 1); `capability.{COOKIE_NAME, is_loopback, secret_ok, RateLimiter}` (Task 3); existing `create_app`, `get_config`, `set_remote_secret` (Task 2, for the admin test).
- Produces: gallery request hooks that 404 tunnel traffic lacking the secret, set the `wl_key` cookie on a valid `?key=`, add `Referrer-Policy: no-referrer`, and 404 `/admin` over the tunnel. `create_app` gains a closure `_via_tunnel() -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gallery_remote.py`:

```python
"""Tests for the shared-secret remote-access gate on the gallery."""

from __future__ import annotations

from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

from wildlife.config import (
    CameraConfig, CaptureConfig, Config, DedupeConfig, DetectionConfig,
    GalleryConfig, RemoteConfig, ResourceGuardConfig, RetentionConfig, StorageConfig,
)
from wildlife.gallery.app import create_app

_SECRET = "opensesame-abc123"
_LAN = {"REMOTE_ADDR": "192.168.1.50"}   # simulate a LAN client
# (test client default REMOTE_ADDR is 127.0.0.1 == "via tunnel")


def _config(tmp_path: Path, *, remote: RemoteConfig) -> Config:
    return Config(
        cameras=[CameraConfig(
            id="north_field", host="192.168.1.101", username="admin", password="secret",
            rtsp_main="rtsp://{username}:{password}@{host}:554/Preview_01_main",
            rtsp_sub="rtsp://{username}:{password}@{host}:554/Preview_01_sub",
        )],
        event_source="reolink_native",
        capture=CaptureConfig(burst_frames=5, burst_interval_ms=200, stream="main",
                              rtsp_timeout_s=10, max_concurrent=1),
        detection=DetectionConfig(model_path="models/yolov8s.pt", device="cpu",
                                  animal_classes=["bird"], confidence_threshold=0.5,
                                  min_box_area_frac=0.01, save_best_only=True),
        dedupe=DedupeConfig(cooldown_s=30),
        storage=StorageConfig(captures_dir=tmp_path / "captures", db_path=tmp_path / "captures.db"),
        retention=RetentionConfig(max_age_days=30),
        gallery=GalleryConfig(host="0.0.0.0", port=8080, page_size=60),
        resource_guard=ResourceGuardConfig(),
        remote=remote,
    )


def _client(config: Config):
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def _enabled(hash_secret: str = _SECRET) -> RemoteConfig:
    return RemoteConfig(enabled=True, base_url="https://cam.example.com",
                        share_secret_hash=generate_password_hash(hash_secret))


def test_disabled_is_open_over_tunnel(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=RemoteConfig(enabled=False)))
    assert client.get("/").status_code == 200  # remote off => unchanged, open


def test_lan_is_open_even_when_enabled(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/", environ_base=_LAN).status_code == 200


def test_tunnel_without_secret_404(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/").status_code == 404  # default client == tunnel, no key/cookie


def test_tunnel_valid_key_sets_cookie_and_serves(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    resp = client.get(f"/?key={_SECRET}")
    assert resp.status_code == 200
    set_cookie = resp.headers.get("Set-Cookie", "")
    assert "wl_key=" in set_cookie and "HttpOnly" in set_cookie and "Secure" in set_cookie


def test_tunnel_cookie_grants_access(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    client.get(f"/?key={_SECRET}")            # sets the cookie on the client jar
    assert client.get("/").status_code == 200  # subsequent request carries the cookie


def test_tunnel_wrong_key_404(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    assert client.get("/?key=nope").status_code == 404


def test_enabled_without_hash_fails_closed_over_tunnel(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=RemoteConfig(enabled=True, share_secret_hash=None)))
    assert client.get("/").status_code == 404
    assert client.get("/", environ_base=_LAN).status_code == 200  # LAN still open


def test_referrer_policy_header_when_enabled(tmp_path: Path) -> None:
    client = _client(_config(tmp_path, remote=_enabled()))
    resp = client.get(f"/?key={_SECRET}")
    assert resp.headers.get("Referrer-Policy") == "no-referrer"


def test_admin_404_over_tunnel_but_auth_on_lan(tmp_path: Path) -> None:
    # /admin only mounts with a config_path; write a real config file for this one.
    from tests.test_admin_config_io import render_base
    from wildlife.admin import config_io as cio
    from wildlife.config import load_config

    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(render_base(tmp_path), "utf-8")
    cio.set_admin_password(str(cfgp), generate_password_hash("supersecret"))
    cio.set_remote_secret(str(cfgp), generate_password_hash(_SECRET))
    app = create_app(load_config(cfgp), config_path=str(cfgp))
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/admin/").status_code == 404             # via tunnel -> blocked
    assert client.get("/admin/", environ_base=_LAN).status_code == 401  # LAN -> auth challenge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gallery_remote.py -v`
Expected: FAIL — the tunnel requests return 200 (no gate yet) so the 404 assertions fail.

- [ ] **Step 3: Write minimal implementation**

In `src/wildlife/gallery/app.py`, add this import near the other `wildlife` imports (after `from wildlife.store import Store`):

```python
from wildlife.remote import capability as _cap
```

Inside `create_app`, immediately after the `get_config` function definition (before `_new_store`), add:

```python
    _rate_limiter = _cap.RateLimiter()

    def _via_tunnel() -> bool:
        """True when the request arrived via the local cloudflared connector AND
        remote access is enabled -- i.e. the shared-secret gate applies. LAN
        clients (non-loopback remote_addr) and remote-disabled configs are False."""
        return get_config().remote.enabled and _cap.is_loopback(request.remote_addr)

    @app.before_request
    def _remote_gate():
        if not _via_tunnel():
            return None  # LAN or remote-disabled -> unchanged behavior
        remote = get_config().remote
        path = request.path
        if remote.block_admin and (path == "/admin" or path.startswith("/admin/")):
            abort(404)  # /admin is never reachable over the tunnel
        if path.startswith("/static/"):
            return None  # styling assets carry no data; keep the shared page rendered
        if not remote.share_secret_hash:
            logger.warning("remote.enabled but no share_secret_hash set; run wildlife-share-secret")
            abort(404)  # fail closed
        ip = request.headers.get("Cf-Connecting-IP") or request.remote_addr
        if _rate_limiter.blocked(ip):
            abort(404)  # treat a rate-limited IP exactly like a bad key (no oracle)
        provided = request.args.get("key")
        if _cap.secret_ok(remote.share_secret_hash, provided):
            _rate_limiter.reset(ip)
            g._set_share_cookie = provided  # emitted in after_request
            return None
        if _cap.secret_ok(remote.share_secret_hash, request.cookies.get(_cap.COOKIE_NAME)):
            return None
        if provided is not None:
            _rate_limiter.record_fail(ip)
        abort(404)

    @app.after_request
    def _remote_headers(resp):
        if getattr(g, "_set_share_cookie", None):
            resp.set_cookie(
                _cap.COOKIE_NAME, g._set_share_cookie,
                max_age=60 * 60 * 24 * 90, secure=True, httponly=True, samesite="Lax",
            )
        if get_config().remote.enabled:
            resp.headers["Referrer-Policy"] = "no-referrer"
        return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gallery_remote.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS — existing gallery/admin/live tests use the test client's default loopback addr but `remote.enabled` defaults False, so `_via_tunnel()` is False and the gate is inert.

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/gallery/app.py tests/test_gallery_remote.py
git commit -m "feat(gallery): shared-secret remote gate (tunnel-only), /admin blocked over tunnel"
```

---

### Task 5: request-aware live embeds (base_path + forced MSE over the tunnel)

**Files:**
- Modify: `src/wildlife/stream/config_gen.py` (emit `api.base_path`)
- Modify: `src/wildlife/gallery/app.py` (`_live_base`, `_camera_live`, `/live` + `/live/<id>` routes pass `remote`)
- Modify: `src/wildlife/gallery/templates/live.html` (Safari hint when remote)
- Test: `tests/test_stream_config.py` (append base_path cases); `tests/test_gallery_remote_live.py` (create)

**Interfaces:**
- Consumes: `_via_tunnel()` (Task 4); `LivestreamConfig.base_path` (Task 1); existing `_stream_iframe_src(base, stream_name, mode)`.
- Produces: `_live_base(remote: bool) -> str`; `_camera_live(base, camera, *, remote: bool) -> dict`; `build_go2rtc_config` emits `api.base_path` when `livestream.base_path` is set. Over the tunnel, embed URLs are same-origin `${base_path}/stream.html?...&mode=mse`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stream_config.py`:

```python
def test_build_go2rtc_config_emits_base_path_when_set() -> None:
    from wildlife.config import LivestreamConfig
    from wildlife.stream.config_gen import build_go2rtc_config
    from tests.test_gallery_live import _build_config  # reuses the minimal Config builder
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        cfg = _build_config(pathlib.Path(d), livestream=LivestreamConfig(enabled=True, base_path="/go2rtc"))
        data = build_go2rtc_config(cfg)
        assert data["api"]["base_path"] == "/go2rtc"


def test_build_go2rtc_config_omits_base_path_when_empty() -> None:
    from wildlife.config import LivestreamConfig
    from wildlife.stream.config_gen import build_go2rtc_config
    from tests.test_gallery_live import _build_config
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        cfg = _build_config(pathlib.Path(d), livestream=LivestreamConfig(enabled=True))
        data = build_go2rtc_config(cfg)
        assert "base_path" not in data["api"]
```

Create `tests/test_gallery_remote_live.py`:

```python
"""Live-embed URL tests: LAN (direct + configured modes) vs tunnel (base_path + MSE)."""

from __future__ import annotations

from pathlib import Path

from werkzeug.security import generate_password_hash

from wildlife.config import LivestreamConfig, RemoteConfig
from wildlife.gallery.app import create_app
from tests.test_gallery_live import _build_config

_SECRET = "livesecret-xyz789"
_LAN = {"REMOTE_ADDR": "192.168.1.50"}


def _remote_config(tmp_path: Path) -> object:
    cfg = _build_config(tmp_path, livestream=LivestreamConfig(enabled=True, base_path="/go2rtc"))
    cfg.remote = RemoteConfig(enabled=True, base_url="https://cam.example.com",
                              share_secret_hash=generate_password_hash(_SECRET))
    return cfg


def _client(config):
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def test_tunnel_embed_uses_base_path_and_mse(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    body = client.get(f"/live?key={_SECRET}").get_data(as_text=True)
    # Same-origin sub-path, forced MSE (WebRTC can't cross the tunnel).
    assert 'src="/go2rtc/stream.html?src=north_field_sub&mode=mse' in body
    assert "http://localhost:1984" not in body  # never the raw go2rtc origin over the tunnel


def test_lan_embed_uses_direct_origin_with_base_path(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    body = client.get("/live", environ_base=_LAN).get_data(as_text=True)
    # LAN goes straight to go2rtc:1984 (under base_path) with the configured mode.
    assert 'src="http://localhost:1984/go2rtc/stream.html?src=north_field_sub&mode=webrtc' in body


def test_remote_hint_shown_only_over_tunnel(tmp_path: Path) -> None:
    client = _client(_remote_config(tmp_path))
    assert "live-note" in client.get(f"/live?key={_SECRET}").get_data(as_text=True)
    assert "live-note" not in client.get("/live", environ_base=_LAN).get_data(as_text=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stream_config.py::test_build_go2rtc_config_emits_base_path_when_set tests/test_gallery_remote_live.py -v`
Expected: FAIL — `api` has no `base_path`; `_live_base()` takes no args / embeds use `http://localhost:1984/stream.html` without the sub-path or `mse`.

- [ ] **Step 3: Write minimal implementation**

In `src/wildlife/stream/config_gen.py`, in `build_go2rtc_config`, replace the `return {...}` block so `api` includes `base_path` when set:

```python
    api: dict = {"listen": ls.api_listen}
    if ls.base_path:
        api["base_path"] = ls.base_path

    return {
        "streams": streams,
        "api": api,
        "webrtc": {"listen": ls.webrtc_listen},
        "rtsp": {"listen": ls.rtsp_listen},
        "log": {"level": "info", "format": "color"},
    }
```

In `src/wildlife/gallery/app.py`, replace `_live_base` and `_camera_live` with:

```python
    def _live_base(remote: bool) -> str:
        """Resolve the browser-reachable go2rtc base for iframe embeds.

        Over the tunnel (``remote``) go2rtc is reverse-routed at a same-origin
        sub-path (``livestream.base_path``, e.g. ``/go2rtc``) and served under
        ``api.base_path``, so we embed a relative URL -- no host, no port, https by
        inheritance. On the LAN we hit go2rtc directly on its api port (honoring an
        explicit ``go2rtc_url`` override), with the same sub-path appended.
        """
        ls = get_config().livestream
        if remote:
            return ls.base_path
        if ls.go2rtc_url:
            return ls.go2rtc_url.rstrip("/") + ls.base_path
        host = request.host.split(":")[0]
        return f"http://{host}:{ls.go2rtc_port}{ls.base_path}"

    def _stream_iframe_src(base: str, stream_name: str, mode: str) -> str:
        """Build the go2rtc ``stream.html`` embed URL for a single stream name."""
        return f"{base}/stream.html?src={stream_name}&mode={mode}"

    def _camera_live(base: str, camera, *, remote: bool) -> dict:
        """Shape a camera into its ``sub``/``main`` iframe URLs.

        Over the tunnel both tiles are forced to ``mode=mse`` because WebRTC cannot
        traverse a Cloudflare Tunnel; on the LAN the configured per-tile transports
        (``sub_mode``/``main_mode``) are kept for best latency.
        """
        ls = get_config().livestream
        sub_mode = "mse" if remote else ls.sub_mode
        main_mode = "mse" if remote else ls.main_mode
        return {
            "id": camera.id,
            "sub_src": _stream_iframe_src(base, f"{camera.id}_sub", sub_mode),
            "main_src": _stream_iframe_src(base, f"{camera.id}_main", main_mode),
        }
```

(Note: `_stream_iframe_src` is unchanged; keep whichever single definition exists — do not duplicate it.)

Update the `/live` route body to compute `remote` and pass it through:

```python
    @app.route("/live")
    def live():
        """Render the live grid of every camera's go2rtc player embed."""
        cfg = get_config()
        ls = cfg.livestream
        if not ls.enabled:
            abort(404)
        remote = _via_tunnel()
        base = _live_base(remote)
        cameras = [_camera_live(base, cam, remote=remote) for cam in cfg.cameras]
        return render_template(
            "live.html", cameras=cameras, default_stream=ls.default_stream,
            allow_main=ls.allow_main, single=False, remote=remote,
        )
```

Update the `/live/<camera_id>` route the same way:

```python
    @app.route("/live/<camera_id>")
    def live_camera(camera_id: str):
        """Render a single enlarged live player for one camera id."""
        cfg = get_config()
        ls = cfg.livestream
        if not ls.enabled:
            abort(404)
        camera = next((c for c in cfg.cameras if c.id == camera_id), None)
        if camera is None:
            abort(404)
        remote = _via_tunnel()
        base = _live_base(remote)
        cameras = [_camera_live(base, camera, remote=remote)]
        return render_template(
            "live.html", cameras=cameras, default_stream=ls.default_stream,
            allow_main=ls.allow_main, single=True, remote=remote,
        )
```

In `src/wildlife/gallery/templates/live.html`, add a hint block immediately after the `<main>` line:

```html
  <main>
    {% if remote %}
    <p class="live-note">Live works best in Safari on iPhone/Mac. If a tile stays black, try the other quality (Sub / Main 4K) or a different browser.</p>
    {% endif %}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream_config.py tests/test_gallery_remote_live.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS — `tests/test_gallery_live.py` still passes because with `remote.enabled` False (its configs) embeds stay `http://localhost:1984/stream.html?...` (base_path defaults to `""`).

- [ ] **Step 6: Commit**

```bash
git add src/wildlife/stream/config_gen.py src/wildlife/gallery/app.py src/wildlife/gallery/templates/live.html tests/test_stream_config.py tests/test_gallery_remote_live.py
git commit -m "feat(live): base_path + forced MSE for live view over the tunnel"
```

---

### Task 6: docs — example config + README remote-access section & runbook

**Files:**
- Modify: `config.example.yaml` (add `livestream.base_path`; add a `remote:` block)
- Modify: `README.md` (new "Remote access (Cloudflare Tunnel)" section)

**Interfaces:** none (documentation). Values must match the code: cookie `wl_key`, `?key=`, `/go2rtc`, `wildlife-share-secret`.

- [ ] **Step 1: Update `config.example.yaml`**

Add a `base_path` line inside the `livestream:` block (after `sub_mode`):

```yaml
  base_path: "" # set "/go2rtc" to serve go2rtc under a sub-path for Cloudflare-Tunnel remote access
```

Add a new top-level block after the `livestream:` block:

```yaml
remote: # optional Cloudflare-Tunnel remote access (see README "Remote access")
  enabled: false # set by `wildlife-share-secret`; gates tunnel traffic with a shared-secret link
  base_url: "" # your public URL, e.g. "https://cam.example.com" (used to build the share link)
  share_secret_hash: null # Werkzeug hash of the shared secret; managed by `wildlife-share-secret`
  block_admin: true # refuse /admin over the tunnel (config editing stays LAN-only)
```

- [ ] **Step 2: Add the README section**

Append a "Remote access (Cloudflare Tunnel)" section to `README.md` covering, in order:

```markdown
## Remote access (Cloudflare Tunnel)

Reach the gallery + live view from anywhere at one URL (e.g. `https://cam.example.com`),
protected by a **shared secret in the link** — no ports forwarded, home IP hidden,
`/admin` unreachable remotely, and your LAN unchanged. Free Cloudflare plan.

**How the gate works.** Requests arriving through the tunnel (from the local
`cloudflared`, i.e. a loopback address) must carry the secret: as `?key=…` on the first
visit, then a `wl_key` cookie. LAN requests (real private IPs) are never gated. `/admin`
returns 404 over the tunnel. Live video is go2rtc's MSE player under `/go2rtc` (WebRTC
can't cross a tunnel), gated at Cloudflare's edge by one WAF rule checking the same cookie.

### One-time setup

1. `brew install cloudflared` (keep current: `brew upgrade cloudflared`).
2. Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create → Cloudflared → name
   it; copy the token; `sudo cloudflared service install <TOKEN>` (boot daemon).
3. Add two **public hostnames** on that tunnel (order matters — the `/go2rtc` one first):
   - `cam` . `example.com`, **Path** `go2rtc` → HTTP `localhost:1984`
   - `cam` . `example.com`, (no path) → HTTP `localhost:8080`
4. In `config.yaml` set `remote.base_url: "https://cam.example.com"` and
   `livestream.base_path: "/go2rtc"`; regenerate go2rtc config
   (`wildlife-stream-config`) and restart the gallery + go2rtc.
5. Run `wildlife-share-secret` → note the printed **share link** and **raw secret**.
6. Add a Cloudflare **WAF custom rule**: *Block* when the path starts with `/go2rtc` and
   the `wl_key` cookie ≠ the raw secret. (Free-plan fallback: a small Cloudflare Worker on
   `cam.example.com/go2rtc/*` that checks the `wl_key` cookie.)
7. Set each camera's **sub-stream to H.264 Main/Baseline** so MSE live plays in every
   browser incl. Safari/iOS.
8. Keep **Total TLS OFF** for the zone so `cam.` stays out of Certificate Transparency logs.

### Using / rotating

- Share `https://cam.example.com/?key=<secret>`; the recipient's browser stores the
  cookie so the key only appears once.
- **Rotate / revoke everyone:** re-run `wildlife-share-secret` and update the WAF rule
  with the new secret. Old links stop working.

> **Security note.** This is view-only "anyone with the link" access: a forwarded link
> grants access until you rotate. `/admin` is never exposed remotely. See
> `docs/superpowers/specs/2026-07-02-remote-access-cloudflare-tunnel-design.md` for the
> full threat model and the go2rtc-exposure trade-off.
```

- [ ] **Step 3: Verify docs match the code**

Run: `grep -n "wl_key\|/go2rtc\|wildlife-share-secret\|base_path" config.example.yaml README.md`
Expected: the cookie name, sub-path, CLI name, and `base_path` appear and match the implementation. Read both files once to confirm accuracy.

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml README.md
git commit -m "docs: remote access (Cloudflare Tunnel) setup + config example"
```

---

### Task 7: production setup script (`scripts/setup_remote.sh`)

**Files:**
- Create: `scripts/setup_remote.sh`
- Modify: `README.md` (point the runbook at the script)

**Interfaces:** none (host orchestration). Runs on the **prod Mac mini**, from the repo root, after Tasks 1–6 are merged and the project is installed in `./.venv`. Calls `wildlife-share-secret`, `wildlife-stream-config`, and `wildlife.admin.config_io.update_sections` — all delivered by earlier tasks. Automates host-side steps; prints the dashboard/camera steps with values filled in.

- [ ] **Step 1: Write the script**

Create `scripts/setup_remote.sh` (make it executable: `chmod +x`):

```bash
#!/usr/bin/env bash
#
# setup_remote.sh -- host-side setup for Cloudflare Tunnel remote access, run on
# the PRODUCTION Mac mini from the repo root (NOT the dev machine). Idempotent
# where practical. Pairs with a few Cloudflare-dashboard + camera steps it prints
# at the end with values filled in.
#
# Automates: install/upgrade cloudflared; connect the tunnel as a boot daemon
# from your dashboard token; set config.yaml (remote.base_url, livestream.base_path);
# mint the share secret; regenerate go2rtc.yaml; restart the gallery + stream services.
#
# Usage:
#   HOST=cam.example.com CF_TUNNEL_TOKEN=eyJ... ./scripts/setup_remote.sh
#   HOST=cam.example.com ./scripts/setup_remote.sh        # host config only; prints tunnel steps
#   ./scripts/setup_remote.sh --dry-run                   # echo actions, change nothing
#
set -euo pipefail

HOST="${HOST:-cam.example.com}"
CONFIG="${CONFIG:-config.yaml}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

run() { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] || "$@"; }
note() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[[ "$(uname)" == "Darwin" ]] || { echo "This script is for macOS (the prod Mac mini)." >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "No $CONFIG here -- run from the repo root on prod." >&2; exit 1; }
command -v brew >/dev/null || { echo "Homebrew required (https://brew.sh)." >&2; exit 1; }
VENV="./.venv/bin"
[[ -x "$VENV/python" ]] || { echo "No ./.venv -- install the project first (see README)." >&2; exit 1; }

# 1. cloudflared
if command -v cloudflared >/dev/null; then
  run brew upgrade cloudflared || true
else
  run brew install cloudflared
fi

# 2. Connect the tunnel as a boot daemon (token from the dashboard tunnel).
if [[ -z "${CF_TUNNEL_TOKEN:-}" ]]; then
  note "No CF_TUNNEL_TOKEN set -- skipping the tunnel service install."
  echo "Create a tunnel in the Cloudflare dashboard (Zero Trust -> Networks -> Tunnels),"
  echo "copy its connector token, then re-run: CF_TUNNEL_TOKEN=... ./scripts/setup_remote.sh"
else
  run sudo cloudflared service install "$CF_TUNNEL_TOKEN"
fi

# 3. Wildlife config: public URL + go2rtc sub-path (comment-preserving write).
note "Updating $CONFIG (remote.base_url, livestream.base_path)"
run "$VENV/python" - "$CONFIG" "$HOST" <<'PY'
import sys
from wildlife.admin.config_io import update_sections
update_sections(sys.argv[1], {
    "remote": {"base_url": f"https://{sys.argv[2]}"},
    "livestream": {"base_path": "/go2rtc"},
})
print("config updated")
PY

# 4. Mint the share secret (prints the share link + the raw secret).
note "Minting the share secret"
run "$VENV/wildlife-share-secret" "$CONFIG"

# 5. Regenerate go2rtc.yaml (adds api.base_path) + restart services.
run "$VENV/wildlife-stream-config" "$CONFIG" go2rtc.yaml
run sudo launchctl kickstart -k system/com.wildlife.gallery || true
run sudo launchctl kickstart -k system/com.wildlife.stream || true

# 6. Finish in the Cloudflare dashboard / on the cameras.
note "ALMOST DONE -- finish these (values filled in for you):"
cat <<EOF

A) Tunnel public hostnames (your tunnel -> Published routes) -- add TWO, /go2rtc FIRST:
     1. Subdomain "${HOST%%.*}", Domain "${HOST#*.}", Path "go2rtc" -> HTTP  localhost:1984
     2. Subdomain "${HOST%%.*}", Domain "${HOST#*.}", (no Path)     -> HTTP  localhost:8080

B) WAF custom rule (Security -> WAF -> Custom rules -> Create):
     Expression:  (starts_with(http.request.uri.path, "/go2rtc") and not http.request.cookies["wl_key"][0] eq "<PASTE THE RAW SECRET PRINTED ABOVE>")
     Action:      Block
   (Free-plan fallback: a Worker on ${HOST}/go2rtc/* that checks the wl_key cookie.)

C) Cameras: set each sub-stream to H.264 Main or Baseline profile.

D) SSL/TLS -> Edge Certificates: confirm Total TLS is OFF (keeps ${HOST} out of CT logs).

Then, from OFF your LAN, open the printed share link and verify the gallery + live view.
EOF
```

- [ ] **Step 2: Make it executable and syntax-check**

Run: `chmod +x scripts/setup_remote.sh && bash -n scripts/setup_remote.sh`
Expected: no output (valid syntax). This script is **prod-only** — do not execute its body here (it has no `config.yaml`, no cloudflared, and would touch `sudo`/services).

- [ ] **Step 3: Lint (if shellcheck is available)**

Run: `command -v shellcheck >/dev/null && shellcheck scripts/setup_remote.sh || echo "shellcheck not installed; skipped"`
Expected: no errors (warnings acceptable; fix any that indicate real bugs). If shellcheck isn't installed, the step is a documented skip.

- [ ] **Step 4: Point the README runbook at the script**

In `README.md`, in the "Remote access (Cloudflare Tunnel)" section's setup steps, add a note near the top:

```markdown
> On the production Mac mini you can run `./scripts/setup_remote.sh` (after creating a
> tunnel in the dashboard and exporting `CF_TUNNEL_TOKEN`) to do the host-side setup —
> it configures `config.yaml`, mints the share secret, regenerates `go2rtc.yaml`,
> restarts services, and prints the remaining dashboard/camera steps with values filled in.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_remote.sh README.md
git commit -m "feat(remote): scripts/setup_remote.sh for prod host-side setup"
```

---

## Self-Review

**1. Spec coverage** (`docs/superpowers/specs/2026-07-02-remote-access-cloudflare-tunnel-design.md`):

- §6.1 `RemoteConfig` + `livestream.base_path` → Task 1 ✅
- §6.2 shared-secret gate (loopback-only, `?key`→cookie, fail-closed, `no-referrer`, rate-limit, 404 no-oracle) → Tasks 3 (helpers) + 4 (wiring) ✅
- §6.2 `wildlife-share-secret` CLI + hash storage → Task 2 ✅
- §6.3 request-aware embeds (base_path + forced MSE; LAN keeps modes) → Task 5 ✅
- §6.4 `config_gen` emits `api.base_path`; WAF rule / Worker fallback → Task 5 (code) + Task 6 (runbook) ✅
- §3.1 `/admin` 404 over tunnel; LAN unchanged → Task 4 ✅
- §7 failure modes (fail-closed, bad key 404, base_path mismatch noted, Safari hint) → Tasks 4 + 5 ✅
- §7.3 camera H.264 Main/Baseline → Task 6 runbook ✅
- §9 runbook, §8 security notes, CT/Total-TLS → Task 6 ✅
- Prod host-side setup automated by `scripts/setup_remote.sh` → Task 7 ✅
- **Edge/WAF steps (dashboard hostnames, WAF rule, Total TLS) + camera codec** remain manual — captured in the Task 6 runbook and **printed by `setup_remote.sh` with values filled in** (Task 7). Intentional: no Cloudflare-API automation in scope (would need a scoped API token and is risky to ship untested from a non-prod box).

**2. Placeholder scan:** No "TBD/TODO"; every code step shows complete code; test steps include real assertions. ✅

**3. Type consistency:** `secret_ok(secret_hash, provided)`, `is_loopback(remote_addr)`, `COOKIE_NAME="wl_key"`, `RateLimiter(max_fails).blocked/record_fail/reset`, `_via_tunnel()`, `_live_base(remote)`, `_camera_live(base, camera, *, remote)`, `set_remote_secret(config_path, secret_hash)` — names/signatures are identical across the tasks that define and consume them. ✅

**4. Non-regression guarantee:** every behavior change is gated behind `remote.enabled` (default False) or `livestream.base_path` (default `""`), so the existing test suite passes unchanged after each task (explicit "run the full suite" steps in Tasks 1, 4, 5).
