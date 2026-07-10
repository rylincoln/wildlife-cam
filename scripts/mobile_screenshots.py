#!/usr/bin/env python3
"""Capture mobile-viewport screenshots of the wildlife gallery for UI review.

Drives a headless Chromium (via Playwright) against a *running* gallery and
saves phone-sized screenshots of the public views (and, with credentials, the
password-gated ``/admin`` views). Intended for eyeballing mobile UI/UX work
without a physical phone.

Setup (one time)::

    uv pip install --python .venv/bin/python playwright   # or: pip install -e '.[screenshots]'
    .venv/bin/python -m playwright install chromium        # downloads the browser

Usage::

    .venv/bin/python scripts/mobile_screenshots.py
    .venv/bin/python scripts/mobile_screenshots.py --out /tmp/shots --device "Pixel 7"
    .venv/bin/python scripts/mobile_screenshots.py --admin-user admin --admin-pass secret

Notes
-----
* Base URL defaults to this machine's **LAN IP** (not 127.0.0.1). When
  ``config.remote.enabled`` is true the gallery 404s loopback requests that lack
  a ``?key=`` (cloudflared -> gallery is also loopback), but a LAN IP is
  non-loopback and bypasses that gate. Override with ``--base-url`` or
  ``$WILDLIFE_GALLERY_URL``.
* ``/admin`` is HTTP Basic auth: it is skipped unless both ``--admin-user`` and
  ``--admin-pass`` are supplied.
* Uses Playwright's built-in device descriptors (viewport, scale, touch, UA).
  List them with ``--device ?``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
from pathlib import Path


def lan_ip() -> str:
    """Best-effort primary LAN IPv4 (the address a browser on the network hits)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the outbound iface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def default_base_url() -> str:
    env = os.environ.get("WILDLIFE_GALLERY_URL")
    if env:
        return env.rstrip("/")
    return f"http://{lan_ip()}:8080"


# (name, path, open_lightbox, admin_only) -- the capture plan.
VIEWS = [
    ("gallery", "/", True, False),
    ("live", "/live", False, False),
    ("admin-dashboard", "/admin", False, True),
    ("admin-captures", "/admin/captures", True, True),
    ("admin-cameras", "/admin/cameras", False, True),
]


async def capture(args) -> int:
    from playwright.async_api import async_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have_admin = bool(args.admin_user and args.admin_pass)
    saved: list[str] = []

    async with async_playwright() as p:
        if args.device == "?":
            print("Available devices:\n  " + "\n  ".join(sorted(p.devices)))
            return 0
        if args.device not in p.devices:
            print(f"Unknown --device {args.device!r}; try --device ? to list.", file=sys.stderr)
            return 2

        browser = await p.chromium.launch()
        ctx_kwargs = dict(p.devices[args.device])
        if have_admin:
            ctx_kwargs["http_credentials"] = {
                "username": args.admin_user,
                "password": args.admin_pass,
            }
        ctx = await browser.new_context(**ctx_kwargs)
        ctx.set_default_timeout(args.timeout * 1000)
        page = await ctx.new_page()

        for name, path, open_lightbox, admin_only in VIEWS:
            if admin_only and not have_admin:
                continue
            url = args.base_url + path
            try:
                await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_timeout(args.settle)
                dest = out / f"{name}.png"
                await page.screenshot(path=str(dest), full_page=args.full_page)
                saved.append(str(dest))
                print(f"  ✓ {name:18s} {url}")
            except Exception as exc:  # noqa: BLE001 - one bad view shouldn't abort the run
                print(f"  ✗ {name:18s} {url}  ({type(exc).__name__}: {exc})", file=sys.stderr)
                continue

            if open_lightbox:
                try:
                    card = await page.query_selector("#grid .card")
                    if card:
                        await card.click()
                        await page.wait_for_selector("#lightbox:not([hidden])", timeout=4000)
                        await page.wait_for_timeout(700)
                        dest = out / f"{name}-lightbox.png"
                        await page.screenshot(path=str(dest), full_page=False)
                        saved.append(str(dest))
                        print(f"  ✓ {name + '-lightbox':18s} (opened first capture)")
                        await page.keyboard.press("Escape")
                        await page.wait_for_timeout(200)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ {name}-lightbox  ({type(exc).__name__}: {exc})", file=sys.stderr)

        await browser.close()

    print(f"\n{len(saved)} screenshot(s) -> {out}/")
    if not have_admin:
        print("(admin views skipped; pass --admin-user/--admin-pass to include them)")
    return 0 if saved else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=default_base_url(),
                    help="Gallery base URL (default: $WILDLIFE_GALLERY_URL or http://<LAN-IP>:8080)")
    ap.add_argument("--out", default="screenshots", help="Output directory (default: ./screenshots)")
    ap.add_argument("--device", default="iPhone 13",
                    help='Playwright device profile (default: "iPhone 13"; use "?" to list)')
    ap.add_argument("--admin-user", default=os.environ.get("WILDLIFE_ADMIN_USER"),
                    help="Admin HTTP Basic username (enables /admin views)")
    ap.add_argument("--admin-pass", default=os.environ.get("WILDLIFE_ADMIN_PASS"),
                    help="Admin HTTP Basic password")
    ap.add_argument("--settle", type=int, default=1200,
                    help="ms to wait after load before shooting (default: 1200)")
    ap.add_argument("--timeout", type=int, default=15, help="Per-navigation timeout in seconds")
    ap.add_argument("--full-page", dest="full_page", action="store_true", default=True,
                    help="Capture the full scroll height (default)")
    ap.add_argument("--no-full-page", dest="full_page", action="store_false",
                    help="Capture only the visible viewport")
    args = ap.parse_args()

    print(f"Gallery: {args.base_url}   device: {args.device}")
    rc = asyncio.run(capture(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
