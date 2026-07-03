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
