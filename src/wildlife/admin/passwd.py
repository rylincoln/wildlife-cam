"""``wildlife-admin-password`` -- set or replace the admin UI password.

The admin routes fail closed until a password hash exists, and you can't set the
first password from the (locked) web UI -- so this CLI bootstraps it. It prompts
without echoing, hashes with Werkzeug, and writes only the hash into
``config.yaml`` via the same validated, comment-preserving writer the UI uses.

Usage::

    wildlife-admin-password                 # edits ./config.yaml
    wildlife-admin-password /path/config.yaml
"""

from __future__ import annotations

import getpass
import sys

from werkzeug.security import generate_password_hash

from wildlife.admin.config_io import ConfigError, set_admin_password

_MIN_LEN = 8


def main() -> int:
    """Prompt for a new admin password and persist its hash. Returns an exit code."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        pw = getpass.getpass("New admin password: ")
        pw2 = getpass.getpass("Confirm password:   ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 1

    if len(pw) < _MIN_LEN:
        print(f"Password must be at least {_MIN_LEN} characters.", file=sys.stderr)
        return 1
    if pw != pw2:
        print("Passwords do not match.", file=sys.stderr)
        return 1

    try:
        set_admin_password(config_path, generate_password_hash(pw))
    except (ConfigError, OSError) as exc:
        print(f"Failed to write {config_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Admin password set in {config_path}. Open /admin and log in (any username).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
