"""HTTP Basic Auth guard for the admin blueprint.

The admin section can rewrite config and (via the reloader) restart services, so
it must not inherit the read-only gallery's no-auth posture. Access is gated by a
single password whose Werkzeug hash lives in ``config.admin.password_hash``.

Design choices:

* **Basic Auth**, not a login session: stateless, needs no ``SECRET_KEY`` for
  the gate itself, and works for both page loads and the AJAX test endpoint.
* **Fail closed**: if admin is enabled but no password hash is set, every admin
  route returns 403 with instructions rather than granting open access -- so a
  fresh deployment can't be reconfigured from the LAN before a password exists.
* The username is ignored; only the password is checked against the stored hash.
"""

from __future__ import annotations

from flask import Response, request
from werkzeug.security import check_password_hash

__all__ = ["auth_challenge", "check_admin_auth"]

_REALM = 'Basic realm="Wildlife Admin", charset="UTF-8"'


def auth_challenge(message: str = "Authentication required.") -> Response:
    """Return a 401 that prompts the browser for Basic Auth credentials."""
    return Response(message, 401, {"WWW-Authenticate": _REALM})


def check_admin_auth(config) -> Response | None:
    """Return ``None`` if the request is authorised, else a blocking response.

    Intended for use in a blueprint ``before_request`` hook. ``config`` is the
    current (live-reloaded) :class:`~wildlife.config.Config`.
    """
    admin = config.admin
    if not admin.enabled:
        # Admin explicitly disabled -> hide the section entirely.
        return Response("Admin is disabled.", 404)
    if not admin.password_hash:
        return Response(
            "Admin is enabled but no password is set. On the host run:\n"
            "    wildlife-admin-password\n"
            "then reload this page.",
            403,
            {"Content-Type": "text/plain; charset=utf-8"},
        )
    auth = request.authorization
    if (
        auth is None
        or auth.password is None
        or not check_password_hash(admin.password_hash, auth.password)
    ):
        return auth_challenge()
    return None
