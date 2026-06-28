"""Read-only Flask gallery for browsing wildlife captures.

Exposes :func:`create_app`, the application factory that builds a Flask app
bound to a validated :class:`wildlife.config.Config`, and :func:`main`, the
console entry point that loads ``config.yaml`` and serves the gallery on the
configured LAN address/port.
"""

from __future__ import annotations

from wildlife.gallery.app import create_app, main

__all__ = ["create_app", "main"]
