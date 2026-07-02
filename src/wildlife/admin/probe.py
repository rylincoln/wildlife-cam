"""Isolated "test this camera" probe for the admin add/edit flow.

This is what powers *test-before-save*: given a set of (possibly unsaved) camera
fields, it verifies the RTSP stream actually decodes a frame and, for Reolink,
that the credentials/host let ``reolink-aio`` connect and enumerate channels/AI
types. A green result means "this camera will work if you save it".

It is designed to run as a **subprocess** (``python -m wildlife.admin.probe``),
for three reasons:

* it keeps the heavy/optional imports (``cv2`` via :func:`wildlife.capture.grab_burst`,
  and ``reolink-aio``) out of the long-lived, hardware-free gallery process;
* a slow or hung RTSP open can be killed by a hard subprocess timeout instead of
  tying up a Flask worker thread; and
* **secrets are read from stdin, never argv** -- so the camera password can't
  leak into the process list (``ps``).

Input (stdin, JSON)::

    {"id","host","username","password","rtsp_main","rtsp_sub","onvif_port",
     "streams":["sub","main"], "timeout_s":10, "reolink":true}

Output (stdout, JSON): see :func:`probe` for the shape.
"""

from __future__ import annotations

import json
import re
import sys

__all__ = ["probe", "main"]

_DEFAULT_TIMEOUT_S = 10

# Mask the password in ``scheme://user:PASSWORD@host...``. A greedy ``.*`` up to
# the LAST ``@`` tolerates delimiter chars ('/', '?', '#', '@', ':') inside the
# password -- ``urlsplit`` mis-parses those and would leak the plaintext.
_REDACT_RE = re.compile(r"(://[^:@/?#]+:).*(@)")


def _redact(url: str) -> str:
    """Mask the password in an RTSP URL before returning it to the browser.

    Leaves credential-less URLs (no ``user:pass@``) untouched.
    """
    return _REDACT_RE.sub(r"\1****\2", url)


def _make_camera(params: dict):
    """Build a validated :class:`CameraConfig` from raw probe params."""
    from wildlife.config import CameraConfig

    return CameraConfig(
        id=str(params.get("id") or "probe"),
        host=str(params["host"]),
        username=str(params["username"]),
        password=str(params.get("password", "")),
        rtsp_main=str(params["rtsp_main"]),
        rtsp_sub=str(params["rtsp_sub"]),
        onvif_port=int(params.get("onvif_port", 8000)),
    )


def _test_stream(camera, stream: str, timeout_s: int) -> dict:
    """Grab a single frame from one stream; return an ok/size/error report."""
    from wildlife.capture import grab_burst  # lazy: pulls in cv2

    url = camera.rtsp_sub if stream == "sub" else camera.rtsp_main
    try:
        frames = grab_burst(camera, n=1, interval_ms=0, stream=stream, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - report any failure to the UI
        return {"ok": False, "url": _redact(url), "error": str(exc)}
    if frames:
        h, w = frames[0].shape[:2]
        return {"ok": True, "url": _redact(url), "width": int(w), "height": int(h)}
    return {
        "ok": False,
        "url": _redact(url),
        "error": "no frame decoded (check credentials, RTSP path, and reachability)",
    }


def _test_reolink(camera, timeout_s: int) -> dict:
    """Connect via reolink-aio and enumerate channels + AI object types."""
    import asyncio

    async def _run() -> dict:
        try:
            from reolink_aio.api import Host
        except ImportError:
            return {"ok": False, "error": "reolink-aio not installed ([cameras] extra)"}

        # Mirror reolink_native's port strategy: try the library default (HTTP:80)
        # first, then the configured ONVIF port.
        last_err = "unknown error"
        for port in (None, camera.onvif_port):
            kwargs = {"timeout": timeout_s}
            if port is not None:
                kwargs["port"] = port
            host = Host(camera.host, camera.username, camera.password, **kwargs)
            try:
                await host.get_host_data()
                channels = list(getattr(host, "channels", None) or [0])
                ai_types: list[str] = []
                for obj in ("animal", "dog_cat", "people", "vehicle"):
                    try:
                        if host.ai_supported(0, obj):
                            ai_types.append(obj)
                    except Exception:  # noqa: BLE001 - version/signature differences
                        pass
                return {
                    "ok": True,
                    "channels": [int(c) for c in channels],
                    "ai_types": ai_types,
                    "model": getattr(host, "model", None),
                }
            except Exception as exc:  # noqa: BLE001 - try next port / report
                last_err = f"{type(exc).__name__}: {exc}"
            finally:
                logout = getattr(host, "logout", None)
                if callable(logout):
                    try:
                        await logout()
                    except Exception:  # noqa: BLE001
                        pass
        return {"ok": False, "error": last_err}

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout=timeout_s * 2 + 2))
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"timed out after {timeout_s * 2 + 2}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def probe(params: dict) -> dict:
    """Run the requested checks and return a JSON-serialisable report.

    Shape::

        {"streams": {"sub": {ok,width,height|error}, "main": {...}},
         "reolink": {attempted, ok, channels, ai_types, model|error}}

    Never raises for expected failures -- every check reports ``ok: False`` with
    an ``error`` string instead, so the UI can render precise diagnostics.
    """
    timeout_s = int(params.get("timeout_s", _DEFAULT_TIMEOUT_S))
    want_streams = params.get("streams") or ["sub", "main"]
    try:
        camera = _make_camera(params)
    except Exception as exc:  # noqa: BLE001 - bad params -> structured error
        return {"streams": {}, "reolink": {"attempted": False}, "error": str(exc)}

    streams = {s: _test_stream(camera, s, timeout_s) for s in want_streams}

    reolink = {"attempted": False}
    if params.get("reolink", True):
        reolink = {"attempted": True, **_test_reolink(camera, timeout_s)}

    return {"streams": streams, "reolink": reolink}


def main() -> int:
    """CLI entry: read JSON params from stdin, print the JSON report to stdout."""
    try:
        params = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        json.dump({"error": f"invalid input: {exc}"}, sys.stdout)
        return 2
    result = probe(params)
    json.dump(result, sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
