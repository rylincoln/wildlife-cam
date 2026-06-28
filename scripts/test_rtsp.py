#!/usr/bin/env python3
"""Hard-gate RTSP smoke test: grab one frame from every configured camera.

This validates the whole "open RTSP -> decode HEVC -> get a numpy frame" path
end to end: camera credentials, the interpolated RTSP URL, network reachability,
and OpenCV's FFmpeg/HEVC decode. It is step 2 of the build order and a hard gate
-- if a camera cannot produce a single frame here, nothing downstream will work.

Run it directly from a checkout::

    python scripts/test_rtsp.py                 # all cameras, config.yaml
    python scripts/test_rtsp.py --stream sub    # use the lighter sub stream
    python scripts/test_rtsp.py --camera north_field

Exit status is ``0`` only when *every* selected camera yielded a frame, so the
script doubles as a CI/pre-flight check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _bootstrap_src_path() -> None:
    """Allow ``python scripts/test_rtsp.py`` without an editable install.

    Adds ``<repo>/src`` to ``sys.path`` if the ``wildlife`` package isn't already
    importable, so the operational scripts run straight from a checkout.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()

from wildlife.capture import grab_burst  # noqa: E402  (after path bootstrap)
from wildlife.config import Config, load_config  # noqa: E402


def _resolve_config(path_arg: str) -> Path:
    """Find the config file, falling back to the repo root next to ``scripts/``."""
    candidate = Path(path_arg)
    if candidate.is_file():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    alt = repo_root / path_arg
    if alt.is_file():
        return alt
    raise SystemExit(
        f"Config file not found: {path_arg!r}. "
        "Copy config.example.yaml to config.yaml and edit it."
    )


def _redact(url: str) -> str:
    """Mask the password in an RTSP URL's userinfo when echoing it to the console.

    Parses the URL and rewrites only the ``user:password@host`` segment so a
    short password (e.g. ``p``) can never clobber unrelated text like the
    ``rtsp`` scheme.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    netloc = f"{user}:****@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _setup_logging(verbose: bool) -> None:
    """Surface capture-module logs (warnings/info) while testing."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _select_cameras(config: Config, only: str | None) -> list:
    """Return the cameras to test, optionally narrowed to a single ``--camera`` id."""
    if only is None:
        return list(config.cameras)
    chosen = [c for c in config.cameras if c.id == only]
    if not chosen:
        known = ", ".join(c.id for c in config.cameras) or "(none)"
        raise SystemExit(f"No camera with id {only!r}. Known cameras: {known}")
    return chosen


def main() -> int:
    """Grab one frame per camera and report success + shape. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--stream",
        choices=("main", "sub"),
        default=None,
        help="Override the configured stream for this test (main|sub).",
    )
    parser.add_argument(
        "--camera", default=None, help="Only test the camera with this id."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level module logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config = load_config(_resolve_config(args.config))
    stream = args.stream or config.capture.stream
    timeout_s = config.capture.rtsp_timeout_s
    cameras = _select_cameras(config, args.camera)

    print(
        f"Testing {len(cameras)} camera(s) on the {stream!r} stream "
        f"(timeout {timeout_s}s)\n"
    )

    failures = 0
    for cam in cameras:
        url = cam.rtsp_sub if stream == "sub" else cam.rtsp_main
        print(f"[{cam.id}] {_redact(url)}")
        frames = grab_burst(
            cam, n=1, interval_ms=0, stream=stream, timeout_s=timeout_s
        )
        if frames:
            frame = frames[0]
            h, w = frame.shape[:2]
            print(f"    OK   shape={frame.shape} dtype={frame.dtype} ({w}x{h})\n")
        else:
            print("    FAIL no frame decoded (check creds / URL / reachability)\n")
            failures += 1

    total = len(cameras)
    ok = total - failures
    print(f"Summary: {ok}/{total} camera(s) returned a frame.")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
