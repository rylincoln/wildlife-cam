#!/usr/bin/env python3
"""Print camera detection events as they arrive (no inference, no capture).

Build-order step 3 and a hard gate: this proves the camera-side motion/animal
events actually reach the Mac via the configured ``event_source``
(``reolink_native`` or ``onvif_bridge``). It starts one
:class:`~wildlife.events.base.EventSource` per camera, each on its own daemon
thread feeding a shared print loop, and echoes every
:class:`~wildlife.models.CameraEvent` until you press Ctrl-C.

Run it directly from a checkout::

    python scripts/test_events.py
    python scripts/test_events.py --camera north_field
    python scripts/test_events.py --source onvif_bridge   # override event_source

Now go walk in front of a camera; you should see events stream in. Ctrl-C stops.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Allow running straight from a checkout without an editable install."""
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()

from wildlife.config import Config, load_config  # noqa: E402
from wildlife.events.base import make_event_source  # noqa: E402

# One lock so concurrent listener threads don't interleave their prints.
_PRINT_LOCK = threading.Lock()


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


def _setup_logging(verbose: bool) -> None:
    """Surface event-source logs (reconnects, backoff, errors)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _emit(line: str) -> None:
    """Thread-safe console print."""
    with _PRINT_LOCK:
        print(line, flush=True)


def _listen(source_kind: str, camera) -> None:
    """Consume one camera's event stream forever, printing each event.

    Runs on a daemon thread. Any exception is reported (not silently swallowed)
    so a camera that can't subscribe is obvious; the thread then exits while the
    others keep running.
    """
    try:
        source = make_event_source(source_kind, camera)
    except Exception as exc:  # noqa: BLE001 - surface setup failures to the user
        _emit(f"[{camera.id}] ERROR creating event source: {exc!r}")
        return

    _emit(f"[{camera.id}] listening via {source_kind} ...")
    try:
        for event in source.stream():
            _emit(
                f"{event.event_ts.isoformat()}  "
                f"{event.camera_id:<16} {event.kind}"
            )
    except Exception as exc:  # noqa: BLE001 - keep other cameras alive
        _emit(f"[{camera.id}] ERROR in event stream: {exc!r}")
        return
    _emit(f"[{camera.id}] event stream ended.")


def _select_cameras(config: Config, only: str | None) -> list:
    """Return the cameras to listen on, optionally narrowed to one ``--camera``."""
    if only is None:
        return list(config.cameras)
    chosen = [c for c in config.cameras if c.id == only]
    if not chosen:
        known = ", ".join(c.id for c in config.cameras) or "(none)"
        raise SystemExit(f"No camera with id {only!r}. Known cameras: {known}")
    return chosen


def main() -> int:
    """Start a listener per camera and print events until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--source",
        choices=("reolink_native", "onvif_bridge"),
        default=None,
        help="Override the configured event_source for this test.",
    )
    parser.add_argument(
        "--camera", default=None, help="Only listen on the camera with this id."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level module logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config = load_config(_resolve_config(args.config))
    source_kind = args.source or config.event_source
    cameras = _select_cameras(config, args.camera)

    print(
        f"Subscribing to {len(cameras)} camera(s) via {source_kind!r}.\n"
        "Trigger a camera (walk in front of it); events print below. "
        "Press Ctrl-C to stop.\n"
    )

    threads: list[threading.Thread] = []
    for cam in cameras:
        thread = threading.Thread(
            target=_listen, args=(source_kind, cam), name=f"events-{cam.id}", daemon=True
        )
        thread.start()
        threads.append(thread)

    try:
        # Idle the main thread; daemon listeners do the work. Exit early only if
        # every listener has already died (e.g. all failed to subscribe).
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
        _emit("\nAll event streams have ended; nothing left to listen on.")
        return 1
    except KeyboardInterrupt:
        _emit("\nStopping (Ctrl-C). Closing listeners.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
