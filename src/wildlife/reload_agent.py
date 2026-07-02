"""Root reloader: apply admin config edits to the worker + go2rtc daemons.

The gallery process runs as an unprivileged user and can't restart the other
LaunchDaemons, so applying a config change is delegated to this tiny job. It is
wired to launchd's ``WatchPaths`` on the reload-trigger file (see
``launchd/com.wildlife.reload.plist``): whenever the admin UI writes a new
``config.yaml`` and touches the trigger, launchd starts this job **as root**.

It then, in order:

1. **Re-validates** ``config.yaml`` through the same loader the daemons use. If
   it's invalid, it logs, records the failure in the status file, and exits
   **without touching any service** -- the last-known-good config keeps running.
   (The admin UI already validates before writing, so this is defence in depth.)
2. Regenerates ``go2rtc.yaml`` from the validated config.
3. Restarts the worker (``com.wildlife.detect``) and go2rtc
   (``com.wildlife.stream``) via ``launchctl kickstart -k``. The gallery is not
   restarted -- it reloads its own config per request.
4. Writes ``.reload.status.json`` next to the config so the admin dashboard can
   show the outcome and timestamp of the last apply.

This runs as a one-shot job (RunAtLoad/KeepAlive off); launchd re-launches it on
every trigger change.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from wildlife.admin.config_io import STATUS_NAME as _STATUS_NAME
from wildlife.config import load_config
from wildlife.stream.config_gen import write_go2rtc_config

logger = logging.getLogger(__name__)

__all__ = ["apply", "main"]

#: LaunchDaemon labels to restart when the config changes (not the gallery).
_SERVICES = ("com.wildlife.detect", "com.wildlife.stream")


def _kickstart(label: str) -> tuple[bool, str]:
    """Restart one system LaunchDaemon; return ``(ok, detail)``."""
    cmd = ["launchctl", "kickstart", "-k", f"system/{label}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, "restarted"
    detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
    return False, detail


def _write_status(config_path: Path, status: dict) -> None:
    """Best-effort write of the apply status JSON next to the config."""
    status["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        (config_path.parent / _STATUS_NAME).write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
    except OSError:  # pragma: no cover - status is advisory only
        logger.warning("Could not write reload status file", exc_info=True)


def apply(config_path: str | Path = "config.yaml", go2rtc_path: str | Path = "go2rtc.yaml") -> int:
    """Validate config, regenerate go2rtc.yaml, and restart services.

    Returns a process exit code (0 = fully applied, non-zero = something failed).
    Never restarts a service when the config is invalid.
    """
    config_path = Path(config_path)
    go2rtc_path = Path(go2rtc_path)

    # 1) Re-validate (defence in depth; the UI already validated).
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - any load/validation failure is fatal here
        logger.error("Config INVALID -- not restarting anything: %s", exc)
        _write_status(config_path, {"ok": False, "stage": "validate", "error": str(exc)})
        return 1

    # 2) Regenerate go2rtc.yaml.
    try:
        write_go2rtc_config(config, go2rtc_path)
        logger.info("Regenerated %s (%d camera(s))", go2rtc_path, len(config.cameras))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to regenerate go2rtc.yaml -- not restarting: %s", exc)
        _write_status(config_path, {"ok": False, "stage": "go2rtc", "error": str(exc)})
        return 1

    # 3) Restart worker + go2rtc.
    restarted: dict[str, str] = {}
    all_ok = True
    for label in _SERVICES:
        ok, detail = _kickstart(label)
        restarted[label] = detail
        all_ok = all_ok and ok
        (logger.info if ok else logger.error)("%s: %s", label, detail)

    _write_status(
        config_path,
        {"ok": all_ok, "stage": "done", "restarted": restarted, "cameras": len(config.cameras)},
    )
    return 0 if all_ok else 2


def main() -> int:
    """Console entry (``wildlife-reload``): ``argv[1]`` config, ``argv[2]`` go2rtc path."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    go2rtc_path = sys.argv[2] if len(sys.argv) > 2 else "go2rtc.yaml"
    logger.info(
        "Reload triggered (uid=%s cwd=%s config=%s)", os.geteuid(), os.getcwd(), config_path
    )
    return apply(config_path, go2rtc_path)


if __name__ == "__main__":
    raise SystemExit(main())
