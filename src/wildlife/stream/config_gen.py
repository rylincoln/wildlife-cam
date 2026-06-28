"""Generate a ``go2rtc.yaml`` config from the wildlife :class:`Config`.

go2rtc is the external companion service that performs the actual WebRTC/MSE
livestreaming; the gallery merely embeds its player. This module is the *single
source of truth* for go2rtc stream naming: for every camera in the wildlife
config it emits exactly two streams,

* ``"<camera.id>_main"`` -> ``[camera.rtsp_main]``
* ``"<camera.id>_sub"``  -> ``[camera.rtsp_sub]``

plus the API / WebRTC / RTSP listen addresses and a log block taken from
:class:`wildlife.config.LivestreamConfig`.

Only the standard library, PyYAML and :mod:`wildlife.config` are imported here so
the module stays free of torch/cv2 and importable in hardware-free tests.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

from wildlife.config import Config, load_config

logger = logging.getLogger(__name__)

__all__ = [
    "build_go2rtc_config",
    "write_go2rtc_config",
    "main",
]


def build_go2rtc_config(config: Config) -> dict:
    """Build the go2rtc configuration mapping for ``config``.

    Two streams are emitted per camera using the frozen naming contract
    ``"<id>_main"`` / ``"<id>_sub"``, each pointing at the camera's
    credential-interpolated ``rtsp_main`` / ``rtsp_sub`` URL. The listen
    addresses and log block come from ``config.livestream``.

    Args:
        config: A validated wildlife configuration tree.

    Returns:
        A plain ``dict`` ready to be serialised to ``go2rtc.yaml`` with the
        shape::

            {"streams": {"<id>_main": [rtsp_main], "<id>_sub": [rtsp_sub], ...},
             "api": {"listen": ...},
             "webrtc": {"listen": ...},
             "rtsp": {"listen": ...},
             "log": {"level": "info", "format": "color"}}
    """
    ls = config.livestream

    streams: dict[str, list[str]] = {}
    for camera in config.cameras:
        streams[f"{camera.id}_main"] = [camera.rtsp_main]
        streams[f"{camera.id}_sub"] = [camera.rtsp_sub]

    return {
        "streams": streams,
        "api": {"listen": ls.api_listen},
        "webrtc": {"listen": ls.webrtc_listen},
        "rtsp": {"listen": ls.rtsp_listen},
        "log": {"level": "info", "format": "color"},
    }


def write_go2rtc_config(config: Config, path: str | Path = "go2rtc.yaml") -> Path:
    """Serialise the go2rtc config for ``config`` to ``path`` as YAML.

    Args:
        config: A validated wildlife configuration tree.
        path: Destination path for the generated ``go2rtc.yaml``.

    Returns:
        The :class:`pathlib.Path` that was written.
    """
    out_path = Path(path)
    data = build_go2rtc_config(config)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out_path


def main() -> None:
    """CLI entry point: render ``go2rtc.yaml`` from a wildlife config file.

    Usage::

        python -m wildlife.stream.config_gen [config_path] [out_path]

    ``config_path`` defaults to ``"config.yaml"`` and ``out_path`` to
    ``"go2rtc.yaml"``. Loads and validates the config, writes the go2rtc file,
    and logs the destination path plus the generated stream names.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "go2rtc.yaml"

    config = load_config(config_path)
    data = build_go2rtc_config(config)
    written = Path(out_path)
    with open(written, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)

    stream_names = ", ".join(data["streams"].keys())
    logger.info("Wrote go2rtc config to %s", written)
    logger.info("Streams (%d): %s", len(data["streams"]), stream_names)


if __name__ == "__main__":
    main()
