"""Optional camera livestreaming support for the wildlife detection system.

This package is *additive*: it does not touch the detector (``worker.py`` /
``capture.py`` / ``detect.py``). Live viewing is delegated to an external
`go2rtc <https://github.com/AlexxIT/go2rtc>`_ binary that runs as a separate
companion service and which the gallery embeds via an ``<iframe>``.

The only responsibility of this module is to translate the validated
:class:`wildlife.config.Config` into a ``go2rtc.yaml`` config file describing two
streams per camera (``<id>_main`` and ``<id>_sub``). It is intentionally
import-light -- only the standard library, PyYAML and :mod:`wildlife.config` --
so it never pulls in torch/cv2.
"""

from __future__ import annotations

from wildlife.stream.config_gen import (
    build_go2rtc_config,
    main,
    write_go2rtc_config,
)

__all__ = [
    "build_go2rtc_config",
    "write_go2rtc_config",
    "main",
]
