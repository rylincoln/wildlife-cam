"""Reolink native event source built on the ``reolink-aio`` library.

This is the *preferred* event engine. ``reolink-aio`` is the same library Home
Assistant's Reolink integration uses; it understands the camera's detection
state (motion plus on-board AI object classes such as people / vehicle /
dog-cat / animal) and is the most robust way to know "something just happened"
without hammering RTSP.

Design
------
The ``reolink-aio`` API is fully asyncio and varies noticeably between versions
(method names, AI object-type spellings, channel discovery). To keep the rest
of the system simple and synchronous -- and to stay resilient to those version
differences -- this source:

1. Runs an asyncio event loop on a background thread (provided by the
   :class:`~wildlife.events.base._QueueBackedEventSource` machinery).
2. Connects a ``reolink_aio.api.Host`` and refreshes its detection state on a
   short interval (``get_states``). Polling is the lowest-common-denominator
   tier of Reolink's own push hierarchy and works on every firmware/library
   combination; true ONVIF/TCP *push* would require running an inbound webhook
   server, which is out of scope for a single-host appliance.
3. Performs rising-edge detection per channel: it emits a
   :class:`~wildlife.models.CameraEvent` the moment a channel transitions from
   "nothing" to "animal or motion active", labelling ``kind`` ``"animal"`` when
   an AI animal class is responsible and ``"motion"`` otherwise. The Mac-side
   YOLO pass remains the precise gate, so a slightly loose trigger here is fine.
4. Reconnects with exponential backoff on any error, alternating the connection
   port across attempts to tolerate setups where the HTTP-API port differs from
   the configured ONVIF port.

All access to ``reolink-aio`` attributes/methods is defensive (``getattr`` +
``try``/``except``) precisely because the surface changes across releases.
``reolink-aio`` itself is imported lazily so importing this module never
requires the optional ``cameras`` dependency group.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from wildlife.events.base import CameraEvent, _QueueBackedEventSource

__all__ = ["ReolinkEventSource"]

logger = logging.getLogger(__name__)

# How often to refresh and check the camera's detection state, in seconds.
_POLL_INTERVAL_S = 1.0
# Exponential-backoff bounds for reconnect attempts, in seconds.
_INITIAL_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
# Per-request timeout handed to reolink-aio's Host so a dead camera fails fast
# (and we retry on backoff) instead of hanging on the library's 30s default.
_CONNECT_TIMEOUT_S = 15
# Candidate AI object types (newest reolink-aio spellings first) that we treat
# as "wildlife". Different firmwares expose different subsets; unsupported ones
# are skipped at runtime.
_ANIMAL_AI_TYPES = ("animal", "dog_cat")


class ReolinkEventSource(_QueueBackedEventSource):
    """Yield :class:`CameraEvent` triggers for one Reolink camera via ``reolink-aio``.

    Parameters
    ----------
    camera:
        The camera's :class:`wildlife.config.CameraConfig`. ``host``,
        ``username``, ``password`` and ``onvif_port`` are used to connect.
    """

    def __init__(self, camera: object) -> None:
        super().__init__(camera)
        # Set on the worker thread in _run; read by _signal_stop from the caller
        # thread to cancel in-flight network awaits on close().
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listen_task: asyncio.Task[None] | None = None

    def _run(self) -> None:
        """Worker-thread entry: drive an asyncio loop running the async listener."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            task = loop.create_task(self._listen_async())
            self._listen_task = task
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                # Expected: close() cancelled the listener mid-await.
                logger.debug("Reolink listener for camera %s cancelled", self.camera.id)
        finally:
            self._listen_task = None
            # Best-effort cleanup of any pending tasks before closing the loop.
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                loop.close()
                self._loop = None

    def _signal_stop(self) -> None:
        """Cancel the in-flight listener task from the caller thread (see base)."""
        loop = self._loop
        task = self._listen_task
        if loop is None or task is None or loop.is_closed():
            return
        # Cancellation must be scheduled on the loop's own thread.
        loop.call_soon_threadsafe(task.cancel)

    async def _listen_async(self) -> None:
        """Connect, poll detection state, and reconnect with backoff until stopped."""
        backoff = _INITIAL_BACKOFF_S
        attempt = 0
        # Alternate connection port across reconnects. Try the library default
        # (``None`` -> reolink-aio auto-selects the HTTP/HTTPS API port, which on
        # typical Reolink firmware is HTTP:80) FIRST -- it is the fast, reliable
        # path -- then fall back to the explicitly configured ONVIF port for
        # setups whose HTTP API lives on a non-standard port. Ordering the ONVIF
        # port first can send reolink-aio down an HTTPS/443 probe that stalls for
        # up to ~90s on cameras with 443 closed (observed on the E1-class units
        # this project targets), delaying the first post-boot connect.
        ports = (None, self.camera.onvif_port)

        while not self._stopping:
            host = None
            port = ports[attempt % len(ports)]
            attempt += 1
            try:
                host = self._make_host(port)
                # Pull capabilities/host metadata; validates credentials + reachability.
                await host.get_host_data()
                channels = self._discover_channels(host)
                logger.info(
                    "Reolink listener connected for camera %s (port=%s, channels=%s)",
                    self.camera.id,
                    port,
                    channels,
                )
                backoff = _INITIAL_BACKOFF_S  # reset after a successful connect
                await self._poll_loop(host, channels)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure -> backoff + retry
                if self._stopping:
                    break
                logger.warning(
                    "Reolink listener for camera %s failed (%s); reconnecting in %.1fs",
                    self.camera.id,
                    exc,
                    backoff,
                )
                # Cancellable wait: close() cancels the task and wakes us instantly.
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
            finally:
                if host is not None:
                    await self._safe_logout(host)

    # -- connection helpers ---------------------------------------------

    def _make_host(self, port: int | None):
        """Construct a ``reolink_aio.api.Host`` (imported lazily).

        Raises a clear error if the optional ``reolink-aio`` dependency is not
        installed.
        """
        try:
            from reolink_aio.api import Host
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "reolink-aio is required for event_source='reolink_native'. "
                "Install it with: pip install 'wildlife-detect[cameras]'"
            ) from exc

        kwargs = {"timeout": _CONNECT_TIMEOUT_S}
        if port is not None:
            kwargs["port"] = port
        return Host(
            self.camera.host,
            self.camera.username,
            self.camera.password,
            **kwargs,
        )

    @staticmethod
    def _discover_channels(host) -> list[int]:
        """Return the camera's channel indices, defaulting to ``[0]``.

        Single-sensor cameras (e.g. the Reolink E1 Outdoor SE) expose one
        channel; NVRs expose several. We tolerate either ``host.channels`` (a
        list) or ``host.num_channels`` (an int), falling back to channel 0.
        """
        channels = getattr(host, "channels", None)
        if channels:
            try:
                return [int(c) for c in channels]
            except TypeError:
                pass
        num = getattr(host, "num_channels", None)
        if isinstance(num, int) and num > 0:
            return list(range(num))
        return [0]

    @staticmethod
    async def _safe_logout(host) -> None:
        """Best-effort session teardown; never raises."""
        logout = getattr(host, "logout", None)
        if logout is None:
            return
        try:
            await logout()
        except Exception:  # noqa: BLE001 - teardown must not mask the real error
            logger.debug("Ignoring error during Reolink logout", exc_info=True)

    # -- polling ---------------------------------------------------------

    async def _poll_loop(self, host, channels: list[int]) -> None:
        """Refresh detection state on an interval and emit rising-edge events."""
        # Per-channel "was something active last tick?" for edge detection.
        active_prev: dict[int, bool] = {ch: False for ch in channels}
        # Cache which animal AI types each channel actually supports.
        animal_types: dict[int, tuple[str, ...]] = {
            ch: self._supported_animal_types(host, ch) for ch in channels
        }

        while not self._stopping:
            await self._refresh_states(host, channels)
            for ch in channels:
                kind = self._active_kind(host, ch, animal_types[ch])
                now_active = kind is not None
                if now_active and not active_prev[ch]:
                    self._emit(
                        CameraEvent(
                            camera_id=self.camera.id,
                            event_ts=datetime.now(timezone.utc),
                            kind=kind,  # type: ignore[arg-type]  # kind is str here
                        )
                    )
                    logger.info(
                        "Reolink event: camera=%s channel=%s kind=%s",
                        self.camera.id,
                        ch,
                        kind,
                    )
                active_prev[ch] = now_active
            await asyncio.sleep(_POLL_INTERVAL_S)

    @staticmethod
    async def _refresh_states(host, channels: list[int]) -> None:
        """Refresh motion/AI state from the camera, across reolink-aio versions.

        Prefers the modern bulk ``get_states`` call; falls back to the older
        per-channel ``get_motion_state``/``get_ai_state`` methods if present.
        """
        get_states = getattr(host, "get_states", None)
        if callable(get_states):
            await get_states()
            return
        for ch in channels:
            for name in ("get_motion_state", "get_ai_state"):
                fn = getattr(host, name, None)
                if callable(fn):
                    try:
                        await fn(ch)
                    except Exception:  # noqa: BLE001 - try the next fetcher
                        logger.debug(
                            "reolink-aio %s(%s) failed", name, ch, exc_info=True
                        )

    def _active_kind(self, host, channel: int, animal_types: tuple[str, ...]) -> str | None:
        """Return ``"animal"``, ``"motion"``, or ``None`` for a channel's state.

        Animal AI classes win over plain motion so the more specific trigger is
        reported when both are active.
        """
        for obj in animal_types:
            if self._ai_detected(host, channel, obj):
                return "animal"
        if self._motion_detected(host, channel):
            return "motion"
        return None

    @staticmethod
    def _supported_animal_types(host, channel: int) -> tuple[str, ...]:
        """Filter :data:`_ANIMAL_AI_TYPES` to those the channel reports supporting.

        If support can't be queried on this library version, all candidates are
        kept and simply evaluate to "not detected" at runtime -- harmless.
        """
        check = getattr(host, "ai_supported", None)
        if not callable(check):
            return _ANIMAL_AI_TYPES
        supported: list[str] = []
        for obj in _ANIMAL_AI_TYPES:
            try:
                if check(channel, obj):
                    supported.append(obj)
            except Exception:  # noqa: BLE001 - signature/spelling varies; keep candidate
                supported.append(obj)
        return tuple(supported) if supported else _ANIMAL_AI_TYPES

    @staticmethod
    def _ai_detected(host, channel: int, object_type: str) -> bool:
        """True if the camera currently reports the given AI object type active."""
        fn = getattr(host, "ai_detected", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(channel, object_type))
        except Exception:  # noqa: BLE001 - unsupported type/signature -> not detected
            return False

    @staticmethod
    def _motion_detected(host, channel: int) -> bool:
        """True if the camera currently reports basic motion active."""
        fn = getattr(host, "motion_detected", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(channel))
        except Exception:  # noqa: BLE001 - tolerate signature differences
            try:
                return bool(fn())
            except Exception:  # noqa: BLE001
                return False
