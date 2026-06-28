"""ONVIF PullPoint event source (fallback) built on ``onvif-zeep``.

Used when the native ``reolink-aio`` path is unavailable or flaky. It opens an
ONVIF *PullPoint* subscription and long-polls the camera for notification
messages. We only care about the *on* edge (a trigger fired), so the well-known
Reolink "no motion-off event" quirk is irrelevant here -- PullPoint property
events are change notifications, so each "state became true" message is already
a clean rising edge and we emit one :class:`~wildlife.models.CameraEvent` per
such message.

``onvif-zeep`` (and the underlying ``zeep`` SOAP client) are synchronous, so
unlike the reolink source there is no asyncio: the background worker thread runs
a blocking ``PullMessages`` loop directly. The subscription is (re)created
inside a reconnect-with-backoff outer loop so expiry or transient faults simply
rebuild it.

The ``onvif`` package is imported lazily so importing this module never requires
the optional ``cameras`` dependency group. All parsing of notification payloads
is defensive because the exact zeep object shape varies by camera and library
version.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from wildlife.events.base import CameraEvent, _QueueBackedEventSource

__all__ = ["OnvifEventSource"]

logger = logging.getLogger(__name__)

# Long-poll timeout per PullMessages call. SOAP calls are not cancellable, so
# this also bounds how long close() waits for an in-flight pull to return.
_PULL_TIMEOUT_S = 5
# Max messages returned per PullMessages call.
_MESSAGE_LIMIT = 100
# Underlying HTTP transport timeouts (seconds): connect/load vs. per-operation.
# The operation timeout must exceed the long-poll window so PullMessages isn't
# cut off server-side mid-wait.
_TRANSPORT_CONNECT_TIMEOUT_S = 5
_OPERATION_TIMEOUT_S = _PULL_TIMEOUT_S + 5
# Exponential-backoff bounds for re-subscribing, in seconds.
_INITIAL_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0

# Substrings (lower-cased) in an ONVIF topic that we classify as an animal AI
# trigger versus generic motion. The Mac-side YOLO pass is the real gate, so
# this only affects the ``kind`` label we record.
_ANIMAL_TOPIC_HINTS = ("animal", "dog", "cat", "pet")
_MOTION_TOPIC_HINTS = ("motion", "cellmotion", "motionalarm")
# SimpleItem names whose truthy value indicates an active trigger.
_STATE_ITEM_NAMES = ("isMotion", "IsMotion", "State", "isAnimal", "IsAnimal")
_TRUE_VALUES = ("true", "1", "yes", "on")


class OnvifEventSource(_QueueBackedEventSource):
    """Yield :class:`CameraEvent` triggers for one camera via ONVIF PullPoint.

    Parameters
    ----------
    camera:
        The camera's :class:`wildlife.config.CameraConfig`. ``host``,
        ``onvif_port``, ``username`` and ``password`` are used to connect.
    """

    # SOAP PullMessages cannot be interrupted, so allow a little longer than the
    # worst-case single blocking operation before warning about a stuck thread.
    _JOIN_TIMEOUT_S = float(_OPERATION_TIMEOUT_S + 5)

    def _run(self) -> None:
        """Worker-thread entry: (re)subscribe and pull messages with backoff."""
        backoff = _INITIAL_BACKOFF_S
        while not self._stopping:
            pullpoint = None
            try:
                camera = self._make_camera()
                pullpoint = self._create_pullpoint(camera)
                logger.info(
                    "ONVIF PullPoint subscribed for camera %s", self.camera.id
                )
                backoff = _INITIAL_BACKOFF_S  # reset after a successful subscribe
                self._pull_loop(pullpoint)
            except Exception as exc:  # noqa: BLE001 - any failure -> backoff + resubscribe
                if self._stopping:
                    break
                logger.warning(
                    "ONVIF listener for camera %s failed (%s); re-subscribing in %.1fs",
                    self.camera.id,
                    exc,
                    backoff,
                )
                self._sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
            finally:
                self._safe_unsubscribe(pullpoint)

    # -- connection / subscription --------------------------------------

    def _make_camera(self):
        """Construct an ``onvif.ONVIFCamera`` (imported lazily).

        Raises a clear error if the optional ``onvif-zeep`` dependency is not
        installed.
        """
        try:
            from onvif import ONVIFCamera
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "onvif-zeep is required for event_source='onvif_bridge'. "
                "Install it with: pip install 'wildlife-detect[cameras]'"
            ) from exc

        # Bound the underlying HTTP transport so a dead camera fails fast (and we
        # retry on backoff) rather than hanging on zeep/requests' long defaults.
        # operation_timeout must exceed the PullMessages long-poll window. If the
        # transport can't be built on this version, fall back to the default.
        transport = None
        try:
            from zeep.transports import Transport

            transport = Transport(
                timeout=_TRANSPORT_CONNECT_TIMEOUT_S,
                operation_timeout=_OPERATION_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - optional hardening; default transport is fine
            logger.debug("Could not build a bounded zeep Transport", exc_info=True)

        # Positional signature is stable across onvif-zeep versions:
        # ONVIFCamera(host, port, user, passwd). The library bundles its own
        # WSDL files, so no wsdl_dir is required in the common case.
        return ONVIFCamera(
            self.camera.host,
            self.camera.onvif_port,
            self.camera.username,
            self.camera.password,
            transport=transport,
        )

    def _create_pullpoint(self, camera):
        """Create a PullPoint subscription and return its service proxy.

        Handles the two common onvif-zeep flavours: a convenience
        ``create_pullpoint_subscription()`` on the camera, and/or the
        lower-level events-service ``CreatePullPointSubscription`` call. Then
        issues ``SetSynchronizationPoint`` (best effort) so the camera starts
        delivering current property state.
        """
        # Preferred: ask the camera to establish the subscription first. Some
        # versions require this before create_pullpoint_service() will work.
        creator = getattr(camera, "create_pullpoint_subscription", None)
        if callable(creator):
            try:
                creator()
            except Exception:  # noqa: BLE001 - fall through to the events-service path
                logger.debug(
                    "create_pullpoint_subscription() failed; trying events service",
                    exc_info=True,
                )
                self._create_subscription_via_events(camera)
        else:
            self._create_subscription_via_events(camera)

        pullpoint = camera.create_pullpoint_service()

        # SetSynchronizationPoint prompts the camera to emit current state;
        # not all firmwares implement it, so this is best-effort.
        sync = getattr(pullpoint, "SetSynchronizationPoint", None)
        if callable(sync):
            try:
                sync()
            except Exception:  # noqa: BLE001 - optional, non-fatal
                logger.debug(
                    "SetSynchronizationPoint not supported by camera %s",
                    self.camera.id,
                    exc_info=True,
                )
        return pullpoint

    @staticmethod
    def _create_subscription_via_events(camera) -> None:
        """Create the PullPoint subscription through the events service proxy."""
        event_service = camera.create_events_service()
        event_service.CreatePullPointSubscription()

    def _pull_loop(self, pullpoint) -> None:
        """Repeatedly PullMessages and emit a CameraEvent per active trigger."""
        while not self._stopping:
            request = pullpoint.create_type("PullMessages")
            request.Timeout = timedelta(seconds=_PULL_TIMEOUT_S)
            request.MessageLimit = _MESSAGE_LIMIT
            # Blocks up to _PULL_TIMEOUT_S waiting for notifications.
            response = pullpoint.PullMessages(request)
            for message in getattr(response, "NotificationMessage", None) or []:
                event = self._parse_message(message)
                if event is not None:
                    self._emit(event)
                    logger.info(
                        "ONVIF event: camera=%s kind=%s",
                        self.camera.id,
                        event.kind,
                    )

    def _safe_unsubscribe(self, pullpoint) -> None:
        """Best-effort PullPoint teardown; never raises."""
        if pullpoint is None:
            return
        unsubscribe = getattr(pullpoint, "Unsubscribe", None)
        if not callable(unsubscribe):
            return
        try:
            unsubscribe()
        except Exception:  # noqa: BLE001 - teardown must not mask the real error
            logger.debug(
                "Ignoring error during ONVIF unsubscribe for camera %s",
                self.camera.id,
                exc_info=True,
            )

    # -- message parsing -------------------------------------------------

    def _parse_message(self, message) -> CameraEvent | None:
        """Turn one ONVIF NotificationMessage into a CameraEvent, or ``None``.

        Returns an event only when the message carries an active (true) trigger
        state. ``kind`` is derived from the ONVIF topic: ``"animal"`` when the
        topic names an animal/pet AI class, otherwise ``"motion"``.
        """
        topic = self._extract_topic(message).lower()

        # Ignore non-animal, non-motion topics (e.g. people/vehicle/face) -- the
        # worker's YOLO pass only cares about wildlife, so we don't burn a burst
        # grab on them. Animal hints take priority over motion.
        if any(hint in topic for hint in _ANIMAL_TOPIC_HINTS):
            kind = "animal"
        elif any(hint in topic for hint in _MOTION_TOPIC_HINTS):
            kind = "motion"
        else:
            return None

        if not self._message_state_is_active(message):
            return None

        return CameraEvent(
            camera_id=self.camera.id,
            event_ts=datetime.now(timezone.utc),
            kind=kind,
        )

    @staticmethod
    def _extract_topic(message) -> str:
        """Pull the topic string from a NotificationMessage, defensively."""
        topic = getattr(message, "Topic", None)
        if topic is None:
            return ""
        # zeep typically exposes mixed content as ``_value_1``.
        value = getattr(topic, "_value_1", None)
        if value is not None:
            return str(value)
        return str(topic)

    @classmethod
    def _message_state_is_active(cls, message) -> bool:
        """True if the message's data SimpleItems indicate an active trigger.

        Walks ``Message -> Data -> SimpleItem`` (the standard ONVIF property
        layout) and returns True when a recognised state item holds a truthy
        value. If no SimpleItems are present at all, we treat the notification
        itself as the trigger (some cameras send bare on-edge notifications).
        """
        simple_items = cls._extract_simple_items(message)
        if not simple_items:
            # No structured state -> treat the notification as the trigger edge.
            return True

        saw_state_item = False
        for item in simple_items:
            name = getattr(item, "Name", None)
            value = getattr(item, "Value", None)
            if name in _STATE_ITEM_NAMES:
                saw_state_item = True
                if str(value).strip().lower() in _TRUE_VALUES:
                    return True
        # If we recognised a state item but none were true, it's an off edge.
        # If we recognised none, fall back to treating it as a trigger.
        return not saw_state_item

    @staticmethod
    def _extract_simple_items(message) -> list:
        """Extract the list of SimpleItem elements from a NotificationMessage."""
        inner = getattr(message, "Message", None)
        # zeep wraps the inner <Message> element content in ``_value_1``.
        data_holder = getattr(inner, "_value_1", inner)
        data = getattr(data_holder, "Data", None)
        if data is None:
            return []
        simple_item = getattr(data, "SimpleItem", None)
        if simple_item is None:
            return []
        if isinstance(simple_item, list):
            return simple_item
        return [simple_item]
