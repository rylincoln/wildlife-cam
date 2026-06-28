"""Event-source abstraction for camera-side motion / animal triggers.

An :class:`EventSource` turns a camera's push/poll detection feed into a simple
synchronous iterator of :class:`~wildlife.models.CameraEvent` values that the
worker can consume without caring about asyncio, SOAP, or vendor quirks::

    src = make_event_source(cfg.event_source, camera)
    for event in src.stream():
        enqueue(event)

Two concrete sources live alongside this module:

* :mod:`wildlife.events.reolink_native` -- preferred, uses ``reolink-aio``.
* :mod:`wildlife.events.onvif_bridge` -- fallback, uses ``onvif-zeep`` PullPoint.

Both share the :class:`_QueueBackedEventSource` machinery defined here: the
real listener runs on a background thread (an asyncio loop for reolink, a
blocking SOAP pull loop for ONVIF) and pushes events onto a thread-safe queue
that :meth:`stream` drains. This keeps the consumer side dead simple and fully
synchronous while the producer side can do whatever the underlying library
needs.

Only :mod:`wildlife.models` and the standard library are imported at module
scope. The heavy/optional vendor libraries (``reolink-aio``, ``onvif-zeep``)
are imported lazily inside the concrete sources so that importing this package
-- including the ``__init__`` re-exports -- never requires the ``cameras``
extra to be installed.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator

# Re-export CameraEvent so callers may ``from wildlife.events.base import CameraEvent``.
from wildlife.models import CameraEvent

__all__ = ["CameraEvent", "EventSource", "make_event_source"]

logger = logging.getLogger(__name__)


class EventSource(ABC):
    """Abstract source of :class:`CameraEvent` triggers for one camera.

    Subclasses implement :meth:`stream`, a generator that yields a
    :class:`CameraEvent` each time the camera reports a (preferably animal,
    falling back to motion) detection. Implementations are expected to reconnect
    on their own with backoff so the generator simply keeps producing across
    transient network/camera drops.
    """

    @abstractmethod
    def stream(self) -> Iterator[CameraEvent]:
        """Yield :class:`CameraEvent` values as the camera reports detections.

        The generator runs until :meth:`close` is called (or the process exits).
        It must not raise on routine connection drops -- it should recover and
        keep yielding.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    def close(self) -> None:
        """Stop listening and release resources. Idempotent; default is a no-op."""


class _QueueBackedEventSource(EventSource):
    """Reusable :class:`EventSource` backed by a background thread + queue.

    Concrete sources implement :meth:`_run`, which executes on a daemon thread
    and emits events via :meth:`_emit`. The thread should periodically consult
    :attr:`_stopping` (or call :meth:`_sleep` for interruptible waits) so that
    :meth:`close` shuts it down promptly.

    :meth:`stream` lazily starts the worker thread on first iteration and then
    blocks on the queue, yielding events until the source is closed (or the
    worker thread terminates and posts the sentinel).
    """

    #: Default seconds to wait for the worker thread to join on :meth:`close`.
    _JOIN_TIMEOUT_S: float = 10.0

    def __init__(self, camera: object) -> None:
        # ``camera`` is a wildlife.config.CameraConfig; typed as object here to
        # avoid importing the pydantic config module into this stdlib-only base.
        self.camera = camera
        self._queue: queue.Queue[object] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        # Per-instance sentinel: pushed onto the queue to signal end-of-stream.
        self._sentinel = object()

    # -- producer side ---------------------------------------------------

    @abstractmethod
    def _run(self) -> None:
        """Listen for camera events on the worker thread, emitting via :meth:`_emit`.

        Runs until :attr:`_stopping` becomes true. May raise; exceptions are
        logged by :meth:`_thread_main` and end the stream cleanly.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    def _thread_main(self) -> None:
        """Worker-thread entry point: run the listener and always post the sentinel."""
        try:
            self._run()
        except Exception:  # noqa: BLE001 - top-level guard for the worker thread
            logger.exception(
                "Event listener thread for camera %s crashed", self._camera_id()
            )
        finally:
            # Guarantee the consumer in stream() is unblocked and terminates.
            self._queue.put(self._sentinel)

    def _emit(self, event: CameraEvent) -> None:
        """Publish a :class:`CameraEvent` to the queue unless shutting down."""
        if self._stop.is_set():
            return
        self._queue.put(event)

    @property
    def _stopping(self) -> bool:
        """True once :meth:`close` has requested shutdown."""
        return self._stop.is_set()

    def _signal_stop(self) -> None:
        """Hook invoked by :meth:`close` to interrupt in-flight blocking work.

        Setting :attr:`_stop` only unblocks waits routed through :meth:`_sleep`.
        A listener parked in a non-cancellable call (e.g. an asyncio network
        ``await`` or a blocking SOAP request) needs an active nudge to wake. The
        default is a no-op; subclasses override it (e.g. to cancel the asyncio
        task that drives the listener). Must be safe to call from another thread
        and must never raise.
        """

    def _sleep(self, seconds: float, step: float = 0.25) -> None:
        """Sleep up to ``seconds``, waking early if shutdown is requested.

        Used by subclasses for backoff/poll intervals so :meth:`close` stays
        responsive instead of blocking on a long ``time.sleep``.
        """
        # ``Event.wait`` returns immediately once the stop flag is set, so we
        # loop in small steps to honour the requested total without oversleeping.
        remaining = seconds
        while remaining > 0.0:
            if self._stop.wait(timeout=min(step, remaining)):
                return
            remaining -= step

    def _camera_id(self) -> str:
        """Best-effort camera id for logging (tolerates a bare object in tests)."""
        return str(getattr(self.camera, "id", self.camera))

    # -- consumer side ---------------------------------------------------

    def _start(self) -> None:
        """Start the worker thread once (idempotent, thread-safe)."""
        with self._thread_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name=f"events-{self._camera_id()}",
                    daemon=True,
                )
                self._thread.start()

    def stream(self) -> Iterator[CameraEvent]:
        """Yield events from the worker thread until closed. See :class:`EventSource`."""
        self._start()
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            # Only CameraEvent instances are ever enqueued besides the sentinel.
            assert isinstance(item, CameraEvent)
            yield item

    def close(self) -> None:
        """Signal the worker thread to stop, unblock the consumer, and join."""
        self._stop.set()
        # Interrupt any non-cancellable blocking work in the listener.
        try:
            self._signal_stop()
        except Exception:  # noqa: BLE001 - shutdown hook must never raise
            logger.debug(
                "Error in _signal_stop for camera %s", self._camera_id(), exc_info=True
            )
        # Wake any consumer parked on queue.get() so stream() can return.
        self._queue.put(self._sentinel)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Event listener thread for camera %s did not stop within %.1fs",
                    self._camera_id(),
                    self._JOIN_TIMEOUT_S,
                )


def make_event_source(kind: str, camera: object) -> EventSource:
    """Construct the configured :class:`EventSource` for a camera.

    Args:
        kind: ``"reolink_native"`` (preferred, ``reolink-aio``) or
            ``"onvif_bridge"`` (fallback, ``onvif-zeep`` PullPoint).
        camera: The camera's :class:`wildlife.config.CameraConfig`.

    Returns:
        A ready-to-use (not yet started) :class:`EventSource`.

    Raises:
        ValueError: If ``kind`` is not a recognised event-source identifier.

    The concrete source modules are imported lazily so that selecting one engine
    never forces the other engine's third-party dependency to be installed.
    """
    if kind == "reolink_native":
        from wildlife.events.reolink_native import ReolinkEventSource

        return ReolinkEventSource(camera)
    if kind == "onvif_bridge":
        from wildlife.events.onvif_bridge import OnvifEventSource

        return OnvifEventSource(camera)
    raise ValueError(
        f"Unknown event_source {kind!r}; expected 'reolink_native' or 'onvif_bridge'"
    )
