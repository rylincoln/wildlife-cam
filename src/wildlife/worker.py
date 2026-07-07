"""Single-consumer worker that ties the wildlife detection pipeline together.

This is the long-running process (started as a ``launchd`` LaunchDaemon in
production) that:

1. Loads and validates configuration.
2. Loads the YOLO model **once** onto the inference device (MPS / CPU).
3. Opens the SQLite store (creating the schema if needed).
4. Starts one :class:`~wildlife.events.base.EventSource` per camera, each running
   in its own daemon thread and feeding a shared :class:`queue.Queue`.
5. Runs a **single** consumer loop that, for every event, applies the resource
   guard (``detect_every_nth_event``), the :class:`~wildlife.gate.Deduper`
   (cooldown + burst cap), grabs an RTSP frame burst, runs inference on each
   frame, gates the detections, and saves either the single best frame or every
   positive frame (per ``save_best_only``).
6. Logs every capture decision (kept / rejected + reason) for tuning.
7. Shuts down gracefully on ``SIGTERM`` / ``SIGINT`` (launchd stop / Ctrl-C):
   stops producers, closes event sources, drains the queue, and closes the DB.

Only the worker (a single thread) ever touches the detector, store, and deduper,
so those components need no internal locking. The producer threads only touch the
shared queue and the source registry (guarded by a lock).

Heavy imports (``detect``/``capture``, which pull in ``torch``/``cv2``) live at
module scope here on purpose: this is an application entry point, not one of the
light, hardware-free library modules.
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
from datetime import datetime
from types import FrameType

from wildlife.audio import AudioAnalyzer
from wildlife.capture import grab_burst
from wildlife.config import CameraConfig, Config, load_config
from wildlife.detect import Detector
from wildlife.events.audio_detection import AudioDetectionSource
from wildlife.events.base import EventSource, make_event_source
from wildlife.events.continuous_motion import EVENT_KIND as CONTINUOUS_EVENT_KIND
from wildlife.gate import Deduper, pick_best, select_keepers
from wildlife.megadetector import MegaDetector, second_pass_decision
from wildlife.models import CameraEvent, Detection
from wildlife.store import Store

__all__ = ["run", "main"]

logger = logging.getLogger(__name__)

# Upper bound on queued, undelivered events. If the consumer ever falls far
# behind (e.g. a storm of motion events during a GPU-contended moment), we drop
# the newest events rather than let memory grow unbounded -- staying lean is a
# hard requirement on the shared 8GB host.
_QUEUE_MAXSIZE = 256
# Reconnect backoff bounds for a flapping event source.
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
# How long the consumer blocks on an empty queue before re-checking shutdown.
_QUEUE_GET_TIMEOUT_S = 0.5
# Grace period to let each producer thread notice shutdown and exit.
_PRODUCER_JOIN_TIMEOUT_S = 2.0


def _now() -> datetime:
    """Return the current wall-clock time used for dedupe and capture stamps.

    Centralised so every timestamp the worker mints comes from one place. The
    gate logic itself never reads the clock -- it is always handed a ``now`` --
    so this is the *only* place the worker samples real time for decisions.
    """
    return datetime.now()


def _rtsp_port(listen: str) -> int:
    """Extract the go2rtc RTSP port from a bind string (e.g. ``":8554"``)."""
    return int(listen.rsplit(":", 1)[-1])


def _ensure_logging() -> None:
    """Install a sensible default logging configuration if none exists.

    When run under ``launchd`` (or directly), stdout/stderr are captured to a log
    file, so a clear, greppable line format is what matters. If the embedding
    application has already configured logging, we leave it untouched.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )


class _Worker:
    """Owns the queue, threads, and pipeline components for one run.

    Constructed from a validated :class:`~wildlife.config.Config`. Call
    :meth:`run` to start producers and block on the consumer loop until a
    shutdown signal arrives.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # camera_id -> CameraConfig, so the consumer can grab the right stream.
        self._cameras: dict[str, CameraConfig] = {c.id: c for c in config.cameras}

        self._queue: "queue.Queue[CameraEvent]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._shutdown = threading.Event()

        # Built in _setup() (kept None until then so failures surface early).
        self._detector: Detector | None = None
        self._megadetector: MegaDetector | None = None
        self._store: Store | None = None
        self._deduper: Deduper | None = None
        self._audio_analyzer: AudioAnalyzer | None = None
        self._audio_sources: list = []

        # Producer bookkeeping. The source registry is touched by producer
        # threads (on reconnect) and by teardown, so guard it with a lock.
        self._producer_threads: list[threading.Thread] = []
        self._sources: dict[str, EventSource] = {}
        self._sources_lock = threading.Lock()

        # Resource-guard counter (consumer-thread-only, so no lock needed).
        self._event_count = 0
        # Per-camera time of the last MegaDetector *rescue* inference, used to
        # throttle the recall net independently of the save-keyed dedupe cooldown
        # (consumer-thread-only, so no lock needed).
        self._md_last_rescue: dict[str, datetime] = {}

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Start producers and run the consumer loop until shutdown.

        Loads the model/store/deduper, installs signal handlers, spins up one
        producer thread per camera, then blocks consuming events on the calling
        (main) thread so that POSIX signals are delivered here. Always tears
        down cleanly via ``finally``.
        """
        if not self._cameras:
            logger.error("No cameras configured; nothing to do. Exiting.")
            return

        # Cover the whole startup path so _teardown() always runs (and closes the
        # store / event sources) even if signal-handler install or producer
        # startup raises after the Store/SQLite connection is opened in _setup().
        try:
            self._setup()
            self._install_signal_handlers()
            self._start_producers()
            logger.info(
                "Worker started: %d camera(s) [%s], event_source=%s, device=%s.",
                len(self._cameras),
                ", ".join(self._cameras),
                self._config.event_source,
                self._detector.device_in_use() if self._detector else "?",
            )
            self._consume()
        finally:
            self._teardown()

    def _setup(self) -> None:
        """Load the model once, open the store, and build the deduper."""
        cfg = self._config

        # Model load is the expensive step; do it exactly once and reuse it for
        # every event across the worker's lifetime.
        self._detector = Detector(cfg.detection.model_path, cfg.detection.device)
        logger.info(
            "Detector ready (requested device=%s, in use=%s).",
            cfg.detection.device,
            self._detector.device_in_use(),
        )

        # Optional MegaDetector second pass. Load once here; a load failure must
        # not take down the vision worker (mirrors the audio analyzer below).
        if cfg.megadetector.enabled:
            try:
                self._megadetector = MegaDetector(
                    weights_path=cfg.megadetector.weights_path,
                    device=cfg.detection.device,
                    confidence=cfg.megadetector.confidence,
                    imgsz=cfg.megadetector.imgsz,
                )
                logger.info(
                    "MegaDetector second pass enabled (in use=%s, classes=%s).",
                    self._megadetector.device_in_use(),
                    cfg.megadetector.classes,
                )
            except Exception:  # noqa: BLE001 - MD is a bonus; never crash the worker
                logger.exception(
                    "megadetector.enabled but MegaDetector failed to load "
                    "(is weights_path=%s present and the [detect] extra installed?). "
                    "Continuing without the second pass.",
                    cfg.megadetector.weights_path,
                )
                self._megadetector = None

        self._store = Store(
            cfg.storage.db_path,
            cfg.storage.captures_dir,
            jpeg_quality=cfg.storage.jpeg_quality,
            thumbnail_px=cfg.storage.thumbnail_px,
        )
        self._store.init_schema()
        logger.info(
            "Store ready (db=%s, captures_dir=%s).",
            cfg.storage.db_path,
            cfg.storage.captures_dir,
        )

        self._deduper = Deduper(
            cfg.dedupe.cooldown_s,
            cfg.resource_guard.max_burst_per_minute,
        )

        if cfg.audio.enabled:
            try:
                self._audio_analyzer = AudioAnalyzer(cfg.audio)
                logger.info("Audio bird-ID enabled (BirdNET loaded).")
            except Exception:  # noqa: BLE001 - audio must not take down the vision worker
                logger.exception(
                    "Audio bird-ID enabled but BirdNET failed to load (is the [audio] extra "
                    "installed / network available for first-run weights?). Continuing without audio."
                )
                self._audio_analyzer = None

    def _install_signal_handlers(self) -> None:
        """Trip the shutdown event on SIGTERM/SIGINT (must run on main thread)."""

        def _handler(signum: int, _frame: FrameType | None) -> None:
            name = signal.Signals(signum).name
            if self._shutdown.is_set():
                logger.warning("Received %s again; shutdown already in progress.", name)
                return
            logger.info("Received %s; initiating graceful shutdown.", name)
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _start_producers(self) -> None:
        """Launch the primary event producer per camera, plus continuous if enabled."""
        continuous_on = self._config.continuous.enabled
        for camera in self._cameras.values():
            primary = threading.Thread(
                target=self._produce,
                args=(camera, self._config.event_source),
                name=f"events-{camera.id}",
                daemon=True,
            )
            primary.start()
            self._producer_threads.append(primary)
            if continuous_on:
                motion = threading.Thread(
                    target=self._produce,
                    args=(camera, "continuous_motion"),
                    name=f"motion-{camera.id}",
                    daemon=True,
                )
                motion.start()
                self._producer_threads.append(motion)

        if self._config.audio.enabled and self._audio_analyzer is not None:
            for camera in self._cameras.values():
                source = AudioDetectionSource(
                    camera, self._config, self._audio_analyzer, self._store
                )
                source.start()
                self._audio_sources.append(source)

    def _teardown(self) -> None:
        """Close event sources, drain the queue, and close the store."""
        logger.info("Tearing down: closing event sources and store.")
        self._shutdown.set()

        for source in self._audio_sources:
            try:
                source.stop()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                logger.exception("Error stopping an audio source.")

        # Best-effort close of every active source so any thread blocked inside
        # stream() is unblocked. EventSource.close() is optional in the contract.
        with self._sources_lock:
            sources = list(self._sources.values())
        # Closed sequentially, so a continuous source's close() blocking up to
        # ~10s on a wedged cv2 read (the read timeout) means N cameras wedged at
        # once cost worst-case ~N x 10s here -- bounded, daemon-threaded, never hangs.
        for source in sources:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup must not raise
                    logger.exception("Error closing an event source.")

        for thread in self._producer_threads:
            thread.join(timeout=_PRODUCER_JOIN_TIMEOUT_S)

        drained = self._drain_queue()
        if drained:
            logger.info("Discarded %d undelivered event(s) during shutdown.", drained)

        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                logger.exception("Error closing the store.")

        logger.info("Worker stopped.")

    def _drain_queue(self) -> int:
        """Empty the queue without further network work; return count discarded."""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            drained += 1
        return drained

    # -- producer side -----------------------------------------------------

    def _produce(self, camera: CameraConfig, kind: str) -> None:
        """Stream events of one ``kind`` from a camera onto the shared queue.

        Each iteration (re)creates the camera's event source for ``kind`` and
        forwards every :class:`CameraEvent` it yields onto the queue, reconnecting
        with capped exponential backoff. The source registry is keyed on
        ``camera.id:kind`` so a camera's multiple producers (e.g. reolink +
        continuous) never clobber each other and both are closed on teardown.
        """
        backoff = _INITIAL_BACKOFF_S
        source_key = f"{camera.id}:{kind}"
        while not self._shutdown.is_set():
            try:
                source = make_event_source(kind, camera, self._config)
            except Exception:  # noqa: BLE001 - keep the producer alive
                logger.exception(
                    "Camera %s: failed to create event source (%s).", camera.id, kind
                )
                if self._shutdown.wait(backoff):
                    return
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue

            with self._sources_lock:
                self._sources[source_key] = source
            logger.info("Camera %s: event source started (%s).", camera.id, kind)

            try:
                for event in source.stream():
                    if self._shutdown.is_set():
                        break
                    self._enqueue(camera, event)
                    backoff = _INITIAL_BACKOFF_S  # healthy stream resets backoff
            except Exception:  # noqa: BLE001 - reconnect rather than die
                logger.exception("Camera %s: event stream error.", camera.id)

            if self._shutdown.is_set():
                break

            logger.warning(
                "Camera %s: event stream ended; reconnecting in %.1fs.",
                camera.id,
                backoff,
            )
            if self._shutdown.wait(backoff):
                break
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

        logger.info("Camera %s: producer thread exiting.", camera.id)

    def _enqueue(self, camera: CameraConfig, event: CameraEvent) -> None:
        """Put one event on the queue, dropping it if the queue is saturated."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(
                "Camera %s: event queue full (%d); dropping event ts=%s.",
                camera.id,
                _QUEUE_MAXSIZE,
                getattr(event, "event_ts", "?"),
            )
            return
        logger.info(
            "Camera %s: event received (kind=%s, ts=%s) -> queued (depth=%d).",
            event.camera_id,
            event.kind,
            event.event_ts,
            self._queue.qsize(),
        )

    # -- consumer side -----------------------------------------------------

    def _consume(self) -> None:
        """Block popping events and handling them until shutdown is requested.

        Any in-flight event is allowed to finish; we only re-check the shutdown
        flag between events. Remaining queued events are discarded in teardown.
        """
        while not self._shutdown.is_set():
            try:
                event = self._queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
            except queue.Empty:
                continue
            try:
                self._handle_event(event)
            except Exception:  # noqa: BLE001 - one bad event must not kill the loop
                logger.exception(
                    "Unhandled error processing event from camera %s.",
                    getattr(event, "camera_id", "?"),
                )
            finally:
                self._queue.task_done()

    def _megadetector_pass(
        self,
        frames: list,
        best_det: Detection | None,
        best_frame,
        positives: list[tuple[object, Detection]],
        save_best_only: bool,
        camera_id: str,
    ) -> tuple[Detection | None, object, list[tuple[object, Detection]], str]:
        """Run MegaDetector once on a representative frame and fold in its verdict.

        Returns ``(best_det, best_frame, positives, reason)`` -- the (possibly
        updated) result plus the MD action string ("" when nothing changed), which
        the caller uses to log an accurate reject reason. Only spends an MD
        inference when it could change the outcome, and never raises -- a failure
        leaves the first-pass result untouched.

        Note: ``suppress_false_positives`` and ``rescue_misses`` do not compose --
        if suppression empties a kept event, the recall net does NOT re-run for a
        non-overlapping animal MD may have seen (``kept`` is evaluated once). This
        only matters when both are enabled (suppression is opt-in, default off).
        """
        assert self._megadetector is not None
        mdc = self._config.megadetector
        kept = (best_det is not None) if save_best_only else bool(positives)
        now = _now()
        if kept:
            # Kept events already passed the save-keyed dedupe cooldown, so
            # override/suppression runs are bounded; run whenever they can act.
            run = mdc.person_override or mdc.suppress_false_positives
        else:
            # Rescue fires on events that never save (e.g. wind-blown vegetation),
            # so it isn't bounded by the dedupe cooldown -- apply its own throttle.
            run = mdc.rescue_misses and self._md_rescue_ready(camera_id, now, mdc.rescue_cooldown_s)
        if not run:
            return best_det, best_frame, positives, ""

        # Representative frame: the winning / most-confident frame, else the middle
        # burst frame (the rescue case, where nothing was kept).
        if save_best_only and best_frame is not None:
            rep = best_frame
        elif positives:
            rep = max(positives, key=lambda fd: fd[1].confidence)[0]
        else:
            rep = frames[len(frames) // 2]

        if not kept:
            # Record the rescue-run time even if it rescues nothing, so the
            # throttle bounds the inference rate on a busy false-trigger camera.
            self._md_last_rescue[camera_id] = now

        try:
            md_dets = self._megadetector.infer(rep)
        except Exception:  # noqa: BLE001 - never let the bonus pass crash an event
            logger.exception(
                "Camera %s: MegaDetector inference failed; keeping first-pass result.",
                camera_id,
            )
            return best_det, best_frame, positives, ""

        best_det, best_frame, positives, reason = second_pass_decision(
            md_dets=md_dets,
            best_det=best_det,
            best_frame=best_frame,
            positives=positives,
            representative_frame=rep,
            save_best_only=save_best_only,
            cfg=mdc,
            min_area_frac=self._config.detection.min_box_area_frac,
        )
        if reason:
            logger.info("Camera %s: MegaDetector %s.", camera_id, reason)
        return best_det, best_frame, positives, reason

    def _md_rescue_ready(self, camera_id: str, now: datetime, cooldown_s: float) -> bool:
        """True if enough time has passed since this camera's last rescue MD run."""
        if cooldown_s <= 0:
            return True
        last = self._md_last_rescue.get(camera_id)
        return last is None or (now - last).total_seconds() >= cooldown_s

    def _handle_event(self, event: CameraEvent) -> None:
        """Run the full pipeline for one event and log the decision.

        Order matches the spec: resource-guard skip -> dedupe/cooldown ->
        burst grab -> per-frame inference -> gate -> save (best or all positives)
        -> mark the deduper. Every branch logs a clear kept/rejected reason.
        """
        assert self._detector is not None and self._store is not None
        assert self._deduper is not None

        camera_id = event.camera_id
        event_ts = event.event_ts
        camera = self._cameras.get(camera_id)
        if camera is None:
            logger.warning("Event for unknown camera %r; ignoring.", camera_id)
            return

        # 1) Resource guard: optionally process only every Nth event.
        self._event_count += 1
        nth = self._config.resource_guard.detect_every_nth_event
        if nth > 1 and (self._event_count % nth) != 0:
            logger.info(
                "Camera %s: REJECTED event #%d (resource_guard detect_every_nth_event=%d).",
                camera_id,
                self._event_count,
                nth,
            )
            return

        # 2) Dedupe: per-camera cooldown + global burst cap. All time injected.
        # v1 semantics: both are keyed off *saves* (mark_saved), so
        # max_burst_per_minute caps the save rate and acts as a backstop while
        # cooldown_s does the dominant throttling. See spec section 5 / README.
        now = _now()
        if not self._deduper.should_process(camera_id, now):
            logger.info(
                "Camera %s: REJECTED (dedupe: within cooldown_s=%d or burst cap "
                "max_burst_per_minute=%d).",
                camera_id,
                self._config.dedupe.cooldown_s,
                self._config.resource_guard.max_burst_per_minute,
            )
            return

        # 3) Capture a short burst from the configured stream. Continuous-motion
        # events route through go2rtc (avoids a second same-IP Reolink session);
        # Reolink events keep the direct path.
        cap = self._config.capture
        is_continuous = event.kind == CONTINUOUS_EVENT_KIND
        if is_continuous:
            rtsp_port = _rtsp_port(self._config.livestream.rtsp_listen)
            burst_url = f"rtsp://127.0.0.1:{rtsp_port}/{camera_id}_{cap.stream}"
        else:
            burst_url = None
        frames = grab_burst(
            camera,
            cap.burst_frames,
            cap.burst_interval_ms,
            cap.stream,
            cap.rtsp_timeout_s,
            rtsp_url=burst_url,
        )
        if not frames:
            logger.warning(
                "Camera %s: REJECTED (no frames captured from %s stream).",
                camera_id,
                cap.stream,
            )
            return

        # 4) Inference + gate, per frame.
        det_cfg = self._config.detection
        save_best_only = det_cfg.save_best_only
        total_raw = 0
        best_det: Detection | None = None
        best_frame = None
        positives: list[tuple[object, Detection]] = []  # (frame, detection)

        for frame in frames:
            raw = self._detector.infer(frame)
            total_raw += len(raw)
            keepers = select_keepers(raw, det_cfg)
            if not keepers:
                continue
            if save_best_only:
                frame_best = pick_best(keepers)
                if frame_best is not None and (
                    best_det is None or frame_best.confidence > best_det.confidence
                ):
                    best_det = frame_best
                    best_frame = frame
            else:
                positives.extend((frame, k) for k in keepers)

        # 4b) Optional MegaDetector second pass (per-event, one representative
        # frame): relabel person, rescue a missed animal, or drop a false positive.
        md_reason = ""
        if self._megadetector is not None:
            best_det, best_frame, positives, md_reason = self._megadetector_pass(
                frames, best_det, best_frame, positives, save_best_only, camera_id
            )

        # 5) Save: the single best frame, or every positive detection.
        source_kind = "continuous" if is_continuous else "reolink"
        capture_ts = _now()
        saved_ids: list[int] = []
        if save_best_only:
            if best_det is not None and best_frame is not None:
                saved_ids.append(
                    self._store.save_capture(
                        camera_id=camera_id,
                        event_ts=event_ts,
                        capture_ts=capture_ts,
                        frame=best_frame,
                        det=best_det,
                        source_kind=source_kind,
                    )
                )
        else:
            for frame, det in positives:
                saved_ids.append(
                    self._store.save_capture(
                        camera_id=camera_id,
                        event_ts=event_ts,
                        capture_ts=capture_ts,
                        frame=frame,
                        det=det,
                        source_kind=source_kind,
                    )
                )

        # 6) Record the decision and (if anything saved) arm the deduper.
        if saved_ids:
            # Anchor the cooldown/burst window to the actual save time
            # (capture_ts), not the event-start `now` sampled before the
            # (possibly slow) burst grab + inference, so the suppression window
            # isn't silently shortened under GPU contention.
            self._deduper.mark_saved(camera_id, capture_ts)
            top = best_det or max(
                (d for _, d in positives), key=lambda d: d.confidence, default=None
            )
            logger.info(
                "Camera %s: KEPT %d capture(s) ids=%s (best label=%s conf=%.3f) "
                "from %d frame(s), %d raw detection(s).",
                camera_id,
                len(saved_ids),
                saved_ids,
                top.label if top else "?",
                top.confidence if top else 0.0,
                len(frames),
                total_raw,
            )
        elif "suppressed" in md_reason:
            # A detection DID pass the gate; MegaDetector dropped it. Attribute the
            # rejection accurately so gate-threshold tuning isn't misdirected.
            logger.info(
                "Camera %s: REJECTED (MegaDetector %s) across %d frame(s), %d raw detection(s).",
                camera_id,
                md_reason,
                len(frames),
                total_raw,
            )
        else:
            logger.info(
                "Camera %s: REJECTED (no detection passed the gate: animal_classes "
                "+ conf>=%.2f + area>=%.3f) across %d frame(s), %d raw detection(s).",
                camera_id,
                det_cfg.confidence_threshold,
                det_cfg.min_box_area_frac,
                len(frames),
                total_raw,
            )


def run(config_path: str = "config.yaml") -> None:
    """Load ``config_path`` and run the worker until a shutdown signal.

    This is the programmatic entry point. It loads and validates configuration,
    constructs a :class:`_Worker`, and blocks until ``SIGTERM``/``SIGINT``.

    Args:
        config_path: Path to the YAML configuration file. Defaults to
            ``"config.yaml"`` in the current working directory.
    """
    _ensure_logging()
    logger.info("Loading configuration from %s", config_path)
    config = load_config(config_path)
    worker = _Worker(config)
    worker.run()


def main() -> None:
    """Console-script entry point: ``argv[1]`` is an optional config path."""
    _ensure_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)


if __name__ == "__main__":
    main()
