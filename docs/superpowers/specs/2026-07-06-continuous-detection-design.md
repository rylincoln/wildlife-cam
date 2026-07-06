# Continuous motion-gated detection — design

**Date:** 2026-07-06
**Status:** Draft for review
**Feature:** Project 1 of "continuous detection + training flywheel." Make the user's
own YOLO the always-on arbiter (not Reolink's person/car AI) via a motion-gated
continuous producer, catching the small/distant/nocturnal wildlife the camera misses.

## 1. Goal

Add an **always-on, motion-gated** detection path that runs alongside today's
event-triggered path: a second per-camera producer watches a cheap motion signal on the
go2rtc sub-restream and, on motion, fires the **existing** capture→YOLO→gate→save
pipeline — so **your fine-tuned model, not Reolink's onboard AI, decides what counts.**
This catches known species Reolink's AI misses (small, distant, nocturnal, unlisted-by-
Reolink) that today are silently never looked at.

## 2. Non-goals (this project)

Deferred to **project 2 (training flywheel)**, so this stays a shippable foundation:
- **No "motion-miss" / candidate / uncertainty capture.** This project saves the same
  **confident keepers** the gate already saves (just triggered by our motion, not
  Reolink). Catching *novel/unlisted* species (motion fired, model saw nothing) and
  uncertainty-band frames is the flywheel's job.
- **No tombstone/soft-delete, no dataset export, no retrain wiring.** Those are project 2/3.
- **No timer background frames.**
- **No new /admin or gallery UI.** (One tiny, forward-looking exception: a `source_kind`
  provenance column so continuous captures are distinguishable — see §4.6.)
- **No new runtime dependency.** Uses cv2/numpy (already the `detect` extra) + the
  existing go2rtc.

## 3. Architecture

A **second producer thread per camera** runs *alongside* the existing Reolink source and
feeds the **same shared queue**, so the single consumer (`_handle_event`), `Deduper`,
gate, and `save_capture` are unchanged in their gate/save/dedupe logic. The only consumer
touch is a minimal burst-source selection keyed on `event.kind` (§4.4).

```
Reolink source ─┐                         ┌─ (unchanged) _handle_event:
                ├─> shared queue.Queue ──▶ │   dedupe → burst → YOLO → gate → save
continuous ─────┘   (one consumer thread)  └─ burst source picked by event.kind
motion source
```

New/changed units:
1. `src/wildlife/motion.py` — pure `MotionDetector` (per-frame motion decision).
2. `src/wildlife/events/continuous_motion.py` — `ContinuousMotionEventSource` (temporal
   orchestration + go2rtc read).
3. `src/wildlife/events/base.py` — register the new source kind.
4. `src/wildlife/worker.py` — dual-producer + composite source-registry key (bug fix) +
   burst routing.
5. `src/wildlife/capture.py` — `grab_burst` gains an optional explicit-URL override.
6. `src/wildlife/config.py` — `ContinuousConfig` + `CameraConfig.motion_mask`.
7. `src/wildlife/store.py` — `source_kind` provenance column (forward hook).

## 4. Components

### 4.1 `MotionDetector` (`motion.py`) — pure per-frame motion decision

cv2/numpy only (mirrors `capture.py`'s hardware confinement). Holds the MOG2 state; each
`update(frame_bgr)` returns whether *this frame* shows motion and whether the scene
changed. Temporal logic (rising edge, refractory, warmup) lives in the event source.

- Constructor: `MotionDetector(downscale_width: int, min_area_frac: float,
  algorithm: str = "mog2", mask_polys: list[list[tuple[float, float]]] | None = None,
  scene_change_thresh: float = 40.0, history: int = 500, var_threshold: int = 16)`.
- `update(frame) -> MotionResult` where `MotionResult(motion: bool, scene_change: bool,
  area_frac: float)`. Pipeline: grayscale → downscale to `downscale_width` → (rasterize
  the ignore mask once at that size and AND it out) → MOG2 `apply()` → threshold →
  morphology **open** (erode→dilate) to kill speckle → `findContours` → `motion =
  largest_contour_area_frac >= min_area_frac`. `scene_change = mean_abs_diff(gray,
  prev_gray) > scene_change_thresh` (whole-frame; catches PTZ / IR-cut flip / exposure
  jump). `algorithm="frame_diff"` swaps MOG2 for `absdiff` vs a rolling reference
  (lighter, weaker-host fallback).
- `reset()` — rebuild the subtractor (called by the source on scene-change/reconnect).

MOG2 defaults: `history=500, varThreshold=16, detectShadows=False`.

### 4.2 `ContinuousMotionEventSource` (`events/continuous_motion.py`)

Subclasses the existing `_QueueBackedEventSource` (reuse its thread/queue/`_emit`/
`_sleep`/`close`/`_signal_stop` verbatim, exactly as `ReolinkEventSource` does). Implements
only `_run()`:
- Open the go2rtc sub restream `rtsp://127.0.0.1:{rtsp_port}/{camera.id}_sub` with
  `capture.py`'s proven FFmpeg knobs: env `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`,
  `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`, `CAP_PROP_BUFFERSIZE=1`, and the open/read
  timeouts from `capture._apply_timeout`.
- Loop: **tight `cap.grab()`-drain to the newest frame**, then `retrieve()` (decode) at
  `sample_fps`; feed the frame to `MotionDetector.update`.
- Emit rules: on a rising edge (no-motion → motion) `self._emit(CameraEvent(
  camera_id=self.camera.id, event_ts=_now(), kind="motion_continuous"))`; enforce a
  producer-side **`refractory_s`** between emits (one animal → one event); suppress emits
  for **`warmup_s`** after (re)connect while MOG2 stabilizes; **`reset()`** the detector +
  apply refractory on `scene_change` or reconnect; obey **`active_hours`**.
- Robustness: a **watchdog** reopens if no frame arrives within a timeout; the worker's
  existing `_produce` backoff reopens on stream end.
- **Timestamp convention:** use naive-local `event_ts` (match `worker._now`), *not*
  Reolink's UTC, so `store` ISO strings stay consistent.

### 4.3 Register the source — `events/base.py`

The factory becomes `make_event_source(kind, camera, config=None)` (add an optional
`config` param — the full validated `Config`; **default `None` keeps existing
callers/tests working**, and reolink/onvif ignore it). Register `kind ==
"continuous_motion" -> ContinuousMotionEventSource(camera, config)` (lazy import, matching
the reolink/onvif pattern). The continuous source pulls everything it needs from that
`Config`: its knobs from `config.continuous`, the restream port from
`config.livestream.rtsp_listen`, and the ignore polygons from `camera.motion_mask` — and
constructs its `MotionDetector(downscale_width=…, min_area_frac=…, algorithm=…,
mask_polys=camera.motion_mask)`. `worker._produce` passes `self._config` into
`make_event_source`.

### 4.4 Worker: dual-producer + composite key + burst routing — `worker.py`

- **Parametrize the producer:** `_produce(self, camera)` → `_produce(self, camera, kind)`
  (it already builds-source/streams/enqueues/backs-off generically; pass `kind` instead
  of reading `self._config.event_source`).
- **Fix a latent bug (found in analysis):** `_produce` stores `self._sources[camera.id] =
  source`. Two producers per camera with the same key **clobber** each other, so
  `_teardown` closes only one and leaks the other's thread/RTSP session. **Key the
  registry on `f"{camera.id}:{kind}"`** everywhere (`_produce`, `_teardown`).
- **Start the second producer:** in `_start_producers`, after the primary
  `self._config.event_source` thread, if `cfg.continuous.enabled` start
  `Thread(target=self._produce, args=(camera, "continuous_motion"),
  name=f"motion-{camera.id}", daemon=True)` per camera. Both push the one shared queue.
- **Burst routing (the one consumer touch):** in `_handle_event`, when
  `event.kind == "motion_continuous"`, grab the burst **through go2rtc** rather than a
  direct Reolink session — build `rtsp://127.0.0.1:{rtsp_port}/{camera.id}_{cap.stream}`
  and pass it to `grab_burst(..., rtsp_url=<that>)`. Reuses the existing `capture.stream`
  config (default `main`, for real classification of small/distant/nocturnal targets).
  Reolink events keep their existing direct path (`rtsp_url=None`). `rtsp_port` comes from
  `livestream.rtsp_listen` (default `":8554"`).

### 4.5 `capture.grab_burst` — explicit-URL override

Add `grab_burst(camera, n, interval_ms, stream, timeout_s, rtsp_url: str | None = None)`:
when `rtsp_url` is given, use it instead of `_select_url(camera, stream)`; otherwise
behavior is byte-for-byte unchanged (existing Reolink path). This is what lets continuous
bursts flow through go2rtc without a second direct camera session.

### 4.6 `source_kind` provenance column — `store.py` (forward hook)

Add one column `source_kind TEXT NOT NULL DEFAULT 'reolink'` to `captures` via the
existing idempotent `_COLUMN_ADDITIONS`/`_migrate` path, added to `_COLUMNS`.
`save_capture` gains `source_kind: str = "reolink"`; `_handle_event` passes
`"continuous" if event.kind == "motion_continuous" else "reolink"`. This is the *only*
schema change here — it makes continuous captures distinguishable for tuning now and is
the seam the flywheel (project 2) builds on. No gallery/admin surfacing yet (both still
show all keepers).

## 5. Config

New `ContinuousConfig` (pydantic; **inert when `enabled: false`**), added to `Config`;
plus optional `CameraConfig.motion_mask`. The worker passes `config.continuous` (and the
camera's `motion_mask` + `livestream.rtsp_listen` port) into the source at construction.

```yaml
continuous: # optional always-on motion-gated detection (your model becomes the gate)
  enabled: false
  sample_fps: 4            # frames/sec sampled from the sub restream for motion (not full fps)
  downscale_width: 480     # px width the motion detector runs at (motion computed downscaled)
  min_area_frac: 0.003     # largest motion contour must be >= this fraction of the downscaled frame
  refractory_s: 8          # producer-side min seconds between motion emits per camera (one animal -> one event)
  warmup_s: 10             # suppress emits this long after (re)connect while MOG2 stabilizes
  algorithm: "mog2"        # mog2 | frame_diff
  active_hours: ""         # optional "HH:MM-HH:MM" local window; empty = 24/7

# per camera:
cameras:
  - id: north_field
    # ...
    motion_mask:           # optional ignore-motion polygons (normalized 0..1 coords) — roads/canopy/flags/water
      - [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]
```

Validation: `sample_fps` ≥ 1; `downscale_width` ≥ 64; `min_area_frac` ∈ (0, 1);
`refractory_s`/`warmup_s` ≥ 0; `algorithm` ∈ {mog2, frame_diff}; `active_hours` empty or
`HH:MM-HH:MM`; `motion_mask` polygons are lists of ≥3 `[x, y]` points with x,y ∈ [0, 1].

**Throttle note:** with two producers on one queue, `resource_guard.max_burst_per_minute`
(global) likely needs raising; per-camera `dedupe.cooldown_s` and
`resource_guard.detect_every_nth_event` now gate continuous for free. Defaults documented.

## 6. Data flow (end to end)

Camera → go2rtc holds one sub upstream and fans it out → `ContinuousMotionEventSource`
reads `rtsp://127.0.0.1:8554/<id>_sub`, samples at `sample_fps`, `MotionDetector.update`
each frame → on rising edge (past warmup, past refractory, motion area ≥ `min_area_frac`,
outside the mask) emits `CameraEvent(kind="motion_continuous")` onto the shared queue →
the unchanged consumer `_handle_event`: `Deduper` cooldown/burst check → `grab_burst`
**through go2rtc main** → per-frame YOLO → gate (`animal_classes` + confidence + box-area)
→ `save_capture(..., source_kind="continuous")` → gallery/notifications as today. Reolink
events flow exactly as before (direct burst, `source_kind="reolink"`).

## 7. Design decisions (approved)

1. **Burst routing = go2rtc, not a direct Reolink session.** Avoids concurrent same-IP
   sessions (which Reolink drops — the reason `capture.py` open/grab/closes). Uses
   `capture.stream` (default main) for classification quality.
2. **Motion masks = normalized polygons in config** (`CameraConfig.motion_mask`), not PNG
   files — config-driven, no image assets to manage, shares geometry with a future zones
   feature. Effectively mandatory in real yards (roads/canopy/flags/water).
3. **Motion algorithm = MOG2** (absorbs swaying vegetation, gradual light, IR-night;
   self-re-learns; surfaces sub-YOLO-size motion), with a `frame_diff` fallback flag.

## 8. Resource budget / duty-cycle

Idle cost is cheap: one persistent sub-decode (a few % CPU) + a downscaled MOG2 loop at
`sample_fps` (low-single-digit % CPU, **no GPU**). The expensive stage (YOLO on MPS) stays
**motion-gated** on the single serialized consumer (`max_concurrent=1`), so the real risk
is **trigger volume** saturating that one consumer / MPS (worst when the co-tenant media
server contends for Metal). Lever: **cut trigger volume** — masks + `min_area_frac` +
`refractory_s` (dominant) — not throughput. `active_hours` duty-cycles by time of day.
Continuous can multiply capture rate, so watch `store` write volume + prune pressure.
(Training contention with the flywheel's retrain is a project-2/3 concern.)

## 9. Failure modes

1. **Masks unset in a busy scene:** MOG2 fires constantly → unusable. Documented as a
   required setup step; `min_area_frac` is a partial backstop.
2. **PTZ / day-night IR-cut flip / exposure jump:** whole-frame "motion." `scene_change`
   detection → `reset()` + refractory swallows it (a startup flood otherwise).
3. **MOG2 warmup:** first ~`history` frames unstable → `warmup_s` suppresses emits after
   (re)connect.
4. **go2rtc not running / sub stream absent:** the source can't open the restream → logs +
   the worker's `_produce` backoff retries; continuous requires go2rtc up (launchd
   ordering: go2rtc before worker). go2rtc opens the camera upstream on first client and
   drops on last, so the persistent reader keeps it alive (avoids session churn).
5. **Wedged/stale RTSP read:** `CAP_PROP_READ_TIMEOUT_MSEC` + a watchdog reopen; the
   grab-drain-to-newest loop avoids processing seconds-stale frames.
6. **Source-registry key collision (the fixed bug):** composite `camera.id:kind` key so
   both producers' sources are tracked and closed in teardown.
7. **`enabled: false`:** no second producer, no schema-write path exercised for continuous;
   worker behavior unchanged (regression-guarded).

## 10. Files touched

- `src/wildlife/motion.py` (new) — `MotionDetector`, `MotionResult`.
- `src/wildlife/events/continuous_motion.py` (new) — `ContinuousMotionEventSource`.
- `src/wildlife/events/base.py` — register `continuous_motion`.
- `src/wildlife/worker.py` — `_produce(camera, kind)`, composite source key, second
  producer, burst routing in `_handle_event`, `source_kind` pass-through.
- `src/wildlife/capture.py` — `grab_burst(..., rtsp_url=None)`.
- `src/wildlife/config.py` — `ContinuousConfig`, `CameraConfig.motion_mask`, wire into
  `Config`, `__all__`.
- `src/wildlife/store.py` — `source_kind` column + `save_capture` param.
- `config.example.yaml` — `continuous` block + a `motion_mask` example.
- `README.md` — a "Continuous detection" section (setup, masks, tuning, go2rtc/launchd).
- Tests: `tests/test_motion.py`, `tests/test_continuous_source.py`,
  `tests/test_worker_continuous.py`, config + store tests (extend existing).

## 11. Testing

- **`MotionDetector`** (guard `pytest.importorskip("cv2")` — keeps the suite hardware-free
  when cv2 is absent): synthetic numpy frames — a moving white blob after a static warm-in
  → `motion=True`; small random speckle / "swaying" noise → `motion=False` (area gate +
  morphology); a blob **inside** a mask polygon → `motion=False`; a whole-frame brightness
  jump → `scene_change=True`; `frame_diff` variant. Deterministic, no camera.
- **`ContinuousMotionEventSource`**: with a fake frame source + a fake `MotionDetector`,
  assert rising-edge emit, `refractory_s` suppression, `warmup_s` suppression,
  `active_hours` gating, and `reset()` on scene-change. (Guard `importorskip` as needed.)
- **`worker` dual-producer**: `_produce(camera, kind)` composite key; second producer
  starts only when `continuous.enabled`; teardown closes **both** sources (composite key);
  `_handle_event` routes a `motion_continuous` event's burst to the go2rtc URL (assert via
  a fake `grab_burst` capturing `rtsp_url`) and Reolink events to `rtsp_url=None`;
  `source_kind` is persisted correctly. (Guard `importorskip("wildlife.worker")`.)
- **`grab_burst`**: `rtsp_url` override is used when given; unchanged when `None`.
- **`ContinuousConfig`**/`motion_mask` validation; `store.source_kind` migration +
  round-trip; `enabled: false` = full no-op regression.

## 12. Build order (phases within this project)

1. `MotionDetector` (`motion.py`) + tests — the pure core, de-risks the key question.
2. `ContinuousConfig` + `CameraConfig.motion_mask` + validation + no-op guard.
3. `grab_burst` URL override + test.
4. `store.source_kind` column + `save_capture` param + migration test.
5. `ContinuousMotionEventSource` + register in `base.py` + tests.
6. Worker dual-producer + composite key + burst routing + `source_kind` pass-through +
   tests (the integration that turns it on).
7. Docs (`config.example.yaml`, README) + go2rtc/launchd ordering note.

## 13. Open items to confirm on review

1. **Burst stream for continuous** = `capture.stream` (default **main**, via go2rtc) —
   confirm main is acceptable per-event camera load, or prefer sub (cheaper, worse
   small/distant classification).
2. **`source_kind` column now** (forward hook for the flywheel) vs deferring all schema to
   project 2. Included as a cheap, high-value seam. Confirm.
3. **Default tuning** (`sample_fps=4`, `downscale_width=480`, `min_area_frac=0.003`,
   `refractory_s=8`, `warmup_s=10`) — starting points; real values come from on-camera
   tuning.
4. Motion masks are a required per-camera setup step in busy scenes — acceptable?
