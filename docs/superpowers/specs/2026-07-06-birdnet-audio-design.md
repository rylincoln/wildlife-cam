# BirdNET audio bird-ID — design

**Date:** 2026-07-06
**Status:** Draft for review
**Feature:** A second detection *modality* alongside the vision pipeline: an always-on,
CPU-side audio gate that identifies birds by song from the camera mics, using BirdNET.
Reuses go2rtc, the per-camera producer-thread pattern, and the `source_kind` seam from
continuous detection. **Scope = project 1 ("engine + gallery visibility"):** detect →
persist → view/hear in the existing gallery. Notifications and a per-camera opt-out are
deferred.

## 1. Goal

Detect and identify birds by sound. A per-camera analyzer reads the camera's audio off
the go2rtc restream, runs BirdNET on rolling 3-second windows, and — after geo/confidence/
repeat-confirmation gating — saves each confirmed detection (species + a **spectrogram**
image + a short **audio clip**) into the existing capture store, tagged
`source_kind='audio'`. Detections appear in the existing gallery grid as spectrogram
thumbnails; the detail view is a **playable spectrogram** — the spectrogram with a
playhead that sweeps in sync with an HTML5 audio player. BirdNET runs CPU-side, so it
never contends with YOLO on the MPS GPU.

## 2. Non-goals (this project)

- **No notifications.** Bird-species ntfy alerts reuse the notifier on the (unmerged)
  `notifications` branch; that integration is a fast-follow once that branch lands.
- **No per-camera audio opt-out** (a single `audio` config applies to all cameras). A
  per-camera flag is a later refinement.
- **No YOLO/vision changes.** The vision pipeline is untouched; audio saves directly and
  does **not** flow through the YOLO consumer queue.
- **No dataset export / retrain / dedicated review UI** (project-2/flywheel territory).
- **No pre/post-roll on clips** — v1 saves the 3-second confirmed window; longer clips are
  a later nicety.
- **No admin UI for audio config** — configured in `config.yaml` like the rest.

## 3. Architecture

BirdNET *is* the model (there is no YOLO stage), so an audio detection is analyzed and
**saved directly** by its own per-camera thread — unlike the motion producer, which
enqueues for the shared YOLO consumer. The worker gains a set of per-camera audio analyzer
threads (started only when `audio.enabled`) that share one loaded BirdNET model and the one
thread-safe `Store`.

```
                          ┌───────────────── worker process ─────────────────┐
Reolink → go2rtc restream │  (unchanged) Reolink/continuous producers → YOLO  │
  <id>_sub (audio track)  │                                                   │
        │                 │  per camera:  AudioDetectionSource (new thread)   │
        └───ffmpeg PCM────▶│    ffmpeg→48k mono PCM (3s win / 1.5s hop)        │
                          │    → AudioAnalyzer.analyze (BirdNET, shared+lock) │
                          │    → RepeatConfirmer (pure) → save_audio_capture ─┼─▶ Store (shared)
                          └───────────────────────────────────────────────────┘
                                     (all CPU-side; YOLO stays on MPS)
```

New/changed units:
1. `src/wildlife/audio.py` (new, hardware-confined) — `AudioAnalyzer` (BirdNET load +
   `analyze`) and `render_spectrogram`.
2. `src/wildlife/audio_gate.py` (new, pure) — `RepeatConfirmer` (repeat-confirmation +
   per-species cooldown state machine).
3. `src/wildlife/_colormap.py` (new, tiny) — a public-domain magma LUT for spectrograms.
4. `src/wildlife/events/audio_detection.py` (new) — the per-camera analyzer thread
   (ffmpeg reader + windowing + orchestration).
5. `src/wildlife/store.py` — `audio_path` column + `save_audio_capture`.
6. `src/wildlife/config.py` — `AudioConfig`.
7. `src/wildlife/worker.py` — load BirdNET once; start/stop the audio threads.
8. `src/wildlife/gallery/app.py` + templates — `/audio/<id>` endpoint, source-kind-aware
   detail rendering (spectrogram + playhead player), a `source_kind` filter/badge.
9. `pyproject.toml` — an `[audio]` optional extra.

## 4. Components

### 4.1 `AudioAnalyzer` (`audio.py`) — hardware-confined BirdNET wrapper

`birdnet`/`soundfile`/`numpy` imported lazily (mirrors `capture.py`/`motion.py`), so
`config`/`store`/`models` stay import-light and the pure logic tests run without BirdNET.

- **Load once** (in the worker, shared across cameras):
  `acoustic = birdnet.load("acoustic", "2.4", "tf")` and `geo = birdnet.load("geo", "2.4",
  "tf")`. (`"tf"` backend = TFLite/LiteRT interpreter, CPU.) On first load, `birdnet`
  downloads weights from Kaggle Hub → **first run needs network + a writable cache.**
- **Geo shortlist:** at startup compute
  `geo.predict(latitude, longitude, week=<week-of-year>)` → a species→probability mapping;
  keep the species names as the `custom_species_list`. `week` is 1–48 (BirdNET's 4-weeks-
  per-month convention); compute from the current local date. (A periodic re-compute as the
  season advances is a later nicety; the worker restarts often enough for v1.)
- **`analyze(pcm_float32: np.ndarray) -> list[tuple[str, float]]`** — a 3-second, 48 kHz
  mono `float32` window in `[-1, 1]`. Calls
  `acoustic.predict_arrays((pcm, 48000), top_k=..., default_confidence_threshold=<conf>,
  bandpass_fmin=<fmin>, custom_species_list=<geo species or None>)`, reads
  `result.to_structured_array()` (columns `input, start_time, end_time, species_name,
  confidence`), and returns `[(species_name, confidence), …]` above threshold. Species
  labels are `"ScientificName_CommonName"`; the common name is the display label.
  **Thread-safe:** `predict_arrays` is serialized behind an internal `threading.Lock`
  (TFLite interpreters aren't reentrant; inference is fast enough to serialize across 2–3
  cameras).
- Config → predict: `confidence_threshold → default_confidence_threshold`;
  `bandpass_fmin → bandpass_fmin` (BirdNET's **built-in** band limiting — the clean wind
  knob, part of its own preprocessing, so no risky external filter);
  `use_geo_filter → custom_species_list` (the geo species, or `None` to disable).

### 4.2 `render_spectrogram(pcm_float32) -> bytes` (`audio.py`) — numpy + Pillow only

Hand-rolled STFT (no matplotlib, no scipy — `scipy.signal.stft` is legacy):
`n_fft=1024`, `hop=256`, Hann window → `np.fft.rfft` over
`sliding_window_view(x, n_fft)[::hop]` → magnitude → `20·log10(mag+1e-6)` with an 80 dB
floor → normalize to `uint8` → **crop to ≤12 kHz** (bird band; ~256 bins) → transpose +
`flipud` (low freq at bottom) → `Image.resize(..., LANCZOS)` to a fixed height → apply the
magma LUT by fancy-indexing (`MAGMA_LUT[img]`) → `Image.fromarray(rgb, "RGB")` → PNG bytes.
Rendered **at detection time from the same in-memory window** that becomes the clip, so the
spectrogram spans exactly the clip's duration (the playhead maps linearly). Two sizes are
produced (full + thumbnail), matching `store.py`'s existing image/thumb convention.

### 4.3 `RepeatConfirmer` (`audio_gate.py`) — pure, unit-testable

No hardware; injected clock (mirrors how the motion temporal logic was pure). Buffers recent
per-species detections and decides when a species is *confirmed* and not in cooldown.

- `RepeatConfirmer(min_confirmations: int, confirm_window_s: float, cooldown_s: float)`.
- `offer(species: str, confidence: float, now: datetime) -> bool` — records the hit;
  returns `True` (fire a save) when this species has ≥ `min_confirmations` hits within the
  trailing `confirm_window_s` **and** is past `cooldown_s` since its last confirmation;
  arms cooldown on a `True`. Species tracked independently; old hits outside the window are
  evicted. Chaotic/broadband noise (wind) rarely reproduces the *same* species, so this is
  the dominant false-positive control.

### 4.4 `AudioDetectionSource` (`events/audio_detection.py`) — per-camera thread

Manages its own daemon thread with `start()` / `stop()` (a stop `threading.Event` + an
interruptible sleep), mirroring the notifier's thread lifecycle — **not** the
`_QueueBackedEventSource` sources, which start lazily via `stream()` and are built around
emitting to a consumer queue this source doesn't use (it saves directly). `cv2` is not
needed; `subprocess`/`numpy` and the lazy `AudioAnalyzer` are used inside the run loop.

- Open ffmpeg on the go2rtc restream (stream from `audio.stream`, default `sub`):
  `ffmpeg -nostdin -loglevel error -rtsp_transport tcp -i
  rtsp://127.0.0.1:{rtsp_port}/{camera.id}_{audio.stream} -vn -map 0:a:0 -ac 1 -ar 48000
  -f s16le -`. Read raw PCM from the subprocess pipe on a dedicated loop, **looping `read()`
  to assemble each hop** (pipe reads are short), maintaining a ring buffer; emit a 3 s
  window (`144000` samples) every **1.5 s hop** (50 % overlap, birdnet-go cadence). Drain
  stderr on a side thread so it can't deadlock the pipe.
- Per window: `float32` scale → `AudioAnalyzer.analyze` → for each `(species, conf)`,
  `RepeatConfirmer.offer(...)`; on `True`, render spectrogram + thumbnail, **encode the clip
  from the same in-memory PCM** (`ffmpeg -f s16le -ar 48000 -ac 1 -i - -c:a aac -b:a 96k
  -movflags +faststart clip.m4a`), and `store.save_audio_capture(...)`.
- Robustness: an empty pipe read = stream dropped → close and **reopen ffmpeg with backoff**
  (reset backoff only after audio actually flows, matching the motion reader's fix). Obey
  `active_hours`. Naive-local `datetime.now()` timestamps.

### 4.5 `store.save_audio_capture` + `audio_path` column (`store.py`)

- Add nullable `audio_path TEXT` to `captures` via the idempotent
  `_COLUMN_ADDITIONS`/`_migrate` path (same pattern as `source_kind`); add to `_COLUMNS`.
- `save_audio_capture(*, camera_id, event_ts, capture_ts, species: str, confidence: float,
  spectrogram_full: bytes, spectrogram_thumb: bytes, clip_bytes: bytes | None,
  source_kind: str = "audio") -> int` — writes the full spectrogram to `image_path`, the
  thumbnail spectrogram to `thumb_path`, the `.m4a` to `audio_path` (or NULL if encoding
  failed), `label`=species, `confidence`, box columns NULL, reusing the dated
  `YYYY/MM/DD` directory + filename conventions. (A sibling to `save_capture`, not a
  reshaping of it — audio has no BGR frame or `Detection`.)

### 4.6 Gallery (`gallery/app.py` + templates)

- **`/audio/<int:capture_id>`** — serves the clip via `send_file(path, mimetype="audio/mp4",
  conditional=True, download_name=...)` (range requests so `<audio>` can seek); 404 if the
  row has no `audio_path`.
- **Source-kind-aware detail:** image rows render as today; `source_kind='audio'` rows
  render the spectrogram (`/image/<id>`) inside a `position:relative` wrapper with a 1 px
  `.playhead` overlay + `<audio controls src="/audio/<id>">`. A small **vanilla-JS**,
  `requestAnimationFrame`-driven playhead (`translateX((currentTime/duration)*imgWidth)`,
  read `img.clientWidth` each frame; `timeupdate` alone is too coarse) — no external JS lib.
- **`source_kind` filter/badge** in the grid so audio and image captures can be told apart
  (the column already exists; this surfaces it). Existing label/camera/date filters work
  unchanged (bird species land in `distinct_labels`).

### 4.7 `worker.py`

- In `_setup`, when `audio.enabled`: build the shared `AudioAnalyzer` (loads both models +
  the geo shortlist) once. If the `[audio]` extra isn't installed, fail with a clear message
  (like reolink-aio's lazy-import error) and leave the rest of the worker running.
- In `_start_producers`, when `audio.enabled`: start one `AudioDetectionSource` thread per
  camera (`name=f"audio-{camera.id}"`), all sharing the analyzer + `Store`. Teardown stops
  them (close the ffmpeg subprocess + join).
- The audio path uses **its own** gate (`RepeatConfirmer` + BirdNET confidence/geo) and does
  not touch the YOLO `Deduper`/`resource_guard`.

## 5. Config

New `AudioConfig` (pydantic; **inert when `enabled: false`**), added to `Config` via
`default_factory`.

```yaml
audio: # optional BirdNET audio bird-ID (CPU-side; needs the go2rtc daemon + camera audio)
  enabled: false
  stream: sub              # sub | main — which go2rtc restream carries the mic
  latitude: 37.2           # for geo/occurrence filtering (rounded placeholder in the example)
  longitude: -107.5
  use_geo_filter: true     # if false, lat/lon optional and no species shortlist is applied
  confidence_threshold: 0.25   # BirdNET score gate; raise to cut wind/noise
  bandpass_fmin: 0             # Hz; raise (e.g. 300) to band-limit low-frequency wind
  min_confirmations: 2         # same species N times within confirm_window_s before saving
  confirm_window_s: 15
  cooldown_s: 30               # per-species suppression after a save
  active_hours: ""             # optional "HH:MM-HH:MM" local; empty = 24/7
```

Validation: `stream ∈ {sub, main}`; `confidence_threshold ∈ [0,1]`; `bandpass_fmin ≥ 0`;
`min_confirmations ≥ 1`; `confirm_window_s`/`cooldown_s ≥ 0`; `active_hours` empty or
`HH:MM-HH:MM` (reuse the `ContinuousConfig` validator style); when `use_geo_filter` is true,
`latitude`/`longitude` must be set (`latitude ∈ [-90,90]`, `longitude ∈ [-180,180]`). The
prod deployment uses the real coordinates **37.228274, -107.519089** in the gitignored
`config.yaml`; `config.example.yaml` ships a rounded placeholder (public repo).

## 6. Data flow (end to end)

Camera mic → RTSP → go2rtc `<id>_{stream}` (audio passthrough) → ffmpeg decodes to 48 kHz
mono s16le PCM on a pipe → per-camera reader frames it into 3 s windows (1.5 s hop) →
`float32` → `AudioAnalyzer.analyze` (BirdNET `predict_arrays` with geo shortlist +
confidence + `bandpass_fmin`) → `RepeatConfirmer.offer` (≥ N same-species hits within the
window, past cooldown) → on confirm: `render_spectrogram` (numpy+Pillow) + AAC/`.m4a` encode
from the same PCM → `store.save_audio_capture(source_kind='audio', …)` → row appears in the
gallery grid (spectrogram thumbnail); the detail view is the spectrogram + synced-playhead
player served from `/audio/<id>`. Vision captures are entirely unaffected.

## 7. Design decisions (approved + research-informed)

1. **Persistence = reuse `captures` with `source_kind='audio'`** (approved Option 1): the
   spectrogram is the row's image/thumbnail, `label`=species, box columns NULL, plus a
   nullable `audio_path` for the clip. One unified timeline; gallery + future notifications
   reuse for near-free. `save_audio_capture` is a sibling of `save_capture`.
2. **Playable spectrogram** (approved): a static spectrogram PNG + HTML5 `<audio>` + a
   vanilla-JS `requestAnimationFrame` playhead. Rendered at detection time from the same
   in-memory window as the clip, so the playhead maps exactly.
3. **`birdnet` PyPI library, `"tf"` (TFLite) backend, CPU** — the maintained, embeddable
   API; `predict_arrays((pcm, 48000))` takes the in-memory window (no temp files). Verified
   against v0.2.16 source. See §8 for the footprint caveat.
4. **False-positive controls, layered & tunable:** BirdNET confidence threshold + geo
   occurrence shortlist (`custom_species_list`) + **repeat-confirmation** (dominant) +
   per-species cooldown; plus BirdNET's built-in `bandpass_fmin` for wind — all knobs, tuned
   on real recordings.
5. **Clip format = AAC in `.m4a`** — ~38 KB per 3 s and the smallest format that plays in
   every named target (desktop Chrome/Safari, iOS Safari) with no version cliff (Opus is
   smaller but unsupported in older Safari and never in Safari-MP4). Encoded from the
   in-memory PCM — **no second RTSP connection.**
6. **Spectrogram = hand-rolled numpy rFFT + Pillow + a CC0 magma LUT** — reuses the store's
   existing numpy/Pillow pattern and adds **no** matplotlib/scipy dependency.
7. **Audio source stream is configurable (`audio.stream`, default `sub`)** — some Reolink
   models only carry audio on the main stream; runtime `ffprobe` confirms which.

## 8. Dependencies & footprint (the one thing to confirm)

The `[audio]` extra pulls `birdnet`, which has **full `tensorflow` (≈1 GB installed) as a
core, non-optional dependency** — not the lean `tflite-runtime` originally assumed — plus
`scipy`, `pandas`, `pyarrow`, `soundfile` (needs system `libsndfile`), and `kagglehub`
(downloads model weights on first load → network + writable cache required once). It is
**CPU-only** (backend `"tf"` runs the TFLite interpreter — no MPS/GPU contention with YOLO),
but it is **not lean**. On the 8 GB Mac mini already running torch (YOLO) + go2rtc + gallery,
TensorFlow's resident memory (~hundreds of MB) coexisting with torch is a real consideration
for 2–3 cameras. Mitigations if memory proves tight: run the audio analyzer in a **separate
process** (isolating TF from torch; the shared `Store` is already multi-process-safe via WAL
+ busy_timeout), or fall back to the **raw BirdNET TFLite model + `ai-edge-litert`** with
hand-rolled pre/post-processing (leaner, but significantly more custom code). v1 keeps the
audio threads in the worker process and documents the footprint; the separate-process split
is a clean escape hatch if needed. **Python 3.11–3.13** required (the app targets 3.12 — OK).
**Licenses:** library code MIT; **model weights CC BY-NC-SA 4.0 (non-commercial)** — fine for
a personal backyard cam; a flag only if this is ever monetized.

## 9. Resource / duty-cycle

Idle cost per camera: one persistent ffmpeg audio decode (cheap) + a BirdNET inference every
1.5 s (fast on CPU, serialized behind the shared lock). 2–3 cameras is comfortable on CPU;
the real cost is TensorFlow's memory (§8). Storage: each confirmed detection adds a small
spectrogram PNG (+ thumbnail) and a ~38 KB `.m4a` — pruned by the existing retention (audio
rows are `captures` rows). `min_confirmations` + `cooldown_s` + `active_hours` keep detection
volume and disk sane.

## 10. Failure modes

1. **No audio on the chosen stream** (mic off / Reolink carries audio only on `main`):
   ffmpeg finds no audio track → log + backoff-retry; the fix is `audio.stream: main` (or a
   camera-config change). Runtime `ffprobe rtsp://127.0.0.1:8554/<id>_sub` confirms.
2. **go2rtc down / stream drop:** empty pipe read → reopen ffmpeg with backoff (reset only
   after audio flows). Continuous requires the go2rtc daemon (same as motion).
3. **`[audio]` extra not installed but `audio.enabled`:** clear startup error; audio
   disabled, rest of the worker unaffected.
4. **First-run weight download fails (no network / Kaggle unreachable):** model load fails →
   logged; audio disabled for that run; retried next start. Document the one-time network
   requirement.
5. **Spectrogram render fails:** skip that detection (the spectrogram is the row's required
   image). **Clip encode fails:** still save the row with `audio_path` NULL (spectrogram
   viewable, no playback). Graceful degradation.
6. **Wind / broadband noise:** confidence + geo + `bandpass_fmin` + **repeat-confirmation**
   (chaotic noise won't reproduce the same species) + cooldown; visually obvious as a
   low-frequency smear on the spectrogram for easy human verification.
7. **Low-rate camera audio (G.711 @ 8 kHz):** BirdNET upsamples fine, but 8 kHz clips
   high-frequency calls — prefer selecting AAC/16000 on the Reolink. Documented, not a code
   fix.
8. **`enabled: false`:** no audio threads, no model load, no schema-write path exercised;
   worker unchanged (regression-guarded).

## 11. Files touched

- `src/wildlife/audio.py` (new) — `AudioAnalyzer`, `render_spectrogram`.
- `src/wildlife/audio_gate.py` (new) — `RepeatConfirmer`.
- `src/wildlife/_colormap.py` (new) — magma LUT.
- `src/wildlife/events/audio_detection.py` (new) — `AudioDetectionSource`.
- `src/wildlife/store.py` — `audio_path` column + `save_audio_capture`.
- `src/wildlife/config.py` — `AudioConfig` (+ `__all__`, wire into `Config`).
- `src/wildlife/worker.py` — load BirdNET once; start/stop audio threads.
- `src/wildlife/gallery/app.py` + `templates/index.html` (+ a small static JS/CSS) —
  `/audio/<id>`, source-kind-aware detail, `source_kind` filter/badge.
- `pyproject.toml` — `[audio]` extra.
- `config.example.yaml`, `README.md` — `audio` block + a "Audio bird-ID" section.
- Tests: `tests/test_audio_gate.py`, `tests/test_audio_config.py`, store tests (extend),
  `tests/test_audio_analyzer.py` (importorskip), gallery audio-route test.

## 12. Testing

- **`RepeatConfirmer`** (pure, injected clock): confirms only at `min_confirmations` within
  `confirm_window_s`; cooldown suppresses repeats; species independent; window eviction. No
  BirdNET.
- **`AudioConfig`** validation: inert default; `stream` enum; lat/lon required + ranged when
  `use_geo_filter`; numeric ranges; `active_hours` format; a `Config` with no `audio:` block
  still validates (inertness regression).
- **`store.save_audio_capture` + `audio_path` migration**: round-trip (source_kind='audio',
  spectrogram as image/thumb, clip path, boxes NULL) + legacy-DB migration adds `audio_path`.
- **`render_spectrogram`** (guard `importorskip("numpy")`/Pillow; no BirdNET needed): a
  synthetic tone PCM → a PNG of the expected fixed dimensions; a pure-silence window → a
  valid (non-crashing) image.
- **`AudioAnalyzer`** (guard `importorskip("birdnet")` — skipped in CI): with the real or a
  monkeypatched model, `analyze` returns `[(species, confidence)]` of the right shape and
  applies the threshold; geo-shortlist plumbing is exercised with a fake model.
- **Gallery `/audio/<id>`**: serves the clip with `audio/mp4` + range support; 404 when
  `audio_path` is NULL; an audio row renders the player markup (assert template branch).
- **`enabled: false`** full no-op regression (mirrors the continuous-detection guard).

## 13. Build order (phases within this project)

1. `RepeatConfirmer` (`audio_gate.py`) + tests — the pure gate, de-risks the false-positive
   logic first.
2. `AudioConfig` + validation + no-op guard.
3. `store.audio_path` column + `save_audio_capture` + migration test.
4. `_colormap.py` + `render_spectrogram` (`audio.py`) + tests (numpy/Pillow only).
5. `AudioAnalyzer` (BirdNET load + `analyze`) in `audio.py` + `[audio]` extra + importorskip
   tests.
6. `AudioDetectionSource` (ffmpeg reader + windowing + orchestration) + tests (fake analyzer/
   confirmer; the ffmpeg/pipe path is exercised lightly, hardware paths untested like the
   event sources).
7. Worker wiring (load once, start/stop threads) + tests.
8. Gallery `/audio/<id>` + spectrogram-player detail + playhead JS + `source_kind` filter.
9. Docs (`config.example.yaml`, README) + `ffprobe`/first-run-download/launchd notes.

## 14. Open items to confirm on review

1. **Footprint (§8):** the `birdnet` library brings **full TensorFlow (~1 GB)** on the 8 GB
   mini (still CPU, no MPS contention). Accept for v1 (documented, with the separate-process
   / raw-tflite escape hatch), or would you rather start with the lean raw-tflite path now?
2. **Clip length:** v1 saves the 3-second confirmed window as the `.m4a`. OK, or want a few
   seconds of pre/post-roll (needs a slightly longer rolling buffer)?
3. **Geo filter default on** (needs your lat/lon, already provided) — good as the default?
4. **Weight download:** first run downloads BirdNET weights from Kaggle Hub (needs network +
   a writable cache once, on the prod mini). Acceptable, or pre-stage the weights?
