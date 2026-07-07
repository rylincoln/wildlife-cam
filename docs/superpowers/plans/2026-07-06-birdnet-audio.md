# BirdNET Audio Bird-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on, CPU-side audio bird-ID modality: per-camera analyzer threads run BirdNET on the go2rtc audio, and confirmed detections are saved (spectrogram + clip) into the existing capture store as `source_kind='audio'` and shown as a playable spectrogram in the gallery.

**Architecture:** BirdNET *is* the model (no YOLO stage), so an audio detection is analyzed and saved **directly** by its own per-camera thread — it does not flow through the YOLO consumer queue. The worker loads BirdNET once and starts one `AudioDetectionSource` thread per camera when `audio.enabled`; each ffmpeg-decodes the go2rtc restream audio to 48 kHz mono PCM, windows it (3 s / 1.5 s hop), runs BirdNET (geo shortlist + confidence + bandpass), applies repeat-confirmation + cooldown, then renders a spectrogram + encodes an AAC clip + saves. The gallery serves the clip from a new `/audio/<id>` and renders audio rows as a spectrogram with a synced playhead.

**Tech Stack:** Python 3.12, `birdnet` (TFLite/CPU) via an optional `[audio]` extra, ffmpeg (subprocess PCM pipe + AAC encode), numpy + Pillow (spectrogram, no matplotlib/scipy), pydantic v2 (config), SQLite (store), Flask + vanilla JS (gallery). Pytest with `importorskip` guards keep the suite hardware-free.

## Global Constraints

- **No new *core* runtime dependency.** BirdNET lives behind the optional `[audio]` extra; `config`/`store`/`models` stay import-light (pydantic/numpy/Pillow/stdlib only). The `birdnet` import is **lazy** (inside `AudioAnalyzer`), so `audio.py`'s pure `render_spectrogram` and everything else import without it.
- **`[audio]` extra pulls full TensorFlow** (a core dep of `birdnet`) — CPU-only, no MPS/GPU. This is expected and documented; do not try to make it lean.
- **Hardware-free test suite.** Guard any test that imports `birdnet` with `pytest.importorskip("birdnet")` (skipped in CI/dev — birdnet is NOT installed in the dev venv). `render_spectrogram`, `RepeatConfirmer`, `AudioConfig`, and `store` tests must run with numpy/Pillow/stdlib only.
- **Inert when `audio.enabled: false`:** no analyzer threads, no BirdNET load, no `save_audio_capture` path exercised; worker/gallery behavior unchanged. Regression-guarded.
- **Persistence = reuse `captures`:** `source_kind='audio'`, spectrogram saved as JPEG in `image_path`/`thumb_path` (so `/image`·`/thumb` serve it unchanged), `label`=bird **common name**, `confidence`=BirdNET score, box columns NULL, clip `.m4a` path in a new nullable `audio_path` column.
- **Audio input:** 48 kHz **mono** `float32` in `[-1, 1]`; window = 3 s = **144000 samples**; hop = 1.5 s = **72000 samples** (50 % overlap).
- **Clip format:** AAC in `.m4a`, encoded from the in-memory PCM (no second RTSP connection); served with Content-Type `audio/mp4` and range support.
- **Timestamps** naive-local via `datetime.now()` (matching the worker), never UTC.
- **Two verify-on-deploy points** (cannot run birdnet in dev — write against the source-verified v0.2.16 API, code defensively, flag them): (a) the exact `GeoPredictionResult` accessor for the species shortlist; (b) that `predict_arrays((pcm, 48000))` + `to_structured_array()` behave as documented. If the geo shortlist can't be built, fall back to `None` (no filter) and log.
- **Prod coordinates** live in the gitignored `config.yaml` (37.228274, -107.519089); `config.example.yaml` ships a rounded placeholder (public repo).
- **Conventions:** `from __future__ import annotations`; Google-style docstrings; ruff-clean (`.venv/bin/ruff check .`); tests via `.venv/bin/python -m pytest`.

---

## File Structure

- **Create** `src/wildlife/audio_gate.py` — `RepeatConfirmer` (pure repeat-confirmation + cooldown state machine).
- **Create** `src/wildlife/_colormap.py` — `MAGMA_LUT` (256×3 uint8, built from public-domain anchor stops).
- **Create** `src/wildlife/audio.py` — `render_spectrogram` (numpy+Pillow) and `AudioAnalyzer` (lazy `birdnet`).
- **Create** `src/wildlife/events/audio_detection.py` — `AudioDetectionSource` (own-thread `start()`/`stop()`; ffmpeg reader + windowing + orchestration).
- **Modify** `src/wildlife/config.py` — `AudioConfig` + wire into `Config`.
- **Modify** `src/wildlife/store.py` — `audio_path` column + `save_audio_capture` (+ extract a shared JPEG-writer).
- **Modify** `src/wildlife/worker.py` — load BirdNET once; start/stop audio threads.
- **Modify** `src/wildlife/gallery/app.py` — `/audio/<id>`, `source_kind` in `_serialize`, `source_kind` filter.
- **Modify** `src/wildlife/gallery/templates/index.html` — audio-row badge + spectrogram-player lightbox + playhead JS.
- **Modify** `pyproject.toml` — `[audio]` extra.
- **Modify** `config.example.yaml`, `README.md` — `audio` block + "Audio bird-ID" section.
- **Create tests** `tests/test_audio_gate.py`, `tests/test_audio_config.py`, `tests/test_spectrogram.py`, `tests/test_audio_analyzer.py`, `tests/test_audio_detection_source.py`, `tests/test_worker_audio.py`, `tests/test_gallery_audio.py`; **extend** `tests/test_store.py`.

**Dependency order:** 1 `RepeatConfirmer` → 2 `AudioConfig` → 3 store → 4 colormap+spectrogram → 5 `AudioAnalyzer`+extra → 6 `AudioDetectionSource` → 7 worker → 8 gallery backend → 9 gallery frontend → 10 docs.

---

### Task 1: `RepeatConfirmer` — pure repeat-confirmation + cooldown

**Files:**
- Create: `src/wildlife/audio_gate.py`
- Test: `tests/test_audio_gate.py`

**Interfaces:**
- Consumes: nothing (leaf, stdlib only).
- Produces: `RepeatConfirmer(min_confirmations: int, confirm_window_s: float, cooldown_s: float)` with `offer(species: str, confidence: float, now: datetime) -> bool` (True = fire a save; arms cooldown).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_gate.py`:

```python
"""Tests for the pure RepeatConfirmer (no hardware)."""

from __future__ import annotations

from datetime import datetime, timedelta

from wildlife.audio_gate import RepeatConfirmer


def _t(sec: float) -> datetime:
    return datetime(2026, 7, 6, 12, 0, 0) + timedelta(seconds=sec)


def test_fires_only_after_min_confirmations_within_window():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False   # 1st hit
    assert rc.offer("robin", 0.9, _t(2)) is True     # 2nd within window -> fire


def test_single_hit_does_not_fire():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False


def test_hits_outside_window_do_not_accumulate():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=10, cooldown_s=0)
    assert rc.offer("robin", 0.9, _t(0)) is False
    # 20s later the first hit has aged out of the 10s window -> still only 1 in-window
    assert rc.offer("robin", 0.9, _t(20)) is False
    assert rc.offer("robin", 0.9, _t(21)) is True    # now 2 within the window


def test_cooldown_suppresses_refire():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False
    assert rc.offer("robin", 0.9, _t(1)) is True     # fire, arm 30s cooldown
    assert rc.offer("robin", 0.9, _t(5)) is False    # within cooldown
    assert rc.offer("robin", 0.9, _t(10)) is False   # still cooling down
    # after cooldown, a fresh pair fires again
    assert rc.offer("robin", 0.9, _t(40)) is False
    assert rc.offer("robin", 0.9, _t(41)) is True


def test_species_tracked_independently():
    rc = RepeatConfirmer(min_confirmations=2, confirm_window_s=15, cooldown_s=30)
    assert rc.offer("robin", 0.9, _t(0)) is False
    assert rc.offer("jay", 0.9, _t(1)) is False       # different species, own count
    assert rc.offer("robin", 0.9, _t(2)) is True      # robin reaches 2
    assert rc.offer("jay", 0.9, _t(3)) is True         # jay reaches 2 independently
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wildlife.audio_gate'`.

- [ ] **Step 3: Write the implementation**

Create `src/wildlife/audio_gate.py`:

```python
"""Repeat-confirmation + per-species cooldown for audio detections.

Pure, hardware-free logic (stdlib only), injected clock — the audio analyzer's
false-positive gate. Chaotic/broadband noise (wind) rarely reproduces the *same*
species across windows, so requiring N same-species hits within a short window is
the dominant defense; a per-species cooldown then throttles a persistently-calling
bird to one saved detection per cooldown period.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

__all__ = ["RepeatConfirmer"]


class RepeatConfirmer:
    """Decide when a species is *confirmed* (fire a save) and not in cooldown.

    Parameters
    ----------
    min_confirmations:
        Number of same-species hits required within ``confirm_window_s`` to fire.
    confirm_window_s:
        Trailing window (seconds) over which hits are counted.
    cooldown_s:
        After a fire, suppress that species for this many seconds.
    """

    def __init__(
        self, min_confirmations: int, confirm_window_s: float, cooldown_s: float
    ) -> None:
        self._min = int(min_confirmations)
        self._window_s = float(confirm_window_s)
        self._cooldown_s = float(cooldown_s)
        self._hits: dict[str, deque[datetime]] = defaultdict(deque)
        self._last_fired: dict[str, datetime] = {}

    def offer(self, species: str, confidence: float, now: datetime) -> bool:
        """Record a hit for ``species``; return True when it fires a save."""
        last = self._last_fired.get(species)
        if last is not None and (now - last).total_seconds() < self._cooldown_s:
            return False  # cooling down; don't even accumulate

        hits = self._hits[species]
        hits.append(now)
        # Evict hits older than the trailing window.
        cutoff = now
        while hits and (cutoff - hits[0]).total_seconds() > self._window_s:
            hits.popleft()

        if len(hits) >= self._min:
            self._last_fired[species] = now
            hits.clear()
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio_gate.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/audio_gate.py tests/test_audio_gate.py
git add src/wildlife/audio_gate.py tests/test_audio_gate.py
git commit -m "feat(audio): add RepeatConfirmer for audio false-positive gating"
```

---

### Task 2: `AudioConfig` + wire into `Config`

**Files:**
- Modify: `src/wildlife/config.py`
- Test: `tests/test_audio_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AudioConfig(enabled=False, stream="sub", latitude=None, longitude=None, use_geo_filter=True, confidence_threshold=0.25, bandpass_fmin=0, min_confirmations=2, confirm_window_s=15.0, cooldown_s=30.0, active_hours="")`; `Config.audio: AudioConfig` (default_factory).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_config.py`:

```python
"""Validation tests for AudioConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildlife.config import AudioConfig, Config


def test_audio_defaults_are_inert():
    a = AudioConfig()
    assert a.enabled is False
    assert a.stream == "sub"
    assert a.use_geo_filter is True
    assert a.confidence_threshold == pytest.approx(0.25)
    assert a.bandpass_fmin == 0
    assert a.min_confirmations == 2
    assert a.confirm_window_s == 15.0
    assert a.cooldown_s == 30.0
    assert a.active_hours == ""


def test_audio_rejects_bad_values():
    with pytest.raises(ValidationError):
        AudioConfig(stream="both")
    with pytest.raises(ValidationError):
        AudioConfig(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        AudioConfig(bandpass_fmin=-1)
    with pytest.raises(ValidationError):
        AudioConfig(min_confirmations=0)
    with pytest.raises(ValidationError):
        AudioConfig(active_hours="25:00-06:00")


def test_geo_filter_requires_lat_lon():
    # use_geo_filter true (default) but no coords -> error
    with pytest.raises(ValidationError):
        AudioConfig(enabled=True, use_geo_filter=True)
    # valid coords ok
    a = AudioConfig(use_geo_filter=True, latitude=37.2, longitude=-107.5)
    assert a.latitude == pytest.approx(37.2)
    # geo off -> coords optional
    assert AudioConfig(use_geo_filter=False).latitude is None


def test_lat_lon_ranges():
    with pytest.raises(ValidationError):
        AudioConfig(use_geo_filter=True, latitude=91.0, longitude=0.0)
    with pytest.raises(ValidationError):
        AudioConfig(use_geo_filter=True, latitude=0.0, longitude=181.0)


def test_config_without_audio_block_still_validates():
    # Mirror the ContinuousConfig inertness regression: a Config with no `audio`
    # key must validate with audio.enabled defaulting False.
    from tests.test_continuous_config import _minimal_config_dict  # reuse helper

    cfg = Config.model_validate(_minimal_config_dict())
    assert cfg.audio.enabled is False
```

> Note: `tests/test_continuous_config.py` already defines `_minimal_config_dict()` (a full minimal config with no optional feature blocks). If importing across test modules is undesirable in this repo, inline a copy of that dict here instead.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'AudioConfig'`.

- [ ] **Step 3: Implement the config**

In `src/wildlife/config.py`, add `"AudioConfig"` to `__all__` (after `"ContinuousConfig"`).

Add the model just before `class Config` (after `ContinuousConfig`):

```python
class AudioConfig(BaseModel):
    """Optional BirdNET audio bird-ID (CPU-side; needs the go2rtc daemon + camera audio).

    A per-camera analyzer reads the camera's audio off the go2rtc restream, runs
    BirdNET on rolling 3-second windows, and saves confirmed detections. Inert when
    ``enabled`` is false.
    """

    enabled: bool = False
    stream: Literal["sub", "main"] = "sub"  # which go2rtc restream carries the mic
    latitude: float | None = None
    longitude: float | None = None
    use_geo_filter: bool = True
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    bandpass_fmin: int = Field(default=0, ge=0)  # Hz; raise to band-limit low-freq wind
    min_confirmations: int = Field(default=2, ge=1)
    confirm_window_s: float = Field(default=15.0, ge=0.0)
    cooldown_s: float = Field(default=30.0, ge=0.0)
    active_hours: str = ""  # optional "HH:MM-HH:MM" local; empty = 24/7

    @field_validator("active_hours")
    @classmethod
    def _validate_active_hours(cls, value: str) -> str:
        """Accept "" (24/7) or a strict "HH:MM-HH:MM" 24-hour window."""
        value = value.strip()
        if not value:
            return ""
        m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", value)
        if not m:
            raise ValueError('active_hours must be "" or "HH:MM-HH:MM"')
        sh, sm, eh, em = (int(g) for g in m.groups())
        for hh, mm in ((sh, sm), (eh, em)):
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError("active_hours has an out-of-range time")
        return value

    @model_validator(mode="after")
    def _geo_needs_coords(self) -> "AudioConfig":
        """When geo filtering is on, latitude/longitude are required and ranged."""
        if self.use_geo_filter:
            if self.latitude is None or self.longitude is None:
                raise ValueError(
                    "use_geo_filter=true requires latitude and longitude"
                )
        if self.latitude is not None and not (-90.0 <= self.latitude <= 90.0):
            raise ValueError("latitude must be in [-90, 90]")
        if self.longitude is not None and not (-180.0 <= self.longitude <= 180.0):
            raise ValueError("longitude must be in [-180, 180]")
        return self
```

Note: `re` is already imported at module top (added in the continuous-detection work). If it is not, add `import re` to the top-of-file imports.

Wire into `Config` (after the `continuous:` field):

```python
    audio: AudioConfig = Field(default_factory=AudioConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio_config.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Verify the example config still loads**

Run: `.venv/bin/python -c "from wildlife.config import load_config; c=load_config('config.example.yaml'); print('audio.enabled =', c.audio.enabled)"`
Expected: prints `audio.enabled = False`.

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/config.py tests/test_audio_config.py
git add src/wildlife/config.py tests/test_audio_config.py
git commit -m "feat(config): add AudioConfig for BirdNET audio bird-ID"
```

---

### Task 3: `store.audio_path` column + `save_audio_capture`

**Files:**
- Modify: `src/wildlife/store.py`
- Test: `tests/test_store.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Store.save_audio_capture(*, camera_id: str, event_ts: datetime, capture_ts: datetime, species: str, confidence: float, spectrogram_rgb: np.ndarray, clip_bytes: bytes | None, source_kind: str = "audio") -> int`; row dicts gain an `"audio_path"` key. A private `_write_image_and_thumb(image, abs_dir, stem, width) -> tuple[str, str]` is extracted and shared with `save_capture`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
def test_save_audio_capture_round_trips(tmp_path):
    import numpy as np
    from datetime import datetime
    from wildlife.store import Store

    store = Store(tmp_path / "c.db", tmp_path / "caps")
    store.init_schema()
    spec = np.zeros((64, 200, 3), dtype=np.uint8)  # RGB spectrogram
    cid = store.save_audio_capture(
        camera_id="cam1",
        event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, 0, 1),
        species="American Robin",
        confidence=0.82,
        spectrogram_rgb=spec,
        clip_bytes=b"\x00\x01\x02\x03",
    )
    row = store.get(cid)
    assert row["source_kind"] == "audio"
    assert row["label"] == "American Robin"
    assert row["confidence"] == pytest.approx(0.82)
    assert row["box_x1"] is None and row["box_x2"] is None
    assert row["image_path"] and row["thumb_path"]
    assert row["audio_path"] and row["audio_path"].endswith(".m4a")
    # the clip and images actually exist on disk
    assert (store.captures_dir / row["audio_path"]).is_file()
    assert (store.captures_dir / row["image_path"]).is_file()
    store.close()


def test_save_audio_capture_without_clip_leaves_audio_path_null(tmp_path):
    import numpy as np
    from datetime import datetime
    from wildlife.store import Store

    store = Store(tmp_path / "c.db", tmp_path / "caps")
    store.init_schema()
    cid = store.save_audio_capture(
        camera_id="cam1",
        event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, 0, 1),
        species="Steller's Jay",
        confidence=0.7,
        spectrogram_rgb=np.zeros((64, 200, 3), dtype=np.uint8),
        clip_bytes=None,
    )
    row = store.get(cid)
    assert row["audio_path"] is None
    assert row["image_path"]  # spectrogram still saved
    store.close()


def test_audio_path_migrates_onto_a_legacy_db(tmp_path):
    import sqlite3
    from wildlife.store import Store

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE captures ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, "
        "event_ts TEXT NOT NULL, capture_ts TEXT NOT NULL, label TEXT NOT NULL, "
        "confidence REAL NOT NULL, box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL, "
        "image_path TEXT NOT NULL, thumb_path TEXT NOT NULL, width INTEGER, height INTEGER)"
    )
    conn.commit()
    conn.close()
    store = Store(db, tmp_path / "caps")
    store.init_schema()
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(captures)")}
    assert "audio_path" in cols
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_store.py -q -k "audio"`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'save_audio_capture'` / missing `audio_path`.

- [ ] **Step 3: Implement the column + method**

In `src/wildlife/store.py`:

Add `audio_path` to the CREATE TABLE in `_SCHEMA_SQL` (after the `source_kind` line):

```sql
    source_kind    TEXT    NOT NULL DEFAULT 'reolink',  -- provenance: reolink | continuous | audio
    audio_path     TEXT                                 -- relative clip path for source_kind='audio'
```

Add `"audio_path"` to `_COLUMNS` (append after `"source_kind"`), and add to `_COLUMN_ADDITIONS`:

```python
    ("audio_path", "TEXT"),
```

Extract the shared JPEG-writer. Add this private method to `Store` (place it just above `save_capture`):

```python
    def _write_image_and_thumb(
        self, image: Image.Image, abs_dir: Path, stem: str, width: int
    ) -> tuple[str, str]:
        """Write ``image`` as a full JPEG + a downscaled thumbnail; return (rel, thumb_rel).

        Shared by :meth:`save_capture` (BGR frames) and :meth:`save_audio_capture`
        (RGB spectrograms). Guards against filename collisions within one event.
        """
        image_abs = abs_dir / f"{stem}.jpg"
        if image_abs.exists():
            n = 1
            while (abs_dir / f"{stem}-{n}.jpg").exists():
                n += 1
            stem = f"{stem}-{n}"
            image_abs = abs_dir / f"{stem}.jpg"
        thumb_abs = abs_dir / f"{stem}_thumb.jpg"

        image.save(image_abs, format="JPEG", quality=self.jpeg_quality)
        thumb = image
        if width > self.thumbnail_px:
            new_h = max(1, round(image.height * self.thumbnail_px / width))
            thumb = image.resize((self.thumbnail_px, new_h), Image.LANCZOS)
        thumb.save(thumb_abs, format="JPEG", quality=self.jpeg_quality)

        rel_dir = abs_dir.relative_to(self.captures_dir)
        return (
            (rel_dir / f"{stem}.jpg").as_posix(),
            (rel_dir / f"{stem}_thumb.jpg").as_posix(),
        )
```

Refactor `save_capture` to use it. Replace the block in `save_capture` that builds `image_abs`/collision-guard/saves the JPEG + thumbnail and computes `image_rel`/`thumb_rel` with:

```python
        image_rel, thumb_rel = self._write_image_and_thumb(image, abs_dir, stem, width)
```

(The `stem` and `image` construction above it, and the INSERT below, stay unchanged — this is a pure refactor; the existing store tests are the regression guard.)

Add `save_audio_capture` (after `save_capture`):

```python
    def save_audio_capture(
        self,
        *,
        camera_id: str,
        event_ts: datetime,
        capture_ts: datetime,
        species: str,
        confidence: float,
        spectrogram_rgb: np.ndarray,
        clip_bytes: bytes | None,
        source_kind: str = "audio",
    ) -> int:
        """Persist one audio detection: spectrogram JPEG + thumbnail + optional clip.

        ``spectrogram_rgb`` is an RGB ``uint8`` image (H, W, 3). The clip (AAC/.m4a
        bytes) is written to ``audio_path`` when given, else that column is NULL.
        Box columns are NULL (audio has no bounding box). Returns the new row id.
        """
        image = Image.fromarray(np.ascontiguousarray(spectrogram_rgb), mode="RGB")
        height, width = int(spectrogram_rgb.shape[0]), int(spectrogram_rgb.shape[1])

        rel_dir = Path(
            f"{capture_ts.year:04d}", f"{capture_ts.month:02d}", f"{capture_ts.day:02d}"
        )
        abs_dir = self.captures_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        ts_compact = capture_ts.strftime("%Y%m%dT%H%M%S_%f")
        stem = "_".join(
            (_sanitize(camera_id), ts_compact, _sanitize(species), f"{confidence:.2f}", "audio")
        )
        image_rel, thumb_rel = self._write_image_and_thumb(image, abs_dir, stem, width)

        audio_rel: str | None = None
        if clip_bytes is not None:
            # Reuse the (collision-resolved) image stem so the clip sits beside it.
            clip_stem = Path(image_rel).stem
            clip_abs = abs_dir / f"{clip_stem}.m4a"
            clip_abs.write_bytes(clip_bytes)
            audio_rel = (rel_dir / f"{clip_stem}.m4a").as_posix()

        with self._write_lock:
            cur = self._conn.execute(
                """
                INSERT INTO captures (
                    camera_id, event_ts, capture_ts, label, confidence,
                    image_path, thumb_path, width, height, source_kind, audio_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id, _iso(event_ts), _iso(capture_ts), species, float(confidence),
                    image_rel, thumb_rel, width, height, source_kind, audio_rel,
                ),
            )
            self._conn.commit()
            capture_id = int(cur.lastrowid)

        logger.info(
            "Saved audio capture id=%d camera=%s species=%s conf=%.3f clip=%s",
            capture_id, camera_id, species, confidence, audio_rel or "-",
        )
        return capture_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: PASS (all store tests — the existing ones still green after the refactor, plus the 3 new).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/store.py tests/test_store.py
git add src/wildlife/store.py tests/test_store.py
git commit -m "feat(store): add audio_path column + save_audio_capture"
```

---

### Task 4: `_colormap.py` + `render_spectrogram`

**Files:**
- Create: `src/wildlife/_colormap.py`
- Create: `src/wildlife/audio.py` (with `render_spectrogram` only for now; `AudioAnalyzer` added in Task 5)
- Test: `tests/test_spectrogram.py`

**Interfaces:**
- Consumes: nothing (numpy + Pillow only).
- Produces: `_colormap.MAGMA_LUT: np.ndarray` (256, 3) uint8; `audio.render_spectrogram(pcm: np.ndarray, sr: int = 48000) -> np.ndarray` (RGB uint8, fixed height 256, `(256, W, 3)`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spectrogram.py`:

```python
"""Tests for the numpy+Pillow spectrogram renderer (no birdnet/matplotlib/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")

from wildlife._colormap import MAGMA_LUT  # noqa: E402
from wildlife.audio import render_spectrogram  # noqa: E402


def test_magma_lut_shape_and_dtype():
    assert MAGMA_LUT.shape == (256, 3)
    assert MAGMA_LUT.dtype == np.uint8


def test_render_spectrogram_returns_fixed_height_rgb():
    # a 3s 48kHz tone
    sr = 48000
    t = np.linspace(0, 3, sr * 3, endpoint=False, dtype=np.float32)
    pcm = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    img = render_spectrogram(pcm, sr=sr)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] == 256  # fixed display height
    assert img.shape[1] > 0


def test_render_spectrogram_handles_silence_without_crashing():
    img = render_spectrogram(np.zeros(48000 * 3, dtype=np.float32), sr=48000)
    assert img.shape[0] == 256 and img.shape[2] == 3


def test_render_spectrogram_accepts_int16_range():
    # int16-scaled input is normalized, not clipped to garbage
    pcm = (np.random.default_rng(0).integers(-3000, 3000, 48000 * 3)).astype(np.float32)
    img = render_spectrogram(pcm, sr=48000)
    assert img.shape[0] == 256
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_spectrogram.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wildlife._colormap'`.

- [ ] **Step 3: Implement the colormap + renderer**

Create `src/wildlife/_colormap.py`:

```python
"""A small magma colormap LUT for spectrograms (no matplotlib dependency).

The viridis-family colormaps (viridis/magma/inferno/plasma) by Stefan van der Walt
and Nathaniel Smith are released into the public domain (CC0); matplotlib merely
embeds the same arrays. We store a handful of anchor stops and interpolate them up
to a 256-entry uint8 lookup table at import.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAGMA_LUT"]

# (position 0..1, R, G, B) anchor stops approximating magma.
_ANCHORS = np.array(
    [
        [0.00, 0, 0, 4],
        [0.14, 28, 16, 68],
        [0.29, 79, 18, 123],
        [0.43, 129, 37, 129],
        [0.57, 181, 54, 122],
        [0.71, 229, 80, 100],
        [0.86, 251, 135, 97],
        [1.00, 252, 253, 191],
    ],
    dtype=np.float64,
)


def _build_lut() -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256)
    pos = _ANCHORS[:, 0]
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, pos, _ANCHORS[:, ch + 1]).round().astype(np.uint8)
    return lut


MAGMA_LUT: np.ndarray = _build_lut()
```

Create `src/wildlife/audio.py`:

```python
"""Audio bird-ID: BirdNET analysis + spectrogram rendering.

Hardware/heavy imports are confined here (mirrors ``capture.py``/``motion.py``):
``birdnet`` is imported LAZILY inside :class:`AudioAnalyzer`, so ``render_spectrogram``
(numpy + Pillow only) and this module import fine without the ``[audio]`` extra —
keeping the pure spectrogram tests hardware-free.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from wildlife._colormap import MAGMA_LUT

__all__ = ["render_spectrogram"]

# STFT params for a 3s / 48kHz window: ~560 columns, 46.9 Hz/bin.
_N_FFT = 1024
_HOP = 256
_FMAX_HZ = 12000  # crop above this (bird band); avoids an empty top third
_DB_FLOOR = 80.0
_OUT_HEIGHT = 256


def render_spectrogram(pcm: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Render a 48kHz mono PCM window to an RGB ``uint8`` spectrogram image.

    Hand-rolled STFT (numpy) → log-magnitude (dB, 80 dB floor) → magma LUT → a
    fixed-height (256 px) RGB array, low frequencies at the bottom. Deterministic,
    no matplotlib/scipy.
    """
    x = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:  # int16-scaled input -> normalize to [-1, 1]
        x = x / 32768.0
    if x.size < _N_FFT:
        x = np.pad(x, (0, _N_FFT - x.size))

    window = np.hanning(_N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, _N_FFT)[::_HOP]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1))  # (frames, n_fft/2+1)

    db = 20.0 * np.log10(spec + 1e-6)
    db = np.maximum(db, db.max() - _DB_FLOOR)

    keep = int(_FMAX_HZ / (sr / _N_FFT)) + 1  # bins up to _FMAX_HZ
    db = db[:, :keep]

    lo, hi = float(db.min()), float(db.max())
    norm = ((db - lo) / (hi - lo + 1e-9) * 255.0).astype(np.uint8)  # (frames, freq)
    norm = np.flipud(norm.T)  # (freq, frames), low freq at bottom
    rgb = MAGMA_LUT[norm]  # (freq, frames, 3)

    pil = Image.fromarray(rgb, mode="RGB")  # size = (frames, freq)
    new_w = max(1, round(pil.width * _OUT_HEIGHT / pil.height))
    pil = pil.resize((new_w, _OUT_HEIGHT), Image.LANCZOS)
    return np.asarray(pil)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_spectrogram.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/_colormap.py src/wildlife/audio.py tests/test_spectrogram.py
git add src/wildlife/_colormap.py src/wildlife/audio.py tests/test_spectrogram.py
git commit -m "feat(audio): numpy+Pillow spectrogram renderer + magma LUT"
```

---

### Task 5: `AudioAnalyzer` (BirdNET wrapper) + `[audio]` extra

**Files:**
- Modify: `src/wildlife/audio.py` (add `AudioAnalyzer`)
- Modify: `pyproject.toml` (`[audio]` extra)
- Test: `tests/test_audio_analyzer.py`

**Interfaces:**
- Consumes: `AudioConfig` (Task 2); `render_spectrogram` (Task 4).
- Produces: `AudioAnalyzer(cfg: AudioConfig)` with `analyze(pcm: np.ndarray) -> list[tuple[str, float]]` (common-name, confidence), thread-safe (internal lock). BirdNET imported lazily.

**Note (verify-on-deploy):** `birdnet` is NOT installed in the dev venv, so `AudioAnalyzer`'s tests use a **fake model** (monkeypatched), and the real-`birdnet` test is `importorskip`-guarded (skipped in CI). Write the wrapper against the source-verified v0.2.16 API; the geo-result accessor and `predict_arrays`/`to_structured_array` behavior are confirmed on the prod mini at deploy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_analyzer.py`:

```python
"""AudioAnalyzer tests using a fake BirdNET (no real birdnet needed)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from wildlife.config import AudioConfig


class _FakeStructured:
    def __init__(self, rows):
        self._rows = rows

    def to_structured_array(self):
        # numpy structured array with the documented columns
        return np.array(
            [(r[0], 0.0, 3.0, r[1], r[2]) for r in self._rows],
            dtype=[
                ("input", "O"), ("start_time", "f4"), ("end_time", "f4"),
                ("species_name", "O"), ("confidence", "f4"),
            ],
        )


class _FakeAcoustic:
    def __init__(self):
        self.last_kwargs = None

    def predict_arrays(self, inp, **kwargs):
        self.last_kwargs = kwargs
        # one segment, one species above threshold
        return _FakeStructured([("", "Turdus migratorius_American Robin", 0.82)])


class _FakeGeo:
    def predict(self, lat, lon, **kwargs):
        class _R:
            def to_structured_array(self_inner):
                return np.array(
                    [("Turdus migratorius_American Robin", 0.5)],
                    dtype=[("species_name", "O"), ("confidence", "f4")],
                )
        return _R()


def _install_fake_birdnet(monkeypatch, acoustic, geo):
    fake = types.ModuleType("birdnet")

    def _load(model_type, version, backend, **kw):
        return acoustic if model_type == "acoustic" else geo

    fake.load = _load
    monkeypatch.setitem(sys.modules, "birdnet", fake)


def test_analyze_returns_common_name_and_confidence(monkeypatch):
    from wildlife.audio import AudioAnalyzer

    acoustic = _FakeAcoustic()
    _install_fake_birdnet(monkeypatch, acoustic, _FakeGeo())
    analyzer = AudioAnalyzer(AudioConfig(use_geo_filter=False))
    out = analyzer.analyze(np.zeros(144000, dtype=np.float32))
    assert len(out) == 1
    name, conf = out[0]
    assert name == "American Robin"
    assert conf == pytest.approx(0.82, abs=1e-4)  # float32 round-trip
    # confidence threshold + bandpass are passed through to predict
    assert acoustic.last_kwargs["default_confidence_threshold"] == pytest.approx(0.25)
    assert acoustic.last_kwargs["bandpass_fmin"] == 0


def test_geo_shortlist_is_built_and_passed(monkeypatch):
    from wildlife.audio import AudioAnalyzer

    acoustic = _FakeAcoustic()
    _install_fake_birdnet(monkeypatch, acoustic, _FakeGeo())
    analyzer = AudioAnalyzer(AudioConfig(use_geo_filter=True, latitude=37.2, longitude=-107.5))
    analyzer.analyze(np.zeros(144000, dtype=np.float32))
    assert acoustic.last_kwargs["custom_species_list"] == ["Turdus migratorius_American Robin"]


def test_geo_failure_falls_back_to_no_filter(monkeypatch):
    from wildlife.audio import AudioAnalyzer

    class _BadGeo:
        def predict(self, *a, **k):
            raise RuntimeError("kaggle down")

    acoustic = _FakeAcoustic()
    _install_fake_birdnet(monkeypatch, acoustic, _BadGeo())
    analyzer = AudioAnalyzer(AudioConfig(use_geo_filter=True, latitude=37.2, longitude=-107.5))
    analyzer.analyze(np.zeros(144000, dtype=np.float32))
    assert acoustic.last_kwargs["custom_species_list"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_analyzer.py -q`
Expected: FAIL — `ImportError: cannot import name 'AudioAnalyzer'`.

- [ ] **Step 3: Implement `AudioAnalyzer`**

In `src/wildlife/audio.py`, add to `__all__`: `"AudioAnalyzer"`. Add imports at top: `import logging`, `import threading`, `from datetime import datetime`. Add:

```python
logger = logging.getLogger(__name__)

_SAMPLE_RATE = 48000
_TOP_K = 1  # take the single best species per 3s window


def _week_of_year(now: datetime) -> int:
    """BirdNET's 1-48 week convention (4 weeks per month)."""
    return (now.month - 1) * 4 + min(4, (now.day - 1) // 7 + 1)


class AudioAnalyzer:
    """Shared, thread-safe BirdNET wrapper. Loads the models once; ``analyze`` a window.

    ``birdnet`` (and thus TensorFlow) is imported lazily here so the rest of the
    package imports without the ``[audio]`` extra.
    """

    def __init__(self, cfg: object) -> None:  # cfg: AudioConfig (typed loosely to avoid import)
        import birdnet  # lazy: pulls TensorFlow

        self._cfg = cfg
        self._lock = threading.Lock()
        self._acoustic = birdnet.load("acoustic", "2.4", "tf")
        self._species_list: list[str] | None = None
        if getattr(cfg, "use_geo_filter", False):
            self._species_list = self._build_geo_shortlist(birdnet, cfg)
        logger.info(
            "AudioAnalyzer ready (geo_filter=%s, species_shortlist=%s).",
            getattr(cfg, "use_geo_filter", False),
            "-" if self._species_list is None else len(self._species_list),
        )

    def _build_geo_shortlist(self, birdnet, cfg) -> list[str] | None:
        """Occurrence shortlist for lat/lon + current week; None on any failure."""
        try:
            geo = birdnet.load("geo", "2.4", "tf")
            week = _week_of_year(datetime.now())
            result = geo.predict(cfg.latitude, cfg.longitude, week=week)
            arr = result.to_structured_array()  # verify column on prod: 'species_name'
            return [str(s) for s in arr["species_name"]]
        except Exception:  # noqa: BLE001 - geo is best-effort; fall back to no filter
            logger.warning("Geo shortlist unavailable; running without it.", exc_info=True)
            return None

    def analyze(self, pcm: np.ndarray) -> list[tuple[str, float]]:
        """Return ``[(common_name, confidence), …]`` for a 48kHz mono float32 window."""
        cfg = self._cfg
        with self._lock:
            result = self._acoustic.predict_arrays(
                (np.asarray(pcm, dtype=np.float32), _SAMPLE_RATE),
                top_k=_TOP_K,
                default_confidence_threshold=cfg.confidence_threshold,
                bandpass_fmin=cfg.bandpass_fmin,
                custom_species_list=self._species_list,
            )
        arr = result.to_structured_array()
        out: list[tuple[str, float]] = []
        for row in arr:
            species_name = str(row["species_name"])
            common = species_name.split("_", 1)[-1]  # "Sci_Common" -> "Common"
            out.append((common, float(row["confidence"])))
        return out
```

In `pyproject.toml`, add the extra (after the `autolabel` block):

```toml
# BirdNET audio bird-ID. NOTE: birdnet pulls full TensorFlow (CPU) + scipy/pandas/
# pyarrow/soundfile/kagglehub -- a large footprint, confined to this optional extra.
audio = [
    "birdnet>=0.2,<0.3",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio_analyzer.py -q`
Expected: PASS (3 passed) — the fake `birdnet` is injected via `sys.modules`, so no real install is needed.

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/audio.py tests/test_audio_analyzer.py
git add src/wildlife/audio.py pyproject.toml tests/test_audio_analyzer.py
git commit -m "feat(audio): AudioAnalyzer BirdNET wrapper + [audio] extra"
```

---

### Task 6: `AudioDetectionSource` — per-camera analyzer thread

**Files:**
- Create: `src/wildlife/events/audio_detection.py`
- Test: `tests/test_audio_detection_source.py`

**Interfaces:**
- Consumes: `AudioConfig`, `AudioAnalyzer` (Task 5), `RepeatConfirmer` (Task 1), `render_spectrogram` (Task 4), `Store.save_audio_capture` (Task 3), `AudioConfig` fields + `livestream.rtsp_listen`.
- Produces: `AudioDetectionSource(camera, config, analyzer, store)` with `start() -> None`, `stop() -> None`, and testable pure helpers `_within_active_hours(now_wall) -> bool` and `_process_window(pcm, now) -> None` (renders/encodes/saves on a confirmed detection). `WIN_SAMPLES=144000`, `HOP_SAMPLES=72000` module constants.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_detection_source.py` (hardware-free: fake analyzer/confirmer/store; the ffmpeg/pipe path is not exercised):

```python
"""AudioDetectionSource orchestration tests (no ffmpeg/birdnet)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from wildlife.config import AudioConfig
from wildlife.events.audio_detection import AudioDetectionSource, HOP_SAMPLES, WIN_SAMPLES


def _source(**audio_overrides):
    cfg = AudioConfig(use_geo_filter=False, **audio_overrides)
    config = SimpleNamespace(audio=cfg, livestream=SimpleNamespace(rtsp_listen=":8554"))
    camera = SimpleNamespace(id="cam1")
    analyzer = SimpleNamespace(analyze=lambda pcm: [("American Robin", 0.9)])
    store = SimpleNamespace(saved=[])
    store.save_audio_capture = lambda **kw: store.saved.append(kw) or 1
    return AudioDetectionSource(camera, config, analyzer, store), store


def test_window_hop_constants():
    assert WIN_SAMPLES == 144000
    assert HOP_SAMPLES == 72000


def test_confirmed_detection_saves_audio_capture(monkeypatch):
    # render_spectrogram is patched to avoid heavy work; assert a save happens on
    # the SECOND window (min_confirmations defaults to 2).
    import wildlife.events.audio_detection as mod

    monkeypatch.setattr(mod, "render_spectrogram", lambda pcm, sr=48000: np.zeros((8, 8, 3), np.uint8))
    src, store = _source(min_confirmations=2, cooldown_s=0)
    monkeypatch.setattr(src, "_encode_clip", lambda pcm: b"clip")  # skip ffmpeg
    win = np.zeros(WIN_SAMPLES, dtype=np.float32)
    src._process_window(win, datetime(2026, 7, 6, 6, 0, 0))
    assert store.saved == []                      # 1st hit: not confirmed
    src._process_window(win, datetime(2026, 7, 6, 6, 0, 2))
    assert len(store.saved) == 1                  # 2nd hit -> saved
    row = store.saved[0]
    assert row["species"] == "American Robin"
    assert row["source_kind"] == "audio"
    assert row["clip_bytes"] == b"clip"


def test_active_hours_gate_blocks_processing():
    src, store = _source(active_hours="20:00-06:00", min_confirmations=1, cooldown_s=0)
    # 12:00 is outside the window -> nothing analyzed/saved
    src._process_window(np.zeros(WIN_SAMPLES, np.float32), datetime(2026, 7, 6, 12, 0, 0))
    assert store.saved == []


def test_within_active_hours_wraps_midnight():
    src, _ = _source(active_hours="20:00-06:00")
    assert src._within_active_hours(datetime(2026, 7, 6, 23, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 3, 0)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 12, 0)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_detection_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wildlife.events.audio_detection'`.

- [ ] **Step 3: Implement the source**

Create `src/wildlife/events/audio_detection.py`:

```python
"""Per-camera BirdNET audio analyzer thread.

Manages its own daemon thread (``start()``/``stop()``), mirroring the notifier's
lifecycle rather than the queue-backed event sources — it saves detections
DIRECTLY (BirdNET is the model; there is no YOLO consumer to feed). ffmpeg decodes
the go2rtc restream audio to 48 kHz mono PCM on a subprocess pipe; the reader frames
it into 3 s windows (1.5 s hop) and, on a confirmed detection, renders a spectrogram,
encodes an AAC clip from the same PCM, and calls ``save_audio_capture``.

Heavy work is confined: ``render_spectrogram`` (numpy/Pillow) is module-level, but
``subprocess``/ffmpeg run only inside the thread. ``AudioAnalyzer`` (birdnet) is
injected, already loaded.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from datetime import datetime

import numpy as np

from wildlife.audio import render_spectrogram
from wildlife.audio_gate import RepeatConfirmer

__all__ = ["AudioDetectionSource", "WIN_SAMPLES", "HOP_SAMPLES"]

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 48000
WIN_SAMPLES = _SAMPLE_RATE * 3          # 144000  (3.0 s)
HOP_SAMPLES = _SAMPLE_RATE * 3 // 2     # 72000   (1.5 s hop, 50% overlap)
_HOP_BYTES = HOP_SAMPLES * 2           # int16
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def _parse_port(listen: str) -> int:
    return int(listen.rsplit(":", 1)[-1])


def _parse_active_hours(spec: str) -> tuple[int, int] | None:
    spec = spec.strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", spec)
    if not m:
        return None
    sh, sm, eh, em = (int(g) for g in m.groups())
    return (sh * 60 + sm, eh * 60 + em)


class AudioDetectionSource:
    """Analyze one camera's audio and save confirmed bird detections directly."""

    def __init__(self, camera: object, config: object, analyzer: object, store: object) -> None:
        cc = config.audio
        self._camera = camera
        self._analyzer = analyzer
        self._store = store
        self._stream = cc.stream
        self._rtsp_port = _parse_port(config.livestream.rtsp_listen)
        self._active_window = _parse_active_hours(cc.active_hours)
        self._confirmer = RepeatConfirmer(cc.min_confirmations, cc.confirm_window_s, cc.cooldown_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name=f"audio-{self._camera.id}", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - best-effort
                pass
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    # -- pure decision helpers (unit-tested) ------------------------------
    def _within_active_hours(self, now_wall: datetime) -> bool:
        if self._active_window is None:
            return True
        start, end = self._active_window
        cur = now_wall.hour * 60 + now_wall.minute
        if start <= end:
            return start <= cur < end
        return cur >= start or cur < end

    def _process_window(self, pcm: np.ndarray, now: datetime) -> None:
        """Analyze one window; on a confirmed species, render+encode+save."""
        if not self._within_active_hours(now):
            return
        for species, conf in self._analyzer.analyze(pcm):
            if not self._confirmer.offer(species, conf, now):
                continue
            try:
                spec_rgb = render_spectrogram(pcm, _SAMPLE_RATE)
            except Exception:  # noqa: BLE001 - a bad render skips the detection
                logger.exception("Camera %s: spectrogram render failed.", self._camera.id)
                continue
            clip = None
            try:
                clip = self._encode_clip(pcm)
            except Exception:  # noqa: BLE001 - save w/o clip rather than lose the detection
                logger.warning("Camera %s: clip encode failed; saving without audio.", self._camera.id)
            self._store.save_audio_capture(
                camera_id=self._camera.id,
                event_ts=now,
                capture_ts=datetime.now(),
                species=species,
                confidence=conf,
                spectrogram_rgb=spec_rgb,
                clip_bytes=clip,
                source_kind="audio",
            )
            logger.info("Camera %s: audio detection %s (%.2f) saved.", self._camera.id, species, conf)

    # -- ffmpeg (hardware; not unit-tested) -------------------------------
    def _decode_cmd(self) -> list[str]:
        url = f"rtsp://127.0.0.1:{self._rtsp_port}/{self._camera.id}_{self._stream}"
        return [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", url, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(_SAMPLE_RATE),
            "-f", "s16le", "-",
        ]

    def _encode_clip(self, pcm: np.ndarray) -> bytes:
        """Encode a float32 window to AAC/.m4a bytes via ffmpeg, from stdin PCM.

        MP4/.m4a with ``+faststart`` needs a *seekable* output (piping to stdout
        fails: "muxer does not support non seekable output"), so we encode to a
        temp file and read the bytes back. The clip is tiny (~38 KB), so this is cheap.
        """
        import os
        import tempfile

        pcm16 = np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()
        fd, tmp = tempfile.mkstemp(suffix=".m4a")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-f", "s16le", "-ar", str(_SAMPLE_RATE), "-ac", "1", "-i", "-",
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", tmp],
                input=pcm16, check=True,
            )
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run(self) -> None:
        """Open ffmpeg, frame the PCM into windows, process each; reconnect on drop."""
        backoff = _INITIAL_BACKOFF_S
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._decode_cmd(), stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=WIN_SAMPLES * 2,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Camera %s: failed to start audio ffmpeg.", self._camera.id)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue

            ring = bytearray()
            delivered = False
            while not self._stop.is_set():
                hop = self._read_exact(self._proc.stdout, _HOP_BYTES)
                if hop is None:
                    break  # stream dropped -> reopen
                delivered = True
                ring += hop
                if len(ring) >= WIN_SAMPLES * 2:
                    window = np.frombuffer(ring[-WIN_SAMPLES * 2:], dtype="<i2")
                    pcm = window.astype(np.float32) / 32768.0
                    try:
                        self._process_window(pcm, datetime.now())
                    except Exception:  # noqa: BLE001 - one bad window mustn't kill the thread
                        logger.exception("Camera %s: audio window error.", self._camera.id)
                    del ring[:_HOP_BYTES]

            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            if self._stop.is_set():
                break
            if delivered:
                backoff = _INITIAL_BACKOFF_S
            logger.warning("Camera %s: audio stream ended; reconnecting in %.1fs.", self._camera.id, backoff)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

    def _read_exact(self, pipe, n: int) -> bytes | None:
        """Read exactly ``n`` bytes (pipe reads are short); None on EOF/stop."""
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            chunk = pipe.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio_detection_source.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/events/audio_detection.py tests/test_audio_detection_source.py
git add src/wildlife/events/audio_detection.py tests/test_audio_detection_source.py
git commit -m "feat(audio): AudioDetectionSource per-camera analyzer thread"
```

---

### Task 7: Worker wiring — load BirdNET once, start/stop audio threads

**Files:**
- Modify: `src/wildlife/worker.py`
- Test: `tests/test_worker_audio.py`

**Interfaces:**
- Consumes: `AudioAnalyzer` (Task 5), `AudioDetectionSource` (Task 6), `Config.audio.enabled`.
- Produces: worker builds `self._audio_analyzer` in `_setup` and starts one `AudioDetectionSource` per camera (tracked in `self._audio_sources`) in `_start_producers` when `audio.enabled`; `_teardown` stops them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_audio.py`:

```python
"""Worker audio wiring (guarded: worker pulls in torch/cv2)."""

from __future__ import annotations

import pytest

pytest.importorskip("wildlife.worker")

from wildlife.config import Config  # noqa: E402
from wildlife import worker as W  # noqa: E402


def _config_dict(audio_enabled: bool):
    d = {
        "cameras": [{
            "id": "cam1", "host": "192.168.1.101", "username": "admin", "password": "x",
            "rtsp_main": "rtsp://{username}:{password}@{host}:554/main",
            "rtsp_sub": "rtsp://{username}:{password}@{host}:554/sub",
        }],
        "event_source": "reolink_native",
        "capture": {"burst_frames": 3, "burst_interval_ms": 100, "stream": "main",
                    "rtsp_timeout_s": 5, "max_concurrent": 1},
        "detection": {"model_path": "models/yolov8s.pt", "device": "cpu",
                      "animal_classes": ["bird"], "confidence_threshold": 0.5,
                      "min_box_area_frac": 0.01, "save_best_only": True},
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": "/tmp/wa_caps", "db_path": "/tmp/wa.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
    }
    if audio_enabled:
        d["audio"] = {"enabled": True, "use_geo_filter": False}
    return d


class _FakeSource:
    def __init__(self, *a, **k):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self, timeout=5.0):
        self.stopped = True


def test_audio_sources_start_only_when_enabled(monkeypatch):
    made = []
    monkeypatch.setattr(W, "AudioDetectionSource",
                        lambda *a, **k: made.append(_FakeSource()) or made[-1])
    monkeypatch.setattr(W, "AudioAnalyzer", lambda cfg: object())

    on = W._Worker(Config.model_validate(_config_dict(True)))
    on._audio_analyzer = object()  # pretend _setup built it
    # Set shutdown first so the primary reolink producer threads _start_producers
    # also launches exit immediately (their _produce loop is `while not shutdown`) —
    # otherwise they'd attempt a real RTSP/reolink connection. The audio-source loop
    # is not shutdown-gated, so the faked sources are still created + started.
    on._shutdown.set()
    on._start_producers()
    assert len(on._audio_sources) == 1 and on._audio_sources[0].started is True

    off = W._Worker(Config.model_validate(_config_dict(False)))
    off._shutdown.set()
    off._start_producers()
    assert off._audio_sources == []


def test_teardown_stops_audio_sources():
    w = W._Worker(Config.model_validate(_config_dict(True)))
    s = _FakeSource()
    w._audio_sources = [s]
    w._teardown()
    assert s.stopped is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_worker_audio.py -q`
Expected: FAIL — `AttributeError: '_Worker' object has no attribute '_audio_sources'` / `module 'wildlife.worker' has no attribute 'AudioDetectionSource'`.

- [ ] **Step 3: Implement the wiring**

In `src/wildlife/worker.py`:

Add imports near the other event imports:

```python
from wildlife.audio import AudioAnalyzer
from wildlife.events.audio_detection import AudioDetectionSource
```

In `_Worker.__init__`, add (after `self._deduper` init):

```python
        self._audio_analyzer: AudioAnalyzer | None = None
        self._audio_sources: list = []
```

In `_setup`, after the deduper is built, add:

```python
        if cfg.audio.enabled:
            self._audio_analyzer = AudioAnalyzer(cfg.audio)
            logger.info("Audio bird-ID enabled (BirdNET loaded).")
```

In `_start_producers`, after the per-camera loop that starts the primary/continuous producers, add:

```python
        if self._config.audio.enabled and self._audio_analyzer is not None:
            for camera in self._cameras.values():
                source = AudioDetectionSource(
                    camera, self._config, self._audio_analyzer, self._store
                )
                source.start()
                self._audio_sources.append(source)
```

In `_teardown`, before closing the store, add:

```python
        for source in self._audio_sources:
            try:
                source.stop()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                logger.exception("Error stopping an audio source.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_worker_audio.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Full suite + lint + commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/wildlife/worker.py tests/test_worker_audio.py
git add src/wildlife/worker.py tests/test_worker_audio.py
git commit -m "feat(worker): load BirdNET once + run per-camera audio threads"
```

Expected: full suite green.

---

### Task 8: Gallery backend — `/audio/<id>`, `source_kind` in payload + filter

**Files:**
- Modify: `src/wildlife/gallery/app.py`
- Modify: `src/wildlife/store.py` (add a `source_kind` query filter)
- Test: `tests/test_gallery_audio.py`

**Interfaces:**
- Consumes: `save_audio_capture` (Task 3); `audio_path`/`source_kind` columns.
- Produces: `GET /audio/<int:capture_id>` (serves the `.m4a`, `audio/mp4`, 404 when absent); `_serialize` adds `"source_kind"` + `"audio_url"`; `Store.query(..., source_kind: str | None = None)` filters on `source_kind`; `_parse_filters` reads `source_kind`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gallery_audio.py`:

```python
"""Gallery audio endpoint + source_kind serialization/filter tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask")

from datetime import datetime  # noqa: E402

from wildlife.config import load_config  # noqa: E402
from wildlife.gallery.app import create_app  # noqa: E402
from wildlife.store import Store  # noqa: E402


def _app(tmp_path):
    # minimal config on disk
    cfg_text = f"""
cameras:
  - id: cam1
    host: 1.2.3.4
    username: u
    password: p
    rtsp_main: "rtsp://{{username}}:{{password}}@{{host}}/main"
    rtsp_sub: "rtsp://{{username}}:{{password}}@{{host}}/sub"
event_source: reolink_native
capture: {{burst_frames: 3, burst_interval_ms: 100, stream: main, rtsp_timeout_s: 5, max_concurrent: 1}}
detection: {{model_path: m.pt, device: cpu, animal_classes: [bird], confidence_threshold: 0.5, min_box_area_frac: 0.01, save_best_only: true}}
dedupe: {{cooldown_s: 0}}
storage: {{captures_dir: "{tmp_path/'caps'}", db_path: "{tmp_path/'c.db'}"}}
retention: {{max_age_days: 30}}
gallery: {{host: 0.0.0.0, port: 8080, page_size: 60}}
resource_guard: {{detect_every_nth_event: 1, max_burst_per_minute: 20}}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)
    config = load_config(cfg_path)
    store = Store(config.storage.db_path, config.storage.captures_dir)
    store.init_schema()
    cid = store.save_audio_capture(
        camera_id="cam1", event_ts=datetime(2026, 7, 6, 6, 0, 0),
        capture_ts=datetime(2026, 7, 6, 6, 0, 1), species="American Robin",
        confidence=0.8, spectrogram_rgb=np.zeros((32, 64, 3), np.uint8), clip_bytes=b"\x00\x01",
    )
    store.close()
    return create_app(config), cid


def test_audio_route_serves_clip(tmp_path):
    app, cid = _app(tmp_path)
    client = app.test_client()
    resp = client.get(f"/audio/{cid}")
    assert resp.status_code == 200
    assert resp.mimetype == "audio/mp4"


def test_audio_route_404_when_no_clip(tmp_path):
    app, _ = _app(tmp_path)
    client = app.test_client()
    assert client.get("/audio/99999").status_code == 404


def test_index_payload_marks_audio_rows(tmp_path):
    app, cid = _app(tmp_path)
    client = app.test_client()
    data = client.get("/api/captures").get_json()
    row = next(c for c in data["captures"] if c["id"] == cid)
    assert row["source_kind"] == "audio"
    assert row["audio_url"].endswith(f"/audio/{cid}")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gallery_audio.py -q`
Expected: FAIL — 404 for `/audio/<id>` (route missing) / `KeyError: 'source_kind'`.

- [ ] **Step 3: Implement the store filter + gallery route/serialize**

In `src/wildlife/store.py`, add a `source_kind` filter. In `_build_filters`, add the parameter and clause:

```python
def _build_filters(
    *,
    camera_id: str | None,
    label: str | None,
    start: datetime | str | None,
    end: datetime | str | None,
    min_confidence: float | None,
    reviewed: bool | None,
    source_kind: str | None = None,
) -> tuple[list[str], list[Any]]:
    ...
    if source_kind is not None:
        clauses.append("source_kind = ?")
        params.append(source_kind)
    return clauses, params
```

Thread `source_kind` through `query` and `count` (add `source_kind: str | None = None` to both signatures and pass it into `_build_filters(...)`).

In `src/wildlife/gallery/app.py`:

Extend `_serialize` (add the two keys before the closing `}`):

```python
            "thumb_url": url_for("thumb", capture_id=cid),
            "image_url": url_for("image", capture_id=cid),
            "source_kind": row.get("source_kind"),
            "audio_url": url_for("audio", capture_id=cid) if row.get("audio_path") else None,
```

Extend `_parse_filters` (add to the returned dict):

```python
    source_kind = (args.get("source_kind") or "").strip() or None
    return {
        ...
        "source_kind": source_kind,
    }
```

Pass it into `_query_page`'s `store.query(...)` call:

```python
        rows = get_store().query(
            camera_id=filters["camera"],
            label=filters["label"],
            start=filters["start"],
            end=filters["end"],
            min_confidence=filters["min_confidence"],
            source_kind=filters["source_kind"],
            limit=page_size + 1,
            offset=offset,
        )
```

Add the `/audio/<id>` route (next to `/image` and `/thumb`):

```python
    @app.route("/audio/<int:capture_id>")
    def audio(capture_id: int):
        """Serve the AAC/.m4a clip for an audio capture (range-enabled)."""
        row = get_store().get(capture_id)
        if not row or not row.get("audio_path"):
            abort(404)
        full = (captures_dir / row["audio_path"]).resolve()
        try:
            full.relative_to(captures_dir)
        except ValueError:
            abort(403)
        if not full.is_file():
            abort(404)
        return send_file(full, mimetype="audio/mp4", conditional=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gallery_audio.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Full suite + lint + commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/wildlife/gallery/app.py src/wildlife/store.py tests/test_gallery_audio.py
git add src/wildlife/gallery/app.py src/wildlife/store.py tests/test_gallery_audio.py
git commit -m "feat(gallery): /audio route + source_kind in payload and filter"
```

---

### Task 9: Gallery frontend — audio badge + spectrogram-player lightbox + playhead

**Files:**
- Modify: `src/wildlife/gallery/templates/index.html`
- Modify: `src/wildlife/gallery/static/style.css` (small additions — playhead + audio-card badge)
- Test: `tests/test_gallery_audio.py` (extend — an automated template-render assertion) + a manual eyeball of the playhead (the JS/playhead itself is not unit-tested).

**Interfaces:**
- Consumes: `source_kind` + `audio_url` in the capture payload (Task 8).

**Note:** This is server-rendered HTML + vanilla JS. The lightbox must, for audio rows, show the spectrogram image plus an `<audio>` element and a synced playhead overlay; image rows behave exactly as today.

- [ ] **Step 1: Add `source_kind`/`audio_url` to the card dataset (both server + JS render paths)**

In `index.html`, add two data attributes to the server-rendered `<article class="card">` (in the `{% for c in captures %}` loop):

```html
      <article class="card"
               data-id="{{ c.id }}"
               data-full="{{ c.image_url }}"
               data-label="{{ c.label }}"
               data-confidence="{{ c.confidence }}"
               data-camera="{{ c.camera_id }}"
               data-ts="{{ c.capture_ts }}"
               data-kind="{{ c.source_kind }}"
               data-audio="{{ c.audio_url or '' }}">
```

Add an audio badge inside the card meta (after the `<span class="cam">` line):

```html
          {% if c.source_kind == 'audio' %}<span class="kind-badge">&#127925;</span>{% endif %}
```

In the JS `renderCard(c)` function, mirror the two dataset fields (after `card.dataset.ts = ...`):

```javascript
        card.dataset.kind = c.source_kind != null ? c.source_kind : "";
        card.dataset.audio = c.audio_url != null ? c.audio_url : "";
```

and add the badge in the JS-built meta (append after setting `.ts`):

```javascript
        if (c.source_kind === "audio") {
          const b = document.createElement("span");
          b.className = "kind-badge";
          b.textContent = "\u{1F3B5}";
          meta.appendChild(b);
        }
```

Add a **Kind filter** to the `<form id="filters">` (after the "Class" `<label>` block). The `source_kind` query param already flows through `_parse_filters` → `store.query` (Task 8) and through the load-more JS (which forwards all query params), so no JS change is needed:

```html
      <label class="field">
        <span>Kind</span>
        <select name="source_kind">
          <option value="">All</option>
          <option value="reolink" {% if filters.source_kind == 'reolink' %}selected{% endif %}>Camera AI</option>
          <option value="continuous" {% if filters.source_kind == 'continuous' %}selected{% endif %}>Motion</option>
          <option value="audio" {% if filters.source_kind == 'audio' %}selected{% endif %}>Audio (birds)</option>
        </select>
      </label>
```

- [ ] **Step 2: Add the spectrogram-player to the lightbox**

In the lightbox markup (`<figure class="lb-figure">`), add an audio panel after the `<img id="lbImg">`:

```html
    <figure class="lb-figure">
      <div class="lb-spinner" id="lbSpinner"></div>
      <div class="lb-spec" id="lbSpec" hidden>
        <img id="lbSpecImg" alt="">
        <div class="playhead" id="lbPlayhead"></div>
      </div>
      <img id="lbImg" alt="">
      <audio id="lbAudio" controls hidden></audio>
      <figcaption id="lbCap" class="lb-cap"></figcaption>
    </figure>
```

- [ ] **Step 3: Branch `openLightbox` on the source kind + drive the playhead**

Replace the `openLightbox(card)` function's body so audio rows show the spectrogram + player. Add references near the other lightbox consts:

```javascript
      const lbSpec = document.getElementById("lbSpec");
      const lbSpecImg = document.getElementById("lbSpecImg");
      const lbPlayhead = document.getElementById("lbPlayhead");
      const lbAudio = document.getElementById("lbAudio");
      let raf = 0;

      function stopPlayhead() { if (raf) cancelAnimationFrame(raf); raf = 0; }
      function tickPlayhead() {
        const d = lbAudio.duration;
        if (d && lbSpecImg.clientWidth) {
          lbPlayhead.style.transform =
            "translateX(" + (lbAudio.currentTime / d) * lbSpecImg.clientWidth + "px)";
        }
        raf = requestAnimationFrame(tickPlayhead);
      }
      lbAudio.addEventListener("play", () => { stopPlayhead(); raf = requestAnimationFrame(tickPlayhead); });
      lbAudio.addEventListener("pause", stopPlayhead);
      lbAudio.addEventListener("ended", stopPlayhead);
      lbAudio.addEventListener("seeked", tickPlayhead);
```

Rewrite `openLightbox` so it chooses the panel by `data-kind`:

```javascript
      function openLightbox(card) {
        currentCard = card;
        const label = card.dataset.label || "";
        const conf = fmtConf(card.dataset.confidence);
        const cam = card.dataset.camera || "";
        const ts = fmtTs(card.dataset.ts);
        const isAudio = card.dataset.kind === "audio";

        stopPlayhead();
        if (isAudio) {
          // spectrogram + audio player
          lbImg.hidden = true;
          lbSpec.hidden = false;
          lbAudio.hidden = false;
          lbSpinner.hidden = false;
          lbSpecImg.onload = () => { lbSpinner.hidden = true; };
          lbSpecImg.onerror = () => { lbSpinner.hidden = true; };
          lbSpecImg.src = card.dataset.full;         // the spectrogram image
          lbPlayhead.style.transform = "translateX(0)";
          lbAudio.src = card.dataset.audio || "";
        } else {
          lbSpec.hidden = true;
          lbAudio.hidden = true;
          lbAudio.removeAttribute("src");
          lbImg.hidden = false;
          lbSpinner.hidden = false;
          lbImg.classList.remove("ready");
          lbImg.onload = () => { lbSpinner.hidden = true; lbImg.classList.add("ready"); };
          lbImg.onerror = () => { lbSpinner.hidden = true; };
          lbImg.src = card.dataset.full;
        }

        lbCap.innerHTML =
          '<strong></strong><span class="lb-conf"></span>' +
          '<span class="lb-cam"></span><span class="lb-ts"></span>';
        lbCap.querySelector("strong").textContent = label;
        lbCap.querySelector(".lb-conf").textContent = conf;
        lbCap.querySelector(".lb-cam").textContent = cam;
        lbCap.querySelector(".lb-ts").textContent = ts;

        lb.hidden = false;
        document.body.classList.add("noscroll");
      }
```

In `closeLightbox`, stop playback + the playhead (add before `lb.hidden = true;`):

```javascript
        stopPlayhead();
        lbAudio.pause();
        lbAudio.removeAttribute("src");
```

- [ ] **Step 4: Add the CSS** (append to `src/wildlife/gallery/static/style.css`):

```css
.kind-badge { margin-left: auto; opacity: .8; }
.lb-spec { position: relative; display: inline-block; max-width: 100%; }
.lb-spec img { display: block; max-width: 100%; height: auto; image-rendering: pixelated; }
.lb-spec .playhead {
  position: absolute; top: 0; bottom: 0; left: 0; width: 1px;
  background: #fff; box-shadow: 0 0 3px #fff; will-change: transform; pointer-events: none;
}
#lbAudio { width: 100%; margin-top: .5rem; }
```

- [ ] **Step 5: Add an automated template-render test + run it**

Append to `tests/test_gallery_audio.py` (reuses the `_app` fixture defined there in Task 8):

```python
def test_index_html_renders_audio_card(tmp_path):
    app, cid = _app(tmp_path)
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'data-kind="audio"' in html
    assert f'data-audio="/audio/{cid}"' in html
    assert 'name="source_kind"' in html  # the Kind filter control is present
```

Run: `.venv/bin/python -m pytest tests/test_gallery_audio.py -q`
Expected: PASS (4 passed — the 3 from Task 8 + this one).

- [ ] **Step 6: Manual eyeball + commit**

Seed an audio capture and load the gallery (if a browser is handy):

```bash
.venv/bin/python - <<'PY'
import numpy as np
from datetime import datetime
from wildlife.store import Store
s = Store("/tmp/wa.db", "/tmp/wa_caps"); s.init_schema()
s.save_audio_capture(camera_id="cam1", event_ts=datetime.now(), capture_ts=datetime.now(),
    species="American Robin", confidence=0.8,
    spectrogram_rgb=(np.random.rand(64,200,3)*255).astype("uint8"), clip_bytes=None)
s.close(); print("seeded")
PY
```

Expected on load: the audio row shows a 🎵 badge; the Kind filter can isolate audio rows; clicking a row opens the spectrogram with an audio player (playback + playhead work when a real clip is present). Then:

```bash
.venv/bin/python -m pytest -q
git add src/wildlife/gallery/templates/index.html src/wildlife/gallery/static/style.css tests/test_gallery_audio.py
git commit -m "feat(gallery): playable-spectrogram lightbox + kind filter for audio"
```

---

### Task 10: Docs — example config + README + deploy notes

**Files:**
- Modify: `config.example.yaml`
- Modify: `README.md`

- [ ] **Step 1: Add the `audio` block to `config.example.yaml`**

Append after the `continuous:` block:

```yaml
audio: # optional BirdNET audio bird-ID (CPU-side; needs the go2rtc daemon + camera audio)
  enabled: false # set true to run a per-camera BirdNET analyzer
  stream: sub # sub | main — which go2rtc restream carries the mic (some Reolink only on main)
  latitude: 37.2 # rounded placeholder — put your real coordinates in config.yaml
  longitude: -107.5
  use_geo_filter: true # restrict to species plausible at your location/season
  confidence_threshold: 0.25 # BirdNET score gate; raise to cut wind/noise
  bandpass_fmin: 0 # Hz; raise (e.g. 300) to band-limit low-frequency wind
  min_confirmations: 2 # same species N times within confirm_window_s before saving
  confirm_window_s: 15
  cooldown_s: 30 # per-species suppression after a save
  active_hours: "" # optional "HH:MM-HH:MM" local; empty = 24/7
```

- [ ] **Step 2: Add an "Audio bird-ID" section to `README.md`**

Add after the "Continuous (motion-gated) detection" section:

```markdown
## Audio bird-ID (optional)

A second detection modality: a per-camera analyzer identifies birds by song using
[BirdNET](https://github.com/birdnet-team/birdnet), reading the camera's audio off the
same go2rtc restream. It runs **CPU-side** (no contention with YOLO on the GPU) and saves
each confirmed detection as a **playable spectrogram** — a spectrogram thumbnail in the
gallery grid; click it for the spectrogram plus an audio player with a synced playhead.

**Install the extra** (heavy — pulls TensorFlow):

    uv pip install -e ".[audio]"

> `birdnet` brings **full TensorFlow (~1 GB)** plus scipy/pandas/pyarrow/soundfile/kagglehub.
> It is CPU-only. On first run it downloads model weights from Kaggle Hub (needs network +
> a writable cache once). On the 8 GB mini, watch memory alongside torch/YOLO.

**Requires the go2rtc daemon** (like continuous detection) and a camera **audio track**.
Some Reolink models only carry audio on the *main* stream — confirm with
`ffprobe rtsp://127.0.0.1:8554/<id>_sub` and set `audio.stream: main` if `_sub` has no audio.
Prefer selecting **AAC/16000** on the camera over G.711 8 kHz (which clips high-frequency
calls).

**Enable it** in `config.yaml` with your real coordinates (they stay in the gitignored
`config.yaml`):

    audio:
      enabled: true
      latitude: 37.228274
      longitude: -107.519089

**Tuning (cut wind/false positives):** `min_confirmations` + `confirm_window_s` are the
dominant lever (wind won't reproduce the *same* species repeatedly); raise
`confidence_threshold`; set `bandpass_fmin` (e.g. 300 Hz) to band-limit low-frequency wind;
`use_geo_filter` trims implausible species by location/season; `active_hours` duty-cycles.
Audio detections are tagged `source_kind = "audio"` in the DB and filterable in the gallery.
```

- [ ] **Step 3: Verify the example config still loads**

Run: `.venv/bin/python -c "from wildlife.config import load_config; c=load_config('config.example.yaml'); print(c.audio.stream, c.audio.enabled)"`
Expected: prints `sub False`.

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml README.md
git commit -m "docs: document BirdNET audio bird-ID"
```

---

## Final verification

- [ ] Full suite: `.venv/bin/python -m pytest -q` → all green (birdnet tests skip via importorskip; the analyzer's fake-model tests run).
- [ ] Lint: `.venv/bin/ruff check .` → clean.
- [ ] Inert-when-disabled: with `audio.enabled: false`, `_start_producers` starts no audio thread and no BirdNET load occurs (covered by `test_audio_sources_start_only_when_enabled`).
- [ ] **Deploy-only checks (prod mini, `[audio]` installed):** `ffprobe` the `_sub` URL to confirm an audio track; confirm `birdnet.load(...).predict_arrays((pcm, 48000)).to_structured_array()` columns and the `GeoPredictionResult` species accessor match §Task-5 (adjust `_build_geo_shortlist` if the geo column name differs); first run downloads weights (network).
