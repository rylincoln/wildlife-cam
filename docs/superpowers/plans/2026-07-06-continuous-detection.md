# Continuous Motion-Gated Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on, motion-gated detection path so the user's fine-tuned YOLO — not Reolink's onboard AI — decides what counts, catching small/distant/nocturnal wildlife the camera misses.

**Architecture:** A second per-camera producer thread runs a cheap MOG2 motion gate on the go2rtc sub-restream and emits `CameraEvent(kind="motion_continuous")` onto the **same** shared queue the existing Reolink producer feeds. The single consumer (`_handle_event` → dedupe → burst → YOLO → gate → save) is reused unchanged except for a minimal `event.kind`-based burst-source selection (route continuous bursts through go2rtc) and a `source_kind` provenance tag.

**Tech Stack:** Python 3, pydantic v2 (config), `cv2`/`numpy` (motion + capture, already in the `detect` extra), SQLite (`store`), threading + `queue.Queue` (worker), pytest with `importorskip` guards to keep the suite hardware-free.

## Global Constraints

- **No new runtime dependency.** `cv2`/`numpy` are already the `detect` extra; go2rtc already runs. Do not add packages.
- **Hardware-free test suite.** Guard any test that imports `cv2`, `wildlife.motion`, `wildlife.capture`, or `wildlife.worker` with `pytest.importorskip(...)`. `config.py` stays pydantic+yaml only (no torch/cv2/numpy); `models.py` stays stdlib-only; `store.py` stays numpy+PIL+stdlib.
- **Two distinct strings — DO NOT SWAP THEM.** The **source kind** (event-source factory key + the value passed to `_produce`) is `"continuous_motion"`, mirroring `"reolink_native"`/`"onvif_bridge"`. The **event kind** (`CameraEvent.kind`, checked by the consumer for burst routing) is `"motion_continuous"`. The event kind is defined once as `EVENT_KIND = "motion_continuous"` in `events/continuous_motion.py` and imported wherever needed — never hand-type it.
- **Inert when `continuous.enabled: false`.** No second producer starts, no continuous burst path is exercised, worker behavior is byte-for-byte unchanged. This is a regression-guarded requirement.
- **Timestamps are naive-local** via `datetime.now()` (matching `worker._now`), never UTC — so `store` ISO strings stay consistent with existing rows.
- **Continuous bursts route through go2rtc**, not a direct Reolink session: `rtsp://127.0.0.1:{rtsp_port}/{camera.id}_{capture.stream}`, where `rtsp_port` is parsed from `livestream.rtsp_listen` (default `":8554"`). Reolink events keep the existing direct path (`rtsp_url=None`).
- **Motion masks are IGNORE polygons**, normalized to `0..1` coordinates.
- **Follow existing conventions:** `from __future__ import annotations`; lazy heavy imports inside event-source `_run` (mirror `reolink_native`'s lazy `reolink-aio` import); Google-style docstrings; keep it ruff-clean (`ruff check .`).

**Default tuning (approved starting points, tuned later on real cameras):** `sample_fps=4`, `downscale_width=480`, `min_area_frac=0.003`, `refractory_s=8`, `warmup_s=10`, `algorithm="mog2"`, `active_hours=""`.

---

## File Structure

- **Create** `src/wildlife/motion.py` — `MotionDetector` + `MotionResult`. Pure per-frame motion decision (cv2/numpy). Holds MOG2 state; no I/O, no temporal logic.
- **Create** `src/wildlife/events/continuous_motion.py` — `ContinuousMotionEventSource` (+ `EVENT_KIND`). Temporal orchestration (rising edge, refractory, warmup, active-hours, scene-change reset) + go2rtc RTSP read. Import-safe (cv2/`MotionDetector` imported lazily in `_run`).
- **Modify** `src/wildlife/events/base.py` — `make_event_source` gains an optional `config` param; register `"continuous_motion"`.
- **Modify** `src/wildlife/capture.py` — `grab_burst` gains `rtsp_url: str | None = None` override.
- **Modify** `src/wildlife/store.py` — `source_kind` provenance column + `save_capture` param.
- **Modify** `src/wildlife/config.py` — `ContinuousConfig` + `CameraConfig.motion_mask`, wired into `Config` + `__all__`.
- **Modify** `src/wildlife/worker.py` — `_produce(camera, kind)`, composite source-registry key, second producer start, burst routing + `source_kind` pass-through in `_handle_event`.
- **Modify** `config.example.yaml`, `README.md` — docs.
- **Create tests** `tests/test_motion.py`, `tests/test_continuous_config.py`, `tests/test_capture_url_override.py`, `tests/test_continuous_source.py`, `tests/test_worker_continuous.py`; **extend** `tests/test_store.py`.

**Dependency order:** Task 1 (motion) → Task 2 (config) → Task 3 (grab_burst) → Task 4 (store) → Task 5 (source, needs config + base) → Task 6 (worker producer-side, needs config + source) → Task 7 (worker consumer-side, needs grab_burst + store) → Task 8 (docs).

---

### Task 1: `MotionDetector` — pure per-frame motion decision

**Files:**
- Create: `src/wildlife/motion.py`
- Test: `tests/test_motion.py`

**Interfaces:**
- Consumes: nothing (leaf module; `cv2`/`numpy` only).
- Produces:
  - `MotionResult(motion: bool, scene_change: bool, area_frac: float)` — frozen dataclass.
  - `MotionDetector(downscale_width: int, min_area_frac: float, algorithm: str = "mog2", mask_polys: list[list[tuple[float, float]]] | None = None, scene_change_thresh: float = 40.0, history: int = 500, var_threshold: int = 16)` with `update(frame_bgr: np.ndarray) -> MotionResult` and `reset() -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_motion.py`:

```python
"""Tests for the pure MotionDetector (cv2/numpy; skipped when cv2 is absent)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from wildlife.motion import MotionDetector, MotionResult  # noqa: E402


def _blank(h: int = 180, w: int = 320, value: int = 0) -> np.ndarray:
    """A solid-gray BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _with_blob(frac_w: float = 0.4, frac_h: float = 0.4) -> np.ndarray:
    """A dark frame with a bright rectangle covering ~frac_w*frac_h of the area."""
    frame = _blank()
    h, w, _ = frame.shape
    bw, bh = int(w * frac_w), int(h * frac_h)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    frame[y0 : y0 + bh, x0 : x0 + bw] = 255
    return frame


def test_moving_blob_after_warmin_is_motion():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01)
    # Warm the background model on a static scene.
    for _ in range(25):
        det.update(_blank())
    result = det.update(_with_blob())
    assert isinstance(result, MotionResult)
    assert result.motion is True
    assert result.area_frac >= 0.01


def test_speckle_noise_is_not_motion():
    rng = np.random.default_rng(0)
    det = MotionDetector(downscale_width=160, min_area_frac=0.02)
    for _ in range(25):
        det.update(_blank())
    # A frame with sparse single-pixel speckle: morphology + area gate reject it.
    noisy = _blank()
    ys = rng.integers(0, noisy.shape[0], size=40)
    xs = rng.integers(0, noisy.shape[1], size=40)
    noisy[ys, xs] = 255
    result = det.update(noisy)
    assert result.motion is False


def test_blob_inside_ignore_mask_is_not_motion():
    # Mask out the entire frame as an ignore region.
    full = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, mask_polys=full)
    for _ in range(25):
        det.update(_blank())
    result = det.update(_with_blob())
    assert result.motion is False


def test_whole_frame_brightness_jump_is_scene_change():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, scene_change_thresh=40.0)
    det.update(_blank(value=0))
    result = det.update(_blank(value=200))  # IR-cut-style flip
    assert result.scene_change is True


def test_frame_diff_algorithm_detects_blob():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01, algorithm="frame_diff")
    for _ in range(5):
        det.update(_blank())
    result = det.update(_with_blob())
    assert result.motion is True


def test_reset_rebuilds_state_without_error():
    det = MotionDetector(downscale_width=160, min_area_frac=0.01)
    det.update(_blank())
    det.reset()
    # After reset the detector still works.
    result = det.update(_with_blob())
    assert isinstance(result, MotionResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_motion.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wildlife.motion'`.

- [ ] **Step 3: Write the implementation**

Create `src/wildlife/motion.py`:

```python
"""Pure per-frame motion decision for continuous, motion-gated detection.

This module mirrors ``capture.py``'s hardware confinement: only ``cv2``/``numpy``
(plus stdlib) are imported, so the heavyweight decode/vision dependency stays out
of the light, hardware-free library modules. It holds no I/O and no temporal
logic — a :class:`MotionDetector` answers one question per frame ("does *this*
frame show motion, and did the scene just change wholesale?"). The rising-edge /
refractory / warmup orchestration lives in the event source that drives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["MotionDetector", "MotionResult"]

# Morphology kernel used to erode-then-dilate the foreground mask, killing
# single-pixel speckle before contour analysis.
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Foreground binarisation threshold for the frame_diff algorithm.
_FRAME_DIFF_THRESH = 25
# Exponential-moving-average weight for the frame_diff reference frame.
_FRAME_DIFF_REF_ALPHA = 0.1


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Outcome of one :meth:`MotionDetector.update` call.

    Attributes:
        motion: True if the largest motion contour meets the area threshold.
        scene_change: True on a whole-frame change (PTZ / IR-cut flip / exposure
            jump) — the caller should ``reset()`` and swallow the transient.
        area_frac: Largest motion contour area as a fraction of the (downscaled)
            frame area; surfaced for tuning/logging.
    """

    motion: bool
    scene_change: bool
    area_frac: float


class MotionDetector:
    """Decide per-frame motion via MOG2 background subtraction (or frame diff).

    Parameters
    ----------
    downscale_width:
        Width (px) the motion computation runs at; taller frames are shrunk to
        this width (aspect preserved) so the vision work stays cheap.
    min_area_frac:
        The largest motion contour must cover at least this fraction of the
        downscaled frame to count as motion.
    algorithm:
        ``"mog2"`` (default; adapts to swaying vegetation / gradual light) or
        ``"frame_diff"`` (lighter absdiff-vs-rolling-reference fallback).
    mask_polys:
        Optional ignore regions as normalised ``0..1`` polygons; motion inside
        them is discarded (roads / canopy / flags / water).
    scene_change_thresh:
        Mean absolute (0..255) whole-frame diff above which ``scene_change`` trips.
    history, var_threshold:
        MOG2 tuning (``detectShadows`` is always False).
    """

    def __init__(
        self,
        downscale_width: int,
        min_area_frac: float,
        algorithm: str = "mog2",
        mask_polys: list[list[tuple[float, float]]] | None = None,
        scene_change_thresh: float = 40.0,
        history: int = 500,
        var_threshold: int = 16,
    ) -> None:
        self._downscale_width = int(downscale_width)
        self._min_area_frac = float(min_area_frac)
        self._algorithm = algorithm
        self._scene_change_thresh = float(scene_change_thresh)
        self._history = int(history)
        self._var_threshold = int(var_threshold)
        self._mask_polys = mask_polys or []

        self._prev_gray: np.ndarray | None = None  # for scene-change diff
        self._ref_gray: np.ndarray | None = None  # frame_diff rolling reference
        self._mask: np.ndarray | None = None  # rasterised keep-mask at work size
        self._subtractor = self._make_subtractor()

    def _make_subtractor(self):
        """Build a fresh MOG2 subtractor (None for the frame_diff algorithm)."""
        if self._algorithm == "mog2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._var_threshold,
                detectShadows=False,
            )
        return None

    def reset(self) -> None:
        """Rebuild the background model and clear cached frames.

        Called by the driving source on a scene change or reconnect so a wholesale
        scene shift does not read as a lasting field of motion.
        """
        self._subtractor = self._make_subtractor()
        self._prev_gray = None
        self._ref_gray = None

    def update(self, frame_bgr: np.ndarray) -> MotionResult:
        """Return the :class:`MotionResult` for one BGR frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self._downscale_width:
            new_h = max(1, round(h * self._downscale_width / w))
            gray = cv2.resize(
                gray, (self._downscale_width, new_h), interpolation=cv2.INTER_AREA
            )

        if self._mask is None and self._mask_polys:
            self._mask = self._rasterize_mask(gray.shape)

        # Whole-frame scene change (catches PTZ / IR-cut flip / exposure jump).
        scene_change = False
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            scene_change = (
                float(cv2.absdiff(gray, self._prev_gray).mean())
                > self._scene_change_thresh
            )
        self._prev_gray = gray

        fg = self._foreground(gray)
        if fg is None:  # frame_diff seeding its reference on the first frame
            return MotionResult(motion=False, scene_change=scene_change, area_frac=0.0)

        if self._mask is not None:
            fg = cv2.bitwise_and(fg, self._mask)
        # Erode-then-dilate: remove speckle without inflating real blobs.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _KERNEL)

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = fg.shape[0] * fg.shape[1]
        largest = max((cv2.contourArea(c) for c in contours), default=0.0)
        area_frac = (largest / frame_area) if frame_area else 0.0
        return MotionResult(
            motion=area_frac >= self._min_area_frac,
            scene_change=scene_change,
            area_frac=area_frac,
        )

    def _foreground(self, gray: np.ndarray) -> np.ndarray | None:
        """Binary foreground mask for ``gray``; None while frame_diff seeds itself."""
        if self._algorithm == "mog2":
            # detectShadows=False -> apply() already yields a 0/255 mask.
            return self._subtractor.apply(gray)
        # frame_diff: absdiff vs a slowly-updated rolling reference.
        if self._ref_gray is None:
            self._ref_gray = gray.copy()
            return None
        diff = cv2.absdiff(gray, self._ref_gray)
        _, fg = cv2.threshold(diff, _FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        self._ref_gray = cv2.addWeighted(
            self._ref_gray, 1.0 - _FRAME_DIFF_REF_ALPHA, gray, _FRAME_DIFF_REF_ALPHA, 0.0
        )
        return fg

    def _rasterize_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Rasterise ignore polygons into a keep-mask (255=keep, 0=ignore)."""
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        for poly in self._mask_polys:
            pts = np.array(
                [[int(round(x * w)), int(round(y * h))] for (x, y) in poly],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 0)
        return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_motion.py -q`
Expected: PASS (6 passed). If `test_moving_blob_after_warmin_is_motion` is flaky, increase the warm-in loop count — synthetic noise-free frames converge quickly but give MOG2 headroom.

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/motion.py tests/test_motion.py
git add src/wildlife/motion.py tests/test_motion.py
git commit -m "feat(motion): add pure MotionDetector for continuous detection"
```

---

### Task 2: `ContinuousConfig` + `CameraConfig.motion_mask`

**Files:**
- Modify: `src/wildlife/config.py`
- Test: `tests/test_continuous_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ContinuousConfig(enabled: bool = False, sample_fps: int = 4, downscale_width: int = 480, min_area_frac: float = 0.003, refractory_s: float = 8.0, warmup_s: float = 10.0, algorithm: Literal["mog2","frame_diff"] = "mog2", active_hours: str = "")`.
  - `CameraConfig.motion_mask: list[list[tuple[float, float]]] | None = None`.
  - `Config.continuous: ContinuousConfig` (default_factory).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_continuous_config.py`:

```python
"""Validation tests for ContinuousConfig and CameraConfig.motion_mask."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wildlife.config import CameraConfig, ContinuousConfig


def _camera(**overrides):
    base = dict(
        id="north_field",
        host="192.168.1.101",
        username="admin",
        password="x",
        rtsp_main="rtsp://{username}:{password}@{host}:554/main",
        rtsp_sub="rtsp://{username}:{password}@{host}:554/sub",
    )
    base.update(overrides)
    return base


def test_continuous_defaults_are_inert():
    cc = ContinuousConfig()
    assert cc.enabled is False
    assert cc.sample_fps == 4
    assert cc.downscale_width == 480
    assert cc.min_area_frac == pytest.approx(0.003)
    assert cc.refractory_s == 8.0
    assert cc.warmup_s == 10.0
    assert cc.algorithm == "mog2"
    assert cc.active_hours == ""


def test_continuous_rejects_bad_values():
    with pytest.raises(ValidationError):
        ContinuousConfig(sample_fps=0)
    with pytest.raises(ValidationError):
        ContinuousConfig(downscale_width=32)
    with pytest.raises(ValidationError):
        ContinuousConfig(min_area_frac=0.0)
    with pytest.raises(ValidationError):
        ContinuousConfig(min_area_frac=1.0)
    with pytest.raises(ValidationError):
        ContinuousConfig(algorithm="optical_flow")


def test_active_hours_accepts_empty_and_valid_windows():
    assert ContinuousConfig(active_hours="").active_hours == ""
    assert ContinuousConfig(active_hours="20:00-06:00").active_hours == "20:00-06:00"
    assert ContinuousConfig(active_hours="06:30-18:45").active_hours == "06:30-18:45"


def test_active_hours_rejects_malformed():
    for bad in ("20:00", "25:00-06:00", "20:61-06:00", "8:00-9:00", "abc"):
        with pytest.raises(ValidationError):
            ContinuousConfig(active_hours=bad)


def test_motion_mask_defaults_none_and_accepts_valid_polygons():
    assert CameraConfig(**_camera()).motion_mask is None
    cam = CameraConfig(
        **_camera(motion_mask=[[[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]])
    )
    assert cam.motion_mask == [[(0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)]]


def test_motion_mask_rejects_short_polygon_and_out_of_range_points():
    with pytest.raises(ValidationError):
        CameraConfig(**_camera(motion_mask=[[[0.0, 0.0], [1.0, 1.0]]]))  # only 2 points
    with pytest.raises(ValidationError):
        CameraConfig(**_camera(motion_mask=[[[0.0, 0.0], [1.0, 0.0], [1.5, 1.0]]]))  # x>1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_continuous_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'ContinuousConfig'`.

- [ ] **Step 3: Implement the config changes**

In `src/wildlife/config.py`, add `"ContinuousConfig"` to `__all__` (after `"RemoteConfig"`).

Add `motion_mask` to `CameraConfig` (after the `onvif_port` field, before `_interpolate_rtsp`):

```python
    motion_mask: list[list[tuple[float, float]]] | None = None

    @field_validator("motion_mask")
    @classmethod
    def _validate_motion_mask(
        cls, value: list[list[tuple[float, float]]] | None
    ) -> list[list[tuple[float, float]]] | None:
        """Each ignore polygon needs >=3 points, all with x,y in [0, 1]."""
        if value is None:
            return None
        for poly in value:
            if len(poly) < 3:
                raise ValueError("each motion_mask polygon needs at least 3 points")
            for x, y in poly:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError("motion_mask points must be normalised to [0, 1]")
        return value
```

Add the `ContinuousConfig` model (place it just before `class Config`):

```python
class ContinuousConfig(BaseModel):
    """Optional always-on, motion-gated detection (your model becomes the gate).

    A second per-camera producer watches a cheap MOG2 motion signal on the go2rtc
    sub restream and fires the existing capture->YOLO->gate->save pipeline on
    motion, so the fine-tuned model (not Reolink's onboard AI) decides what counts.
    Entirely inert when ``enabled`` is false.
    """

    enabled: bool = False
    sample_fps: int = Field(default=4, ge=1)  # frames/sec sampled for motion
    downscale_width: int = Field(default=480, ge=64)  # px width motion runs at
    min_area_frac: float = Field(default=0.003, gt=0.0, lt=1.0)  # largest contour gate
    refractory_s: float = Field(default=8.0, ge=0.0)  # min seconds between emits
    warmup_s: float = Field(default=10.0, ge=0.0)  # suppress emits after (re)connect
    algorithm: Literal["mog2", "frame_diff"] = "mog2"
    active_hours: str = ""  # "HH:MM-HH:MM" local window; empty = 24/7

    @field_validator("active_hours")
    @classmethod
    def _validate_active_hours(cls, value: str) -> str:
        """Accept "" (24/7) or a strict "HH:MM-HH:MM" 24-hour window."""
        value = value.strip()
        if not value:
            return ""
        import re

        m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", value)
        if not m:
            raise ValueError('active_hours must be "" or "HH:MM-HH:MM"')
        sh, sm, eh, em = (int(g) for g in m.groups())
        for hh, mm in ((sh, sm), (eh, em)):
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError("active_hours has an out-of-range time")
        return value
```

Wire it into `Config` (after the `remote:` field):

```python
    continuous: ContinuousConfig = Field(default_factory=ContinuousConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_continuous_config.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Verify existing config still validates (no-op regression)**

Run: `.venv/bin/python -c "from wildlife.config import load_config; c = load_config('config.example.yaml'); print('continuous.enabled =', c.continuous.enabled)"`
Expected: prints `continuous.enabled = False` (a config with no `continuous:` block gets the inert default).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/config.py tests/test_continuous_config.py
git add src/wildlife/config.py tests/test_continuous_config.py
git commit -m "feat(config): add ContinuousConfig and CameraConfig.motion_mask"
```

---

### Task 3: `grab_burst` explicit-URL override

**Files:**
- Modify: `src/wildlife/capture.py:85-91` (signature) + body URL selection
- Test: `tests/test_capture_url_override.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `grab_burst(camera, n, interval_ms, stream, timeout_s, rtsp_url: str | None = None) -> list[np.ndarray]` — when `rtsp_url` is given it is opened instead of `_select_url(camera, stream)`; otherwise byte-for-byte unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capture_url_override.py`:

```python
"""grab_burst honours an explicit rtsp_url override (cv2 monkeypatched)."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from wildlife import capture  # noqa: E402


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in that records the URL and opens 'closed'."""

    opened_urls: list[str] = []

    def __init__(self, url, _backend):
        _FakeCapture.opened_urls.append(url)

    def set(self, *_args):
        return True

    def isOpened(self):
        return False  # short-circuits grab_burst -> returns [] fast

    def release(self):
        pass


class _Camera:
    id = "north_field"
    rtsp_main = "rtsp://cam/main"
    rtsp_sub = "rtsp://cam/sub"


def test_explicit_rtsp_url_is_used(monkeypatch):
    _FakeCapture.opened_urls = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    capture.grab_burst(_Camera(), 3, 100, "main", 5, rtsp_url="rtsp://127.0.0.1:8554/x_main")
    assert _FakeCapture.opened_urls == ["rtsp://127.0.0.1:8554/x_main"]


def test_no_override_falls_back_to_select_url(monkeypatch):
    _FakeCapture.opened_urls = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    capture.grab_burst(_Camera(), 3, 100, "sub", 5)
    assert _FakeCapture.opened_urls == ["rtsp://cam/sub"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_capture_url_override.py -q`
Expected: FAIL — `TypeError: grab_burst() got an unexpected keyword argument 'rtsp_url'`.

- [ ] **Step 3: Implement the override**

In `src/wildlife/capture.py`, change the `grab_burst` signature to add the parameter:

```python
def grab_burst(
    camera: "CameraConfig",
    n: int,
    interval_ms: int,
    stream: str,
    timeout_s: int,
    rtsp_url: str | None = None,
) -> list[np.ndarray]:
```

Add one line to the docstring's Args (after the `stream:` entry):

```
        rtsp_url: Optional explicit RTSP URL to open instead of the camera's
            own main/sub URL — used to route continuous bursts through go2rtc.
```

Change the URL selection line (currently `url = _select_url(camera, stream)`):

```python
    url = rtsp_url if rtsp_url else _select_url(camera, stream)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_capture_url_override.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/capture.py tests/test_capture_url_override.py
git add src/wildlife/capture.py tests/test_capture_url_override.py
git commit -m "feat(capture): grab_burst accepts an explicit rtsp_url override"
```

---

### Task 4: `store.source_kind` provenance column

**Files:**
- Modify: `src/wildlife/store.py` (`_SCHEMA_SQL`, `_COLUMNS`, `_COLUMN_ADDITIONS`, `save_capture`)
- Test: `tests/test_store.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Store.save_capture(..., source_kind: str = "reolink") -> int`; every row dict (from `get`/`query`) gains a `"source_kind"` key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py` (reuse the module's existing frame/Detection/Store helpers; the snippet below constructs its own to be self-contained — adapt to match the file's existing fixtures if present):

```python
def test_source_kind_defaults_to_reolink(tmp_path):
    import numpy as np

    from wildlife.models import Detection
    from wildlife.store import Store

    store = Store(tmp_path / "c.db", tmp_path / "caps")
    store.init_schema()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    det = Detection("bird", 0.9, (0.0, 0.0, 10.0, 10.0), 0.25)
    from datetime import datetime

    cid = store.save_capture(
        camera_id="cam1",
        event_ts=datetime(2026, 7, 6, 12, 0, 0),
        capture_ts=datetime(2026, 7, 6, 12, 0, 1),
        frame=frame,
        det=det,
    )
    assert store.get(cid)["source_kind"] == "reolink"
    store.close()


def test_source_kind_continuous_round_trips(tmp_path):
    import numpy as np
    from datetime import datetime

    from wildlife.models import Detection
    from wildlife.store import Store

    store = Store(tmp_path / "c.db", tmp_path / "caps")
    store.init_schema()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    det = Detection("deer", 0.8, (0.0, 0.0, 10.0, 10.0), 0.25)
    cid = store.save_capture(
        camera_id="cam1",
        event_ts=datetime(2026, 7, 6, 12, 0, 0),
        capture_ts=datetime(2026, 7, 6, 12, 0, 1),
        frame=frame,
        det=det,
        source_kind="continuous",
    )
    assert store.get(cid)["source_kind"] == "continuous"
    store.close()


def test_source_kind_migrates_onto_a_legacy_db(tmp_path):
    """A pre-existing captures table without source_kind gains it, defaulting reolink."""
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
    conn.execute(
        "INSERT INTO captures (camera_id, event_ts, capture_ts, label, confidence, "
        "image_path, thumb_path) VALUES ('cam1','t','t','bird',0.9,'a.jpg','a_thumb.jpg')"
    )
    conn.commit()
    conn.close()

    store = Store(db, tmp_path / "caps")
    store.init_schema()  # runs _migrate
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(captures)")}
    assert "source_kind" in cols
    rows = store.query(camera_id="cam1")
    assert rows[0]["source_kind"] == "reolink"
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_store.py -q -k source_kind`
Expected: FAIL — `KeyError: 'source_kind'` (column not in `_COLUMNS`) / `TypeError` on the `source_kind=` kwarg.

- [ ] **Step 3: Implement the column**

In `src/wildlife/store.py`:

Add the column to the CREATE TABLE in `_SCHEMA_SQL` (after the `reviewed_at` line, before the closing `);`):

```sql
    reviewed_at    TEXT,              -- ISO8601 of the last human action
    source_kind    TEXT NOT NULL DEFAULT 'reolink'  -- provenance: reolink | continuous
```

Add `"source_kind"` to `_COLUMNS` (append after `"reviewed_at"`).

Add the migration entry to `_COLUMN_ADDITIONS` (append after the `reviewed_at` tuple):

```python
    ("source_kind", "TEXT NOT NULL DEFAULT 'reolink'"),
```

Add the `source_kind` parameter to `save_capture` (after `det: Detection,`):

```python
        source_kind: str = "reolink",
```

Update the INSERT to persist it — change the column list, placeholders, and values tuple:

```python
            cur = self._conn.execute(
                """
                INSERT INTO captures (
                    camera_id, event_ts, capture_ts, label, confidence,
                    box_x1, box_y1, box_x2, box_y2,
                    image_path, thumb_path, width, height, source_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    _iso(event_ts),
                    _iso(capture_ts),
                    det.label,
                    float(det.confidence),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    image_rel,
                    thumb_rel,
                    width,
                    height,
                    source_kind,
                ),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: PASS (all store tests, including the 3 new ones).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/store.py tests/test_store.py
git add src/wildlife/store.py tests/test_store.py
git commit -m "feat(store): add source_kind provenance column"
```

---

### Task 5: `ContinuousMotionEventSource` + register in factory

**Files:**
- Create: `src/wildlife/events/continuous_motion.py`
- Modify: `src/wildlife/events/base.py` (`make_event_source` signature + dispatch)
- Test: `tests/test_continuous_source.py`

**Interfaces:**
- Consumes: `_QueueBackedEventSource` + `CameraEvent` from `events/base`; `MotionDetector`/`MotionResult` from `motion` (lazy); `ContinuousConfig` fields + `CameraConfig.motion_mask` + `LivestreamConfig.rtsp_listen` off the passed `config`; `capture._apply_timeout` (lazy).
- Produces:
  - `EVENT_KIND = "motion_continuous"` (module constant).
  - `ContinuousMotionEventSource(camera, config)` — subclass of `_QueueBackedEventSource`; testable pure methods `_consider(result, now_mono: float, now_wall: datetime) -> str | None` (returns `"emit"`, `"reset"`, or `None`) and `_within_active_hours(now_wall: datetime) -> bool`.
  - `make_event_source(kind, camera, config=None)` now accepts and dispatches `"continuous_motion"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_continuous_source.py` (no cv2 needed — exercises the pure temporal logic via a duck-typed config/camera):

```python
"""Temporal-logic tests for ContinuousMotionEventSource (hardware-free)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from wildlife.config import ContinuousConfig
from wildlife.events.continuous_motion import EVENT_KIND, ContinuousMotionEventSource


def _result(motion=False, scene_change=False, area_frac=0.0):
    return SimpleNamespace(motion=motion, scene_change=scene_change, area_frac=area_frac)


def _source(**cc_overrides):
    cc = ContinuousConfig(**cc_overrides)
    config = SimpleNamespace(
        continuous=cc, livestream=SimpleNamespace(rtsp_listen=":8554")
    )
    camera = SimpleNamespace(id="cam1", motion_mask=None)
    return ContinuousMotionEventSource(camera, config)


def test_event_kind_constant():
    assert EVENT_KIND == "motion_continuous"


def test_rising_edge_emits_once():
    src = _source(warmup_s=0, refractory_s=0)
    assert src._consider(_result(motion=False), 0.0, datetime(2026, 7, 6, 12)) is None
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) == "emit"
    # sustained motion (no new rising edge) does not re-emit
    assert src._consider(_result(motion=True), 2.0, datetime(2026, 7, 6, 12)) is None


def test_warmup_suppresses_emit():
    src = _source(warmup_s=10, refractory_s=0)
    src._warmup_until_mono = 100.0
    assert src._consider(_result(motion=True), 50.0, datetime(2026, 7, 6, 12)) is None
    # after warmup, sustained motion is not a rising edge -> a fresh edge is needed
    src._consider(_result(motion=False), 101.0, datetime(2026, 7, 6, 12))
    assert src._consider(_result(motion=True), 102.0, datetime(2026, 7, 6, 12)) == "emit"


def test_refractory_suppresses_second_emit():
    src = _source(warmup_s=0, refractory_s=8)
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) == "emit"
    src._consider(_result(motion=False), 2.0, datetime(2026, 7, 6, 12))
    # rising edge within refractory window -> suppressed
    assert src._consider(_result(motion=True), 5.0, datetime(2026, 7, 6, 12)) is None
    src._consider(_result(motion=False), 6.0, datetime(2026, 7, 6, 12))
    # rising edge after refractory window -> emits
    assert src._consider(_result(motion=True), 12.0, datetime(2026, 7, 6, 12)) == "emit"


def test_scene_change_requests_reset_and_suppresses():
    src = _source(warmup_s=0, refractory_s=8)
    action = src._consider(_result(motion=True, scene_change=True), 1.0, datetime(2026, 7, 6, 12))
    assert action == "reset"
    # the reset also arms refractory, so an immediate rising edge is suppressed
    assert src._consider(_result(motion=True), 2.0, datetime(2026, 7, 6, 12)) is None


def test_active_hours_gate_blocks_emit_outside_window():
    src = _source(warmup_s=0, refractory_s=0, active_hours="20:00-06:00")
    # 12:00 is outside the 20:00-06:00 window -> no emit even on a rising edge
    assert src._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 12)) is None
    # 22:00 is inside the window -> emits
    src2 = _source(warmup_s=0, refractory_s=0, active_hours="20:00-06:00")
    assert src2._consider(_result(motion=True), 1.0, datetime(2026, 7, 6, 22)) == "emit"


def test_within_active_hours_wraps_midnight():
    src = _source(active_hours="20:00-06:00")
    assert src._within_active_hours(datetime(2026, 7, 6, 23)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 3)) is True
    assert src._within_active_hours(datetime(2026, 7, 6, 12)) is False


def test_make_event_source_dispatches_continuous():
    from wildlife.events.base import make_event_source

    config = SimpleNamespace(
        continuous=ContinuousConfig(), livestream=SimpleNamespace(rtsp_listen=":8554")
    )
    camera = SimpleNamespace(id="cam1", motion_mask=None)
    src = make_event_source("continuous_motion", camera, config)
    assert isinstance(src, ContinuousMotionEventSource)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_continuous_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wildlife.events.continuous_motion'`.

- [ ] **Step 3: Implement the event source**

Create `src/wildlife/events/continuous_motion.py`:

```python
"""Continuous, motion-gated event source.

A second per-camera producer that turns the fine-tuned Mac-side YOLO into the
always-on arbiter. It reads the go2rtc **sub** restream, runs a cheap
:class:`~wildlife.motion.MotionDetector` on sampled frames, and emits a
``CameraEvent(kind="motion_continuous")`` on a motion rising edge — which the
worker's existing consumer turns into a capture->YOLO->gate->save pass.

Threading/queue/backoff machinery is inherited verbatim from
:class:`~wildlife.events.base._QueueBackedEventSource` (exactly as
``ReolinkEventSource`` does); only :meth:`_run` is implemented here. The temporal
decision (rising edge, refractory, warmup, active-hours, scene-change reset) is
factored into the pure :meth:`_consider` so it is unit-testable without a camera.

``cv2``, :class:`~wildlife.motion.MotionDetector`, and ``capture._apply_timeout``
are imported lazily inside :meth:`_run` so importing this module never requires
the ``detect`` extra — keeping the temporal tests hardware-free.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime

from wildlife.events.base import CameraEvent, _QueueBackedEventSource

__all__ = ["ContinuousMotionEventSource", "EVENT_KIND"]

logger = logging.getLogger(__name__)

# CameraEvent.kind emitted by this source; the worker routes bursts by matching
# this exact value. (Distinct from the *source* kind "continuous_motion".)
EVENT_KIND = "motion_continuous"

# RTSP open/read timeout (ms) for the persistent motion reader; a wedged read
# fails within this window so the outer loop reopens (acts as the watchdog).
_RTSP_TIMEOUT_MS = 10_000
# Reconnect backoff bounds when the restream will not open / ends.
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def _parse_port(listen: str) -> int:
    """Extract the port from a go2rtc bind string like ``":8554"`` / ``"0.0.0.0:8554"``."""
    return int(listen.rsplit(":", 1)[-1])


def _parse_active_hours(spec: str) -> tuple[int, int] | None:
    """Parse ``"HH:MM-HH:MM"`` into (start_minute, end_minute); ``""`` -> None (24/7)."""
    spec = spec.strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", spec)
    if not m:  # config validation should already guarantee the format
        return None
    sh, sm, eh, em = (int(g) for g in m.groups())
    return (sh * 60 + sm, eh * 60 + em)


class ContinuousMotionEventSource(_QueueBackedEventSource):
    """Emit ``motion_continuous`` events from a go2rtc sub restream's motion.

    Parameters
    ----------
    camera:
        The camera's :class:`wildlife.config.CameraConfig` (needs ``id`` and,
        optionally, ``motion_mask``).
    config:
        The full validated :class:`wildlife.config.Config` — this source reads
        ``config.continuous`` (its knobs), ``config.livestream.rtsp_listen`` (the
        restream port), and ``camera.motion_mask`` (ignore polygons).
    """

    def __init__(self, camera: object, config: object) -> None:
        super().__init__(camera)
        cc = config.continuous
        self._sample_interval_s = 1.0 / max(1, cc.sample_fps)
        self._downscale_width = cc.downscale_width
        self._min_area_frac = cc.min_area_frac
        self._algorithm = cc.algorithm
        self._refractory_s = cc.refractory_s
        self._warmup_s = cc.warmup_s
        self._mask_polys = getattr(camera, "motion_mask", None) or []
        self._rtsp_port = _parse_port(config.livestream.rtsp_listen)
        self._active_window = _parse_active_hours(cc.active_hours)

        # Temporal state (monotonic clock for warmup/refractory).
        self._active = False  # previous frame's motion state (for rising edge)
        self._last_emit_mono: float | None = None
        self._warmup_until_mono = 0.0
        self._detector = None  # built in _run() (needs cv2)

    # -- pure decision logic (unit-tested without a camera) ---------------

    def _within_active_hours(self, now_wall: datetime) -> bool:
        """True if ``now_wall`` falls in the configured window (or none is set)."""
        if self._active_window is None:
            return True
        start, end = self._active_window
        cur = now_wall.hour * 60 + now_wall.minute
        if start <= end:
            return start <= cur < end
        return cur >= start or cur < end  # window wraps past midnight

    def _consider(self, result, now_mono: float, now_wall: datetime) -> str | None:
        """Fold one motion result into the temporal state; return the action.

        Returns ``"emit"`` (fire a CameraEvent), ``"reset"`` (rebuild the motion
        model after a scene change), or ``None`` (do nothing). ``result`` needs
        ``.motion`` and ``.scene_change`` attributes.
        """
        if not self._within_active_hours(now_wall):
            self._active = bool(result.motion)  # track edge; never emit off-hours
            return None

        if result.scene_change:
            # Whole-scene shift: rebuild the model and start a fresh refractory so
            # the transient does not read as a lasting field of motion.
            self._active = False
            self._last_emit_mono = now_mono
            return "reset"

        motion = bool(result.motion)
        rising = motion and not self._active
        self._active = motion
        if not rising:
            return None
        if now_mono < self._warmup_until_mono:
            return None  # model still stabilising after (re)connect
        if (
            self._last_emit_mono is not None
            and (now_mono - self._last_emit_mono) < self._refractory_s
        ):
            return None  # one animal -> one event
        self._last_emit_mono = now_mono
        return "emit"

    def _arm_warmup(self) -> None:
        """Suppress emits for ``warmup_s`` and clear the edge after (re)connect."""
        self._warmup_until_mono = time.monotonic() + self._warmup_s
        self._active = False

    # -- worker-thread run loop (hardware; lazy heavy imports) ------------

    def _run(self) -> None:
        """Open the sub restream, sample motion, and emit rising-edge events."""
        import cv2  # lazy: confine the heavy decode dependency to runtime

        from wildlife.motion import MotionDetector

        url = f"rtsp://127.0.0.1:{self._rtsp_port}/{self.camera.id}_sub"
        backoff = _INITIAL_BACKOFF_S
        while not self._stopping:
            cap = self._open_capture(cv2, url)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                logger.warning(
                    "Camera %s: continuous restream will not open (%s); retry in %.1fs.",
                    self.camera.id,
                    url,
                    backoff,
                )
                self._sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue

            backoff = _INITIAL_BACKOFF_S
            self._detector = MotionDetector(
                self._downscale_width,
                self._min_area_frac,
                self._algorithm,
                self._mask_polys,
            )
            self._arm_warmup()
            logger.info("Camera %s: continuous motion source reading %s.", self.camera.id, url)
            try:
                self._read_loop(cap)
            except Exception:  # noqa: BLE001 - reopen rather than die
                logger.exception("Camera %s: continuous read loop error.", self.camera.id)
            finally:
                cap.release()

    def _open_capture(self, cv2, url: str):
        """Open a TCP-forced, single-buffered FFmpeg capture with open/read timeouts."""
        from wildlife.capture import _apply_timeout

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        _apply_timeout(cap, _RTSP_TIMEOUT_MS // 1000)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - backend dependent
            pass
        return cap

    def _read_loop(self, cap) -> None:
        """Grab frames (paced by the stream), decode at sample_fps, emit on motion."""
        last_sample = 0.0
        while not self._stopping:
            # grab() blocks on the socket -> the loop is paced by the stream, not a
            # busy spin; a wedged read fails the RTSP timeout and ends the loop.
            if not cap.grab():
                return  # stream ended -> outer loop reopens
            now = time.monotonic()
            if (now - last_sample) < self._sample_interval_s:
                continue  # drain to the newest frame without decoding
            last_sample = now
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            result = self._detector.update(frame)
            action = self._consider(result, time.monotonic(), datetime.now())
            if action == "reset":
                self._detector.reset()
                self._arm_warmup()
            elif action == "emit":
                self._emit(
                    CameraEvent(
                        camera_id=self.camera.id,
                        event_ts=datetime.now(),
                        kind=EVENT_KIND,
                    )
                )
                logger.info(
                    "Camera %s: continuous motion event (area_frac=%.4f).",
                    self.camera.id,
                    result.area_frac,
                )
```

In `src/wildlife/events/base.py`, update `make_event_source` — new signature, docstring line, dispatch, and error message:

```python
def make_event_source(kind: str, camera: object, config: object = None) -> EventSource:
```

Add to the docstring's Args (after the `camera:` entry):

```
        config: The full validated ``wildlife.config.Config``. Required by the
            ``continuous_motion`` source (its knobs, restream port, and motion
            mask); ignored by reolink/onvif. Defaults to ``None`` so existing
            callers keep working.
```

Add the dispatch branch (after the `onvif_bridge` branch, before the `raise`):

```python
    if kind == "continuous_motion":
        from wildlife.events.continuous_motion import ContinuousMotionEventSource

        return ContinuousMotionEventSource(camera, config)
```

Update the `raise ValueError` message:

```python
    raise ValueError(
        f"Unknown event_source {kind!r}; expected 'reolink_native', "
        "'onvif_bridge', or 'continuous_motion'"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_continuous_source.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/events/continuous_motion.py src/wildlife/events/base.py tests/test_continuous_source.py
git add src/wildlife/events/continuous_motion.py src/wildlife/events/base.py tests/test_continuous_source.py
git commit -m "feat(events): add ContinuousMotionEventSource + register in factory"
```

---

### Task 6: Worker producer-side — parametrized `_produce`, composite key, second producer

**Files:**
- Modify: `src/wildlife/worker.py` (`_produce`, `_start_producers`)
- Test: `tests/test_worker_continuous.py` (create)

**Interfaces:**
- Consumes: `make_event_source(kind, camera, config)` (Task 5); `Config.continuous.enabled` (Task 2).
- Produces: `_produce(self, camera, kind: str)`; source registry keyed on `f"{camera.id}:{kind}"`; a second `"continuous_motion"` producer thread per camera when `continuous.enabled`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_continuous.py`:

```python
"""Worker dual-producer wiring (guarded: worker pulls in torch/cv2)."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("wildlife.worker")

from wildlife.config import Config  # noqa: E402
from wildlife.worker import _Worker  # noqa: E402


def _config_dict(continuous_enabled: bool):
    return {
        "cameras": [
            {
                "id": "cam1",
                "host": "192.168.1.101",
                "username": "admin",
                "password": "x",
                "rtsp_main": "rtsp://{username}:{password}@{host}:554/main",
                "rtsp_sub": "rtsp://{username}:{password}@{host}:554/sub",
            }
        ],
        "event_source": "reolink_native",
        "capture": {
            "burst_frames": 3,
            "burst_interval_ms": 100,
            "stream": "main",
            "rtsp_timeout_s": 5,
            "max_concurrent": 1,
        },
        "detection": {
            "model_path": "models/yolov8s.pt",
            "device": "cpu",
            "animal_classes": ["bird"],
            "confidence_threshold": 0.5,
            "min_box_area_frac": 0.01,
            "save_best_only": True,
        },
        "dedupe": {"cooldown_s": 0},
        "storage": {"captures_dir": "/tmp/wc_caps", "db_path": "/tmp/wc.db"},
        "retention": {"max_age_days": 30},
        "gallery": {"host": "0.0.0.0", "port": 8080, "page_size": 60},
        "resource_guard": {"detect_every_nth_event": 1, "max_burst_per_minute": 20},
        "continuous": {"enabled": continuous_enabled},
    }


def _make_worker(continuous_enabled: bool) -> _Worker:
    return _Worker(Config.model_validate(_config_dict(continuous_enabled)))


class _FakeSource:
    """Stops the producer loop after one iteration; records close()."""

    def __init__(self, shutdown: threading.Event) -> None:
        self._sd = shutdown
        self.closed = False

    def stream(self):
        self._sd.set()  # make _produce exit after this pass
        return iter(())

    def close(self):
        self.closed = True


def test_produce_uses_composite_source_key(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    camera = worker._cameras["cam1"]
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(worker._shutdown),
    )
    worker._produce(camera, "reolink_native")
    worker._shutdown.clear()
    worker._produce(camera, "continuous_motion")
    assert set(worker._sources) == {"cam1:reolink_native", "cam1:continuous_motion"}


def test_teardown_closes_both_sources():
    worker = _make_worker(continuous_enabled=True)
    a, b = _FakeSource(worker._shutdown), _FakeSource(worker._shutdown)
    worker._sources = {"cam1:reolink_native": a, "cam1:continuous_motion": b}
    worker._teardown()
    assert a.closed and b.closed


def _stop_and_join(worker: _Worker) -> None:
    """Let the (daemon) producer threads exit and join them, so nothing leaks."""
    worker._shutdown.set()
    for t in worker._producer_threads:
        t.join(timeout=1.0)


def test_second_producer_starts_only_when_enabled(monkeypatch):
    # Each fake is bound to ITS worker's _shutdown so _produce exits after one
    # pass — otherwise the daemon threads spin and, once monkeypatch reverts,
    # would call the real make_event_source and open real RTSP sessions.
    on = _make_worker(continuous_enabled=True)
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(on._shutdown),
    )
    on._start_producers()
    assert len(on._producer_threads) == 2  # reolink + continuous
    _stop_and_join(on)

    off = _make_worker(continuous_enabled=False)
    monkeypatch.setattr(
        "wildlife.worker.make_event_source",
        lambda kind, cam, config=None: _FakeSource(off._shutdown),
    )
    off._start_producers()
    assert len(off._producer_threads) == 1  # reolink only
    _stop_and_join(off)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_worker_continuous.py -q`
Expected: FAIL — `test_produce_uses_composite_source_key` (key is `"cam1"`, not composite) and `test_second_producer_starts_only_when_enabled` (only 1 thread) fail.

- [ ] **Step 3: Implement the producer-side changes**

In `src/wildlife/worker.py`, change `_produce` to take `kind` and use a composite registry key. Replace the signature and the two affected lines:

```python
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
```

And change the registry write inside that method (currently `self._sources[camera.id] = source`):

```python
            with self._sources_lock:
                self._sources[source_key] = source
```

Update `_start_producers` to pass the kind and start the optional second producer:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_worker_continuous.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/wildlife/worker.py tests/test_worker_continuous.py
git add src/wildlife/worker.py tests/test_worker_continuous.py
git commit -m "feat(worker): dual producers with composite source key"
```

---

### Task 7: Worker consumer-side — burst routing + `source_kind` pass-through

**Files:**
- Modify: `src/wildlife/worker.py` (`_handle_event`)
- Test: `tests/test_worker_continuous.py` (extend)

**Interfaces:**
- Consumes: `grab_burst(..., rtsp_url=...)` (Task 3); `Store.save_capture(..., source_kind=...)` (Task 4); `EVENT_KIND` from `events/continuous_motion`; `LivestreamConfig.rtsp_listen`.
- Produces: continuous events (`event.kind == EVENT_KIND`) grab their burst from `rtsp://127.0.0.1:{rtsp_port}/{camera.id}_{capture.stream}` and persist with `source_kind="continuous"`; all other events keep `rtsp_url=None` and `source_kind="reolink"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker_continuous.py`:

```python
from datetime import datetime  # noqa: E402

from wildlife.events.continuous_motion import EVENT_KIND  # noqa: E402
from wildlife.gate import Deduper  # noqa: E402
from wildlife.models import CameraEvent  # noqa: E402


def _prime_consumer(worker: _Worker) -> None:
    """Give _handle_event the non-None collaborators it asserts on."""
    worker._detector = object()
    worker._store = object()
    worker._deduper = Deduper(0, 10_000)  # always processes


def test_continuous_event_routes_burst_through_go2rtc(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    _prime_consumer(worker)
    captured = {}

    def _fake_grab(camera, n, interval, stream, timeout, rtsp_url=None):
        captured["url"] = rtsp_url
        return []  # empty -> _handle_event returns before touching detector/store

    monkeypatch.setattr("wildlife.worker.grab_burst", _fake_grab)
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind=EVENT_KIND)
    )
    assert captured["url"] == "rtsp://127.0.0.1:8554/cam1_main"


def test_reolink_event_uses_direct_burst(monkeypatch):
    worker = _make_worker(continuous_enabled=True)
    _prime_consumer(worker)
    captured = {}

    def _fake_grab(camera, n, interval, stream, timeout, rtsp_url=None):
        captured["url"] = rtsp_url
        return []

    monkeypatch.setattr("wildlife.worker.grab_burst", _fake_grab)
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind="animal")
    )
    assert captured["url"] is None


def test_continuous_capture_persists_source_kind(monkeypatch, tmp_path):
    import numpy as np

    from wildlife.models import Detection
    from wildlife.store import Store

    class _FakeDetector:
        def infer(self, _frame):
            return [Detection("bird", 0.95, (0.0, 0.0, 40.0, 40.0), 0.3)]

    worker = _make_worker(continuous_enabled=True)
    worker._detector = _FakeDetector()
    worker._store = Store(tmp_path / "c.db", tmp_path / "caps")
    worker._store.init_schema()
    worker._deduper = Deduper(0, 10_000)

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "wildlife.worker.grab_burst",
        lambda *a, **k: [frame],
    )
    worker._handle_event(
        CameraEvent(camera_id="cam1", event_ts=datetime(2026, 7, 6, 12), kind=EVENT_KIND)
    )
    rows = worker._store.query(camera_id="cam1")
    assert rows and rows[0]["source_kind"] == "continuous"
    worker._store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_worker_continuous.py -q -k "routes_burst or direct_burst or persists_source_kind"`
Expected: FAIL — `_handle_event` does not yet pass `rtsp_url` / `source_kind` (the go2rtc URL assertion fails; `source_kind` is `"reolink"`).

- [ ] **Step 3: Implement the consumer-side routing**

In `src/wildlife/worker.py`, add the `EVENT_KIND` import near the other event imports:

```python
from wildlife.events.base import EventSource, make_event_source
from wildlife.events.continuous_motion import EVENT_KIND as CONTINUOUS_EVENT_KIND
```

In `_handle_event`, replace the burst-grab block (step 3, currently `frames = grab_burst(camera, cap.burst_frames, cap.burst_interval_ms, cap.stream, cap.rtsp_timeout_s)`) with kind-aware routing:

```python
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
```

Add the module-level helper near `_now` (after the `_now` function):

```python
def _rtsp_port(listen: str) -> int:
    """Extract the go2rtc RTSP port from a bind string (e.g. ``":8554"``)."""
    return int(listen.rsplit(":", 1)[-1])
```

Compute the provenance tag once (right after the `is_continuous` line is fine, or just before the save block). Add near the top of the save section (step 5, before `capture_ts = _now()`):

```python
        source_kind = "continuous" if is_continuous else "reolink"
```

Pass it into **both** `save_capture` calls (the `save_best_only` branch and the all-positives loop). For each `self._store.save_capture(...)` call, add the argument:

```python
                        det=best_det,
                        source_kind=source_kind,
```

and for the loop:

```python
                        det=det,
                        source_kind=source_kind,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_worker_continuous.py -q`
Expected: PASS (all 6 in the file).

- [ ] **Step 5: Full suite + lint + commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/wildlife/worker.py tests/test_worker_continuous.py
git add src/wildlife/worker.py tests/test_worker_continuous.py
git commit -m "feat(worker): route continuous bursts via go2rtc + tag source_kind"
```

Expected: full suite green.

---

### Task 8: Documentation — example config + README

**Files:**
- Modify: `config.example.yaml`
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the `continuous` block to `config.example.yaml`**

Append after the `remote:` block:

```yaml
continuous: # optional always-on motion-gated detection (your model becomes the gate)
  enabled: false # set true to run a second, MOG2-motion-gated producer per camera
  sample_fps: 4 # frames/sec sampled from the sub restream for motion (not full fps)
  downscale_width: 480 # px width the motion detector runs at (motion computed downscaled)
  min_area_frac: 0.003 # largest motion contour must be >= this fraction of the frame
  refractory_s: 8 # min seconds between motion emits per camera (one animal -> one event)
  warmup_s: 10 # suppress emits this long after (re)connect while MOG2 stabilizes
  algorithm: mog2 # mog2 | frame_diff
  active_hours: "" # optional "HH:MM-HH:MM" local window; empty = 24/7
```

Add a commented `motion_mask` example under the `north_field` camera (after its `onvif_port: 8000` line):

```yaml
    # motion_mask: # optional ignore-motion polygons (normalized 0..1) — roads/canopy/water
    #   - [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]] # ignore the bottom 30% (e.g. a road)
```

- [ ] **Step 2: Add a "Continuous detection" section to `README.md`**

Add a new section (place it after the Remote access / Livestream material). Content to include:

```markdown
## Continuous (motion-gated) detection

By default the pipeline only looks when Reolink's onboard AI fires. Continuous
detection adds a second, always-on producer per camera that runs a cheap MOG2
motion gate on the go2rtc **sub** restream and fires your own YOLO on motion — so
*your* fine-tuned model, not Reolink's person/car AI, decides what counts. This
catches small, distant, and nocturnal wildlife the camera silently ignores.

**Requires go2rtc** (the `livestream` feature): the motion reader consumes
`rtsp://127.0.0.1:<rtsp_listen port>/<camera id>_sub`, and bursts are grabbed back
through go2rtc's main restream (no second direct camera session). Under launchd,
start go2rtc **before** the worker.

Enable it in `config.yaml`:

    continuous:
      enabled: true

**Motion masks (effectively required in busy scenes).** A yard with a road,
swaying canopy, a flag, or moving water will trip MOG2 constantly. Add per-camera
`motion_mask` ignore polygons (normalized 0..1 coordinates) to blank those regions:

    cameras:
      - id: north_field
        # ...
        motion_mask:
          - [[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]  # ignore the bottom 30%

**Tuning knobs** (start with the defaults, then adjust on real footage):
`min_area_frac` and `refractory_s` are the dominant levers for cutting trigger
volume; `active_hours` duty-cycles by time of day; `algorithm: frame_diff` is a
lighter fallback for a weaker host. Continuous captures are tagged
`source_kind = "continuous"` in the database (vs `"reolink"`) so you can tell the
two paths apart when tuning.

Because a second producer now feeds the same queue, the global
`resource_guard.max_burst_per_minute` cap (which limits how often inference
fires) may need raising so continuous events aren't starved by Reolink ones; the
per-camera `dedupe.cooldown_s` and `resource_guard.detect_every_nth_event`
throttles apply to continuous events for free.
```

- [ ] **Step 3: Verify the example config still loads**

Run: `.venv/bin/python -c "from wildlife.config import load_config; c = load_config('config.example.yaml'); print(c.continuous.sample_fps, c.cameras[0].motion_mask)"`
Expected: prints `4 None` (the commented mask stays inactive).

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml README.md
git commit -m "docs: document continuous motion-gated detection"
```

---

## Final verification

- [ ] Run the whole suite: `.venv/bin/python -m pytest -q` → all green.
- [ ] Lint everything touched: `.venv/bin/ruff check .` → clean.
- [ ] Confirm inert-when-disabled: with `continuous.enabled: false`, `_start_producers` starts exactly one thread per camera and no continuous burst path is reachable (covered by `test_second_producer_starts_only_when_enabled`).
