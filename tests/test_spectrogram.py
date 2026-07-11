"""Tests for the numpy+Pillow spectrogram renderer (no birdnet/matplotlib/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")

from wildlife._colormap import MAGMA_LUT  # noqa: E402
from wildlife.audio import render_spectrogram, signal_quality  # noqa: E402


def _tone(freq: float, sr: int, n: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


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


# --- signal_quality (band-limited SNR) ------------------------------------

def test_signal_quality_silence_is_zero():
    assert signal_quality(np.zeros(48000 * 3, np.float32), 48000) == 0.0


def test_signal_quality_clear_burst_beats_uniform_noise():
    sr, n = 48000, 48000 * 3
    rng = np.random.default_rng(0)
    # A short 2 kHz burst over a near-silent background -> high peak/floor contrast.
    burst = (1e-3 * rng.standard_normal(n)).astype(np.float32)
    b0, b1 = sr * 1, sr * 1 + sr // 3  # 1/3 s burst
    burst[b0:b1] += _tone(2000, sr, b1 - b0, amp=0.5)
    # Uniform in-band noise: energy roughly constant across frames -> low contrast.
    uniform = (0.1 * rng.standard_normal(n)).astype(np.float32)
    assert signal_quality(burst, sr) > signal_quality(uniform, sr)
    assert signal_quality(burst, sr) > 10.0  # a clear call stands well above the floor


def test_signal_quality_ignores_subband_rumble():
    # A loud 200 Hz rumble sits below the 1 kHz band edge -> not counted as signal.
    sr, n = 48000, 48000 * 3
    assert signal_quality(_tone(200, sr, n, amp=0.8), sr) < 5.0


def test_signal_quality_continuous_tone_has_low_contrast():
    # A tone filling the whole window has peak ~= floor -> lower SNR than the same
    # tone as a short burst over quiet background (contrast is what the metric rewards).
    sr, n = 48000, 48000 * 3
    cont = _tone(2000, sr, n, amp=0.5)
    burst = (1e-3 * np.random.default_rng(1).standard_normal(n)).astype(np.float32)
    b0, b1 = sr * 1, sr * 1 + sr // 3
    burst[b0:b1] += _tone(2000, sr, b1 - b0, amp=0.5)
    assert signal_quality(cont, sr) < signal_quality(burst, sr)
