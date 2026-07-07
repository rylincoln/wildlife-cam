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
