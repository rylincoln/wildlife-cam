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


class _FakeSession:
    """Stand-in for a held-open birdnet prediction session (context manager)."""

    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True
        return False

    def run_arrays(self, inp):
        # one segment, one species above threshold
        return _FakeStructured([("", "Turdus migratorius_American Robin", 0.82)])


class _FakeAcoustic:
    def __init__(self):
        self.last_kwargs = None
        self.session = None

    def predict_session(self, **kwargs):
        # The shortlist/threshold/bandpass are baked into the warm session here.
        self.last_kwargs = kwargs
        self.session = _FakeSession()
        return self.session


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
