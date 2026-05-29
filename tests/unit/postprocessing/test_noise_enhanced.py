"""
Unit tests for NoiseEnhanced — A-weighted acoustic noise analysis.

Tests cover:
- Init with default and custom config
- A-weighting computation
- 1/3 octave band analysis
- LAeq computation
- Peak frequency detection
- Write output files
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.postprocessing.noise_enhanced import (
    NoiseEnhanced,
    a_weighting,
    ThirdOctaveBand,
    _THIRD_OCTAVE_CENTRES,
)


class TestAWeighting:
    """Tests for A-weighting correction function."""

    def test_a_weighting_shape(self):
        """A-weighting returns correct shape."""
        freqs = torch.tensor([100.0, 1000.0, 10000.0], dtype=torch.float64)
        a = a_weighting(freqs)
        assert a.shape == (3,)

    def test_a_weighting_1khz_near_zero(self):
        """A-weighting at 1000 Hz should be close to 0 dB."""
        freqs = torch.tensor([1000.0], dtype=torch.float64)
        a = a_weighting(freqs)
        assert abs(a.item()) < 3.0  # Should be ~0 dB

    def test_a_weighting_low_freq_attenuation(self):
        """A-weighting at low frequencies should be negative."""
        freqs = torch.tensor([10.0, 50.0, 100.0], dtype=torch.float64)
        a = a_weighting(freqs)
        assert (a < 0).all()

    def test_a_weighting_high_freq_attenuation(self):
        """A-weighting at very high frequencies should be negative."""
        freqs = torch.tensor([15000.0, 20000.0], dtype=torch.float64)
        a = a_weighting(freqs)
        assert (a < 0).all()

    def test_a_weighting_peak_near_3khz(self):
        """Peak A-weighting should be near 3-4 kHz."""
        freqs = torch.linspace(1000, 10000, 100, dtype=torch.float64)
        a = a_weighting(freqs)
        peak_idx = a.argmax()
        peak_freq = freqs[peak_idx].item()
        assert 2000 < peak_freq < 5000

    def test_a_weighting_finite(self):
        """A-weighting produces finite values."""
        freqs = torch.logspace(1, 4, 50, dtype=torch.float64)
        a = a_weighting(freqs)
        assert torch.isfinite(a).all()


class TestThirdOctaveBand:
    """Tests for ThirdOctaveBand data class."""

    def test_creation(self):
        band = ThirdOctaveBand(
            centre=1000.0, lower=890.9, upper=1122.5,
            spl=80.0, spl_a=78.0,
        )
        assert band.centre == 1000.0
        assert band.spl == 80.0
        assert band.spl_a == 78.0

    def test_repr(self):
        band = ThirdOctaveBand(1000.0, 890.9, 1122.5, 80.0, 78.0)
        r = repr(band)
        assert "1000.0 Hz" in r
        assert "dBA" in r


class TestNoiseEnhancedInit:
    """Tests for NoiseEnhanced initialisation."""

    def test_init_defaults(self):
        ne = NoiseEnhanced()
        assert ne.name == "noiseEnhanced"
        assert ne._use_a_weighting is True
        assert ne._use_third_octave is True

    def test_init_with_config(self):
        config = {
            "fields": ["p"],
            "aWeighting": False,
            "thirdOctaveBands": False,
            "windowSize": 1024,
        }
        ne = NoiseEnhanced("ne1", config)
        assert ne.name == "ne1"
        assert ne._use_a_weighting is False
        assert ne._use_third_octave is False
        assert ne._window_size == 1024

    def test_inherits_noise(self):
        """NoiseEnhanced is a subclass of Noise."""
        from pyfoam.postprocessing.noise import Noise
        ne = NoiseEnhanced()
        assert isinstance(ne, Noise)


class TestNoiseEnhancedFinalise:
    """Tests for enhanced spectral analysis."""

    def _make_ne_with_data(self, n_samples=100, dt=0.01, freq=100.0):
        """Helper: create NoiseEnhanced with synthetic sine wave data."""
        ne = NoiseEnhanced("ne", {"fields": ["p"]})
        for i in range(n_samples):
            t = i * dt
            ne._times.append(t)
            ne._pressure_history.append(
                torch.tensor([101325.0 + 0.1 * math.sin(2 * math.pi * freq * t)])
            )
        return ne

    def test_finalise_computes_a_weighting(self):
        """finalise computes A-weighted spectra."""
        ne = self._make_ne_with_data()
        ne.finalise()

        assert ne.spl_a is not None
        assert ne.spl_a_total is not None
        assert ne.a_weighted_psd is not None

    def test_finalise_computes_third_octave(self):
        """finalise computes 1/3 octave band results."""
        ne = self._make_ne_with_data(n_samples=200)
        ne.finalise()

        assert ne.third_octave_results is not None
        assert len(ne.third_octave_results) == 1  # 1 probe
        assert len(ne.third_octave_results[0]) == len(_THIRD_OCTAVE_CENTRES)

    def test_finalise_computes_laeq(self):
        """finalise computes LAeq."""
        ne = self._make_ne_with_data()
        ne.finalise()

        assert ne.laeq is not None
        assert torch.isfinite(ne.laeq).all()

    def test_finalise_computes_peak_frequency(self):
        """finalise finds peak frequency."""
        ne = self._make_ne_with_data(n_samples=200, freq=200.0)
        ne.finalise()

        assert ne.peak_frequencies is not None
        assert ne.peak_spl is not None

    def test_finalise_too_few_samples(self):
        """finalise with < 2 samples handles gracefully."""
        ne = NoiseEnhanced("ne", {"fields": ["p"]})
        ne._times.append(0.0)
        ne._pressure_history.append(torch.tensor([1.0]))
        ne.finalise()

        assert ne.spl_a is None  # Base class didn't compute spectra

    def test_a_weighting_disabled(self):
        """A-weighting can be disabled."""
        ne = NoiseEnhanced("ne", {"fields": ["p"], "aWeighting": False})
        for i in range(100):
            ne._times.append(i * 0.01)
            ne._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))
        ne.finalise()

        assert ne.spl_a is None
        assert ne._a_weighted_psd is None

    def test_third_octave_disabled(self):
        """1/3 octave analysis can be disabled."""
        ne = NoiseEnhanced("ne", {"fields": ["p"], "thirdOctaveBands": False})
        for i in range(100):
            ne._times.append(i * 0.01)
            ne._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))
        ne.finalise()

        assert ne.third_octave_results is None


class TestNoiseEnhancedWrite:
    """Tests for output file writing."""

    def test_write_a_weighted(self, tmp_path):
        """Writing A-weighted SPL files."""
        ne = NoiseEnhanced("ne", {"fields": ["p"], "writeSpectrum": True})
        ne.set_output_path(tmp_path)

        for i in range(100):
            ne._times.append(i * 0.01)
            ne._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))
        ne.finalise()
        ne.write()

        assert (tmp_path / "spl_a_overall.dat").exists()
        assert (tmp_path / "spl_overall.dat").exists()

    def test_write_third_octave(self, tmp_path):
        """Writing 1/3 octave band files."""
        ne = NoiseEnhanced("ne", {"fields": ["p"]})
        ne.set_output_path(tmp_path)

        for i in range(200):
            ne._times.append(i * 0.01)
            ne._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))
        ne.finalise()
        ne.write()

        assert (tmp_path / "third_octave_probe0.dat").exists()

    def test_write_peak_frequency(self, tmp_path):
        """Writing peak frequency file."""
        ne = NoiseEnhanced("ne", {"fields": ["p"]})
        ne.set_output_path(tmp_path)

        for i in range(200):
            ne._times.append(i * 0.01)
            ne._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))
        ne.finalise()
        ne.write()

        assert (tmp_path / "peak_frequency.dat").exists()

    def test_write_no_data(self, tmp_path):
        """Writing with no data is skipped."""
        ne = NoiseEnhanced("ne")
        ne.set_output_path(tmp_path)
        ne.write()

        assert not (tmp_path / "spl_a_overall.dat").exists()


class TestNoiseEnhancedRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing.noise_enhanced import NoiseEnhanced
        FunctionObjectRegistry.register("noiseEnhanced", NoiseEnhanced)
        assert "noiseEnhanced" in FunctionObjectRegistry.list_registered()
