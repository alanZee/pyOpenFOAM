"""Tests for noise_enhanced — enhanced noise analysis."""

from __future__ import annotations

import numpy as np
import pytest

from pyfoam.tools.noise_enhanced import noise_analysis, NoiseResult


def _sine_signal(freq, duration, fs, amplitude=1.0):
    """Generate a sine wave signal."""
    t = np.arange(0, duration, 1.0 / fs)
    return amplitude * np.sin(2.0 * np.pi * freq * t)


class TestNoiseAnalysis:
    """Test the noise_analysis function."""

    def test_basic_analysis(self):
        """Basic analysis should produce valid result."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0)
        assert isinstance(result, NoiseResult)
        assert result.n_samples == len(p)
        assert result.sample_rate == 1000.0

    def test_psd_shape(self):
        """PSD should have correct shape (n_fft/2 + 1)."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0, n_fft=256)
        assert result.frequencies.shape == result.psd.shape
        assert result.frequencies.shape[0] == 256 // 2 + 1

    def test_spl_shape(self):
        """SPL should have same shape as PSD."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0, n_fft=256)
        assert result.spl.shape == result.psd.shape

    def test_peak_at_sine_frequency(self):
        """Peak frequency should be near the sine wave frequency."""
        fs = 1000.0
        freq = 100.0
        p = _sine_signal(freq, 2.0, fs)
        result = noise_analysis(p, sample_rate=fs, n_fft=1024)
        # Allow +-one frequency bin resolution
        df = fs / 1024.0
        assert abs(result.peak_frequency - freq) < 2 * df + 1e-6

    def test_overall_spl_finite(self):
        """Overall SPL should be finite and positive."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0)
        assert np.isfinite(result.overall_spl_db)
        assert result.overall_spl_db > 0.0

    def test_a_weighting_shift(self):
        """A-weighted SPL should differ from unweighted."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0, a_weighting=True)
        # Low-frequency A-weighting should reduce SPL
        # (100 Hz gets about -19 dB)
        assert not np.allclose(result.spl, result.spl_a_weighted)

    def test_no_a_weighting(self):
        """With a_weighting=False, spl_a_weighted should equal spl."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0, a_weighting=False)
        np.testing.assert_array_equal(result.spl, result.spl_a_weighted)

    def test_octave_bands_populated(self):
        """1/3-octave bands should be computed."""
        p = _sine_signal(1000.0, 1.0, 44100.0)
        result = noise_analysis(p, sample_rate=44100.0)
        assert len(result.octave_bands) > 0
        # Should contain standard octave band labels
        assert any("1kHz" in k for k in result.octave_bands)

    def test_short_signal_raises(self):
        """Too-short signal should raise ValueError."""
        p = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 4"):
            noise_analysis(p, sample_rate=1000.0)

    def test_window_types(self):
        """Different window types should work."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        for win in ["hanning", "hamming", "none"]:
            result = noise_analysis(p, sample_rate=1000.0, window=win)
            assert np.all(np.isfinite(result.spl))

    def test_overlap_parameter(self):
        """Different overlap values should produce valid results."""
        p = _sine_signal(100.0, 2.0, 1000.0)
        for ov in [0.0, 0.25, 0.5, 0.75]:
            result = noise_analysis(p, sample_rate=1000.0, overlap=ov)
            assert np.all(np.isfinite(result.psd))

    def test_white_noise_flat_psd(self):
        """White noise should have roughly flat PSD."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(10000)
        result = noise_analysis(p, sample_rate=1000.0, n_fft=1024)
        # PSD should not have huge variations (within 20 dB)
        psd_mid = result.psd[10:-10]
        ratio = psd_mid.max() / (psd_mid.min() + 1e-40)
        assert ratio < 100.0  # <20 dB variation

    def test_frequencies_positive(self):
        """Frequency array should be non-negative."""
        p = _sine_signal(100.0, 1.0, 1000.0)
        result = noise_analysis(p, sample_rate=1000.0)
        assert np.all(result.frequencies >= 0.0)
