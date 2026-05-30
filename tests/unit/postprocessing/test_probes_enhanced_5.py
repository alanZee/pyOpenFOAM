"""Tests for ProbesEnhanced5.

Tests cover:
- Signal quality assessment
- Wavelet transform basics
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced_5 import (
    ProbesEnhanced5,
    WaveletResult,
    SignalQuality,
    AutoPlacementResult,
)
from pyfoam.postprocessing.probes_enhanced_4 import ProbesEnhanced4


class TestProbesEnhanced5:
    """Tests for ProbesEnhanced5."""

    def test_inherits_from_enhanced4(self):
        probes = ProbesEnhanced5("test", {})
        assert isinstance(probes, ProbesEnhanced4)

    def test_default_params(self):
        probes = ProbesEnhanced5("test", {})
        assert probes._wavelet_enabled is False
        assert probes._wavelet_type == "morlet"
        assert probes._n_scales == 32
        assert probes._auto_placement is False
        assert probes._signal_quality_check is True

    def test_custom_params(self):
        probes = ProbesEnhanced5("test", {
            "waveletAnalysis": True,
            "waveletType": "mexican_hat",
            "nWaveletScales": 64,
            "autoPlacement": True,
            "signalQualityCheck": False,
        })
        assert probes._wavelet_enabled is True
        assert probes._wavelet_type == "mexican_hat"
        assert probes._n_scales == 64
        assert probes._auto_placement is True
        assert probes._signal_quality_check is False

    def test_signal_quality_sine_wave(self):
        probes = ProbesEnhanced5("test", {})
        # Clean sine wave
        t = torch.linspace(0, 1, 256, dtype=torch.float64)
        signal = torch.sin(2 * 3.14159 * 10 * t)

        sq = probes.assess_signal_quality(signal, dt=1.0 / 256)
        assert isinstance(sq, SignalQuality)
        assert sq.snr_db > 0  # Clean signal should have positive SNR

    def test_signal_quality_noisy(self):
        probes = ProbesEnhanced5("test", {})
        signal = torch.randn(100, dtype=torch.float64)

        sq = probes.assess_signal_quality(signal, dt=0.01)
        assert isinstance(sq, SignalQuality)

    def test_signal_quality_short_signal(self):
        probes = ProbesEnhanced5("test", {})
        signal = torch.tensor([1.0, 2.0], dtype=torch.float64)

        sq = probes.assess_signal_quality(signal, dt=0.01)
        assert isinstance(sq, SignalQuality)

    def test_wavelet_result_dataclass(self):
        wr = WaveletResult()
        assert wr.scales is None
        assert wr.peak_frequency == 0.0

    def test_signal_quality_dataclass(self):
        sq = SignalQuality(snr_db=10.0, has_aliasing=False)
        assert sq.snr_db == pytest.approx(10.0)
        assert sq.has_aliasing is False

    def test_auto_placement_dataclass(self):
        ap = AutoPlacementResult(locations=[[0.1, 0.2, 0.3]], scores=[0.9])
        assert ap.n_suggested == 0  # default
        assert len(ap.locations) == 1
