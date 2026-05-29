"""Tests for ProbesEnhanced3.

Tests cover:
- Per-probe spectral analysis
- Coherence matrix
- Signal quality assessment
"""

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced_3 import (
    ProbesEnhanced3,
    ProbeSpectrumResult,
    CoherenceMatrix,
)
from pyfoam.postprocessing.probes_enhanced_2 import ProbesEnhanced2


class TestProbesEnhanced3:
    """Tests for ProbesEnhanced3."""

    def test_inherits_from_enhanced2(self):
        probes = ProbesEnhanced3("test", {
            "probeGroups": [
                {"name": "g1", "locations": [[0.5, 0.5, 0.5]], "fields": ["p"]},
            ],
        })
        assert isinstance(probes, ProbesEnhanced2)

    def test_default_params(self):
        probes = ProbesEnhanced3("test", {})
        assert probes._compute_spectrum_per_probe is True
        assert probes._n_fft_points == 1024
        assert probes._compute_coherence_matrix is False

    def test_custom_params(self):
        probes = ProbesEnhanced3("test", {
            "computeSpectrumPerProbe": False,
            "nFFTPoints": 512,
            "computeCoherenceMatrix": True,
        })
        assert probes._compute_spectrum_per_probe is False
        assert probes._n_fft_points == 512
        assert probes._compute_coherence_matrix is True

    def test_compute_probe_spectrum_no_data(self):
        probes = ProbesEnhanced3("test", {})
        result = probes.compute_probe_spectrum("p", 0)
        assert result is None

    def test_compute_probe_spectrum_with_data(self):
        """Test spectral analysis with synthetic signal data."""
        probes = ProbesEnhanced3("test", {})

        # Manually inject probe data
        probes._results["p"] = {
            0: [float(torch.sin(torch.tensor(2 * 3.14159 * 10 * t / 100)).item())
                for t in range(200)],
        }
        probes._times = [t * 0.01 for t in range(200)]

        result = probes.compute_probe_spectrum("p", 0, dt=0.01)
        assert result is not None
        assert isinstance(result, ProbeSpectrumResult)
        assert result.frequencies.numel() > 0
        assert result.psd.numel() > 0
        assert result.peak_frequency >= 0
        assert result.n_samples > 0

    def test_compute_probe_spectrum_caching(self):
        """Repeated calls should return cached result."""
        probes = ProbesEnhanced3("test", {})
        probes._results["p"] = {
            0: [float(i % 10) for i in range(100)],
        }
        probes._times = [t * 0.01 for t in range(100)]

        r1 = probes.compute_probe_spectrum("p", 0, dt=0.01)
        r2 = probes.compute_probe_spectrum("p", 0, dt=0.01)
        assert r1 is r2  # Same object (cached)

    def test_compute_probe_spectrum_snr(self):
        """Pure sinusoidal signal should have high SNR."""
        probes = ProbesEnhanced3("test", {})
        t_vals = torch.linspace(0, 1, 256, dtype=torch.float64)
        freq = 50.0
        signal = torch.sin(2 * 3.14159 * freq * t_vals).tolist()
        probes._results["p"] = {0: signal}
        probes._times = t_vals.tolist()

        result = probes.compute_probe_spectrum("p", 0, dt=float(t_vals[1] - t_vals[0]))
        assert result is not None
        assert result.snr_estimate > 1.0  # Should detect strong peak

    def test_coherence_matrix_no_data(self):
        probes = ProbesEnhanced3("test", {})
        result = probes.compute_coherence_matrix("p")
        assert result is None

    def test_coherence_matrix_insufficient_probes(self):
        """Single probe cannot produce coherence matrix."""
        probes = ProbesEnhanced3("test", {})
        probes._results["p"] = {0: [1.0, 2.0, 3.0, 4.0]}
        probes._times = [0.0, 1.0, 2.0, 3.0]
        result = probes.compute_coherence_matrix("p")
        assert result is None

    def test_properties_empty(self):
        probes = ProbesEnhanced3("test", {})
        assert probes.probe_spectra == {}
        assert probes.coherence_matrices == {}


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
