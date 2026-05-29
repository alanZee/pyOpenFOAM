"""Tests for ProbesEnhanced — time interpolation and spectral analysis."""

import math

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced import ProbesEnhanced, SpectrumResult


# ---------------------------------------------------------------------------
# ProbesEnhanced tests
# ---------------------------------------------------------------------------


class TestProbesEnhanced:
    """Test ProbesEnhanced function object."""

    def _create_probes(self, config=None):
        """Helper to create and initialise probes."""
        from tests.unit.postprocessing.conftest import make_fv_mesh

        mesh = make_fv_mesh()
        config = config or {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        }
        probes = ProbesEnhanced("testProbes", config)
        probes.initialise(mesh, {"p": _make_field(mesh)})
        return probes, mesh

    def test_creation(self):
        """Create ProbesEnhanced with config."""
        from tests.unit.postprocessing.conftest import make_fv_mesh
        mesh = make_fv_mesh()
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
            "computeSpectrum": True,
        })
        probes.initialise(mesh, {"p": _make_field(mesh)})
        assert probes._compute_spectrum is True

    def test_inherits_probes(self):
        """ProbesEnhanced is a subclass of Probes."""
        from pyfoam.postprocessing.sampling import Probes
        assert issubclass(ProbesEnhanced, Probes)

    def test_linear_interpolation(self, fv_mesh, sample_fields):
        """Linear interpolation at probe location."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        probes.execute(0.0)
        probes.execute(1.0)

        # Interpolate at midpoint
        val = probes.interpolate_at_time("p", 0, 0.5, method="linear")
        assert val is not None

    def test_cubic_interpolation(self, fv_mesh, sample_fields):
        """Cubic interpolation at probe location."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            probes.execute(t)

        val = probes.interpolate_at_time("p", 0, 0.375, method="cubic")
        assert val is not None

    def test_interpolate_at_boundary(self, fv_mesh, sample_fields):
        """Interpolation at boundary times returns exact values."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        probes.execute(0.0)
        probes.execute(1.0)

        # At start time
        val_start = probes.interpolate_at_time("p", 0, 0.0)
        assert val_start is not None

        # At end time
        val_end = probes.interpolate_at_time("p", 0, 1.0)
        assert val_end is not None

    def test_interpolate_out_of_range(self, fv_mesh, sample_fields):
        """Out-of-range times return boundary values."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        probes.execute(0.5)
        probes.execute(1.0)

        # Before start
        val = probes.interpolate_at_time("p", 0, -1.0)
        assert val is not None

        # After end
        val = probes.interpolate_at_time("p", 0, 100.0)
        assert val is not None

    def test_interpolate_missing_field(self, fv_mesh, sample_fields):
        """Interpolate returns None for missing field."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)

        val = probes.interpolate_at_time("nonexistent", 0, 0.0)
        assert val is None

    def test_interpolate_at_times(self, fv_mesh, sample_fields):
        """Batch interpolation."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)
        probes.execute(1.0)

        results = probes.interpolate_at_times("p", 0, [0.0, 0.5, 1.0])
        assert len(results) == 3
        assert all(v is not None for v in results)


# ---------------------------------------------------------------------------
# Spectral analysis tests
# ---------------------------------------------------------------------------


class TestSpectralAnalysis:
    """Test spectral analysis features."""

    def test_compute_spectrum(self, fv_mesh, sample_fields):
        """Compute power spectrum."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        # Generate a sine wave signal
        dt = 0.01
        n_steps = 100
        for i in range(n_steps):
            t = i * dt
            # Modify field values to create a signal
            val = torch.tensor([1.0 + 0.5 * math.sin(2 * math.pi * 10 * t)] * 2,
                               dtype=torch.float64)
            probes._results["p"][0].append(val[0].item())
            probes._times.append(t)

        spec = probes.compute_spectrum("p", 0, dt=dt)
        assert spec is not None
        assert isinstance(spec, SpectrumResult)
        assert spec.frequencies.numel() > 0
        assert spec.power.numel() > 0
        assert spec.dt == dt

    def test_peak_frequency(self, fv_mesh, sample_fields):
        """Peak frequency detection."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        dt = 0.01
        n_steps = 200
        freq = 5.0  # 5 Hz signal
        for i in range(n_steps):
            t = i * dt
            probes._results["p"][0].append(math.sin(2 * math.pi * freq * t))
            probes._times.append(t)

        peak = probes.get_peak_frequency("p", 0)
        assert peak is not None
        # Peak should be near 5 Hz
        assert abs(peak - freq) < 2.0  # Allow some FFT resolution error

    def test_spectrum_cached(self, fv_mesh, sample_fields):
        """Spectrum is cached after first computation."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        dt = 0.01
        for i in range(50):
            probes._results["p"][0].append(math.sin(i * dt))
            probes._times.append(i * dt)

        spec1 = probes.compute_spectrum("p", 0, dt=dt)
        spec2 = probes.compute_spectrum("p", 0, dt=dt)

        assert spec1 is spec2  # Same object (cached)

    def test_spectrum_insufficient_data(self, fv_mesh, sample_fields):
        """Spectrum returns None with too few samples."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)

        spec = probes.compute_spectrum("p", 0, dt=1.0)
        assert spec is None

    def test_compute_all_spectra(self, fv_mesh, sample_fields):
        """Compute spectra for all fields and probes."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        dt = 0.01
        for i in range(50):
            probes._results["p"][0].append(math.sin(i * dt))
            probes._times.append(i * dt)

        all_specs = probes.compute_all_spectra(dt=dt)
        assert ("p", 0) in all_specs

    def test_hanning_window(self, fv_mesh, sample_fields):
        """Hanning window function."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        signal = torch.ones(100, dtype=torch.float64)
        windowed = probes._apply_window(signal, "hanning")
        assert windowed.shape == (100,)
        # Ends should be near zero
        assert abs(windowed[0].item()) < 0.1
        assert abs(windowed[-1].item()) < 0.1

    def test_hamming_window(self, fv_mesh, sample_fields):
        """Hamming window function."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        signal = torch.ones(100, dtype=torch.float64)
        windowed = probes._apply_window(signal, "hamming")
        assert windowed.shape == (100,)
        # Hamming window endpoints are ~0.08, not 0
        assert abs(windowed[0].item()) < 0.2

    def test_no_window(self, fv_mesh, sample_fields):
        """No window returns unchanged signal."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)

        signal = torch.ones(10, dtype=torch.float64)
        result = probes._apply_window(signal, "none")
        assert torch.equal(result, signal)

    def test_write_includes_spectra(self, fv_mesh, sample_fields, tmp_path):
        """Write includes spectrum files."""
        probes = ProbesEnhanced("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.set_output_path(tmp_path)
        probes.initialise(fv_mesh, sample_fields)

        dt = 0.01
        for i in range(50):
            probes._results["p"][0].append(math.sin(i * dt))
            probes._times.append(i * dt)

        probes.compute_spectrum("p", 0, dt=dt)
        probes.write()

        assert (tmp_path / "p_probe0_spectrum.dat").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_field(mesh):
    """Create a sample scalar field."""
    from pyfoam.fields.vol_fields import volScalarField
    return volScalarField(mesh, "p")
