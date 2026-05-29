"""
Unit tests for Noise — FFT-based acoustic noise analysis.

Tests cover:
- Init with default and custom config
- Window function generation (rectangular, hanning, hamming)
- SPL computation from known signal
- Probe sampling and pressure history
- FFT spectrum computation
- Overall SPL calculation
- Writing output files
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.postprocessing.noise import Noise, P_REF


class TestNoiseInit:
    """Tests for Noise function object initialisation."""

    def test_init_defaults(self):
        noise = Noise()
        assert noise.name == "noise"
        assert noise._p_ref == P_REF
        assert noise._window_func == "hanning"
        assert noise._write_spectrum is True

    def test_init_with_config(self):
        config = {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
            "pRef": 1e-3,
            "windowFunction": "hamming",
            "writeSpectrum": False,
        }
        noise = Noise("noise1", config)
        assert noise.name == "noise1"
        assert noise._p_ref == 1e-3
        assert noise._window_func == "hamming"
        assert noise._write_spectrum is False
        assert noise._probe_locations == [[0.5, 0.5, 0.5]]

    def test_init_invalid_window(self):
        with pytest.raises(ValueError, match="Unknown window function"):
            Noise("test", {"windowFunction": "blackman"})

    def test_valid_window_functions(self):
        for wf in ["rectangular", "hanning", "hamming"]:
            noise = Noise("test", {"windowFunction": wf})
            assert noise._window_func == wf


class TestNoiseWindowFunctions:
    """Tests for window function generation."""

    def test_rectangular_window(self):
        noise = Noise("test", {"windowFunction": "rectangular"})
        w = noise._create_window(10, torch.float64)
        assert w.shape == (10,)
        assert torch.allclose(w, torch.ones(10, dtype=torch.float64))

    def test_hanning_window(self):
        noise = Noise("test", {"windowFunction": "hanning"})
        w = noise._create_window(10, torch.float64)
        assert w.shape == (10,)
        # Hanning window: 0 at endpoints, 1 at centre
        assert abs(w[0].item()) < 1e-10
        assert abs(w[-1].item()) < 1e-10
        # Central value should be close to 1
        mid = len(w) // 2
        assert w[mid].item() > 0.9

    def test_hamming_window(self):
        noise = Noise("test", {"windowFunction": "hamming"})
        w = noise._create_window(10, torch.float64)
        assert w.shape == (10,)
        # Hamming window: ~0.08 at endpoints
        assert abs(w[0].item() - 0.08) < 0.01
        assert abs(w[-1].item() - 0.08) < 0.01

    def test_window_symmetry(self):
        for wf in ["hanning", "hamming"]:
            noise = Noise("test", {"windowFunction": wf})
            w = noise._create_window(11, torch.float64)
            # Window should be symmetric
            assert torch.allclose(w, w.flip(0), atol=1e-10)


class TestNoiseSPL:
    """Tests for SPL computation from pressure signals."""

    def test_spl_known_sine_wave(self):
        """SPL of a 1 Pa amplitude sine wave should be ~90 dB."""
        # p_rms = A / sqrt(2) for a sine wave
        # SPL = 20 * log10(p_rms / p_ref)
        dt = 0.001
        t = torch.arange(0, 1.0, dt, dtype=torch.float64)
        freq = 100.0
        p = torch.sin(2.0 * math.pi * freq * t)  # 1 Pa amplitude

        spl = Noise.compute_spl_from_signal(p)
        p_rms = 1.0 / math.sqrt(2)
        expected_spl = 20.0 * math.log10(p_rms / P_REF)
        assert abs(spl - expected_spl) < 1.0, (
            f"SPL mismatch: got {spl:.2f}, expected ~{expected_spl:.2f}"
        )

    def test_spl_known_amplitude(self):
        """SPL of a constant pressure signal."""
        p = torch.full((100,), 1.0, dtype=torch.float64)
        spl = Noise.compute_spl_from_signal(p)
        expected = 20.0 * math.log10(1.0 / P_REF)
        assert abs(spl - expected) < 0.1

    def test_spl_silent(self):
        """SPL of zero signal should be very low (not NaN)."""
        p = torch.zeros(100, dtype=torch.float64)
        spl = Noise.compute_spl_from_signal(p)
        assert math.isfinite(spl)
        assert spl < 0  # Below reference

    def test_spl_custom_p_ref(self):
        """SPL with custom reference pressure."""
        p = torch.full((100,), 1.0, dtype=torch.float64)
        spl = Noise.compute_spl_from_signal(p, p_ref=1.0)
        assert abs(spl) < 0.1  # 0 dB when p = p_ref


class TestNoiseExecute:
    """Tests for noise sampling execution."""

    def test_execute_collects_samples(self, fv_mesh, sample_fields):
        """Execute stores pressure samples."""
        noise = Noise("noise", {"fields": ["p"]})
        noise.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.1, 0.2, 0.3]:
            noise.execute(t)

        assert len(noise.times) == 4
        assert len(noise.pressure_history) == 4

    def test_execute_disabled(self, fv_mesh, sample_fields):
        """Disabled noise does not collect samples."""
        noise = Noise("noise", {"enabled": False, "fields": ["p"]})
        noise.initialise(fv_mesh, sample_fields)
        noise.execute(0.0)

        assert len(noise.times) == 0

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        """Missing field produces warning, not error."""
        noise = Noise("noise", {"fields": ["nonexistent"]})
        noise.initialise(fv_mesh, sample_fields)
        noise.execute(0.0)  # Should not raise

        assert len(noise.times) == 0


class TestNoiseFinalise:
    """Tests for FFT analysis after collection."""

    def test_finalise_computes_spectra(self, fv_mesh, sample_fields):
        """finalise computes FFT spectra from collected data."""
        noise = Noise("noise", {"fields": ["p"]})
        noise.initialise(fv_mesh, sample_fields)

        # Collect synthetic sine wave samples
        n_samples = 100
        dt = 0.01
        for i in range(n_samples):
            t = i * dt
            # Override: directly inject known pressure
            noise._times.append(t)
            noise._pressure_history.append(
                torch.tensor([100.0 + 0.1 * math.sin(2 * math.pi * 10 * t)])
            )

        noise.finalise()

        assert noise.frequencies is not None
        assert noise.psd is not None
        assert noise.spl is not None
        assert noise.spl_total is not None

        assert noise.frequencies.shape[0] == n_samples // 2 + 1
        assert noise.psd.shape[0] == noise.frequencies.shape[0]
        assert noise.spl.shape == noise.psd.shape
        assert noise.spl_total.shape == (1,)  # 1 probe

    def test_finalise_too_few_samples(self, fv_mesh, sample_fields):
        """finalise with < 2 samples does not crash."""
        noise = Noise("noise", {"fields": ["p"]})
        noise.initialise(fv_mesh, sample_fields)
        noise.execute(0.0)
        noise.finalise()

        assert noise.frequencies is None  # Not enough data

    def test_spl_total_finite(self, fv_mesh, sample_fields):
        """SPL values are finite."""
        noise = Noise("noise", {"fields": ["p"]})
        noise.initialise(fv_mesh, sample_fields)

        for i in range(50):
            noise._times.append(i * 0.01)
            noise._pressure_history.append(torch.tensor([101325.0 + 0.01 * i]))

        noise.finalise()

        assert torch.isfinite(noise.spl_total).all()


class TestNoiseWrite:
    """Tests for output file writing."""

    def test_write_spectrum(self, fv_mesh, sample_fields, tmp_path):
        """Writing spectrum data to files."""
        noise = Noise("noise", {"fields": ["p"], "writeSpectrum": True})
        noise.set_output_path(tmp_path)
        noise.initialise(fv_mesh, sample_fields)

        for i in range(50):
            noise._times.append(i * 0.01)
            noise._pressure_history.append(torch.tensor([1.0 + 0.01 * i]))

        noise.finalise()
        noise.write()

        assert (tmp_path / "spectrum_probe0.dat").exists()
        assert (tmp_path / "spl_overall.dat").exists()

    def test_write_skipped_without_data(self, tmp_path):
        """Writing is skipped if no data collected."""
        noise = Noise("noise")
        noise.set_output_path(tmp_path)
        noise.write()

        assert not (tmp_path / "spl_overall.dat").exists()


class TestNoiseRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing.noise import Noise
        FunctionObjectRegistry.register("noise", Noise)
        assert "noise" in FunctionObjectRegistry.list_registered()

    def test_create_from_registry(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing.noise import Noise
        FunctionObjectRegistry.register("noise", Noise)
        fo = FunctionObjectRegistry.create("noise", {"name": "noise1"})
        assert isinstance(fo, Noise)
        assert fo.name == "noise1"
