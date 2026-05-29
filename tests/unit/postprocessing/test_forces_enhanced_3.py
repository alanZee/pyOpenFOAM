"""Tests for ForcesEnhanced3.

Tests cover:
- Moment coefficients
- Force spectrum computation
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.forces_enhanced_3 import (
    ForcesEnhanced3,
    MomentCoefficients,
    ForceSpectrum,
)
from pyfoam.postprocessing.forces_enhanced_2 import ForcesEnhanced2


class TestForcesEnhanced3:
    """Tests for ForcesEnhanced3."""

    def test_inherits_from_enhanced2(self):
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
        })
        assert isinstance(fo, ForcesEnhanced2)

    def test_default_params(self):
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
        })
        assert fo._compute_moment_coeffs is True
        assert fo._compute_force_spectrum is False
        assert fo._n_fft_points == 256

    def test_custom_params(self):
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
            "computeMomentCoeffs": False,
            "computeForceSpectrum": True,
            "nFFTPoints": 512,
        })
        assert fo._compute_moment_coeffs is False
        assert fo._compute_force_spectrum is True
        assert fo._n_fft_points == 512

    def test_execute(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
            "Uref": 1.0,
            "computeMomentCoeffs": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.moment_coefficients) == 1

    def test_moment_coefficients_values(self, fv_mesh, sample_fields):
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
            "Uref": 1.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        mc = fo.get_latest_moment()
        assert mc is not None
        assert isinstance(mc, MomentCoefficients)
        assert mc.time == pytest.approx(0.0)
        assert isinstance(mc.Cm_roll, float)
        assert isinstance(mc.Cm_pitch, float)
        assert isinstance(mc.Cm_yaw, float)

    def test_get_latest_moment_no_data(self):
        fo = ForcesEnhanced3("test", {"patches": ["bottom"]})
        assert fo.get_latest_moment() is None

    def test_force_spectrum_none_initially(self):
        fo = ForcesEnhanced3("test", {"patches": ["bottom"]})
        assert fo.force_spectrum is None

    def test_spectrum_insufficient_data(self, fv_mesh, sample_fields):
        """With only 1 time step, spectrum should not be computed."""
        fo = ForcesEnhanced3("test", {
            "patches": ["bottom"],
            "rhoInf": 1.0,
            "computeForceSpectrum": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        # Not enough data for spectrum (< 4 time steps)
        assert fo.force_spectrum is None


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
