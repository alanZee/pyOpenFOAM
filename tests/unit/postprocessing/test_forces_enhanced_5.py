"""Tests for ForcesEnhanced5.

Tests cover:
- FSI coupling data
- Fatigue spectrum
- Moment PSD
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.forces_enhanced_5 import (
    ForcesEnhanced5,
    FSIForceData,
    FatigueSpectrum,
    MomentPSD,
)
from pyfoam.postprocessing.forces_enhanced_4 import ForcesEnhanced4


class TestForcesEnhanced5:
    """Tests for ForcesEnhanced5."""

    def test_inherits_from_enhanced4(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert isinstance(forces, ForcesEnhanced4)

    def test_default_params(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces._fsi_enabled is False
        assert forces._fatigue_enabled is False
        assert forces._moment_psd is False
        assert forces._sn_exponent == pytest.approx(3.0)

    def test_custom_params(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
            "fsiInterface": True,
            "fatigueEstimation": True,
            "momentPSD": True,
            "fatigueSNExponent": 5.0,
            "multiReferencePoints": [[1.0, 0.0, 0.0]],
        })
        assert forces._fsi_enabled is True
        assert forces._fatigue_enabled is True
        assert forces._moment_psd is True
        assert forces._sn_exponent == pytest.approx(5.0)
        assert forces._multi_cofr == [[1.0, 0.0, 0.0]]

    def test_fsi_force_data_dataclass(self):
        fsi = FSIForceData(
            time=1.0,
            total_force=torch.tensor([10.0, 5.0, 0.0]),
            total_moment=torch.tensor([0.0, 0.0, 1.0]),
            pressure_force=torch.tensor([8.0, 4.0, 0.0]),
            viscous_force=torch.tensor([2.0, 1.0, 0.0]),
            force_rms=torch.tensor([0.5, 0.3, 0.1]),
            patch_name="wall",
        )
        assert fsi.time == pytest.approx(1.0)
        assert fsi.patch_name == "wall"

    def test_fatigue_spectrum_dataclass(self):
        fs = FatigueSpectrum(
            time=1.0,
            n_cycles=100,
            damage_estimate=0.001,
        )
        assert fs.n_cycles == 100
        assert fs.damage_estimate == pytest.approx(0.001)

    def test_moment_psd_dataclass(self):
        mpsd = MomentPSD(
            peak_frequency_drag=10.0,
            peak_frequency_lift=5.0,
        )
        assert mpsd.peak_frequency_drag == pytest.approx(10.0)

    def test_empty_fsi_data(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces.fsi_data == []
        assert forces.get_latest_fsi() is None

    def test_empty_fatigue_spectra(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces.fatigue_spectra == []
        assert forces.get_latest_fatigue() is None

    def test_empty_moment_psd_results(self):
        forces = ForcesEnhanced5("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces.moment_psd_results == []
