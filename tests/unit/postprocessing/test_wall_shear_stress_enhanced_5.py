"""Tests for WallShearStressEnhanced5.

Tests cover:
- Anisotropy tensor
- Coherent structure detection
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress_enhanced_5 import (
    WallShearStressEnhanced5,
    AnisotropyTensor,
    CoherentStructure,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_4 import WallShearStressEnhanced4


class TestWallShearStressEnhanced5:
    """Tests for WallShearStressEnhanced5."""

    def test_inherits_from_enhanced4(self):
        wss = WallShearStressEnhanced5("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert isinstance(wss, WallShearStressEnhanced4)

    def test_default_params(self):
        wss = WallShearStressEnhanced5("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss._anisotropy_enabled is False
        assert wss._coherent_enabled is False
        assert wss._pressure_coupling is False
        assert wss._multiscale_enabled is False
        assert wss._n_time_scales == 5

    def test_custom_params(self):
        wss = WallShearStressEnhanced5("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
            "anisotropyTensor": True,
            "coherentStructureDetection": True,
            "pressureGradientCoupling": True,
            "multiscaleDecomposition": True,
            "nTimeScales": 8,
        })
        assert wss._anisotropy_enabled is True
        assert wss._coherent_enabled is True
        assert wss._pressure_coupling is True
        assert wss._multiscale_enabled is True
        assert wss._n_time_scales == 8

    def test_anisotropy_tensor_dataclass(self):
        at = AnisotropyTensor(
            patch_name="wall",
            time=1.0,
            b_ij=torch.zeros(3, 3),
            invariant_II=0.5,
            invariant_III=0.1,
            anisotropy_ratio=2.0,
        )
        assert at.patch_name == "wall"
        assert at.anisotropy_ratio == pytest.approx(2.0)

    def test_coherent_structure_dataclass(self):
        cs = CoherentStructure(
            patch_name="wall",
            time=1.0,
            n_ejections=10,
            n_sweeps=12,
            mean_ejection_duration=3.5,
            mean_sweep_duration=4.0,
            burst_period=50.0,
        )
        assert cs.n_ejections == 10
        assert cs.burst_period == pytest.approx(50.0)

    def test_empty_anisotropy_results(self):
        wss = WallShearStressEnhanced5("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss.anisotropy_results == []
        assert wss.get_latest_anisotropy("wall") is None

    def test_empty_coherent_structures(self):
        wss = WallShearStressEnhanced5("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss.coherent_structures == []
        assert wss.get_latest_coherent("wall") is None
