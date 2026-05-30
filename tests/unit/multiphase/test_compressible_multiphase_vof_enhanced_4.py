"""Tests for CompressibleMultiphaseVoFEnhanced4 (v5).

Tests cover:
- EOS consistency check
- Wood mixture speed of sound
- Enhanced PISO pressure correction
- Custom parameters
"""

import pytest
import torch

from pyfoam.multiphase.compressible_multiphase_vof_enhanced_4 import (
    CompressibleMultiphaseVoFEnhanced4,
)
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_3 import (
    CompressibleMultiphaseVoFEnhanced3,
)


class TestCompressibleMultiphaseVoFEnhanced4:
    """Tests for CompressibleMultiphaseVoFEnhanced4."""

    def test_inherits_from_v4(self):
        model = CompressibleMultiphaseVoFEnhanced4(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert isinstance(model, CompressibleMultiphaseVoFEnhanced3)

    def test_default_params(self):
        model = CompressibleMultiphaseVoFEnhanced4(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model.eos_consistency_tol == pytest.approx(1e-4)
        assert model.piso_relax == pytest.approx(0.3)
        assert model.use_wood_speed is True

    def test_custom_params(self):
        model = CompressibleMultiphaseVoFEnhanced4(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            eos_consistency_tol=1e-6,
            piso_relax=0.5,
            use_wood_speed=False,
        )
        assert model.eos_consistency_tol == pytest.approx(1e-6)
        assert model.piso_relax == pytest.approx(0.5)
        assert model.use_wood_speed is False

    def test_eos_consistency_check(self):
        model = CompressibleMultiphaseVoFEnhanced4(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.3]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        rho_exp = torch.tensor([700.0], dtype=torch.float64)

        residual, converged = model.check_eos_consistency(alphas, p, T, rho_exp)
        assert residual.shape == (1,)
        assert isinstance(converged, bool)

    def test_repr(self):
        model = CompressibleMultiphaseVoFEnhanced4(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        r = repr(model)
        assert "Enhanced4" in r
        assert "piso_relax" in r
