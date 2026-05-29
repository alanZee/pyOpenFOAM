"""Tests for CompressibleMultiphaseVoFEnhanced3 (v4).

Tests cover:
- Relaxed EOS update
- Corrected mixture density (harmonic mean)
- Transonic Mach limiter
- Inheritance
"""

import pytest
import torch

from pyfoam.multiphase.compressible_multiphase_vof_enhanced_3 import (
    CompressibleMultiphaseVoFEnhanced3,
)
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_2 import (
    CompressibleMultiphaseVoFEnhanced2,
)


class TestCompressibleMultiphaseVoFEnhanced3:
    """Tests for CompressibleMultiphaseVoFEnhanced3."""

    def test_inherits_from_v3(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert isinstance(model, CompressibleMultiphaseVoFEnhanced2)

    def test_default_params(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model.relaxation_factor == pytest.approx(0.8)
        assert model.mixing_correction is True
        assert model.transonic_limiter is True

    def test_relaxed_eos_update_shape(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            n_eos_iter=2,
        )
        alphas = torch.tensor([[0.3]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        rho_target = torch.tensor([500.0], dtype=torch.float64)

        p_new, T_new = model.relaxed_eos_update(alphas, p, T, rho_target)
        assert p_new.shape == (1,)
        assert T_new.shape == (1,)
        assert float(p_new[0].item()) > 0
        assert float(T_new[0].item()) > 0

    def test_corrected_mixture_density_shape(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.3], [0.8]], dtype=torch.float64)
        p = torch.tensor([101325.0, 101325.0], dtype=torch.float64)
        T = torch.tensor([300.0, 300.0], dtype=torch.float64)

        rho = model.corrected_mixture_density(alphas, p, T)
        assert rho.shape == (2,)
        assert (rho > 0).all()

    def test_transonic_limiter_reduces_velocity(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            Ma_max=0.5,
        )
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        # High velocity (supersonic)
        U_high = torch.tensor([500.0], dtype=torch.float64)

        U_limited = model.transonic_mach_limiter(alphas, p, T, U_high)
        assert float(U_limited[0].item()) <= float(U_high[0].item())

    def test_transonic_limiter_disabled(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            transonic_limiter=False,
        )
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        U = torch.tensor([500.0], dtype=torch.float64)

        U_out = model.transonic_mach_limiter(alphas, p, T, U)
        assert float(U_out[0].item()) == pytest.approx(500.0)

    def test_repr(self):
        model = CompressibleMultiphaseVoFEnhanced3(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        r = repr(model)
        assert "Enhanced3" in r
        assert "relaxation" in r
