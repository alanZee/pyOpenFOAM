"""Tests for CompressibleMultiphaseVoFEnhanced.

Tests cover:
- EOS coupling (internal energy, drho/dp)
- Pressure iteration
- Mach limiter
- Properties and repr
"""

import pytest
import torch

from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF
from pyfoam.multiphase.compressible_multiphase_vof_enhanced import (
    CompressibleMultiphaseVoFEnhanced,
)


class TestCompressibleMultiphaseVoFEnhanced:
    """Tests for CompressibleMultiphaseVoFEnhanced."""

    def test_inherits_from_base(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
        assert isinstance(model, CompressibleMultiphaseVoF)

    def test_default_params(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
        assert model.n_piso == 2
        assert model.Ma_max == pytest.approx(0.9)

    def test_custom_params(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
            n_piso=3, Ma_max=0.8,
        )
        assert model.n_piso == 3
        assert model.Ma_max == pytest.approx(0.8)

    def test_mixture_internal_energy(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        e = model.mixture_internal_energy(alphas, p, T)
        assert e.shape == (1,)
        assert float(e[0].item()) > 0

    def test_limit_mach(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
            Ma_max=0.5,
        )
        U_mag = torch.tensor([100.0, 200.0], dtype=torch.float64)
        a_mix = torch.tensor([340.0, 340.0], dtype=torch.float64)
        U_limited = model.limit_mach(U_mag, a_mix)
        assert float(U_limited[0].item()) == pytest.approx(100.0)
        assert float(U_limited[1].item()) == pytest.approx(0.5 * 340.0)

    def test_iterate_pressure(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
            n_piso=3,
        )
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        rho_target = model.mixture_density(alphas, p, T)
        p_corrected = model.iterate_pressure(alphas, p, T, rho_target)
        assert p_corrected.shape == (1,)
        assert float(p_corrected[0].item()) > 0

    def test_phase_density(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        rho_gas = model.phase_density(0, p, T)
        assert float(rho_gas[0].item()) == pytest.approx(101325.0 / (287.0 * 300.0), rel=1e-3)

    def test_repr(self):
        model = CompressibleMultiphaseVoFEnhanced(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0], [1.8e-5, 1e-3],
            R=[287.0, None], gamma=[1.4, None],
        )
        r = repr(model)
        assert "CompressibleMultiphaseVoFEnhanced" in r
        assert "n_piso" in r
