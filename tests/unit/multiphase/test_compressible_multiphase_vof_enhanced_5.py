"""Tests for CompressibleMultiphaseVoFEnhanced5 (v6).

Tests cover:
- Energy coupling parameters
- Latent heat
- Phase change heat source
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.multiphase.compressible_multiphase_vof_enhanced_5 import (
    CompressibleMultiphaseVoFEnhanced5,
)
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_4 import (
    CompressibleMultiphaseVoFEnhanced4,
)


class TestCompressibleMultiphaseVoFEnhanced5:
    """Tests for CompressibleMultiphaseVoFEnhanced5."""

    def test_inherits_from_v5(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert isinstance(model, CompressibleMultiphaseVoFEnhanced4)

    def test_default_params(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model.n_energy_iter == 3
        assert model.energy_relax == pytest.approx(0.7)
        assert model.latent_heat[0] == pytest.approx(2.26e6)

    def test_custom_params(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            n_energy_iter=5,
            energy_relax=0.5,
            latent_heat=[1e6, 0.0],
        )
        assert model.n_energy_iter == 5
        assert model.energy_relax == pytest.approx(0.5)
        assert model.latent_heat == [1e6, 0.0]

    def test_phase_change_heat(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
            latent_heat=[2.26e6, 0.0],
        )
        alphas = torch.tensor([[0.3]], dtype=torch.float64)
        T = torch.tensor([373.15], dtype=torch.float64)
        m_dot = torch.tensor([0.01], dtype=torch.float64)

        Q = model.phase_change_heat(alphas, T, m_dot)
        assert Q.shape == (1,)
        assert Q[0] < 0  # Evaporation cools

    def test_phase_change_heat_no_mass_transfer(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.3]], dtype=torch.float64)
        T = torch.tensor([373.15], dtype=torch.float64)

        Q = model.phase_change_heat(alphas, T)
        assert Q.shape == (1,)
        assert (Q == 0).all()

    def test_repr(self):
        model = CompressibleMultiphaseVoFEnhanced5(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        r = repr(model)
        assert "Enhanced5" in r
        assert "n_energy_iter" in r
