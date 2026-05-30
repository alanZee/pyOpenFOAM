"""Tests for enhanced elastic material models v6."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_6 import (
    ChabocheKinematicHardening,
    JohnsonCookModel,
    ConcreteDamagedPlasticityModel,
)


class TestChabocheKinematicHardening:
    """Test Chaboche combined hardening model."""

    def test_creation(self):
        model = ChabocheKinematicHardening()
        assert model.yield_stress == pytest.approx(250e6)
        assert model.accumulated_plastic_strain == 0.0

    def test_elastic_stress(self):
        model = ChabocheKinematicHardening(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_elasticity_matrix(self):
        model = ChabocheKinematicHardening()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_back_stress_initial(self):
        model = ChabocheKinematicHardening()
        alpha = model.back_stress
        assert torch.allclose(alpha, torch.zeros(6, dtype=torch.float64))

    def test_cyclic_update(self):
        model = ChabocheKinematicHardening(
            sigma_y=250e6, C=1e10, gamma=100.0, Q=50e6, b=10.0,
        )
        model.update_cyclic(d_eps_p=0.001, direction=1.0)
        assert model.accumulated_plastic_strain == pytest.approx(0.001)
        assert model.back_stress.norm().item() > 0

    def test_isotropic_hardening(self):
        model = ChabocheKinematicHardening(sigma_y=250e6, Q=50e6, b=10.0)
        sy_before = model.yield_stress
        model.update_cyclic(d_eps_p=0.01)
        assert model.yield_stress > sy_before

    def test_no_update_zero_strain(self):
        model = ChabocheKinematicHardening()
        model.update_cyclic(d_eps_p=0.0, direction=1.0)
        assert model.accumulated_plastic_strain == 0.0

    def test_reset_state(self):
        model = ChabocheKinematicHardening()
        model.update_cyclic(d_eps_p=0.1)
        model.reset_state()
        assert model.accumulated_plastic_strain == 0.0
        assert model.back_stress.norm().item() == 0.0

    def test_repr(self):
        model = ChabocheKinematicHardening()
        r = repr(model)
        assert "Chaboche" in r


class TestJohnsonCookModel:
    """Test Johnson-Cook material model."""

    def test_creation(self):
        model = JohnsonCookModel()
        assert model.accumulated_plastic_strain == 0.0

    def test_elasticity_matrix(self):
        model = JohnsonCookModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_elastic_stress(self):
        model = JohnsonCookModel()
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_flow_stress_reference(self):
        """At reference conditions, flow stress = A + B*eps_p^n."""
        model = JohnsonCookModel(A=350e6, B=275e6, n=0.36)
        sigma = model.flow_stress(plastic_strain=0.0, strain_rate=1.0)
        assert sigma == pytest.approx(350e6)

    def test_flow_stress_strain_hardening(self):
        """Higher plastic strain -> higher flow stress."""
        model = JohnsonCookModel(A=350e6, B=275e6, n=0.36)
        s0 = model.flow_stress(plastic_strain=0.0)
        s1 = model.flow_stress(plastic_strain=0.1)
        assert s1 > s0

    def test_flow_stress_rate_sensitivity(self):
        """Higher strain rate -> higher flow stress."""
        model = JohnsonCookModel(C=0.022)
        s_low = model.flow_stress(plastic_strain=0.1, strain_rate=1.0)
        s_high = model.flow_stress(plastic_strain=0.1, strain_rate=1000.0)
        assert s_high > s_low

    def test_flow_stress_thermal_softening(self):
        """Higher temperature -> lower flow stress."""
        model = JohnsonCookModel(T_ref=293.15, T_melt=1793.0, m=1.0)
        model.set_temperature(293.15)
        s_cold = model.flow_stress(plastic_strain=0.1)
        model.set_temperature(1000.0)
        s_hot = model.flow_stress(plastic_strain=0.1)
        assert s_hot < s_cold

    def test_update_plastic_strain(self):
        model = JohnsonCookModel()
        model.update_plastic_strain(0.01)
        assert model.accumulated_plastic_strain == pytest.approx(0.01)

    def test_no_negative_plastic_strain(self):
        model = JohnsonCookModel()
        model.update_plastic_strain(-0.01)
        assert model.accumulated_plastic_strain == 0.0

    def test_reset_state(self):
        model = JohnsonCookModel()
        model.update_plastic_strain(0.1)
        model.set_temperature(500.0)
        model.reset_state()
        assert model.accumulated_plastic_strain == 0.0

    def test_repr(self):
        model = JohnsonCookModel()
        r = repr(model)
        assert "JohnsonCook" in r


class TestConcreteDamagedPlasticityModel:
    """Test concrete damaged plasticity model."""

    def test_creation(self):
        model = ConcreteDamagedPlasticityModel()
        assert model.tension_damage == 0.0
        assert model.compression_damage == 0.0

    def test_elasticity_matrix(self):
        model = ConcreteDamagedPlasticityModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_elastic_stress(self):
        model = ConcreteDamagedPlasticityModel()
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_no_damage_small_strain(self):
        model = ConcreteDamagedPlasticityModel(E=30e9, ft=3e6)
        # Strain below threshold
        model.update_tension_damage(0.00001)
        assert model.tension_damage == 0.0

    def test_tension_damage_growth(self):
        model = ConcreteDamagedPlasticityModel(E=30e9, ft=3e6, Gf=100.0)
        # eps_0 = ft/E = 3e6/30e9 = 1e-4
        model.update_tension_damage(2e-4)  # Above threshold
        assert model.tension_damage > 0

    def test_tension_damage_capped(self):
        model = ConcreteDamagedPlasticityModel(E=30e9, ft=3e6, Gf=0.1)
        model.update_tension_damage(1.0)  # Very large strain
        assert model.tension_damage <= 1.0

    def test_compression_damage(self):
        model = ConcreteDamagedPlasticityModel(E=30e9, fc=30e6, Gc=10000.0)
        model.update_compression_damage(-0.002)  # Above threshold
        assert model.compression_damage > 0

    def test_elasticity_degrades(self):
        model = ConcreteDamagedPlasticityModel()
        C_initial = model.elasticity_matrix.clone()
        model._d_t = 0.5
        C_damaged = model.elasticity_matrix
        assert C_damaged.norm().item() < C_initial.norm().item()

    def test_not_failed_initially(self):
        model = ConcreteDamagedPlasticityModel()
        assert model.is_failed() is False

    def test_failed_at_full_damage(self):
        model = ConcreteDamagedPlasticityModel()
        model._d_t = 1.0
        assert model.is_failed() is True

    def test_reset_state(self):
        model = ConcreteDamagedPlasticityModel()
        model._d_t = 0.5
        model._d_c = 0.3
        model.reset_state()
        assert model.tension_damage == 0.0
        assert model.compression_damage == 0.0

    def test_repr(self):
        model = ConcreteDamagedPlasticityModel()
        r = repr(model)
        assert "Concrete" in r
