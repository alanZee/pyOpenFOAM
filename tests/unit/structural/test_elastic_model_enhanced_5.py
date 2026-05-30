"""Tests for enhanced elastic material models v5."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_5 import (
    GradientPlasticityModel,
    CoupledDamagePlasticityModel,
    HyperelasticOgdenModel,
)


class TestGradientPlasticityModel:
    """Test strain gradient plasticity model."""

    def test_creation(self):
        model = GradientPlasticityModel()
        assert model.internal_length == pytest.approx(1e-4)
        assert model.yield_stress == pytest.approx(250e6)
        assert model.accumulated_plastic_strain == 0.0

    def test_elastic_stress(self):
        model = GradientPlasticityModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_effective_yield_stress_no_gradient(self):
        model = GradientPlasticityModel(sigma_y=250e6, l=1e-4)
        sy = model.effective_yield_stress(strain_gradient_norm=0.0)
        assert sy == pytest.approx(250e6)

    def test_effective_yield_stress_with_gradient(self):
        model = GradientPlasticityModel(sigma_y=250e6, l=1e-4)
        sy = model.effective_yield_stress(strain_gradient_norm=100.0)
        # sy = 250e6 + (1e-4)^2 * 100^2 = 250e6 + 1e-4
        assert sy > 250e6

    def test_check_yield(self):
        model = GradientPlasticityModel(sigma_y=100e6)
        small_stress = torch.tensor([1e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert model.check_yield(small_stress) is False

    def test_hardening(self):
        model = GradientPlasticityModel(sigma_y=250e6, hardening_modulus=1e9)
        sy_before = model.yield_stress
        model.update_hardening(d_eps_p=0.01)
        assert model.yield_stress > sy_before
        assert model.accumulated_plastic_strain == pytest.approx(0.01)

    def test_reset_state(self):
        model = GradientPlasticityModel()
        model.update_hardening(d_eps_p=0.1)
        model.reset_state()
        assert model.accumulated_plastic_strain == 0.0
        assert model.yield_stress == pytest.approx(250e6)

    def test_elasticity_matrix(self):
        model = GradientPlasticityModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_repr(self):
        model = GradientPlasticityModel()
        r = repr(model)
        assert "GradientPlasticity" in r


class TestCoupledDamagePlasticityModel:
    """Test Lemaitre damage model coupled with plasticity."""

    def test_creation(self):
        model = CoupledDamagePlasticityModel()
        assert model.damage == 0.0
        assert model.accumulated_plastic_strain == 0.0

    def test_elastic_stress(self):
        model = CoupledDamagePlasticityModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_damage_growth(self):
        model = CoupledDamagePlasticityModel(
            sigma_y=250e6, S=1e6, R=1.0, beta=1.0,
        )
        d_before = model.damage
        model.update_damage(d_eps_p=0.01)
        assert model.damage > d_before

    def test_damage_capped_at_one(self):
        model = CoupledDamagePlasticityModel(S=1e-6, R=2.0)
        for _ in range(1000):
            model.update_damage(d_eps_p=0.1)
        assert model.damage <= 1.0

    def test_not_failed_initially(self):
        model = CoupledDamagePlasticityModel()
        assert model.is_failed() is False

    def test_effective_stress(self):
        model = CoupledDamagePlasticityModel()
        stress = torch.tensor([100e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        effective = model.effective_stress(stress)
        # No damage: effective == nominal
        assert torch.allclose(effective, stress)

    def test_elasticity_degrades(self):
        model = CoupledDamagePlasticityModel()
        C_initial = model.elasticity_matrix.clone()
        model._D = 0.5
        C_damaged = model.elasticity_matrix
        assert C_damaged.norm().item() < C_initial.norm().item()

    def test_zero_d_eps_p_no_damage(self):
        model = CoupledDamagePlasticityModel()
        d = model.update_damage(d_eps_p=0.0)
        assert d == 0.0

    def test_reset_state(self):
        model = CoupledDamagePlasticityModel()
        model.update_damage(d_eps_p=0.1)
        model.reset_state()
        assert model.damage == 0.0

    def test_repr(self):
        model = CoupledDamagePlasticityModel()
        r = repr(model)
        assert "CoupledDamage" in r


class TestHyperelasticOgdenModel:
    """Test Ogden hyperelastic model."""

    def test_creation(self):
        model = HyperelasticOgdenModel()
        assert model.n_terms == 2
        assert model.bulk_modulus == pytest.approx(1e9)

    def test_custom_terms(self):
        model = HyperelasticOgdenModel(mu=[1.0, 0.5], alpha=[2.0, 4.0])
        assert model.n_terms == 2

    def test_mismatched_params_raises(self):
        with pytest.raises(ValueError):
            HyperelasticOgdenModel(mu=[1.0], alpha=[2.0, 4.0])

    def test_strain_energy_undeformed(self):
        model = HyperelasticOgdenModel(mu=[1.0, 0.0], alpha=[2.0, 4.0])
        stretches = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        W = model.strain_energy(stretches)
        assert W == pytest.approx(0.0)  # No deformation

    def test_strain_energy_stretched(self):
        model = HyperelasticOgdenModel(mu=[1.0, 0.0], alpha=[2.0, 4.0])
        stretches = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
        W = model.strain_energy(stretches)
        assert W > 0

    def test_stress_from_stretches_undeformed(self):
        model = HyperelasticOgdenModel(mu=[1.0, 0.0], alpha=[2.0, 4.0])
        stretches = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        stresses = model.stress_from_stretches(stretches)
        # Undeformed: stress should be zero (no volumetric change)
        assert stresses.norm().item() < 1e-10

    def test_elasticity_matrix(self):
        model = HyperelasticOgdenModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_stress_method(self):
        model = HyperelasticOgdenModel()
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_reset_state(self):
        model = HyperelasticOgdenModel()
        model.reset_state()  # Should not raise

    def test_repr(self):
        model = HyperelasticOgdenModel()
        r = repr(model)
        assert "Ogden" in r
