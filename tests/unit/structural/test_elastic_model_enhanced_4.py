"""Tests for enhanced elastic material models v4."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_4 import (
    GursonDamageModel,
    CrystalPlasticityModel,
    PhaseFieldFractureModel,
)


class TestGursonDamageModel:
    """Test Gurson-Tvergaard-Needleman porous plasticity."""

    def test_creation(self):
        model = GursonDamageModel()
        assert model.void_fraction == pytest.approx(0.001)
        assert model.void_fraction_initial == pytest.approx(0.001)

    def test_elastic_stress(self):
        model = GursonDamageModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_yield_function_below_yield(self):
        model = GursonDamageModel(sigma_y=250e6)
        stress = torch.tensor([1e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        f = model.gurson_yield_function(stress)
        assert f < 0  # below yield

    def test_yield_function_above_yield(self):
        model = GursonDamageModel(sigma_y=100e6, f0=0.001)
        stress = torch.tensor([500e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        f = model.gurson_yield_function(stress)
        assert f > 0  # above yield

    def test_void_fraction_growth(self):
        model = GursonDamageModel(f0=0.001)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        f_before = model.void_fraction
        model.update_void_fraction(strain, d_strain=0.01)
        assert model.void_fraction > f_before

    def test_void_fraction_capped(self):
        model = GursonDamageModel(f0=0.5)
        strain = torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.update_void_fraction(strain, d_strain=10.0)
        assert model.void_fraction <= 0.99

    def test_reset_state(self):
        model = GursonDamageModel(f0=0.001)
        model.update_void_fraction(
            torch.tensor([0.1, 0, 0, 0, 0, 0], dtype=torch.float64),
            d_strain=0.1,
        )
        model.reset_state()
        assert model.void_fraction == pytest.approx(0.001)

    def test_elasticity_matrix(self):
        model = GursonDamageModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_repr(self):
        model = GursonDamageModel()
        r = repr(model)
        assert "GursonDamage" in r


class TestCrystalPlasticityModel:
    """Test crystal plasticity model."""

    def test_creation(self):
        model = CrystalPlasticityModel()
        assert model.n_slip_systems == 12
        assert model.accumulated_slip == 0.0

    def test_custom_slip_systems(self):
        model = CrystalPlasticityModel(n_slip_systems=6)
        assert model.n_slip_systems == 6

    def test_elastic_stress(self):
        model = CrystalPlasticityModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_resolved_shear_stresses(self):
        model = CrystalPlasticityModel(n_slip_systems=4)
        stress = torch.tensor([100e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        tau = model.resolved_shear_stresses(stress)
        assert tau.shape == (4,)
        assert tau.min().item() >= 0

    def test_check_yielding(self):
        model = CrystalPlasticityModel(tau_crss=10e6)
        small_stress = torch.tensor([1e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert model.check_yielding(small_stress) is False

        large_stress = torch.tensor([500e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert model.check_yielding(large_stress) is True

    def test_hardening(self):
        model = CrystalPlasticityModel(tau_crss=50e6, hardening_rate=500e6)
        crss_before = model.critical_resolved_shear_stress
        model.update_hardening(d_gamma=0.01)
        assert model.critical_resolved_shear_stress > crss_before
        assert model.accumulated_slip == pytest.approx(0.01)

    def test_reset_state(self):
        model = CrystalPlasticityModel()
        model.update_hardening(d_gamma=0.1)
        model.reset_state()
        assert model.accumulated_slip == 0.0

    def test_repr(self):
        model = CrystalPlasticityModel()
        r = repr(model)
        assert "CrystalPlasticity" in r


class TestPhaseFieldFractureModel:
    """Test phase-field fracture model."""

    def test_creation(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base, Gc=1000.0, length_scale=0.01)
        assert model.phase_field == 1.0
        assert model.damage == 0.0
        assert model.critical_energy_release_rate == 1000.0

    def test_degradation_function(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        assert model.degradation(1.0) == pytest.approx(1.0 + 1e-6)
        assert model.degradation(0.0) == pytest.approx(1e-6)

    def test_elastic_energy_density(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        psi = model.elastic_energy_density(strain)
        assert psi > 0

    def test_degraded_stress(self):
        base = LinearElasticModel(youngs_modulus=210e9)
        model = PhaseFieldFractureModel(base)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)

        stress_intact = model.degraded_stress(strain)
        model._phi = 0.5
        stress_damaged = model.degraded_stress(strain)

        # Damaged stress should be smaller
        assert stress_damaged.norm().item() < stress_intact.norm().item()

    def test_degraded_stiffness(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        C_intact = model.degraded_stiffness()
        model._phi = 0.5
        C_damaged = model.degraded_stiffness()
        assert C_damaged.norm().item() < C_intact.norm().item()

    def test_phase_field_evolution(self):
        base = LinearElasticModel(youngs_modulus=210e9)
        model = PhaseFieldFractureModel(base, Gc=1.0, length_scale=0.001)
        large_strain = torch.tensor([0.1, 0, 0, 0, 0, 0], dtype=torch.float64)
        phi_before = model.phase_field
        model.update_phase_field(large_strain, dt=1.0)
        # High strain energy should cause damage
        assert model.phase_field <= phi_before

    def test_reset_state(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        model._phi = 0.3
        model.reset_state()
        assert model.phase_field == 1.0

    def test_elasticity_matrix(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_stress_method(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_repr(self):
        base = LinearElasticModel()
        model = PhaseFieldFractureModel(base)
        r = repr(model)
        assert "PhaseField" in r
