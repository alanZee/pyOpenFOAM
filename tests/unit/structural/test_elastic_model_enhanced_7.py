"""Tests for enhanced elastic material models v7."""

import pytest
import torch

from pyfoam.structural.elastic_model_enhanced_7 import (
    ThermomechanicalCouplingModel,
    PorousElasticModel,
    FatigueDamageModel,
)


class TestThermomechanicalCouplingModel:
    """Test coupled thermo-mechanical model."""

    def test_initial_E(self):
        model = ThermomechanicalCouplingModel(E=210e9)
        assert model.current_E == pytest.approx(210e9)

    def test_temperature_affects_E(self):
        model = ThermomechanicalCouplingModel(E=210e9, beta=1e-4, T_ref=293.15)
        model.set_temperature(500.0)
        assert model.current_E < 210e9

    def test_thermal_strain(self):
        model = ThermomechanicalCouplingModel(
            alpha=12e-6, T_ref=293.15
        )
        model.set_temperature(393.15)  # dT = 100
        eps_th = model.thermal_strain
        assert eps_th[0].item() == pytest.approx(12e-6 * 100)
        assert eps_th[3].item() == pytest.approx(0.0)  # No shear

    def test_stress_thermal_expansion(self):
        model = ThermomechanicalCouplingModel(
            E=210e9, nu=0.3, alpha=12e-6, T_ref=293.15
        )
        model.set_temperature(393.15)
        # Zero total strain => pure thermal stress
        strain = torch.zeros(6, dtype=torch.float64)
        stress = model.stress(strain)
        # Thermal expansion induces compressive stress
        assert stress[0].item() < 0

    def test_elasticity_matrix_shape(self):
        model = ThermomechanicalCouplingModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_thermal_diffusivity(self):
        model = ThermomechanicalCouplingModel(
            thermal_conductivity=50.0,
            specific_heat=500.0,
            density=7800.0,
        )
        alpha = model.thermal_diffusivity
        expected = 50.0 / (7800.0 * 500.0)
        assert abs(alpha - expected) < 1e-10

    def test_reset(self):
        model = ThermomechanicalCouplingModel(T_ref=293.15)
        model.set_temperature(500.0)
        model.reset_state()
        assert model.temperature_change == pytest.approx(0.0)

    def test_repr(self):
        model = ThermomechanicalCouplingModel(E=210e9)
        r = repr(model)
        assert "ThermomechanicalCouplingModel" in r


class TestPorousElasticModel:
    """Test poroelastic model."""

    def test_effective_stress(self):
        model = PorousElasticModel(E=10e9, nu=0.2, biot_coefficient=0.8)
        model.set_pore_pressure(1e6)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        # Effective stress should be less than total stress due to pore pressure
        total_stress = model._model.stress(strain)
        assert stress[0].item() < total_stress[0].item()

    def test_zero_pore_pressure(self):
        model = PorousElasticModel()
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        total = model._model.stress(strain)
        assert torch.allclose(stress, total)

    def test_elasticity_matrix(self):
        model = PorousElasticModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_undrained_poisson_ratio(self):
        model = PorousElasticModel(E=10e9, nu=0.2, biot_coefficient=0.8, biot_modulus=20e9)
        nu_u = model.undrained_poisson_ratio
        # Undrained should be higher than drained
        assert nu_u > 0.2

    def test_repr(self):
        model = PorousElasticModel()
        r = repr(model)
        assert "PorousElasticModel" in r


class TestFatigueDamageModel:
    """Test fatigue damage model."""

    def test_initial_no_damage(self):
        model = FatigueDamageModel()
        assert model.cumulative_damage == pytest.approx(0.0)

    def test_damage_increases(self):
        model = FatigueDamageModel()
        model.update_fatigue(200e6, 1000, yield_stress=250e6)
        assert model.cumulative_damage > 0

    def test_below_endurance_no_damage(self):
        model = FatigueDamageModel(endurance_limit_ratio=0.5)
        model.update_fatigue(100e6, 1000, yield_stress=250e6)
        assert model.cumulative_damage == pytest.approx(0.0)

    def test_current_E_decreases(self):
        model = FatigueDamageModel(E=210e9)
        model.update_fatigue(300e6, 10000, yield_stress=250e6)
        assert model.current_E < 210e9

    def test_is_failed(self):
        model = FatigueDamageModel()
        model._D = 1.0
        assert model.is_failed() is True

    def test_stress_degradation(self):
        model = FatigueDamageModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress_undamaged = model.stress(strain)
        model._D = 0.5
        stress_damaged = model.stress(strain)
        assert stress_damaged[0].item() < stress_undamaged[0].item()

    def test_elasticity_matrix_shape(self):
        model = FatigueDamageModel()
        assert model.elasticity_matrix.shape == (6, 6)

    def test_reset(self):
        model = FatigueDamageModel()
        model.update_fatigue(300e6, 1000, yield_stress=250e6)
        model.reset_state()
        assert model.cumulative_damage == pytest.approx(0.0)

    def test_repr(self):
        model = FatigueDamageModel()
        r = repr(model)
        assert "FatigueDamageModel" in r
