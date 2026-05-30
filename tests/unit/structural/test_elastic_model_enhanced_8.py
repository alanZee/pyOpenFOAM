"""Tests for enhanced elastic material models v8."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_7 import (
    ThermomechanicalCouplingModel,
    PorousElasticModel,
    FatigueDamageModel,
)
from pyfoam.structural.elastic_model_enhanced_8 import (
    MicromechanicalModel,
    ThermoelasticDamageModel,
    PhaseFieldBrittleFracture,
)


class TestMicromechanicalModel:
    """Test MicromechanicalModel."""

    def test_effective_stiffness_shape(self):
        model = MicromechanicalModel(
            E_matrix=3.0e9, nu_matrix=0.35,
            E_inclusion=200e9, nu_inclusion=0.3,
            volume_fraction=0.3,
        )
        C = model.effective_stiffness
        assert C.shape == (6, 6)

    def test_stress(self):
        model = MicromechanicalModel()
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0  # Tensile stress

    def test_volume_fraction(self):
        model = MicromechanicalModel(volume_fraction=0.3)
        assert model.volume_fraction == 0.3

    def test_stiffness_between_bounds(self):
        """Effective stiffness should be between matrix and inclusion."""
        model = MicromechanicalModel(
            E_matrix=3.0e9, E_inclusion=200e9, volume_fraction=0.3
        )
        C_eff = model.effective_stiffness
        C_m = model.matrix_stiffness
        C_i = model.inclusion_stiffness
        # Voigt-Reuss-Hill average should be between bounds
        assert C_eff[0, 0].item() >= C_m[0, 0].item() * 0.9
        assert C_eff[0, 0].item() <= C_i[0, 0].item() * 1.1

    def test_repr(self):
        model = MicromechanicalModel()
        r = repr(model)
        assert "MicromechanicalModel" in r


class TestThermoelasticDamageModel:
    """Test ThermoelasticDamageModel."""

    def test_stress_no_damage(self):
        model = ThermoelasticDamageModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_damage_reduces_stress(self):
        model = ThermoelasticDamageModel(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress_undamaged = model.stress(strain)

        model.set_damage(0.5)
        stress_damaged = model.stress(strain)
        assert stress_damaged[0].item() < stress_undamaged[0].item()

    def test_thermal_strain(self):
        model = ThermoelasticDamageModel(alpha=12e-6, T_ref=293.15)
        model.set_temperature(393.15)  # dT = 100
        eps_th = model.thermal_strain
        assert abs(eps_th[0].item() - 12e-6 * 100) < 1e-10

    def test_damage_update(self):
        model = ThermoelasticDamageModel(damage_resistance=10.0)
        strain = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        D = model.update_damage(strain, d_time=1.0)
        assert D > 0
        assert D <= 1.0

    def test_reset_state(self):
        model = ThermoelasticDamageModel()
        model.set_damage(0.5)
        model.set_temperature(500.0)
        model.reset_state()
        assert model.damage == 0.0

    def test_repr(self):
        model = ThermoelasticDamageModel()
        r = repr(model)
        assert "ThermoelasticDamageModel" in r


class TestPhaseFieldBrittleFracture:
    """Test PhaseFieldBrittleFracture."""

    def test_phase_field_initially_zero(self):
        model = PhaseFieldBrittleFracture()
        assert model.phase_field == 0.0

    def test_degraded_stiffness(self):
        model = PhaseFieldBrittleFracture()
        model.set_phase_field(0.0)
        factor_undamaged = model.degraded_stiffness_factor
        model.set_phase_field(0.5)
        factor_damaged = model.degraded_stiffness_factor
        assert factor_damaged < factor_undamaged

    def test_stress(self):
        model = PhaseFieldBrittleFracture(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_strain_energy_density(self):
        model = PhaseFieldBrittleFracture(E=210e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        psi = model.compute_strain_energy_density(strain)
        assert psi > 0

    def test_phase_field_update(self):
        model = PhaseFieldBrittleFracture(
            fracture_energy=100.0,
            regularization_length=0.01,
            E=210e9,
        )
        strain = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        phi = model.update_phase_field(strain, d_time=1.0)
        assert phi >= 0
        assert phi <= 1.0

    def test_is_fractured(self):
        model = PhaseFieldBrittleFracture()
        model.set_phase_field(0.99)
        assert model.is_fractured() is True
        model.set_phase_field(0.5)
        assert model.is_fractured() is False

    def test_reset_state(self):
        model = PhaseFieldBrittleFracture()
        model.set_phase_field(0.5)
        model.reset_state()
        assert model.phase_field == 0.0

    def test_repr(self):
        model = PhaseFieldBrittleFracture()
        r = repr(model)
        assert "PhaseFieldBrittleFracture" in r
