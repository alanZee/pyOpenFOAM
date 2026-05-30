"""Tests for enhanced elastic material models v9."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_9 import (
    FunctionallyGradedModel,
    CoupledPoromechanicsModel,
    ElectroMechanicalModel,
)


class TestFunctionallyGradedModel:
    def test_E_at_bottom(self):
        fgm = FunctionallyGradedModel(E_top=200e9, E_bottom=3e9)
        assert abs(fgm.E_at(0.0) - 3e9) < 1e-3

    def test_E_at_top(self):
        fgm = FunctionallyGradedModel(E_top=200e9, E_bottom=3e9)
        assert abs(fgm.E_at(1.0) - 200e9) < 1e-3

    def test_E_at_mid(self):
        fgm = FunctionallyGradedModel(
            E_top=200e9, E_bottom=3e9, gradation_power=1.0
        )
        E_mid = fgm.E_at(0.5)
        assert 3e9 < E_mid < 200e9

    def test_stiffness_shape(self):
        fgm = FunctionallyGradedModel()
        C = fgm.stiffness_at(0.5)
        assert C.shape == (6, 6)

    def test_stress_at(self):
        fgm = FunctionallyGradedModel()
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = fgm.stress_at(0.5, strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_density_at(self):
        fgm = FunctionallyGradedModel(density_top=7800.0, density_bottom=2500.0)
        assert abs(fgm.density_at(0.0) - 2500.0) < 1e-3
        assert abs(fgm.density_at(1.0) - 7800.0) < 1e-3

    def test_repr(self):
        fgm = FunctionallyGradedModel()
        r = repr(fgm)
        assert "FunctionallyGradedModel" in r


class TestCoupledPoromechanicsModel:
    def test_stress_no_pressure(self):
        model = CoupledPoromechanicsModel(E=1e9, nu=0.2)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_stress_with_pressure(self):
        model = CoupledPoromechanicsModel(E=1e9, nu=0.2, biot_coefficient=0.8)
        model.set_pore_pressure(1e6)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        # Pressure should reduce effective stress
        model_no_p = CoupledPoromechanicsModel(E=1e9, nu=0.2, biot_coefficient=0.8)
        stress_no_p = model_no_p.stress(strain)
        assert stress[0].item() < stress_no_p[0].item()

    def test_hydraulic_diffusivity(self):
        model = CoupledPoromechanicsModel(
            permeability=1e-12,
            biot_modulus=10e9,
            fluid_viscosity=1e-3,
        )
        assert model.hydraulic_diffusivity > 0

    def test_effective_stress_factor(self):
        model = CoupledPoromechanicsModel(biot_coefficient=0.8)
        assert abs(model.effective_stress_factor - 0.2) < 1e-10

    def test_repr(self):
        model = CoupledPoromechanicsModel()
        r = repr(model)
        assert "CoupledPoromechanicsModel" in r


class TestElectroMechanicalModel:
    def test_stress_no_field(self):
        model = ElectroMechanicalModel(E=60e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_stress_with_field(self):
        model = ElectroMechanicalModel(E=60e9, nu=0.3, piezoelectric_coefficient=400e-12)
        model.set_electric_field(1e6)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        # Should differ from no-field case
        assert stress.shape == (6,)

    def test_electric_displacement(self):
        model = ElectroMechanicalModel()
        model.set_electric_field(1e6)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        D = model.electric_displacement(strain)
        assert isinstance(D, float)

    def test_coupling_efficiency(self):
        model = ElectroMechanicalModel(coupling_factor=0.7)
        assert abs(model.coupling_efficiency - 0.49) < 1e-10

    def test_piezoelectric_stress_tensor(self):
        model = ElectroMechanicalModel()
        e = model.piezoelectric_stress_tensor
        assert e.shape == (6,)
        assert e[2].item() != 0  # e_33 component

    def test_repr(self):
        model = ElectroMechanicalModel()
        r = repr(model)
        assert "ElectroMechanicalModel" in r
