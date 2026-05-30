"""Tests for enhanced elastic material models v10."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_10 import (
    TopologyOptimisationModel,
    MagnetostrictiveModel,
    FlexoelectricModel,
)


class TestTopologyOptimisationModel:
    def test_E_at_full_density(self):
        simp = TopologyOptimisationModel(E_0=210e9, penalisation=3.0)
        assert abs(simp.E_at(1.0) - 210e9) < 1e-3

    def test_E_at_half_density(self):
        simp = TopologyOptimisationModel(E_0=210e9, penalisation=3.0)
        E_half = simp.E_at(0.5)
        assert 0 < E_half < 210e9

    def test_E_at_min_density(self):
        simp = TopologyOptimisationModel(E_0=210e9, density_min=1e-3)
        assert simp.E_at(0.0) > 0  # density_min clamped

    def test_stiffness_shape(self):
        simp = TopologyOptimisationModel()
        C = simp.stiffness_at(0.5)
        assert C.shape == (6, 6)

    def test_sensitivity(self):
        simp = TopologyOptimisationModel(E_0=210e9, penalisation=3.0)
        sens = simp.sensitivity(1.0)
        assert sens > 0

    def test_penalisation(self):
        simp = TopologyOptimisationModel(penalisation=3.0)
        assert simp.penalisation == 3.0

    def test_repr(self):
        simp = TopologyOptimisationModel()
        r = repr(simp)
        assert "TopologyOptimisationModel" in r


class TestMagnetostrictiveModel:
    def test_stress_no_field(self):
        model = MagnetostrictiveModel(E=100e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_stress_with_field(self):
        model = MagnetostrictiveModel(E=100e9, nu=0.3, magnetostrictive_coefficient=1e-8)
        model.set_magnetic_field(1e6)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_magnetostrictive_strain(self):
        model = MagnetostrictiveModel(magnetostrictive_coefficient=1e-8)
        model.set_magnetic_field(1e6)
        assert model.magnetostrictive_strain > 0

    def test_magnetic_flux_density(self):
        model = MagnetostrictiveModel(magnetic_permeability=1.256e-6)
        model.set_magnetic_field(1e6)
        assert model.magnetic_flux_density > 0

    def test_repr(self):
        model = MagnetostrictiveModel()
        r = repr(model)
        assert "MagnetostrictiveModel" in r


class TestFlexoelectricModel:
    def test_stress_no_gradient(self):
        model = FlexoelectricModel(E=100e9, nu=0.3)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress[0].item() > 0

    def test_stress_with_gradient(self):
        model = FlexoelectricModel(E=100e9, nu=0.3, flexoelectric_coefficient=1e-5)
        grad = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        model.set_strain_gradient(grad)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_polarisation(self):
        model = FlexoelectricModel(flexoelectric_coefficient=1e-5)
        grad = torch.tensor([0.01, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        model.set_strain_gradient(grad)
        P = model.polarisation()
        assert P.shape == (6,)
        assert P[0].item() > 0

    def test_coefficient(self):
        model = FlexoelectricModel(flexoelectric_coefficient=1e-5)
        assert model.flexoelectric_coefficient == 1e-5

    def test_repr(self):
        model = FlexoelectricModel()
        r = repr(model)
        assert "FlexoelectricModel" in r
