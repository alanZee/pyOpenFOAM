"""Tests for enhanced elastic material models v3."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.elastic_model_enhanced_3 import (
    OrthotropicPlasticModel,
    ViscoelasticMaxwellModel,
    DamageModel,
)


class TestOrthotropicPlasticModel:
    """Test orthotropic elasticity + Hill yield."""

    def test_creation(self):
        model = OrthotropicPlasticModel()
        assert model.elasticity_matrix.shape == (6, 6)
        assert model.equivalent_plastic_strain == 0.0

    def test_isotropic_default(self):
        """Default parameters give isotropic response."""
        model = OrthotropicPlasticModel()
        hp = model.hill_parameters
        assert hp["F"] > 0

    def test_elastic_below_yield(self):
        """Small strain stays elastic."""
        model = OrthotropicPlasticModel(yield_1=250e6, yield_2=250e6, yield_3=250e6)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert model.equivalent_plastic_strain == 0.0

    def test_plastic_above_yield(self):
        """Large strain triggers plasticity."""
        model = OrthotropicPlasticModel(yield_1=250e6, yield_2=250e6, yield_3=250e6)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert model.equivalent_plastic_strain > 0

    def test_reset_state(self):
        model = OrthotropicPlasticModel()
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain)
        model.reset_state()
        assert model.equivalent_plastic_strain == 0.0

    def test_hill_yield_function(self):
        """Hill yield function evaluates correctly."""
        model = OrthotropicPlasticModel()
        stress = torch.zeros(6, dtype=torch.float64)
        f = model.hill_yield_function(stress)
        # Zero stress: f = -sigma_y^2 < 0
        assert f < 0

    def test_repr(self):
        model = OrthotropicPlasticModel()
        r = repr(model)
        assert "OrthotropicPlastic" in r


class TestViscoelasticMaxwellModel:
    """Test Maxwell viscoelastic model."""

    def test_creation(self):
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        assert model.n_elements == 1
        assert len(model.relaxation_times) == 1

    def test_two_elements(self):
        model = ViscoelasticMaxwellModel(
            E_inf=1e9, E_1=5e8, eta_1=1e6, E_2=2e8, eta_2=2e6,
        )
        assert model.n_elements == 2

    def test_relaxation_time(self):
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        tau = model.relaxation_times[0]
        assert tau == pytest.approx(1e6 / 5e8)  # eta/E = 0.002

    def test_stress_response(self):
        """Stress responds to strain input."""
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain, dt=0.001)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_relaxation(self):
        """Stress relaxes over time under constant strain."""
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)

        # First step
        s1 = model.stress(strain, dt=0.001).clone()
        # Several more steps at the same strain
        for _ in range(10):
            s2 = model.stress(strain, dt=0.001)

        # Stress should have relaxed (decreased)
        assert s2[0].item() <= s1[0].item()

    def test_reset_state(self):
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain, dt=0.001)
        model.reset_state()
        # After reset, q should be 0, so next step gives only E_inf contribution
        stress = model.stress(strain, dt=0.001)
        assert stress[0].item() == pytest.approx(1e9 * 0.001, rel=0.1)

    def test_repr(self):
        model = ViscoelasticMaxwellModel(E_inf=1e9, E_1=5e8, eta_1=1e6)
        r = repr(model)
        assert "Maxwell" in r
        assert "n_elements=1" in r


class TestDamageModel:
    """Test isotropic damage model."""

    def test_creation(self):
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.01)
        assert model.damage == 0.0
        assert model.stiffness_reduction == 1.0

    def test_undamaged_stress(self):
        """Small strain: no damage, stress equals elastic."""
        base = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        model = DamageModel(base, damage_strain=0.01)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        elastic_stress = base.stress(strain)
        assert torch.allclose(stress, elastic_stress, rtol=0.01)

    def test_damage_growth(self):
        """Damage increases with strain."""
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.005)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain)
        assert model.damage > 0

    def test_damage_bounded(self):
        """Damage doesn't exceed max_damage."""
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.0001, max_damage=0.9)
        strain = torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain)
        assert model.damage <= 0.9

    def test_stiffness_degradation(self):
        """Effective stiffness decreases with damage."""
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.005)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain)
        C = model.elasticity_matrix
        C_base = base.elasticity_matrix
        # Degraded stiffness should be less
        assert C.norm().item() < C_base.norm().item()

    def test_damage_irreversible(self):
        """Damage can only grow."""
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.005)
        strain_large = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain_large)
        d1 = model.damage

        strain_small = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain_small)
        d2 = model.damage
        assert d2 >= d1

    def test_reset_state(self):
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.005)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.stress(strain)
        model.reset_state()
        assert model.damage == 0.0

    def test_repr(self):
        base = LinearElasticModel()
        model = DamageModel(base, damage_strain=0.01)
        r = repr(model)
        assert "DamageModel" in r
