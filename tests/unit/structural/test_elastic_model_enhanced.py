"""Tests for enhanced elastic material models."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced import (
    AnisotropicElasticModel,
    OrthotropicElasticModel,
    IsotropicPlasticModel,
)


class TestAnisotropicElasticModel:
    """Test fully anisotropic elastic model."""

    def test_creation_with_symmetric_matrix(self):
        """Create with valid symmetric positive-definite C."""
        C = torch.eye(6, dtype=torch.float64) * 1e9
        model = AnisotropicElasticModel(C)
        assert model.elasticity_matrix.shape == (6, 6)

    def test_non_square_raises(self):
        """Non-square C raises ValueError."""
        with pytest.raises(ValueError, match="6x6"):
            AnisotropicElasticModel(torch.zeros(3, 4, dtype=torch.float64))

    def test_non_symmetric_raises(self):
        """Non-symmetric C raises ValueError."""
        C = torch.eye(6, dtype=torch.float64)
        C[0, 1] = 1.0  # Not symmetric
        with pytest.raises(ValueError, match="symmetric"):
            AnisotropicElasticModel(C)

    def test_stress_strain(self):
        """sigma = C : epsilon."""
        C = torch.eye(6, dtype=torch.float64) * 2e9
        model = AnisotropicElasticModel(C)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert abs(stress[0].item() - 2e6) < 1e-3

    def test_strain_stress_roundtrip(self):
        """stress(strain(stress)) recovers original."""
        C = torch.eye(6, dtype=torch.float64) * 1e9
        model = AnisotropicElasticModel(C)
        stress_in = torch.tensor([100, 50, 0, 10, 0, 20], dtype=torch.float64)
        strain = model.strain(stress_in)
        stress_out = model.stress(strain)
        assert torch.allclose(stress_out, stress_in, atol=1e-3)

    def test_batch_stress(self):
        """Batch computation works."""
        C = torch.eye(6, dtype=torch.float64) * 1e9
        model = AnisotropicElasticModel(C)
        strain = torch.tensor([[0.001, 0, 0, 0, 0, 0], [0, 0.002, 0, 0, 0, 0]], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (2, 6)

    def test_compliance_matrix(self):
        """S = C^{-1}."""
        C = torch.eye(6, dtype=torch.float64) * 2e9
        model = AnisotropicElasticModel(C)
        S = model.compliance_matrix()
        assert torch.allclose(S, torch.eye(6, dtype=torch.float64) / 2e9, atol=1e-15)

    def test_repr(self):
        C = torch.eye(6, dtype=torch.float64)
        model = AnisotropicElasticModel(C)
        assert "Anisotropic" in repr(model)


class TestOrthotropicElasticModel:
    """Test orthotropic elastic model."""

    def test_creation_default(self):
        """Default: isotropic steel-like."""
        model = OrthotropicElasticModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_elasticity_matrix_symmetric(self):
        model = OrthotropicElasticModel(E1=12e9, E2=1e9, E3=1e9)
        C = model.elasticity_matrix
        assert torch.allclose(C, C.T, atol=1e-3)

    def test_elasticity_matrix_positive_definite(self):
        model = OrthotropicElasticModel()
        C = model.elasticity_matrix
        eigvals = torch.linalg.eigvalsh(C)
        assert (eigvals > 0).all()

    def test_youngs_moduli(self):
        model = OrthotropicElasticModel(E1=12e9, E2=1e9, E3=2e9)
        assert model.youngs_moduli == (12e9, 1e9, 2e9)

    def test_poisson_ratios(self):
        model = OrthotropicElasticModel(nu23=0.4, nu13=0.3, nu12=0.1)
        assert model.poisson_ratios == (0.4, 0.3, 0.1)

    def test_shear_moduli(self):
        model = OrthotropicElasticModel(G23=1e9, G13=2e9, G12=5e9)
        assert model.shear_moduli == (1e9, 2e9, 5e9)

    def test_stress_strain(self):
        """Uniaxial stress along axis 1 — sigma_xx dominated by E1."""
        model = OrthotropicElasticModel(E1=100e9, E2=10e9, E3=10e9)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        # sigma_xx ~ C[0,0] * eps_xx which depends on all elastic constants
        # For orthotropic, C[0,0] ≈ E1*(1-nu23*nu32)/(det) > E1
        # Just check it's in the right ballpark
        assert stress[0].item() > 0
        assert abs(stress[0].item() - 100e9 * 0.001) < 5e6

    def test_stress_strain_roundtrip(self):
        model = OrthotropicElasticModel()
        stress_in = torch.tensor([100e6, 50e6, 0, 10e6, 0, 20e6], dtype=torch.float64)
        strain = model.strain(stress_in)
        stress_out = model.stress(strain)
        assert torch.allclose(stress_out, stress_in, atol=1e3)

    def test_compliance_matrix(self):
        model = OrthotropicElasticModel()
        S = model.compliance_matrix
        assert S.shape == (6, 6)

    def test_repr(self):
        model = OrthotropicElasticModel(E1=12e9, E2=1e9, E3=1e9)
        r = repr(model)
        assert "Orthotropic" in r


class TestIsotropicPlasticModel:
    """Test isotropic elastic + plastic model."""

    def test_creation(self):
        model = IsotropicPlasticModel()
        assert model.yield_stress == 250e6
        assert model.hardening_modulus == 0.0
        assert model.equivalent_plastic_strain == 0.0

    def test_elastic_stress_below_yield(self):
        """Stress below yield stays elastic."""
        model = IsotropicPlasticModel(yield_stress=250e6, hardening_modulus=0.0)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress, is_plastic = model.return_mapping(strain)
        assert not is_plastic
        assert model.equivalent_plastic_strain == 0.0

    def test_plastic_stress_above_yield(self):
        """Stress above yield gets corrected."""
        model = IsotropicPlasticModel(yield_stress=250e6, hardening_modulus=0.0)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress, is_plastic = model.return_mapping(strain)
        assert is_plastic
        assert model.equivalent_plastic_strain > 0

    def test_plastic_stress_within_yield_surface(self):
        """Returned stress should be on or within yield surface."""
        model = IsotropicPlasticModel(yield_stress=250e6, hardening_modulus=1e9)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress, is_plastic = model.return_mapping(strain)

        if is_plastic:
            from pyfoam.structural.elastic_model import VonMisesYield
            criterion = VonMisesYield(yield_stress=model.yield_stress)
            vm = criterion.von_mises_stress(stress)
            assert vm.item() <= model.yield_stress + 1e3  # Small tolerance

    def test_hardening_increases_yield(self):
        """Hardening increases the yield stress."""
        model = IsotropicPlasticModel(
            yield_stress=250e6, hardening_modulus=1e9
        )
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.return_mapping(strain)
        assert model.yield_stress > 250e6

    def test_reset_plastic_strain(self):
        """reset_plastic_strain clears accumulated strain."""
        model = IsotropicPlasticModel(yield_stress=250e6)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.return_mapping(strain)
        assert model.equivalent_plastic_strain > 0
        model.reset_plastic_strain()
        assert model.equivalent_plastic_strain == 0.0

    def test_stress_method(self):
        """stress() method works for single strain."""
        model = IsotropicPlasticModel(yield_stress=250e6)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_stress_batch(self):
        """stress() works for batch of strains."""
        model = IsotropicPlasticModel(yield_stress=250e6)
        model.reset_plastic_strain()
        strain = torch.tensor([
            [0.0001, 0, 0, 0, 0, 0],
            [0.01, 0, 0, 0, 0, 0],
        ], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (2, 6)

    def test_elasticity_matrix(self):
        """Elasticity matrix is available."""
        model = IsotropicPlasticModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_repr(self):
        model = IsotropicPlasticModel(yield_stress=300e6, hardening_modulus=1e9)
        r = repr(model)
        assert "IsotropicPlasticModel" in r
        assert "300" in r or "3.0" in r
