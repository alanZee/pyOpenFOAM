"""Tests for enhanced elastic material models v2."""

import pytest
import torch

from pyfoam.structural.elastic_model_enhanced_2 import (
    TransverselyIsotropicModel,
    HyperelasticNeoHookean,
    CombinedPlasticityModel,
)


class TestTransverselyIsotropicModel:
    """Test transversely isotropic elastic model."""

    def test_creation_default(self):
        """Default: isotropic."""
        model = TransverselyIsotropicModel()
        assert model.elasticity_matrix.shape == (6, 6)

    def test_creation_anisotropic(self):
        """Anisotropic with different E_axial and E_transverse."""
        model = TransverselyIsotropicModel(
            E_axial=150e9,
            E_transverse=10e9,
        )
        Ea, Et = model.youngs_moduli
        assert Ea == 150e9
        assert Et == 10e9

    def test_elasticity_symmetric(self):
        model = TransverselyIsotropicModel(E_axial=100e9, E_transverse=10e9)
        C = model.elasticity_matrix
        assert torch.allclose(C, C.T, atol=1e-3)

    def test_elasticity_positive_definite(self):
        model = TransverselyIsotropicModel()
        C = model.elasticity_matrix
        eigvals = torch.linalg.eigvalsh(C)
        assert (eigvals > 0).all()

    def test_stress_strain(self):
        """sigma = C : epsilon."""
        model = TransverselyIsotropicModel()
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)
        assert stress[0].item() > 0

    def test_stress_strain_roundtrip(self):
        model = TransverselyIsotropicModel()
        stress_in = torch.tensor([100e6, 50e6, 0, 10e6, 0, 20e6], dtype=torch.float64)
        strain = model.strain(stress_in)
        stress_out = model.stress(strain)
        assert torch.allclose(stress_out, stress_in, atol=1e3)

    def test_batch_stress(self):
        model = TransverselyIsotropicModel()
        strain = torch.tensor([
            [0.001, 0, 0, 0, 0, 0],
            [0, 0.002, 0, 0, 0, 0],
        ], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (2, 6)

    def test_shear_modulus(self):
        model = TransverselyIsotropicModel(G_axial=6e9)
        assert model.shear_modulus == 6e9

    def test_invalid_symmetry_axis(self):
        with pytest.raises(ValueError, match="symmetry_axis"):
            TransverselyIsotropicModel(symmetry_axis=3)

    def test_repr(self):
        model = TransverselyIsotropicModel(E_axial=100e9, E_transverse=10e9)
        r = repr(model)
        assert "TransverselyIsotropic" in r


class TestHyperelasticNeoHookean:
    """Test Neo-Hookean hyperelastic model."""

    def test_creation(self):
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        assert model.mu == 1e6
        assert model.kappa == 1e9

    def test_zero_deformation(self):
        """Zero deformation: zero strain energy and stress."""
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        W = model.strain_energy(grad_u)
        assert W.item() == pytest.approx(0.0, abs=1e-10)

    def test_strain_energy_positive(self):
        """Non-zero deformation has positive strain energy."""
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        grad_u = torch.tensor([
            [0.01, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=torch.float64)
        W = model.strain_energy(grad_u)
        assert W.item() > 0

    def test_pk2_stress_symmetric(self):
        """PK2 stress is symmetric."""
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        grad_u = torch.tensor([
            [0.01, 0.001, 0],
            [0.001, 0.02, 0],
            [0, 0, 0],
        ], dtype=torch.float64)
        S = model.pk2_stress(grad_u)
        assert torch.allclose(S, S.T, atol=1e-6)

    def test_cauchy_stress_symmetric(self):
        """Cauchy stress is symmetric."""
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        grad_u = torch.tensor([
            [0.01, 0, 0],
            [0, 0.02, 0],
            [0, 0, 0],
        ], dtype=torch.float64)
        sigma = model.cauchy_stress(grad_u)
        assert torch.allclose(sigma, sigma.T, atol=1e-6)

    def test_stress_voigt_shape(self):
        """Voigt stress has correct shape."""
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        grad_u[0, 0] = 0.01
        stress_v = model.stress_voigt(grad_u)
        assert stress_v.shape == (6,)

    def test_repr(self):
        model = HyperelasticNeoHookean(mu=1e6, kappa=1e9)
        r = repr(model)
        assert "NeoHookean" in r


class TestCombinedPlasticityModel:
    """Test combined isotropic + kinematic hardening model."""

    def test_creation(self):
        model = CombinedPlasticityModel()
        assert model.yield_stress == 250e6
        assert model.equivalent_plastic_strain == 0.0

    def test_elastic_below_yield(self):
        """Stress below yield stays elastic."""
        model = CombinedPlasticityModel(yield_stress=250e6)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress, is_plastic = model.return_mapping(strain)
        assert not is_plastic

    def test_plastic_above_yield(self):
        """Stress above yield gets corrected."""
        model = CombinedPlasticityModel(yield_stress=250e6)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress, is_plastic = model.return_mapping(strain)
        assert is_plastic
        assert model.equivalent_plastic_strain > 0

    def test_back_stress_updates(self):
        """Back stress updates during plasticity."""
        model = CombinedPlasticityModel(yield_stress=250e6, kinematic_C=1e9)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.return_mapping(strain)
        bs = model.back_stress
        assert bs.norm().item() > 0

    def test_isotropic_hardening(self):
        """Isotropic hardening variable increases."""
        model = CombinedPlasticityModel(yield_stress=250e6, isotropic_Q=50e6)
        initial_yield = model.yield_stress
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.return_mapping(strain)
        assert model.yield_stress >= initial_yield

    def test_reset_state(self):
        """Reset clears all state."""
        model = CombinedPlasticityModel(yield_stress=250e6)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        model.return_mapping(strain)
        model.reset_state()
        assert model.equivalent_plastic_strain == 0.0
        assert model.back_stress.norm().item() == 0.0

    def test_stress_method_single(self):
        """stress() works for single strain."""
        model = CombinedPlasticityModel()
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (6,)

    def test_stress_method_batch(self):
        """stress() works for batch of strains."""
        model = CombinedPlasticityModel()
        model.reset_state()
        strain = torch.tensor([
            [0.0001, 0, 0, 0, 0, 0],
            [0.01, 0, 0, 0, 0, 0],
        ], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (2, 6)

    def test_elasticity_matrix(self):
        model = CombinedPlasticityModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_repr(self):
        model = CombinedPlasticityModel(yield_stress=300e6)
        r = repr(model)
        assert "CombinedPlasticity" in r
