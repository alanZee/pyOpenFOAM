"""Tests for elastic material models and yield criteria."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield


class TestLinearElasticModel:
    """Test isotropic linear elastic constitutive model."""

    def test_default_params(self):
        """Default steel-like parameters."""
        model = LinearElasticModel()
        assert model.youngs_modulus == 210e9
        assert model.poisson_ratio == 0.3

    def test_elasticity_matrix_shape(self):
        """C matrix is 6x6."""
        model = LinearElasticModel()
        C = model.elasticity_matrix
        assert C.shape == (6, 6)

    def test_elasticity_matrix_symmetric(self):
        """C matrix is symmetric."""
        model = LinearElasticModel()
        C = model.elasticity_matrix
        assert torch.allclose(C, C.T, atol=1e-10)

    def test_elasticity_matrix_positive_definite(self):
        """C matrix is positive definite."""
        model = LinearElasticModel()
        C = model.elasticity_matrix
        eigvals = torch.linalg.eigvalsh(C)
        assert (eigvals > 0).all()

    def test_shear_modulus(self):
        """G = E / (2*(1+nu))."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        G = model.shear_modulus
        expected = 210e9 / (2 * 1.3)
        assert abs(G - expected) < 1e3

    def test_bulk_modulus(self):
        """K = E / (3*(1-2*nu))."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        K = model.bulk_modulus
        expected = 210e9 / (3 * 0.4)
        assert abs(K - expected) < 1e3

    def test_invalid_poisson_ratio_raises(self):
        """Poisson's ratio >= 0.5 or < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Poisson"):
            LinearElasticModel(poisson_ratio=0.5)
        with pytest.raises(ValueError, match="Poisson"):
            LinearElasticModel(poisson_ratio=-0.1)

    def test_uniaxial_stress(self):
        """Uniaxial strain eps_xx=0.001 gives sigma_xx = E*(1-nu)*eps/((1+nu)*(1-2nu))."""
        E = 210e9
        nu = 0.3
        model = LinearElasticModel(youngs_modulus=E, poisson_ratio=nu)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        stress = model.stress(strain)
        # sigma_xx = C[0,0] * eps_xx
        expected_sxx = model.elasticity_matrix[0, 0] * 0.001
        assert abs(stress[0].item() - expected_sxx.item()) < 1e3

    def test_stress_strain_roundtrip(self):
        """stress(strain(stress)) recovers original stress."""
        model = LinearElasticModel()
        stress_in = torch.tensor([100e6, -50e6, 0, 10e6, 0, 20e6], dtype=torch.float64)
        strain = model.strain(stress_in)
        stress_out = model.stress(strain)
        assert torch.allclose(stress_out, stress_in, atol=1e3)

    def test_batch_stress(self):
        """Batch stress computation works."""
        model = LinearElasticModel()
        strain = torch.tensor([
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        stress = model.stress(strain)
        assert stress.shape == (2, 6)

    def test_repr(self):
        """__repr__ includes material properties."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        r = repr(model)
        assert "LinearElasticModel" in r
        assert "2.1" in r or "210" in r


class TestVonMisesYield:
    """Test von Mises yield criterion."""

    def test_uniaxial_yield(self):
        """Uniaxial stress sigma_xx = sigma_y gives yielding."""
        sigma_y = 250e6
        criterion = VonMisesYield(yield_stress=sigma_y)
        stress = torch.tensor([sigma_y, 0, 0, 0, 0, 0], dtype=torch.float64)
        vm = criterion.von_mises_stress(stress)
        assert abs(vm.item() - sigma_y) < 1e3

    def test_hydrostatic_no_yield(self):
        """Hydrostatic stress does not cause yielding (von Mises is deviatoric)."""
        criterion = VonMisesYield(yield_stress=250e6)
        p = 1e9  # large hydrostatic pressure
        stress = torch.tensor([p, p, p, 0, 0, 0], dtype=torch.float64)
        vm = criterion.von_mises_stress(stress)
        assert vm.item() < 1e-3

    def test_is_yielding(self):
        """is_yielding returns True when VM stress > yield stress."""
        criterion = VonMisesYield(yield_stress=250e6)
        stress_safe = torch.tensor([200e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        stress_yield = torch.tensor([300e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert not criterion.is_yielding(stress_safe).item()
        assert criterion.is_yielding(stress_yield).item()

    def test_safety_factor(self):
        """Safety factor = sigma_y / sigma_vm."""
        sigma_y = 250e6
        criterion = VonMisesYield(yield_stress=sigma_y)
        stress = torch.tensor([125e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        sf = criterion.safety_factor(stress)
        # VM of uniaxial = |sigma_xx| = 125e6, SF = 250/125 = 2.0
        assert abs(sf.item() - 2.0) < 1e-6

    def test_pure_shear_vm_stress(self):
        """Pure shear: VM = sqrt(3) * tau."""
        tau = 100e6
        criterion = VonMisesYield(yield_stress=250e6)
        stress = torch.tensor([0, 0, 0, tau, 0, 0], dtype=torch.float64)
        vm = criterion.von_mises_stress(stress)
        expected = (3.0 ** 0.5) * tau
        assert abs(vm.item() - expected) / expected < 1e-10

    def test_batch_vm_stress(self):
        """Batch computation works."""
        criterion = VonMisesYield(yield_stress=250e6)
        stress = torch.tensor([
            [250e6, 0, 0, 0, 0, 0],
            [0, 0, 0, 100e6, 0, 0],
        ], dtype=torch.float64)
        vm = criterion.von_mises_stress(stress)
        assert vm.shape == (2,)

    def test_repr(self):
        """__repr__ includes yield stress."""
        criterion = VonMisesYield(yield_stress=250e6)
        r = repr(criterion)
        assert "VonMisesYield" in r
