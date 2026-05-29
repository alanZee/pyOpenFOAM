"""Tests for EnhancedStressSolver — iterative stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver import StressSolver
from pyfoam.structural.stress_solver_enhanced import (
    EnhancedStressSolver,
    IterativeStressResult,
)


class TestIterativeStressResult:
    """Test IterativeStressResult dataclass."""

    def test_creation(self):
        stress = torch.zeros(6, dtype=torch.float64)
        result = IterativeStressResult(stress=stress, n_iterations=5, converged=True)
        assert result.n_iterations == 5
        assert result.converged
        assert result.is_plastic is False

    def test_defaults(self):
        stress = torch.zeros(6, dtype=torch.float64)
        result = IterativeStressResult(stress=stress)
        assert result.n_iterations == 0
        assert result.residual == 0.0


class TestEnhancedStressSolver:
    """Test enhanced stress solver."""

    def setup_method(self):
        self.model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        self.yield_criterion = VonMisesYield(yield_stress=250e6)
        self.solver = EnhancedStressSolver(self.model, self.yield_criterion)

    def test_inherits_stress_solver(self):
        assert issubclass(EnhancedStressSolver, StressSolver)

    def test_solve_iterative_converges(self):
        """Iterative solver converges for linear material."""
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = self.solver.solve_iterative(strain)
        assert result.converged
        assert result.n_iterations >= 1
        assert result.stress.shape == (6,)

    def test_solve_iterative_matches_direct(self):
        """Iterative result matches direct solve for linear material."""
        strain = torch.tensor([0.001, -0.0003, -0.0003, 0, 0, 0], dtype=torch.float64)
        iter_result = self.solver.solve_iterative(strain)
        direct = self.solver.solve(strain)
        assert torch.allclose(iter_result.stress, direct, atol=1e-3)

    def test_solve_iterative_has_von_mises(self):
        """Iterative result includes von Mises when yield criterion set."""
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = self.solver.solve_iterative(strain)
        assert result.von_mises is not None

    def test_solve_iterative_no_yield(self):
        """No von Mises when yield criterion not set."""
        solver = EnhancedStressSolver(self.model)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_iterative(strain)
        assert result.von_mises is None

    def test_solve_nonlinear_basic(self):
        """Nonlinear solve with linear model."""
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = self.solver.solve_nonlinear(strain)
        assert result.stress.shape == (6,)

    def test_solve_nonlinear_with_plastic_model(self):
        """Nonlinear solve with plastic model."""
        from pyfoam.structural.elastic_model_enhanced import IsotropicPlasticModel

        plastic = IsotropicPlasticModel(
            youngs_modulus=210e9, poisson_ratio=0.3, yield_stress=250e6
        )
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = self.solver.solve_nonlinear(strain, nonlinear_model=plastic)
        assert result.stress.shape == (6,)

    def test_strain_energy_release_rate(self):
        """G = U * A."""
        stress = torch.tensor([100e6, 0, 0, 0, 0, 0], dtype=torch.float64)
        G = self.solver.strain_energy_release_rate(stress, crack_area=0.01)
        assert G.item() > 0

    def test_stress_invariant_I1(self):
        """I1 = sigma_xx + sigma_yy + sigma_zz."""
        stress = torch.tensor([100e6, 50e6, 30e6, 0, 0, 0], dtype=torch.float64)
        I1 = self.solver.stress_invariant_I1(stress)
        assert I1.item() == pytest.approx(180e6, rel=1e-6)

    def test_stress_invariant_J2_hydrostatic(self):
        """J2 = 0 for hydrostatic stress."""
        p = 100e6
        stress = torch.tensor([p, p, p, 0, 0, 0], dtype=torch.float64)
        J2 = self.solver.stress_invariant_J2(stress)
        assert abs(J2.item()) < 1e-3

    def test_stress_invariant_J2_pure_shear(self):
        """J2 = tau^2 for pure shear."""
        tau = 100e6
        stress = torch.tensor([0, 0, 0, tau, 0, 0], dtype=torch.float64)
        J2 = self.solver.stress_invariant_J2(stress)
        assert J2.item() == pytest.approx(tau ** 2, rel=1e-6)

    def test_repr(self):
        r = repr(self.solver)
        assert "EnhancedStressSolver" in r
