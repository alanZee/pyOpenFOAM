"""Tests for EnhancedStressSolver3 — v3 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_2 import EnhancedStressSolver2
from pyfoam.structural.stress_solver_enhanced_3 import (
    EnhancedStressSolver3,
    NonlinearStressResult,
)


class TestNonlinearStressResult:
    """Test NonlinearStressResult dataclass."""

    def test_defaults(self):
        result = NonlinearStressResult(
            stress=torch.zeros(6, dtype=torch.float64),
        )
        assert result.n_iterations == 0
        assert result.converged is True
        assert result.residual_history == []
        assert result.line_search_steps == 0
        assert result.final_step_size == 1.0

    def test_with_data(self):
        result = NonlinearStressResult(
            stress=torch.zeros(6, dtype=torch.float64),
            n_iterations=5,
            residual_history=[1.0, 0.5, 0.1, 0.01, 0.001],
        )
        assert len(result.residual_history) == 5


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v2(self):
        assert issubclass(EnhancedStressSolver3, EnhancedStressSolver2)


class TestEnhancedStressSolver3:
    """Test EnhancedStressSolver3."""

    def test_creation(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver3(model)
        assert solver._model is model

    def test_solve_nonlinear_converges(self):
        """Nonlinear solve converges for linear elastic (single iteration)."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver3(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_nonlinear(strain)
        assert result.converged
        assert result.n_iterations <= 3  # Linear should converge quickly

    def test_solve_nonlinear_with_yield(self):
        """Nonlinear solve with yield criterion."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        criterion = VonMisesYield(yield_stress=250e6)
        solver = EnhancedStressSolver3(model, criterion)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_nonlinear(strain)
        assert result.converged

    def test_residual_history(self):
        """Residual history is populated."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver3(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_nonlinear(strain)
        assert len(result.residual_history) > 0

    def test_consistent_tangent(self):
        """Consistent tangent stiffness computed."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver3(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        C = solver.consistent_tangent(strain)
        assert C.shape == (6, 6)
        # Should be approximately the elasticity matrix
        C_expected = model.elasticity_matrix
        assert torch.allclose(C, C_expected, rtol=0.1)

    def test_consistent_tangent_numerical(self):
        """Consistent tangent matches analytical for linear elastic."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver3(model)

        strain = torch.tensor([0.0, 0.001, 0, 0, 0, 0], dtype=torch.float64)
        C = solver.consistent_tangent(strain, delta=1e-6)
        C_expected = model.elasticity_matrix
        # Should be close to the linear elastic matrix
        for i in range(6):
            for j in range(6):
                if abs(C_expected[i, j].item()) > 1e6:
                    assert abs(C[i, j].item() - C_expected[i, j].item()) / abs(
                        C_expected[i, j].item()
                    ) < 0.05

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver3(model)
        r = repr(solver)
        assert "EnhancedStressSolver3" in r
