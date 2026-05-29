"""Tests for EnhancedDisplacementSolver3 — v3 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_2 import EnhancedDisplacementSolver2
from pyfoam.structural.displacement_solver_enhanced_3 import (
    EnhancedDisplacementSolver3,
    LargeDeformationResult,
)


class TestLargeDeformationResult:
    """Test LargeDeformationResult dataclass."""

    def test_defaults(self):
        result = LargeDeformationResult()
        assert result.n_steps == 0
        assert result.all_converged is True
        assert result.max_divergence_recoveries == 0

    def test_with_data(self):
        result = LargeDeformationResult(
            n_steps=5,
            all_converged=True,
            max_divergence_recoveries=1,
        )
        assert result.n_steps == 5


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v2(self):
        assert issubclass(EnhancedDisplacementSolver3, EnhancedDisplacementSolver2)


class TestUpdatedLagrangianTangent:
    """Test updated Lagrangian tangent stiffness."""

    def test_zero_displacement(self):
        """Zero displacement gives standard material stiffness."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        u = torch.zeros(2, dtype=torch.float64)
        K = solver.updated_lagrangian_tangent_1d(u, area=0.01, length=1.0)
        assert K.shape == (2, 2)
        # Should be symmetric
        assert torch.allclose(K, K.T, atol=1.0)
        # Diagonal should be positive
        assert K[0, 0].item() > 0

    def test_tension_stiffening(self):
        """Tension increases tangent stiffness (geometric stiffness)."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        u_zero = torch.zeros(2, dtype=torch.float64)
        K_zero = solver.updated_lagrangian_tangent_1d(u_zero, area=0.01, length=1.0)

        u_tension = torch.tensor([0.0, 0.001], dtype=torch.float64)
        K_tension = solver.updated_lagrangian_tangent_1d(
            u_tension, area=0.01, length=1.0,
        )

        # Tangent stiffness should be different (includes geometric stiffness)
        assert not torch.allclose(K_zero, K_tension, atol=1e-3)


class TestNonlinearSolve1D:
    """Test nonlinear 1D solve with geometric nonlinearity."""

    def test_basic_solve(self):
        """Nonlinear solve produces valid result."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        result = solver.solve_nonlinear_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=5,
        )
        assert isinstance(result, LargeDeformationResult)
        assert result.n_steps > 0

    def test_tensile_positive_displacement(self):
        """Tensile load produces positive displacement."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        result = solver.solve_nonlinear_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=10,
        )
        assert result.final_displacement[1].item() > 0

    def test_single_step(self):
        """Single load step works."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        result = solver.solve_nonlinear_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=1,
        )
        assert result.n_steps >= 1

    def test_final_stress(self):
        """Final stress tensor is computed."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        result = solver.solve_nonlinear_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=5,
        )
        assert result.final_stress is not None
        assert result.final_stress.shape == (3, 3)
        # Stress should be positive for tension
        assert result.final_stress[0, 0].item() > 0 or result.final_displacement[1].item() > 0

    def test_convergence_for_small_load(self):
        """Small load should converge."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver3(model)

        result = solver.solve_nonlinear_1d(
            area=0.01,
            length=1.0,
            total_force=1e3,
            n_steps=5,
        )
        assert result.all_converged

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver3(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver3" in r
