"""Tests for EnhancedDisplacementSolver2 — v2 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced import EnhancedDisplacementSolver
from pyfoam.structural.displacement_solver_enhanced_2 import (
    EnhancedDisplacementSolver2,
    ArcLengthResult,
    LoadStepResult,
)


class TestLoadStepResult:
    """Test LoadStepResult dataclass."""

    def test_defaults(self):
        result = LoadStepResult()
        assert result.load_factor == 1.0
        assert result.n_iterations == 0
        assert result.converged is True

    def test_with_displacement(self):
        disp = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = LoadStepResult(displacement=disp, load_factor=0.5)
        assert torch.allclose(result.displacement, disp)


class TestArcLengthResult:
    """Test ArcLengthResult dataclass."""

    def test_defaults(self):
        result = ArcLengthResult()
        assert result.n_steps == 0
        assert result.all_converged is True

    def test_with_steps(self):
        steps = [
            LoadStepResult(load_factor=0.5, converged=True),
            LoadStepResult(load_factor=1.0, converged=True),
        ]
        result = ArcLengthResult(
            load_steps=steps,
            n_steps=2,
            final_load_factor=1.0,
        )
        assert result.n_steps == 2


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_enhanced(self):
        assert issubclass(EnhancedDisplacementSolver2, EnhancedDisplacementSolver)


class TestEnhancedDisplacementSolver2:
    """Test EnhancedDisplacementSolver2."""

    def test_creation(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver2(model)
        assert solver._model is model

    def test_total_lagrangian_stress(self):
        """Total Lagrangian stress computation."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        grad_u = torch.tensor([
            [0.001, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=torch.float64)
        P = solver.total_lagrangian_stress(grad_u)
        assert P.shape == (3, 3)
        # For small deformation, P[0,0] should be ~ E * 0.001
        assert P[0, 0].item() > 0

    def test_total_lagrangian_zero_deformation(self):
        """Zero deformation gives zero stress."""
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver2(model)

        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        P = solver.total_lagrangian_stress(grad_u)
        assert torch.allclose(P, torch.zeros(3, 3, dtype=torch.float64), atol=1e-3)

    def test_arc_length_basic(self):
        """Arc-length method produces valid result."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        result = solver.solve_arc_length_1d(
            area=0.01,
            length=1.0,
            reference_force=1e6,
            load_increment=0.1,
            max_arc_length=0.01,
            max_steps=5,
        )
        assert isinstance(result, ArcLengthResult)
        assert result.n_steps == 5
        assert len(result.load_steps) == 5

    def test_arc_length_load_factor_grows(self):
        """Load factor increases over steps."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        result = solver.solve_arc_length_1d(
            area=0.01,
            length=1.0,
            reference_force=1e6,
            load_increment=0.1,
            max_steps=5,
        )
        assert result.final_load_factor > 0

    def test_incremental_basic(self):
        """Incremental load stepping produces valid result."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        result = solver.solve_incremental_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=5,
        )
        assert isinstance(result, ArcLengthResult)
        assert result.n_steps == 5

    def test_incremental_displacement_positive(self):
        """Tensile load produces positive displacement."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        result = solver.solve_incremental_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=10,
        )
        assert result.final_displacement[1].item() > 0

    def test_incremental_single_step(self):
        """Single load step."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver2(model)

        result = solver.solve_incremental_1d(
            area=0.01,
            length=1.0,
            total_force=1e6,
            n_steps=1,
        )
        assert result.n_steps == 1

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver2(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver2" in r
