"""Tests for EnhancedStressSolver2 — v2 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced import EnhancedStressSolver
from pyfoam.structural.stress_solver_enhanced_2 import (
    EnhancedStressSolver2,
    AdaptiveStressResult,
)


class TestAdaptiveStressResult:
    """Test AdaptiveStressResult dataclass."""

    def test_creation(self):
        result = AdaptiveStressResult(
            stress=torch.zeros(6, dtype=torch.float64),
        )
        assert result.n_iterations == 0
        assert result.converged is True
        assert result.final_relaxation == 1.0
        assert result.residual_history == []

    def test_with_history(self):
        result = AdaptiveStressResult(
            stress=torch.zeros(6, dtype=torch.float64),
            residual_history=[1.0, 0.5, 0.1],
        )
        assert len(result.residual_history) == 3


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_enhanced(self):
        assert issubclass(EnhancedStressSolver2, EnhancedStressSolver)


class TestEnhancedStressSolver2:
    """Test EnhancedStressSolver2."""

    def test_creation(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)
        assert solver._model is model

    def test_solve_adaptive_converges(self):
        """Adaptive solve converges for linear elastic."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver2(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_adaptive(strain)
        assert result.converged
        assert result.residual < 1e-8

    def test_solve_adaptive_with_yield(self):
        """Adaptive solve with yield criterion."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        criterion = VonMisesYield(yield_stress=250e6)
        solver = EnhancedStressSolver2(model, criterion)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_adaptive(strain)
        assert result.von_mises is not None

    def test_solve_adaptive_residual_history(self):
        """Residual history is populated."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_adaptive(strain)
        assert len(result.residual_history) > 0

    def test_solve_adaptive_custom_params(self):
        """Custom relaxation parameters."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)

        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_adaptive(
            strain,
            initial_relaxation=0.5,
            min_relaxation=0.1,
            max_relaxation=1.5,
        )
        assert result.converged
        assert result.final_relaxation >= 0.1

    def test_principal_stress_directions(self):
        """Principal stresses and directions computed."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)

        stress = torch.tensor([100e6, 50e6, 0, 0, 0, 0], dtype=torch.float64)
        eigvals, eigvecs = solver.principal_stress_directions(stress)
        assert eigvals.shape == (3,)
        assert eigvecs.shape == (3, 3)
        # Principal stresses should be sorted descending
        assert eigvals[0] >= eigvals[1] >= eigvals[2]

    def test_extrapolate_to_nodes(self):
        """Extrapolation from integration points to nodes."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)

        # 2 integration points, 4 nodes, 6 stress components
        ip_stress = torch.tensor([
            [100e6, 50e6, 0, 0, 0, 0],
            [200e6, 100e6, 0, 0, 0, 0],
        ], dtype=torch.float64)

        # Simple extrapolation matrix
        extrap = torch.tensor([
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=torch.float64)

        node_stress = solver.extrapolate_to_nodes(ip_stress, extrap)
        assert node_stress.shape == (4, 6)

    def test_solve_batch(self):
        """Batch solve produces results for each strain."""
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)

        strains = torch.tensor([
            [0.001, 0, 0, 0, 0, 0],
            [0, 0.002, 0, 0, 0, 0],
        ], dtype=torch.float64)

        results = solver.solve_batch(strains)
        assert len(results) == 2
        assert all(r.converged for r in results)

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver2(model)
        r = repr(solver)
        assert "EnhancedStressSolver2" in r
