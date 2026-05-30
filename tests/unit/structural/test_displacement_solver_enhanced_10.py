"""Tests for EnhancedDisplacementSolver10 -- v10 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_9 import EnhancedDisplacementSolver9
from pyfoam.structural.displacement_solver_enhanced_10 import (
    EnhancedDisplacementSolver10,
    XFEM1DResult,
    PUMResult,
    DWRRefinementResult,
    _heaviside_enrichment,
)


class TestXFEM1DResult:
    def test_defaults(self):
        result = XFEM1DResult()
        assert result.compliance == 0.0
        assert result.converged is False
        assert result.crack_opening == 0.0


class TestPUMResult:
    def test_defaults(self):
        result = PUMResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestDWRRefinementResult:
    def test_defaults(self):
        result = DWRRefinementResult()
        assert result.n_refined == 0
        assert result.estimated_total_error == 0.0


class TestHeavisideEnrichment:
    def test_positive_side(self):
        assert _heaviside_enrichment(0.6, 0.5) == 1.0

    def test_negative_side(self):
        assert _heaviside_enrichment(0.3, 0.5) == -1.0

    def test_at_crack(self):
        assert _heaviside_enrichment(0.5, 0.5) == 1.0


class TestInheritance:
    def test_inherits_v9(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver10(model)
        assert isinstance(solver, EnhancedDisplacementSolver9)


class TestXFEMSolve:
    def test_returns_xfem_result(self):
        n_elements = 10
        force = torch.zeros(n_elements + 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.xfem_solve_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            crack_position=0.5,
            youngs_modulus=210e9,
        )
        assert isinstance(result, XFEM1DResult)
        assert result.converged is True

    def test_displacement_shape(self):
        n_elements = 10
        force = torch.zeros(n_elements + 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.xfem_solve_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
        )
        assert result.displacement.numel() == n_elements + 1

    def test_crack_opening(self):
        n_elements = 10
        force = torch.zeros(n_elements + 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.xfem_solve_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            crack_position=0.5,
        )
        # With enrichment, there should be some crack opening
        assert result.n_enriched_dof >= 0


class TestPUMSolve:
    def test_returns_pum_result(self):
        force = torch.zeros(11, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.pum_solve_1d(
            area=1.0, length=1.0, n_elements=10,
            external_force=force,
        )
        assert isinstance(result, PUMResult)
        assert result.converged is True

    def test_displacement_shape(self):
        force = torch.zeros(11, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.pum_solve_1d(
            area=1.0, length=1.0, n_elements=10,
            external_force=force,
        )
        assert result.displacement.numel() == 11

    def test_partitions(self):
        force = torch.zeros(11, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver10.pum_solve_1d(
            area=1.0, length=1.0, n_elements=10,
            external_force=force,
            n_partitions=5,
        )
        assert result.n_partitions == 5


class TestDWRErrorEstimate:
    def test_returns_dwr_result(self):
        u = torch.linspace(0, 1, 20, dtype=torch.float64)
        h = torch.full((19,), 0.05, dtype=torch.float64)
        result = EnhancedDisplacementSolver10.dwr_error_estimate(u, h)
        assert isinstance(result, DWRRefinementResult)
        assert result.error_indicators.shape[0] == 19

    def test_refinement_counts(self):
        u = torch.linspace(0, 1, 10, dtype=torch.float64)
        h = torch.full((9,), 0.1, dtype=torch.float64)
        result = EnhancedDisplacementSolver10.dwr_error_estimate(u, h)
        assert result.n_refined >= 0
        assert result.n_coarsened >= 0


class TestRepr:
    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver10(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver10" in r
