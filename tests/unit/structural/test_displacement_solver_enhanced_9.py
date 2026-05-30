"""Tests for EnhancedDisplacementSolver9 -- v9 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_8 import EnhancedDisplacementSolver8
from pyfoam.structural.displacement_solver_enhanced_9 import (
    EnhancedDisplacementSolver9,
    IsogeometricResult,
    MeshlessResult,
    RefinementResult9,
    _bspline_basis,
)


class TestIsogeometricResult:
    def test_defaults(self):
        result = IsogeometricResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestMeshlessResult:
    def test_defaults(self):
        result = MeshlessResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestRefinementResult9:
    def test_defaults(self):
        result = RefinementResult9()
        assert result.n_refined == 0
        assert result.total_dof == 0


class TestBSplineBasis:
    def test_zeroth_order(self):
        # 使用无重复的均匀节点向量
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert _bspline_basis(0, 0, knots, 0.1) == 1.0
        assert _bspline_basis(1, 0, knots, 0.3) == 1.0
        assert _bspline_basis(2, 0, knots, 0.7) == 1.0
        # 不在任何区间内的值
        assert _bspline_basis(0, 0, knots, 0.3) == 0.0

    def test_partition_of_unity(self):
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        total = sum(_bspline_basis(i, 2, knots, 0.5) for i in range(3))
        assert abs(total - 1.0) < 1e-10


class TestInheritance:
    def test_inherits_v8(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver9(model)
        assert isinstance(solver, EnhancedDisplacementSolver8)


class TestIsogeometricSolve:
    def test_returns_iga_result(self):
        n_knots = 8
        force = torch.zeros(n_knots - 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver9.isogeometric_solve_1d(
            area=1.0, length=1.0, n_knots=n_knots,
            external_force=force,
            polynomial_order=2,
            youngs_modulus=210e9,
        )
        assert isinstance(result, IsogeometricResult)
        assert result.converged is True

    def test_displacement_shape(self):
        n_knots = 8
        force = torch.zeros(n_knots - 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver9.isogeometric_solve_1d(
            area=1.0, length=1.0, n_knots=n_knots,
            external_force=force,
        )
        assert result.displacement.numel() == n_knots - 1

    def test_computed_compliance(self):
        n_knots = 8
        force = torch.zeros(n_knots - 1, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver9.isogeometric_solve_1d(
            area=1.0, length=1.0, n_knots=n_knots,
            external_force=force,
        )
        assert result.compliance > 0


class TestMeshlessRPIM:
    def test_returns_meshless_result(self):
        force = torch.zeros(10, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver9.meshless_rpim_solve_1d(
            area=1.0, length=1.0, n_nodes=10,
            external_force=force,
        )
        assert isinstance(result, MeshlessResult)
        assert result.converged is True

    def test_displacement_shape(self):
        force = torch.zeros(10, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver9.meshless_rpim_solve_1d(
            area=1.0, length=1.0, n_nodes=10,
            external_force=force,
        )
        assert result.displacement.numel() == 10


class TestAdaptiveRefinement:
    def test_returns_refinement_result(self):
        u = torch.linspace(0, 1, 20, dtype=torch.float64)
        h = torch.full((19,), 0.05, dtype=torch.float64)
        result = EnhancedDisplacementSolver9.adaptive_refine(u, h)
        assert isinstance(result, RefinementResult9)
        assert result.refined_mesh_size.shape[0] == 19

    def test_error_indicator_shape(self):
        u = torch.linspace(0, 1, 10, dtype=torch.float64)
        h = torch.full((9,), 0.1, dtype=torch.float64)
        result = EnhancedDisplacementSolver9.adaptive_refine(u, h)
        assert result.error_indicator.shape[0] == 9


class TestRepr:
    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver9(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver9" in r
