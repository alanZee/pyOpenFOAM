"""Tests for EnhancedDisplacementSolver6 -- v6 enhanced displacement solver."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_5 import EnhancedDisplacementSolver5
from pyfoam.structural.displacement_solver_enhanced_6 import (
    EnhancedDisplacementSolver6,
    BucklingResult,
    ContactResult6,
    GeometricNonlinearResult,
)


class TestBucklingResult:
    """Test BucklingResult dataclass."""

    def test_defaults(self):
        r = BucklingResult()
        assert r.load_factor == 0.0
        assert r.n_modes == 0


class TestContactResult6:
    """Test ContactResult6 dataclass."""

    def test_defaults(self):
        r = ContactResult6()
        assert r.n_contact_nodes == 0
        assert r.penetration == 0.0


class TestGeometricNonlinearResult:
    """Test GeometricNonlinearResult dataclass."""

    def test_defaults(self):
        r = GeometricNonlinearResult()
        assert r.n_iterations == 0
        assert r.converged is False


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(EnhancedDisplacementSolver6, EnhancedDisplacementSolver5)


class TestGeometricStiffness:
    """Test geometric stiffness matrix."""

    def test_shape(self):
        K_sigma = EnhancedDisplacementSolver6.assemble_geometric_stiffness_1d(
            stress_axial=100.0, area=0.01, length=1.0, n_elements=4,
        )
        assert K_sigma.shape == (5, 5)

    def test_symmetry(self):
        K_sigma = EnhancedDisplacementSolver6.assemble_geometric_stiffness_1d(
            stress_axial=100.0, area=0.01, length=1.0, n_elements=3,
        )
        assert torch.allclose(K_sigma, K_sigma.T)

    def test_zero_stress_zero_matrix(self):
        K_sigma = EnhancedDisplacementSolver6.assemble_geometric_stiffness_1d(
            stress_axial=0.0, area=0.01, length=1.0, n_elements=3,
        )
        assert torch.allclose(K_sigma, torch.zeros_like(K_sigma))


class TestBucklingAnalysis:
    """Test linearized buckling analysis."""

    def test_buckling_result_shape(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver6(model)
        # 使用压缩载荷（负值）以产生正特征值
        result = solver.buckling_analysis_1d(
            area=0.01, length=1.0, n_elements=5,
            applied_load=-1000.0,
        )
        assert isinstance(result, BucklingResult)
        assert result.load_factor > 0
        assert result.buckling_mode.shape[0] == 6  # n_elements + 1

    def test_critical_load_positive(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver6(model)
        result = solver.buckling_analysis_1d(
            area=0.01, length=1.0, n_elements=4,
            applied_load=-100.0,
        )
        assert result.critical_load > 0


class TestContactMechanics:
    """Test contact force computation."""

    def test_no_contact(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver6(model, penalty_stiffness=1e10)
        # All displacements below contact position
        u = torch.tensor([0.0, 0.001, 0.002], dtype=torch.float64)
        result = solver.compute_contact_force_1d(
            u, contact_position=0.01, n_elements=2,
        )
        assert result.n_contact_nodes == 0

    def test_contact_detected(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver6(model, penalty_stiffness=1e10)
        u = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64)
        result = solver.compute_contact_force_1d(
            u, contact_position=0.01, n_elements=2,
        )
        assert result.n_contact_nodes == 1
        assert result.penetration == pytest.approx(0.04)
        assert result.contact_force[2].item() < 0

    def test_contact_pressure(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver6(model, penalty_stiffness=1e10)
        u = torch.tensor([0.0, 0.02], dtype=torch.float64)
        result = solver.compute_contact_force_1d(
            u, contact_position=0.01, n_elements=1,
        )
        # pressure = k_penalty * penetration = 1e10 * 0.01 = 1e8
        assert result.contact_pressure == pytest.approx(1e8)


class TestGeometricNonlinear:
    """Test geometric nonlinear solver."""

    def test_small_load_converges(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver6(model)
        force = torch.tensor([100.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.solve_geometric_nonlinear_1d(
            area=0.01, length=1.0, n_elements=4,
            external_force=force,
        )
        assert result.converged is True
        assert result.displacement.shape[0] == 4

    def test_zero_force_zero_displacement(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver6(model)
        force = torch.zeros(3, dtype=torch.float64)
        result = solver.solve_geometric_nonlinear_1d(
            area=0.01, length=1.0, n_elements=3,
            external_force=force,
        )
        assert torch.allclose(result.displacement, torch.zeros(3, dtype=torch.float64), atol=1e-10)


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver6(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver6" in r
