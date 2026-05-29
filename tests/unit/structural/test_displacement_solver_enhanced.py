"""Tests for EnhancedDisplacementSolver — large deformation displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver import DisplacementSolver
from pyfoam.structural.displacement_solver_enhanced import (
    EnhancedDisplacementSolver,
    NonlinearSolveResult,
)


class TestNonlinearSolveResult:
    """Test NonlinearSolveResult dataclass."""

    def test_creation(self):
        result = NonlinearSolveResult(
            displacement=torch.zeros(2, dtype=torch.float64),
            n_iterations=5,
            converged=True,
        )
        assert result.n_iterations == 5
        assert result.converged

    def test_defaults(self):
        result = NonlinearSolveResult(displacement=torch.zeros(2))
        assert result.residual == 0.0
        assert result.strain_energy == 0.0


class TestEnhancedDisplacementSolver:
    """Test enhanced displacement solver with large deformation."""

    def setup_method(self):
        self.model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        self.solver = EnhancedDisplacementSolver(self.model)

    def test_inherits_displacement_solver(self):
        assert issubclass(EnhancedDisplacementSolver, DisplacementSolver)

    # ------------------------------------------------------------------
    # Green-Lagrange strain
    # ------------------------------------------------------------------

    def test_green_lagrange_small_deformation(self):
        """For small deformation, GL strain ≈ Cauchy strain."""
        grad_u = torch.tensor([
            [0.001, 0.002, 0.0],
            [0.002, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        gl_strain = self.solver.green_lagrange_strain(grad_u)
        cauchy_strain = self.solver.strain_from_displacement_gradient(grad_u)
        # For small strains, GL ≈ Cauchy (the difference is 0.5 * grad_u^2)
        # With grad_u entries ~0.002, the difference ~0.000002
        assert torch.allclose(gl_strain, cauchy_strain, atol=1e-4)

    def test_green_lagrange_zero_displacement(self):
        """Zero displacement gives zero strain."""
        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        gl_strain = self.solver.green_lagrange_strain(grad_u)
        assert torch.allclose(gl_strain, torch.zeros(6, dtype=torch.float64))

    def test_green_lagrange_shape(self):
        """Output shape is (6,)."""
        grad_u = torch.eye(3, dtype=torch.float64) * 0.01
        gl_strain = self.solver.green_lagrange_strain(grad_u)
        assert gl_strain.shape == (6,)

    def test_green_lagrange_large_deformation(self):
        """Large deformation gives different GL vs Cauchy."""
        grad_u = torch.tensor([
            [0.5, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        gl_strain = self.solver.green_lagrange_strain(grad_u)
        cauchy_strain = self.solver.strain_from_displacement_gradient(grad_u)
        # For large deformation, they should differ
        diff = (gl_strain - cauchy_strain).abs().max()
        assert diff > 0.01

    # ------------------------------------------------------------------
    # Deformation gradient
    # ------------------------------------------------------------------

    def test_deformation_gradient_identity(self):
        """Zero grad_u gives identity F."""
        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        F = self.solver.deformation_gradient(grad_u)
        assert torch.allclose(F, torch.eye(3, dtype=torch.float64))

    def test_jacobian_det_identity(self):
        """Zero grad_u gives J = 1."""
        grad_u = torch.zeros(3, 3, dtype=torch.float64)
        J = self.solver.jacobian_det(grad_u)
        assert J.item() == pytest.approx(1.0)

    def test_jacobian_det_expansion(self):
        """Uniform expansion: J > 1."""
        grad_u = torch.eye(3, dtype=torch.float64) * 0.1
        J = self.solver.jacobian_det(grad_u)
        assert J.item() > 1.0

    # ------------------------------------------------------------------
    # Geometric stiffness
    # ------------------------------------------------------------------

    def test_geometric_stiffness_1d_tension(self):
        """Positive axial force gives positive stiffness."""
        K_geo = self.solver.geometric_stiffness_1d(axial_force=1000.0, length=1.0)
        assert K_geo.shape == (2, 2)
        assert K_geo[0, 0].item() == pytest.approx(1000.0)

    def test_geometric_stiffness_1d_symmetric(self):
        """Geometric stiffness is symmetric."""
        K_geo = self.solver.geometric_stiffness_1d(axial_force=500.0, length=0.5)
        assert torch.allclose(K_geo, K_geo.T)

    # ------------------------------------------------------------------
    # Newton-Raphson solver
    # ------------------------------------------------------------------

    def test_solve_nonlinear_1d_converges(self):
        """1D nonlinear solve converges."""
        result = self.solver.solve_nonlinear_1d(
            area=0.01, length=1.0, force=1e6,
        )
        assert result.converged
        assert result.n_iterations >= 1

    def test_solve_nonlinear_1d_fixed_end(self):
        """Fixed end has zero displacement."""
        result = self.solver.solve_nonlinear_1d(
            area=0.01, length=1.0, force=1e6,
        )
        assert abs(result.displacement[0].item()) < 1e-12

    def test_solve_nonlinear_1d_free_end(self):
        """Free end has positive displacement under tension."""
        result = self.solver.solve_nonlinear_1d(
            area=0.01, length=1.0, force=1e6,
        )
        assert result.displacement[1].item() > 0

    def test_solve_nonlinear_1d_zero_force(self):
        """Zero force gives zero displacement."""
        result = self.solver.solve_nonlinear_1d(
            area=0.01, length=1.0, force=0.0,
        )
        assert torch.allclose(
            result.displacement, torch.zeros(2, dtype=torch.float64), atol=1e-12
        )

    def test_solve_nonlinear_1d_strain_energy_positive(self):
        """Nonzero force gives positive strain energy."""
        result = self.solver.solve_nonlinear_1d(
            area=0.01, length=1.0, force=1e6,
        )
        assert result.strain_energy > 0

    # ------------------------------------------------------------------
    # Updated Lagrangian
    # ------------------------------------------------------------------

    def test_updated_lagrangian_same_config(self):
        """Same configuration gives zero increment."""
        grad_u = torch.eye(3, dtype=torch.float64) * 0.01
        dE = self.solver.updated_lagrangian_strain_increment(grad_u, grad_u)
        assert torch.allclose(dE, torch.zeros(6, dtype=torch.float64), atol=1e-12)

    def test_updated_lagrangian_increment_shape(self):
        """Output shape is (6,)."""
        grad_u_old = torch.zeros(3, 3, dtype=torch.float64)
        grad_u_new = torch.eye(3, dtype=torch.float64) * 0.01
        dE = self.solver.updated_lagrangian_strain_increment(grad_u_old, grad_u_new)
        assert dE.shape == (6,)

    def test_repr(self):
        r = repr(self.solver)
        assert "EnhancedDisplacementSolver" in r
