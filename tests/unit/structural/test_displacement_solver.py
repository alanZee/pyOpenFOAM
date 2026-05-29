"""Tests for displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver import DisplacementSolver


class TestDisplacementSolver:
    """Test the displacement solver."""

    def setup_method(self):
        """Set up solver with steel-like material."""
        self.model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        self.solver = DisplacementSolver(self.model)

    def test_strain_from_displacement_gradient(self):
        """Strain = 0.5 * (grad_u + grad_u^T)."""
        grad_u = torch.tensor([
            [0.001, 0.002, 0.0],
            [0.0,   0.0,   0.0],
            [0.0,   0.0,   0.0],
        ], dtype=torch.float64)
        strain = self.solver.strain_from_displacement_gradient(grad_u)
        assert strain.shape == (6,)
        # eps_xx = 0.001
        assert abs(strain[0].item() - 0.001) < 1e-12
        # gamma_xy = 2 * 0.5 * (0.002 + 0) = 0.002
        assert abs(strain[5].item() - 0.002) < 1e-12

    def test_strain_symmetric(self):
        """Symmetric grad_u gives gamma = 2 * eps_offdiagonal."""
        grad_u = torch.tensor([
            [0.0, 0.001, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        strain = self.solver.strain_from_displacement_gradient(grad_u)
        # eps_xx = 0, eps_yy = 0, gamma_xy = 2 * 0.001 = 0.002
        assert abs(strain[5].item() - 0.002) < 1e-12

    def test_solve_from_strain(self):
        """Displacement from uniform strain."""
        strain = torch.tensor([0.001, -0.0003, -0.0003, 0, 0, 0], dtype=torch.float64)
        disp = self.solver.solve_from_strain(strain, element_size=1.0)
        assert torch.allclose(disp, strain[:3])

    def test_solve_from_strain_scaled(self):
        """Displacement scales with element size."""
        strain = torch.tensor([0.001, 0.0, 0.0, 0, 0, 0], dtype=torch.float64)
        disp = self.solver.solve_from_strain(strain, element_size=0.5)
        assert abs(disp[0].item() - 0.0005) < 1e-12

    def test_stiffness_matrix_1d(self):
        """1D bar stiffness matrix is 2x2."""
        K = self.solver.stiffness_matrix_1d(area=0.01, length=1.0)
        assert K.shape == (2, 2)
        # K = E*A/L
        k = 210e9 * 0.01 / 1.0
        assert abs(K[0, 0].item() - k) < 1e3
        assert abs(K[0, 1].item() + k) < 1e3

    def test_stiffness_matrix_1d_symmetric(self):
        """1D stiffness matrix is symmetric."""
        K = self.solver.stiffness_matrix_1d(area=0.01, length=1.0)
        assert torch.allclose(K, K.T)

    def test_stiffness_matrix_2d_truss(self):
        """2D truss stiffness matrix is 4x4."""
        K = self.solver.stiffness_matrix_2d_truss(area=0.01, length=1.0, angle=0.0)
        assert K.shape == (4, 4)

    def test_stiffness_matrix_2d_truss_symmetric(self):
        """2D truss stiffness matrix is symmetric."""
        K = self.solver.stiffness_matrix_2d_truss(area=0.01, length=1.0, angle=0.5)
        assert torch.allclose(K, K.T, atol=1e-10)

    def test_solve_bar_simple(self):
        """Solve a simple 2-node bar with one end fixed."""
        # Bar with E*A/L = 100
        K = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float64)
        F = torch.tensor([0.0, 10.0], dtype=torch.float64)
        # Fix node 0 (DOF 0)
        u = self.solver.solve_bar(K, F, fixed_dofs=[0])
        # u_1 = F_1 / K_11 = 10 / 100 = 0.1
        assert abs(u[0].item()) < 1e-12
        assert abs(u[1].item() - 0.1) < 1e-10

    def test_solve_bar_all_fixed(self):
        """All DOFs fixed gives zero displacement."""
        K = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float64)
        F = torch.tensor([10.0, 10.0], dtype=torch.float64)
        u = self.solver.solve_bar(K, F, fixed_dofs=[0, 1])
        assert torch.allclose(u, torch.zeros(2, dtype=torch.float64))

    def test_strain_energy(self):
        """Strain energy is positive for nonzero strain."""
        strain = torch.tensor([0.001, 0.0, 0.0, 0, 0, 0], dtype=torch.float64)
        U = self.solver.strain_energy(strain)
        assert U.item() > 0

    def test_strain_energy_zero_strain(self):
        """Zero strain gives zero strain energy."""
        strain = torch.zeros(6, dtype=torch.float64)
        U = self.solver.strain_energy(strain)
        assert abs(U.item()) < 1e-20

    def test_repr(self):
        """__repr__ includes model info."""
        r = repr(self.solver)
        assert "DisplacementSolver" in r
