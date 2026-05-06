"""Tests for PBiCGSTAB (Preconditioned Bi-Conjugate Gradient Stabilised) solver."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver
from pyfoam.solvers.linear_solver import create_solver


class TestPBiCGSTABBasic:
    """Basic PBiCGSTAB solver tests."""

    def test_identity_system(self):
        """Solve I x = b → x = b."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-8)

    def test_symmetric_2cell(self):
        """PBiCGSTAB should also work for symmetric systems."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        # b = A @ [1, 2] = [2, 7]
        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([1.0, 2.0], dtype=CFD_DTYPE), atol=1e-6)

    def test_asymmetric_2cell(self):
        """Solve an asymmetric 2x2 system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 5.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0], dtype=CFD_DTYPE)

        # A = [[4, -1], [-2, 5]]
        # x_exact = [1, 1]
        # b = [4-1, -2+5] = [3, 3]
        b = torch.tensor([3.0, 3.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([1.0, 1.0], dtype=CFD_DTYPE), atol=1e-4)

    def test_asymmetric_3cell(self, asymmetric_matrix_3cell):
        """PBiCGSTAB on a 3-cell asymmetric system."""
        mat = asymmetric_matrix_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-8, max_iter=500, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-4

    def test_with_dilu_preconditioner(self, asymmetric_matrix_3cell):
        """PBiCGSTAB with DILU preconditioner."""
        mat = asymmetric_matrix_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-8, max_iter=500, preconditioner="DILU")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # DILU-preconditioned PBiCGSTAB should converge
        assert torch.norm(r_final) < 0.1

    def test_with_dic_preconditioner(self, asymmetric_matrix_3cell):
        """PBiCGSTAB with DIC preconditioner."""
        mat = asymmetric_matrix_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-8, max_iter=500, preconditioner="DIC")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-4


class TestPBiCGSTABConvergence:
    """PBiCGSTAB convergence behaviour tests."""

    def test_convergence_info(self, asymmetric_matrix_3cell):
        """Solver returns correct convergence information."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-6, max_iter=200)
        x, iters, residual = solver(mat, b, x0)

        assert isinstance(iters, int)
        assert isinstance(residual, float)
        assert iters > 0

    def test_10cell_convergence(self, asymmetric_matrix_10cell):
        """PBiCGSTAB converges on a 10-cell system."""
        mat = asymmetric_matrix_10cell
        n_cells = mat.n_cells

        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-8, max_iter=500, preconditioner="DILU")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # DILU-preconditioned PBiCGSTAB should converge
        assert torch.norm(r_final) < 0.1

    def test_zero_rhs(self):
        """PBiCGSTAB with zero RHS should give zero solution."""
        mat = LduMatrix(3, torch.tensor([0, 1], dtype=INDEX_DTYPE),
                        torch.tensor([1, 2], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0, -2.0], dtype=CFD_DTYPE)

        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSTABSolver(tolerance=1e-10, max_iter=100)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-10)


class TestPBiCGSTABFactory:
    """Test solver creation via factory."""

    def test_create_pbicgstab(self):
        """Create PBiCGSTAB via factory function."""
        solver = create_solver("PBiCGSTAB", tolerance=1e-8)
        assert isinstance(solver, PBiCGSTABSolver)

    def test_create_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("pbicgstab")
        assert isinstance(solver, PBiCGSTABSolver)

    def test_solve_via_factory(self, asymmetric_matrix_3cell):
        """Factory-created solver works end-to-end."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = create_solver("PBiCGSTAB", tolerance=1e-6, max_iter=200)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # Factory-created solver should converge
        assert torch.norm(r_final) < 0.1

    def test_repr(self):
        """Solver repr includes key parameters."""
        solver = PBiCGSTABSolver(tolerance=1e-8, max_iter=500)
        r = repr(solver)
        assert "PBiCGSTABSolver" in r
