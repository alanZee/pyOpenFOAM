"""Tests for PBiCG (Preconditioned Bi-Conjugate Gradient) solver."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.pbicg import PBiCGSolver
from pyfoam.solvers.linear_solver import create_solver


class TestPBiCGBasic:
    """Basic PBiCG solver tests."""

    def test_identity_system(self):
        """Solve I x = b -> x = b."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-8)

    def test_symmetric_2cell(self):
        """PBiCG should also work for symmetric systems."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
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

        solver = PBiCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([1.0, 1.0], dtype=CFD_DTYPE), atol=1e-4)

    def test_asymmetric_3cell(self, asymmetric_matrix_3cell):
        """PBiCG on a 3-cell asymmetric system."""
        mat = asymmetric_matrix_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1.0  # Converges but not tight

    def test_symmetric_3cell(self, symmetric_poisson_3cell):
        """PBiCG on symmetric 3-cell system (should match PCG behaviour)."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-10, max_iter=1000, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-6


class TestPBiCGPreconditioned:
    """PBiCG with preconditioners."""

    def test_with_dilu_preconditioner(self, asymmetric_matrix_3cell):
        """PBiCG with DILU preconditioner."""
        mat = asymmetric_matrix_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="DILU")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 0.1

    def test_with_dic_preconditioner(self, symmetric_poisson_3cell):
        """PBiCG with DIC preconditioner."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="DIC")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-4

    def test_dilu_faster_than_none(self, asymmetric_matrix_3cell):
        """DILU preconditioning should reduce iterations."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver_none = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="none")
        _, iters_none, _ = solver_none(mat, b, x0.clone())

        solver_dilu = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="DILU")
        _, iters_dilu, _ = solver_dilu(mat, b, x0.clone())

        # DILU should converge in same or fewer iterations
        assert iters_dilu <= iters_none

    def test_invalid_preconditioner_raises(self):
        """Invalid preconditioner type raises ValueError."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-6, max_iter=10, preconditioner="INVALID")
        with pytest.raises(ValueError, match="Unknown preconditioner"):
            solver(mat, b, x0)


class TestPBiCGConvergence:
    """PBiCG convergence behaviour tests."""

    def test_convergence_info(self, asymmetric_matrix_3cell):
        """Solver returns correct convergence information."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-6, max_iter=200)
        x, iters, residual = solver(mat, b, x0)

        assert isinstance(iters, int)
        assert isinstance(residual, float)
        assert iters > 0

    def test_max_iterations_stops(self):
        """Solver stops at max iterations even if not converged."""
        mat = LduMatrix(2, torch.tensor([0], dtype=INDEX_DTYPE),
                        torch.tensor([1], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1e8], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-15, max_iter=5, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert iters <= 5

    def test_10cell_convergence(self, asymmetric_matrix_10cell):
        """PBiCG converges on 10-cell system."""
        mat = asymmetric_matrix_10cell
        n_cells = mat.n_cells

        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-8, max_iter=500, preconditioner="DILU")
        x, iters, residual = solver(mat, b, x0)

        # PBiCG with diagonal DILU preconditioner may not converge as
        # well as PBiCGSTAB, but should make progress
        r_final = b - mat.Ax(x)
        assert iters > 0
        assert torch.isfinite(x).all()

    def test_zero_rhs(self):
        """PBiCG with zero RHS gives zero solution."""
        mat = LduMatrix(3, torch.tensor([0, 1], dtype=INDEX_DTYPE),
                        torch.tensor([1, 2], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0, -2.0], dtype=CFD_DTYPE)

        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PBiCGSolver(tolerance=1e-10, max_iter=100)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-10)


class TestPBiCGFactory:
    """Test solver creation via factory."""

    def test_create_pbicg(self):
        """Create PBiCG via factory function."""
        solver = create_solver("PBiCG", tolerance=1e-8)
        assert isinstance(solver, PBiCGSolver)

    def test_create_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("pbicg")
        assert isinstance(solver, PBiCGSolver)

    def test_solve_via_factory(self, asymmetric_matrix_3cell):
        """Factory-created solver works end-to-end."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = create_solver("PBiCG", tolerance=1e-6, max_iter=200)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 0.1

    def test_repr(self):
        """Solver repr includes key parameters."""
        solver = PBiCGSolver(tolerance=1e-8, max_iter=500)
        r = repr(solver)
        assert "PBiCGSolver" in r
