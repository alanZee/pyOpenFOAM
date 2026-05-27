"""Tests for Diagonal solver (explicit systems)."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.diagonal import DiagonalSolver
from pyfoam.solvers.linear_solver import create_solver


class TestDiagonalBasic:
    """Basic Diagonal solver tests."""

    def test_identity_system(self):
        """Solve I x = b -> x = b."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-8)

    def test_scaled_diagonal(self):
        """Solve diag(2) x = b -> x = b/2."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 2.0], dtype=CFD_DTYPE)

        b = torch.tensor([4.0, 6.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([2.0, 3.0], dtype=CFD_DTYPE), atol=1e-8)

    def test_varying_diagonal(self):
        """Solve with different diagonal entries."""
        diag_vals = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=CFD_DTYPE)
        mat = LduMatrix(4, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = diag_vals

        b = torch.tensor([10.0, 12.0, 15.0, 28.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(4, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        expected = b / diag_vals
        assert torch.allclose(x, expected, atol=1e-8)

    def test_ignores_off_diagonal(self):
        """Diagonal solver ignores off-diagonal coefficients."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([4.0, 8.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        # Only diagonal is used: x = b / diag = [1, 2]
        assert torch.allclose(x, torch.tensor([1.0, 2.0], dtype=CFD_DTYPE), atol=1e-8)

    def test_converges_in_one_pass(self):
        """Diagonal solver always converges in one iteration."""
        mat = LduMatrix(5, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(5, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=100)
        x, iters, residual = solver(mat, b, x0)

        assert iters == 1
        expected = b / mat.diag
        assert torch.allclose(x, expected, atol=1e-8)

    def test_zero_rhs(self):
        """Diagonal with zero RHS gives zero solution."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)

        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-10)

    def test_near_zero_diagonal_safe(self):
        """Diagonal solver handles near-zero diagonal entries safely."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1e-35, 2.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = DiagonalSolver(tolerance=1e-10, max_iter=1)
        x, iters, residual = solver(mat, b, x0)

        # First element clamped, result is large but finite
        assert torch.isfinite(x).all()
        assert torch.allclose(x[1], torch.tensor(1.0, dtype=CFD_DTYPE), atol=1e-8)


class TestDiagonalFactory:
    """Test solver creation via factory."""

    def test_create_diagonal(self):
        """Create Diagonal via factory function."""
        solver = create_solver("diagonal", tolerance=1e-8)
        assert isinstance(solver, DiagonalSolver)
        assert solver.tolerance == 1e-8

    def test_create_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("DIAGONAL")
        assert isinstance(solver, DiagonalSolver)

    def test_solve_via_factory(self):
        """Factory-created solver works end-to-end."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)

        b = torch.tensor([4.0, 9.0, 16.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = create_solver("diagonal", tolerance=1e-6)
        x, iters, residual = solver(mat, b, x0)

        expected = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        assert torch.allclose(x, expected, atol=1e-8)

    def test_repr(self):
        """Solver repr includes class name."""
        solver = DiagonalSolver(tolerance=1e-8, max_iter=500)
        r = repr(solver)
        assert "DiagonalSolver" in r
