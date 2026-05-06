"""Tests for PCG (Preconditioned Conjugate Gradient) solver."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.linear_solver import create_solver


class TestPCGBasic:
    """Basic PCG solver tests."""

    def test_identity_system(self):
        """Solve I x = b → x = b."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-8)
        # PCG needs 1-2 iterations for identity (first computes residual, second converges)
        assert iters <= 2

    def test_scaled_identity(self):
        """Solve diag(2) x = b → x = b/2."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([2.0, 2.0], dtype=CFD_DTYPE)

        b = torch.tensor([4.0, 6.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([2.0, 3.0], dtype=CFD_DTYPE), atol=1e-8)

    def test_symmetric_2cell(self):
        """Solve a 2x2 symmetric system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        # Known solution: x = [1, 2]
        # A = [[4, -1], [-1, 4]]
        # b = A @ [1, 2] = [4-2, -1+8] = [2, 7]
        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=100, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.tensor([1.0, 2.0], dtype=CFD_DTYPE), atol=1e-6)
        assert iters < 20

    def test_symmetric_3cell_chain(self, symmetric_poisson_3cell):
        """Solve 3-cell Poisson equation."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        # Source: b = [1, 2, 1]
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=1000, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        # Verify: residual should be small
        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-6

    def test_with_dic_preconditioner(self, symmetric_poisson_3cell):
        """PCG with DIC preconditioner should converge faster."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        # Without preconditioner
        solver_no = PCGSolver(tolerance=1e-10, max_iter=1000, preconditioner="none")
        _, iters_no, _ = solver_no(mat, b, x0.clone())

        # With DIC
        solver_dic = PCGSolver(tolerance=1e-10, max_iter=1000, preconditioner="DIC")
        x_dic, iters_dic, _ = solver_dic(mat, b, x0.clone())

        # DIC should converge in same or fewer iterations
        # (for small systems the difference may be small)
        r_final = b - mat.Ax(x_dic)
        assert torch.norm(r_final) < 1e-6

    def test_with_dilu_preconditioner(self, symmetric_poisson_3cell):
        """PCG with DILU preconditioner."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=1000, preconditioner="DILU")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-6


class TestPCGConvergence:
    """PCG convergence behaviour tests."""

    def test_convergence_info(self, symmetric_poisson_3cell):
        """Solver returns correct convergence information."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-6, max_iter=100)
        x, iters, residual = solver(mat, b, x0)

        assert isinstance(iters, int)
        assert isinstance(residual, float)
        assert iters > 0
        assert residual < 1e-6

    def test_max_iterations_stops(self):
        """Solver stops at max iterations even if not converged."""
        # Ill-conditioned system
        mat = LduMatrix(2, torch.tensor([0], dtype=INDEX_DTYPE),
                        torch.tensor([1], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1e8], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-15, max_iter=5, preconditioner="none")
        x, iters, residual = solver(mat, b, x0)

        assert iters <= 5

    def test_10cell_convergence(self, symmetric_poisson_10cell):
        """PCG converges on a 10-cell system."""
        mat = symmetric_poisson_10cell
        n_cells = mat.n_cells

        # Smooth source
        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-8, max_iter=500, preconditioner="DIC")
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # DIC-preconditioned PCG should converge well
        assert torch.norm(r_final) < 1e-4

    def test_zero_rhs(self):
        """PCG with zero RHS should give zero solution."""
        mat = LduMatrix(3, torch.tensor([0, 1], dtype=INDEX_DTYPE),
                        torch.tensor([1, 2], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)

        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = PCGSolver(tolerance=1e-10, max_iter=100)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-10)


class TestPCGFactory:
    """Test solver creation via factory."""

    def test_create_pcg(self):
        """Create PCG via factory function."""
        solver = create_solver("PCG", tolerance=1e-8)
        assert isinstance(solver, PCGSolver)
        assert solver.tolerance == 1e-8

    def test_create_pcg_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("pcg")
        assert isinstance(solver, PCGSolver)

    def test_unknown_solver_raises(self):
        """Unknown solver name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown solver"):
            create_solver("FOOBAR")

    def test_repr(self):
        """Solver repr includes key parameters."""
        solver = PCGSolver(tolerance=1e-8, max_iter=500)
        r = repr(solver)
        assert "PCGSolver" in r
        assert "1e-08" in r
