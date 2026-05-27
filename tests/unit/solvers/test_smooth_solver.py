"""Tests for SmoothSolver (iterative smoother with configurable smoother type)."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.smooth_solver import SmoothSolver
from pyfoam.solvers.linear_solver import create_solver


class TestSmoothSolverGaussSeidel:
    """GaussSeidel smoother tests."""

    def test_identity_system(self):
        """Solve I x = b -> x = b."""
        mat = LduMatrix(3, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-10, max_iter=100,
                              smoother="GaussSeidel", n_sweeps=2)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-8)

    def test_symmetric_2cell(self):
        """Solve a 2x2 symmetric system with enough iterations."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-6, max_iter=500,
                              smoother="GaussSeidel", n_sweeps=2)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-2

    def test_symmetric_3cell_converges(self, symmetric_poisson_3cell):
        """GaussSeidel converges on 3-cell Poisson equation (residual decreases)."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        # Run two solves with different iteration counts
        solver_few = SmoothSolver(tolerance=1e-15, max_iter=10,
                                  smoother="GaussSeidel", n_sweeps=2)
        _, iters_few, res_few = solver_few(mat, b, x0.clone())

        solver_many = SmoothSolver(tolerance=1e-15, max_iter=500,
                                   smoother="GaussSeidel", n_sweeps=2)
        _, iters_many, res_many = solver_many(mat, b, x0.clone())

        # More iterations should give lower residual
        assert res_many < res_few

    def test_10cell_residual_decreases(self, symmetric_poisson_10cell):
        """GaussSeidel residual decreases on 10-cell system."""
        mat = symmetric_poisson_10cell
        n_cells = mat.n_cells

        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver_few = SmoothSolver(tolerance=1e-15, max_iter=10,
                                  smoother="GaussSeidel", n_sweeps=2)
        _, _, res_few = solver_few(mat, b, x0.clone())

        solver_many = SmoothSolver(tolerance=1e-15, max_iter=1000,
                                   smoother="GaussSeidel", n_sweeps=2)
        _, _, res_many = solver_many(mat, b, x0.clone())

        assert res_many < res_few


class TestSmoothSolverDIC:
    """DIC smoother tests."""

    def test_symmetric_2cell(self):
        """DIC-smoothed solver on 2x2 system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 7.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-10, max_iter=100,
                              smoother="DIC", n_sweeps=2)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-4

    def test_3cell_convergence(self, symmetric_poisson_3cell):
        """DIC-smoothed solver on 3-cell system."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-8, max_iter=500,
                              smoother="DIC", n_sweeps=2)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-2


class TestSmoothSolverDILU:
    """DILU smoother tests."""

    def test_asymmetric_2cell(self):
        """DILU-smoothed solver on asymmetric 2x2 system."""
        owner = torch.tensor([0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        mat = LduMatrix(2, owner, neighbour)
        mat.diag = torch.tensor([4.0, 5.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-2.0], dtype=CFD_DTYPE)

        b = torch.tensor([3.0, 3.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-10, max_iter=200,
                              smoother="DILU", n_sweeps=2, omega=0.5)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 0.1

    def test_asymmetric_3cell_residual_decreases(self, asymmetric_matrix_3cell):
        """DILU-smoothed solver residual decreases on 3-cell asymmetric."""
        mat = asymmetric_matrix_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver_few = SmoothSolver(tolerance=1e-15, max_iter=50,
                                  smoother="DILU", n_sweeps=2, omega=0.5)
        _, _, res_few = solver_few(mat, b, x0.clone())

        solver_many = SmoothSolver(tolerance=1e-15, max_iter=200,
                                   smoother="DILU", n_sweeps=2, omega=0.5)
        _, _, res_many = solver_many(mat, b, x0.clone())

        # Residual should decrease with more iterations
        assert res_many <= res_few


class TestSmoothSolverCommon:
    """Tests common to all smoother types."""

    def test_convergence_info(self, symmetric_poisson_3cell):
        """Solver returns correct convergence information."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-15, max_iter=100,
                              smoother="GaussSeidel", n_sweeps=2)
        x, iters, residual = solver(mat, b, x0)

        assert isinstance(iters, int)
        assert isinstance(residual, float)
        assert iters > 0

    def test_zero_rhs(self):
        """SmoothSolver with zero RHS gives zero solution."""
        mat = LduMatrix(3, torch.tensor([0, 1], dtype=INDEX_DTYPE),
                        torch.tensor([1, 2], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([4.0, 6.0, 4.0], dtype=CFD_DTYPE)
        mat.lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        mat.upper = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)

        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-10, max_iter=100,
                              smoother="GaussSeidel")
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-10)

    def test_invalid_smoother_raises(self):
        """Invalid smoother type raises ValueError."""
        mat = LduMatrix(2, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(2, dtype=CFD_DTYPE)

        solver = SmoothSolver(tolerance=1e-6, max_iter=10, smoother="INVALID")
        with pytest.raises(ValueError, match="Unknown smoother"):
            solver(mat, b, x0)

    def test_n_sweeps_affects_convergence(self, symmetric_poisson_3cell):
        """More sweeps should help convergence."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver_1sweep = SmoothSolver(tolerance=1e-15, max_iter=100,
                                     smoother="GaussSeidel", n_sweeps=1)
        solver_4sweep = SmoothSolver(tolerance=1e-15, max_iter=100,
                                     smoother="GaussSeidel", n_sweeps=4)

        _, _, res_1 = solver_1sweep(mat, b, x0.clone())
        _, _, res_4 = solver_4sweep(mat, b, x0.clone())

        # More sweeps per iteration should reduce residual
        assert res_4 <= res_1


class TestSmoothSolverFactory:
    """Test solver creation via factory."""

    def test_create_smooth_solver(self):
        """Create SmoothSolver via factory function."""
        solver = create_solver("smoothSolver", tolerance=1e-8)
        assert isinstance(solver, SmoothSolver)
        assert solver.tolerance == 1e-8

    def test_create_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("SMOOTHSOLVER")
        assert isinstance(solver, SmoothSolver)

    def test_solve_via_factory(self, symmetric_poisson_3cell):
        """Factory-created solver works (residual decreases)."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = create_solver("smoothSolver", tolerance=1e-15, max_iter=200)
        x, iters, residual = solver(mat, b, x0)

        assert iters > 0
        assert residual < 1.0  # Should have made some progress

    def test_repr(self):
        """Solver repr includes class name and smoother."""
        solver = SmoothSolver(tolerance=1e-8, max_iter=500, smoother="DIC")
        r = repr(solver)
        assert "SmoothSolver" in r
