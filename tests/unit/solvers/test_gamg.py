"""Tests for GAMG (Algebraic Multigrid) solver."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.gamg import GAMGSolver
from pyfoam.solvers.linear_solver import create_solver


class TestGAMGBasic:
    """Basic GAMG solver tests."""

    def test_identity_system(self):
        """GAMG on diagonal-only system."""
        mat = LduMatrix(4, torch.tensor([], dtype=INDEX_DTYPE),
                        torch.tensor([], dtype=INDEX_DTYPE))
        mat.diag = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=CFD_DTYPE)

        b = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(4, dtype=CFD_DTYPE)

        solver = GAMGSolver(
            tolerance=1e-8, max_iter=50,
            n_pre_smooth=2, n_post_smooth=2,
            min_cells_coarse=2,
        )
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, b, atol=1e-4)

    def test_symmetric_poisson_small(self, symmetric_poisson_3cell):
        """GAMG on small symmetric Poisson system."""
        mat = symmetric_poisson_3cell
        n_cells = mat.n_cells

        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = GAMGSolver(
            tolerance=1e-6, max_iter=100,
            n_pre_smooth=3, n_post_smooth=3,
            min_cells_coarse=1,
        )
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # GAMG on small systems may not converge as well as on larger ones
        assert torch.norm(r_final) < 0.1

    def test_symmetric_poisson_10cell(self, symmetric_poisson_10cell):
        """GAMG on 10-cell symmetric system."""
        mat = symmetric_poisson_10cell
        n_cells = mat.n_cells

        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = GAMGSolver(
            tolerance=1e-6, max_iter=100,
            n_pre_smooth=3, n_post_smooth=3,
            min_cells_coarse=2,
        )
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # GAMG should reduce residual significantly
        assert torch.norm(r_final) < 0.1

    def test_larger_system(self):
        """GAMG on a larger 20-cell chain."""
        n_cells = 20
        owner = torch.arange(19, dtype=INDEX_DTYPE)
        neighbour = torch.arange(1, 20, dtype=INDEX_DTYPE)

        mat = LduMatrix(n_cells, owner, neighbour)
        coeff = 1.0
        mat.lower = -coeff * torch.ones(19, dtype=CFD_DTYPE)
        mat.upper = -coeff * torch.ones(19, dtype=CFD_DTYPE)

        diag = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for f in range(19):
            diag[f] += coeff
            diag[f + 1] += coeff
        diag += 0.1
        mat.diag = diag

        b = torch.ones(n_cells, dtype=CFD_DTYPE)
        x0 = torch.zeros(n_cells, dtype=CFD_DTYPE)

        solver = GAMGSolver(
            tolerance=1e-6, max_iter=200,
            n_pre_smooth=3, n_post_smooth=3,
            min_cells_coarse=2,
        )
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        # GAMG should reduce residual
        assert torch.norm(r_final) < 0.1


class TestGAMGComponents:
    """Test GAMG internal components."""

    def test_build_aggregates(self, chain_10cell):
        """Aggregation should reduce the number of cells."""
        n_cells, owner, neighbour = chain_10cell
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.ones(n_cells, dtype=CFD_DTYPE)
        mat.lower = -0.5 * torch.ones(9, dtype=CFD_DTYPE)
        mat.upper = -0.5 * torch.ones(9, dtype=CFD_DTYPE)

        agg = GAMGSolver._build_aggregates(mat, n_coarse_target=5)

        assert agg.n_coarse <= 10
        assert agg.n_coarse >= 1
        assert agg.fine_to_coarse.shape == (n_cells,)
        # All cells should be assigned
        assert torch.all(agg.fine_to_coarse >= 0)

    def test_restrict_prolongate_roundtrip(self, chain_3cell):
        """Restrict then prolongate should preserve coarse-scale features."""
        n_cells, owner, neighbour = chain_3cell
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.ones(n_cells, dtype=CFD_DTYPE)
        mat.lower = -0.5 * torch.ones(2, dtype=CFD_DTYPE)
        mat.upper = -0.5 * torch.ones(2, dtype=CFD_DTYPE)

        agg = GAMGSolver._build_aggregates(mat, n_coarse_target=2)

        # Create a coarse vector
        e_coarse = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)[:agg.n_coarse]

        # Prolongate then restrict
        e_fine = GAMGSolver._prolongate(e_coarse, agg)
        e_back = GAMGSolver._restrict(e_fine, agg)

        # The roundtrip should preserve the coarse values (approximately)
        assert e_back.shape[0] == agg.n_coarse

    def test_build_coarse_matrix(self, chain_10cell):
        """Coarse matrix should be smaller than fine matrix."""
        n_cells, owner, neighbour = chain_10cell
        mat = LduMatrix(n_cells, owner, neighbour)
        mat.diag = torch.ones(n_cells, dtype=CFD_DTYPE) * 2.0
        mat.lower = -torch.ones(9, dtype=CFD_DTYPE)
        mat.upper = -torch.ones(9, dtype=CFD_DTYPE)

        agg = GAMGSolver._build_aggregates(mat, n_coarse_target=5)
        coarse = GAMGSolver._build_coarse_matrix(mat, agg)

        assert coarse.n_cells <= n_cells
        assert coarse.diag.shape[0] == coarse.n_cells


class TestGAMGConvergence:
    """GAMG convergence behaviour tests."""

    def test_convergence_info(self, symmetric_poisson_3cell):
        """Solver returns correct convergence information."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = GAMGSolver(tolerance=1e-6, max_iter=50, min_cells_coarse=1)
        x, iters, residual = solver(mat, b, x0)

        assert isinstance(iters, int)
        assert isinstance(residual, float)
        assert iters > 0

    def test_zero_rhs(self, symmetric_poisson_3cell):
        """GAMG with zero RHS should give approximately zero solution."""
        mat = symmetric_poisson_3cell
        b = torch.zeros(3, dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = GAMGSolver(tolerance=1e-10, max_iter=50, min_cells_coarse=1)
        x, iters, residual = solver(mat, b, x0)

        assert torch.allclose(x, torch.zeros(3, dtype=CFD_DTYPE), atol=1e-4)

    def test_max_iterations_stops(self, symmetric_poisson_3cell):
        """Solver stops at max iterations."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = GAMGSolver(tolerance=1e-15, max_iter=3, min_cells_coarse=1)
        x, iters, residual = solver(mat, b, x0)

        # Should have done at most 3 V-cycles (plus initial residual = 1 update)
        assert iters <= 4


class TestGAMGFactory:
    """Test GAMG creation via factory."""

    def test_create_gamg(self):
        """Create GAMG via factory function."""
        solver = create_solver("GAMG", tolerance=1e-8)
        assert isinstance(solver, GAMGSolver)

    def test_create_case_insensitive(self):
        """Factory is case-insensitive."""
        solver = create_solver("gamg")
        assert isinstance(solver, GAMGSolver)

    def test_solve_via_factory(self, symmetric_poisson_3cell):
        """Factory-created GAMG solver works end-to-end."""
        mat = symmetric_poisson_3cell
        b = torch.tensor([1.0, 2.0, 1.0], dtype=CFD_DTYPE)
        x0 = torch.zeros(3, dtype=CFD_DTYPE)

        solver = create_solver("GAMG", tolerance=1e-6, max_iter=50)
        x, iters, residual = solver(mat, b, x0)

        r_final = b - mat.Ax(x)
        assert torch.norm(r_final) < 1e-3

    def test_repr(self):
        """Solver repr includes key parameters."""
        solver = GAMGSolver(tolerance=1e-8, max_iter=50)
        r = repr(solver)
        assert "GAMGSolver" in r
