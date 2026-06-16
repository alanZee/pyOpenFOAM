"""
Scipy-based sparse linear solver for LDU matrices.

Uses scipy.sparse.linalg for high-performance solves.
"""

from __future__ import annotations

import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, bicgstab, spsolve, factorized

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["ScipyPCGSolver", "ScipyBiCGStabSolver", "ScipyDirectSolver"]


def _ldu_to_csr(matrix: LduMatrix) -> sparse.csr_matrix:
    """Convert LDU matrix to scipy CSR format."""
    n = matrix.n_cells
    diag = matrix.diag.detach().cpu().numpy()
    lower = matrix.lower.detach().cpu().numpy()
    upper = matrix.upper.detach().cpu().numpy()
    owner = matrix.owner.detach().cpu().numpy()
    neighbour = matrix.neighbour.detach().cpu().numpy()

    rows = np.concatenate([np.arange(n), owner, neighbour])
    cols = np.concatenate([np.arange(n), neighbour, owner])
    data = np.concatenate([diag, upper, lower])

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


class ScipyPCGSolver(LinearSolverBase):
    """Scipy-based PCG solver for symmetric positive-definite matrices."""

    def __init__(self, tolerance: float = 1e-6, rel_tol: float = 0.01,
                 max_iter: int = 1000, min_iter: int = 0,
                 verbose: bool = False, preconditioner: str = "DIC") -> None:
        super().__init__(tolerance=tolerance, rel_tol=rel_tol,
                        max_iter=max_iter, min_iter=min_iter, verbose=verbose)

    def _solve(self, matrix: LduMatrix, source: torch.Tensor,
               x0: torch.Tensor, monitor: ResidualMonitor,
               max_iter: int) -> tuple[torch.Tensor, ConvergenceInfo]:
        A_csr = _ldu_to_csr(matrix)
        b = source.detach().cpu().numpy().flatten()
        x0_np = x0.detach().cpu().numpy().flatten()

        x, info = cg(A_csr, b, x0=x0_np, rtol=self._tolerance,
                     maxiter=max_iter, atol=0)

        result = torch.from_numpy(x.reshape(-1)).to(source.device, source.dtype)
        residual = float(np.linalg.norm(A_csr @ x - b))
        conv = ConvergenceInfo(
            converged=(info == 0),
            iterations=max_iter,
            final_residual=residual,
            initial_residual=residual,
            residual_ratio=1.0,
            tolerance=self._tolerance,
        )
        return result, conv


class ScipyBiCGStabSolver(LinearSolverBase):
    """Scipy-based BiCGStab solver for non-symmetric matrices."""

    def __init__(self, tolerance: float = 1e-6, rel_tol: float = 0.01,
                 max_iter: int = 1000, min_iter: int = 0,
                 verbose: bool = False, preconditioner: str = "DILU") -> None:
        super().__init__(tolerance=tolerance, rel_tol=rel_tol,
                        max_iter=max_iter, min_iter=min_iter, verbose=verbose)

    def _solve(self, matrix: LduMatrix, source: torch.Tensor,
               x0: torch.Tensor, monitor: ResidualMonitor,
               max_iter: int) -> tuple[torch.Tensor, ConvergenceInfo]:
        A_csr = _ldu_to_csr(matrix)
        b = source.detach().cpu().numpy().flatten()
        x0_np = x0.detach().cpu().numpy().flatten()

        x, info = bicgstab(A_csr, b, x0=x0_np, rtol=self._tolerance,
                           maxiter=max_iter, atol=0)

        result = torch.from_numpy(x.reshape(-1)).to(source.device, source.dtype)
        residual = float(np.linalg.norm(A_csr @ x - b))
        conv = ConvergenceInfo(
            converged=(info == 0),
            iterations=max_iter,
            final_residual=residual,
            initial_residual=residual,
            residual_ratio=1.0,
            tolerance=self._tolerance,
        )
        return result, conv


class ScipyDirectSolver(LinearSolverBase):
    """Scipy sparse direct solver (LU decomposition).

    Much faster than iterative solvers for small-medium systems.
    Uses factorized() to cache LU decomposition when matrix doesn't change.
    """

    def __init__(self, tolerance: float = 1e-6, rel_tol: float = 0.01,
                 max_iter: int = 1, min_iter: int = 0,
                 verbose: bool = False, preconditioner: str = "none") -> None:
        super().__init__(tolerance=tolerance, rel_tol=rel_tol,
                        max_iter=max_iter, min_iter=min_iter, verbose=verbose)
        self._solve_cached = None
        self._cached_n = -1

    def _solve(self, matrix: LduMatrix, source: torch.Tensor,
               x0: torch.Tensor, monitor: ResidualMonitor,
               max_iter: int) -> tuple[torch.Tensor, ConvergenceInfo]:
        A_csr = _ldu_to_csr(matrix)
        b = source.detach().cpu().numpy().flatten()

        # Use cached factorization if matrix size matches
        if self._solve_cached is None or self._cached_n != A_csr.shape[0]:
            self._solve_cached = factorized(A_csr.tocsc())
            self._cached_n = A_csr.shape[0]

        x = self._solve_cached(b)

        result = torch.from_numpy(x.reshape(-1)).to(source.device, source.dtype)
        residual = float(np.linalg.norm(A_csr @ x - b))
        conv = ConvergenceInfo(
            converged=(residual < self._tolerance),
            iterations=1,
            final_residual=residual,
            initial_residual=residual,
            residual_ratio=0.0,
            tolerance=self._tolerance,
        )
        return result, conv
