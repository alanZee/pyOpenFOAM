"""
Diagonal solver for explicit systems (diagonal matrices only).

Solves A x = b where A is diagonal: x = b / diag(A).

This solver is used for explicit equations in OpenFOAM where the
off-diagonal coefficients are zero (e.g., explicit velocity correction,
cell-limited corrections).  It performs a single-pass division with
no iterations.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["DiagonalSolver"]


class DiagonalSolver(LinearSolverBase):
    """Diagonal solver for explicit (diagonal-only) systems.

    Computes ``x = b / diag(A)`` in a single pass.  Off-diagonal
    coefficients are ignored (assumed zero).

    This solver always converges in one "iteration" (a single division).

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance (unused, kept for API compatibility).
    rel_tol : float
        Relative convergence tolerance (unused).
    max_iter : int
        Maximum iterations (always 1).
    min_iter : int
        Minimum iterations (always 1).
    verbose : bool
        If True, log the operation.
    """

    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Solve by dividing source by diagonal: x = b / diag(A).

        Args:
            matrix: The LDU matrix A (only diagonal is used).
            source: Right-hand side b.
            x0: Initial guess (unused).
            monitor: Residual monitor.
            max_iter: Maximum iterations (always 1).

        Returns:
            Tuple of ``(solution, convergence_info)``.
        """
        device = matrix.device
        dtype = matrix.dtype

        source = source.to(device=device, dtype=dtype)

        diag = matrix.diag
        # Clamp to avoid division by zero
        safe_diag = diag.abs().clamp(min=1e-30)

        x = source / safe_diag

        # Compute residual for reporting: r = b - diag * x
        # (off-diagonal terms are assumed zero)
        r = source - diag * x
        monitor.update(r, 0)
        info = monitor.build_info(converged=True)

        return x, info
