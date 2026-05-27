"""
Smooth solver with configurable smoother for LDU matrices.

The smoothSolver in OpenFOAM applies iterative smoothing using one of:

- **GaussSeidel**: Sequential Gauss-Seidel relaxation.  Each cell is
  updated in-place using the latest values from already-updated cells.
  Converges well for diagonally dominant systems.
- **DIC**: Diagonal Incomplete Cholesky preconditioned Richardson iteration.
  Computes z = DIC⁻¹ · r, then x = x + ω · z (with optional relaxation).
- **DILU**: Diagonal Incomplete LU preconditioned Richardson iteration.
  Similar to DIC but for asymmetric matrices.

Algorithm for GaussSeidel smoother::

    for iter = 0, 1, ..., nSweeps:
        for each cell i:
            sum = b[i] - sum_{j != i} A[i,j] * x[j]
            x[i] = sum / A[i,i]

Algorithm for DIC/DILU smoothers::

    for iter = 0, 1, ..., nSweeps:
        r = b - A · x
        z = M⁻¹ · r
        x = x + ω · z

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather, scatter_add
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.preconditioners import (
    DICPreconditioner,
    DILUPreconditioner,
    Preconditioner,
)
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["SmoothSolver"]


class SmoothSolver(LinearSolverBase):
    """Iterative smoother solver with configurable smoother type.

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rel_tol : float
        Relative convergence tolerance.
    max_iter : int
        Maximum number of outer iterations (each iteration applies
        ``n_sweeps`` smoothing passes).
    min_iter : int
        Minimum iterations before convergence check.
    verbose : bool
        If True, log residuals.
    smoother : str
        Smoother type: ``"GaussSeidel"`` (default), ``"DIC"``, or ``"DILU"``.
    n_sweeps : int
        Number of smoothing passes per outer iteration (default 2,
        matching OpenFOAM's ``nSweeps``).
    omega : float
        Under-relaxation factor for DIC/DILU smoothers (default 1.0).
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        max_iter: int = 1000,
        min_iter: int = 0,
        verbose: bool = False,
        smoother: str = "GaussSeidel",
        n_sweeps: int = 2,
        omega: float = 1.0,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            rel_tol=rel_tol,
            max_iter=max_iter,
            min_iter=min_iter,
            verbose=verbose,
        )
        self._smoother = smoother
        self._n_sweeps = n_sweeps
        self._omega = omega

    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Run the smoothing iterations.

        Args:
            matrix: The LDU matrix A.
            source: Right-hand side b.
            x0: Initial guess.
            monitor: Residual monitor.
            max_iter: Maximum outer iterations.

        Returns:
            Tuple of ``(solution, convergence_info)``.
        """
        device = matrix.device
        dtype = matrix.dtype

        source = source.to(device=device, dtype=dtype)
        x = x0.to(device=device, dtype=dtype).clone()

        if self._smoother == "GaussSeidel":
            return self._solve_gauss_seidel(matrix, source, x, monitor, max_iter)
        elif self._smoother in ("DIC", "DILU"):
            return self._solve_precond_richardson(matrix, source, x, monitor, max_iter)
        else:
            raise ValueError(
                f"Unknown smoother '{self._smoother}'. "
                f"Use 'GaussSeidel', 'DIC', or 'DILU'."
            )

    def _solve_gauss_seidel(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Gauss-Seidel smoothing: sequential in-place cell updates.

        For each cell i:
            x[i] = (b[i] - sum_{j != i} A[i,j] * x[j]) / A[i,i]

        Uses the LDU structure to compute the off-diagonal sum for each row.
        In LDU format, a cell can have multiple faces creating duplicate
        entries for the same matrix position, so we use the full Ax formula
        per row to correctly accumulate all contributions.
        """
        diag = matrix.diag
        owner = matrix.owner
        neighbour = matrix.neighbour
        lower = matrix.lower
        upper = matrix.upper
        n_cells = matrix.n_cells
        n_faces = matrix.n_internal_faces

        # Initial residual check
        r = source - matrix.Ax(x)
        if monitor.update(r, 0):
            info = monitor.build_info(converged=True)
            return x, info

        for i in range(1, max_iter + 1):
            # Apply n_sweeps Gauss-Seidel passes
            for _sweep in range(self._n_sweeps):
                for c in range(n_cells):
                    # Compute (A*x)[c] using the full row to handle
                    # duplicate entries in LDU format
                    row_ax = float(diag[c]) * float(x[c])
                    for f in range(n_faces):
                        p = int(owner[f])
                        n = int(neighbour[f])
                        if p == c:
                            row_ax += float(lower[f]) * float(x[n])
                        elif n == c:
                            row_ax += float(upper[f]) * float(x[p])

                    # off_diag_sum = (A*x)[c] - diag[c]*x[c]
                    off_diag_sum = row_ax - float(diag[c]) * float(x[c])

                    d = float(diag[c])
                    if abs(d) > 1e-30:
                        x[c] = (float(source[c]) - off_diag_sum) / d

            # Check convergence after all sweeps
            r = source - matrix.Ax(x)
            if monitor.update(r, i):
                info = monitor.build_info(converged=True)
                return x, info

        info = monitor.build_info(converged=False)
        return x, info

    def _solve_precond_richardson(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Preconditioned Richardson iteration: x = x + omega * M⁻¹ · r.

        Uses DIC or DILU preconditioner.
        """
        # Build preconditioner
        if self._smoother == "DIC":
            precond: Preconditioner | None = DICPreconditioner(matrix)
        else:
            precond = DILUPreconditioner(matrix)

        # Initial residual check
        r = source - matrix.Ax(x)
        if monitor.update(r, 0):
            info = monitor.build_info(converged=True)
            return x, info

        for i in range(1, max_iter + 1):
            for _sweep in range(self._n_sweeps):
                r = source - matrix.Ax(x)
                z = precond.apply_full(r)
                x = x + self._omega * z

            # Check convergence after all sweeps
            r = source - matrix.Ax(x)
            if monitor.update(r, i):
                info = monitor.build_info(converged=True)
                return x, info

        info = monitor.build_info(converged=False)
        return x, info
