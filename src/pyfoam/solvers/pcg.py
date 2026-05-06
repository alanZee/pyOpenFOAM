"""
Preconditioned Conjugate Gradient (PCG) solver for symmetric positive-definite
LDU matrices.

The PCG algorithm with preconditioner M:

::

    r₀ = b - A·x₀
    z₀ = M⁻¹·r₀
    p₀ = z₀
    for i = 0, 1, ...
        α_i  = (r_i · z_i) / (p_i · A · p_i)
        x_{i+1} = x_i + α_i · p_i
        r_{i+1} = r_i - α_i · A · p_i
        z_{i+1} = M⁻¹ · r_{i+1}
        β_i  = (r_{i+1} · z_{i+1}) / (r_i · z_i)
        p_{i+1} = z_{i+1} + β_i · p_i

PCG is the standard solver for the pressure equation in SIMPLE/PISO
(pressure is symmetric positive-definite after discretisation).

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.preconditioners import (
    DICPreconditioner,
    DILUPreconditioner,
    Preconditioner,
)
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["PCGSolver"]


class PCGSolver(LinearSolverBase):
    """Preconditioned Conjugate Gradient solver.

    Suitable for symmetric positive-definite matrices (e.g., pressure
    equation in incompressible flow).

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rel_tol : float
        Relative convergence tolerance.
    max_iter : int
        Maximum iterations.
    min_iter : int
        Minimum iterations before convergence check.
    verbose : bool
        If True, log residuals.
    preconditioner : str
        Preconditioner type: ``"DIC"`` (default) or ``"DILU"`` or ``"none"``.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        max_iter: int = 1000,
        min_iter: int = 0,
        verbose: bool = False,
        preconditioner: str = "DIC",
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            rel_tol=rel_tol,
            max_iter=max_iter,
            min_iter=min_iter,
            verbose=verbose,
        )
        self._precond_type = preconditioner.upper()

    def _make_preconditioner(self, matrix: LduMatrix) -> Preconditioner | None:
        """Create the preconditioner instance."""
        if self._precond_type == "DIC":
            return DICPreconditioner(matrix)
        elif self._precond_type == "DILU":
            return DILUPreconditioner(matrix)
        elif self._precond_type == "NONE":
            return None
        else:
            raise ValueError(
                f"Unknown preconditioner '{self._precond_type}'. "
                f"Use 'DIC', 'DILU', or 'none'."
            )

    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Run the PCG algorithm.

        Args:
            matrix: The LDU matrix A.
            source: Right-hand side b.
            x0: Initial guess.
            monitor: Residual monitor.
            max_iter: Maximum iterations.

        Returns:
            Tuple of ``(solution, convergence_info)``.
        """
        device = matrix.device
        dtype = matrix.dtype

        # Ensure tensors are on correct device/dtype
        source = source.to(device=device, dtype=dtype)
        x = x0.to(device=device, dtype=dtype).clone()

        # Build preconditioner
        precond = self._make_preconditioner(matrix)

        # Initial residual: r = b - A·x
        r = source - matrix.Ax(x)

        # Check initial convergence
        if monitor.update(r, 0):
            info = monitor.build_info(converged=True)
            return x, info

        # Precondition: z = M⁻¹·r
        if precond is not None:
            z = precond.apply(r)
        else:
            z = r.clone()

        # p = z
        p = z.clone()

        # rz = r · z
        rz = torch.dot(r, z)

        for i in range(1, max_iter + 1):
            # A·p
            Ap = matrix.Ax(p)

            # α = (r·z) / (p·A·p)
            pAp = torch.dot(p, Ap)
            if abs(float(pAp)) < 1e-30:
                # Breakdown: p·A·p ≈ 0
                info = monitor.build_info(converged=False)
                return x, info

            alpha = rz / pAp

            # x = x + α·p
            x = x + alpha * p

            # r = r - α·A·p
            r = r - alpha * Ap

            # Check convergence
            if monitor.update(r, i):
                info = monitor.build_info(converged=True)
                return x, info

            # z = M⁻¹·r
            if precond is not None:
                z = precond.apply(r)
            else:
                z = r.clone()

            # β = (r_new·z_new) / (r_old·z_old)
            rz_new = torch.dot(r, z)
            beta = rz_new / rz
            rz = rz_new

            # p = z + β·p
            p = z + beta * p

        # Max iterations reached
        info = monitor.build_info(converged=False)
        return x, info
