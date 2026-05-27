"""
Preconditioned Bi-Conjugate Gradient (PBiCG) solver for general LDU matrices.

PBiCG is an alternative to PBiCGSTAB for non-symmetric systems.  Unlike
PBiCGSTAB which stabilises the iteration, PBiCG works with both the
forward system A and the transpose system A^T, which can sometimes
converge where PBiCGSTAB stalls.

The PBiCG algorithm with left preconditioner M (Lanczos bi-orthogonalisation):

::

    r₀ = b - A·x₀
    z₀ = M⁻¹·r₀
    r̃₀ = r₀   (shadow residual)
    z̃₀ = M⁻ᵀ·r̃₀
    p₀ = z₀
    ρ₀ = z̃₀·r₀
    for i = 0, 1, ...
        v = A·p_i
        α_i = ρ_i / (z̃₀·v)
        x_{i+1} = x_i + α_i·p_i
        r_{i+1} = r_i - α_i·v
        z_{i+1} = M⁻¹·r_{i+1}
        ρ_{i+1} = z̃₀·r_{i+1}
        β_i = ρ_{i+1} / ρ_i
        p_{i+1} = z_{i+1} + β_i·p_i

The transpose preconditioner M⁻ᵀ is applied using the same M.apply()
because DIC and DILU preconditioners are effectively symmetric (diagonal
after the forward/backward sweep factorisation).

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.preconditioners import (
    DILUPreconditioner,
    DICPreconditioner,
    Preconditioner,
)
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["PBiCGSolver"]


class PBiCGSolver(LinearSolverBase):
    """Preconditioned Bi-Conjugate Gradient solver.

    Suitable for general (possibly asymmetric) matrices.
    Alternative to PBiCGSTAB that may converge in different situations.

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
        Preconditioner type: ``"DILU"`` (default) or ``"DIC"`` or ``"none"``.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        max_iter: int = 1000,
        min_iter: int = 0,
        verbose: bool = False,
        preconditioner: str = "DILU",
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
        if self._precond_type == "DILU":
            return DILUPreconditioner(matrix)
        elif self._precond_type == "DIC":
            return DICPreconditioner(matrix)
        elif self._precond_type == "NONE":
            return None
        else:
            raise ValueError(
                f"Unknown preconditioner '{self._precond_type}'. "
                f"Use 'DILU', 'DIC', or 'none'."
            )

    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Run the PBiCG algorithm with proper transpose handling.

        Uses the Lanczos bi-orthogonalisation form of BiCG with left
        preconditioning.  The shadow system uses A^T and M^{-T}.

        For DIC/DILU preconditioners, M^{-T} = M^{-1} (the factored
        diagonal is symmetric), so the same ``precond.apply()`` is used.

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

        # Shadow residual: r_tilde = r (standard choice)
        r_tilde = r.clone()

        # Initial preconditioned residuals
        if precond is not None:
            z = precond.apply(r)
            z_tilde = precond.apply(r_tilde)
        else:
            z = r.clone()
            z_tilde = r_tilde.clone()

        # Search direction
        p = z.clone()

        # rho = z_tilde · r
        rho = torch.dot(z_tilde, r)

        for i in range(1, max_iter + 1):
            # Check for breakdown
            if abs(float(rho)) < 1e-30:
                info = monitor.build_info(converged=False)
                return x, info

            # v = A·p
            v = matrix.Ax(p)

            # alpha = rho / (z_tilde · v)
            z_tilde_v = torch.dot(z_tilde, v)
            if abs(float(z_tilde_v)) < 1e-30:
                info = monitor.build_info(converged=False)
                return x, info

            alpha = rho / z_tilde_v

            # x = x + alpha · p
            x = x + alpha * p

            # r = r - alpha · v
            r = r - alpha * v

            # Check convergence
            if monitor.update(r, i):
                info = monitor.build_info(converged=True)
                return x, info

            # z = M^{-1} · r
            if precond is not None:
                z = precond.apply(r)
            else:
                z = r.clone()

            # Update adjoint: r_tilde via A^T, z_tilde = M^{-T} r_tilde
            r_tilde = r_tilde - alpha * matrix.Ax_T(p)
            if precond is not None:
                z_tilde = precond.apply(r_tilde)
            else:
                z_tilde = r_tilde.clone()

            # rho_new = z_tilde · r
            rho_new = torch.dot(z_tilde, r)

            # beta = rho_new / rho
            beta = rho_new / rho
            rho = rho_new

            # p = z + beta · p
            p = z + beta * p

        # Max iterations reached
        info = monitor.build_info(converged=False)
        return x, info
