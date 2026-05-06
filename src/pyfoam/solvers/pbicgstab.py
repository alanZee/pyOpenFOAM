"""
Preconditioned Bi-Conjugate Gradient Stabilised (PBiCGSTAB) solver for
general (possibly asymmetric) LDU matrices.

The PBiCGSTAB algorithm:

::

    r₀ = b - A·x₀
    r̃₀ = r₀  (shadow residual, can be any vector with r̃₀·r₀ ≠ 0)
    ρ₀ = r̃₀·r₀
    p₀ = r₀
    for i = 0, 1, ...
        y = M⁻¹·p_i
        v = A·y
        α_i = ρ_i / (r̃₀·v)
        s = r_i - α_i·v
        z = M⁻¹·s
        t = A·z
        ω_i = (t·z) / (t·t)
        x_{i+1} = x_i + α_i·y + ω_i·z
        r_{i+1} = s - ω_i·t
        ρ_{i+1} = r̃₀·r_{i+1}
        β_i = (ρ_{i+1}/ρ_i)·(α_i/ω_i)
        p_{i+1} = r_{i+1} + β_i·(p_i - ω_i·v)

PBiCGSTAB is the standard solver for asymmetric systems (e.g., momentum
equation with upwind convection).

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.preconditioners import (
    DILUPreconditioner,
    Preconditioner,
)
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["PBiCGSTABSolver"]


class PBiCGSTABSolver(LinearSolverBase):
    """Preconditioned Bi-Conjugate Gradient Stabilised solver.

    Suitable for general (possibly asymmetric) matrices.

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
            from pyfoam.solvers.preconditioners import DICPreconditioner
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
        """Run the PBiCGSTAB algorithm.

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

        # rho = r_tilde · r
        rho = torch.dot(r_tilde, r)

        # p = r
        p = r.clone()

        # v is allocated inside the loop (first iteration)
        v = torch.zeros_like(r)

        for i in range(1, max_iter + 1):
            # Check for breakdown
            if abs(float(rho)) < 1e-30:
                info = monitor.build_info(converged=False)
                return x, info

            # y = M⁻¹·p
            if precond is not None:
                y = precond.apply(p)
            else:
                y = p.clone()

            # v = A·y
            v = matrix.Ax(y)

            # α = ρ / (r_tilde · v)
            r_tilde_v = torch.dot(r_tilde, v)
            if abs(float(r_tilde_v)) < 1e-30:
                info = monitor.build_info(converged=False)
                return x, info

            alpha = rho / r_tilde_v

            # s = r - α·v
            s = r - alpha * v

            # Check convergence on s (half-step)
            if monitor.update(s, i):
                # x = x + α·y (just the correction so far)
                x = x + alpha * y
                info = monitor.build_info(converged=True)
                return x, info

            # z = M⁻¹·s
            if precond is not None:
                z = precond.apply(s)
            else:
                z = s.clone()

            # t = A·z
            t = matrix.Ax(z)

            # ω = (t·z) / (t·t)
            t_t = torch.dot(t, t)
            if abs(float(t_t)) < 1e-30:
                # t ≈ 0, just use the correction from s
                x = x + alpha * y
                r = s.clone()
                if monitor.update(r, i):
                    info = monitor.build_info(converged=True)
                    return x, info
                # Reset
                rho_new = torch.dot(r_tilde, r)
                beta = (rho_new / rho) * alpha
                rho = rho_new
                p = r + beta * (p - v)  # v was from p, approximate
                continue

            omega = torch.dot(t, z) / t_t

            # x = x + α·y + ω·z
            x = x + alpha * y + omega * z

            # r = s - ω·t
            r = s - omega * t

            # Check convergence
            if monitor.update(r, i):
                info = monitor.build_info(converged=True)
                return x, info

            # ρ_new = r_tilde · r
            rho_new = torch.dot(r_tilde, r)

            # β = (ρ_new / ρ) · (α / ω)
            if abs(float(omega)) < 1e-30:
                info = monitor.build_info(converged=False)
                return x, info

            beta = (rho_new / rho) * (alpha / omega)
            rho = rho_new

            # p = r + β·(p - ω·v)
            p = r + beta * (p - omega * v)

        # Max iterations reached
        info = monitor.build_info(converged=False)
        return x, info
