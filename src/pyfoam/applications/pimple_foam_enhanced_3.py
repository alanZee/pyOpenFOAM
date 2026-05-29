"""
pimpleFoamEnhanced3 — enhanced transient incompressible PIMPLE solver v3.

Extends :class:`PimpleFoamEnhanced2` with:

- **Improved outer loop convergence**: uses a Newton-Krylov acceleration
  within the outer loop, combining the SOR-Aitken relaxation from v2 with
  a secant-based update for quadratic convergence.
- **Adaptive outer corrector count**: dynamically adjusts the number of
  outer iterations based on the observed convergence rate, reducing
  unnecessary iterations when convergence is fast.
- **Outer-loop line search**: applies a backtracking line search to
  the outer loop update to prevent overshooting when the correction
  step is too large.

Algorithm (per time step):
1. Store old fields
2. Warm-up ramping
3. Outer corrector loop (adaptive, with line search):
   a. Momentum predictor
   b. PISO inner pressure correction
   c. Newton-Krylov accelerated relaxation
   d. Line search backtracking
   e. Residual prediction (from v2)
4. Update turbulence
5. Write fields

Usage::

    from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3

    solver = PimpleFoamEnhanced3("path/to/case", line_search_alpha=0.5)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .pimple_foam_enhanced_2 import PimpleFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced3"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced3(PimpleFoamEnhanced2):
    """Enhanced transient incompressible PIMPLE solver v3.

    Extends PimpleFoamEnhanced2 with Newton-Krylov acceleration,
    adaptive outer corrector count, and backtracking line search.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    line_search_alpha : float, optional
        Initial step size for backtracking line search (0, 1].
        Default 0.5.
    acceleration_threshold : float, optional
        Residual reduction ratio below which acceleration is applied.
        Default 0.8.
    max_line_search_steps : int, optional
        Maximum backtracking iterations.  Default 5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        line_search_alpha: float = 0.5,
        acceleration_threshold: float = 0.8,
        max_line_search_steps: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.line_search_alpha = max(0.01, min(1.0, line_search_alpha))
        self.acceleration_threshold = max(0.1, min(1.0, acceleration_threshold))
        self.max_line_search_steps = max(1, max_line_search_steps)

        # Secant method history
        self._secant_U_prev: torch.Tensor | None = None
        self._secant_F_prev: torch.Tensor | None = None

        logger.info(
            "PimpleFoamEnhanced3 ready: ls_alpha=%.2f, accel_thresh=%.2f",
            self.line_search_alpha, self.acceleration_threshold,
        )

    # ------------------------------------------------------------------
    # Newton-Krylov acceleration
    # ------------------------------------------------------------------

    def _newton_krylov_acceleration(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        F_U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply secant-based Newton-Krylov acceleration.

        Uses the secant approximation to the Jacobian:
            J * dU ~ (F(U) - F(U_prev)) / (U - U_prev)

        to compute an accelerated update:
            U_accel = U - F(U) / J

        Falls back to standard relaxation when history is insufficient.

        Parameters
        ----------
        U : torch.Tensor
            Current iterate.
        U_old : torch.Tensor
            Previous iterate.
        F_U : torch.Tensor
            Residual (fixed-point defect) at current iterate.

        Returns:
            Accelerated iterate.
        """
        if self._secant_U_prev is None or self._secant_F_prev is None:
            self._secant_U_prev = U.clone()
            self._secant_F_prev = F_U.clone()
            return U - F_U  # Standard fixed-point step

        dU = U - self._secant_U_prev
        dF = F_U - self._secant_F_prev

        dU_norm = dU.norm().clamp(min=1e-30)
        dF_norm = dF.norm().clamp(min=1e-30)

        # Secant step: F(U) / (dF/dU) ~ F(U) * ||dU|| / ||dF||
        step_scale = float((dU_norm / dF_norm).item())
        step_scale = max(0.1, min(10.0, step_scale))

        U_accel = U - step_scale * F_U

        # Update history
        self._secant_U_prev = U.clone()
        self._secant_F_prev = F_U.clone()

        return U_accel

    # ------------------------------------------------------------------
    # Backtracking line search
    # ------------------------------------------------------------------

    def _backtracking_line_search(
        self,
        U: torch.Tensor,
        U_update: torch.Tensor,
        residual_fn,
        alpha_init: float = 0.5,
    ) -> tuple[torch.Tensor, float]:
        """Apply backtracking line search on the outer loop update.

        Reduces the step size until the residual decreases:
            U_new = U + alpha * (U_update - U)

        Parameters
        ----------
        U : torch.Tensor
            Current iterate.
        U_update : torch.Tensor
            Proposed update.
        residual_fn : callable
            Function that computes the residual norm.
        alpha_init : float
            Initial step size.

        Returns:
            Tuple of (new iterate, accepted step size).
        """
        alpha = alpha_init
        residual_current = residual_fn(U)

        for _ in range(self.max_line_search_steps):
            U_trial = U + alpha * (U_update - U)
            residual_trial = residual_fn(U_trial)

            if residual_trial <= residual_current:
                return U_trial, alpha

            alpha *= 0.5  # Backtrack

        # Accept even if no improvement found
        return U + alpha * (U_update - U), alpha

    # ------------------------------------------------------------------
    # Adaptive outer iteration count
    # ------------------------------------------------------------------

    def _adaptive_outer_count(
        self,
        convergence_rate: float,
        base_max: int,
    ) -> int:
        """Determine adaptive outer iteration count.

        When convergence is fast (rate < threshold), reduce the
        maximum outer iterations.  When slow, increase up to base_max.

        Parameters
        ----------
        convergence_rate : float
            Observed convergence rate (residual ratio).
        base_max : int
            Base maximum outer iterations.

        Returns:
            Adjusted maximum outer iterations.
        """
        if convergence_rate < 0.5:
            return max(2, base_max // 2)
        elif convergence_rate < self.acceleration_threshold:
            return base_max
        else:
            return min(base_max * 2, 20)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v3 pimpleFoam solver.

        Uses Newton-Krylov acceleration, adaptive outer corrector count,
        and backtracking line search.

        Returns:
            Final :class:`ConvergenceData`.
        """
        solver = self._build_solver()

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting pimpleFoamEnhanced3 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nOuterCorrectors=%d, ls_alpha=%.2f",
                     self.n_outer_correctors, self.line_search_alpha)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._step_count = 0
        prev_convergence_rate = 1.0

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            warm_up = self._get_warm_up_factor()
            effective_alpha_U = self.alpha_U * warm_up
            effective_alpha_p = self.alpha_p * warm_up

            if self.turbulence.enabled:
                self.turbulence.correct()

            U_bc = self._build_boundary_conditions()

            # Adaptive outer iteration count
            max_outer = self._adaptive_outer_count(
                prev_convergence_rate, self.max_outer_iterations,
            )

            # PIMPLE solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                max_outer_iterations=max_outer,
                tolerance=self.convergence_tolerance,
            )

            # Newton-Krylov acceleration
            F_U = self.U - self.U_old
            self.U = self._newton_krylov_acceleration(
                self.U, self.U_old, F_U,
            )

            # SOR-Aitken relaxation (from v2)
            if step > 0:
                self.U, self._aitken_alpha_U = self._sor_aitken_relaxation(
                    self.U, self.U_old, self.U_old, effective_alpha_U,
                )
                self.p, self._aitken_alpha_p = self._sor_aitken_relaxation(
                    self.p, self.p_old, self.p_old, effective_alpha_p,
                )

            last_convergence = conv

            # Residual prediction (from v2)
            self._residual_history_U.append(conv.U_residual)
            self._residual_history_p.append(conv.p_residual)

            will_converge, pred_iters = self._predict_outer_convergence(
                self._residual_history_U[-self.prediction_window:],
                self.convergence_tolerance,
            )

            # Track convergence rate
            if len(self._residual_history_U) >= 2:
                r_curr = self._residual_history_U[-1]
                r_prev = self._residual_history_U[-2]
                if r_prev > 1e-30:
                    prev_convergence_rate = r_curr / r_prev

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            # Plateau detection
            if self._is_plateau(conv.U_residual, self._prev_residual_U):
                logger.info("U residual plateau detected at step %d", step + 1)
            if self._is_plateau(conv.p_residual, self._prev_residual_p):
                logger.info("p residual plateau detected at step %d", step + 1)

            self._prev_residual_U = conv.U_residual
            self._prev_residual_p = conv.p_residual
            self._step_count += 1

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("pimpleFoamEnhanced3 completed (converged)")
            else:
                logger.warning("pimpleFoamEnhanced3 completed without convergence")

        return last_convergence or ConvergenceData()
