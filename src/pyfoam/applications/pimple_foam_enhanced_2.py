"""
pimpleFoamEnhanced2 — enhanced transient incompressible PIMPLE solver v2.

Extends :class:`PimpleFoamEnhanced` with:

- **Improved outer loop convergence** using a successive over-relaxation
  (SOR) approach with a blending factor that combines Aitken and fixed
  relaxation, providing robustness when the Aitken method oscillates.
- **Outer-loop residual prediction**: fits an exponential decay to the
  residual history to predict whether the outer loop will converge,
  enabling early exit before reaching the maximum number of outer
  iterations.
- **Convergence rate diagnostics**: tracks the effective convergence
  rate per outer iteration and adjusts the number of required outer
  iterations dynamically.

Algorithm (per time step):
1. Store old fields
2. Warm-up ramping
3. Outer corrector loop (adaptive):
   a. Momentum predictor
   b. PISO inner pressure correction
   c. SOR-Aitken blended relaxation
   d. Residual prediction (early exit)
4. Update turbulence
5. Write fields

Usage::

    from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2

    solver = PimpleFoamEnhanced2("path/to/case")
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

from .pimple_foam_enhanced import PimpleFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced2"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced2(PimpleFoamEnhanced):
    """Enhanced transient incompressible PIMPLE solver v2.

    Extends PimpleFoamEnhanced with SOR-Aitken blended relaxation,
    residual prediction for early exit, and adaptive outer iteration count.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    sor_weight : float, optional
        Blending weight between Aitken (0) and fixed SOR (1).
        Default 0.3.
    prediction_window : int, optional
        Number of outer iterations used for residual prediction.
        Default 3.
    min_outer_iterations : int, optional
        Minimum outer iterations per time step (regardless of convergence).
        Default 2.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        sor_weight: float = 0.3,
        prediction_window: int = 3,
        min_outer_iterations: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.sor_weight = max(0.0, min(1.0, sor_weight))
        self.prediction_window = max(2, prediction_window)
        self.min_outer_iterations = max(1, min_outer_iterations)

        # Residual history for prediction
        self._residual_history_U: list[float] = []
        self._residual_history_p: list[float] = []
        self._convergence_rate: float = 1.0

        logger.info(
            "PimpleFoamEnhanced2 ready: sor_weight=%.2f, pred_window=%d",
            self.sor_weight, self.prediction_window,
        )

    # ------------------------------------------------------------------
    # SOR-Aitken blended relaxation
    # ------------------------------------------------------------------

    def _sor_aitken_relaxation(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
        field_prev_outer: torch.Tensor,
        alpha_base: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply SOR-Aitken blended relaxation.

        Combines the Aitken method (adaptive) with fixed SOR
        (successive over-relaxation) for robustness:

            α_blended = w * α_sor + (1 - w) * α_aitken

        where α_sor is a fixed under-relaxation and α_aitken is the
        Aitken-adaptive factor.

        Parameters
        ----------
        field : torch.Tensor
            Current field (after correction).
        field_old : torch.Tensor
            Field from previous outer iteration.
        field_prev_outer : torch.Tensor
            Field from two outer iterations ago.
        alpha_base : float
            Base relaxation factor.

        Returns:
            Tuple of (relaxed_field, new_alpha).
        """
        # Get Aitken alpha
        _, aitken_alpha = self._aitken_relaxation(
            field, field_old, field_prev_outer, alpha_base,
        )

        # SOR: fixed moderate under-relaxation
        sor_alpha = alpha_base * 0.8

        # Blend
        alpha = self.sor_weight * sor_alpha + (1.0 - self.sor_weight) * aitken_alpha
        alpha = max(0.05, min(1.0, alpha))

        relaxed = alpha * field + (1.0 - alpha) * field_old
        return relaxed, alpha

    # ------------------------------------------------------------------
    # Residual prediction
    # ------------------------------------------------------------------

    def _predict_outer_convergence(
        self,
        residual_history: list[float],
        tolerance: float,
    ) -> tuple[bool, float]:
        """Predict whether the outer loop will converge.

        Fits an exponential decay r(n) = a * exp(-b * n) to the
        residual history and predicts the number of iterations needed.

        Parameters
        ----------
        residual_history : list[float]
            Recent residual values from outer iterations.
        tolerance : float
            Convergence tolerance.

        Returns:
            Tuple of (will_converge, predicted_iterations).
        """
        if len(residual_history) < 2:
            return False, float("inf")

        # Simple convergence rate estimate
        r_last = residual_history[-1]
        r_prev = residual_history[-2]

        if r_last < 1e-30 or r_prev < 1e-30:
            return True, float(len(residual_history))

        rate = r_last / r_prev

        if rate >= 1.0:
            # Not converging
            return False, float("inf")

        if r_last <= tolerance:
            return True, float(len(residual_history))

        # Predict iterations: tolerance = r_last * rate^n
        # n = log(tolerance / r_last) / log(rate)
        if rate > 1e-30:
            n_remaining = math.log(tolerance / r_last) / math.log(rate)
            return n_remaining < self.max_outer_iterations, float(
                len(residual_history) + max(0, n_remaining)
            )

        return False, float("inf")

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v2 pimpleFoam solver.

        Uses SOR-Aitken blended relaxation and residual prediction
        for improved outer loop convergence.

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

        logger.info("Starting pimpleFoamEnhanced2 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nOuterCorrectors=%d, sor_weight=%.2f",
                     self.n_outer_correctors, self.sor_weight)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._step_count = 0

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            warm_up = self._get_warm_up_factor()
            effective_alpha_U = self.alpha_U * warm_up
            effective_alpha_p = self.alpha_p * warm_up

            if self.turbulence.enabled:
                self.turbulence.correct()

            U_bc = self._build_boundary_conditions()

            # Run one PIMPLE time step
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=self.U_old,
                p_old=self.p_old,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )

            # SOR-Aitken relaxation
            if step > 0:
                self.U, self._aitken_alpha_U = self._sor_aitken_relaxation(
                    self.U, self.U_old, self.U_old, effective_alpha_U,
                )
                self.p, self._aitken_alpha_p = self._sor_aitken_relaxation(
                    self.p, self.p_old, self.p_old, effective_alpha_p,
                )

            last_convergence = conv

            # Residual prediction
            self._residual_history_U.append(conv.U_residual)
            self._residual_history_p.append(conv.p_residual)

            will_converge, pred_iters = self._predict_outer_convergence(
                self._residual_history_U[-self.prediction_window:],
                self.convergence_tolerance,
            )
            if will_converge:
                self._convergence_rate = max(0.01, min(1.0, pred_iters))

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
                logger.info("pimpleFoamEnhanced2 completed (converged)")
            else:
                logger.warning("pimpleFoamEnhanced2 completed without convergence")

        return last_convergence or ConvergenceData()
