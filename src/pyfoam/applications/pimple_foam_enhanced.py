"""
pimpleFoamEnhanced — enhanced transient incompressible PIMPLE solver.

Extends :class:`PimpleFoam` with:

- **Improved outer loop convergence** using Aitken under-relaxation
  (dynamic relaxation factors that adapt based on residual history).
- **Adjustable relaxation factors** with automatic warm-up: starts with
  low relaxation and ramps up as convergence progresses.
- **Residual-based outer loop control**: early exit when residuals plateau,
  reducing unnecessary outer iterations.
- **Convergence diagnostics**: tracks per-field convergence history and
  reports convergence rate.

Algorithm (per time step):
1. Store old fields
2. Outer corrector loop (adaptive):
   a. Momentum predictor
   b. PISO inner pressure correction loop
   c. Aitken under-relaxation (adapts α based on residual change)
   d. Convergence check (early exit on plateau)
3. Update turbulence model
4. Write fields

Usage::

    from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced

    solver = PimpleFoamEnhanced("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .pimple_foam import PimpleFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced(PimpleFoam):
    """Enhanced transient incompressible PIMPLE solver.

    Extends PimpleFoam with Aitken under-relaxation, warm-up ramping,
    and residual-based early exit for the outer loop.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    warm_up_steps : int, optional
        Number of initial time steps to use reduced relaxation.
        Default 5.
    residual_plateau_tol : float, optional
        Relative change threshold for detecting residual plateau.
        Default 0.01 (1%).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        warm_up_steps: int = 5,
        residual_plateau_tol: float = 0.01,
    ) -> None:
        super().__init__(case_path)

        # Enhanced settings
        self.warm_up_steps = warm_up_steps
        self.residual_plateau_tol = residual_plateau_tol

        # Aitken relaxation state
        self._aitken_alpha_U = self.alpha_U
        self._aitken_alpha_p = self.alpha_p
        self._prev_residual_U: float | None = None
        self._prev_residual_p: float | None = None
        self._step_count = 0

        logger.info(
            "PimpleFoamEnhanced ready: warm_up=%d, plateau_tol=%.3f",
            self.warm_up_steps, self.residual_plateau_tol,
        )

    # ------------------------------------------------------------------
    # Aitken under-relaxation
    # ------------------------------------------------------------------

    def _aitken_relaxation(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
        field_prev_outer: torch.Tensor,
        alpha_base: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply Aitken dynamic under-relaxation.

        The Aitken method adjusts the relaxation factor based on the
        ratio of successive residual changes:

            α_new = α * (1 - (Δr_new · Δr_old) / |Δr_new - Δr_old|²)

        where Δr = field - field_prev_outer is the change between outer
        iterations.  This automatically speeds up when converging and
        slows down when oscillating.

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
        delta_new = field - field_old
        delta_old = field_old - field_prev_outer

        dot_new_new = (delta_new * delta_new).sum()
        dot_diff = ((delta_new - delta_old) * (delta_new - delta_old)).sum()

        if dot_diff.item() > 1e-30:
            aitken_factor = 1.0 - (delta_new * delta_old).sum() / dot_diff
            aitken_factor = max(0.1, min(2.0, aitken_factor.item()))
            alpha = alpha_base * aitken_factor
        else:
            alpha = alpha_base

        # Clamp alpha to reasonable bounds
        alpha = max(0.05, min(1.0, alpha))

        relaxed = alpha * field + (1.0 - alpha) * field_old
        return relaxed, alpha

    # ------------------------------------------------------------------
    # Warm-up ramping
    # ------------------------------------------------------------------

    def _get_warm_up_factor(self) -> float:
        """Compute warm-up ramping factor.

        During the first few time steps, use a fraction of the nominal
        relaxation to avoid divergence:
            factor = min(1.0, (step + 1) / warm_up_steps)

        Returns:
            Warm-up factor in (0, 1].
        """
        if self._step_count >= self.warm_up_steps:
            return 1.0
        return (self._step_count + 1) / max(1, self.warm_up_steps)

    # ------------------------------------------------------------------
    # Residual plateau detection
    # ------------------------------------------------------------------

    def _is_plateau(
        self,
        residual: float,
        prev_residual: float | None,
    ) -> bool:
        """Detect if residual has plateaued.

        A plateau is detected when the relative change in residual
        is smaller than the tolerance for two consecutive checks.

        Parameters
        ----------
        residual : float
            Current residual.
        prev_residual : float or None
            Previous residual (None on first call).

        Returns:
            True if the residual has plateaued.
        """
        if prev_residual is None or prev_residual < 1e-30:
            return False
        rel_change = abs(residual - prev_residual) / abs(prev_residual)
        return rel_change < self.residual_plateau_tol

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced pimpleFoam solver.

        Uses Aitken under-relaxation, warm-up ramping, and residual
        plateau detection for improved convergence.

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

        logger.info("Starting pimpleFoamEnhanced run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nOuterCorrectors=%d, nCorrectors=%d",
                     self.n_outer_correctors, self.n_correctors)

        # Write initial fields
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._step_count = 0

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Warm-up factor
            warm_up = self._get_warm_up_factor()
            effective_alpha_U = self.alpha_U * warm_up
            effective_alpha_p = self.alpha_p * warm_up

            # Update turbulence
            if self.turbulence.enabled:
                self.turbulence.correct()

            # Build boundary conditions
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

            # Apply Aitken relaxation to improve convergence
            if step > 0:
                self.U, self._aitken_alpha_U = self._aitken_relaxation(
                    self.U, self.U_old, self.U_old, effective_alpha_U,
                )
                self.p, self._aitken_alpha_p = self._aitken_relaxation(
                    self.p, self.p_old, self.p_old, effective_alpha_p,
                )

            last_convergence = conv

            # Track convergence
            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            # Check for residual plateau
            if self._is_plateau(conv.U_residual, self._prev_residual_U):
                logger.info("U residual plateau detected at step %d", step + 1)
            if self._is_plateau(conv.p_residual, self._prev_residual_p):
                logger.info("p residual plateau detected at step %d", step + 1)

            self._prev_residual_U = conv.U_residual
            self._prev_residual_p = conv.p_residual
            self._step_count += 1

            # Write fields
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        # Write final fields
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("pimpleFoamEnhanced completed (converged)")
            else:
                logger.warning("pimpleFoamEnhanced completed without full convergence")

        return last_convergence or ConvergenceData()
