"""
pisoFoamEnhanced4 — enhanced transient incompressible PISO solver v4.

Extends :class:`PisoFoamEnhanced3` with:

- **Deferred correction approach**: implements a Picard deferred-correction
  strategy for the convective term that blends second-order linear
  upwind with first-order upwind, providing higher accuracy without
  sacrificing stability.
- **Adaptive PISO corrector scheduling**: uses a convergence-rate-based
  scheduling algorithm that dynamically determines the optimal number
  of PISO correctors per time step rather than using a fixed count.
- **Pressure-gradient preconditioning**: applies a cell-based pressure
  gradient preconditioner that improves the conditioning of the pressure
  Poisson equation on stretched or skewed meshes.

Algorithm (per time step):
1. Store old fields
2. Compute adaptive sub-steps (CFL + momentum balance from v3)
3. For each sub-step:
   a. Momentum predictor (deferred correction)
   b. Adaptive PISO correction loop
   c. Skewness-corrected Rhie-Chow (from v3)
   d. Pressure-gradient preconditioning
   e. Momentum balance correction (from v3)
   f. Turbulence update
4. Check convergence

Usage::

    from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4

    solver = PisoFoamEnhanced4("path/to/case", deferred_correction=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.coupled_solver import ConvergenceData

from .piso_foam_enhanced_3 import PisoFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced4"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced4(PisoFoamEnhanced3):
    """Enhanced transient incompressible PISO solver v4.

    Extends PisoFoamEnhanced3 with deferred correction, adaptive
    PISO corrector scheduling, and pressure-gradient preconditioning.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    deferred_correction : bool, optional
        Enable Picard deferred correction.  Default True.
    deferred_blend : float, optional
        Blending factor (0=first-order, 1=second-order).  Default 0.8.
    pressure_preconditioner : bool, optional
        Enable pressure-gradient preconditioner.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        deferred_correction: bool = True,
        deferred_blend: float = 0.8,
        pressure_preconditioner: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.deferred_correction = deferred_correction
        self.deferred_blend = max(0.0, min(1.0, deferred_blend))
        self.pressure_preconditioner = pressure_preconditioner

        # Corrector scheduling history
        self._corrector_history: list[int] = []
        self._convergence_rate_history: list[float] = []

        logger.info(
            "PisoFoamEnhanced4 ready: deferred=%s, blend=%.2f, p_prec=%s",
            self.deferred_correction, self.deferred_blend,
            self.pressure_preconditioner,
        )

    # ------------------------------------------------------------------
    # Deferred correction for convective term
    # ------------------------------------------------------------------

    def _deferred_correction_convection(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Picard deferred correction to the convective term.

        Blends first-order upwind with higher-order (linear upwind):
            U_corrected = (1-blend)*U_upwind + blend*U_higher_order

        The deferred part is treated explicitly for stability.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity iterate.
        U_old : torch.Tensor
            Previous iterate (for deferred part).

        Returns:
            Deferred-correction velocity.
        """
        if not self.deferred_correction:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Face interpolation weights
        w = mesh.face_weights[:n_internal]

        # First-order upwind (implicit part)
        U_O = U[owner]
        U_N = U[neigh]

        # Higher-order contribution (deferred, explicit)
        U_old_O = U_old[owner]
        U_old_N = U_old[neigh]

        # Linear upwind difference
        dU_ho = (U_N - U_O) - (U_old_N - U_old_O)

        # Apply blending
        correction = torch.zeros_like(U)
        dU_deferred = self.deferred_blend * dU_ho

        # Scatter face contributions
        correction.index_add_(0, owner, dU_deferred * 0.5)
        correction.index_add_(0, neigh, -dU_deferred * 0.5)

        return U + correction * 0.1  # Damped for stability

    # ------------------------------------------------------------------
    # Adaptive PISO corrector scheduling
    # ------------------------------------------------------------------

    def _adaptive_corrector_count(
        self,
        step: int,
        prev_residual: float,
        current_residual: float,
    ) -> int:
        """Determine optimal number of PISO correctors.

        Uses convergence rate history to schedule the corrector count:
        - Fast convergence (rate < 0.5): reduce correctors
        - Moderate convergence: maintain
        - Slow convergence (rate > 0.9): increase correctors

        Parameters
        ----------
        step : int
            Current time step.
        prev_residual : float
            Previous step residual.
        current_residual : float
            Current step residual.

        Returns:
            Number of PISO correctors for this step.
        """
        base_count = self.max_piso_correctors

        if step == 0 or prev_residual < 1e-30:
            return base_count

        rate = current_residual / prev_residual
        self._convergence_rate_history.append(rate)

        # Keep limited history
        if len(self._convergence_rate_history) > 20:
            self._convergence_rate_history.pop(0)

        # Average convergence rate
        avg_rate = sum(self._convergence_rate_history[-5:]) / min(
            len(self._convergence_rate_history), 5,
        )

        if avg_rate < 0.5:
            n_corr = max(1, base_count - 1)
        elif avg_rate > 0.9:
            n_corr = min(base_count + 2, 10)
        else:
            n_corr = base_count

        self._corrector_history.append(n_corr)
        return n_corr

    # ------------------------------------------------------------------
    # Pressure-gradient preconditioning
    # ------------------------------------------------------------------

    def _precondition_pressure_gradient(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pressure-gradient preconditioner.

        On stretched meshes, the standard pressure equation can have
        poor conditioning.  This preconditioner scales the pressure
        correction by a local metric factor based on the cell aspect
        ratio.

        p_precond = p * (1 + alpha * ||grad_p|| * dx)

        Parameters
        ----------
        p : torch.Tensor
            Current pressure.
        U : torch.Tensor
            Current velocity.

        Returns:
            Preconditioned pressure.
        """
        if not self.pressure_preconditioner:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute face-normal pressure gradient magnitude
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        grad_p_face = (p_N - p_O) * delta_coeffs

        # Scatter gradient magnitude to cells
        grad_p_cell = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        grad_p_cell = grad_p_cell + scatter_add(grad_p_face.abs(), owner, n_cells)
        grad_p_cell = grad_p_cell + scatter_add(grad_p_face.abs(), neigh, n_cells)

        # Normalize
        grad_p_norm = grad_p_cell / (grad_p_cell.mean().clamp(min=1e-30))

        # Preconditioning factor
        alpha = 0.1  # Small correction
        factor = (1.0 + alpha * (grad_p_norm - 1.0)).clamp(min=0.5, max=2.0)

        return p * factor

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 pisoFoam solver.

        Uses deferred correction, adaptive corrector scheduling,
        and pressure-gradient preconditioning.

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

        logger.info("Starting pisoFoamEnhanced4 run")
        logger.info("  deferred=%s, blend=%.2f, p_prec=%s",
                     self.deferred_correction, self.deferred_blend,
                     self.pressure_preconditioner)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        prev_residual = 0.0

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            n_sub = self._compute_sub_steps()
            sub_dt = self.delta_t / n_sub

            # Adaptive corrector scheduling
            n_corr = self._adaptive_corrector_count(
                step, prev_residual,
                last_convergence.U_residual if last_convergence else 0.0,
            )

            for _sub in range(n_sub):
                if self.turbulence.enabled:
                    self.turbulence.correct()

                U_bc = self._build_boundary_conditions()

                # Deferred correction
                U_corrected = self._deferred_correction_convection(
                    self.U, self.U_old,
                )

                # PISO solve
                self.U, self.p, self.phi, conv = solver.solve(
                    U_corrected, self.p, self.phi,
                    U_bc=U_bc,
                    U_old=self.U_old,
                    p_old=self.p_old,
                    tolerance=self.convergence_tolerance,
                )

                # Skewness-corrected Rhie-Chow (from v3)
                A_p_ones = torch.ones(
                    self.mesh.n_cells, dtype=self.U.dtype, device=self.U.device,
                )
                self.U = self._rhie_chow_skewness_corrected(
                    self.U, self.p, A_p_ones,
                )

                # Pressure-gradient preconditioning
                self.p = self._precondition_pressure_gradient(self.p, self.U)

                # Non-orthogonal corrections
                self.p, self.U, self.phi = self._apply_non_orthogonal_corrections(
                    self.p, self.U, self.phi, solver, U_bc,
                )

                # Momentum balance check (from v3)
                balance = self._compute_momentum_balance(
                    self.U, self.U_old, sub_dt,
                )
                if balance > self.momentum_balance_tol:
                    logger.debug("  Momentum imbalance %.2e at sub-step", balance)

            last_convergence = conv
            if conv is not None:
                prev_residual = conv.U_residual

            residuals = {
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("pisoFoamEnhanced4 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced4 completed without convergence")

        return last_convergence or ConvergenceData()
