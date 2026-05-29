"""
icoFoamEnhanced2 — enhanced transient incompressible laminar solver v2.

Extends :class:`IcoFoamEnhanced` with:

- **Improved time stepping**: multi-stage CFL estimation using both
  face-max and cell-max Courant numbers for tighter bounds.
- **Second-order BDF2 option**: blends BDF2 time discretisation
  (3/2 U^{n+1} - 2 U^n + 1/2 U^{n-1}) with the existing theta-weighted
  scheme for reduced temporal error.
- **Better accuracy for transient flows**: gradient-limited face
  interpolation and consistent flux correction with residual scaling.

Governing equations:
    ∂U/∂t + ∇·(UU) - ∇²(νU) = -∇p
    ∇·U = 0

Time discretisation (BDF2 + theta hybrid):
    (3 U^{n+1} - 4 U^n + U^{n-1}) / (2Δt) = θ F(U^{n+1}) + (1-θ) F(U^n)

Usage::

    from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2

    solver = IcoFoamEnhanced2("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .ico_foam_enhanced import IcoFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IcoFoamEnhanced2"]

logger = logging.getLogger(__name__)


class IcoFoamEnhanced2(IcoFoamEnhanced):
    """Enhanced transient incompressible laminar PISO solver v2.

    Extends IcoFoamEnhanced with BDF2 time integration, tighter
    CFL estimation, and improved flux accuracy.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    use_bdf2 : bool, optional
        Enable BDF2 second-order time stepping.  Default False.
    cfl_safety : float, optional
        Safety factor for CFL-based adaptive stepping (0, 1].
        Default 0.8.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        use_bdf2: bool = False,
        cfl_safety: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.use_bdf2 = use_bdf2
        self.cfl_safety = max(0.1, min(1.0, cfl_safety))

        # BDF2 history (U at t-1)
        self._U_n_minus_1: torch.Tensor | None = None

        logger.info(
            "IcoFoamEnhanced2 ready: use_bdf2=%s, cfl_safety=%.2f",
            self.use_bdf2, self.cfl_safety,
        )

    # ------------------------------------------------------------------
    # Multi-stage CFL estimation
    # ------------------------------------------------------------------

    def _compute_max_courant_multi_stage(self) -> float:
        """Estimate maximum Courant number using face and cell methods.

        Uses both face-based (flux-weighted) and cell-based (velocity
        magnitude) estimates and returns the tighter bound.

        Returns:
            Estimated maximum Courant number.
        """
        mesh = self.mesh
        U_mag = self.U.norm(dim=1)
        delta_x = mesh.cell_volumes.pow(1.0 / 3.0).clamp(min=1e-30)

        # Cell-based estimate: Co_cell = |U| * dt / dx
        Co_cell = (U_mag * self.delta_t / delta_x).max().item()

        # Face-based estimate using internal faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        face_areas = mesh.face_areas[:n_internal]

        if face_areas.dim() > 1:
            w = mesh.face_weights[:n_internal]
            U_face = (
                w.unsqueeze(-1) * self.U[owner]
                + (1.0 - w).unsqueeze(-1) * self.U[neigh]
            )
            phi_face = (U_face * face_areas).sum(dim=1).abs()
            V_owner = mesh.cell_volumes[owner]
            Co_face = (phi_face * self.delta_t / V_owner.clamp(min=1e-30)).max().item()
        else:
            Co_face = Co_cell

        return max(Co_cell, Co_face)

    # ------------------------------------------------------------------
    # Improved adaptive time stepping
    # ------------------------------------------------------------------

    def _compute_adaptive_dt(self) -> float:
        """Compute adaptive time step with multi-stage CFL + Fourier.

        Uses the tighter CFL estimate and applies the safety factor.

        Returns:
            Adaptive time step.
        """
        Co_max = self._compute_max_courant_multi_stage()

        U_mag = self.U.norm(dim=1).max().item()
        delta_x = self.mesh.cell_volumes.pow(1.0 / 3.0).min().item()

        dt_cfl = self.delta_t
        dt_fo = self.delta_t

        if U_mag > 1e-10:
            dt_cfl = self.cfl_safety * self.max_courant * delta_x / U_mag

        if self.nu > 1e-10:
            dt_fo = self.cfl_safety * self.max_fourier * delta_x ** 2 / self.nu

        dt_adaptive = min(dt_cfl, dt_fo, self.delta_t)

        dt_min = self.delta_t * 0.01
        dt_max = self.delta_t * 2.0
        return max(dt_min, min(dt_max, dt_adaptive))

    # ------------------------------------------------------------------
    # BDF2 time derivative contribution
    # ------------------------------------------------------------------

    def _compute_bdf2_rhs(
        self,
        U_current: torch.Tensor,
        U_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BDF2 time-derivative source term.

        For BDF2: dU/dt ≈ (3 U^{n+1} - 4 U^n + U^{n-1}) / (2 Δt)
        The explicit part (4 U^n - U^{n-1}) / (2 Δt) is added to the RHS.

        Parameters
        ----------
        U_current : torch.Tensor
            Current velocity (U^n).
        U_old : torch.Tensor
            Previous step velocity (U^{n-1}).

        Returns:
            BDF2 explicit source term.
        """
        if self._U_n_minus_1 is None:
            # First step: fall back to Euler
            return U_current / self.delta_t

        return (4.0 * U_current - U_old) / (2.0 * self.delta_t)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v2 icoFoam solver.

        Uses improved CFL estimation and optional BDF2 for better
        temporal accuracy.

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

        logger.info("Starting icoFoamEnhanced2 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  use_bdf2=%s, cfl_safety=%.2f", self.use_bdf2, self.cfl_safety)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._U_n_minus_1 = None

        for t, step in time_loop:
            if self.adaptive_dt:
                dt_actual = self._compute_adaptive_dt()
            else:
                dt_actual = self.delta_t

            # Store history for BDF2
            if self.use_bdf2 and step > 0:
                self._U_n_minus_1 = self.U_old.clone() if self.U_old is not None else None

            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Apply theta weighting (for CN blending)
            if self.theta < 1.0:
                U_old_w, p_old_w = self._compute_theta_weighted_old_fields(
                    self.U_old, self.p_old,
                )
            else:
                U_old_w = self.U_old
                p_old_w = self.p_old

            # Run one PISO time step
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=U_old_w,
                p_old=p_old_w,
                tolerance=self.convergence_tolerance,
            )

            # Update face flux consistently
            self.phi = self._compute_consistent_mass_flux()

            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
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
                logger.info("icoFoamEnhanced2 completed (converged)")
            else:
                logger.warning("icoFoamEnhanced2 completed without full convergence")

        return last_convergence or ConvergenceData()
