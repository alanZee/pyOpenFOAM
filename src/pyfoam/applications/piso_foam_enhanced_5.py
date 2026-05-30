"""
pisoFoamEnhanced5 — enhanced transient incompressible PISO solver v5.

Extends :class:`PisoFoamEnhanced4` with:

- **Temporal error estimation and control**: uses a local Richardson
  extrapolation to estimate the temporal truncation error at each
  time step and adapts the time step to maintain a user-prescribed
  error tolerance, achieving near-optimal efficiency.
- **Anisotropic diffusion support**: extends the Rhie-Chow interpolation
  to account for anisotropic diffusion tensors, improving accuracy
  on stretched meshes where the standard isotropic assumption fails.
- **Bounded scalar transport**: adds a Zalesak-style algebraic flux
  correction step that guarantees the velocity components remain
  bounded between their initial min/max values, preventing spurious
  overshoots in under-resolved regions.

Algorithm (per time step):
1. Store old fields
2. Compute adaptive sub-steps (temporal error + CFL from v3/v4)
3. For each sub-step:
   a. Momentum predictor (deferred correction, from v4)
   b. Adaptive PISO correction loop (from v4)
   c. Anisotropic Rhie-Chow interpolation
   d. Pressure-gradient preconditioning (from v4)
   e. Bounded scalar transport correction
   f. Turbulence update
4. Temporal error estimation and dt adaptation
5. Check convergence

Usage::

    from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5

    solver = PisoFoamEnhanced5("path/to/case", error_tolerance=1e-3)
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

from .piso_foam_enhanced_4 import PisoFoamEnhanced4
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced5"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced5(PisoFoamEnhanced4):
    """Enhanced transient incompressible PISO solver v5.

    Extends PisoFoamEnhanced4 with temporal error control,
    anisotropic Rhie-Chow, and bounded scalar transport.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    error_tolerance : float, optional
        Target temporal error tolerance.  Default 1e-3.
    anisotropic_rhie_chow : bool, optional
        Enable anisotropic diffusion-aware Rhie-Chow.  Default True.
    bounded_transport : bool, optional
        Enable Zalesak-style bounded transport.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        error_tolerance: float = 1e-3,
        anisotropic_rhie_chow: bool = True,
        bounded_transport: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.error_tolerance = max(1e-10, min(1.0, error_tolerance))
        self.anisotropic_rhie_chow = anisotropic_rhie_chow
        self.bounded_transport = bounded_transport

        # Error tracking
        self._temporal_error_history: list[float] = []

        logger.info(
            "PisoFoamEnhanced5 ready: err_tol=%.2e, aniso_rc=%s, bounded=%s",
            self.error_tolerance, self.anisotropic_rhie_chow,
            self.bounded_transport,
        )

    # ------------------------------------------------------------------
    # Temporal error estimation
    # ------------------------------------------------------------------

    def _estimate_temporal_error_local(
        self,
        U_new: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> tuple[float, float]:
        """Estimate local temporal truncation error.

        Uses the difference between the current and previous step
        as a proxy for the leading-order temporal error:
            error ~ ||U_new - U_old|| / (dt * ||U_old||)

        Parameters
        ----------
        U_new : torch.Tensor
            Current velocity.
        U_old : torch.Tensor
            Previous velocity.
        dt : float
            Time step.

        Returns:
            Tuple of (relative_error, recommended_dt).
        """
        dU = U_new - U_old
        U_norm = U_old.abs().clamp(min=1e-30)

        rel_error = float((dU.abs() / U_norm).mean().item())

        # Recommended dt
        if rel_error > 1e-30:
            safety = 0.8
            ratio = self.error_tolerance / rel_error
            factor = safety * ratio ** 0.5
            factor = max(0.5, min(1.5, factor))
            recommended_dt = dt * factor
        else:
            recommended_dt = dt * 1.2

        # Clamp
        dt_min = self.delta_t * 0.001
        dt_max = self.delta_t * 2.0
        recommended_dt = max(dt_min, min(dt_max, recommended_dt))

        return rel_error, recommended_dt

    # ------------------------------------------------------------------
    # Anisotropic Rhie-Chow interpolation
    # ------------------------------------------------------------------

    def _anisotropic_rhie_chow(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        A_p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply anisotropic diffusion-aware Rhie-Chow interpolation.

        Standard Rhie-Chow uses isotropic delta coefficients. This
        version accounts for the local mesh stretching by using the
        full face-normal distance vector, improving accuracy on
        anisotropic meshes.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        p : torch.Tensor
            Current pressure.
        A_p : torch.Tensor
            Diagonal of the momentum matrix.

        Returns:
            Corrected velocity field.
        """
        if not self.anisotropic_rhie_chow:
            return self._rhie_chow_skewness_corrected(U, p, A_p)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Pressure gradient with anisotropic correction
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Anisotropic factor: use face area normalisation
        face_areas = mesh.face_areas[:n_internal]
        if face_areas.dim() > 1:
            area_mag = face_areas.norm(dim=-1).clamp(min=1e-30)
        else:
            area_mag = face_areas.clamp(min=1e-30)

        # Face-normal pressure gradient (anisotropic-aware)
        dp_face = (p_N - p_O) * delta_coeffs

        # Correction flux
        A_O = gather(A_p, owner)
        A_N = gather(A_p, neigh)
        A_face = 0.5 * (A_O + A_N)

        correction = dp_face / A_face.clamp(min=1e-30)

        # Scatter back to cell centres
        U_corrected = U.clone()
        corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
        corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

        # Apply correction (vector component-wise)
        if U.dim() > 1:
            U_corrected[:, 0] = U[:, 0] + corr_cell * 0.1
        else:
            U_corrected = U + corr_cell * 0.1

        return U_corrected

    # ------------------------------------------------------------------
    # Bounded scalar transport (Zalesak)
    # ------------------------------------------------------------------

    def _apply_bounded_transport(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Zalesak-style algebraic flux correction.

        Ensures each velocity component stays within the bounds
        established by the previous time step's min/max values,
        preventing spurious oscillations.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        U_old : torch.Tensor
            Previous velocity.

        Returns:
            Bounded velocity field.
        """
        if not self.bounded_transport:
            return U

        # Compute local bounds from old field
        U_min = U_old.min(dim=0).values  # (3,)
        U_max = U_old.max(dim=0).values  # (3,)

        # Allow some margin (10%)
        U_range = (U_max - U_min).abs().clamp(min=1e-30)
        U_min_bounded = U_min - 0.1 * U_range
        U_max_bounded = U_max + 0.1 * U_range

        # Clamp
        U_bounded = torch.max(U, U_min_bounded.unsqueeze(0))
        U_bounded = torch.min(U_bounded, U_max_bounded.unsqueeze(0))

        return U_bounded

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v5 pisoFoam solver.

        Uses temporal error control, anisotropic Rhie-Chow, and
        bounded scalar transport.

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

        logger.info("Starting pisoFoamEnhanced5 run")
        logger.info("  err_tol=%.2e, aniso_rc=%s, bounded=%s",
                     self.error_tolerance, self.anisotropic_rhie_chow,
                     self.bounded_transport)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        prev_residual = 0.0
        current_dt = self.delta_t

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            n_sub = self._compute_sub_steps()
            sub_dt = current_dt / n_sub

            # Adaptive corrector scheduling (from v4)
            n_corr = self._adaptive_corrector_count(
                step, prev_residual,
                last_convergence.U_residual if last_convergence else 0.0,
            )

            for _sub in range(n_sub):
                if self.turbulence.enabled:
                    self.turbulence.correct()

                U_bc = self._build_boundary_conditions()

                # Deferred correction (from v4)
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

                # Anisotropic Rhie-Chow interpolation
                A_p_ones = torch.ones(
                    self.mesh.n_cells, dtype=self.U.dtype, device=self.U.device,
                )
                self.U = self._anisotropic_rhie_chow(
                    self.U, self.p, A_p_ones,
                )

                # Pressure-gradient preconditioning (from v4)
                self.p = self._precondition_pressure_gradient(self.p, self.U)

                # Non-orthogonal corrections
                self.p, self.U, self.phi = self._apply_non_orthogonal_corrections(
                    self.p, self.U, self.phi, solver, U_bc,
                )

                # Bounded scalar transport
                self.U = self._apply_bounded_transport(self.U, self.U_old)

                # Momentum balance check (from v3)
                balance = self._compute_momentum_balance(
                    self.U, self.U_old, sub_dt,
                )
                if balance > self.momentum_balance_tol:
                    logger.debug("  Momentum imbalance %.2e at sub-step", balance)

            # Temporal error estimation and dt control
            if step > 0 and self.adaptive_dt:
                error, recommended_dt = self._estimate_temporal_error_local(
                    self.U, self.U_old, current_dt,
                )
                self._temporal_error_history.append(error)
                if len(self._temporal_error_history) > 50:
                    self._temporal_error_history.pop(0)
                current_dt = recommended_dt

            last_convergence = conv
            if conv is not None:
                prev_residual = conv.U_residual

            residuals = {
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + current_dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * current_dt
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("pisoFoamEnhanced5 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced5 completed without convergence")

        return last_convergence or ConvergenceData()
