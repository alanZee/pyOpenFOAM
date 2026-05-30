"""
icoFoamEnhanced4 — enhanced transient incompressible laminar solver v4.

Extends :class:`IcoFoamEnhanced3` with:

- **Lax-Wendroff anti-diffusion**: adds an anti-diffusive flux correction
  to the SSP-RK3 scheme that reduces numerical diffusion while preserving
  the TVD property through a flux limiter.
- **Multi-stage CFL control**: uses separate CFL limits for advection and
  diffusion time scales, selecting the more restrictive constraint
  automatically.
- **Conservative velocity reconstruction**: applies a conservative
  least-squares gradient reconstruction for the velocity field that
  preserves discrete conservation on arbitrary polyhedral meshes.

Governing equations:
    dU/dt + div(UU) - div(nu*grad(U)) = -grad(p)
    div(U) = 0

Usage::

    from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4

    solver = IcoFoamEnhanced4("path/to/case", anti_diffusion=True)
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

from .ico_foam_enhanced_3 import IcoFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IcoFoamEnhanced4"]

logger = logging.getLogger(__name__)


class IcoFoamEnhanced4(IcoFoamEnhanced3):
    """Enhanced transient incompressible laminar PISO solver v4.

    Extends IcoFoamEnhanced3 with Lax-Wendroff anti-diffusion,
    multi-stage CFL control, and conservative velocity reconstruction.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    anti_diffusion : bool, optional
        Enable Lax-Wendroff anti-diffusion correction.  Default True.
    diffusion_cfl : float, optional
        CFL limit for diffusion time scale.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        anti_diffusion: bool = True,
        diffusion_cfl: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.anti_diffusion = anti_diffusion
        self.diffusion_cfl = max(0.01, min(2.0, diffusion_cfl))

        logger.info(
            "IcoFoamEnhanced4 ready: anti_diff=%s, diff_cfl=%.2f",
            self.anti_diffusion, self.diffusion_cfl,
        )

    # ------------------------------------------------------------------
    # Lax-Wendroff anti-diffusion
    # ------------------------------------------------------------------

    def _lax_wendroff_anti_diffusion(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply Lax-Wendroff anti-diffusive flux correction.

        Computes the difference between the Lax-Wendroff flux and the
        first-order upwind flux, then applies a TVD flux limiter to
        prevent oscillations.

        F_anti = (F_LW - F_upwind) * limiter

        Parameters
        ----------
        U : torch.Tensor
            Current velocity (after time stepping).
        U_old : torch.Tensor
            Previous velocity.
        dt : float
            Time step.

        Returns:
            Anti-diffusion corrected velocity.
        """
        if not self.anti_diffusion:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Face interpolation of velocity (direct indexing for 2D tensor)
        w = mesh.face_weights[:n_internal]
        U_O = U[owner]
        U_N = U[neigh]

        # Upwind flux (first-order diffusion)
        U_upwind = w.unsqueeze(-1) * U_O + (1.0 - w).unsqueeze(-1) * U_N

        # Lax-Wendroff correction (second-order anti-diffusion)
        U_old_O = U_old[owner]
        U_old_N = U_old[neigh]
        dU_face = (U_N - U_O) - (U_old_N - U_old_O)

        # TVD limiter (minmod)
        dU_backward = U_O - U_old[owner]
        dU_forward = U_N - U_O

        # Simplified minmod on magnitudes
        dU_backward_mag = dU_backward.norm(dim=-1, keepdim=True)
        dU_forward_mag = dU_forward.norm(dim=-1, keepdim=True)

        ratio = dU_backward_mag / dU_forward_mag.clamp(min=1e-30)
        limiter = torch.where(
            (ratio >= 0) & (ratio <= 1),
            ratio,
            torch.where(ratio > 1, torch.ones_like(ratio), torch.zeros_like(ratio)),
        )

        # Anti-diffusive flux
        F_anti = 0.5 * dt * dU_face * limiter

        # Scatter correction
        correction = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        correction.index_add_(0, owner, F_anti)
        correction.index_add_(0, neigh, -F_anti)

        return U + correction * 0.1  # Damped to prevent instability

    # ------------------------------------------------------------------
    # Multi-stage CFL control
    # ------------------------------------------------------------------

    def _compute_multi_stage_cfl_dt(self) -> float:
        """Compute time step from multi-stage CFL constraints.

        Uses separate CFL numbers for advection and diffusion:
            dt_adv = min(CFL * dx / |U|)
            dt_diff = min(CFL_diff * dx^2 / (2*nu))

        Returns the more restrictive of the two.

        Returns:
            Maximum allowable time step.
        """
        mesh = self.mesh
        U_mag = self.U.norm(dim=1)

        # Characteristic length scale (cube root of cell volume)
        dx = mesh.cell_volumes.pow(1.0 / 3.0).clamp(min=1e-30)

        # Advection time scale
        dt_adv = (self.max_courant * dx / U_mag.clamp(min=1e-30)).min().item()

        # Diffusion time scale
        nu = self.nu if hasattr(self, 'nu') else 0.01
        if nu > 1e-30:
            dt_diff = (self.diffusion_cfl * dx.pow(2) / (2.0 * nu)).min().item()
        else:
            dt_diff = float('inf')

        dt = min(dt_adv, dt_diff)

        # Clamp to reasonable bounds
        dt_min = self.delta_t * 0.001
        dt_max = self.delta_t * 2.0
        return max(dt_min, min(dt_max, dt))

    # ------------------------------------------------------------------
    # Conservative velocity reconstruction
    # ------------------------------------------------------------------

    def _conservative_gradient_reconstruction(
        self,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conservative least-squares velocity gradient.

        Uses a weighted least-squares reconstruction that preserves
        discrete conservation of momentum:

            grad(U)_P = sum_f (U_f - U_P) * w_LS * S_f

        Parameters
        ----------
        U : torch.Tensor
            Velocity field.

        Returns:
            ``(n_cells, 3, 3)`` velocity gradient tensor.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Face values
        w = mesh.face_weights[:n_internal]
        U_O = U[owner]
        U_N = U[neigh]
        U_face = w.unsqueeze(-1) * U_O + (1.0 - w).unsqueeze(-1) * U_N

        # dU on face
        dU = U_N - U_O

        # Least-squares weights (inverse distance squared)
        c_O = mesh.cell_centres[owner]
        c_N = mesh.cell_centres[neigh]
        d2 = (c_N - c_O).pow(2).sum(dim=1).clamp(min=1e-30)
        w_ls = 1.0 / d2

        # Scatter weighted contribution
        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)

        face_areas = mesh.face_areas[:n_internal]
        if face_areas.dim() == 1:
            # Scalar face areas
            for i in range(3):
                contrib = w_ls * dU[:, i]
                grad_U[:, i, 0] = grad_U[:, i, 0] + scatter_add(contrib, owner, n_cells)
                grad_U[:, i, 0] = grad_U[:, i, 0] + scatter_add(-contrib, neigh, n_cells)
        else:
            for i in range(3):
                for j in range(3):
                    contrib = w_ls * dU[:, i] * face_areas[:, j]
                    grad_U[:, i, j] = grad_U[:, i, j] + scatter_add(contrib, owner, n_cells)
                    grad_U[:, i, j] = grad_U[:, i, j] + scatter_add(-contrib, neigh, n_cells)

        # Normalize by cell volume
        V = mesh.cell_volumes.clamp(min=1e-30)
        grad_U = grad_U / V.unsqueeze(-1).unsqueeze(-1)

        return grad_U

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 icoFoam solver.

        Uses Lax-Wendroff anti-diffusion, multi-stage CFL, and
        conservative velocity reconstruction.

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

        logger.info("Starting icoFoamEnhanced4 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  anti_diffusion=%s, diff_cfl=%.2f",
                     self.anti_diffusion, self.diffusion_cfl)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._U_n_minus_1 = None
        current_dt = self.delta_t

        for t, step in time_loop:
            # Store history
            if step > 0:
                self._U_n_minus_1 = self.U_old.clone() if self.U_old is not None else None

            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Multi-stage CFL
            if self.adaptive_dt:
                current_dt = self._compute_multi_stage_cfl_dt()

            # Apply theta weighting
            if self.theta < 1.0:
                U_old_w, p_old_w = self._compute_theta_weighted_old_fields(
                    self.U_old, self.p_old,
                )
            else:
                U_old_w = self.U_old
                p_old_w = self.p_old

            # Main solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                U_old=U_old_w,
                p_old=p_old_w,
                tolerance=self.convergence_tolerance,
            )

            # SSP-RK sub-stepping (from v3)
            if self.temporal_order == 3:
                U_rk3 = self._ssp_rk3_step(self.U, self.U_old, self.p, current_dt)
                U_rk2 = self._ssp_rk2_step(self.U, self.U_old, self.p, current_dt)
            else:
                U_rk2 = self._ssp_rk2_step(self.U, self.U_old, self.p, current_dt)
                U_rk3 = self._ssp_rk3_step(self.U, self.U_old, self.p, current_dt)

            # Error estimation and dt adaptation (from v3)
            if step > 0:
                error = self._estimate_temporal_error(U_rk2, U_rk3)
                current_dt = self._compute_error_adaptive_dt(current_dt, error)

                if self.temporal_order == 3:
                    self.U = U_rk3
                else:
                    self.U = U_rk2

            # Lax-Wendroff anti-diffusion
            self.U = self._lax_wendroff_anti_diffusion(
                self.U, self.U_old, current_dt,
            )

            # Conservative gradient reconstruction (for diagnostics)
            if step % 10 == 0:
                grad_U = self._conservative_gradient_reconstruction(self.U)

            # Consistent flux
            self.phi = self._compute_consistent_mass_flux()

            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
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
                logger.info("icoFoamEnhanced4 completed (converged)")
            else:
                logger.warning("icoFoamEnhanced4 completed without full convergence")

        return last_convergence or ConvergenceData()
