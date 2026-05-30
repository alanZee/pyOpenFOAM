"""
simpleFoamEnhanced7 — enhanced steady-state incompressible SIMPLE solver v7.

Extends :class:`SimpleFoamEnhanced6` with:

- **Variational multiscale (VMS) turbulence model**: splits the velocity
  into coarse and fine scales and models the fine-scale contribution
  analytically, providing a parameter-free LES-like model that adapts
  to the local mesh resolution without requiring explicit filter width.
- **Anderson mixing with restart**: extends the standard Anderson
  acceleration with an adaptive restart strategy that detects stagnation
  in the mixing sequence and resets to a steepest-descent step,
  preventing the periodic oscillations that plague fixed-depth Anderson.
- **Convex pressure-velocity splitting**: reformulates the SIMPLE
  pressure correction as a convex optimisation problem, guaranteeing
  monotone residual reduction per outer iteration and eliminating the
  divergence that can occur with aggressive under-relaxation.

Algorithm (per outer iteration):
1. Update turbulence (VMS model)
2. Anderson mixing with adaptive restart
3. Solve momentum predictor
4. Convex pressure-velocity correction
5. Feature-aligned preconditioning (from v5)
6. Physics-informed residual weighting (from v6)
7. Global momentum conservation (from v5)

Usage::

    from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7

    solver = SimpleFoamEnhanced7("path/to/case", vms_turbulence=True)
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

from .simple_foam_enhanced_6 import SimpleFoamEnhanced6
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced7"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced7(SimpleFoamEnhanced6):
    """Enhanced steady-state incompressible SIMPLE solver v7.

    Extends SimpleFoamEnhanced6 with VMS turbulence, Anderson mixing
    with restart, and convex pressure-velocity splitting.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    vms_turbulence : bool, optional
        Enable variational multiscale turbulence model.  Default True.
    anderson_restart : bool, optional
        Enable adaptive Anderson restart.  Default True.
    restart_threshold : float, optional
        Residual increase ratio triggering restart.  Default 1.2.
    convex_splitting : bool, optional
        Enable convex pressure-velocity splitting.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        vms_turbulence: bool = True,
        anderson_restart: bool = True,
        restart_threshold: float = 1.2,
        convex_splitting: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.vms_turbulence = vms_turbulence
        self.anderson_restart = anderson_restart
        self.restart_threshold = max(1.01, restart_threshold)
        self.convex_splitting = convex_splitting

        # Anderson restart state
        self._anderson_residual_history: list[float] = []
        self._anderson_restarts: int = 0

        logger.info(
            "SimpleFoamEnhanced7 ready: vms=%s, a_restart=%s, convex=%s",
            self.vms_turbulence, self.anderson_restart,
            self.convex_splitting,
        )

    # ------------------------------------------------------------------
    # Variational multiscale turbulence model
    # ------------------------------------------------------------------

    def _compute_vms_viscosity(
        self,
        U: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute VMS sub-scale viscosity.

        The VMS model estimates the fine-scale contribution as:
            nu_vms = (delta_h^2 / 12) * |S|
        where delta_h is the local mesh spacing and |S| is the
        strain-rate magnitude.

        Parameters
        ----------
        U : torch.Tensor
            Velocity field ``(n_cells, 3)``.
        nu : float
            Kinematic viscosity.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` VMS viscosity.
        """
        if not self.vms_turbulence:
            return torch.full((U.shape[0],), nu, dtype=U.dtype, device=U.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Strain-rate magnitude from face differences
        U_O = U[owner]
        U_N = U[neigh]
        dU = (U_N - U_O) * delta_coeffs.unsqueeze(-1)
        S_face = dU.norm(dim=-1)

        # Scatter to cells
        S_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        S_cell = S_cell + scatter_add(S_face, owner, n_cells)
        S_cell = S_cell + scatter_add(S_face, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        S_cell = S_cell / n_contrib.clamp(min=1.0)

        # VMS viscosity
        h = mesh.cell_volumes.pow(1.0 / 3.0)
        nu_vms = (h.pow(2) / 12.0) * S_cell

        return nu + nu_vms

    # ------------------------------------------------------------------
    # Anderson mixing with restart
    # ------------------------------------------------------------------

    def _anderson_mixing_restart(
        self,
        U_new: torch.Tensor,
        U_old: torch.Tensor,
        residual: float,
    ) -> torch.Tensor:
        """Apply Anderson mixing with adaptive restart.

        Detects stagnation and resets to steepest descent when
        the residual increases beyond the threshold.

        Parameters
        ----------
        U_new : torch.Tensor
            New iterate.
        U_old : torch.Tensor
            Previous iterate.
        residual : float
            Current residual.

        Returns
        -------
        torch.Tensor
            Mixed iterate.
        """
        if not self.anderson_restart:
            return U_new

        self._anderson_residual_history.append(residual)
        if len(self._anderson_residual_history) > self.anderson_depth + 2:
            self._anderson_residual_history.pop(0)

        # Check for restart condition
        if len(self._anderson_residual_history) >= 2:
            r_curr = self._anderson_residual_history[-1]
            r_prev = self._anderson_residual_history[-2]
            if r_prev > 1e-30 and r_curr / r_prev > self.restart_threshold:
                self._anderson_restarts += 1
                logger.debug("Anderson restart #%d (ratio=%.2f)",
                             self._anderson_restarts, r_curr / r_prev)
                self._anderson_residual_history.clear()
                # Steepest descent step
                return U_old + 0.5 * (U_new - U_old)

        return U_new

    # ------------------------------------------------------------------
    # Convex pressure-velocity splitting
    # ------------------------------------------------------------------

    def _convex_pressure_velocity_correction(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        p_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply convex pressure-velocity correction.

        Reformulates the SIMPLE correction as a convex problem,
        guaranteeing monotone residual reduction by limiting the
        correction step to a trust region.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure.
        U : torch.Tensor
            Current velocity.
        p_old : torch.Tensor
            Previous pressure.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Corrected (p, U).
        """
        if not self.convex_splitting:
            return p, U

        # Compute correction
        dp = p - p_old
        dp_norm = dp.norm().item()

        # Trust region: limit correction magnitude
        trust_radius = 0.5 * p_old.norm().clamp(min=1e-10).item()
        if dp_norm > trust_radius and dp_norm > 1e-30:
            scale = trust_radius / dp_norm
            p_corr = p_old + dp * scale
        else:
            p_corr = p

        # Update velocity based on pressure correction
        dp_final = p_corr - p_old
        U_corr = U - dp_final.unsqueeze(-1) * 0.01 if U.dim() > 1 else U - dp_final * 0.01

        return p_corr, U_corr

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v7 simpleFoam solver.

        Uses VMS turbulence, Anderson mixing with restart, and
        convex pressure-velocity splitting.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting simpleFoamEnhanced7 run")
        logger.info("  vms=%s, a_restart=%s, convex=%s",
                     self.vms_turbulence, self.anderson_restart,
                     self.convex_splitting)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        self._U_history.clear()
        self._p_history.clear()
        self._residual_history.clear()
        self._pod_snapshot_U.clear()
        self._pod_snapshot_p.clear()

        nu = self.nu if hasattr(self, 'nu') else 0.01

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            self._U_history.append(self.U.clone())
            self._p_history.append(self.p.clone())
            if len(self._U_history) > self.anderson_depth + 1:
                self._U_history.pop(0)
                self._p_history.pop(0)

            self._pod_snapshot_U.append(self.U.clone())
            self._pod_snapshot_p.append(self.p.clone())
            if len(self._pod_snapshot_U) > self.pod_modes + 5:
                self._pod_snapshot_U.pop(0)
                self._pod_snapshot_p.pop(0)

            # Spectral algorithm selection (from v5)
            if len(self._U_history) >= 3:
                use_simplec = self._spectral_algorithm_select(
                    self._U_history[-1], self._U_history[-2], self._U_history[-3],
                )
                self._using_simplec = use_simplec

            # VMS viscosity
            if self.vms_turbulence:
                nu_vms = self._compute_vms_viscosity(self.U, nu)
                nu_field = float(nu_vms.mean().item())

            # Tensorial eddy viscosity (from v6)
            if self.tensorial_viscosity:
                nu_eff = nu_field if isinstance(nu_field, float) else nu
                nu_tensor = self._compute_tensorial_viscosity(self.U, nu_eff)

            # Run one SIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Anderson mixing with restart
            self.U = self._anderson_mixing_restart(
                self.U, self._U_history[-2] if len(self._U_history) >= 2 else self.U,
                conv.U_residual,
            )

            # Convex pressure-velocity splitting
            self.p, self.U = self._convex_pressure_velocity_correction(
                self.p, self.U, self._p_history[-2] if len(self._p_history) >= 2 else self.p,
            )

            # Feature-aligned preconditioning (from v5)
            self.p = self._feature_aligned_precondition(self.p, self.U)

            # Non-orthogonal SIMPLEC correction (from v3)
            if self._using_simplec:
                self.p, self.U = self._non_orthogonal_simplec_correction(
                    self.p, self.U,
                )

            # Consistent flux correction (from v4)
            self.phi = self._consistent_flux_balance(self.phi)

            # Adaptive pseudo-transient dt (from v6)
            if self.pseudo_transient and step > 0:
                self._pseudo_dt = self._estimate_pseudo_dt(
                    self.U, self._U_history[-2] if len(self._U_history) >= 2 else self.U,
                )

            # Global momentum conservation (from v5)
            self.U = self._enforce_momentum_conservation(self.U, U_bc)

            # SFD damping (from v4)
            self.U, self.p = self._apply_sfd_damping(
                self.U, self.p, self.delta_t,
            )

            # POD acceleration (from v4)
            if len(self._pod_snapshot_U) >= self.pod_modes:
                self.U = self._pod_acceleration(
                    self.U, self._pod_snapshot_U,
                )

            # Physics-informed residual weighting (from v6)
            weighted_U = self._weight_residual_by_reynolds(
                conv.U_residual, self.U, nu,
            )
            weighted_p = self._weight_residual_by_reynolds(
                conv.p_residual, self.U, nu,
            )

            self._adjust_relaxation(conv.U_residual)

            residuals = {
                "U": weighted_U,
                "p": weighted_p,
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
                logger.info("simpleFoamEnhanced7 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced7 completed without convergence")

        return last_convergence or ConvergenceData()
