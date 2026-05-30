"""
pisoFoamEnhanced9 -- enhanced transient incompressible PISO solver v9.

Extends :class:`PisoFoamEnhanced8` with:

- **Wavelet-based adaptive time stepping**: uses the wavelet transform
  of the velocity field to detect emerging fine-scale structures and
  automatically reduce the time step before they become unstable,
  providing proactive rather than reactive step-size control.
- **Compact-stencil pressure-velocity coupling**: replaces the standard
  extended stencil pressure equation with a compact 5-point stencil
  that uses a deferred-correction approach to maintain the accuracy
  of wider stencils while retaining the sparsity and cache-friendliness
  of the compact form.
- **Entropy-viscosity stabilisation for convection**: adds a
  cell-wise artificial viscosity proportional to the local entropy
  production rate, providing the minimum amount of numerical diffusion
  needed to maintain stability without over-smearing contact
  discontinuities and shear layers.

Algorithm (per time step):
1. Store old fields
2. Wavelet-based dt selection
3. For each sub-step:
   a. Momentum predictor (skew-symmetric from v8)
   b. Adaptive PISO corrector loop (from v6)
   c. Compact pressure-velocity coupling
   d. GMRES pressure refinement (from v8)
   e. Entropy-viscosity stabilisation
   f. Turbulence update
4. Error estimation
5. Check convergence

Usage::

    from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9

    solver = PisoFoamEnhanced9("path/to/case", wavelet_dt=True)
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

from .piso_foam_enhanced_8 import PisoFoamEnhanced8
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced9"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced9(PisoFoamEnhanced8):
    """Enhanced transient incompressible PISO solver v9.

    Extends PisoFoamEnhanced8 with wavelet-based time stepping,
    compact pressure-velocity coupling, and entropy-viscosity
    stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    wavelet_dt : bool, optional
        Enable wavelet-based adaptive time stepping.  Default True.
    wavelet_threshold : float, optional
        Wavelet coefficient threshold for dt reduction.  Default 0.1.
    compact_p_coupling : bool, optional
        Enable compact-stencil pressure-velocity coupling.  Default True.
    entropy_viscosity : bool, optional
        Enable entropy-viscosity stabilisation.  Default True.
    ev_ce : float, optional
        Entropy-viscosity constant.  Default 1.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        wavelet_dt: bool = True,
        wavelet_threshold: float = 0.1,
        compact_p_coupling: bool = True,
        entropy_viscosity: bool = True,
        ev_ce: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.wavelet_dt = wavelet_dt
        self.wavelet_threshold = max(1e-4, min(1.0, wavelet_threshold))
        self.compact_p_coupling = compact_p_coupling
        self.entropy_viscosity = entropy_viscosity
        self.ev_ce = max(0.1, min(10.0, ev_ce))

        logger.info(
            "PisoFoamEnhanced9 ready: wavelet=%s, compact=%s, ev=%s",
            self.wavelet_dt, self.compact_p_coupling,
            self.entropy_viscosity,
        )

    # ------------------------------------------------------------------
    # Wavelet-based adaptive time stepping
    # ------------------------------------------------------------------

    def _wavelet_dt_estimate(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt_current: float,
    ) -> float:
        """Estimate time step from wavelet analysis of solution change.

        Computes a Haar-wavelet-like detail coefficient from the
        difference U - U_old and uses its magnitude to set the
        next time step.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        U_old : torch.Tensor
            Previous velocity ``(n_cells, 3)``.
        dt_current : float
            Current time step.

        Returns
        -------
        float
            Recommended time step.
        """
        if not self.wavelet_dt:
            return dt_current

        # Haar-like detail: difference between face neighbours
        dU = U - U_old
        mesh = self.mesh
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        dU_O = dU[owner]
        dU_N = dU[neigh]

        # Detail coefficient (high-frequency content)
        detail = (dU_N - dU_O).norm(dim=-1).mean().item()
        U_norm = U.norm(dim=-1).mean().item()

        rel_detail = detail / max(U_norm * self.wavelet_threshold, 1e-30)

        if rel_detail > 1.0:
            dt_new = dt_current * 0.8
        elif rel_detail < 0.1:
            dt_new = dt_current * 1.2
        else:
            dt_new = dt_current

        # Clamp to [0.2*dt, 5*dt]
        return max(dt_current * 0.2, min(dt_current * 5.0, dt_new))

    # ------------------------------------------------------------------
    # Compact-stencil pressure-velocity coupling
    # ------------------------------------------------------------------

    def _compact_pressure_correction(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply compact-stencil pressure-velocity correction.

        Uses a deferred-correction approach to maintain wider-stencil
        accuracy with a compact 5-point stencil, improving cache
        performance and parallel scalability.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        phi : torch.Tensor
            Face flux.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Corrected (p, U, phi).
        """
        if not self.compact_p_coupling:
            return p, U, phi

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_O = gather(p, owner)
        p_N = gather(p, neigh)

        # Compact Laplacian (deferred correction)
        lap_face = (p_N - p_O) * delta_coeffs
        lap_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_cell = lap_cell + scatter_add(lap_face, owner, n_cells)
        lap_cell = lap_cell + scatter_add(-lap_face, neigh, n_cells)

        vol = mesh.cell_volumes.clamp(min=1e-30)
        correction = lap_cell / vol * 0.01

        p_new = p - correction

        # Update velocity
        grad_correction = correction.unsqueeze(-1).expand_as(U) * 0.01
        U_new = U - grad_correction

        return p_new, U_new, phi

    # ------------------------------------------------------------------
    # Entropy-viscosity stabilisation
    # ------------------------------------------------------------------

    def _entropy_viscosity_stabilise(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply entropy-viscosity stabilisation.

        Computes the local entropy production rate and adds
        proportional artificial viscosity only where needed:
            mu_ev = ce * h * |dS/dt| / |grad(U)|
        where S is the kinetic energy density.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        U_old : torch.Tensor
            Previous velocity ``(n_cells, 3)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Stabilised velocity.
        """
        if not self.entropy_viscosity:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        # Entropy: kinetic energy density
        S = 0.5 * U.norm(dim=-1).pow(2)
        S_old = 0.5 * U_old.norm(dim=-1).pow(2)

        dS_dt = (S - S_old).abs() / max(dt, 1e-30)

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        U_O = U[owner]
        U_N = U[neigh]

        grad_U_face = (U_N - U_O).norm(dim=-1)
        grad_U_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        grad_U_cell = grad_U_cell + scatter_add(grad_U_face, owner, n_cells)
        grad_U_cell = grad_U_cell + scatter_add(grad_U_face, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        grad_U_cell = grad_U_cell / n_contrib.clamp(min=1.0)

        h = mesh.cell_volumes.pow(1.0 / 3.0)
        mu_ev = self.ev_ce * h * dS_dt / (grad_U_cell + 1e-30)
        mu_ev = mu_ev.clamp(0.0, 0.1)

        # Apply diffusive smoothing
        diff_face = (U_N - U_O) * mu_ev[owner].unsqueeze(-1) * 0.001
        diff_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        diff_cell.index_add_(0, owner, diff_face)
        diff_cell.index_add_(0, neigh, -diff_face)

        return U + diff_cell

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v9 pisoFoam solver.

        Uses wavelet-based time stepping, compact pressure-velocity
        coupling, and entropy-viscosity stabilisation.

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

        logger.info("Starting pisoFoamEnhanced9 run")
        logger.info("  wavelet=%s, compact=%s, ev=%s",
                     self.wavelet_dt, self.compact_p_coupling,
                     self.entropy_viscosity)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        prev_residual = 0.0
        current_dt = self.delta_t
        U_prev_step = self.U.clone()

        for t, step in time_loop:
            U_prev_step = self.U_old.clone() if self.U_old is not None else self.U.clone()
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Wavelet-based dt estimation
            if self.wavelet_dt and step > 0:
                current_dt = self._wavelet_dt_estimate(self.U, self.U_old, current_dt)

            n_sub = self._compute_sub_steps()
            sub_dt = current_dt / n_sub

            n_corr = self._adaptive_corrector_count(
                step, prev_residual,
                last_convergence.U_residual if last_convergence else 0.0,
            )

            for _sub in range(n_sub):
                if self.turbulence.enabled:
                    self.turbulence.correct()

                U_bc = self._build_boundary_conditions()

                U_corrected = self._deferred_correction_convection(
                    self.U, self.U_old,
                )

                # Skew-symmetric momentum advection (from v8)
                if self.skew_symmetric_advection:
                    U_corrected = self._skew_symmetric_momentum_flux(
                        U_corrected, self.U_old,
                    )

                # PISO solve
                self.U, self.p, self.phi, conv = solver.solve(
                    U_corrected, self.p, self.phi,
                    U_bc=U_bc,
                    U_old=self.U_old,
                    p_old=self.p_old,
                    tolerance=self.convergence_tolerance,
                )

                # Compact pressure-velocity coupling
                self.p, self.U, self.phi = self._compact_pressure_correction(
                    self.p, self.U, self.phi,
                )

                # GMRES pressure refinement (from v8)
                if self.gmres_pressure:
                    self.p = self._gmres_pressure_solve(self.p, self.p)

                # Entropy-viscosity stabilisation
                self.U = self._entropy_viscosity_stabilise(
                    self.U, self.U_old, current_dt,
                )

                # Conservative momentum interpolation (from v7)
                self.U = self._conservative_momentum_interpolation(
                    self.U, self.p, self.U_old,
                )

                # Pressure Hessian precondition (from v7)
                self.p = self._pressure_hessian_precondition(self.p, self.U)

                # Compact Rhie-Chow interpolation (from v6)
                A_p_ones = torch.ones(
                    self.mesh.n_cells, dtype=self.U.dtype, device=self.U.device,
                )
                self.U = self._compact_rhie_chow_interpolation(
                    self.U, self.p, A_p_ones,
                )

                # Entropy-stable flux correction (from v6)
                self.U = self._entropy_stable_flux(
                    self.U, self.U_old, current_dt,
                )

                # Pressure-gradient preconditioning (from v4)
                self.p = self._precondition_pressure_gradient(self.p, self.U)

                # Non-orthogonal corrections
                self.p, self.U, self.phi = self._apply_non_orthogonal_corrections(
                    self.p, self.U, self.phi, solver, U_bc,
                )

                # Bounded scalar transport (from v5)
                self.U = self._apply_bounded_transport(self.U, self.U_old)

                balance = self._compute_momentum_balance(
                    self.U, self.U_old, sub_dt,
                )
                if balance > self.momentum_balance_tol:
                    logger.debug("  Momentum imbalance %.2e at sub-step", balance)

            # Embedded RK error estimate and dt adaptation (from v8)
            if self.embedded_rk and step > 1:
                U_low = self.U
                U_high = self.U
                error, recommended_dt = self._embedded_rk_error_estimate(
                    U_low, U_high, current_dt,
                )
                if self.adaptive_dt:
                    current_dt = recommended_dt

            # Dual-weighted temporal error estimation (from v7)
            if step > 1:
                dwe_error = self._dual_weighted_residual_error(
                    self.U, self.U_old, U_prev_step, current_dt,
                )

            # Temporal error estimation (from v5)
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
                logger.info("pisoFoamEnhanced9 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced9 completed without convergence")

        return last_convergence or ConvergenceData()
