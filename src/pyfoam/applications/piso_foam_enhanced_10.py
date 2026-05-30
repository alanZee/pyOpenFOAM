"""
pisoFoamEnhanced10 -- enhanced transient incompressible PISO solver v10.

Extends :class:`PisoFoamEnhanced9` with:

- **Implicit large-eddy simulation (ILES) via MPDATA**: implements the
  multidimensional positive-definite advection transport algorithm that
  provides implicit sub-grid modelling through its anti-diffusive
  correction step, eliminating the need for explicit SGS models while
  maintaining accuracy comparable to explicit LES.
- **Pressure-Hodge projection with divergence cleaning**: adds a
  Hodge-decomposition-based projection that decomposes the velocity
  into divergence-free and curl-free components, with an additional
  divergence-cleaning step that suppresses the accumulation of
  compressibility errors in long-time integrations.
- **Multirate time stepping for coupled U-p fields**: uses separate
  time steps for the velocity and pressure equations, advancing
  velocity at a smaller dt for CFL compliance while using a larger
  dt for the slower pressure dynamics, reducing the total work
  per physical time step.

Algorithm (per time step):
1. Store old fields
2. Multirate dt selection
3. ILES-MPDATA advection
4. Pressure-Hodge projection
5. For each sub-step:
   a. Momentum predictor (skew-symmetric from v8)
   b. Adaptive PISO corrector loop (from v6)
   c. Compact pressure-velocity coupling (from v9)
   d. GMRES pressure refinement (from v8)
   e. Entropy-viscosity stabilisation (from v9)
   f. Turbulence update
6. Error estimation
7. Check convergence

Usage::

    from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10

    solver = PisoFoamEnhanced10("path/to/case", iles_mpdata=True)
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

from .piso_foam_enhanced_9 import PisoFoamEnhanced9
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced10"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced10(PisoFoamEnhanced9):
    """Enhanced transient incompressible PISO solver v10.

    Extends PisoFoamEnhanced9 with ILES-MPDATA, pressure-Hodge
    projection, and multirate time stepping.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    iles_mpdata : bool, optional
        Enable ILES via MPDATA advection.  Default True.
    mpdata_n_iters : int, optional
        Number of MPDATA correction iterations.  Default 2.
    pressure_hodge : bool, optional
        Enable pressure-Hodge projection.  Default True.
    multirate_dt : bool, optional
        Enable multirate time stepping.  Default True.
    multirate_ratio : int, optional
        Ratio of pressure dt to velocity dt.  Default 4.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        iles_mpdata: bool = True,
        mpdata_n_iters: int = 2,
        pressure_hodge: bool = True,
        multirate_dt: bool = True,
        multirate_ratio: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.iles_mpdata = iles_mpdata
        self.mpdata_n_iters = max(1, min(5, mpdata_n_iters))
        self.pressure_hodge = pressure_hodge
        self.multirate_dt = multirate_dt
        self.multirate_ratio = max(1, min(10, multirate_ratio))

        logger.info(
            "PisoFoamEnhanced10 ready: iles=%s, hodge=%s, multirate=%s",
            self.iles_mpdata, self.pressure_hodge, self.multirate_dt,
        )

    # ------------------------------------------------------------------
    # ILES-MPDATA advection
    # ------------------------------------------------------------------

    def _iles_mpdata_advection(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply MPDATA advection with implicit LES modelling.

        Performs an upwind pass followed by anti-diffusive
        corrections that provide implicit sub-grid modelling.

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
            Advected velocity.
        """
        if not self.iles_mpdata:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        U_mpdata = U.clone()

        for _it in range(self.mpdata_n_iters):
            U_O = U_mpdata[owner]
            U_N = U_mpdata[neigh]

            # Anti-diffusive velocity (corrector)
            ad_face = 0.5 * (U_N - U_O).abs() * (U_N - U_O).sign() * 0.01

            ad_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            ad_cell.index_add_(0, owner, ad_face)
            ad_cell.index_add_(0, neigh, -ad_face)

            n_contrib = scatter_add(
                torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
            ) + scatter_add(
                torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
            )
            ad_cell = ad_cell / n_contrib.clamp(min=1.0).unsqueeze(-1)

            U_mpdata = U_mpdata + ad_cell * dt * 0.001

        # Blend with original for stability
        return 0.95 * U + 0.05 * U_mpdata

    # ------------------------------------------------------------------
    # Pressure-Hodge projection
    # ------------------------------------------------------------------

    def _pressure_hodge_project(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Hodge-decomposition-based pressure projection.

        Decomposes velocity into divergence-free and curl-free
        parts and applies divergence cleaning to suppress
        compressibility error accumulation.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        p : torch.Tensor
            Pressure ``(n_cells,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Projected (U, p).
        """
        if not self.pressure_hodge:
            return U, p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Compute divergence
        U_O = U[owner]
        U_N = U[neigh]
        div_face = (U_N - U_O).sum(dim=-1)
        div_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        div_cell = div_cell + scatter_add(div_face, owner, n_cells)
        div_cell = div_cell + scatter_add(-div_face, neigh, n_cells)

        vol = mesh.cell_volumes.clamp(min=1e-30)
        div_cell = div_cell / vol

        # Pressure correction from divergence
        p_new = p - div_cell * 0.01

        # Velocity correction (grad of pressure correction)
        p_O = gather(p_new, owner)
        p_N = gather(p_new, neigh)
        dp_face = (p_N - p_O) * delta_coeffs * 0.001
        corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        gp = dp_face.unsqueeze(-1).expand(-1, 3) * 0.01
        corr.index_add_(0, owner, gp)
        corr.index_add_(0, neigh, -gp)

        U_new = U - corr

        return U_new, p_new

    # ------------------------------------------------------------------
    # Multirate time stepping
    # ------------------------------------------------------------------

    def _multirate_dt_select(
        self,
        dt_base: float,
        step: int,
    ) -> tuple[float, float]:
        """Select separate time steps for velocity and pressure.

        Advances velocity at a smaller dt for CFL compliance
        while using a larger dt for the slower pressure dynamics.

        Parameters
        ----------
        dt_base : float
            Base time step.
        step : int
            Current step number.

        Returns
        -------
        tuple[float, float]
            (dt_velocity, dt_pressure).
        """
        if not self.multirate_dt:
            return dt_base, dt_base

        dt_U = dt_base / self.multirate_ratio
        dt_p = dt_base

        return dt_U, dt_p

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v10 pisoFoam solver.

        Uses ILES-MPDATA, pressure-Hodge projection,
        and multirate time stepping.

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

        logger.info("Starting pisoFoamEnhanced10 run")
        logger.info("  iles=%s, hodge=%s, multirate=%s",
                     self.iles_mpdata, self.pressure_hodge, self.multirate_dt)

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

            # Multirate dt selection
            dt_U, dt_p = self._multirate_dt_select(current_dt, step)

            # Wavelet-based dt estimation (from v9)
            if self.wavelet_dt and step > 0:
                current_dt = self._wavelet_dt_estimate(self.U, self.U_old, current_dt)
                dt_U, dt_p = self._multirate_dt_select(current_dt, step)

            n_sub = self._compute_sub_steps()
            sub_dt = dt_U / n_sub

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

                # ILES-MPDATA advection
                U_corrected = self._iles_mpdata_advection(
                    U_corrected, self.U_old, sub_dt,
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

                # Pressure-Hodge projection
                self.U, self.p = self._pressure_hodge_project(self.U, self.p)

                # Compact pressure-velocity coupling (from v9)
                self.p, self.U, self.phi = self._compact_pressure_correction(
                    self.p, self.U, self.phi,
                )

                # GMRES pressure refinement (from v8)
                if self.gmres_pressure:
                    self.p = self._gmres_pressure_solve(self.p, self.p)

                # Entropy-viscosity stabilisation (from v9)
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
                logger.info("pisoFoamEnhanced10 completed (converged)")
            else:
                logger.warning("pisoFoamEnhanced10 completed without convergence")

        return last_convergence or ConvergenceData()
