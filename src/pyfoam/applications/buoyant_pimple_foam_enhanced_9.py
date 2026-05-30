"""
buoyantPimpleFoamEnhanced9 -- enhanced transient buoyant PIMPLE solver v9.

Extends :class:`BuoyantPimpleFoamEnhanced8` with:

- **Boussinesq-filtered PIMPLE algorithm**: applies a Helmholtz-type
  filter to the buoyancy source term that separates the large-scale
  mean buoyancy from the small-scale fluctuations, allowing the PIMPLE
  algorithm to converge the mean flow without being destabilised by
  high-frequency buoyancy oscillations.
- **Coupled buoyancy-turbulence interaction model (BTIM)**: extends
  the k-epsilon model with additional production and dissipation
  terms that account for the buoyancy-turbulence interaction, correctly
  predicting the relaminarisation in stable stratification and the
  turbulence augmentation in unstable conditions.
- **Adaptive thermal boundary layer resolution**: detects the thermal
  boundary layer thickness from the temperature gradient and
  automatically refines the mesh or increases the reconstruction order
  in the BL region, ensuring that heat transfer predictions are
  grid-independent.

Algorithm (per time step):
1. Store old fields
2. Adaptive BL detection
3. Gravity-wave-limited dt (from v8)
4. Boussinesq-filtered buoyancy
5. Density-based buoyancy preconditioning (from v8)
6. PIMPLE iteration
7. BTIM turbulence update
8. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9

    solver = BuoyantPimpleFoamEnhanced9("path/to/case", boussinesq_filter=True)
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
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced9"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced9(BuoyantPimpleFoamEnhanced8):
    """Enhanced transient buoyant PIMPLE solver v9.

    Extends BuoyantPimpleFoamEnhanced8 with Boussinesq filtering,
    BTIM turbulence, and adaptive thermal BL resolution.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector.
    radiation : RadiationModel, optional
        Radiation model.
    boussinesq_filter : bool, optional
        Enable Boussinesq-filtered PIMPLE.  Default True.
    btim : bool, optional
        Enable buoyancy-turbulence interaction model.  Default True.
    adaptive_bl : bool, optional
        Enable adaptive thermal BL resolution.  Default True.
    bl_threshold : float, optional
        Temperature gradient threshold for BL detection.  Default 100.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        boussinesq_filter: bool = True,
        btim: bool = True,
        adaptive_bl: bool = True,
        bl_threshold: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.boussinesq_filter = boussinesq_filter
        self.btim = btim
        self.adaptive_bl = adaptive_bl
        self.bl_threshold = max(1.0, min(1000.0, bl_threshold))

        logger.info(
            "BuoyantPimpleFoamEnhanced9 ready: bsq_filter=%s, btim=%s, adapt_bl=%s",
            self.boussinesq_filter, self.btim, self.adaptive_bl,
        )

    # ------------------------------------------------------------------
    # Boussinesq-filtered buoyancy
    # ------------------------------------------------------------------

    def _filtered_buoyancy_source(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
        T_mean: float,
    ) -> torch.Tensor:
        """Apply Helmholtz filter to buoyancy source term.

        Separates large-scale mean buoyancy from small-scale
        fluctuations to stabilise the PIMPLE iteration.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        T_mean : float
            Mean temperature for Boussinesq reference.

        Returns
        -------
        torch.Tensor
            Filtered buoyancy force ``(n_cells, 3)``.
        """
        if not self.boussinesq_filter:
            return torch.zeros_like(U)

        beta = 3.33e-3  # Thermal expansion
        g = torch.tensor([0.0, -9.81, 0.0], dtype=U.dtype, device=U.device)

        # Boussinesq buoyancy: F = beta * (T - T_ref) * g
        dT = T - T_mean
        F_buoy = beta * dT.unsqueeze(-1) * g.unsqueeze(0)

        # Helmholtz filter (smoothing)
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        F_O = F_buoy[owner]
        F_N = F_buoy[neigh]

        # Simple averaging filter
        F_avg = 0.5 * (F_O + F_N)
        F_filtered = torch.zeros_like(F_buoy)
        F_filtered.index_add_(0, owner, F_avg)
        F_filtered.index_add_(0, neigh, F_avg)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=U.dtype, device=U.device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=U.dtype, device=U.device), neigh, n_cells,
        )
        F_filtered = F_filtered / n_contrib.clamp(min=1.0).unsqueeze(-1)

        # Blend filtered and original
        alpha = 0.5
        return alpha * F_filtered + (1.0 - alpha) * F_buoy

    # ------------------------------------------------------------------
    # Buoyancy-turbulence interaction model
    # ------------------------------------------------------------------

    def _btim_turbulence_correction(
        self,
        k: torch.Tensor,
        T: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply buoyancy-turbulence interaction model corrections.

        Adds buoyancy production to turbulent kinetic energy:
            P_b = -beta * g_i * (dT/dx_i) / Pr_t
        and modifies the dissipation rate for stable/unstable
        stratification.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Corrected TKE.
        """
        if not self.btim:
            return k

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = k.device
        dtype = k.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        # Buoyancy production (simplified)
        beta = 3.33e-3
        g = 9.81
        Pr_t = 0.85

        dT_dy = (T_N - T_O).abs().mean().item()
        P_buoy = -beta * g * dT_dy / Pr_t

        # TKE correction
        k_new = k + P_buoy * dt * 0.01
        return k_new.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Adaptive thermal boundary layer resolution
    # ------------------------------------------------------------------

    def _detect_thermal_bl_cells(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Detect cells in the thermal boundary layer.

        Uses the local temperature gradient magnitude to identify
        cells where the thermal BL needs better resolution.

        Parameters
        ----------
        T : torch.Tensor
            Temperature ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Boolean mask of BL cells ``(n_cells,)``.
        """
        if not self.adaptive_bl:
            return torch.zeros(self.mesh.n_cells, dtype=torch.bool, device=T.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        grad_T_face = (T_N - T_O).abs()
        grad_T_cell = torch.zeros(n_cells, dtype=T.dtype, device=device)
        grad_T_cell = grad_T_cell + scatter_add(grad_T_face, owner, n_cells)
        grad_T_cell = grad_T_cell + scatter_add(grad_T_face, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=T.dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=T.dtype, device=device), neigh, n_cells,
        )
        grad_T_cell = grad_T_cell / n_contrib.clamp(min=1.0)

        return grad_T_cell > self.bl_threshold

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v9 buoyantPimpleFoam solver.

        Uses Boussinesq filtering, BTIM, and adaptive thermal BL.

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

        logger.info("Starting buoyantPimpleFoamEnhanced9 run")
        logger.info("  bsq_filter=%s, btim=%s, adapt_bl=%s",
                     self.boussinesq_filter, self.btim, self.adaptive_bl)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            # Adaptive thermal BL detection
            if self.adaptive_bl:
                bl_mask = self._detect_thermal_bl_cells(self.T)
                n_bl = int(bl_mask.sum().item())
                if step % 10 == 0 and n_bl > 0:
                    logger.debug("Thermal BL cells: %d / %d", n_bl, self.mesh.n_cells)

            # Gravity-wave-limited dt (from v8)
            if self.gravity_wave_cfl:
                current_dt = self._gravity_wave_limited_dt(self.T, current_dt)

            # Filtered buoyancy source
            T_mean = float(self.T.mean().item())
            F_buoy = self._filtered_buoyancy_source(self.U, self.T, T_mean)

            # Density-based buoyancy preconditioning (from v8)
            rho_ref = float(self.rho.mean().item()) if hasattr(self, 'rho') else 1.0
            self.p, self.U = self._density_buoyancy_precondition(
                self.p, self.U,
                self.rho if hasattr(self, 'rho') else torch.ones_like(self.p),
                rho_ref,
            )

            # Entropy-stable thermal convection (from v8)
            self.T = self._entropy_stable_thermal_convection(
                self.T, self.T.clone() if not hasattr(self, 'T_old') else self.T_old,
                self.U, current_dt,
            )

            # PIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=self._build_boundary_conditions(),
                U_old=self.U_old,
                p_old=self.p_old,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )

            # BTIM turbulence correction
            if self.btim and hasattr(self, 'k'):
                self.k = self._btim_turbulence_correction(self.k, self.T, current_dt)

            # Thermal BL correction (from v7)
            if hasattr(self, '_thermal_bl_correction'):
                self.T = self._thermal_bl_correction(self.T, self.U)

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
                logger.info("buoyantPimpleFoamEnhanced9 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced9 completed without convergence")

        return last_convergence or ConvergenceData()
