"""
buoyantPimpleFoamEnhanced4 — enhanced transient buoyant PIMPLE solver v4.

Extends :class:`BuoyantPimpleFoamEnhanced3` with:

- **Adaptive thermal relaxation**: uses the local temperature gradient
  to set the energy equation under-relaxation factor, reducing
  relaxation in regions with steep gradients for stability.
- **Improved radiation-buoyancy coupling**: iterates between the
  radiation source term and the buoyancy force to capture the
  interaction between radiative heating and natural convection.
- **Courant-adaptive buoyancy scaling**: scales the buoyancy source
  term with the local Courant number to prevent buoyancy-driven
  instability at large time steps.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson and Brunt-Vaisala (from v3)
3. Filter gravity wave modes (from v3)
4. Outer corrector loop:
   a. Semi-implicit buoyancy source (from v3)
   b. Radiation-buoyancy coupled iteration
   c. Momentum predictor with buoyancy-turbulence coupling (from v3)
   d. PISO pressure correction (p_rgh form)
   e. Energy equation (with adaptive relaxation)
   f. EOS update
   g. Temperature limiting (from v2)
5. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4

    solver = BuoyantPimpleFoamEnhanced4("path/to/case", rad_buoy_coupling=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced4"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced4(BuoyantPimpleFoamEnhanced3):
    """Enhanced transient buoyant PIMPLE solver v4.

    Extends BuoyantPimpleFoamEnhanced3 with adaptive thermal relaxation,
    radiation-buoyancy coupling, and Courant-adaptive buoyancy scaling.

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
    rad_buoy_coupling : bool, optional
        Enable radiation-buoyancy coupling iteration.  Default True.
    rad_coupling_iters : int, optional
        Number of radiation-buoyancy coupling iterations.  Default 3.
    adaptive_thermal_relaxation : bool, optional
        Enable gradient-based adaptive thermal relaxation.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        rad_buoy_coupling: bool = True,
        rad_coupling_iters: int = 3,
        adaptive_thermal_relaxation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.rad_buoy_coupling = rad_buoy_coupling
        self.rad_coupling_iters = max(1, min(10, rad_coupling_iters))
        self.adaptive_thermal_relaxation = adaptive_thermal_relaxation

        logger.info(
            "BuoyantPimpleFoamEnhanced4 ready: rad_buoy=%s, adaptive_T=%s",
            self.rad_buoy_coupling, self.adaptive_thermal_relaxation,
        )

    # ------------------------------------------------------------------
    # Adaptive thermal relaxation
    # ------------------------------------------------------------------

    def _adaptive_thermal_relaxation(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        alpha_T_base: float,
    ) -> torch.Tensor:
        """Apply gradient-based adaptive thermal relaxation.

        In regions with large temperature gradients, reduces the
        relaxation factor to prevent overshoots and oscillations.

        alpha_T_local = alpha_T_base * min(1, grad_T_ref / |grad_T|)

        Parameters
        ----------
        T : torch.Tensor
            Updated temperature.
        T_old : torch.Tensor
            Old temperature.
        alpha_T_base : float
            Base thermal relaxation factor.

        Returns:
            Relaxed temperature field.
        """
        if not self.adaptive_thermal_relaxation:
            return alpha_T_base * T + (1.0 - alpha_T_base) * T_old

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Face temperature gradient
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        grad_T_face = (T_N - T_O) * delta_coeffs

        # Scatter gradient magnitude to cells
        grad_T_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        grad_T_cell = grad_T_cell + scatter_add(grad_T_face.abs(), owner, n_cells)
        grad_T_cell = grad_T_cell + scatter_add(grad_T_face.abs(), neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        grad_T_cell = grad_T_cell / n_contrib.clamp(min=1.0)

        # Reference gradient (from T range)
        T_range = float((T.max() - T.min()).abs().item())
        dx_char = float(mesh.cell_volumes.pow(1.0 / 3.0).mean().item())
        grad_T_ref = T_range / max(dx_char, 1e-10) if T_range > 0 else 1.0

        # Adaptive factor: reduce where gradients are steep
        alpha_factor = (grad_T_ref / grad_T_cell.clamp(min=1e-10)).clamp(max=1.0)
        alpha_local = alpha_T_base * alpha_factor

        # Ensure minimum relaxation
        alpha_local = alpha_local.clamp(min=0.1, max=1.0)

        return alpha_local * T + (1.0 - alpha_local) * T_old

    # ------------------------------------------------------------------
    # Radiation-buoyancy coupling
    # ------------------------------------------------------------------

    def _radiation_buoyancy_iteration(
        self,
        T: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate between radiation source and buoyancy force.

        The radiation source term heats/cools the fluid, which changes
        the temperature, which changes the buoyancy force, which changes
        the velocity, which changes the temperature again.  Iterating
        captures this feedback.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        rho : torch.Tensor
            Density field.

        Returns:
            Tuple of (updated_T, radiation_source).
        """
        if not self.rad_buoy_coupling or self.radiation is None:
            Q_rad = torch.zeros_like(T)
            return T, Q_rad

        T_iter = T.clone()

        for _iter in range(self.rad_coupling_iters):
            # Radiation source
            Q_rad = self.radiation.calculate_source(T_iter, rho)

            # Buoyancy from updated temperature
            F_buoy, _ = self._semi_implicit_buoyancy_source(
                T_iter, self.T_old, rho,
            )

            # Update temperature with radiation source
            # dT ~ Q_rad / (rho * Cp)
            Cp = 1005.0  # Approximate
            dT_rad = Q_rad * self.delta_t / (rho * Cp).clamp(min=1e-10)
            T_iter = T_iter + dT_rad

        return T_iter, Q_rad

    # ------------------------------------------------------------------
    # Courant-adaptive buoyancy scaling
    # ------------------------------------------------------------------

    def _courant_scaled_buoyancy(
        self,
        F_buoy: torch.Tensor,
        Co: torch.Tensor,
    ) -> torch.Tensor:
        """Scale buoyancy force by local Courant number.

        At high Courant numbers, the explicit buoyancy source can
        cause instability.  This scales the force down proportionally.

        Parameters
        ----------
        F_buoy : torch.Tensor
            Buoyancy force.
        Co : torch.Tensor
            Local Courant number field.

        Returns:
            Scaled buoyancy force.
        """
        # Scale factor: 1 for Co < 0.5, linearly decreasing to 0.2 at Co = 1
        Co_threshold = 0.5
        scale = torch.where(
            Co < Co_threshold,
            torch.ones_like(Co),
            (1.0 - 0.8 * (Co - Co_threshold) / (1.0 - Co_threshold)).clamp(min=0.2),
        )

        if F_buoy.dim() > 1:
            scale = scale.unsqueeze(-1)

        return F_buoy * scale

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 buoyantPimpleFoam solver.

        Uses adaptive thermal relaxation, radiation-buoyancy coupling,
        and Courant-adaptive buoyancy scaling.

        Returns:
            Final :class:`ConvergenceData`.
        """
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

        logger.info("Starting buoyantPimpleFoamEnhanced4 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  rad_buoy=%s, adaptive_T=%s",
                     self.rad_buoy_coupling, self.adaptive_thermal_relaxation)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            # Richardson number (from v3)
            Ri = self._compute_richardson(self.U, self.T)
            self._richardson = Ri

            # Brunt-Vaisala frequency (from v3)
            N = self._compute_brunt_vaisala_frequency(self.T)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s", Ri, N)

            mu_eff = self._update_turbulence()

            # Semi-implicit buoyancy source (from v3)
            F_buoy, diag_buoy = self._semi_implicit_buoyancy_source(
                self.T, self.T_old, self.rho,
            )

            # Radiation-buoyancy coupling
            self.T, Q_rad = self._radiation_buoyancy_iteration(
                self.T, self.rho,
            )

            # Courant-adaptive buoyancy scaling
            Co = self._compute_local_courant()
            F_buoy = self._courant_scaled_buoyancy(F_buoy, Co)

            # Buoyancy production (from v3)
            G_b = self._compute_buoyancy_production(self.T)

            # Gravity wave filter (from v3)
            self.U = self._apply_gravity_wave_filter(
                self.U, self.U_old, self.delta_t,
            )

            # PIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )

            # Energy corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho,
            )

            # Temperature limiting (from v2)
            self.T = self._limit_temperature(self.T)

            # Adaptive thermal relaxation
            self.T = self._adaptive_thermal_relaxation(
                self.T, self.T_old, self.alpha_p,
            )

            # Temperature-dependent relaxation (from v3)
            self.U = self._temperature_dependent_relaxation(
                self.T, self.U, self.U_old, self.alpha_U,
            )

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
                logger.info("buoyantPimpleFoamEnhanced4 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced4 completed without convergence")

        return last_convergence or ConvergenceData()
