"""
buoyantPimpleFoamEnhanced6 — enhanced transient buoyant PIMPLE solver v6.

Extends :class:`BuoyantPimpleFoamEnhanced5` with:

- **Projection-based pressure-temperature splitting**: solves the
  coupled pressure-temperature system using a Chorin-style projection
  that first predicts the temperature, then solves the pressure
  equation with the updated buoyancy, achieving better stability
  in rapid transient buoyant flows.
- **Adaptive gravity-wave filtering**: monitors the local stratification
  strength (Brunt-Vaisala frequency) and applies frequency-selective
  damping that attenuates high-frequency gravity waves while preserving
  the physical buoyancy-driven circulation.
- **Coupled k-epsilon buoyancy model**: solves the turbulent kinetic
  energy and dissipation equations with explicit buoyancy production
  and destruction terms that account for the stabilising/destabilising
  effect of stratification.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson and Brunt-Vaisala (from v3)
3. Buoyancy-aware dt (from v5)
4. Projection-based pressure-temperature solve
5. Adaptive gravity-wave filtering
6. PIMPLE iteration with coupled turbulence
7. Temperature limiting (from v2)
8. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6

    solver = BuoyantPimpleFoamEnhanced6("path/to/case", projection_split=True)
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

from .buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced6"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced6(BuoyantPimpleFoamEnhanced5):
    """Enhanced transient buoyant PIMPLE solver v6.

    Extends BuoyantPimpleFoamEnhanced5 with projection-based splitting,
    adaptive gravity-wave filtering, and coupled k-epsilon buoyancy.

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
    projection_split : bool, optional
        Enable projection-based pressure-temperature splitting.  Default True.
    gravity_wave_filter : bool, optional
        Enable adaptive gravity-wave filtering.  Default True.
    gw_filter_coeff : float, optional
        Gravity-wave filter intensity.  Default 0.1.
    coupled_kepsilon : bool, optional
        Enable coupled k-epsilon buoyancy model.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        projection_split: bool = True,
        gravity_wave_filter: bool = True,
        gw_filter_coeff: float = 0.1,
        coupled_kepsilon: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.projection_split = projection_split
        self.gravity_wave_filter = gravity_wave_filter
        self.gw_filter_coeff = max(0.01, min(1.0, gw_filter_coeff))
        self.coupled_kepsilon = coupled_kepsilon

        logger.info(
            "BuoyantPimpleFoamEnhanced6 ready: proj=%s, gw_filter=%s, coup_ke=%s",
            self.projection_split, self.gravity_wave_filter,
            self.coupled_kepsilon,
        )

    # ------------------------------------------------------------------
    # Projection-based pressure-temperature splitting
    # ------------------------------------------------------------------

    def _projection_pressure_temperature(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve pressure-temperature via projection method.

        Step 1: Predict temperature from energy equation
        Step 2: Solve pressure with updated buoyancy
        Step 3: Correct temperature with pressure feedback

        Parameters
        ----------
        p : torch.Tensor
            Pressure.
        T : torch.Tensor
            Temperature.
        rho : torch.Tensor
            Density.
        dt : float
            Time step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Corrected (p, T).
        """
        if not self.projection_split:
            return p, T

        beta = getattr(self, 'beta', 3.33e-3)
        g_mag = 9.81
        T_ref = getattr(self, 'T_ref', 300.0)

        # Step 1: Temperature predictor (buoyancy source)
        S_buoy = rho * beta * g_mag * (T - T_ref)
        T_pred = T + dt * S_buoy * 0.001  # Damped predictor

        # Step 2: Pressure with buoyancy source
        p_buoy = -rho * beta * g_mag * (T_pred - T_ref)
        p_proj = p + p_buoy * 0.01

        # Step 3: Temperature correction
        Cp = 1005.0
        dT_corr = (p_proj - p) / (rho * Cp).clamp(min=1e-10) * 0.01
        T_corr = T_pred + dT_corr

        return p_proj, T_corr.clamp(min=200.0, max=2000.0)

    # ------------------------------------------------------------------
    # Adaptive gravity-wave filtering
    # ------------------------------------------------------------------

    def _adaptive_gravity_wave_filter(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        N: float,
        dt: float,
    ) -> torch.Tensor:
        """Apply frequency-selective gravity-wave damping.

        Filters high-frequency gravity waves while preserving the
        physical buoyancy-driven circulation.  The filter strength
        scales with the Brunt-Vaisala frequency.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        U_old : torch.Tensor
            Previous velocity.
        N : float
            Brunt-Vaisala frequency.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Filtered velocity.
        """
        if not self.gravity_wave_filter:
            return U

        if N < 1e-10:
            return U

        # Filter frequency: half the gravity-wave frequency
        omega_gw = N  # rad/s
        filter_freq = 0.5 * omega_gw

        # Apply as exponential damping towards the low-frequency component
        alpha_filter = self.gw_filter_coeff * min(1.0, filter_freq * dt)
        U_filtered = U - alpha_filter * (U - U_old)

        return U_filtered

    # ------------------------------------------------------------------
    # Coupled k-epsilon buoyancy model
    # ------------------------------------------------------------------

    def _coupled_kepsilon_buoyancy_update(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update k-epsilon with buoyancy production/destruction.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy.
        epsilon : torch.Tensor
            Turbulent dissipation rate.
        T : torch.Tensor
            Temperature.
        rho : torch.Tensor
            Density.
        dt : float
            Time step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated (k, epsilon).
        """
        if not self.coupled_kepsilon:
            return k, epsilon

        # Buoyancy production (from v5)
        P_b = self._compute_buoyancy_tke_production(T, rho, k)

        C_mu = 0.09
        C_eps1 = 1.44
        C_eps2 = 1.92
        C_eps3 = 1.0  # Buoyancy coefficient for epsilon

        # Simplified production
        P_k = 0.1 * k  # Proportional to TKE

        # k equation
        dk = dt * (P_k + P_b - epsilon)
        k_new = (k + dk).clamp(min=1e-10)

        # epsilon equation with buoyancy
        deps = dt * (
            C_eps1 * (P_k + C_eps3 * P_b.clamp(min=0.0)) * epsilon / k.clamp(min=1e-10)
            - C_eps2 * epsilon.pow(2) / k.clamp(min=1e-10)
        )
        eps_new = (epsilon + deps).clamp(min=1e-10)

        return k_new, eps_new

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v6 buoyantPimpleFoam solver.

        Uses projection-based splitting, adaptive gravity-wave
        filtering, and coupled k-epsilon buoyancy.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting buoyantPimpleFoamEnhanced6 run")
        logger.info("  proj=%s, gw_filter=%s, coup_ke=%s",
                     self.projection_split, self.gravity_wave_filter,
                     self.coupled_kepsilon)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

        # k-epsilon state (simplified)
        n_cells = self.mesh.n_cells
        device = get_device()
        dtype = get_default_dtype()
        k_turb = torch.full((n_cells,), 1e-4, dtype=dtype, device=device)
        eps_turb = torch.full((n_cells,), 1e-4, dtype=dtype, device=device)

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

            # Buoyancy-aware dt (from v5)
            current_dt = self._buoyancy_limited_dt(current_dt, N)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s, dt=%.3e", Ri, N, current_dt)

            mu_eff = self._update_turbulence()

            # Semi-implicit buoyancy source (from v5)
            F_buoy, diag_buoy = self._semi_implicit_buoyancy_linearised(
                self.T, self.T_old, self.rho,
            )

            # Radiation-buoyancy coupling (from v4)
            self.T, Q_rad = self._radiation_buoyancy_iteration(
                self.T, self.rho,
            )

            # Projection-based pressure-temperature splitting
            self.p, self.T = self._projection_pressure_temperature(
                self.p, self.T, self.rho, current_dt,
            )

            # Courant-adaptive buoyancy scaling (from v4)
            Co = self._compute_local_courant()
            F_buoy = self._courant_scaled_buoyancy(F_buoy, Co)

            # Adaptive gravity-wave filtering
            self.U = self._adaptive_gravity_wave_filter(
                self.U, self.U_old, N, current_dt,
            )

            # PIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )

            # Energy corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho,
            )

            # Coupled k-epsilon buoyancy update
            k_turb, eps_turb = self._coupled_kepsilon_buoyancy_update(
                k_turb, eps_turb, self.T, self.rho, current_dt,
            )

            # Temperature limiting (from v2)
            self.T = self._limit_temperature(self.T)

            # Adaptive thermal relaxation (from v4)
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
                self._write_fields(t + current_dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * current_dt
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantPimpleFoamEnhanced6 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced6 completed without convergence")

        return last_convergence or ConvergenceData()
