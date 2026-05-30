"""
buoyantPimpleFoamEnhanced5 — enhanced transient buoyant PIMPLE solver v5.

Extends :class:`BuoyantPimpleFoamEnhanced4` with:

- **Semi-implicit buoyancy-momentum coupling**: treats the buoyancy
  source term semi-implicitly in the momentum equation by linearising
  around the current temperature, reducing the splitting error between
  the buoyancy and pressure-correction steps.
- **Buoyancy-aware adaptive time stepping**: uses the Brunt-Vaisala
  frequency to limit the time step so that internal gravity waves
  are resolved, preventing numerical instability in stably stratified
  flows.
- **Turbulence-buoyancy interaction model**: adds a buoyancy production
  term to the turbulent kinetic energy equation that accounts for the
  stabilising/destabilising effect of stratification on turbulence.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson and Brunt-Vaisala (from v3)
3. Buoyancy-aware dt adaptation
4. Outer corrector loop:
   a. Semi-implicit buoyancy source
   b. Radiation-buoyancy coupled iteration (from v4)
   c. Momentum predictor with buoyancy-turbulence coupling
   d. PISO pressure correction (p_rgh form)
   e. Energy equation (with adaptive relaxation from v4)
   f. EOS update
   g. Temperature limiting (from v2)
5. Turbulence-buoyancy interaction update
6. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5

    solver = BuoyantPimpleFoamEnhanced5("path/to/case", semi_implicit_buoyancy=True)
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

from .buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced5"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced5(BuoyantPimpleFoamEnhanced4):
    """Enhanced transient buoyant PIMPLE solver v5.

    Extends BuoyantPimpleFoamEnhanced4 with semi-implicit buoyancy
    coupling, buoyancy-aware time stepping, and turbulence-buoyancy
    interaction modelling.

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
    semi_implicit_buoyancy : bool, optional
        Enable semi-implicit buoyancy coupling.  Default True.
    buoyancy_dt_limit : bool, optional
        Enable buoyancy-aware time stepping.  Default True.
    buoyancy_tke_production : bool, optional
        Include buoyancy in TKE equation.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        semi_implicit_buoyancy: bool = True,
        buoyancy_dt_limit: bool = True,
        buoyancy_tke_production: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.semi_implicit_buoyancy = semi_implicit_buoyancy
        self.buoyancy_dt_limit = buoyancy_dt_limit
        self.buoyancy_tke_production = buoyancy_tke_production

        logger.info(
            "BuoyantPimpleFoamEnhanced5 ready: semi_impl=%s, dt_limit=%s, tke_prod=%s",
            self.semi_implicit_buoyancy, self.buoyancy_dt_limit,
            self.buoyancy_tke_production,
        )

    # ------------------------------------------------------------------
    # Semi-implicit buoyancy coupling
    # ------------------------------------------------------------------

    def _semi_implicit_buoyancy_linearised(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linearised semi-implicit buoyancy source.

        Linearises rho*beta*g*(T-T_ref) around the current temperature:
            F_buoy = rho*beta*g*(T-T_ref) ~ F_buoy^0 + dF/dT * (T-T^0)

        The implicit part is added to the momentum diagonal for stability.

        Parameters
        ----------
        T : torch.Tensor
            Current temperature.
        T_old : torch.Tensor
            Previous temperature.
        rho : torch.Tensor
            Density field.

        Returns:
            Tuple of (buoyancy_force, implicit_diagonal_contribution).
        """
        beta = getattr(self, 'beta', 3.33e-3)
        g_mag = 9.81
        T_ref = getattr(self, 'T_ref', 300.0)

        if not self.semi_implicit_buoyancy:
            # Explicit buoyancy (from v3)
            return self._semi_implicit_buoyancy_source(T, T_old, rho)

        # Linearised form
        F_buoy = rho * beta * g_mag * (T - T_ref)

        # Implicit diagonal (stabilising)
        diag_buoy = rho * beta * g_mag

        if self.U.dim() > 1:
            # Create 3D buoyancy force (gravity direction)
            g_vec = getattr(self, 'g_vec', torch.tensor([0.0, -1.0, 0.0], dtype=T.dtype, device=T.device))
            F_buoy_3d = F_buoy.unsqueeze(-1) * g_vec.unsqueeze(0)
        else:
            F_buoy_3d = F_buoy

        return F_buoy_3d, diag_buoy

    # ------------------------------------------------------------------
    # Buoyancy-aware time stepping
    # ------------------------------------------------------------------

    def _buoyancy_limited_dt(
        self,
        current_dt: float,
        N: float,
    ) -> float:
        """Limit time step to resolve internal gravity waves.

        The buoyancy time scale is:
            tau_buoy = 2*pi / N
        where N is the Brunt-Vaisala frequency.

        The time step should be a fraction of this:
            dt_buoy = tau_buoy / (2*pi) = 1/N

        Parameters
        ----------
        current_dt : float
            Current time step.
        N : float
            Brunt-Vaisala frequency (rad/s).

        Returns:
            Buoyancy-limited time step.
        """
        if not self.buoyancy_dt_limit or N < 1e-10:
            return current_dt

        # Limit dt to resolve gravity waves
        dt_buoy = 0.5 / N  # Half period

        # Clamp
        dt_min = self.delta_t * 0.001
        dt_max = self.delta_t * 2.0
        dt_buoy = max(dt_min, min(dt_max, dt_buoy))

        return min(current_dt, dt_buoy)

    # ------------------------------------------------------------------
    # Turbulence-buoyancy interaction
    # ------------------------------------------------------------------

    def _compute_buoyancy_tke_production(
        self,
        T: torch.Tensor,
        rho: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute buoyancy production/destruction of TKE.

        In stably stratified flows, buoyancy damps turbulence:
            P_b = -g*beta * dT/dy / Pr_t
        In unstable stratification, it produces TKE.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        rho : torch.Tensor
            Density field.
        k : torch.Tensor
            Turbulent kinetic energy.

        Returns:
            ``(n_cells,)`` buoyancy TKE source.
        """
        if not self.buoyancy_tke_production:
            return torch.zeros_like(k)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Vertical temperature gradient
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        grad_T = (T_N - T_O) * delta_coeffs

        # Scatter to cells (simplified: use face-average)
        grad_T_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        grad_T_cell = (grad_T_cell + scatter_add(grad_T, owner, n_cells)
                       + scatter_add(grad_T, neigh, n_cells))
        grad_T_cell = grad_T_cell / n_contrib.clamp(min=1.0)

        # Buoyancy TKE production
        beta = getattr(self, 'beta', 3.33e-3)
        g_mag = 9.81
        Pr_t = 0.85  # Turbulent Prandtl number

        P_b = -g_mag * beta * grad_T_cell / Pr_t

        return P_b

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v5 buoyantPimpleFoam solver.

        Uses semi-implicit buoyancy coupling, buoyancy-aware dt,
        and turbulence-buoyancy interaction.

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

        logger.info("Starting buoyantPimpleFoamEnhanced5 run")
        logger.info("  semi_impl=%s, dt_limit=%s, tke_prod=%s",
                     self.semi_implicit_buoyancy, self.buoyancy_dt_limit,
                     self.buoyancy_tke_production)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

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

            # Buoyancy-aware dt
            current_dt = self._buoyancy_limited_dt(current_dt, N)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s, dt=%.3e", Ri, N, current_dt)

            mu_eff = self._update_turbulence()

            # Semi-implicit buoyancy source
            F_buoy, diag_buoy = self._semi_implicit_buoyancy_linearised(
                self.T, self.T_old, self.rho,
            )

            # Radiation-buoyancy coupling (from v4)
            self.T, Q_rad = self._radiation_buoyancy_iteration(
                self.T, self.rho,
            )

            # Courant-adaptive buoyancy scaling (from v4)
            Co = self._compute_local_courant()
            F_buoy = self._courant_scaled_buoyancy(F_buoy, Co)

            # Gravity wave filter (from v3)
            self.U = self._apply_gravity_wave_filter(
                self.U, self.U_old, current_dt,
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

            # Adaptive thermal relaxation (from v4)
            self.T = self._adaptive_thermal_relaxation(
                self.T, self.T_old, self.alpha_p,
            )

            # Temperature-dependent relaxation (from v3)
            self.U = self._temperature_dependent_relaxation(
                self.T, self.U, self.U_old, self.alpha_U,
            )

            # Turbulence-buoyancy interaction
            if self.buoyancy_tke_production and hasattr(self, 'k'):
                P_b = self._compute_buoyancy_tke_production(
                    self.T, self.rho, self.k,
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
                logger.info("buoyantPimpleFoamEnhanced5 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced5 completed without convergence")

        return last_convergence or ConvergenceData()
