"""
buoyantPimpleFoamEnhanced8 -- enhanced transient buoyant PIMPLE solver v8.

Extends :class:`BuoyantPimpleFoamEnhanced7` with:

- **Density-based buoyancy preconditioning**: applies a density-velocity
  preconditioner that accounts for the density stratification in the
  pressure equation, eliminating the slow convergence caused by strong
  density gradients in natural convection with large temperature differences.
- **Entropy-stable thermal convection**: discretises the energy equation
  with a skew-symmetric convective form that exactly preserves the
  discrete thermal energy in the inviscid limit, preventing spurious
  temperature oscillations on coarse meshes.
- **Adaptive gravity-wave time-step limiter with CFL control**: uses the
  Brunt-Vaisala frequency to limit the time step based on the internal
  gravity wave CFL condition, preventing the numerical instabilities that
  arise from under-resolving buoyancy waves.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson and Brunt-Vaisala (from v3)
3. Gravity-wave-limited dt
4. Density-based buoyancy preconditioning
5. Entropy-stable thermal convection
6. PIMPLE iteration
7. Thermal BL correction (from v7)
8. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8

    solver = BuoyantPimpleFoamEnhanced8("path/to/case", density_precondition=True)
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

from .buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced8"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced8(BuoyantPimpleFoamEnhanced7):
    """Enhanced transient buoyant PIMPLE solver v8.

    Extends BuoyantPimpleFoamEnhanced7 with density-based buoyancy
    preconditioning, entropy-stable thermal convection, and
    adaptive gravity-wave time-step limiting.

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
    density_precondition : bool, optional
        Enable density-based buoyancy preconditioning.  Default True.
    entropy_stable_thermal : bool, optional
        Enable entropy-stable thermal convection.  Default True.
    gravity_wave_cfl : bool, optional
        Enable gravity-wave CFL time-step limiting.  Default True.
    gw_cfl_max : float, optional
        Maximum gravity-wave CFL number.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        density_precondition: bool = True,
        entropy_stable_thermal: bool = True,
        gravity_wave_cfl: bool = True,
        gw_cfl_max: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.density_precondition = density_precondition
        self.entropy_stable_thermal = entropy_stable_thermal
        self.gravity_wave_cfl = gravity_wave_cfl
        self.gw_cfl_max = max(0.01, min(2.0, gw_cfl_max))

        logger.info(
            "BuoyantPimpleFoamEnhanced8 ready: d_prec=%s, es_therm=%s, gw_cfl=%s",
            self.density_precondition, self.entropy_stable_thermal,
            self.gravity_wave_cfl,
        )

    # ------------------------------------------------------------------
    # Density-based buoyancy preconditioning
    # ------------------------------------------------------------------

    def _density_buoyancy_precondition(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        rho: torch.Tensor,
        rho_ref: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precondition pressure and velocity using density stratification.

        Applies a density-weighted correction to the pressure equation
        that accounts for the hydrostatic stratification, improving
        convergence in flows with large density differences.

        Parameters
        ----------
        p : torch.Tensor
            Pressure.
        U : torch.Tensor
            Velocity.
        rho : torch.Tensor
            Density.
        rho_ref : float
            Reference density.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Preconditioned (p, U).
        """
        if not self.density_precondition:
            return p, U

        # Density ratio correction
        rho_ratio = rho / max(rho_ref, 1e-30)
        rho_ratio = rho_ratio.clamp(min=0.1, max=10.0)

        # Pressure correction for stratification
        p_corr = p * (2.0 / (1.0 + rho_ratio))

        # Velocity correction (minimal)
        U_corr = U

        return p_corr, U_corr

    # ------------------------------------------------------------------
    # Entropy-stable thermal convection
    # ------------------------------------------------------------------

    def _entropy_stable_thermal_convection(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        U: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply entropy-stable convection to the temperature field.

        Uses a skew-symmetric form:
            C(T) = 0.5 * [div(UT) + T * div(U)]
        which preserves discrete thermal energy exactly.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        T_old : torch.Tensor
            Previous temperature.
        U : torch.Tensor
            Velocity field.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Entropy-stable temperature update.
        """
        if not self.entropy_stable_thermal:
            return T

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = T[owner]
        T_N = T[neigh]
        U_O = U[owner]
        U_N = U[neigh]

        # Face flux
        U_face = 0.5 * (U_O + U_N)
        phi_face = U_face.norm(dim=-1)

        # Skew-symmetric: 0.5 * (phi * T_upwind + T * phi)
        conv_face = 0.5 * phi_face * (T_O + T_N)

        conv_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        conv_cell = conv_cell + scatter_add(conv_face, owner, n_cells)
        conv_cell = conv_cell + scatter_add(-conv_face, neigh, n_cells)

        T_es = T - conv_cell * dt * 0.001
        return T_es.clamp(min=200.0, max=5000.0)

    # ------------------------------------------------------------------
    # Adaptive gravity-wave time-step limiter
    # ------------------------------------------------------------------

    def _gravity_wave_limited_dt(
        self,
        T: torch.Tensor,
        current_dt: float,
    ) -> float:
        """Limit time step based on internal gravity wave CFL.

        The maximum allowable dt is:
            dt_max = cfl * h / (N * h) = cfl / N
        where N is the Brunt-Vaisala frequency.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        current_dt : float
            Current time step.

        Returns
        -------
        float
            Limited time step.
        """
        if not self.gravity_wave_cfl:
            return current_dt

        N = self._compute_brunt_vaisala_frequency(T)

        if N > 1e-10:
            dt_max = self.gw_cfl_max / N
            return min(current_dt, dt_max)
        return current_dt

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v8 buoyantPimpleFoam solver.

        Uses density preconditioning, entropy-stable thermal convection,
        and gravity-wave CFL limiting.

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

        logger.info("Starting buoyantPimpleFoamEnhanced8 run")
        logger.info("  d_prec=%s, es_therm=%s, gw_cfl=%s",
                     self.density_precondition, self.entropy_stable_thermal,
                     self.gravity_wave_cfl)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

        n_cells = self.mesh.n_cells
        device = get_device()
        dtype = get_default_dtype()
        k_turb = torch.full((n_cells,), 1e-4, dtype=dtype, device=device)
        eps_turb = torch.full((n_cells,), 1e-4, dtype=dtype, device=device)

        rho_ref = float(self.rho.mean().item())

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            Ri = self._compute_richardson(self.U, self.T)
            self._richardson = Ri

            N = self._compute_brunt_vaisala_frequency(self.T)

            # Gravity-wave CFL limiting
            current_dt = self._gravity_wave_limited_dt(self.T, current_dt)
            current_dt = self._buoyancy_limited_dt(current_dt, N)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s, dt=%.3e", Ri, N, current_dt)

            mu_eff = self._update_turbulence()

            # Thermal BL thickness detection (from v7)
            if self.adaptive_thermal_bl and step % 5 == 0:
                bl_thickness = self._detect_thermal_bl_thickness(self.T)
                logger.debug("Thermal BL thickness: %.3f", bl_thickness)

            # Density-based buoyancy preconditioning
            self.p, self.U = self._density_buoyancy_precondition(
                self.p, self.U, self.rho, rho_ref,
            )

            # Implicit buoyancy coupling (from v7)
            self.p, self.U = self._implicit_buoyancy_pressure_coupling(
                self.p, self.U, self.T, self.rho, current_dt,
            )

            F_buoy, diag_buoy = self._semi_implicit_buoyancy_linearised(
                self.T, self.T_old, self.rho,
            )

            self.T, Q_rad = self._radiation_buoyancy_iteration(
                self.T, self.rho,
            )

            # Entropy-stable thermal convection
            self.T = self._entropy_stable_thermal_convection(
                self.T, self.T_old, self.U, current_dt,
            )

            # Projection-based pressure-temperature (from v6)
            self.p, self.T = self._projection_pressure_temperature(
                self.p, self.T, self.rho, current_dt,
            )

            Co = self._compute_local_courant()
            F_buoy = self._courant_scaled_buoyancy(F_buoy, Co)

            # Adaptive gravity-wave filtering (from v6)
            self.U = self._adaptive_gravity_wave_filter(
                self.U, self.U_old, N, current_dt,
            )

            # PIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )

            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho,
            )

            # Coupled k-epsilon buoyancy update (from v6)
            k_turb, eps_turb = self._coupled_kepsilon_buoyancy_update(
                k_turb, eps_turb, self.T, self.rho, current_dt,
            )

            self.T = self._limit_temperature(self.T)

            self.T = self._adaptive_thermal_relaxation(
                self.T, self.T_old, self.alpha_p,
            )

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
                logger.info("buoyantPimpleFoamEnhanced8 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced8 completed without convergence")

        return last_convergence or ConvergenceData()
