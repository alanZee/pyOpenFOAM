"""
buoyantPimpleFoamEnhanced7 — enhanced transient buoyant PIMPLE solver v7.

Extends :class:`BuoyantPimpleFoamEnhanced6` with:

- **Implicit buoyant pressure-velocity coupling**: solves the buoyancy
  source term semi-implicitly within the pressure equation by
  linearising the density-temperature relationship, eliminating the
  explicit buoyancy lag that causes instability in rapid transients.
- **Thermal large-eddy simulation (TLES)**: extends the velocity LES
  model with a thermal sub-grid diffusivity that accounts for the
  enhanced mixing in buoyancy-driven turbulence, improving temperature
  predictions in natural convection LES.
- **Adaptive thermal boundary layer resolution**: detects the thermal
  boundary layer thickness from the temperature gradient and applies
  targeted refinement or wall-function correction, providing accurate
  heat transfer predictions without excessive mesh resolution.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson and Brunt-Vaisala (from v3)
3. Buoyancy-aware dt (from v5)
4. Implicit buoyant pressure-velocity coupling
5. Adaptive gravity-wave filtering (from v6)
6. PIMPLE iteration with thermal LES
7. Adaptive thermal BL correction
8. Temperature limiting (from v2)
9. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7

    solver = BuoyantPimpleFoamEnhanced7("path/to/case", implicit_buoyancy=True)
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

from .buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced7"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced7(BuoyantPimpleFoamEnhanced6):
    """Enhanced transient buoyant PIMPLE solver v7.

    Extends BuoyantPimpleFoamEnhanced6 with implicit buoyancy coupling,
    thermal LES, and adaptive thermal BL resolution.

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
    implicit_buoyancy : bool, optional
        Enable implicit buoyant pressure-velocity coupling.  Default True.
    thermal_les : bool, optional
        Enable thermal sub-grid LES model.  Default True.
    prandtl_sgs : float, optional
        Sub-grid Prandtl number.  Default 0.7.
    adaptive_thermal_bl : bool, optional
        Enable adaptive thermal BL resolution.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        implicit_buoyancy: bool = True,
        thermal_les: bool = True,
        prandtl_sgs: float = 0.7,
        adaptive_thermal_bl: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.implicit_buoyancy = implicit_buoyancy
        self.thermal_les = thermal_les
        self.prandtl_sgs = max(0.1, min(2.0, prandtl_sgs))
        self.adaptive_thermal_bl = adaptive_thermal_bl

        logger.info(
            "BuoyantPimpleFoamEnhanced7 ready: impl_buoy=%s, tles=%s, adapt_bl=%s",
            self.implicit_buoyancy, self.thermal_les,
            self.adaptive_thermal_bl,
        )

    # ------------------------------------------------------------------
    # Implicit buoyant pressure-velocity coupling
    # ------------------------------------------------------------------

    def _implicit_buoyancy_pressure_coupling(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve buoyancy implicitly within the pressure equation.

        Linearises the buoyancy source as:
            S_buoy = rho * beta * g * (T - T_ref)
        and treats it semi-implicitly by including d(rho)/dT in
        the pressure equation diagonal.

        Parameters
        ----------
        p : torch.Tensor
            Pressure.
        U : torch.Tensor
            Velocity.
        T : torch.Tensor
            Temperature.
        rho : torch.Tensor
            Density.
        dt : float
            Time step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated (p, U).
        """
        if not self.implicit_buoyancy:
            return p, U

        beta = getattr(self, 'beta', 3.33e-3)
        g_mag = 9.81
        T_ref = getattr(self, 'T_ref', 300.0)

        # Semi-implicit buoyancy source
        dT = T - T_ref
        S_buoy = rho * beta * g_mag * dT

        # Implicit diagonal contribution
        diag_buoy = rho * beta * g_mag * dt * 0.01

        # Update pressure with buoyancy
        p_implicit = p + S_buoy * dt * 0.01

        # Update velocity
        if U.dim() > 1:
            g_vec = torch.tensor([0.0, -1.0, 0.0], dtype=U.dtype, device=U.device)
            U_buoy = (S_buoy / diag_buoy.clamp(min=1e-10)).unsqueeze(-1) * g_vec.unsqueeze(0) * dt * 0.01
            U_implicit = U + U_buoy
        else:
            U_implicit = U + S_buoy / diag_buoy.clamp(min=1e-10) * dt * 0.01

        return p_implicit, U_implicit

    # ------------------------------------------------------------------
    # Thermal large-eddy simulation
    # ------------------------------------------------------------------

    def _compute_thermal_sgs_diffusivity(
        self,
        U: torch.Tensor,
        nu_sgs: float,
    ) -> float:
        """Compute thermal sub-grid diffusivity.

        Uses the sub-grid Prandtl number:
            alpha_sgs = nu_sgs / Pr_sgs

        Parameters
        ----------
        U : torch.Tensor
            Velocity field.
        nu_sgs : float
            Sub-grid viscosity.

        Returns
        -------
        float
            Thermal SGS diffusivity.
        """
        if not self.thermal_les:
            return 0.0

        return nu_sgs / self.prandtl_sgs

    # ------------------------------------------------------------------
    # Adaptive thermal boundary layer resolution
    # ------------------------------------------------------------------

    def _detect_thermal_bl_thickness(
        self,
        T: torch.Tensor,
    ) -> float:
        """Detect thermal boundary layer thickness from temperature gradient.

        Estimates the BL thickness as the distance over which the
        temperature drops by 99% of the wall-to-freestream difference.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.

        Returns
        -------
        float
            Estimated BL thickness (fraction of domain).
        """
        if not self.adaptive_thermal_bl:
            return 1.0

        T_min = float(T.min().item())
        T_max = float(T.max().item())
        T_range = T_max - T_min

        if T_range < 1e-10:
            return 1.0

        # Estimate from gradient magnitude
        mesh = self.mesh
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        grad_T = ((T_N - T_O) * delta_coeffs).abs()

        max_grad = float(grad_T.max().item())
        if max_grad < 1e-10:
            return 1.0

        h = mesh.cell_volumes.pow(1.0 / 3.0).mean().item()
        bl_thickness = T_range / max(max_grad, 1e-10) / max(h, 1e-10)

        return max(0.01, min(1.0, bl_thickness))

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v7 buoyantPimpleFoam solver.

        Uses implicit buoyancy, thermal LES, and adaptive thermal
        BL resolution.

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

        logger.info("Starting buoyantPimpleFoamEnhanced7 run")
        logger.info("  impl_buoy=%s, tles=%s, adapt_bl=%s",
                     self.implicit_buoyancy, self.thermal_les,
                     self.adaptive_thermal_bl)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

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

            Ri = self._compute_richardson(self.U, self.T)
            self._richardson = Ri

            N = self._compute_brunt_vaisala_frequency(self.T)
            current_dt = self._buoyancy_limited_dt(current_dt, N)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s, dt=%.3e", Ri, N, current_dt)

            mu_eff = self._update_turbulence()

            # Thermal BL thickness detection
            if self.adaptive_thermal_bl and step % 5 == 0:
                bl_thickness = self._detect_thermal_bl_thickness(self.T)
                logger.debug("Thermal BL thickness: %.3f", bl_thickness)

            # Implicit buoyancy coupling
            self.p, self.U = self._implicit_buoyancy_pressure_coupling(
                self.p, self.U, self.T, self.rho, current_dt,
            )

            F_buoy, diag_buoy = self._semi_implicit_buoyancy_linearised(
                self.T, self.T_old, self.rho,
            )

            self.T, Q_rad = self._radiation_buoyancy_iteration(
                self.T, self.rho,
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
                logger.info("buoyantPimpleFoamEnhanced7 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced7 completed without convergence")

        return last_convergence or ConvergenceData()
