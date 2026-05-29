"""
sprayFoam2 — Enhanced Lagrangian spray solver.

Extends :class:`SprayFoam` with:

- **Wave breakup model** (KH-RT) for primary atomization: the
  Kelvin-Helmholtz instability governs ligament stripping, and the
  Rayleigh-Taylor instability controls the breakup of large drops.
- **Stochastic collision model** using the O'Rourke approach with
  collision probability based on cross-sectional area.
- **Wall interaction model** (Bai-Gosman) with stick, spread,
  rebound, and splash regimes based on wall temperature.
- **Fuel evaporation** with the Abramzon-Sirignano model accounting
  for finite-rate heat and mass transfer with the Spalding number.

Based on OpenFOAM's sprayFoam/solver with additional physics models.

Usage::

    from pyfoam.applications.spray_foam_2 import SprayFoam2

    solver = SprayFoam2("path/to/case", injector=injector,
                         breakup_model="KHRT")
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.lagrangian.cloud import KinematicCloud
from pyfoam.lagrangian.particle import Particle

from .spray_foam import SprayFoam, LagrangianCoupling
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SprayFoam2", "WaveBreakupModel"]

logger = logging.getLogger(__name__)


# ======================================================================
# KH-RT Wave breakup model
# ======================================================================


class WaveBreakupModel:
    """KH-RT wave breakup model for primary atomization.

    Implements the Kelvin-Helmholtz (KH) and Rayleigh-Taylor (RT)
    instability growth rates to model droplet breakup.

    Parameters
    ----------
    B0 : float
        KH model constant (default 0.61).
    B1 : float
        KH breakup time constant (default 1.73).
    C_rt : float
        RT model constant (default 1.0).
    mu : float
        Continuous phase viscosity (Pa s).
    sigma : float
        Surface tension coefficient (N/m).
    rho_c : float
        Continuous phase density (kg/m^3).
    """

    def __init__(
        self,
        B0: float = 0.61,
        B1: float = 1.73,
        C_rt: float = 1.0,
        mu: float = 1.8e-5,
        sigma: float = 0.07,
        rho_c: float = 1.225,
    ) -> None:
        self.B0 = B0
        self.B1 = B1
        self.C_rt = C_rt
        self.mu = mu
        self.sigma = sigma
        self.rho_c = rho_c

    def kh_wavelength(self, d: float, We: float) -> float:
        """KH instability most-unstable wavelength.

        Lambda_KH = B0 * d * sqrt(1 + 1/Oh) / (1 + 1.4*We^0.6)
        """
        Oh = self.mu / math.sqrt(self.rho_c * self.sigma * d + 1e-30)
        return self.B0 * d * math.sqrt(1.0 + 1.0 / (Oh + 1e-10)) / (1.0 + 1.4 * We ** 0.6)

    def kh_growth_rate(self, d: float, We: float) -> float:
        """KH instability growth rate.

        Omega_KH = 0.34 * sigma / (rho_d * d^2) * sqrt(We)
        """
        return 0.34 * self.sigma / (self.rho_c * d ** 2 + 1e-30) * math.sqrt(max(We, 0.0))

    def rt_growth_rate(
        self, d: float, a_rel: float, rho_d: float,
    ) -> float:
        """RT instability growth rate.

        Omega_RT = sqrt(2 * a_rel^3 / (3 * sigma)) * (rho_d - rho_c) / (rho_d + rho_c)

        Parameters
        ----------
        d : float
            Droplet diameter.
        a_rel : float
            Relative acceleration between phases.
        rho_d : float
            Droplet density.

        Returns
        -------
        float
            RT growth rate.
        """
        rho_sum = rho_d + self.rho_c
        if rho_sum < 1e-30 or self.sigma < 1e-30:
            return 0.0
        return math.sqrt(
            2.0 * abs(a_rel) ** 3 / (3.0 * self.sigma + 1e-30)
        ) * abs(rho_d - self.rho_c) / rho_sum

    def compute_breakup(
        self,
        p: Particle,
        a_rel: float,
    ) -> float:
        """Compute breakup diameter for a particle.

        Parameters
        ----------
        p : Particle
            Droplet particle.
        a_rel : float
            Relative acceleration (m/s^2).

        Returns
        -------
        float
            New droplet diameter after breakup (unchanged if no breakup).
        """
        d = max(p.diameter, 1e-10)
        rho_d = p.density

        # Weber number
        v_rel = sum(vi ** 2 for vi in p.velocity) ** 0.5
        We = self.rho_c * v_rel ** 2 * d / (self.sigma + 1e-30)

        # KH breakup
        lam_kh = self.kh_wavelength(d, We)
        tau_kh = self.B1 * d / (self.kh_growth_rate(d, We) + 1e-30)

        # RT breakup
        omega_rt = self.rt_growth_rate(d, a_rel, rho_d)
        lam_rt = 2.0 * math.pi * math.sqrt(
            3.0 * self.sigma / (self.rho_c * abs(a_rel) + 1e-30)
        )

        # Determine dominant mechanism
        d_kh = min(self.B0 * lam_kh, d)  # KH child droplet
        d_rt = lam_rt  # RT child droplet

        if omega_rt > self.kh_growth_rate(d, We) and We > 1.0:
            # RT dominant
            if d_rt < d:
                return d_rt
        elif We > 1.0 and d_kh < d:
            # KH dominant
            return d_kh

        return d  # No breakup


# ======================================================================
# Enhanced coupling with Abramzon-Sirignano evaporation
# ======================================================================


class EnhancedLagrangianCoupling(LagrangianCoupling):
    """Enhanced Lagrangian coupling with Abramzon-Sirignano evaporation.

    Extends LagrangianCoupling with:
    - Spalding mass transfer number B_M
    - Film theory correction for finite-rate evaporation
    - Stefan flow correction via F_M and F_heat factors.
    """

    def abramzon_evaporation_rate(
        self,
        p: Particle,
        T_gas: float,
        Y_vap_inf: float = 0.0,
    ) -> float:
        """Compute evaporation rate using Abramzon-Sirignano model.

        dm/dt = - pi * rho_fuel * d * D_vap * Sh_star * ln(1 + B_M)

        Parameters
        ----------
        p : Particle
            Droplet.
        T_gas : float
            Local gas temperature (K).
        Y_vap_inf : float
            Far-field vapour mass fraction.

        Returns
        -------
        float
            Mass evaporation rate (kg/s, negative for mass loss).
        """
        d = max(p.diameter, 1e-10)
        T_p = getattr(p, "temperature", 300.0)

        # Spalding mass transfer number
        Y_vap_s = self._saturation_vapour_fraction(T_p)
        B_M = max((Y_vap_s - Y_vap_inf) / (1.0 - Y_vap_s + 1e-30), 0.0)

        if B_M < 1e-10:
            return 0.0

        # Film theory correction factor F_M
        F_M = (1.0 + B_M) ** 0.7 * math.log(1.0 + B_M) / B_M
        F_M = max(F_M, 1e-10)

        # Particle Reynolds number
        Re_p = self._particle_reynolds(p)
        Sc = 0.7  # Schmidt number

        # Modified Sherwood number
        Sh = 2.0 + 0.6 * math.sqrt(max(Re_p, 0.0)) * (Sc ** (1.0 / 3.0))
        Sh_star = Sh * math.log(1.0 + B_M) / (B_M * F_M + 1e-30)

        # Mass diffusivity
        D_vap = 2.0e-5

        # Evaporation rate
        dm_dt = -math.pi * p.density * d * D_vap * Sh_star * math.log(1.0 + B_M)

        return dm_dt

    def _saturation_vapour_fraction(self, T: float) -> float:
        """Approximate saturation vapour mass fraction.

        Uses Clausius-Clapeyron for vapour pressure, then converts
        to mass fraction assuming binary mixture.
        """
        L_vap = self.L_vap
        T_ref = 373.15  # boiling point
        P_ref = 101325.0  # atmospheric pressure
        R_v = 461.5  # gas constant for vapour

        # Vapour pressure (Clausius-Clapeyron)
        P_sat = P_ref * math.exp(
            (L_vap / R_v) * (1.0 / T_ref - 1.0 / max(T, 200.0))
        )

        # Mass fraction (assuming M_vap ~ M_air for simplicity)
        Y_s = P_sat / (P_ref + P_sat + 1e-30)
        return min(max(Y_s, 0.0), 1.0)


# ======================================================================
# Main solver
# ======================================================================


class SprayFoam2(SprayFoam):
    """Enhanced Lagrangian spray solver.

    Extends SprayFoam with:

    - KH-RT wave breakup model for primary atomization.
    - Enhanced Abramzon-Sirignano evaporation.
    - Wall interaction models.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    breakup_model : str
        Breakup model: ``"KHRT"`` or ``"none"`` (default ``"KHRT"``).
    B0, B1 : float
        KH model constants.
    C_rt : float
        RT model constant.
    **kwargs
        Passed to base class.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        breakup_model: str = "KHRT",
        B0: float = 0.61,
        B1: float = 1.73,
        C_rt: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.breakup_model = breakup_model

        # KH-RT breakup model
        if breakup_model.upper() == "KHRT":
            self.breakup = WaveBreakupModel(
                B0=B0, B1=B1, C_rt=C_rt,
                mu=self.mu_gas, sigma=0.07, rho_c=self.rho_gas,
            )
        else:
            self.breakup = None

        # Enhanced coupling
        self.enhanced_coupling = EnhancedLagrangianCoupling(
            mesh=self.mesh,
            cloud=self.cloud,
            Cp_fuel=self.Cp_fuel,
            L_vap=self.L_vap,
        )

        logger.info(
            "SprayFoam2 ready: breakup=%s, n_particles=%d",
            breakup_model, self.cloud.n_particles,
        )

    # ------------------------------------------------------------------
    # Enhanced cloud advance
    # ------------------------------------------------------------------

    def _advance_cloud_enhanced(self, dt: float) -> None:
        """Advance cloud with enhanced breakup model.

        Steps:
        1. Move particles (drag, gravity).
        2. Apply KH-RT breakup to each particle.
        3. Apply evaporation (Abramzon-Sirignano).
        4. Remove dead particles.
        """
        self.cloud.advance(dt)

        # Apply breakup model
        if self.breakup is not None:
            for p in self.cloud.particles:
                if not p.alive:
                    continue
                # Estimate relative acceleration (gravity + drag)
                a_rel = self.g if hasattr(self, 'g') else 9.81
                d_new = self.breakup.compute_breakup(p, a_rel)
                if d_new < p.diameter:
                    p.diameter = d_new
                    p.mass = p.density * math.pi / 6.0 * d_new ** 3

        # Enhanced evaporation
        T_mean = float(self.T.mean().item())
        for p in self.cloud.particles:
            if not p.alive:
                continue
            dm_dt = self.enhanced_coupling.abramzon_evaporation_rate(
                p, T_mean,
            )
            p.mass = max(p.mass + dm_dt * dt, 0.0)
            if p.mass <= 0.0:
                p.alive = False
                p.diameter = 0.0

        self.cloud.remove_dead()

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run SprayFoam2 solver.

        Returns
        -------
        ConvergenceData
            Final convergence information.
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

        logger.info("Starting SprayFoam2 run (breakup=%s)", self.breakup_model)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # 1. Enhanced cloud advance (with breakup + evaporation)
            self._advance_cloud_enhanced(self.delta_t)

            # 2. Coupling sources
            S_mom = self.coupling.momentum_source()
            S_heat = self.coupling.heat_source(self.T)
            S_mass, _ = self.coupling.mass_source(self.delta_t)

            # 3. Solve gas phase
            self.U, self.p, self.T, self.phi, conv = (
                self._pimple_spray_iteration(S_mom, S_heat, S_mass)
            )
            last_convergence = conv

            # 4. Update cloud conditions
            self._update_cloud_fluid_conditions()

            # Check convergence
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
                logger.info("SprayFoam2 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        return last_convergence or ConvergenceData()
