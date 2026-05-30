"""
sprayFoamEnhanced3 — enhanced Lagrangian spray solver v3.

Extends :class:`SprayFoamEnhanced2` with:

- **Improved evaporation model**: uses a convection-corrected d^2-law
  evaporation model with the Ranz-Marshall correlation for the Sherwood
  number, accounting for the effect of relative velocity on the mass
  transfer rate.
- **Dynamic drag model**: implements a dynamic drag coefficient that
  accounts for droplet deformation (via the TAB distortion parameter),
  providing higher drag for distorted droplets approaching breakup.
- **Parcel population balance**: tracks the size distribution of
  parcels using a simple method-of-moments approach, allowing
  statistical analysis of the spray characteristics.

Based on OpenFOAM's sprayFoam with enhanced physics.

Usage::

    from pyfoam.applications.spray_foam_enhanced_3 import SprayFoamEnhanced3

    solver = SprayFoamEnhanced3("path/to/case", dynamic_drag=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .spray_foam_enhanced_2 import SprayFoamEnhanced2, TABBreakupModel
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SprayFoamEnhanced3", "ParcelMomentTracker"]

logger = logging.getLogger(__name__)


@dataclass
class ParcelMomentTracker:
    """Tracks size distribution moments of the parcel population.

    Uses a simple 2-moment method (number density and mean size):
        M_0 = sum(n_i)          (total number density)
        M_1 = sum(n_i * d_i)    (mean size moment)
        M_3 = sum(n_i * d_i^3)  (volume moment)

    Attributes
    ----------
    M_0 : float
        Total number density.
    M_1 : float
        First size moment.
    M_3 : float
        Third size moment (related to volume/mass).
    """
    M_0: float = 0.0
    M_1: float = 0.0
    M_3: float = 0.0

    @property
    def mean_diameter(self) -> float:
        """Sauter mean diameter (d32)."""
        if self.M_3 > 0 and self.M_0 > 0:
            return self.M_3 / self.M_1 if self.M_1 > 0 else 0.0
        return 0.0

    @property
    def total_particles(self) -> float:
        """Total particle count."""
        return self.M_0

    def update(self, diameters: list[float]) -> None:
        """Update moments from current diameter list."""
        self.M_0 = len(diameters)
        self.M_1 = sum(d for d in diameters)
        self.M_3 = sum(d ** 3 for d in diameters)


class SprayFoamEnhanced3(SprayFoamEnhanced2):
    """Enhanced Lagrangian spray solver v3.

    Extends SprayFoamEnhanced2 with convection-corrected evaporation,
    dynamic drag, and parcel population balance tracking.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    dynamic_drag : bool, optional
        Enable dynamic drag coefficient.  Default True.
    C_d_base : float, optional
        Base drag coefficient.  Default 0.44.
    population_balance : bool, optional
        Enable parcel population balance tracking.  Default True.
    **kwargs
        Passed to base class.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        dynamic_drag: bool = True,
        C_d_base: float = 0.44,
        population_balance: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.dynamic_drag = dynamic_drag
        self.C_d_base = max(0.1, min(2.0, C_d_base))
        self.population_balance = population_balance

        # Population balance tracker
        self.moment_tracker = ParcelMomentTracker()

        logger.info(
            "SprayFoamEnhanced3 ready: dynamic_drag=%s, pop_balance=%s",
            self.dynamic_drag, self.population_balance,
        )

    # ------------------------------------------------------------------
    # Convection-corrected evaporation
    # ------------------------------------------------------------------

    def _compute_evaporation_rate_enhanced(
        self,
        d: float,
        T_droplet: float,
        T_gas: float,
        v_rel: float,
        rho_l: float,
    ) -> float:
        """Compute evaporation rate with Ranz-Marshall correction.

        Uses the d^2-law with Sherwood number correction:
            Sh = 2 + 0.6 * Re^0.5 * Sc^(1/3)
            dm/dt = -pi * D_ab * Sh * rho_g * ln(1 + B_M) * d

        Parameters
        ----------
        d : float
            Droplet diameter.
        T_droplet : float
            Droplet temperature.
        T_gas : float
            Gas temperature.
        v_rel : float
            Relative velocity.
        rho_l : float
            Liquid density.

        Returns:
            Mass evaporation rate (kg/s).
        """
        # Gas properties
        nu_gas = 1.5e-5  # Kinematic viscosity
        D_ab = 2.5e-5  # Binary diffusion coefficient
        Sc = nu_gas / max(D_ab, 1e-30)
        rho_g = self.rho_gas if hasattr(self, 'rho_gas') else 1.2

        # Reynolds and Sherwood numbers
        Re = v_rel * d / max(nu_gas, 1e-30)
        Sh = 2.0 + 0.6 * math.sqrt(max(Re, 0.0)) * Sc ** (1.0 / 3.0)

        # Spalding mass transfer number (simplified)
        B_M = max((T_gas - T_droplet) / max(T_droplet, 1.0), 0.0)

        # Evaporation rate
        dm_dt = -math.pi * D_ab * Sh * rho_g * math.log(1.0 + B_M) * d

        return max(dm_dt, 0.0)  # Only evaporation (negative mass change)

    # ------------------------------------------------------------------
    # Dynamic drag model
    # ------------------------------------------------------------------

    def _dynamic_drag_coefficient(
        self,
        d: float,
        Re: float,
        distortion: float,
    ) -> float:
        """Compute dynamic drag coefficient accounting for deformation.

        The drag coefficient increases with droplet distortion:
            C_d = C_d_sphere * (1 + 2.632 * y)

        where y is the TAB distortion parameter.

        Parameters
        ----------
        d : float
            Droplet diameter.
        Re : float
            Reynolds number.
        distortion : float
            TAB distortion parameter (0 = sphere, 1 = breakup).

        Returns:
            Drag coefficient.
        """
        if not self.dynamic_drag:
            return self.C_d_base

        # Schiller-Naumann correlation for sphere
        if Re < 1000:
            C_d_sphere = (24.0 / max(Re, 1e-10)) * (
                1.0 + 0.15 * Re ** 0.687
            )
        else:
            C_d_sphere = 0.44

        # Deformation correction
        C_d = C_d_sphere * (1.0 + 2.632 * max(min(distortion, 1.0), 0.0))

        return min(C_d, 5.0)  # Cap for stability

    # ------------------------------------------------------------------
    # Population balance update
    # ------------------------------------------------------------------

    def _update_population_balance(self) -> None:
        """Update parcel population balance statistics."""
        if not self.population_balance:
            return

        diameters = [
            p.diameter for p in self.cloud.particles
            if p.alive and p.diameter > 0
        ]
        self.moment_tracker.update(diameters)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run SprayFoamEnhanced3 solver.

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

        logger.info("Starting SprayFoamEnhanced3 run")
        logger.info("  dynamic_drag=%s, pop_balance=%s",
                     self.dynamic_drag, self.population_balance)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Enhanced cloud advance (from v2)
            self._advance_cloud_enhanced_v2(self.delta_t)

            # Coupling sources
            S_mom = self.coupling.momentum_source()
            S_heat = self.coupling.heat_source(self.T)
            S_mass, _ = self.coupling.mass_source(self.delta_t)

            # Turbulence source
            self._S_k = self._compute_turbulence_source(self.delta_t)

            # Solve gas phase
            self.U, self.p, self.T, self.phi, conv = (
                self._pimple_spray_iteration(S_mom, S_heat, S_mass)
            )
            last_convergence = conv

            # Update cloud conditions
            self._update_cloud_fluid_conditions()

            # Population balance
            if step % 5 == 0:
                self._update_population_balance()
                if step % 20 == 0 and self.population_balance:
                    logger.info(
                        "  d32=%.2e, N_p=%d",
                        self.moment_tracker.mean_diameter,
                        int(self.moment_tracker.total_particles),
                    )

            # Convergence
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
                logger.info("SprayFoamEnhanced3 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("SprayFoamEnhanced3 completed")
        logger.info("  d32=%.2e, N_p=%d",
                     self.moment_tracker.mean_diameter,
                     int(self.moment_tracker.total_particles))
        logger.info("  coalescence=%d, bounce=%d, fragmentation=%d",
                     self._n_coalescence, self._n_bounce, self._n_fragmentation)

        return last_convergence or ConvergenceData()
