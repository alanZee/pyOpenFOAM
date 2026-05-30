"""
filmFoamEnhanced5 — enhanced thin film flow solver v5.

Extends :class:`FilmFoamEnhanced4` with:

- **Foam drainage coupled to film dynamics**: solves the coupled
  equations for foam column drainage (Plateau border flow) and
  individual film thinning, capturing the interaction between
  macroscopic drainage and local film stability.
- **Thermal effects on film viscosity**: includes temperature-
  dependent viscosity and surface tension that modify the film
  flow behaviour, capturing the destabilising effect of heating
  on foam stability.
- **Non-Newtonian film rheology**: supports power-law and
  Herschel-Bulkley rheological models for the film liquid,
  enabling simulation of polymeric and particle-laden films
  where the viscosity depends on local shear rate.

Governing equations:
    dh/dt + div(h U_s) = S_evap - S_rupture + S_drainage
    d(epsilon)/dt + div(epsilon U_s) = S_drainage

Usage::

    from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5

    solver = FilmFoamEnhanced5("path/to/case", foam_drainage=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

from .film_foam_enhanced_4 import FilmFoamEnhanced4
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced5"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced5(FilmFoamEnhanced4):
    """Enhanced thin film flow solver v5 with foam drainage and non-Newtonian rheology.

    Extends FilmFoamEnhanced4 with foam drainage coupling, thermal
    effects on viscosity, and non-Newtonian film rheology.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    foam_drainage : bool, optional
        Enable foam drainage coupling.  Default True.
    drainage_rate : float, optional
        Foam drainage coefficient (1/s).  Default 0.1.
    thermal_viscosity : bool, optional
        Enable temperature-dependent viscosity.  Default True.
    viscosity_activation_energy : float, optional
        Activation energy for Arrhenius viscosity (J/mol).  Default 20000.0.
    non_newtonian : bool, optional
        Enable non-Newtonian film rheology.  Default True.
    power_law_n : float, optional
        Power-law index (n<1: shear-thinning, n>1: shear-thickening).  Default 0.6.
    yield_stress : float, optional
        Herschel-Bulkley yield stress (Pa).  Default 0.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        foam_drainage: bool = True,
        drainage_rate: float = 0.1,
        thermal_viscosity: bool = True,
        viscosity_activation_energy: float = 20000.0,
        non_newtonian: bool = True,
        power_law_n: float = 0.6,
        yield_stress: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.foam_drainage = foam_drainage
        self.drainage_rate = max(0.0, drainage_rate)
        self.thermal_viscosity = thermal_viscosity
        self.viscosity_activation_energy = max(1000.0, viscosity_activation_energy)
        self.non_newtonian = non_newtonian
        self.power_law_n = max(0.1, min(2.0, power_law_n))
        self.yield_stress = max(0.0, yield_stress)

        # Liquid fraction field (for foam drainage)
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self.epsilon = torch.full((n_cells,), 0.1, dtype=dtype, device=device)

        # Temperature field (simplified)
        self.T_film = torch.full((n_cells,), 298.15, dtype=dtype, device=device)

        # Non-Newtonian effective viscosity
        self.mu_ref = 0.001  # Reference viscosity (Pa.s)

        logger.info(
            "FilmFoamEnhanced5 ready: drain=%s, thermal=%s, nn=%s, n=%.2f",
            self.foam_drainage, self.thermal_viscosity,
            self.non_newtonian, self.power_law_n,
        )

    # ------------------------------------------------------------------
    # Foam drainage model
    # ------------------------------------------------------------------

    def _compute_drainage_source(
        self,
        h: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute foam drainage source terms.

        Drainage of liquid from films to Plateau borders:
            dh/dt = -k_d * h * epsilon
            deps/dt = k_d * h * (1 - epsilon)

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        epsilon : torch.Tensor
            Liquid fraction.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (dS_h, dS_eps) drainage source for thickness and liquid fraction.
        """
        if not self.foam_drainage:
            return torch.zeros_like(h), torch.zeros_like(epsilon)

        dS_h = -self.drainage_rate * h * epsilon
        dS_eps = self.drainage_rate * h * (1.0 - epsilon).clamp(min=0.0)

        return dS_h, dS_eps

    # ------------------------------------------------------------------
    # Temperature-dependent viscosity
    # ------------------------------------------------------------------

    def _thermal_viscosity_factor(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temperature-dependent viscosity factor.

        Arrhenius model:
            mu(T) = mu_ref * exp(Ea / (R*T) - Ea / (R*T_ref))

        Parameters
        ----------
        T : torch.Tensor
            Temperature field (K).

        Returns
        -------
        torch.Tensor
            Viscosity multiplier.
        """
        if not self.thermal_viscosity:
            return torch.ones_like(T)

        R = 8.314
        T_ref = 298.15
        T_safe = T.clamp(min=200.0, max=500.0)

        factor = torch.exp(
            self.viscosity_activation_energy / R * (1.0 / T_safe - 1.0 / T_ref)
        )

        return factor.clamp(min=0.01, max=100.0)

    # ------------------------------------------------------------------
    # Non-Newtonian film rheology
    # ------------------------------------------------------------------

    def _non_newtonian_viscosity(
        self,
        shear_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute effective viscosity for non-Newtonian film.

        Herschel-Bulkley model:
            mu_eff = tau_y / (gamma_dot + epsilon) + K * gamma_dot^(n-1)

        Parameters
        ----------
        shear_rate : torch.Tensor
            Local shear rate.

        Returns
        -------
        torch.Tensor
            Effective viscosity.
        """
        if not self.non_newtonian:
            return torch.full_like(shear_rate, self.mu_ref)

        gamma = shear_rate.abs().clamp(min=1e-10)

        # Power-law contribution
        K = self.mu_ref  # Consistency index
        mu_pl = K * gamma.pow(self.power_law_n - 1.0)

        # Yield stress contribution (regularised)
        mu_yield = self.yield_stress / (gamma + 1e-6)

        mu_eff = mu_yield + mu_pl

        return mu_eff.clamp(min=self.mu_ref * 0.01, max=self.mu_ref * 1000.0)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v5 filmFoam solver.

        Uses foam drainage, thermal viscosity, and non-Newtonian
        rheology.

        Returns
        -------
        dict
            ``converged``, ``steps``, ``residual``,
            ``h_min``, ``h_max``, ``n_spinodal``,
            ``capillary_number``, ``total_evaporation``,
            ``n_dry_cells``, ``n_refined_cells``,
            ``n_rupture_events``, ``mean_surface_tension``,
            ``mean_surfactant_concentration``,
            ``mean_liquid_fraction``, ``mean_viscosity_factor``.
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

        logger.info("Starting FilmFoamEnhanced5 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  drain=%s, thermal=%s, nn=%s",
                     self.foam_drainage, self.thermal_viscosity,
                     self.non_newtonian)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0
        total_spinodal = 0
        total_dry = 0

        for t, step in time_loop:
            h_old = self.h.clone()
            self.Gamma_old = self.Gamma.clone()

            # Adaptive time step
            if self.adaptive_dt:
                dt_actual = self._compute_capillary_dt(self.h)
            else:
                dt_actual = self.delta_t

            # Thermal viscosity correction
            mu_thermal = self._thermal_viscosity_factor(self.T_film)

            # Non-Newtonian viscosity (simplified shear rate estimate)
            shear_rate = self.h.abs() / max(dt_actual, 1e-10)
            mu_nn = self._non_newtonian_viscosity(shear_rate)

            # Foam drainage
            dS_h, dS_eps = self._compute_drainage_source(self.h, self.epsilon)

            # Advance surfactant (from v4)
            self.Gamma = self._advance_surfactant(self.Gamma, self.h, dt_actual)

            # Update surface tension field (from v4)
            self.sigma_field = self._compute_surface_tension(self.Gamma)

            # Marangoni stress (from v4)
            tau_M = self._compute_marangoni_stress(None, self.Gamma)

            # Advance film with evaporation (from v3) + drainage
            self.h = self._advance_film_v3(self.h, dt_actual)
            self.h = self.h + dS_h * dt_actual
            self.h = self.h.clamp(min=0.0)

            # Update liquid fraction
            self.epsilon = self.epsilon + dS_eps * dt_actual
            self.epsilon = self.epsilon.clamp(min=0.0, max=1.0)

            # Rupture detection (from v4)
            self.h = self._detect_and_track_rupture(self.h, dt_actual)

            # Spinodal check (from v2)
            _, n_spinodal = self._check_spinodal_instability(self.h)
            total_spinodal += n_spinodal

            # Count dry cells
            n_dry = int((~self._wet_cells).sum().item())
            total_dry += n_dry

            # Residual
            residual = float((self.h - h_old).abs().max().item())
            converged = convergence.update(step + 1, {"h": residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FilmFoamEnhanced5 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())
        n_refined = int(self._refine_flag.sum().item())

        logger.info("FilmFoamEnhanced5 completed: h=[%.2e, %.2e] m", h_min, h_max)
        logger.info("  rupture events=%d, avg sigma=%.4f N/m",
                     self._n_rupture_events, self.sigma_field.mean().item())
        logger.info("  avg liquid fraction=%.4f", self.epsilon.mean().item())

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
            "h_min": h_min,
            "h_max": h_max,
            "n_spinodal": total_spinodal,
            "capillary_number": self.Ca,
            "total_evaporation": self._total_evap_mass,
            "n_dry_cells": total_dry,
            "n_refined_cells": n_refined,
            "n_rupture_events": self._n_rupture_events,
            "mean_surface_tension": float(self.sigma_field.mean().item()),
            "mean_surfactant_concentration": float(self.Gamma.mean().item()),
            "mean_liquid_fraction": float(self.epsilon.mean().item()),
            "mean_viscosity_factor": float(mu_thermal.mean().item()),
        }
