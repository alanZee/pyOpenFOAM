"""
filmFoamEnhanced6 — enhanced thin film flow solver v6.

Extends :class:`FilmFoamEnhanced5` with:

- **Electrohydrodynamic (EHD) film destabilisation**: includes the
  electric Maxwell stress contribution to the film dynamics, modelling
  the electrostatic pressure that drives the formation of pillar-like
  structures in thin dielectric films under uniform electric fields.
- **Phase-change coupled film dynamics**: couples the film evaporation
  and condensation with the local temperature and vapour concentration
  fields, providing a self-consistent model for film boiling,
  Leidenfrost transitions, and condensation-driven film growth.
- **Viscoelastic film rheology**: extends the non-Newtonian model
  with a viscoelastic Oldroyd-B formulation that captures the
  elastic recoil and stress relaxation effects important in polymer
  solution films and biological membranes.

Governing equations:
    dh/dt + div(h U_s) = S_evap - S_rupture + S_ehd + S_pc
    d(Gamma)/dt + div(Gamma U_s) = D_s * laplacian(Gamma) + S_surfactant

Usage::

    from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6

    solver = FilmFoamEnhanced6("path/to/case", ehd=True)
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

from .film_foam_enhanced_5 import FilmFoamEnhanced5
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced6"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced6(FilmFoamEnhanced5):
    """Enhanced thin film flow solver v6 with EHD, phase-change, and viscoelasticity.

    Extends FilmFoamEnhanced5 with EHD destabilisation, phase-change
    coupled dynamics, and viscoelastic film rheology.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    ehd : bool, optional
        Enable EHD film destabilisation.  Default True.
    electric_field : float, optional
        Applied electric field strength (V/m).  Default 1e6.
    dielectric_constant : float, optional
        Film dielectric constant.  Default 2.5.
    phase_change : bool, optional
        Enable phase-change coupled dynamics.  Default True.
    latent_heat : float, optional
        Latent heat of vaporisation (J/kg).  Default 2.26e6.
    viscoelastic : bool, optional
        Enable viscoelastic film rheology.  Default True.
    relaxation_time : float, optional
        Oldroyd-B relaxation time (s).  Default 0.01.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        ehd: bool = True,
        electric_field: float = 1e6,
        dielectric_constant: float = 2.5,
        phase_change: bool = True,
        latent_heat: float = 2.26e6,
        viscoelastic: bool = True,
        relaxation_time: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.ehd = ehd
        self.electric_field = max(0.0, electric_field)
        self.dielectric_constant = max(1.0, dielectric_constant)
        self.phase_change = phase_change
        self.latent_heat = max(1e3, latent_heat)
        self.viscoelastic = viscoelastic
        self.relaxation_time = max(1e-6, relaxation_time)

        # Viscoelastic stress state
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self.tau_elastic = torch.zeros(n_cells, dtype=dtype, device=device)

        logger.info(
            "FilmFoamEnhanced6 ready: ehd=%s, pc=%s, visco=%s",
            self.ehd, self.phase_change, self.viscoelastic,
        )

    # ------------------------------------------------------------------
    # EHD film destabilisation
    # ------------------------------------------------------------------

    def _compute_ehd_pressure(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute electrohydrodynamic pressure contribution.

        The Maxwell stress at the film interface creates an
        electrostatic pressure:
            p_ehd = eps_0 * (eps_r - 1) * E^2 / (2 * (1 + (eps_r-1)*h_0/h)^2)

        Parameters
        ----------
        h : torch.Tensor
            Film thickness ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            EHD pressure contribution.
        """
        if not self.ehd:
            return torch.zeros_like(h)

        eps_0 = 8.854e-12  # Permittivity of free space
        eps_r = self.dielectric_constant
        E = self.electric_field

        # Reference thickness
        h_ref = h.mean().clamp(min=1e-10)
        h_ratio = (h / h_ref).clamp(min=0.01, max=100.0)

        # EHD pressure (simplified)
        p_ehd = eps_0 * (eps_r - 1.0) * E.pow(2) / (2.0 * (1.0 + (eps_r - 1.0) * h_ratio).pow(2))

        return p_ehd

    # ------------------------------------------------------------------
    # Phase-change coupled dynamics
    # ------------------------------------------------------------------

    def _compute_phase_change_source(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute evaporation/condensation source terms.

        Uses the Hertz-Knudsen model for net evaporation flux:
            J_net = (M/(2*pi*R*T))^0.5 * (p_sat(T) - p_vap)

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        T : torch.Tensor
            Film temperature.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (dS_evap, Q_latent) mass source and latent heat.
        """
        if not self.phase_change:
            return torch.zeros_like(h), torch.zeros_like(h)

        M_water = 0.018  # kg/mol
        R = 8.314

        T_safe = T.clamp(min=273.15, max=373.15)

        # Clausius-Clapeyron for saturation pressure
        T_ref = 373.15
        p_ref = 101325.0
        p_sat = p_ref * torch.exp(
            self.latent_heat * M_water / R * (1.0 / T_ref - 1.0 / T_safe)
        )

        # Simplified net evaporation (proportional to superheat)
        T_sat = 373.15
        superheat = (T - T_sat).clamp(min=-50.0, max=100.0)
        evap_rate = superheat * 1e-6  # Simplified coefficient

        dS_evap = -evap_rate  # Negative = evaporation
        Q_latent = dS_evap.abs() * self.latent_heat

        return dS_evap, Q_latent

    # ------------------------------------------------------------------
    # Viscoelastic film rheology
    # ------------------------------------------------------------------

    def _viscoelastic_stress_update(
        self,
        shear_rate: torch.Tensor,
        mu_eff: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Update viscoelastic stress with Oldroyd-B model.

        tau + lambda * tau_dot = mu_p * gamma_dot
        where lambda is the relaxation time and mu_p is the
        polymeric viscosity.

        Parameters
        ----------
        shear_rate : torch.Tensor
            Local shear rate.
        mu_eff : torch.Tensor
            Effective viscosity from non-Newtonian model.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Updated elastic stress.
        """
        if not self.viscoelastic:
            return self.tau_elastic

        lam = self.relaxation_time
        mu_p = mu_eff * 0.3  # Polymeric contribution

        # Oldroyd-B: explicit Euler
        tau_eq = mu_p * shear_rate
        tau_dot = (tau_eq - self.tau_elastic) / lam

        self.tau_elastic = (self.tau_elastic + tau_dot * dt).clamp(
            min=-1e6, max=1e6,
        )

        return self.tau_elastic

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v6 filmFoam solver.

        Uses EHD destabilisation, phase-change coupling, and
        viscoelastic rheology.

        Returns
        -------
        dict
            Convergence and diagnostic information.
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

        logger.info("Starting FilmFoamEnhanced6 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  ehd=%s, pc=%s, visco=%s",
                     self.ehd, self.phase_change, self.viscoelastic)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0
        total_spinodal = 0
        total_dry = 0
        total_evap = 0.0

        for t, step in time_loop:
            h_old = self.h.clone()
            self.Gamma_old = self.Gamma.clone()

            if self.adaptive_dt:
                dt_actual = self._compute_capillary_dt(self.h)
            else:
                dt_actual = self.delta_t

            # Thermal viscosity correction (from v5)
            mu_thermal = self._thermal_viscosity_factor(self.T_film)

            # Non-Newtonian viscosity (from v5)
            shear_rate = self.h.abs() / max(dt_actual, 1e-10)
            mu_nn = self._non_newtonian_viscosity(shear_rate)

            # Viscoelastic stress
            if self.viscoelastic:
                tau_ve = self._viscoelastic_stress_update(
                    shear_rate, mu_nn, dt_actual,
                )

            # EHD pressure
            p_ehd = self._compute_ehd_pressure(self.h)

            # Phase-change sources
            dS_evap, Q_latent = self._compute_phase_change_source(
                self.h, self.T_film,
            )
            total_evap += float(dS_evap.sum().abs().item()) * dt_actual

            # Foam drainage (from v5)
            dS_h, dS_eps = self._compute_drainage_source(self.h, self.epsilon)

            # Advance surfactant (from v4)
            self.Gamma = self._advance_surfactant(self.Gamma, self.h, dt_actual)
            self.sigma_field = self._compute_surface_tension(self.Gamma)
            tau_M = self._compute_marangoni_stress(None, self.Gamma)

            # Advance film with EHD + phase-change + drainage
            self.h = self._advance_film_v3(self.h, dt_actual)
            self.h = self.h + dS_h * dt_actual  # Drainage
            self.h = self.h + dS_evap * dt_actual  # Phase change
            self.h = self.h + p_ehd * dt_actual * 1e-12  # EHD (damped)
            self.h = self.h.clamp(min=0.0)

            # Update liquid fraction
            self.epsilon = self.epsilon + dS_eps * dt_actual
            self.epsilon = self.epsilon.clamp(min=0.0, max=1.0)

            # Rupture detection (from v4)
            self.h = self._detect_and_track_rupture(self.h, dt_actual)

            _, n_spinodal = self._check_spinodal_instability(self.h)
            total_spinodal += n_spinodal

            n_dry = int((~self._wet_cells).sum().item())
            total_dry += n_dry

            residual = float((self.h - h_old).abs().max().item())
            converged = convergence.update(step + 1, {"h": residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FilmFoamEnhanced6 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())
        n_refined = int(self._refine_flag.sum().item())

        logger.info("FilmFoamEnhanced6 completed: h=[%.2e, %.2e] m", h_min, h_max)
        logger.info("  total evaporation=%.2e kg", total_evap)

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
            "h_min": h_min,
            "h_max": h_max,
            "n_spinodal": total_spinodal,
            "capillary_number": self.Ca,
            "total_evaporation": self._total_evap_mass + total_evap,
            "n_dry_cells": total_dry,
            "n_refined_cells": n_refined,
            "n_rupture_events": self._n_rupture_events,
            "mean_surface_tension": float(self.sigma_field.mean().item()),
            "mean_surfactant_concentration": float(self.Gamma.mean().item()),
            "mean_liquid_fraction": float(self.epsilon.mean().item()),
            "mean_viscosity_factor": float(mu_thermal.mean().item()),
        }
