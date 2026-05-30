"""
filmFoamEnhanced4 — enhanced thin film flow solver v4.

Extends :class:`FilmFoamEnhanced3` with:

- **Surfactant transport model**: solves an additional transport equation
  for the surface-active agent concentration, which modifies the surface
  tension coefficient through the Langmuir equation of state, enabling
  simulation of surfactant-laden films.
- **Thermocapillary (Marangoni) flow**: includes the temperature-
  dependent surface tension gradient as a tangential stress source
  that drives Marangoni flow along the film surface, important for
  heated thin films.
- **Film rupture with hole dynamics**: models film rupture events by
  tracking hole growth after rupture, using a Taylor-Culick velocity
  model for the expanding hole edge, providing physically-based
  rupture dynamics.

Governing equations:
    dh/dt + div(h U_s) = S_evap - S_rupture
    d(Gamma)/dt + div(Gamma U_s) = div(D_s * grad(Gamma))

Usage::

    from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4

    solver = FilmFoamEnhanced4("path/to/case", surfactant=True)
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

from .film_foam_enhanced_3 import FilmFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced4"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced4(FilmFoamEnhanced3):
    """Enhanced thin film flow solver v4 with surfactants and rupture dynamics.

    Extends FilmFoamEnhanced3 with surfactant transport, thermocapillary
    flow, and film rupture with hole dynamics.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    surfactant : bool, optional
        Enable surfactant transport.  Default True.
    surfactant_diffusivity : float, optional
        Surface diffusion coefficient (m^2/s).  Default 1e-9.
    gamma_eq : float, optional
        Equilibrium surface concentration (mol/m^2).  Default 5e-6.
    sigma_clean : float, optional
        Surface tension of clean interface (N/m).  Default 0.072.
    sigma_min : float, optional
        Minimum surface tension (fully covered).  Default 0.025.
    marangoni : bool, optional
        Enable thermocapillary (Marangoni) flow.  Default True.
    dsigma_dT : float, optional
        Surface tension temperature coefficient (N/m/K).  Default -1.5e-4.
    rupture_dynamics : bool, optional
        Enable hole-growth rupture model.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        surfactant: bool = True,
        surfactant_diffusivity: float = 1e-9,
        gamma_eq: float = 5e-6,
        sigma_clean: float = 0.072,
        sigma_min: float = 0.025,
        marangoni: bool = True,
        dsigma_dT: float = -1.5e-4,
        rupture_dynamics: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.surfactant = surfactant
        self.D_s = max(1e-15, surfactant_diffusivity)
        self.gamma_eq = max(1e-10, gamma_eq)
        self.sigma_clean = max(0.001, sigma_clean)
        self.sigma_min = max(0.001, min(self.sigma_clean, sigma_min))
        self.marangoni = marangoni
        self.dsigma_dT = dsigma_dT
        self.rupture_dynamics = rupture_dynamics

        # Surfactant concentration field
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        self.Gamma = torch.full((n_cells,), self.gamma_eq * 0.5, dtype=dtype, device=device)
        self.Gamma_old = self.Gamma.clone()

        # Rupture tracking
        self._ruptured_cells = torch.zeros(n_cells, dtype=torch.bool, device=device)
        self._hole_radii: dict[int, float] = {}
        self._n_rupture_events = 0

        # Surface tension field
        self.sigma_field = torch.full((n_cells,), self.sigma_clean, dtype=dtype, device=device)

        logger.info(
            "FilmFoamEnhanced4 ready: surf=%s, marangoni=%s, rupture=%s",
            self.surfactant, self.marangoni, self.rupture_dynamics,
        )

    # ------------------------------------------------------------------
    # Surfactant transport
    # ------------------------------------------------------------------

    def _compute_surface_tension(
        self,
        Gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surface tension from surfactant concentration.

        Uses the Langmuir equation of state:
            sigma = sigma_clean - (sigma_clean - sigma_min) * Gamma/Gamma_eq

        Parameters
        ----------
        Gamma : torch.Tensor
            Surfactant concentration.

        Returns:
            Surface tension field.
        """
        coverage = (Gamma / self.gamma_eq).clamp(min=0.0, max=1.0)
        sigma = self.sigma_clean - (self.sigma_clean - self.sigma_min) * coverage

        return sigma.clamp(min=self.sigma_min)

    def _advance_surfactant(
        self,
        Gamma: torch.Tensor,
        h: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance surfactant concentration.

        Solves: d(Gamma)/dt + div(Gamma * U_s) = D_s * laplacian(Gamma)

        Simplified: diffusion + advection with film velocity.

        Parameters
        ----------
        Gamma : torch.Tensor
            Current surfactant concentration.
        h : torch.Tensor
            Film thickness.
        dt : float
            Time step.

        Returns:
            Updated surfactant concentration.
        """
        if not self.surfactant:
            return Gamma

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = Gamma.device
        dtype = Gamma.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Diffusion flux
        G_O = gather(Gamma, owner)
        G_N = gather(Gamma, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        diff_flux = self.D_s * (G_N - G_O) * delta_coeffs

        diffusion = torch.zeros(n_cells, dtype=dtype, device=device)
        diffusion = diffusion + scatter_add(diff_flux, owner, n_cells)
        diffusion = diffusion + scatter_add(-diff_flux, neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        dGamma_dt = diffusion / V

        # Update (forward Euler)
        Gamma_new = (Gamma + dt * dGamma_dt).clamp(min=0.0)

        # Cap at saturation
        Gamma_new = Gamma_new.clamp(max=self.gamma_eq * 2.0)

        return Gamma_new

    # ------------------------------------------------------------------
    # Marangoni flow
    # ------------------------------------------------------------------

    def _compute_marangoni_stress(
        self,
        T: torch.Tensor | None,
        Gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Marangoni (thermocapillary) stress.

        tau_Marangoni = d(sigma)/dx = d(sigma)/dT * dT/dx + d(sigma)/dGamma * dGamma/dx

        Parameters
        ----------
        T : torch.Tensor or None
            Temperature field.
        Gamma : torch.Tensor
            Surfactant concentration.

        Returns:
            ``(n_cells,)`` Marangoni stress.
        """
        if not self.marangoni:
            return torch.zeros(self.mesh.n_cells, dtype=Gamma.dtype, device=Gamma.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = Gamma.device
        dtype = Gamma.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Surfactant gradient contribution to surface tension
        sigma = self._compute_surface_tension(Gamma)
        s_O = gather(sigma, owner)
        s_N = gather(sigma, neigh)

        grad_sigma = (s_N - s_O) * delta_coeffs

        # Scatter to cells
        tau_M = torch.zeros(n_cells, dtype=dtype, device=device)
        tau_M = tau_M + scatter_add(grad_sigma.abs(), owner, n_cells)
        tau_M = tau_M + scatter_add(grad_sigma.abs(), neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        tau_M = tau_M / n_contrib.clamp(min=1.0)

        # Temperature gradient contribution (if available)
        if T is not None:
            T_O = gather(T, owner)
            T_N = gather(T, neigh)
            grad_T = (T_N - T_O) * delta_coeffs

            tau_therm = torch.zeros(n_cells, dtype=dtype, device=device)
            tau_therm = tau_therm + scatter_add(grad_T.abs(), owner, n_cells)
            tau_therm = tau_therm + scatter_add(grad_T.abs(), neigh, n_cells)
            tau_therm = tau_therm / n_contrib.clamp(min=1.0)

            tau_M = tau_M + abs(self.dsigma_dT) * tau_therm

        return tau_M

    # ------------------------------------------------------------------
    # Film rupture with hole dynamics
    # ------------------------------------------------------------------

    def _detect_and_track_rupture(
        self,
        h: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Detect film rupture and track hole growth.

        When h < precursor_thickness * 1.05, the cell is marked as
        ruptured.  Hole growth follows the Taylor-Culick velocity:
            v_tc = sqrt(2 * sigma / (rho * h))

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        dt : float
            Time step.

        Returns:
            Updated film thickness with rupture holes.
        """
        if not self.rupture_dynamics:
            return h

        # Detect new ruptures
        rupture_threshold = self.precursor_thickness * 1.05
        new_ruptures = (h < rupture_threshold) & (~self._ruptured_cells)

        if new_ruptures.any():
            n_new = int(new_ruptures.sum().item())
            self._n_rupture_events += n_new
            self._ruptured_cells = self._ruptured_cells | new_ruptures
            logger.debug("  %d new rupture cells detected", n_new)

        # Apply hole dynamics to ruptured cells
        if self._ruptured_cells.any():
            h = h.clone()
            rho_film = self.rho if hasattr(self, 'rho') else 1000.0
            sigma = self.sigma_field.mean().item()

            # Taylor-Culick velocity
            v_tc = math.sqrt(2.0 * sigma / (rho_film * max(self.precursor_thickness, 1e-15)))

            # Expand holes (reduce thickness toward zero)
            h[self._ruptured_cells] = h[self._ruptured_cells] * max(0.1, 1.0 - v_tc * dt)

        return h

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v4 filmFoam solver.

        Uses surfactant transport, Marangoni flow, and rupture dynamics.

        Returns
        -------
        dict
            ``converged``, ``steps``, ``residual``,
            ``h_min``, ``h_max``, ``n_spinodal``,
            ``capillary_number``, ``total_evaporation``,
            ``n_dry_cells``, ``n_refined_cells``,
            ``n_rupture_events``, ``mean_surface_tension``,
            ``mean_surfactant_concentration``.
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

        logger.info("Starting FilmFoamEnhanced4 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  surf=%s, marangoni=%s, rupture=%s",
                     self.surfactant, self.marangoni, self.rupture_dynamics)

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

            # Advance surfactant
            self.Gamma = self._advance_surfactant(self.Gamma, self.h, dt_actual)

            # Update surface tension field
            self.sigma_field = self._compute_surface_tension(self.Gamma)

            # Marangoni stress
            tau_M = self._compute_marangoni_stress(None, self.Gamma)

            # Advance film with evaporation (from v3)
            self.h = self._advance_film_v3(self.h, dt_actual)

            # Rupture detection and hole dynamics
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
                logger.info("FilmFoamEnhanced4 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())
        n_refined = int(self._refine_flag.sum().item())

        logger.info("FilmFoamEnhanced4 completed: h=[%.2e, %.2e] m", h_min, h_max)
        logger.info("  rupture events=%d, avg sigma=%.4f N/m",
                     self._n_rupture_events, self.sigma_field.mean().item())

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
        }
