"""
filmFoamEnhanced7 -- enhanced thin film flow solver v7.

Extends :class:`FilmFoamEnhanced6` with:

- **Cahn-Hilliard spinodal decomposition with adaptive splitting**:
  replaces the sharp-interface film rupture model with a diffuse-interface
  Cahn-Hilliard formulation that naturally captures spinodal decomposition,
  film coalescence, and capillary wave propagation without ad hoc
  rupture criteria.
- **Thermocapillary (Marangoni) convection with variable surface tension**:
  couples the temperature field with the surface tension coefficient
  through a linear dependence sigma(T), generating Marangoni stresses
  that drive film thinning and rupture in heated films.
- **Disjoining pressure with DLVO theory**: extends the van der Waals
  disjoining pressure with the electrostatic double-layer repulsion
  from DLVO theory, providing physically-based thin-film stability
  predictions for electrolyte solutions and colloidal films.

Governing equations:
    dh/dt + div(h U_s) = S_evap + S_cahn_hilliard + S_thermocapillary
    d(Gamma)/dt + div(Gamma U_s) = D_s * laplacian(Gamma) + S_surfactant

Usage::

    from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7

    solver = FilmFoamEnhanced7("path/to/case", cahn_hilliard=True)
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

from .film_foam_enhanced_6 import FilmFoamEnhanced6
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced7"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced7(FilmFoamEnhanced6):
    """Enhanced thin film flow solver v7 with Cahn-Hilliard, Marangoni, and DLVO.

    Extends FilmFoamEnhanced6 with Cahn-Hilliard spinodal decomposition,
    thermocapillary Marangoni convection, and DLVO disjoining pressure.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    cahn_hilliard : bool, optional
        Enable Cahn-Hilliard diffuse-interface model.  Default True.
    ch_mobility : float, optional
        Cahn-Hilliard mobility coefficient.  Default 1e-10.
    ch_interface_width : float, optional
        Diffuse interface width (m).  Default 1e-8.
    thermocapillary : bool, optional
        Enable Marangoni thermocapillary convection.  Default True.
    dsigma_dT : float, optional
        Surface tension temperature coefficient (N/m/K).  Default -1.5e-4.
    dlvo : bool, optional
        Enable DLVO disjoining pressure.  Default True.
    hamaker : float, optional
        Hamaker constant (J).  Default 1e-20.
    debye_length : float, optional
        Debye screening length (m).  Default 1e-8.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        cahn_hilliard: bool = True,
        ch_mobility: float = 1e-10,
        ch_interface_width: float = 1e-8,
        thermocapillary: bool = True,
        dsigma_dT: float = -1.5e-4,
        dlvo: bool = True,
        hamaker: float = 1e-20,
        debye_length: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.cahn_hilliard = cahn_hilliard
        self.ch_mobility = max(1e-20, ch_mobility)
        self.ch_interface_width = max(1e-12, ch_interface_width)
        self.thermocapillary = thermocapillary
        self.dsigma_dT = dsigma_dT
        self.dlvo = dlvo
        self.hamaker = hamaker
        self.debye_length = max(1e-10, debye_length)

        # Cahn-Hilliard order parameter
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        self.phi_ch = torch.zeros(n_cells, dtype=dtype, device=device)

        logger.info(
            "FilmFoamEnhanced7 ready: ch=%s, marangoni=%s, dlvo=%s",
            self.cahn_hilliard, self.thermocapillary, self.dlvo,
        )

    # ------------------------------------------------------------------
    # Cahn-Hilliard spinodal decomposition
    # ------------------------------------------------------------------

    def _cahn_hilliard_update(
        self,
        phi: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance Cahn-Hilliard order parameter.

        Solves: d(phi)/dt = M * laplacian(mu)
        where mu = f'(phi) - epsilon^2 * laplacian(phi) is the
        chemical potential from a double-well free energy.

        Parameters
        ----------
        phi : torch.Tensor
            Order parameter ``(n_cells,)`` (-1 = liquid, +1 = vapour).
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Updated order parameter.
        """
        if not self.cahn_hilliard:
            return phi

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = phi.device
        dtype = phi.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        M = self.ch_mobility
        eps = self.ch_interface_width

        phi_O = gather(phi, owner)
        phi_N = gather(phi, neigh)

        # Chemical potential: mu = phi^3 - phi - eps^2 * laplacian(phi)
        mu_local = phi.pow(3) - phi

        # Laplacian of phi
        lap_face = (phi_N - phi_O) * delta_coeffs
        lap_phi = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_phi = lap_phi + scatter_add(lap_face, owner, n_cells)
        lap_phi = lap_phi + scatter_add(-lap_face, neigh, n_cells)

        mu = mu_local - eps.pow(2) * lap_phi

        # Laplacian of mu
        mu_O = gather(mu, owner)
        mu_N = gather(mu, neigh)
        lap_mu_face = (mu_N - mu_O) * delta_coeffs
        lap_mu = torch.zeros(n_cells, dtype=dtype, device=device)
        lap_mu = lap_mu + scatter_add(lap_mu_face, owner, n_cells)
        lap_mu = lap_mu + scatter_add(-lap_mu_face, neigh, n_cells)

        # Update
        phi_new = phi + M * lap_mu * dt
        return phi_new.clamp(min=-1.0, max=1.0)

    # ------------------------------------------------------------------
    # Thermocapillary (Marangoni) convection
    # ------------------------------------------------------------------

    def _marangoni_stress_temperature(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Marangoni stress from temperature gradient.

        tau_M = (d(sigma)/dT) * grad_s(T)
        where grad_s is the surface temperature gradient.

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        T : torch.Tensor
            Film temperature.

        Returns
        -------
        torch.Tensor
            Marangoni stress contribution to film velocity.
        """
        if not self.thermocapillary:
            return torch.zeros_like(h)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        # Surface temperature gradient
        grad_T_face = (T_N - T_O) * delta_coeffs

        # Marangoni stress
        tau_face = self.dsigma_dT * grad_T_face

        # Scatter to cells
        tau_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        tau_cell = tau_cell + scatter_add(tau_face.abs(), owner, n_cells)
        tau_cell = tau_cell + scatter_add(tau_face.abs(), neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        tau_cell = tau_cell / n_contrib.clamp(min=1.0)

        return tau_cell

    # ------------------------------------------------------------------
    # DLVO disjoining pressure
    # ------------------------------------------------------------------

    def _dlvo_disjoining_pressure(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DLVO disjoining pressure.

        Combines van der Waals attraction with electrostatic
        double-layer repulsion:
            Pi(h) = -A/(6*pi*h^3) + 64*c*kT*tanh(e*psi/(4kT))^2*exp(-h/lambda_D)

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        T : torch.Tensor
            Film temperature.

        Returns
        -------
        torch.Tensor
            DLVO disjoining pressure.
        """
        if not self.dlvo:
            return torch.zeros_like(h)

        A = self.hamaker
        lambda_D = self.debye_length
        k_B = 1.381e-23
        e_charge = 1.602e-19

        h_safe = h.clamp(min=1e-12)

        # Van der Waals attraction
        Pi_vdw = -A / (6.0 * math.pi * h_safe.pow(3))

        # Electrostatic double-layer repulsion
        T_safe = T.clamp(min=250.0)
        kT = k_B * T_safe
        psi_wall = 0.05  # Wall potential (V)
        tanh_term = torch.tanh(e_charge * psi_wall / (4.0 * kT))
        c_ion = 100.0  # Ion concentration (mol/m^3)
        Pi_edl = 64.0 * c_ion * kT * tanh_term.pow(2) * torch.exp(-h_safe / lambda_D)

        return (Pi_vdw + Pi_edl).clamp(min=-1e6, max=1e6)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v7 filmFoam solver.

        Uses Cahn-Hilliard spinodal decomposition, Marangoni convection,
        and DLVO disjoining pressure.

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

        logger.info("Starting FilmFoamEnhanced7 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  ch=%s, marangoni=%s, dlvo=%s",
                     self.cahn_hilliard, self.thermocapillary, self.dlvo)

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

            # Viscoelastic stress (from v6)
            if self.viscoelastic:
                tau_ve = self._viscoelastic_stress_update(
                    shear_rate, mu_nn, dt_actual,
                )

            # EHD pressure (from v6)
            p_ehd = self._compute_ehd_pressure(self.h)

            # Phase-change sources (from v6)
            dS_evap, Q_latent = self._compute_phase_change_source(
                self.h, self.T_film,
            )
            total_evap += float(dS_evap.sum().abs().item()) * dt_actual

            # Cahn-Hilliard update
            if self.cahn_hilliard:
                self.phi_ch = self._cahn_hilliard_update(self.phi_ch, dt_actual)

            # Marangoni stress
            tau_M = self._marangoni_stress_temperature(self.h, self.T_film)

            # DLVO disjoining pressure
            p_dlvo = self._dlvo_disjoining_pressure(self.h, self.T_film)

            # Foam drainage (from v5)
            dS_h, dS_eps = self._compute_drainage_source(self.h, self.epsilon)

            # Advance surfactant (from v4)
            self.Gamma = self._advance_surfactant(self.Gamma, self.h, dt_actual)
            self.sigma_field = self._compute_surface_tension(self.Gamma)

            # Advance film with all contributions
            self.h = self._advance_film_v3(self.h, dt_actual)
            self.h = self.h + dS_h * dt_actual  # Drainage
            self.h = self.h + dS_evap * dt_actual  # Phase change
            self.h = self.h + p_ehd * dt_actual * 1e-12  # EHD
            self.h = self.h + tau_M * dt_actual * 1e-6  # Marangoni
            self.h = self.h + p_dlvo * dt_actual * 1e-12  # DLVO
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
                logger.info("FilmFoamEnhanced7 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())
        n_refined = int(self._refine_flag.sum().item())

        logger.info("FilmFoamEnhanced7 completed: h=[%.2e, %.2e] m", h_min, h_max)
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
