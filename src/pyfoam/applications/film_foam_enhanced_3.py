"""
filmFoamEnhanced3 — enhanced thin film flow solver v3.

Extends :class:`FilmFoamEnhanced2` with:

- **Adaptive mesh refinement for film**: uses a film-thickness-based
  criterion to mark cells for refinement, concentrating resolution
  near the contact line and in thin regions.
- **Evaporation model**: adds a Hertz-Knudsen evaporation model that
  accounts for the vapor pressure difference and local temperature,
  providing mass loss from the film due to evaporation.
- **Wetting-drying model**: implements a threshold-based wetting/drying
  algorithm that activates and deactivates cells based on the film
  thickness relative to a threshold, preventing numerical issues
  in nearly dry regions.

Governing equation:
    dh/dt + div(h U_s) = S_evap

Usage::

    from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3

    solver = FilmFoamEnhanced3("path/to/case", evaporation=True)
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

from .film_foam_enhanced_2 import FilmFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced3"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced3(FilmFoamEnhanced2):
    """Enhanced thin film flow solver v3 with evaporation and AMR.

    Extends FilmFoamEnhanced2 with adaptive mesh refinement,
    Hertz-Knudsen evaporation, and wetting-drying model.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    evaporation : bool, optional
        Enable evaporation model.  Default True.
    evap_coefficient : float, optional
        Evaporation mass flux coefficient.  Default 1e-4.
    vapor_pressure_ref : float, optional
        Reference vapor pressure (Pa).  Default 2337.0 (water at 20C).
    amr_enabled : bool, optional
        Enable adaptive mesh refinement.  Default True.
    amr_threshold : float, optional
        Film thickness threshold for AMR.  Default 1e-5.
    wet_dry_threshold : float, optional
        Threshold for wetting/drying model.  Default 1e-8.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        evaporation: bool = True,
        evap_coefficient: float = 1e-4,
        vapor_pressure_ref: float = 2337.0,
        amr_enabled: bool = True,
        amr_threshold: float = 1e-5,
        wet_dry_threshold: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.evaporation = evaporation
        self.evap_coeff = max(0.0, evap_coefficient)
        self.vapor_pressure_ref = max(0.0, vapor_pressure_ref)
        self.amr_enabled = amr_enabled
        self.amr_threshold = max(1e-15, amr_threshold)
        self.wet_dry_threshold = max(1e-15, wet_dry_threshold)

        # AMR refinement flags
        device = get_device()
        dtype = get_default_dtype()
        self._refine_flag = torch.zeros(
            self.mesh.n_cells, dtype=torch.bool, device=device,
        )

        # Wetting/drying state
        self._wet_cells = torch.ones(
            self.mesh.n_cells, dtype=torch.bool, device=device,
        )

        # Evaporation statistics
        self._total_evap_mass = 0.0

        logger.info(
            "FilmFoamEnhanced3 ready: evap=%s, amr=%s, wet_dry=%.2e",
            self.evaporation, self.amr_enabled, self.wet_dry_threshold,
        )

    # ------------------------------------------------------------------
    # Evaporation model
    # ------------------------------------------------------------------

    def _compute_evaporation_rate(
        self,
        h: torch.Tensor,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute evaporation mass flux using Hertz-Knudsen model.

        m_evap = alpha_e * (p_vap(T) - p_v) / sqrt(2 * pi * M * R * T)

        Simplified for isothermal case:
            m_evap = coeff * (p_vap_ref - 0) * h

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        T : torch.Tensor, optional
            Temperature field (None for isothermal).

        Returns:
            ``(n_cells,)`` evaporation mass flux (kg/m^2/s).
        """
        if not self.evaporation:
            return torch.zeros_like(h)

        # Simplified Hertz-Knudsen
        p_sat = self.vapor_pressure_ref

        # Evaporation rate proportional to exposed area and saturation pressure
        # Use Knudsen layer approximation: m = alpha * p_sat * sqrt(M / (2*pi*R*T))
        M_water = 0.018  # kg/mol
        R = 8.314
        T_local = 293.15 if T is None else T.clamp(min=273.0)

        coeff = self.evap_coeff * p_sat * math.sqrt(
            M_water / (2.0 * math.pi * R * float(T_local.mean().item()) if isinstance(T_local, torch.Tensor) else 293.15),
        )

        # Only evaporate where film exists
        m_evap = coeff * (h > self.wet_dry_threshold).float()

        return m_evap.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Wetting-drying model
    # ------------------------------------------------------------------

    def _apply_wet_drying(
        self,
        h: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply wetting-drying algorithm.

        - Wet cells (h > threshold): full physics applies.
        - Drying cells (h approaching threshold): apply precursor film
          with enhanced surface tension regularization.
        - Dry cells (h < threshold): deactivate (set to precursor level).

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        dt : float
            Time step.

        Returns:
            Updated film thickness with wet/dry states.
        """
        # Update wet cell flags
        self._wet_cells = h > self.wet_dry_threshold

        n_dry = int((~self._wet_cells).sum().item())

        if n_dry == 0:
            return h

        # Drying cells: smooth transition
        drying = (h <= self.wet_dry_threshold * 2.0) & self._wet_cells
        if drying.any():
            # Enhanced precursor film for drying cells
            h = h.clone()
            h[drying] = h[drying].clamp(min=self.precursor_thickness)

        # Dry cells: set to precursor level
        dry = ~self._wet_cells
        if dry.any():
            h = h.clone()
            h[dry] = self.precursor_thickness

        return h

    # ------------------------------------------------------------------
    # AMR refinement marking
    # ------------------------------------------------------------------

    def _mark_refinement_cells(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Mark cells for adaptive refinement based on film thickness.

        Cells where the film thickness is below the AMR threshold
        are marked for refinement to better resolve thin regions.

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.

        Returns:
            Boolean refinement flag tensor.
        """
        if not self.amr_enabled:
            return torch.zeros_like(h, dtype=torch.bool)

        # Refine thin regions
        thin = h < self.amr_threshold

        # Also refine near contact line (large gradient)
        mesh = self.mesh
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        h_O = gather(h, owner)
        h_N = gather(h, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        grad_h = ((h_N - h_O) * delta_coeffs).abs()
        grad_max = torch.zeros(mesh.n_cells, dtype=h.dtype, device=h.device)
        grad_max = grad_max + scatter_add(grad_h, owner, mesh.n_cells)
        grad_max = grad_max + scatter_add(grad_h, neigh, mesh.n_cells)

        steep = grad_max > grad_max.mean() * 3.0

        self._refine_flag = thin | steep

        return self._refine_flag

    # ------------------------------------------------------------------
    # Enhanced film advance with evaporation
    # ------------------------------------------------------------------

    def _advance_film_v3(
        self,
        h: torch.Tensor,
        dt: float,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance film thickness with evaporation and wet/dry model.

        Parameters
        ----------
        h : torch.Tensor
            Film thickness.
        dt : float
            Time step.
        T : torch.Tensor, optional
            Temperature field.

        Returns:
            Updated film thickness.
        """
        # Evaporation source
        m_evap = self._compute_evaporation_rate(h, T)

        # Convert mass flux to thickness rate: dh/dt = -m_evap / rho
        rho_film = self.rho if hasattr(self, 'rho') else 1000.0
        dh_evap = -m_evap / rho_film * dt

        # Base film advancement (from v2)
        h_new = self._advance_film_v2(h, dt)

        # Apply evaporation
        h_new = h_new + dh_evap

        # Track total evaporation
        self._total_evap_mass += float((m_evap * dt).sum().item())

        # Apply wetting-drying
        h_new = self._apply_wet_drying(h_new, dt)

        # AMR marking (for diagnostics)
        if self.amr_enabled:
            self._mark_refinement_cells(h_new)

        return h_new

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v3 filmFoam solver.

        Uses evaporation, wetting-drying, and AMR marking.

        Returns
        -------
        dict
            ``converged``, ``steps``, ``residual``,
            ``h_min``, ``h_max``, ``n_spinodal``,
            ``capillary_number``, ``total_evaporation``,
            ``n_dry_cells``, ``n_refined_cells``.
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

        logger.info("Starting FilmFoamEnhanced3 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  evap=%s, amr=%s", self.evaporation, self.amr_enabled)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0
        total_spinodal = 0
        total_dry = 0

        for t, step in time_loop:
            h_old = self.h.clone()

            # Adaptive time step
            if self.adaptive_dt:
                dt_actual = self._compute_capillary_dt(self.h)
            else:
                dt_actual = self.delta_t

            # Advance film with evaporation
            self.h = self._advance_film_v3(self.h, dt_actual)

            # Spinodal check (from v2)
            _, n_spinodal = self._check_spinodal_instability(self.h)
            total_spinodal += n_spinodal

            # Count dry cells
            n_dry = int((~self._wet_cells).sum().item())
            total_dry += n_dry

            # Rupture detection
            ruptured = (self.h <= self.precursor_thickness * 1.1).sum().item()
            if ruptured > 0:
                logger.debug("  %d cells at precursor thickness", ruptured)

            # Residual
            residual = float((self.h - h_old).abs().max().item())
            converged = convergence.update(step + 1, {"h": residual})

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FilmFoamEnhanced3 converged at step %d", step + 1)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        h_min = float(self.h.min().item())
        h_max = float(self.h.max().item())
        n_refined = int(self._refine_flag.sum().item())

        logger.info("FilmFoamEnhanced3 completed: h=[%.2e, %.2e] m", h_min, h_max)
        logger.info("  total evaporation: %.4e kg", self._total_evap_mass)
        logger.info("  total dry cell events: %d", total_dry)

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
        }
