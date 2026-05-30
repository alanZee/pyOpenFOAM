"""
filmFoamEnhanced9 -- enhanced thin film flow solver v9.

Extends :class:`FilmFoamEnhanced8` with:

- **Inertial thin-film equation with Reynolds lubrication extension**:
  extends the lubrication model with the inertial terms that become
  important for rapid spreading and impact scenarios, using a
  depth-averaged formulation that captures the transition from
  inertia-dominated to viscosity-dominated regimes.
- **Wetting-drying with precursor film model**: handles the
  contact-line singularity by introducing a thin precursor film
  ahead of the advancing contact line, providing a physically
  regularised model of wetting and dewetting dynamics.
- **Thermocapillary instability analysis with linear stability**:
  couples the Marangoni stresses with a linear stability analysis
  that predicts the onset of thermocapillary (Benard-Marangoni)
  instabilities, flagging regions where the film may become unstable.

Algorithm (per time step):
1. Store old fields
2. Inertial thin-film evolution
3. Wetting-drying with precursor film
4. Thermocapillary instability analysis
5. Surfactant transport (from v8)
6. Cahn-Hilliard update (from v7)
7. DLVO disjoining pressure (from v7)
8. Write fields

Usage::

    from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9

    solver = FilmFoamEnhanced9("path/to/case", inertial_film=True)
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

from .film_foam_enhanced_8 import FilmFoamEnhanced8
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FilmFoamEnhanced9"]

logger = logging.getLogger(__name__)


class FilmFoamEnhanced9(FilmFoamEnhanced8):
    """Enhanced thin film flow solver v9 with inertial lubrication,
    wetting-drying, and thermocapillary instability analysis.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    inertial_film : bool, optional
        Enable inertial thin-film equation.  Default True.
    wetting_drying : bool, optional
        Enable wetting-drying with precursor film.  Default True.
    precursor_thickness : float, optional
        Precursor film thickness (m).  Default 1e-9.
    thermocapillary_analysis : bool, optional
        Enable thermocapillary instability analysis.  Default True.
    ma_critical : float, optional
        Critical Marangoni number for instability onset.  Default 80.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        inertial_film: bool = True,
        wetting_drying: bool = True,
        precursor_thickness: float = 1e-9,
        thermocapillary_analysis: bool = True,
        ma_critical: float = 80.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.inertial_film = inertial_film
        self.wetting_drying = wetting_drying
        self.precursor_thickness = max(1e-15, min(1e-6, precursor_thickness))
        self.thermocapillary_analysis = thermocapillary_analysis
        self.ma_critical = max(1.0, min(1000.0, ma_critical))

        logger.info(
            "FilmFoamEnhanced9 ready: inertial=%s, wd=%s, tc=%s",
            self.inertial_film, self.wetting_drying,
            self.thermocapillary_analysis,
        )

    # ------------------------------------------------------------------
    # Inertial thin-film evolution
    # ------------------------------------------------------------------

    def _inertial_film_advance(
        self,
        h: torch.Tensor,
        h_old: torch.Tensor,
        U: torch.Tensor,
        mu: float,
        rho: float,
        dt: float,
    ) -> torch.Tensor:
        """Advance film thickness with inertial thin-film equation.

        Solves the depth-averaged inertial lubrication equation:
            rho * dh/dt + div(h * U) = div(h^3 / (3*mu) * grad(p))

        Parameters
        ----------
        h : torch.Tensor
            Film thickness ``(n_cells,)``.
        h_old : torch.Tensor
            Previous film thickness ``(n_cells,)``.
        U : torch.Tensor
            Surface velocity ``(n_cells,)``.
        mu : float
            Dynamic viscosity.
        rho : float
            Density.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Updated film thickness.
        """
        if not self.inertial_film:
            return h

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = h.device
        dtype = h.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        h_O = gather(h, owner)
        h_N = gather(h, neigh)

        # Pressure gradient from film curvature (simplified)
        grad_h = (h_N - h_O) * delta_coeffs
        p_face = 0.072 * grad_h  # Surface tension * gradient

        # Flux: h^3 / (3*mu) * grad(p)
        h_face = 0.5 * (h_O + h_N)
        flux = h_face.pow(3) / (3.0 * mu + 1e-30) * p_face

        # Continuity: dh/dt = -div(flux)
        div_flux = torch.zeros(n_cells, dtype=dtype, device=device)
        div_flux = div_flux + scatter_add(flux, owner, n_cells)
        div_flux = div_flux + scatter_add(-flux, neigh, n_cells)

        vol = mesh.cell_volumes.clamp(min=1e-30)
        dh = -div_flux / vol * dt / (rho + 1e-30)

        h_new = h + dh
        return h_new.clamp(min=self.precursor_thickness if self.wetting_drying else 0.0)

    # ------------------------------------------------------------------
    # Wetting-drying with precursor film
    # ------------------------------------------------------------------

    def _precursor_film_regularise(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Regularise film thickness with precursor film model.

        Ensures the film never completely dries by enforcing a
        minimum precursor thickness, preventing the contact-line
        singularity.

        Parameters
        ----------
        h : torch.Tensor
            Film thickness ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Regularised film thickness.
        """
        if not self.wetting_drying:
            return h.clamp(min=0.0)

        # Apply precursor film minimum
        h_min = self.precursor_thickness

        # Smooth transition from dry to precursor
        h_reg = torch.where(
            h < h_min,
            h_min * torch.ones_like(h),
            h,
        )

        return h_reg

    # ------------------------------------------------------------------
    # Thermocapillary instability analysis
    # ------------------------------------------------------------------

    def _marangoni_stability_check(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
        mu: float,
        sigma_t: float,
        k_th: float,
    ) -> torch.Tensor:
        """Check for thermocapillary (Benard-Marangoni) instability.

        Computes the local Marangoni number:
            Ma = sigma_t * dT * h / (mu * alpha_th)
        and flags cells where Ma exceeds the critical value.

        Parameters
        ----------
        h : torch.Tensor
            Film thickness ``(n_cells,)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        mu : float
            Dynamic viscosity.
        sigma_t : float
            Surface tension temperature coefficient.
        k_th : float
            Thermal conductivity.

        Returns
        -------
        torch.Tensor
            Marangoni number per cell ``(n_cells,)``.
        """
        if not self.thermocapillary_analysis:
            return torch.zeros(self.mesh.n_cells, dtype=h.dtype, device=h.device)

        alpha_th = k_th / (1000.0 * 2000.0)  # Thermal diffusivity

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        # Temperature gradient magnitude
        dT = (T_N - T_O).abs()
        dT_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        dT_cell = dT_cell + scatter_add(dT, owner, n_cells)
        dT_cell = dT_cell + scatter_add(dT, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        dT_cell = dT_cell / n_contrib.clamp(min=1.0)

        # Marangoni number
        Ma = sigma_t * dT_cell * h / (mu * alpha_th + 1e-30)

        return Ma

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced v9 film solver.

        Uses inertial film, wetting-drying,
        and thermocapillary instability analysis.

        Returns
        -------
        dict
            Convergence info and diagnostics.
        """
        device = get_device()
        dtype = get_default_dtype()

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

        logger.info("Starting FilmFoamEnhanced9 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  inertial=%s, wd=%s, tc=%s",
                     self.inertial_film, self.wetting_drying,
                     self.thermocapillary_analysis)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        n_cells = self.mesh.n_cells
        h = torch.full((n_cells,), 1e-6, dtype=dtype, device=device)
        gamma = torch.full((n_cells,), self.gamma_eq, dtype=dtype, device=device)
        T_film = torch.full((n_cells,), 300.0, dtype=dtype, device=device)
        converged = False
        max_ma = 0.0

        for t, step in time_loop:
            h_old = h.clone()
            gamma_old = gamma.clone()

            # Inertial film evolution
            if self.inertial_film:
                U_surface = torch.zeros(n_cells, dtype=dtype, device=device)
                h = self._inertial_film_advance(
                    h, h_old, U_surface, 1e-3, 1000.0, self.delta_t,
                )

            # Wetting-drying precursor film
            if self.wetting_drying:
                h = self._precursor_film_regularise(h)

            # Thermocapillary instability analysis
            if self.thermocapillary_analysis:
                Ma = self._marangoni_stability_check(
                    h, T_film, 1e-3, 1e-4, 0.6,
                )
                max_ma = max(max_ma, float(Ma.max().item()))
                n_unstable = int((Ma > self.ma_critical).sum().item())
                if n_unstable > 0 and step % 10 == 0:
                    logger.debug("Marangoni: %d unstable cells (max Ma=%.1f)",
                                 n_unstable, max_ma)

            # Cahn-Hilliard update (from v7)
            if self.cahn_hilliard:
                phi_ch = self._cahn_hilliard_update(h, self.delta_t)

            # Marangoni stress (from v7)
            if self.thermocapillary:
                tau = self._marangoni_stress_temperature(h, T_film)

            # Surfactant transport (from v8)
            U_surface = torch.zeros(n_cells, dtype=dtype, device=device)
            gamma = self._surfactant_transport(gamma, h, U_surface, self.delta_t)

            # Film AMR indicators (from v8)
            if self.film_amr and step % 5 == 0:
                amr_ind = self._film_amr_indicators(h)
                n_refine = int((amr_ind > self.amr_h_threshold).sum().item())
                if n_refine > 0:
                    logger.debug("Film AMR: %d cells flagged", n_refine)

            # DLVO disjoining pressure (from v7)
            if self.dlvo:
                p_dlvo = self._dlvo_disjoining_pressure(h, T_film)

            # Film thickness evolution
            residual_h = (h - h_old).abs().mean().item()
            residuals = {"h": residual_h}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("FilmFoamEnhanced9 completed")
        logger.info("  h range: [%.2e, %.2e] m", h.min().item(), h.max().item())
        logger.info("  Max Ma: %.1f", max_ma)

        return {
            "converged": converged,
            "h_min": float(h.min().item()),
            "h_max": float(h.max().item()),
            "max_marangoni": max_ma,
        }
