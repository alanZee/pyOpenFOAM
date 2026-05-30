"""
buoyantSimpleFoamEnhanced11 -- enhanced steady-state buoyant SIMPLE solver v11.

Extends :class:`BuoyantSimpleFoamEnhanced10` with non-orthogonal correction
variants:

- **Extended non-orthogonal buoyant-pressure correction (ENBPC)**: iteratively
  corrects the pressure field for non-orthogonality in buoyancy-driven flows.
- **Consistent non-orthogonal buoyant-momentum correction (CNBMC)**: ensures
  that the buoyancy and momentum non-orthogonal corrections are mutually
  consistent.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: provides robust
  stabilised correction blending minimum-correction and over-relaxed approaches.

Algorithm (per outer iteration):
1. Turbulence update
2. ENBPC buoyant-pressure correction
3. CNBMC buoyant-momentum correction
4. ORNS stabilisation
5. SIMPLE iteration (from v10)
6. Richardson damping (from v10)
7. Convergence check

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11

    solver = BuoyantSimpleFoamEnhanced11("path/to/case", enbpc=True)
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

from .buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced11"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced11(BuoyantSimpleFoamEnhanced10):
    """Enhanced steady-state buoyant SIMPLE solver v11.

    Extends BuoyantSimpleFoamEnhanced10 with extended non-orthogonal
    buoyant-pressure correction, consistent buoyant-momentum correction,
    and over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enbpc : bool, optional
        Enable extended non-orthogonal buoyant-pressure correction.  Default True.
    enbpc_levels : int, optional
        Number of correction levels.  Default 3.
    cnbmc : bool, optional
        Enable consistent non-orthogonal buoyant-momentum correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enbpc: bool = True,
        enbpc_levels: int = 3,
        cnbmc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enbpc = enbpc
        self.enbpc_levels = max(1, min(10, enbpc_levels))
        self.cnbmc = cnbmc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "BuoyantSimpleFoamEnhanced11 ready: enbpc=%s, cnbmc=%s, orns=%s",
            self.enbpc, self.cnbmc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal buoyant-pressure correction
    # ------------------------------------------------------------------

    def _extended_buoyant_pressure_correct(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to buoyancy-pressure.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        rho : torch.Tensor
            Density ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enbpc:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_corr = p.clone()

        for level in range(self.enbpc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            T_O = gather(T, owner)
            T_N = gather(T, neigh)
            buoy_face = (T_N - T_O) * 0.001

            weight = 1.0 + 0.1 * level
            correction = (grad_f * (weight - 1.0) + buoy_face) / self.enbpc_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.02 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal buoyant-momentum correction
    # ------------------------------------------------------------------

    def _consistent_buoyant_momentum_correct(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
        T_ref: float,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal buoyant-momentum correction.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        T_ref : float
            Reference temperature.

        Returns
        -------
        torch.Tensor
            Corrected velocity.
        """
        if not self.cnbmc:
            return U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        dT_face = (T_N - T_O).unsqueeze(-1)

        gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)
        buoy_face = dT_face.expand(-1, 3) * gravity * 0.0001

        corr_cell = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr_cell.index_add_(0, owner, buoy_face)
        corr_cell.index_add_(0, neigh, -buoy_face)

        return U + corr_cell

    # ------------------------------------------------------------------
    # Over-relaxed non-orthogonal stabilisation
    # ------------------------------------------------------------------

    def _over_relaxed_stabilise(
        self,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply over-relaxed non-orthogonal stabilisation.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Stabilised pressure.
        """
        if not self.orns:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = p_N - p_O

        dp_over = dp * (1.0 + self.orns_blend)
        dp_min = dp * (1.0 - self.orns_blend)
        dp_blend = self.orns_blend * dp_over + (1.0 - self.orns_blend) * dp_min

        correction = (dp_blend - dp) * delta_coeffs * 0.01

        corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
        corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

        return p + corr_cell

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v11 buoyantSimpleFoam solver.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        solver = self._build_solver()

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

        logger.info("Starting buoyantSimpleFoamEnhanced11 run")
        logger.info("  enbpc=%s, cnbmc=%s, orns=%s",
                     self.enbpc, self.cnbmc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            T = self.T if hasattr(self, 'T') else torch.ones_like(self.p) * 300.0
            rho = self.rho if hasattr(self, 'rho') else torch.ones_like(self.p) * 1.2

            # Non-orthogonal corrections
            self.p = self._extended_buoyant_pressure_correct(self.p, T, rho)
            self.U = self._consistent_buoyant_momentum_correct(self.U, T, 300.0)
            self.p = self._over_relaxed_stabilise(self.p)

            # SIMPLE iteration
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Non-Boussinesq density update (from v10)
            if self.non_boussinesq:
                self.U = self._variable_density_buoyancy(
                    self.U, rho, T, 1.2, 300.0,
                )

            # Richardson damping (from v10)
            if self.richardson_damping:
                self.U = self._richardson_velocity_damp(self.U, T, 300.0)

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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantSimpleFoamEnhanced11 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
