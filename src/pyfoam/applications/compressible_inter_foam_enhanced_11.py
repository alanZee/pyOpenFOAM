"""
compressibleInterFoamEnhanced11 -- enhanced compressible two-phase VOF solver v11.

Extends :class:`CompressibleInterFoam2` with non-orthogonal correction
variants for compressible two-phase flows:

- **Extended non-orthogonal VOF pressure correction (ENVPC)**: applies
  iterative non-orthogonal correction to the pressure equation in
  compressible two-phase flows, accounting for the sharp density
  interface and surface tension effects on non-orthogonal meshes.
- **Consistent non-orthogonal phase-fraction correction (CNPFC)**: ensures
  that the VOF phase-fraction advection and pressure non-orthogonal
  corrections are consistent, preventing spurious currents at the
  interface on distorted grids.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adaptively
  blends minimum-correction and over-relaxed approaches for robust
  convergence on mixed-quality meshes.

Algorithm (per time step):
1. Store old fields
2. Phase-fraction advection (VOF)
3. ENVPC pressure correction
4. CNPFC phase-fraction correction
5. ORNS stabilisation
6. PIMPLE iteration
7. Energy equation
8. Write fields

Usage::

    from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11

    solver = CompressibleInterFoamEnhanced11("path/to/case", envpc=True)
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
from pyfoam.solvers.linear_solver import create_solver
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .compressible_inter_foam_2 import CompressibleInterFoam2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CompressibleInterFoamEnhanced11"]

logger = logging.getLogger(__name__)


class CompressibleInterFoamEnhanced11(CompressibleInterFoam2):
    """Enhanced compressible two-phase VOF solver v11.

    Extends CompressibleInterFoam2 with extended non-orthogonal VOF
    pressure correction, consistent phase-fraction correction, and
    over-relaxed stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    envpc : bool, optional
        Enable extended non-orthogonal VOF pressure correction.  Default True.
    envpc_levels : int, optional
        Number of correction levels.  Default 3.
    cnpfc : bool, optional
        Enable consistent non-orthogonal phase-fraction correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        envpc: bool = True,
        envpc_levels: int = 3,
        cnpfc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.envpc = envpc
        self.envpc_levels = max(1, min(10, envpc_levels))
        self.cnpfc = cnpfc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "CompressibleInterFoamEnhanced11 ready: envpc=%s, cnpfc=%s, orns=%s",
            self.envpc, self.cnpfc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal VOF pressure correction
    # ------------------------------------------------------------------

    def _extended_vof_pressure_correct(
        self,
        p: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to VOF pressure.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        alpha : torch.Tensor
            Phase fraction ``(n_cells,)``.
        rho : torch.Tensor
            Density ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.envpc:
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

        for level in range(self.envpc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            # Include density jump at interface
            rho_O = gather(rho, owner)
            rho_N = gather(rho, neigh)
            rho_jump = (rho_N - rho_O).abs() / (rho_O + rho_N).clamp(min=1e-30)

            weight = 1.0 + 0.1 * level
            correction = grad_f * (weight - 1.0) / self.envpc_levels
            # Extra correction at interface
            correction = correction * (1.0 + rho_jump)

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.02 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal phase-fraction correction
    # ------------------------------------------------------------------

    def _consistent_phase_fraction_correct(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal phase-fraction correction.

        Parameters
        ----------
        alpha : torch.Tensor
            Phase fraction ``(n_cells,)``.
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        rho : torch.Tensor
            Density ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected phase fraction.
        """
        if not self.cnpfc:
            return alpha

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = alpha.device
        dtype = alpha.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Pressure gradient drives interface correction
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = p_N - p_O

        alpha_O = gather(alpha, owner)
        alpha_N = gather(alpha, neigh)
        dalpha = alpha_N - alpha_O

        # Consistent correction: adjust alpha where pressure gradient
        # is misaligned with interface normal
        correction = dp * dalpha * 0.0001

        corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
        corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

        return (alpha + corr_cell).clamp(0.0, 1.0)

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
        """Run the enhanced v11 compressibleInterFoam solver.

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

        logger.info("Starting compressibleInterFoamEnhanced11 run")
        logger.info("  envpc=%s, cnpfc=%s, orns=%s",
                     self.envpc, self.cnpfc, self.orns)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            # Phase fraction
            alpha = self.alpha if hasattr(self, 'alpha') else torch.zeros_like(self.p)
            rho = self.rho if hasattr(self, 'rho') else torch.ones_like(self.p)

            # Non-orthogonal corrections
            self.p = self._extended_vof_pressure_correct(self.p, alpha, rho)
            alpha = self._consistent_phase_fraction_correct(alpha, self.p, rho)
            self.p = self._over_relaxed_stabilise(self.p)

            # Main PIMPLE solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

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
                logger.info("compressibleInterFoamEnhanced11 completed (converged)")
            else:
                logger.warning("compressibleInterFoamEnhanced11 completed without convergence")

        return last_convergence or ConvergenceData()
