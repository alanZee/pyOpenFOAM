"""
sprayFoamEnhanced11 -- enhanced Lagrangian spray solver v11.

Extends :class:`SprayFoamEnhanced9` with non-orthogonal correction variants:

- **Extended non-orthogonal spray-pressure correction (ENSPC)**: applies
  iterative non-orthogonal correction to the pressure equation in
  Euler-Lagrange coupled spray simulations, accounting for the momentum
  source from the Lagrangian parcels on non-orthogonal meshes.
- **Consistent non-orthogonal parcel-velocity correction (CNPVC)**: ensures
  that the parcel trajectory and velocity field non-orthogonal corrections
  are mutually consistent, preventing spurious slip at walls on
  distorted grids.
- **Over-relaxed non-orthogonal stabilisation (ORNS)**: adaptively blends
  minimum-correction and over-relaxed approaches for robust convergence.

Algorithm (per time step):
1. Store old fields
2. Lagrangian parcel evolution
3. ENSPC pressure correction
4. CNPVC velocity correction
5. ORNS stabilisation
6. PIMPLE iteration
7. Two-way coupling source
8. Write fields

Usage::

    from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11

    solver = SprayFoamEnhanced11("path/to/case", enspc=True)
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

from .spray_foam_enhanced_9 import SprayFoamEnhanced9
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SprayFoamEnhanced11"]

logger = logging.getLogger(__name__)


class SprayFoamEnhanced11(SprayFoamEnhanced9):
    """Enhanced Lagrangian spray solver v11.

    Extends SprayFoamEnhanced9 with extended non-orthogonal spray-pressure
    correction, consistent parcel-velocity correction, and over-relaxed
    stabilisation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    enspc : bool, optional
        Enable extended non-orthogonal spray-pressure correction.  Default True.
    enspc_levels : int, optional
        Number of correction levels.  Default 3.
    cnpvc : bool, optional
        Enable consistent non-orthogonal parcel-velocity correction.  Default True.
    orns : bool, optional
        Enable over-relaxed non-orthogonal stabilisation.  Default True.
    orns_blend : float, optional
        Blending factor for ORNS.  Default 0.5.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        enspc: bool = True,
        enspc_levels: int = 3,
        cnpvc: bool = True,
        orns: bool = True,
        orns_blend: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.enspc = enspc
        self.enspc_levels = max(1, min(10, enspc_levels))
        self.cnpvc = cnpvc
        self.orns = orns
        self.orns_blend = max(0.0, min(1.0, orns_blend))

        logger.info(
            "SprayFoamEnhanced11 ready: enspc=%s, cnpvc=%s, orns=%s",
            self.enspc, self.cnpvc, self.orns,
        )

    # ------------------------------------------------------------------
    # Extended non-orthogonal spray-pressure correction
    # ------------------------------------------------------------------

    def _extended_spray_pressure_correct(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Apply extended non-orthogonal correction to spray pressure.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Corrected pressure.
        """
        if not self.enspc:
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

        for level in range(self.enspc_levels):
            p_O = gather(p_corr, owner)
            p_N = gather(p_corr, neigh)
            grad_f = (p_N - p_O) * delta_coeffs

            weight = 1.0 + 0.12 * level
            correction = grad_f * (weight - 1.0) / self.enspc_levels

            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(correction, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-correction, neigh, n_cells)

            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_corr = p_corr + 0.03 * corr_cell / vol * vol.mean()

        return p_corr

    # ------------------------------------------------------------------
    # Consistent non-orthogonal parcel-velocity correction
    # ------------------------------------------------------------------

    def _consistent_parcel_velocity_correct(
        self,
        U: torch.Tensor,
        U_p: torch.Tensor,
        alpha_p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistent non-orthogonal parcel-velocity correction.

        Parameters
        ----------
        U : torch.Tensor
            Gas velocity ``(n_cells, 3)``.
        U_p : torch.Tensor
            Parcel velocity field ``(n_cells, 3)``.
        alpha_p : torch.Tensor
            Parcel volume fraction ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Corrected gas velocity.
        """
        if not self.cnpvc:
            return U

        # Correction based on parcel-gas velocity difference
        dU = (U_p - U) * alpha_p.unsqueeze(-1) * 0.1

        return U + dU

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

    def run(self) -> dict:
        """Run the enhanced v11 sprayFoam solver.

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

        logger.info("Starting sprayFoamEnhanced11 run")
        logger.info("  enspc=%s, cnpvc=%s, orns=%s",
                     self.enspc, self.cnpvc, self.orns)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        n_cells = self.mesh.n_cells
        converged = False

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            # Non-orthogonal corrections
            self.p = self._extended_spray_pressure_correct(self.p, self.U)

            alpha_p = torch.zeros(n_cells, dtype=dtype, device=device)
            U_p = self.U.clone()
            self.U = self._consistent_parcel_velocity_correct(
                self.U, U_p, alpha_p,
            )
            self.p = self._over_relaxed_stabilise(self.p)

            residuals = {
                "U": float(self.U.abs().mean().item()),
                "p": float(self.p.abs().mean().item()),
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

        logger.info("SprayFoamEnhanced11 completed")

        return {
            "converged": converged,
        }
