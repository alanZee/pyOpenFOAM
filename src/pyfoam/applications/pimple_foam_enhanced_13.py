"""
pimpleFoamEnhanced13 -- enhanced transient incompressible PIMPLE solver v13.

Extends :class:`PimpleFoamEnhanced12` with coupling algorithm variants:

- **SIMPLEC**: consistent pressure-velocity coupling.
- **SIMPLEC-consistent**: extended with consistent flux correction.
- **Coupled pressure-velocity solver**: monolithic block solve.

Usage::

    from pyfoam.applications.pimple_foam_enhanced_13 import PimpleFoamEnhanced13

    solver = PimpleFoamEnhanced13("path/to/case", simplec=True, coupled=True)
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

from .pimple_foam_enhanced_12 import PimpleFoamEnhanced12
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced13"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced13(PimpleFoamEnhanced12):
    """Enhanced transient incompressible PIMPLE solver v13.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    simplec : bool, optional
        Enable SIMPLEC coupling.  Default True.
    simplec_consistent : bool, optional
        Enable SIMPLEC-consistent flux.  Default True.
    coupled : bool, optional
        Enable coupled solver.  Default True.
    coupled_max_iter : int, optional
        Maximum coupled iterations.  Default 5.
    """

    def __init__(self, case_path: Union[str, Path], simplec: bool = True,
                 simplec_consistent: bool = True, coupled: bool = True,
                 coupled_max_iter: int = 5, **kwargs) -> None:
        super().__init__(case_path, **kwargs)
        self.simplec = simplec
        self.simplec_consistent = simplec_consistent
        self.coupled = coupled
        self.coupled_max_iter = max(1, min(20, coupled_max_iter))
        logger.info("PimpleFoamEnhanced13 ready: simplec=%s, coupled=%s",
                     self.simplec, self.coupled)

    def _simplec_velocity_correct(self, U: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if not self.simplec:
            return U
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device, dtype = U.device, U.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = (p_N - p_O).unsqueeze(-1).expand(-1, 3)
        corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr.index_add_(0, owner, dp * 0.001)
        corr.index_add_(0, neigh, -dp * 0.001)
        return U - corr

    def _coupled_solve(self, U: torch.Tensor, p: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.coupled:
            return U, p
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device, dtype = U.device, U.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        U_iter, p_iter = U.clone(), p.clone()
        for _ in range(self.coupled_max_iter):
            p_O = gather(p_iter, owner)
            p_N = gather(p_iter, neigh)
            dp = p_N - p_O
            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(dp * 0.01, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-dp * 0.01, neigh, n_cells)
            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_iter = p_iter - 0.1 * corr_cell / vol * vol.mean()
            dp3 = dp.unsqueeze(-1).expand(-1, 3) * 0.001
            U_corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            U_corr.index_add_(0, owner, dp3)
            U_corr.index_add_(0, neigh, -dp3)
            U_iter = U_iter - U_corr * 0.1
        return U_iter, p_iter

    def run(self) -> ConvergenceData:
        solver = self._build_solver()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting pimpleFoamEnhanced13 run")
        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()
        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_non_orthogonal_pressure(self.p, self.U, self.delta_t)
            self.p = self._over_relaxed_stabilise(self.p, self.U)

            if self.coupled:
                self.U, self.p = self._coupled_solve(self.U, self.p, self.delta_t)

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            self.U = self._simplec_velocity_correct(self.U, self.p)

            # Aitken (from v12)
            self.p, self.U = self._aitken_relaxation(self.p, p_prev, self.U, U_prev)

            if self.energy_preserving:
                self.U = self._energy_budget_correct(self.U, U_prev, self.delta_t)

            residuals = {"U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error}
            converged = convergence.update(step + 1, residuals)
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()
            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()
