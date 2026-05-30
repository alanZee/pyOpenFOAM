"""
sprayFoamEnhanced13 -- enhanced Lagrangian spray solver v13.

Extends :class:`SprayFoamEnhanced12` with coupling algorithm variants.

Usage::

    from pyfoam.applications.spray_foam_enhanced_13 import SprayFoamEnhanced13

    solver = SprayFoamEnhanced13("path/to/case", simplec=True, coupled=True)
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

from .spray_foam_enhanced_12 import SprayFoamEnhanced12
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SprayFoamEnhanced13"]

logger = logging.getLogger(__name__)


class SprayFoamEnhanced13(SprayFoamEnhanced12):
    """Enhanced Lagrangian spray solver v13.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    simplec : bool, optional
        Enable SIMPLEC coupling.  Default True.
    coupled : bool, optional
        Enable coupled solver.  Default True.
    """

    def __init__(self, case_path: Union[str, Path], simplec: bool = True,
                 coupled: bool = True, coupled_max_iter: int = 5, **kwargs) -> None:
        super().__init__(case_path, **kwargs)
        self.simplec = simplec
        self.coupled = coupled
        self.coupled_max_iter = max(1, min(20, coupled_max_iter))
        logger.info("SprayFoamEnhanced13 ready: simplec=%s, coupled=%s", self.simplec, self.coupled)

    def _simplec_correct(self, U: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
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

    def _coupled_solve(self, U, p, dt):
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
        return U_iter, p_iter

    def run(self) -> dict:
        device = get_device()
        dtype = get_default_dtype()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting sprayFoamEnhanced13 run")
        self._write_fields(self.start_time)
        time_loop.mark_written()
        converged = False

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            self.p = self._extended_spray_pressure_correct(self.p, self.U)
            self.p = self._over_relaxed_stabilise(self.p)

            if self.coupled:
                self.U, self.p = self._coupled_solve(self.U, self.p, self.delta_t)

            self.U = self._simplec_correct(self.U, self.p)
            self.p = self._aitken_correct(self.p, p_prev)

            residuals = {"U": float(self.U.abs().mean().item())}
            converged = convergence.update(step + 1, residuals)
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()
            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return {"converged": converged}
