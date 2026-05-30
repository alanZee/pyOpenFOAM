"""
buoyantPimpleFoamEnhanced13 -- enhanced transient buoyant PIMPLE solver v13.

Extends :class:`BuoyantPimpleFoamEnhanced12` with coupling algorithm variants.

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_13 import BuoyantPimpleFoamEnhanced13

    solver = BuoyantPimpleFoamEnhanced13("path/to/case", simplec=True, coupled=True)
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

from .buoyant_pimple_foam_enhanced_12 import BuoyantPimpleFoamEnhanced12
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced13"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced13(BuoyantPimpleFoamEnhanced12):
    """Enhanced transient buoyant PIMPLE solver v13.

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
        logger.info("BuoyantPimpleFoamEnhanced13 ready: simplec=%s, coupled=%s", self.simplec, self.coupled)

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

    def _coupled_buoyancy_solve(self, U, p, T, rho, dt):
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

    def run(self) -> ConvergenceData:
        solver = self._build_solver()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting buoyantPimpleFoamEnhanced13 run")
        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()
        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()
            T = self.T if hasattr(self, 'T') else torch.ones_like(self.p) * 300.0
            rho = self.rho if hasattr(self, 'rho') else torch.ones_like(self.p) * 1.2

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_buoyancy_pressure_correct(self.p, T, rho)
            self.p = self._over_relaxed_stabilise(self.p)

            if self.coupled:
                self.U, self.p = self._coupled_buoyancy_solve(self.U, self.p, T, rho, self.delta_t)

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            self.U = self._simplec_correct(self.U, self.p)
            self.p, self.U = self._aitken_relaxation(self.p, p_prev, self.U, U_prev)
            self.U = self._field_relax(self.U, U_prev)

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
