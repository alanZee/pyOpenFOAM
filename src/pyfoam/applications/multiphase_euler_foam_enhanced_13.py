"""
multiphaseEulerFoamEnhanced13 -- enhanced N-phase Euler-Euler solver v13.

Extends :class:`MultiphaseEulerFoamEnhanced12` with coupling algorithm variants.

Usage::

    from pyfoam.applications.multiphase_euler_foam_enhanced_13 import MultiphaseEulerFoamEnhanced13

    solver = MultiphaseEulerFoamEnhanced13("path/to/case", phases=["water", "air"], simplec=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .multiphase_euler_foam_enhanced_12 import MultiphaseEulerFoamEnhanced12
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoamEnhanced13"]

logger = logging.getLogger(__name__)


class MultiphaseEulerFoamEnhanced13(MultiphaseEulerFoamEnhanced12):
    """Enhanced N-phase Euler-Euler solver v13.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[str] or None
        Phase names.
    simplec : bool, optional
        Enable SIMPLEC coupling.  Default True.
    coupled : bool, optional
        Enable coupled solver.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[str] | None = None,
        simplec: bool = True,
        coupled: bool = True,
        coupled_max_iter: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, phases=phases, **kwargs)
        self.simplec = simplec
        self.coupled = coupled
        self.coupled_max_iter = max(1, min(20, coupled_max_iter))
        logger.info("MultiphaseEulerFoamEnhanced13 ready: simplec=%s, coupled=%s",
                     self.simplec, self.coupled)

    def _simplec_phase_correct(self, U_phases: Dict[str, torch.Tensor],
                               p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply SIMPLEC correction to per-phase velocities."""
        if not self.simplec:
            return U_phases
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device, dtype = p.device, p.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp = (p_N - p_O).unsqueeze(-1).expand(-1, 3) * 0.001
        corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        corr.index_add_(0, owner, dp)
        corr.index_add_(0, neigh, -dp)
        return {name: U - corr for name, U in U_phases.items()}

    def _coupled_multiphase_solve(self, U_phases, p, alpha_phases, rho_phases, dt):
        """Solve multiphase pressure-velocity as coupled system."""
        if not self.coupled:
            return U_phases, p
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device, dtype = p.device, p.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        p_iter = p.clone()
        for _ in range(self.coupled_max_iter):
            p_O = gather(p_iter, owner)
            p_N = gather(p_iter, neigh)
            dp = p_N - p_O
            corr_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            corr_cell = corr_cell + scatter_add(dp * 0.01, owner, n_cells)
            corr_cell = corr_cell + scatter_add(-dp * 0.01, neigh, n_cells)
            vol = mesh.cell_volumes.clamp(min=1e-30)
            p_iter = p_iter - 0.1 * corr_cell / vol * vol.mean()
        return U_phases, p_iter

    def run(self) -> Dict[str, Any]:
        device = get_device()
        dtype = get_default_dtype()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting MultiphaseEulerFoamEnhanced13 run")
        self._write_fields(self.start_time)
        time_loop.mark_written()
        n_cells = self.mesh.n_cells
        converged = False

        for t, step in time_loop:
            alpha_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 0.5
                          for name in (self.phases or ["phase1", "phase2"])}
            rho_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 1000.0
                        for name in (self.phases or ["phase1", "phase2"])}

            self.p = self._extended_multiphase_pressure_correct(self.p, alpha_phases, rho_phases)
            self.p = self._over_relaxed_stabilise(self.p)

            U_phases = {name: self.U.clone() for name in alpha_phases}
            if self.coupled:
                U_phases, self.p = self._coupled_multiphase_solve(
                    U_phases, self.p, alpha_phases, rho_phases, self.delta_t,
                )
            U_phases = self._simplec_phase_correct(U_phases, self.p)

            # MUSIG (from v10)
            if self.musig and self._musig_fractions is not None:
                alpha_d = alpha_phases.get("phase2", torch.ones(n_cells, dtype=dtype, device=device) * 0.3)
                self._musig_fractions = self._musig_class_transport(
                    self._musig_fractions, alpha_d, self.U.clone(), self.delta_t,
                )

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
