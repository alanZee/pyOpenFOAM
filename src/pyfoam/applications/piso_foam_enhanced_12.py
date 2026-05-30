"""
pisoFoamEnhanced12 -- enhanced transient incompressible PISO solver v12.

Extends :class:`PisoFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: dynamically adjusts relaxation
  based on residual evolution.
- **Aitken under-relaxation**: delta-squared acceleration for PISO
  corrector iterations.
- **Field-based under-relaxation (FBUR)**: spatially-varying relaxation
  factors based on local flow features.

Usage::

    from pyfoam.applications.piso_foam_enhanced_12 import PisoFoamEnhanced12

    solver = PisoFoamEnhanced12("path/to/case", aur=True, aitken=True)
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

from .piso_foam_enhanced_11 import PisoFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PisoFoamEnhanced12"]

logger = logging.getLogger(__name__)


class PisoFoamEnhanced12(PisoFoamEnhanced11):
    """Enhanced transient incompressible PISO solver v12.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    aur : bool, optional
        Enable adaptive under-relaxation.  Default True.
    aur_growth : float, optional
        Growth factor for AUR.  Default 1.05.
    aitken : bool, optional
        Enable Aitken under-relaxation.  Default True.
    aitken_depth : int, optional
        History depth for Aitken method.  Default 3.
    fbur : bool, optional
        Enable field-based under-relaxation.  Default True.
    fbur_wall_damping : float, optional
        Wall damping factor.  Default 0.3.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        aur: bool = True,
        aur_growth: float = 1.05,
        aitken: bool = True,
        aitken_depth: int = 3,
        fbur: bool = True,
        fbur_wall_damping: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.aur = aur
        self.aur_growth = max(1.0, min(2.0, aur_growth))
        self.aitken = aitken
        self.aitken_depth = max(2, min(10, aitken_depth))
        self.fbur = fbur
        self.fbur_wall_damping = max(0.05, min(1.0, fbur_wall_damping))

        self._aitken_p_history: list[torch.Tensor] = []
        self._aur_prev_residual = float('inf')
        self._aur_alpha = 0.7

        logger.info(
            "PisoFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
            self.aur, self.aitken, self.fbur,
        )

    def _adaptive_relaxation(self, residual: float, step: int) -> float:
        if not self.aur:
            return 0.7
        if step > 0 and residual < self._aur_prev_residual:
            self._aur_alpha = min(0.95, self._aur_alpha * self.aur_growth)
        elif step > 0:
            self._aur_alpha = max(0.1, self._aur_alpha * 0.9)
        self._aur_prev_residual = residual
        return self._aur_alpha

    def _aitken_relaxation(self, p: torch.Tensor, p_old: torch.Tensor) -> torch.Tensor:
        if not self.aitken:
            return p
        self._aitken_p_history.append(p.clone())
        if len(self._aitken_p_history) > self.aitken_depth:
            self._aitken_p_history.pop(0)
        if len(self._aitken_p_history) < 2:
            return p
        dp_curr = p - self._aitken_p_history[-2]
        dp_prev = (self._aitken_p_history[-2] - self._aitken_p_history[-3]
                   if len(self._aitken_p_history) >= 3
                   else torch.zeros_like(p))
        dp_diff = dp_curr - dp_prev
        dp_sq = (dp_diff * dp_diff).sum().clamp(min=1e-30)
        if dp_sq > 1e-30:
            alpha = float(((dp_curr * dp_diff).sum() / dp_sq).clamp(0.1, 2.0).item())
        else:
            alpha = 1.0
        return p_old + alpha * (p - p_old)

    def _field_based_relaxation(self, p: torch.Tensor, p_old: torch.Tensor) -> torch.Tensor:
        if not self.fbur:
            return p
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        wall_proximity = 1.0 - (n_contrib / n_contrib.max().clamp(min=1.0))
        alpha_field = (1.0 - self.fbur_wall_damping * wall_proximity).clamp(0.1, 1.0)
        return p_old + alpha_field * (p - p_old)

    def run(self) -> ConvergenceData:
        solver = self._build_solver()
        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting pisoFoamEnhanced12 run")
        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()
        self._aitken_p_history.clear()
        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()
            residual_val = last_convergence.U_residual if last_convergence else 1.0
            self._adaptive_relaxation(residual_val, step)

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_non_orthogonal_project(self.p, self.U)
            self.U = self._consistent_rhie_chow_correct(self.U, self.p, U_prev, p_prev, self.delta_t)
            self.p = self._over_relaxed_stabilise(self.p)
            self.U, self.p = self._pressure_hodge_project(self.U, self.p)

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            self.p = self._aitken_relaxation(self.p, p_prev)
            self.p = self._field_based_relaxation(self.p, p_prev)

            residuals = {"U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error}
            converged = convergence.update(step + 1, residuals)
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()
            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()
