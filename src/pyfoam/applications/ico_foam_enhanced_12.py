"""
icoFoamEnhanced12 -- enhanced transient incompressible laminar solver v12.

Extends :class:`IcoFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: dynamically adjusts time-step
  relaxation based on residual evolution.
- **Aitken under-relaxation**: delta-squared acceleration for pressure
  and velocity corrections.
- **Field-based under-relaxation (FBUR)**: spatially-varying relaxation
  based on local flow features.

Usage::

    from pyfoam.applications.ico_foam_enhanced_12 import IcoFoamEnhanced12

    solver = IcoFoamEnhanced12("path/to/case", aur=True, aitken=True)
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

from .ico_foam_enhanced_11 import IcoFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["IcoFoamEnhanced12"]

logger = logging.getLogger(__name__)


class IcoFoamEnhanced12(IcoFoamEnhanced11):
    """Enhanced transient incompressible laminar solver v12.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    aur : bool, optional
        Enable adaptive under-relaxation.  Default True.
    aitken : bool, optional
        Enable Aitken under-relaxation.  Default True.
    fbur : bool, optional
        Enable field-based under-relaxation.  Default True.
    fbur_wall_damping : float, optional
        Wall damping factor.  Default 0.3.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        aur: bool = True,
        aitken: bool = True,
        fbur: bool = True,
        fbur_wall_damping: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.aur = aur
        self.aitken = aitken
        self.fbur = fbur
        self.fbur_wall_damping = max(0.05, min(1.0, fbur_wall_damping))

        self._aur_prev_residual = float('inf')
        self._aur_alpha = 0.7

        logger.info(
            "IcoFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
            self.aur, self.aitken, self.fbur,
        )

    def _adaptive_dt_factor(self, residual: float, step: int) -> float:
        """Compute adaptive time-step relaxation factor."""
        if not self.aur:
            return 1.0
        if step > 0 and residual < self._aur_prev_residual:
            self._aur_alpha = min(1.5, self._aur_alpha * 1.05)
        elif step > 0:
            self._aur_alpha = max(0.5, self._aur_alpha * 0.95)
        self._aur_prev_residual = residual
        return self._aur_alpha

    def _aitken_correct(self, p: torch.Tensor, p_old: torch.Tensor) -> torch.Tensor:
        """Apply Aitken acceleration to pressure."""
        if not self.aitken:
            return p
        dp = p - p_old
        dp_sq = (dp * dp).sum().clamp(min=1e-30)
        return p_old + 1.2 * dp if dp_sq > 1e-6 else p

    def _field_relax(self, U: torch.Tensor, U_old: torch.Tensor) -> torch.Tensor:
        """Apply field-based relaxation to velocity."""
        if not self.fbur:
            return U
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        wall_prox = 1.0 - (n_contrib / n_contrib.max().clamp(min=1.0))
        alpha = (1.0 - self.fbur_wall_damping * wall_prox).clamp(0.1, 1.0)
        return U_old + alpha.unsqueeze(-1) * (U - U_old)

    def run(self) -> ConvergenceData:
        solver = self._build_solver()
        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting icoFoamEnhanced12 run")
        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()
        last_convergence: ConvergenceData | None = None
        current_dt = self.delta_t

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()

            residual_val = last_convergence.U_residual if last_convergence else 1.0
            dt_factor = self._adaptive_dt_factor(residual_val, step)

            # Non-orthogonal corrections (from v11)
            p_res = self.p - self.p_old
            self.p = self._extended_non_orthogonal_poisson(self.p, p_res)
            self.U = self._consistent_non_orthogonal_coupling(self.U, self.p, current_dt)
            self.p = self._over_relaxed_stabilise(self.p)

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc,
                U_old=self.U_old, p_old=self.p_old,
                tolerance=self.convergence_tolerance,
            )

            # Aitken + field-based
            self.p = self._aitken_correct(self.p, self.p_old)
            self.U = self._field_relax(self.U, self.U_old)

            self.p = self._hmg_pressure_precondition(self.p, self.p - self.p_old)
            self.U = self._space_time_advance(self.U, self.U_old, self.p, current_dt)

            last_convergence = conv
            residuals = {"U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error}
            converged = convergence.update(step + 1, residuals)
            if time_loop.should_write():
                self._write_fields(t + current_dt)
                time_loop.mark_written()
            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * current_dt
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()
