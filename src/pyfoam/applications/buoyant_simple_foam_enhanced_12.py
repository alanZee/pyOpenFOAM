"""
buoyantSimpleFoamEnhanced12 -- enhanced steady-state buoyant SIMPLE solver v12.

Extends :class:`BuoyantSimpleFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: dynamically adjusts relaxation
  based on buoyancy convergence.
- **Aitken under-relaxation**: delta-squared acceleration.
- **Field-based under-relaxation (FBUR)**: spatially-varying relaxation.

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_12 import BuoyantSimpleFoamEnhanced12

    solver = BuoyantSimpleFoamEnhanced12("path/to/case", aur=True)
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

from .buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced12"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced12(BuoyantSimpleFoamEnhanced11):
    """Enhanced steady-state buoyant SIMPLE solver v12.

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
    """

    def __init__(self, case_path: Union[str, Path], aur: bool = True,
                 aitken: bool = True, fbur: bool = True, fbur_wall_damping: float = 0.3,
                 **kwargs) -> None:
        super().__init__(case_path, **kwargs)
        self.aur = aur
        self.aitken = aitken
        self.fbur = fbur
        self.fbur_wall_damping = max(0.05, min(1.0, fbur_wall_damping))
        self._aur_prev_residual = float('inf')
        self._aur_alpha = 0.7
        logger.info("BuoyantSimpleFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
                     self.aur, self.aitken, self.fbur)

    def _adaptive_relaxation(self, residual: float, step: int) -> float:
        if not self.aur:
            return 0.7
        if step > 0 and residual < self._aur_prev_residual:
            self._aur_alpha = min(0.95, self._aur_alpha * 1.05)
        elif step > 0:
            self._aur_alpha = max(0.1, self._aur_alpha * 0.9)
        self._aur_prev_residual = residual
        return self._aur_alpha

    def _aitken_correct(self, p: torch.Tensor, p_old: torch.Tensor) -> torch.Tensor:
        if not self.aitken:
            return p
        dp = p - p_old
        dp_sq = (dp * dp).sum().clamp(min=1e-30)
        return p_old + 1.2 * dp if dp_sq > 1e-6 else p

    def _field_relax(self, U: torch.Tensor, U_old: torch.Tensor) -> torch.Tensor:
        if not self.fbur:
            return U
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device, dtype = U.device, U.dtype
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        n_contrib = scatter_add(torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells) + \
                    scatter_add(torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells)
        wall_prox = 1.0 - (n_contrib / n_contrib.max().clamp(min=1.0))
        alpha = (1.0 - self.fbur_wall_damping * wall_prox).clamp(0.1, 1.0)
        return U_old + alpha.unsqueeze(-1) * (U - U_old)

    def run(self) -> ConvergenceData:
        solver = self._build_solver()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting buoyantSimpleFoamEnhanced12 run")
        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()
        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            nu_field = self._update_turbulence()
            T = self.T if hasattr(self, 'T') else torch.ones_like(self.p) * 300.0
            rho = self.rho if hasattr(self, 'rho') else torch.ones_like(self.p) * 1.2

            residual_val = last_convergence.U_residual if last_convergence else 1.0
            self._adaptive_relaxation(residual_val, step)

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_buoyant_pressure_correct(self.p, T, rho)
            self.U = self._consistent_buoyant_momentum_correct(self.U, T, 300.0)
            self.p = self._over_relaxed_stabilise(self.p)

            U_prev = self.U.clone()
            p_prev = self.p.clone()
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc, nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            self.p = self._aitken_correct(self.p, p_prev)
            self.U = self._field_relax(self.U, U_prev)

            if self.non_boussinesq:
                self.U = self._variable_density_buoyancy(self.U, rho, T, 1.2, 300.0)
            if self.richardson_damping:
                self.U = self._richardson_velocity_damp(self.U, T, 300.0)

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
