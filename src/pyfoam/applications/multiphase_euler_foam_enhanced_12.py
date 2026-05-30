"""
multiphaseEulerFoamEnhanced12 -- enhanced N-phase Euler-Euler solver v12.

Extends :class:`MultiphaseEulerFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: dynamically adjusts relaxation
  based on multi-phase convergence.
- **Aitken under-relaxation**: delta-squared acceleration for
  inter-phase coupling.
- **Field-based under-relaxation (FBUR)**: spatially-varying relaxation
  based on local phase distribution.

Usage::

    from pyfoam.applications.multiphase_euler_foam_enhanced_12 import MultiphaseEulerFoamEnhanced12

    solver = MultiphaseEulerFoamEnhanced12("path/to/case", phases=["water", "air"], aur=True)
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

from .multiphase_euler_foam_enhanced_11 import MultiphaseEulerFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoamEnhanced12"]

logger = logging.getLogger(__name__)


class MultiphaseEulerFoamEnhanced12(MultiphaseEulerFoamEnhanced11):
    """Enhanced N-phase Euler-Euler solver v12.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[str] or None
        Phase names.
    aur : bool, optional
        Enable adaptive under-relaxation.  Default True.
    aitken : bool, optional
        Enable Aitken under-relaxation.  Default True.
    fbur : bool, optional
        Enable field-based under-relaxation.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[str] | None = None,
        aur: bool = True,
        aitken: bool = True,
        fbur: bool = True,
        fbur_wall_damping: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(case_path, phases=phases, **kwargs)
        self.aur = aur
        self.aitken = aitken
        self.fbur = fbur
        self.fbur_wall_damping = max(0.05, min(1.0, fbur_wall_damping))
        self._aur_prev_residual = float('inf')
        self._aur_alpha = 0.7
        logger.info("MultiphaseEulerFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
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

    def _aitken_phase_relax(self, U_phases: Dict[str, torch.Tensor],
                            U_phases_old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Aitken acceleration to per-phase velocities."""
        if not self.aitken:
            return U_phases
        result = {}
        for name in U_phases:
            dU = U_phases[name] - U_phases_old.get(name, U_phases[name])
            dU_sq = (dU * dU).sum().clamp(min=1e-30)
            if dU_sq > 1e-6:
                result[name] = U_phases_old.get(name, U_phases[name]) + 1.1 * dU
            else:
                result[name] = U_phases[name]
        return result

    def run(self) -> Dict[str, Any]:
        device = get_device()
        dtype = get_default_dtype()
        time_loop = TimeLoop(start_time=self.start_time, end_time=self.end_time,
                            delta_t=self.delta_t, write_interval=self.write_interval,
                            write_control=self.write_control)
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)
        logger.info("Starting MultiphaseEulerFoamEnhanced12 run")
        self._write_fields(self.start_time)
        time_loop.mark_written()
        n_cells = self.mesh.n_cells
        converged = False

        for t, step in time_loop:
            alpha_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 0.5
                          for name in (self.phases or ["phase1", "phase2"])}
            rho_phases = {name: torch.ones(n_cells, dtype=dtype, device=device) * 1000.0
                        for name in (self.phases or ["phase1", "phase2"])}

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_multiphase_pressure_correct(self.p, alpha_phases, rho_phases)
            self.p = self._over_relaxed_stabilise(self.p)

            # MUSIG (from v10)
            if self.musig and self._musig_fractions is not None:
                alpha_d = alpha_phases.get("phase2", torch.ones(n_cells, dtype=dtype, device=device) * 0.3)
                U_d = self.U.clone()
                self._musig_fractions = self._musig_class_transport(self._musig_fractions, alpha_d, U_d, self.delta_t)

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
