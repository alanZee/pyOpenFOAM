"""
pimpleFoamEnhanced12 -- enhanced transient incompressible PIMPLE solver v12.

Extends :class:`PimpleFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: dynamically adjusts relaxation
  factors based on residual evolution, increasing them during smooth
  convergence and decreasing during oscillations.
- **Aitken under-relaxation**: applies the Aitken delta-squared method
  to accelerate PIMPLE outer iteration convergence.
- **Field-based under-relaxation (FBUR)**: assigns spatially-varying
  relaxation factors based on local flow features.

Algorithm (per time step):
1. Store old fields
2. AUR dynamic relaxation
3. Outer corrector loop with Aitken/FBUR
4. Non-orthogonal corrections (from v11)
5. Update turbulence
6. Write fields

Usage::

    from pyfoam.applications.pimple_foam_enhanced_12 import PimpleFoamEnhanced12

    solver = PimpleFoamEnhanced12("path/to/case", aur=True, aitken=True)
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

from .pimple_foam_enhanced_11 import PimpleFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["PimpleFoamEnhanced12"]

logger = logging.getLogger(__name__)


class PimpleFoamEnhanced12(PimpleFoamEnhanced11):
    """Enhanced transient incompressible PIMPLE solver v12.

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
        self._aitken_U_history: list[torch.Tensor] = []
        self._aur_alpha_U = 0.7
        self._aur_alpha_p = 0.3
        self._aur_prev_residual = float('inf')

        logger.info(
            "PimpleFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
            self.aur, self.aitken, self.fbur,
        )

    # ------------------------------------------------------------------
    # Adaptive under-relaxation
    # ------------------------------------------------------------------

    def _adaptive_relaxation(
        self,
        residual: float,
        step: int,
    ) -> tuple[float, float]:
        """Compute adaptive relaxation factors."""
        if not self.aur:
            return self.alpha_U, self.alpha_p

        if step > 0 and residual < self._aur_prev_residual:
            self._aur_alpha_U = min(0.95, self._aur_alpha_U * self.aur_growth)
            self._aur_alpha_p = min(0.5, self._aur_alpha_p * self.aur_growth)
        elif step > 0:
            self._aur_alpha_U = max(0.1, self._aur_alpha_U * 0.9)
            self._aur_alpha_p = max(0.05, self._aur_alpha_p * 0.9)

        self._aur_prev_residual = residual
        return self._aur_alpha_U, self._aur_alpha_p

    # ------------------------------------------------------------------
    # Aitken under-relaxation
    # ------------------------------------------------------------------

    def _aitken_relaxation(
        self,
        p: torch.Tensor,
        p_old: torch.Tensor,
        U: torch.Tensor,
        U_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Aitken delta-squared acceleration."""
        if not self.aitken:
            return p, U

        self._aitken_p_history.append(p.clone())
        self._aitken_U_history.append(U.clone())

        if len(self._aitken_p_history) > self.aitken_depth:
            self._aitken_p_history.pop(0)
            self._aitken_U_history.pop(0)

        if len(self._aitken_p_history) < 2:
            return p, U

        dp_curr = p - self._aitken_p_history[-2]
        dp_prev = (self._aitken_p_history[-2] - self._aitken_p_history[-3]
                   if len(self._aitken_p_history) >= 3
                   else self._aitken_p_history[-2] - self._aitken_p_history[-2])

        dp_diff = dp_curr - dp_prev
        dp_sq = (dp_diff * dp_diff).sum().clamp(min=1e-30)

        if dp_sq > 1e-30:
            alpha = float(((dp_curr * dp_diff).sum() / dp_sq).clamp(0.1, 2.0).item())
        else:
            alpha = 1.0

        return p_old + alpha * (p - p_old), U_old + alpha * (U - U_old)

    # ------------------------------------------------------------------
    # Field-based under-relaxation
    # ------------------------------------------------------------------

    def _field_based_relaxation(
        self,
        p: torch.Tensor,
        p_old: torch.Tensor,
        U: torch.Tensor,
        U_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply spatially-varying under-relaxation."""
        if not self.fbur:
            return p, U

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

        p_relaxed = p_old + alpha_field * (p - p_old)
        U_relaxed = U_old + alpha_field.unsqueeze(-1) * (U - U_old)

        return p_relaxed, U_relaxed

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v12 pimpleFoam solver."""
        solver = self._build_solver()

        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)

        logger.info("Starting pimpleFoamEnhanced12 run")
        logger.info("  aur=%s, aitken=%s, fbur=%s", self.aur, self.aitken, self.fbur)

        U_bc = self._build_boundary_conditions()
        self._write_fields(self.start_time)
        time_loop.mark_written()

        self._aitken_p_history.clear()
        self._aitken_U_history.clear()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            residual_val = last_convergence.U_residual if last_convergence else 1.0
            self._adaptive_relaxation(residual_val, step)

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_non_orthogonal_pressure(self.p, self.U, self.delta_t)
            self.U = self._consistent_non_orthogonal_momentum(self.U, self.p)
            self.p = self._over_relaxed_stabilise(self.p, self.U)

            # VMS + OINN (from v10)
            self.p = self._vms_pressure_stabilise(self.p, self.U, self.delta_t)
            self.U = self._oinn_correct(self.U, U_prev, self.delta_t)

            # Main solve
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi, U_bc=U_bc,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Aitken + field-based
            self.p, self.U = self._aitken_relaxation(self.p, p_prev, self.U, U_prev)
            self.p, self.U = self._field_based_relaxation(self.p, p_prev, self.U, U_prev)

            if self.energy_preserving:
                self.U = self._energy_budget_correct(self.U, U_prev, self.delta_t)

            residuals = {
                "U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error,
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

        if last_convergence and last_convergence.converged:
            logger.info("pimpleFoamEnhanced12 completed (converged)")
        else:
            logger.warning("pimpleFoamEnhanced12 completed without convergence")

        return last_convergence or ConvergenceData()
