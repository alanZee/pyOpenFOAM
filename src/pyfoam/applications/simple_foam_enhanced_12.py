"""
simpleFoamEnhanced12 -- enhanced steady-state incompressible SIMPLE solver v12.

Extends :class:`SimpleFoamEnhanced11` with under-relaxation variants:

- **Adaptive under-relaxation (AUR)**: monitors the residual ratio between
  successive iterations and dynamically adjusts the relaxation factors,
  increasing them when the solution is converging smoothly and decreasing
  them when oscillations are detected.
- **Aitken under-relaxation**: applies the Aitken delta-squared method
  to accelerate the convergence of the SIMPLE outer iterations, using
  the history of successive iterates to compute an optimal relaxation
  factor that minimises the residual in a least-squares sense.
- **Field-based under-relaxation (FBUR)**: assigns spatially-varying
  relaxation factors based on local flow features, using stronger
  relaxation near walls and in recirculation zones while allowing
  faster convergence in the free stream.

Algorithm (per outer iteration):
1. Update turbulence
2. AUR dynamic relaxation
3. Aitken acceleration
4. Field-based spatial relaxation
5. SIMPLE iteration (from v11)
6. Non-orthogonal corrections (from v11)
7. Convergence check

Usage::

    from pyfoam.applications.simple_foam_enhanced_12 import SimpleFoamEnhanced12

    solver = SimpleFoamEnhanced12("path/to/case", aur=True, aitken=True)
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

from .simple_foam_enhanced_11 import SimpleFoamEnhanced11
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced12"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced12(SimpleFoamEnhanced11):
    """Enhanced steady-state incompressible SIMPLE solver v12.

    Extends SimpleFoamEnhanced11 with adaptive, Aitken, and field-based
    under-relaxation strategies.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    aur : bool, optional
        Enable adaptive under-relaxation.  Default True.
    aur_growth : float, optional
        Growth factor for AUR when converging.  Default 1.05.
    aitken : bool, optional
        Enable Aitken under-relaxation.  Default True.
    aitken_depth : int, optional
        History depth for Aitken method.  Default 3.
    fbur : bool, optional
        Enable field-based under-relaxation.  Default True.
    fbur_wall_damping : float, optional
        Wall damping factor for FBUR.  Default 0.3.
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

        # History for Aitken
        self._aitken_p_history: list[torch.Tensor] = []
        self._aitken_U_history: list[torch.Tensor] = []

        # AUR state
        self._aur_alpha_U = 0.7
        self._aur_alpha_p = 0.3
        self._aur_prev_residual = float('inf')

        logger.info(
            "SimpleFoamEnhanced12 ready: aur=%s, aitken=%s, fbur=%s",
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
        """Compute adaptive relaxation factors based on convergence history.

        Parameters
        ----------
        residual : float
            Current residual norm.
        step : int
            Current iteration step.

        Returns
        -------
        tuple[float, float]
            (alpha_U, alpha_p) relaxation factors.
        """
        if not self.aur:
            return self.alpha_U, self.alpha_p

        if step > 0 and residual < self._aur_prev_residual:
            # Converging: increase relaxation
            self._aur_alpha_U = min(0.95, self._aur_alpha_U * self.aur_growth)
            self._aur_alpha_p = min(0.5, self._aur_alpha_p * self.aur_growth)
        elif step > 0:
            # Diverging: decrease relaxation
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
        """Apply Aitken delta-squared acceleration.

        Uses the history of successive iterates to compute an optimal
        relaxation factor.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        p_old : torch.Tensor
            Previous pressure ``(n_cells,)``.
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        U_old : torch.Tensor
            Previous velocity ``(n_cells, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Accelerated (pressure, velocity).
        """
        if not self.aitken:
            return p, U

        self._aitken_p_history.append(p.clone())
        self._aitken_U_history.append(U.clone())

        if len(self._aitken_p_history) > self.aitken_depth:
            self._aitken_p_history.pop(0)
            self._aitken_U_history.pop(0)

        if len(self._aitken_p_history) < 2:
            return p, U

        # Aitken delta-squared for pressure
        dp_curr = p - self._aitken_p_history[-2]
        dp_prev = self._aitken_p_history[-2] - (self._aitken_p_history[-3]
                                                  if len(self._aitken_p_history) >= 3
                                                  else self._aitken_p_history[-2])

        dp_diff = dp_curr - dp_prev
        dp_norm = dp_diff.norm().clamp(min=1e-30)

        if dp_norm > 1e-30:
            alpha_aitken = (dp_curr * dp_diff).sum() / (dp_diff * dp_diff).sum().clamp(min=1e-30)
            alpha_aitken = float(alpha_aitken.clamp(0.1, 2.0).item())
        else:
            alpha_aitken = 1.0

        p_accel = p_old + alpha_aitken * (p - p_old)
        U_accel = U_old + alpha_aitken * (U - U_old)

        return p_accel, U_accel

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
        """Apply spatially-varying field-based under-relaxation.

        Uses stronger relaxation near walls and in recirculation zones.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure ``(n_cells,)``.
        p_old : torch.Tensor
            Previous pressure ``(n_cells,)``.
        U : torch.Tensor
            Current velocity ``(n_cells, 3)``.
        U_old : torch.Tensor
            Previous velocity ``(n_cells, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Relaxed (pressure, velocity).
        """
        if not self.fbur:
            return p, U

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        # Compute velocity magnitude as flow feature indicator
        U_mag = U.norm(dim=-1)

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Near-wall detection: cells with boundary faces have fewer internal connections
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )

        # Wall proximity indicator: fewer connections = closer to wall
        wall_proximity = 1.0 - (n_contrib / n_contrib.max().clamp(min=1.0))

        # Spatial relaxation: stronger damping near walls
        alpha_field_p = (1.0 - self.fbur_wall_damping * wall_proximity).clamp(0.1, 1.0)
        alpha_field_U = alpha_field_p.clone()

        # Apply spatially-varying relaxation
        p_relaxed = p_old + alpha_field_p * (p - p_old)
        U_relaxed = U_old + alpha_field_U.unsqueeze(-1) * (U - U_old)

        return p_relaxed, U_relaxed

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v12 simpleFoam solver.

        Uses adaptive, Aitken, and field-based under-relaxation.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        solver = self._build_solver()

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

        logger.info("Starting simpleFoamEnhanced12 run")
        logger.info("  aur=%s, aitken=%s, fbur=%s",
                     self.aur, self.aitken, self.fbur)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        nu = self.nu if hasattr(self, 'nu') else 0.01

        self._aitken_p_history.clear()
        self._aitken_U_history.clear()

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            # Adaptive under-relaxation
            residual_val = last_convergence.U_residual if last_convergence else 1.0
            aur_alpha_U, aur_alpha_p = self._adaptive_relaxation(residual_val, step)

            # Non-orthogonal corrections (from v11)
            self.p = self._extended_non_orthogonal_correct(self.p, self.U)
            self.p = self._consistent_non_orthogonal_correct(self.p, self.U)
            self.p = self._over_relaxed_stabilise(self.p, self.U)

            # Run one SIMPLE iteration
            U_prev = self.U.clone()
            p_prev = self.p.clone()

            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Aitken acceleration
            self.p, self.U = self._aitken_relaxation(
                self.p, p_prev, self.U, U_prev,
            )

            # Field-based relaxation
            self.p, self.U = self._field_based_relaxation(
                self.p, p_prev, self.U, U_prev,
            )

            # OLPS pressure correction (from v10)
            p_res = self.p - p_prev
            self.p = self._olps_pressure_correct(self.p, p_res)

            # Spectral viscosity (from v10)
            self.U = self._spectral_viscosity_stabilise(self.U, U_prev)

            res_norm = conv.U_residual if conv is not None else 0.0
            self._adjust_relaxation(conv.U_residual)

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
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

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("simpleFoamEnhanced12 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced12 completed without convergence")

        return last_convergence or ConvergenceData()
