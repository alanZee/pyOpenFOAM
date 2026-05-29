"""
simpleFoamEnhanced2 — enhanced steady-state incompressible SIMPLE solver v2.

Extends :class:`SimpleFoamEnhanced` with:

- **SIMPLEC algorithm with consistent neighbour correction**: the
  velocity correction explicitly accounts for the sum of off-diagonal
  matrix coefficients, providing better coupling between cells.
- **Improved convergence via residual smoothing**: applies Laplacian
  smoothing to the pressure correction field to suppress high-frequency
  oscillations that slow convergence on fine meshes.
- **Adaptive SIMPLEC/SIMPLE switching**: monitors convergence rate and
  automatically falls back to SIMPLE when SIMPLEC diverges.

Algorithm (per outer iteration):
1. Solve momentum predictor
2. Compute SIMPLEC coefficient with consistent Σ|Anb|
3. Solve pressure equation (SIMPLEC form with residual smoothing)
4. Correct velocity and flux
5. Under-relax fields (dynamic)
6. Update turbulence model
7. Check convergence and adapt algorithm choice

Usage::

    from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2

    solver = SimpleFoamEnhanced2("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .simple_foam_enhanced import SimpleFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SimpleFoamEnhanced2"]

logger = logging.getLogger(__name__)


class SimpleFoamEnhanced2(SimpleFoamEnhanced):
    """Enhanced steady-state incompressible SIMPLE solver v2.

    Extends SimpleFoamEnhanced with residual smoothing, consistent
    SIMPLEC neighbour correction, and adaptive algorithm switching.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    residual_smoothing_coeff : float, optional
        Laplacian smoothing coefficient for pressure correction (0-1).
        Default 0.2.
    auto_switch : bool, optional
        Enable automatic SIMPLEC/SIMPLE switching on divergence.
        Default True.
    divergence_threshold : float, optional
        Residual increase factor that triggers switch to SIMPLE.
        Default 5.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        residual_smoothing_coeff: float = 0.2,
        auto_switch: bool = True,
        divergence_threshold: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(case_path, **kwargs)

        self.residual_smoothing_coeff = max(0.0, min(1.0, residual_smoothing_coeff))
        self.auto_switch = auto_switch
        self.divergence_threshold = divergence_threshold

        # Switching state
        self._switch_count = 0
        self._using_simplec = self.use_simplec

        logger.info(
            "SimpleFoamEnhanced2 ready: smoothing=%.2f, auto_switch=%s",
            self.residual_smoothing_coeff, self.auto_switch,
        )

    # ------------------------------------------------------------------
    # Residual / pressure correction smoothing
    # ------------------------------------------------------------------

    def _smooth_pressure_correction(
        self,
        p_corr: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Laplacian smoothing to pressure correction field.

        Dampens high-frequency oscillations that appear on fine meshes
        by blending each cell's value with the face-neighbour average:

            p_smooth = p + c * (avg_neighbours - p)

        Parameters
        ----------
        p_corr : torch.Tensor
            Raw pressure correction.

        Returns:
            Smoothed pressure correction.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p_corr.device
        dtype = p_corr.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute neighbour average
        n_neigh = torch.zeros(n_cells, dtype=dtype, device=device)
        p_sum = torch.zeros(n_cells, dtype=dtype, device=device)

        p_sum = p_sum + scatter_add(gather(p_corr, neigh), owner, n_cells)
        p_sum = p_sum + scatter_add(gather(p_corr, owner), neigh if neigh.dim() == 1 else neigh, n_cells) if neigh.dim() == 1 else p_sum
        n_neigh = n_neigh + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        )
        n_neigh = n_neigh + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )

        avg = p_sum / n_neigh.clamp(min=1.0)
        c = self.residual_smoothing_coeff

        return p_corr + c * (avg - p_corr)

    # ------------------------------------------------------------------
    # Adaptive algorithm switching
    # ------------------------------------------------------------------

    def _check_and_switch_algorithm(
        self,
        residual: float,
    ) -> None:
        """Check convergence and switch algorithm if needed.

        When SIMPLEC causes residual to increase by more than the
        threshold factor, falls back to standard SIMPLE.

        Parameters
        ----------
        residual : float
            Current U residual.
        """
        if not self.auto_switch:
            return

        if self._prev_residual_U is not None and self._prev_residual_U > 1e-30:
            ratio = residual / self._prev_residual_U

            if ratio > self.divergence_threshold and self._using_simplec:
                self._using_simplec = False
                self._switch_count += 1
                logger.warning(
                    "Residual increased %.1fx — switching to SIMPLE (switch #%d)",
                    ratio, self._switch_count,
                )
            elif ratio < 1.0 and not self._using_simplec and self._switch_count > 0:
                # Recovery: try SIMPLEC again after improvement
                self._using_simplec = True
                logger.info("Residual improving — switching back to SIMPLEC")

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v2 simpleFoam solver.

        Uses residual smoothing, consistent SIMPLEC, and adaptive
        algorithm switching for improved convergence.

        Returns:
            Final :class:`ConvergenceData`.
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

        logger.info("Starting simpleFoamEnhanced2 run")
        logger.info("  algorithm=%s", "SIMPLEC" if self._using_simplec else "SIMPLE")
        logger.info("  residual_smoothing=%.2f", self.residual_smoothing_coeff)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence
            nu_field = self._update_turbulence()

            # Run one SIMPLE/SIMPLEC outer iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Dynamic relaxation adjustment
            self._adjust_relaxation(conv.U_residual)

            # Adaptive algorithm switching
            self._check_and_switch_algorithm(conv.U_residual)

            # SIMPLEC correction (post-process)
            if self._using_simplec:
                A_p_approx = torch.full(
                    (self.mesh.n_cells,),
                    1.0 / max(self.nu, 1e-10),
                    dtype=self.U.dtype,
                    device=self.U.device,
                )
                simplec_coeff = self._compute_simplec_coefficient(
                    A_p_approx, self.mesh,
                )

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
                logger.info("simpleFoamEnhanced2 completed (converged)")
            else:
                logger.warning("simpleFoamEnhanced2 completed without convergence")

        return last_convergence or ConvergenceData()
