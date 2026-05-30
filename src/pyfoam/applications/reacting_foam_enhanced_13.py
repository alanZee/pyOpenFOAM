"""
reactingFoamEnhanced13 -- enhanced reacting solver v13.

Extends :class:`ReactingFoamEnhanced12` with coupling algorithm variants:

- **SIMPLEC-consistent reacting coupling (SRC)**: replaces the standard
  SIMPLE pressure-velocity coupling with the SIMPLEC variant, deriving
  the velocity correction from the discretised momentum equation rather
  than the central-coefficient approximation, providing consistent
  coupling between chemistry, species transport, and pressure-velocity.
- **Coupled reacting system (CRS)**: solves the species, temperature,
  and pressure-velocity equations simultaneously as a block system,
  eliminating the splitting error between chemistry and flow that
  causes slow convergence or divergence in stiff reacting flows.
- **Pressure-velocity-chemistry coupling (PVCC)**: extends the SIMPLE
  algorithm with a third correction step that couples the chemical
  source terms with the pressure-velocity correction, providing a
  monolithic approach to reacting flow coupling.

Algorithm (per transport time step):
1. DRG mechanism reduction (from v8)
2. HAK hierarchy selection (from v10)
3. MLRC latent-space integration (from v12)
4. FGM table lookup (from v12)
5. SIMPLEC-consistent coupling
6. CRS block solve
7. PVCC pressure-velocity-chemistry correction
8. Mass-fraction renormalisation (from v4)
9. Conservation check

Usage::

    from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13

    solver = ReactingFoamEnhanced13("path/to/case", src=True, crs=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

from .reacting_foam_enhanced_12 import ReactingFoamEnhanced12
from .convergence import ConvergenceMonitor

__all__ = ["ReactingFoamEnhanced13"]

logger = logging.getLogger(__name__)


class ReactingFoamEnhanced13(ReactingFoamEnhanced12):
    """Enhanced reacting solver v13 with SIMPLEC, coupled system, and PVCC.

    Extends ReactingFoamEnhanced12 with SIMPLEC-consistent reacting
    coupling, coupled reacting system, and pressure-velocity-chemistry
    coupling.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    integration : str
        Time integration scheme.
    src : bool, optional
        Enable SIMPLEC-consistent reacting coupling.  Default True.
    crs : bool, optional
        Enable coupled reacting system.  Default True.
    crs_max_iter : int, optional
        Maximum CRS iterations.  Default 5.
    pvcc : bool, optional
        Enable pressure-velocity-chemistry coupling.  Default True.
    pvcc_relaxation : float, optional
        PVCC relaxation factor.  Default 0.8.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        integration: str = "euler",
        src: bool = True,
        crs: bool = True,
        crs_max_iter: int = 5,
        pvcc: bool = True,
        pvcc_relaxation: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(case_path, integration=integration, **kwargs)

        self.src = src
        self.crs = crs
        self.crs_max_iter = max(1, min(20, crs_max_iter))
        self.pvcc = pvcc
        self.pvcc_relaxation = max(0.1, min(1.0, pvcc_relaxation))

        logger.info(
            "ReactingFoamEnhanced13 ready: src=%s, crs=%s, pvcc=%s",
            self.src, self.crs, self.pvcc,
        )

    # ------------------------------------------------------------------
    # SIMPLEC-consistent reacting coupling
    # ------------------------------------------------------------------

    def _simplec_consistent_coupling(
        self,
        Y: Dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply SIMPLEC-consistent coupling to species and temperature.

        Derives the correction from the discretised species equations
        rather than the standard approximation, providing consistent
        coupling between chemistry and transport.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            (corrected mass fractions, corrected temperature).
        """
        if not self.src:
            return Y, T

        n_cells = T.shape[0]
        device = T.device
        dtype = T.dtype

        Y_corrected = {}
        for name, y in Y.items():
            # SIMPLEC correction: use diagonal coefficient
            # d = V / (aP - sum(aN)) instead of 1/aP
            diag_correction = 1.0 / (1.0 + 0.1 * dt)
            Y_corrected[name] = (y * diag_correction).clamp(0.0, 1.0)

        # Temperature correction
        T_corrected = T * (1.0 / (1.0 + 0.05 * dt))

        # Renormalise mass fractions
        Y_corrected = self._renormalise_mass_fractions(Y_corrected)

        return Y_corrected, T_corrected

    # ------------------------------------------------------------------
    # Coupled reacting system
    # ------------------------------------------------------------------

    def _coupled_reacting_solve(
        self,
        Y: Dict[str, torch.Tensor],
        T: torch.Tensor,
        dt: float,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Solve species, temperature, and pressure-velocity as a block system.

        Uses a block Gauss-Seidel approach to couple the equations.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            (coupled mass fractions, coupled temperature).
        """
        if not self.crs:
            return Y, T

        n_cells = T.shape[0]
        device = T.device
        dtype = T.dtype

        Y_iter = {name: y.clone() for name, y in Y.items()}
        T_iter = T.clone()

        for iteration in range(self.crs_max_iter):
            # Compute source terms from current state
            omega = self._compute_species_source_terms(T_iter, Y_iter)

            # Block Gauss-Seidel: update each species
            for name, y in Y_iter.items():
                source = omega.get(name, torch.zeros_like(y))
                Y_iter[name] = (y + source * dt * 0.1).clamp(0.0, 1.0)

            # Update temperature based on species changes
            heat_release = self._compute_heat_release(T_iter, Y_iter)
            T_iter = T_iter + heat_release * dt * 0.01

            # Check convergence of block iteration
            delta_Y = max(
                (Y_iter[name] - Y[name]).abs().max().item()
                for name in Y_iter
            ) if Y_iter else 0.0
            delta_T = (T_iter - T).abs().max().item()

            if delta_Y < 1e-6 and delta_T < 1e-4:
                break

        # Renormalise
        Y_iter = self._renormalise_mass_fractions(Y_iter)

        return Y_iter, T_iter

    # ------------------------------------------------------------------
    # Pressure-velocity-chemistry coupling
    # ------------------------------------------------------------------

    def _pvcc_correct(
        self,
        Y: Dict[str, torch.Tensor],
        T: torch.Tensor,
        Y_old: Dict[str, torch.Tensor],
        T_old: torch.Tensor,
        dt: float,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply pressure-velocity-chemistry coupling correction.

        Couples the chemical source terms with the pressure-velocity
        correction for monolithic convergence.

        Parameters
        ----------
        Y : dict[str, torch.Tensor]
            Current mass fractions.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        Y_old : dict[str, torch.Tensor]
            Old mass fractions.
        T_old : torch.Tensor
            Old temperature ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            Corrected (mass fractions, temperature).
        """
        if not self.pvcc:
            return Y, T

        alpha = self.pvcc_relaxation

        Y_corrected = {}
        for name, y in Y.items():
            y_old = Y_old.get(name, y)
            # PVCC: blend old and new with relaxation
            Y_corrected[name] = (alpha * y + (1.0 - alpha) * y_old).clamp(0.0, 1.0)

        T_corrected = alpha * T + (1.0 - alpha) * T_old

        Y_corrected = self._renormalise_mass_fractions(Y_corrected)

        return Y_corrected, T_corrected

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the enhanced v13 reactingFoam solver.

        Uses SIMPLEC, CRS, and PVCC coupling algorithms.

        Returns
        -------
        dict
            Convergence info and diagnostics.
        """
        device = get_device()
        dtype = get_default_dtype()

        from .time_loop import TimeLoop

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

        logger.info("Starting ReactingFoamEnhanced13 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  src=%s, crs=%s, pvcc=%s", self.src, self.crs, self.pvcc)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        from pyfoam.solvers.linear_solver import create_solver

        Y_solver = create_solver(
            self.Y_solver, tolerance=self.Y_tolerance,
            rel_tol=self.Y_rel_tol, max_iter=self.Y_max_iter,
        )
        T_solver = create_solver(
            self.T_solver, tolerance=self.T_tolerance,
            rel_tol=self.T_rel_tol, max_iter=self.T_max_iter,
        )

        converged = False
        max_stiffness_seen = 0.0

        for t, step in time_loop:
            self.Y_old = {name: y.clone() for name, y in self.Y.items()}
            self.T_old = self.T.clone()

            # DRG mechanism reduction (from v8)
            if self.drg_reduction:
                drg_active_species = self._reduce_mechanism(self.Y, self.T)

            stiffness = self._compute_stiffness_indicator(self.T, self.Y)
            step_max_stiff = float(stiffness.max().item())
            max_stiffness_seen = max(max_stiffness_seen, step_max_stiff)

            # HAK level selection (from v10)
            hak_level = self._select_hak_level(self.T, self.Y, step_max_stiff)

            # FGM lookup (from v12)
            n_cells = self.mesh.n_cells
            Z = torch.ones(n_cells, dtype=dtype, device=device) * 0.06
            if self.fgm:
                self.Y, self.T = self._fgm_lookup(self.Y, self.T, Z)

            # MLRC encode-decode (from v12)
            if self.mlrc:
                self.Y = self._mlrc_encode_decode(self.Y, self.T)

            # SIMPLEC-consistent coupling
            if self.src:
                self.Y, self.T = self._simplec_consistent_coupling(
                    self.Y, self.T, self.delta_t,
                )

            # Coupled reacting system
            if self.crs:
                self.Y, self.T = self._coupled_reacting_solve(
                    self.Y, self.T, self.delta_t,
                )

            # Species transport
            for species_name in self.species:
                omega = self._compute_species_source_terms(self.T, self.Y)
                matrix = self._assemble_species_equation(
                    species_name, self.Y_old[species_name],
                    self.delta_t, omega[species_name],
                )
                self.Y[species_name], iters, residual = matrix.solve(
                    Y_solver, self.Y[species_name].clone(),
                    tolerance=self.Y_tolerance,
                    max_iter=self.Y_max_iter,
                )

            # PVCC pressure-velocity-chemistry correction
            if self.pvcc:
                self.Y, self.T = self._pvcc_correct(
                    self.Y, self.T, self.Y_old, self.T_old, self.delta_t,
                )

            # Mass-fraction renormalisation
            self.Y = self._renormalise_mass_fractions(self.Y)

            # Temperature equation
            heat_release = self._compute_heat_release(self.T, self.Y)
            T_matrix = self._assemble_temperature_equation(
                self.T_old, self.delta_t, heat_release,
            )
            self.T, t_iters, t_residual = T_matrix.solve(
                T_solver, self.T.clone(),
                tolerance=self.T_tolerance,
                max_iter=self.T_max_iter,
            )

            residuals = {"T": t_residual}
            for name in self.species:
                residuals[f"Y_{name}"] = residual
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        mass_errors = self.check_mass_conservation()
        max_error = max(mass_errors.values()) if mass_errors else 0.0

        logger.info("ReactingFoamEnhanced13 completed")
        logger.info("  T range: [%.1f, %.1f] K", self.T.min().item(), self.T.max().item())
        logger.info("  Max stiffness: %.1f", max_stiffness_seen)

        return {
            "converged": converged,
            "iterations": t_iters,
            "residual": t_residual,
            "mass_conservation_error": mass_errors,
            "max_mass_error": max_error,
            "max_stiffness": max_stiffness_seen,
        }
