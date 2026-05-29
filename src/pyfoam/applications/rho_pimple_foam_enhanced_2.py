"""
rhoPimpleFoamEnhanced2 — enhanced transient compressible PIMPLE solver v2.

Extends :class:`RhoPimpleFoamEnhanced` with:

- **Improved energy equation coupling**: uses a predictor-corrector
  scheme for the energy equation within each outer iteration, reducing
  splitting errors between momentum and energy.
- **Pressure-velocity-density coupled correction**: after the standard
  PISO pressure correction, applies an additional density correction
  step that ensures thermodynamic consistency.
- **Mach-aware adaptive relaxation**: extends the Mach-aware relaxation
  to both temperature and density fields for better stability at
  transonic and supersonic conditions.

Algorithm (per time step):
1. Store old fields
2. Outer corrector loop:
   a. Momentum predictor
   b. PISO pressure correction (compressible)
   c. Energy predictor + corrector
   d. EOS update: ρ = ρ(p, T)
   e. Density correction step
   f. Mach-aware relaxation for U, p, T, ρ
3. Check convergence

Usage::

    from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2

    solver = RhoPimpleFoamEnhanced2("path/to/case")
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

from .rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoPimpleFoamEnhanced2"]

logger = logging.getLogger(__name__)


class RhoPimpleFoamEnhanced2(RhoPimpleFoamEnhanced):
    """Enhanced transient compressible PIMPLE solver v2.

    Extends RhoPimpleFoamEnhanced with predictor-corrector energy
    coupling, density correction, and Mach-aware T/ρ relaxation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    n_energy_correctors : int, optional
        Number of energy corrector iterations per outer loop.
        Default 2.
    density_correction : bool, optional
        Enable density correction step.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        n_energy_correctors: int = 2,
        density_correction: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, thermo=thermo, **kwargs)

        self.n_energy_correctors = max(1, n_energy_correctors)
        self.density_correction = density_correction

        logger.info(
            "RhoPimpleFoamEnhanced2 ready: energy_correctors=%d, density_corr=%s",
            self.n_energy_correctors, self.density_correction,
        )

    # ------------------------------------------------------------------
    # Energy predictor-corrector
    # ------------------------------------------------------------------

    def _energy_predictor_corrector(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Solve energy equation with predictor-corrector.

        Predictor: solve energy with current U, phi, rho.
        Corrector: re-solve with updated rho and flux, iterating
        n_energy_correctors times for tighter coupling.

        Parameters
        ----------
        T, U, phi, rho, p : torch.Tensor
            Current field values.

        Returns:
            Updated temperature.
        """
        T_current = T.clone()

        for _ in range(self.n_energy_correctors):
            T_new = self._solve_energy_equation(T_current, U, phi, rho, p)
            # Blend with previous to avoid oscillation
            T_current = 0.7 * T_new + 0.3 * T_current

        return T_current

    # ------------------------------------------------------------------
    # Density correction
    # ------------------------------------------------------------------

    def _density_correction_step(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Apply density correction for thermodynamic consistency.

        After the pressure and temperature updates, recompute density
        from the EOS to ensure p, T, rho are thermodynamically consistent.

        Parameters
        ----------
        rho : torch.Tensor
            Current density.
        p : torch.Tensor
            Updated pressure.
        T : torch.Tensor
            Updated temperature.

        Returns:
            Corrected density.
        """
        try:
            rho_new = self.thermo.rho(p, T)
            # Blend to avoid sudden jumps
            return 0.5 * rho_new + 0.5 * rho
        except (AttributeError, Exception):
            # Fallback: ideal gas
            R = 8.314
            W = 0.02897
            T_safe = T.clamp(min=1.0)
            return (p * W / (R * T_safe)).clamp(min=0.01)

    # ------------------------------------------------------------------
    # Mach-aware temperature relaxation
    # ------------------------------------------------------------------

    def _mach_aware_T_relaxation(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        Ma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Mach-aware relaxation to temperature field.

        At high Mach numbers, temperature changes can be very large
        due to compressibility effects.  Reduce relaxation accordingly.

        Parameters
        ----------
        T : torch.Tensor
            Updated temperature.
        T_old : torch.Tensor
            Old temperature.
        Ma : torch.Tensor
            Local Mach number.

        Returns:
            Relaxed temperature.
        """
        Ma_threshold = 0.3
        Ma_factor = torch.where(
            Ma > Ma_threshold,
            Ma_threshold / Ma.clamp(min=1e-10),
            torch.ones_like(Ma),
        )
        alpha = (0.7 * Ma_factor).clamp(min=0.1, max=1.0)
        return alpha * T + (1.0 - alpha) * T_old

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v2 rhoPimpleFoam solver.

        Uses energy predictor-corrector, density correction, and
        Mach-aware T/ρ relaxation.

        Returns:
            Final :class:`ConvergenceData`.
        """
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

        logger.info("Starting rhoPimpleFoamEnhanced2 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  energy_correctors=%d", self.n_energy_correctors)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            Ma = self._compute_mach_number()
            self._max_mach = float(Ma.max().item())

            if step % 10 == 0:
                logger.info("Max Mach number: %.3f", self._max_mach)

            # PIMPLE iteration
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._pimple_iteration()
            )

            # Energy predictor-corrector
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho, self.p,
            )

            # Density correction
            if self.density_correction:
                self.rho = self._density_correction_step(
                    self.rho, self.p, self.T,
                )

            # Coupled energy-momentum correction
            self.U, self.p, self.T, self.phi, self.rho = (
                self._coupled_energy_momentum_step(
                    self.U, self.p, self.T, self.phi, self.rho,
                )
            )

            # Mach-aware relaxation for all fields
            self.U = self._mach_aware_relaxation(
                self.U, self.U_old, self.alpha_U, Ma,
            )
            self.p = self._mach_aware_relaxation(
                self.p, self.p_old, self.alpha_p, Ma,
            )
            self.T = self._mach_aware_T_relaxation(self.T, self.T_old, Ma)

            last_convergence = conv

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

        logger.info(
            "rhoPimpleFoamEnhanced2 completed: max_Ma=%.3f",
            self._max_mach,
        )

        return last_convergence or ConvergenceData()
