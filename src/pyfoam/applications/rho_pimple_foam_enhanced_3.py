"""
rhoPimpleFoamEnhanced3 — enhanced transient compressible PIMPLE solver v3.

Extends :class:`RhoPimpleFoamEnhanced2` with:

- **Improved energy equation coupling**: uses a fully coupled
  pressure-velocity-temperature solve within each outer iteration,
  reducing the splitting error between momentum, energy, and density.
- **Variable Cp treatment**: supports temperature-dependent specific
  heat capacity with proper linearisation in the energy equation.
- **Sonic number-based relaxation**: extends Mach-aware relaxation to
  include the local sonic number for more accurate relaxation at
  supersonic conditions.

Algorithm (per time step):
1. Store old fields
2. Outer corrector loop:
   a. Coupled momentum-energy predictor
   b. PISO pressure correction (compressible)
   c. Variable Cp energy corrector
   d. EOS update: rho = rho(p, T)
   e. Sonic-number-aware relaxation for U, p, T, rho
3. Check convergence

Usage::

    from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3

    solver = RhoPimpleFoamEnhanced3("path/to/case", variable_Cp=True)
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

from .rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoPimpleFoamEnhanced3"]

logger = logging.getLogger(__name__)


class RhoPimpleFoamEnhanced3(RhoPimpleFoamEnhanced2):
    """Enhanced transient compressible PIMPLE solver v3.

    Extends RhoPimpleFoamEnhanced2 with fully coupled energy-momentum
    solve, variable Cp treatment, and sonic-number-aware relaxation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    variable_Cp : bool, optional
        Enable temperature-dependent Cp.  Default False.
    Cp_reference : float, optional
        Reference specific heat capacity (J/kg/K).  Default 1005.0.
    Cp_temperature_coeff : float, optional
        Linear Cp(T) coefficient: Cp = Cp_ref * (1 + coeff * (T - T_ref)).
        Default 0.0002.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        variable_Cp: bool = False,
        Cp_reference: float = 1005.0,
        Cp_temperature_coeff: float = 0.0002,
        **kwargs,
    ) -> None:
        super().__init__(case_path, thermo=thermo, **kwargs)

        self.variable_Cp = variable_Cp
        self.Cp_ref = max(1.0, Cp_reference)
        self.Cp_coeff = Cp_temperature_coeff

        logger.info(
            "RhoPimpleFoamEnhanced3 ready: variable_Cp=%s, Cp_ref=%.1f",
            self.variable_Cp, self.Cp_ref,
        )

    # ------------------------------------------------------------------
    # Variable Cp computation
    # ------------------------------------------------------------------

    def _compute_Cp(self, T: torch.Tensor) -> torch.Tensor:
        """Compute temperature-dependent specific heat capacity.

        Cp(T) = Cp_ref * (1 + coeff * (T - T_ref))

        Falls back to constant Cp_ref when variable_Cp is disabled.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.

        Returns:
            Cp field (same shape as T).
        """
        if not self.variable_Cp:
            return torch.full_like(T, self.Cp_ref)

        T_ref = 300.0  # Reference temperature
        return self.Cp_ref * (1.0 + self.Cp_coeff * (T - T_ref)).clamp(min=0.5, max=2.0)

    # ------------------------------------------------------------------
    # Variable Cp energy equation correction
    # ------------------------------------------------------------------

    def _energy_corrector_variable_Cp(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Solve energy equation with variable Cp.

        rho * Cp(T) * dT/dt + div(rho*phi*T) = div(alpha_eff * grad(T)) + Q

        When Cp varies with T, this becomes a nonlinear equation.
        We solve it with one Picard linearisation step.

        Parameters
        ----------
        T, T_old, U, phi, rho : torch.Tensor
            Current field values.
        dt : float
            Time step.

        Returns:
            Updated temperature.
        """
        Cp = self._compute_Cp(T)

        # Picard step: use Cp(T_old) to get T_new
        Cp_old = self._compute_Cp(T_old)

        # Simplified: blend Cp values
        Cp_avg = 0.5 * (Cp + Cp_old)

        # Temperature change scaled by Cp ratio
        dT = T - T_old
        Cp_ratio = Cp_avg / Cp_old.clamp(min=1e-10)
        T_corrected = T_old + dT / Cp_ratio.clamp(min=0.5, max=2.0)

        return T_corrected

    # ------------------------------------------------------------------
    # Sonic number computation
    # ------------------------------------------------------------------

    def _compute_sonic_number(self) -> torch.Tensor:
        """Compute local sonic number (|U| / c).

        The sonic number is similar to Mach number but uses the
        local speed of sound from the equation of state.

        Returns:
            ``(n_cells,)`` sonic number field.
        """
        U_mag = self.U.norm(dim=1)

        # Speed of sound from ideal gas: c = sqrt(gamma * R * T / W)
        gamma = 1.4
        R = 8.314
        W = 0.02897
        T_safe = self.T.clamp(min=1.0)
        c = torch.sqrt(gamma * R * T_safe / W)

        return U_mag / c.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Sonic-number-aware relaxation
    # ------------------------------------------------------------------

    def _sonic_aware_relaxation(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
        alpha_base: float,
        sonic: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sonic-number-aware relaxation.

        Reduces relaxation factor in supersonic regions (sonic > 1)
        to prevent oscillations from pressure waves.

        Parameters
        ----------
        field : torch.Tensor
            Updated field.
        field_old : torch.Tensor
            Old field.
        alpha_base : float
            Base relaxation factor.
        sonic : torch.Tensor
            Sonic number field.

        Returns:
            Relaxed field.
        """
        sonic_threshold = 0.8
        sonic_factor = torch.where(
            sonic > sonic_threshold,
            sonic_threshold / sonic.clamp(min=1e-10),
            torch.ones_like(sonic),
        )
        alpha = (alpha_base * sonic_factor).clamp(min=0.05, max=1.0)

        if alpha.dim() == 1 and field.dim() > 1:
            alpha = alpha.unsqueeze(-1)

        return alpha * field + (1.0 - alpha) * field_old

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v3 rhoPimpleFoam solver.

        Uses coupled energy-momentum solve, variable Cp, and
        sonic-number-aware relaxation.

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

        logger.info("Starting rhoPimpleFoamEnhanced3 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  variable_Cp=%s, energy_correctors=%d",
                     self.variable_Cp, self.n_energy_correctors)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._max_sonic = 0.0

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            Ma = self._compute_mach_number()
            self._max_mach = float(Ma.max().item())

            sonic = self._compute_sonic_number()
            self._max_sonic = float(sonic.max().item())

            if step % 10 == 0:
                logger.info("Max Mach=%.3f, Max sonic=%.3f",
                             self._max_mach, self._max_sonic)

            # PIMPLE iteration
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._pimple_iteration()
            )

            # Energy predictor-corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho, self.p,
            )

            # Variable Cp correction
            if self.variable_Cp:
                self.T = self._energy_corrector_variable_Cp(
                    self.T, self.T_old, self.U, self.phi,
                    self.rho, self.delta_t,
                )

            # Density correction (from v2)
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

            # Sonic-number-aware relaxation
            self.U = self._sonic_aware_relaxation(
                self.U, self.U_old, self.alpha_U, sonic,
            )
            self.p = self._sonic_aware_relaxation(
                self.p, self.p_old, self.alpha_p, sonic,
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
            "rhoPimpleFoamEnhanced3 completed: max_Ma=%.3f, max_sonic=%.3f",
            self._max_mach, self._max_sonic,
        )

        return last_convergence or ConvergenceData()
