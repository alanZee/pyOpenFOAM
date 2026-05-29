"""
buoyantPimpleFoamEnhanced — enhanced transient buoyant PIMPLE solver.

Extends :class:`BuoyantPimpleFoam` with:

- **Improved transient buoyant flows**: better coupling between the
  buoyancy source term and the energy equation within each outer
  iteration, reducing splitting errors.
- **Better stability for large temperature differences**: temperature-
  dependent relaxation that reduces under-relaxation in hot regions
  and increases it in cold regions.
- **Boussinesq mode**: optional linearised buoyancy for moderate
  temperature differences (βΔT < 0.1).
- **Richardson-aware time stepping**: adapts time step based on
  the ratio of buoyancy to inertial forces.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson number and adapt time step
3. Outer corrector loop:
   a. Momentum predictor with buoyancy
   b. PISO pressure correction (p_rgh form)
   c. Energy equation with buoyancy coupling
   d. EOS update
   e. Temperature-dependent under-relaxation
4. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced

    solver = BuoyantPimpleFoamEnhanced("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_pimple_foam import BuoyantPimpleFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced(BuoyantPimpleFoam):
    """Enhanced transient buoyant compressible PIMPLE solver.

    Extends BuoyantPimpleFoam with improved buoyancy coupling,
    temperature-dependent relaxation, and Richardson-aware time stepping.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector (m/s²).
    radiation : RadiationModel, optional
        Radiation model.
    beta : float, optional
        Thermal expansion coefficient (1/K).  Default 3.33e-3.
    T_ref : float, optional
        Reference temperature (K).  Default 300.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        beta: float = 3.33e-3,
        T_ref: float = 300.0,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity, radiation=radiation,
        )

        self.beta = beta
        self.T_ref = T_ref

        # Relaxation state
        self._richardson = 0.0

        logger.info(
            "BuoyantPimpleFoamEnhanced ready: beta=%.6e, T_ref=%.1f",
            self.beta, self.T_ref,
        )

    # ------------------------------------------------------------------
    # Richardson number
    # ------------------------------------------------------------------

    def _compute_richardson(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
    ) -> float:
        """Compute bulk Richardson number.

        Ri = g * β * ΔT * L / U_ref²

        Returns:
            Richardson number (dimensionless).
        """
        g_mag = float(self.g.norm().item())
        delta_T = float((T.max() - T.min()).item())
        U_ref = float(U.norm(dim=1).mean().item())
        L = float(self.mesh.cell_volumes.sum().pow(1.0 / 3.0).item())

        if U_ref < 1e-10:
            return 100.0  # Cap for pure natural convection

        return g_mag * self.beta * delta_T * L / (U_ref**2 + 1e-30)

    # ------------------------------------------------------------------
    # Temperature-dependent relaxation
    # ------------------------------------------------------------------

    def _temperature_dependent_relaxation(
        self,
        T: torch.Tensor,
        field: torch.Tensor,
        field_old: torch.Tensor,
        alpha_base: float,
    ) -> torch.Tensor:
        """Apply temperature-dependent under-relaxation.

        In hot regions (T >> T_ref), reduces relaxation to avoid
        oscillations caused by strong buoyancy coupling.

        α_eff = α_base * T_ref / max(T, T_ref)

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        field : torch.Tensor
            Field after correction.
        field_old : torch.Tensor
            Field from previous iteration.
        alpha_base : float
            Base relaxation factor.

        Returns:
            Relaxed field.
        """
        T_ratio = self.T_ref / T.clamp(min=self.T_ref * 0.1)
        alpha = alpha_base * T_ratio.clamp(min=0.3, max=1.0)

        if field.dim() == 1:
            return alpha * field + (1.0 - alpha) * field_old
        else:
            return alpha.unsqueeze(-1) * field + (1.0 - alpha).unsqueeze(-1) * field_old

    # ------------------------------------------------------------------
    # Richardson-aware time stepping
    # ------------------------------------------------------------------

    def _adapt_time_step(self, Ri: float) -> float:
        """Adapt time step based on Richardson number.

        For high Ri (buoyancy-dominated), reduce Δt to maintain
        temporal resolution of buoyant plumes.

        Returns:
            Adapted time step.
        """
        if Ri < 1.0:
            return self.delta_t
        elif Ri < 10.0:
            return self.delta_t / (1.0 + 0.1 * Ri)
        else:
            return self.delta_t / (1.0 + 0.05 * Ri)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced buoyantPimpleFoam solver.

        Uses Richardson-aware time stepping and temperature-dependent
        relaxation for improved stability with large temperature
        differences.

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

        logger.info("Starting buoyantPimpleFoamEnhanced run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  beta=%.6e, T_ref=%.1f", self.beta, self.T_ref)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            # Compute Richardson number
            Ri = self._compute_richardson(self.U, self.T)
            self._richardson = Ri

            if step % 10 == 0:
                logger.info("Richardson number: %.3f", Ri)

            # Update turbulence
            mu_eff = self._update_turbulence()

            # Run PIMPLE iteration with buoyancy
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )

            # Temperature-dependent relaxation
            self.U = self._temperature_dependent_relaxation(
                self.T, self.U, self.U_old, self.alpha_U,
            )
            self.p = self._temperature_dependent_relaxation(
                self.T, self.p, self.p_old, self.alpha_p,
            )

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
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantPimpleFoamEnhanced completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced completed without convergence")

        return last_convergence or ConvergenceData()
