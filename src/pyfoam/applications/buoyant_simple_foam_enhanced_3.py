"""
buoyantSimpleFoamEnhanced3 — enhanced steady-state buoyant SIMPLE solver v3.

Extends :class:`BuoyantSimpleFoamEnhanced2` with:

- **Improved Boussinesq approximation**: uses a second-order
  linearisation of the buoyancy source that includes both the
  linear and quadratic temperature correction terms.
- **Multi-level Richardson number**: classifies the flow into
  forced, mixed, and natural convection regimes using both bulk
  and gradient Richardson numbers, with regime-specific relaxation
  strategies.
- **Buoyancy-driven pressure correction**: modifies the pressure
  equation to include a buoyancy-induced correction that accelerates
  convergence for pure natural convection problems.

Algorithm (per outer iteration):
1. Update turbulence
2. Classify flow regime (bulk + gradient Richardson)
3. Solve momentum predictor with second-order Boussinesq buoyancy
4. Solve pressure equation (with buoyancy-driven correction)
5. Correct velocity and flux
6. Solve energy equation
7. Update density and check convergence

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3

    solver = BuoyantSimpleFoamEnhanced3("path/to/case", quadratic_boussinesq=True)
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

from .buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced3"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced3(BuoyantSimpleFoamEnhanced2):
    """Enhanced steady-state buoyant SIMPLE solver v3.

    Extends BuoyantSimpleFoamEnhanced2 with second-order Boussinesq,
    multi-level Richardson classification, and buoyancy-driven
    pressure correction.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector (m/s^2).
    radiation : RadiationModel, optional
        Radiation model.
    quadratic_boussinesq : bool, optional
        Include quadratic temperature term in Boussinesq.  Default False.
    buoyancy_pressure_correction : bool, optional
        Enable buoyancy-driven pressure correction.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        quadratic_boussinesq: bool = False,
        buoyancy_pressure_correction: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.quadratic_boussinesq = quadratic_boussinesq
        self.buoyancy_pressure_correction = buoyancy_pressure_correction

        # Flow regime classification
        self._flow_regime = "forced"  # forced, mixed, natural

        logger.info(
            "BuoyantSimpleFoamEnhanced3 ready: quadratic=%s, buoy_pressure=%s",
            self.quadratic_boussinesq, self.buoyancy_pressure_correction,
        )

    # ------------------------------------------------------------------
    # Multi-level Richardson classification
    # ------------------------------------------------------------------

    def _classify_flow_regime(
        self,
        Ri_bulk: float,
        Ri_gradient_field: torch.Tensor,
    ) -> str:
        """Classify the flow regime using bulk and gradient Richardson.

        - Ri_bulk < 0.1: forced convection (shear-dominated)
        - 0.1 <= Ri_bulk < 10: mixed convection
        - Ri_bulk >= 10: natural convection (buoyancy-dominated)

        Parameters
        ----------
        Ri_bulk : float
            Bulk Richardson number.
        Ri_gradient_field : torch.Tensor
            Gradient Richardson number field.

        Returns:
            Flow regime string: "forced", "mixed", or "natural".
        """
        Ri_grad_max = float(Ri_gradient_field.max().item())
        Ri_combined = max(Ri_bulk, Ri_grad_max)

        if Ri_combined < 0.1:
            return "forced"
        elif Ri_combined < 10.0:
            return "mixed"
        else:
            return "natural"

    # ------------------------------------------------------------------
    # Second-order Boussinesq buoyancy
    # ------------------------------------------------------------------

    def _compute_quadratic_boussinesq(
        self,
        T: torch.Tensor,
        rho0: float,
    ) -> torch.Tensor:
        """Compute quadratic Boussinesq buoyancy force.

        F_b = rho0 * beta * (T_ref - T) * g
            + rho0 * beta^2 * (T_ref - T)^2 * g  [quadratic term]

        The quadratic term improves accuracy for large temperature
        differences where the linear approximation breaks down.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        rho0 : float
            Reference density.

        Returns:
            ``(n_cells, 3)`` buoyancy force.
        """
        T_diff = T - self.T_ref

        # Linear term
        F_linear = -rho0 * self.beta * T_diff.unsqueeze(-1) * self.g.unsqueeze(0)

        if not self.quadratic_boussinesq:
            return F_linear

        # Quadratic correction
        F_quad = rho0 * self.beta ** 2 * T_diff.pow(2).unsqueeze(-1) * self.g.unsqueeze(0)

        return F_linear + F_quad

    # ------------------------------------------------------------------
    # Buoyancy-driven pressure correction
    # ------------------------------------------------------------------

    def _buoyancy_pressure_correction(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Apply buoyancy-driven correction to pressure.

        In natural convection, the pressure field contains a dominant
        hydrostatic component.  Subtracting this component and
        correcting for it improves the conditioning of the pressure
        equation.

        p_corrected = p - p_hydrostatic(T)

        Parameters
        ----------
        p : torch.Tensor
            Current pressure.
        T : torch.Tensor
            Temperature field.

        Returns:
            Corrected pressure.
        """
        if not self.buoyancy_pressure_correction:
            return p

        p_hydro = self._compute_hydrostatic_pressure(T)
        return p - p_hydro

    # ------------------------------------------------------------------
    # Regime-specific relaxation
    # ------------------------------------------------------------------

    def _regime_relaxation(
        self,
        regime: str,
        alpha_U_base: float,
        alpha_p_base: float,
    ) -> tuple[float, float]:
        """Get regime-specific relaxation factors.

        - Forced: standard relaxation
        - Mixed: slightly reduced for stability
        - Natural: heavily reduced U relaxation, increased p relaxation

        Parameters
        ----------
        regime : str
            Flow regime.
        alpha_U_base, alpha_p_base : float
            Base relaxation factors.

        Returns:
            Tuple of (alpha_U, alpha_p).
        """
        if regime == "forced":
            return alpha_U_base, alpha_p_base
        elif regime == "mixed":
            return alpha_U_base * 0.9, alpha_p_base * 0.95
        else:  # natural
            return alpha_U_base * 0.7, alpha_p_base * 1.1

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v3 buoyantSimpleFoam solver.

        Uses second-order Boussinesq, multi-level Richardson
        classification, and buoyancy-driven pressure correction.

        Returns:
            Final :class:`ConvergenceData`.
        """
        use_boussinesq = self._should_use_boussinesq()
        if use_boussinesq:
            logger.info("Using Boussinesq approximation (beta=%.4e)", self.beta)
        else:
            logger.info("Using variable-density mode")

        rho0 = float(self.rho.mean().item())

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

        logger.info("Starting buoyantSimpleFoamEnhanced3 run")
        logger.info("  quadratic=%s, buoyancy_pressure=%s",
                     self.quadratic_boussinesq, self.buoyancy_pressure_correction)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            mu_eff = self._update_turbulence()

            # Richardson numbers
            Ri = self._compute_richardson_number(self.U, self.T)
            self._richardson = Ri
            Ri_field = self._compute_gradient_richardson_field(self.U, self.T)

            # Classify flow regime
            self._flow_regime = self._classify_flow_regime(Ri, Ri_field)

            if step % 10 == 0:
                logger.info(
                    "Ri=%.3f, regime=%s, Ri_grad_max=%.3f",
                    Ri, self._flow_regime, float(Ri_field.max().item()),
                )

            # Regime-specific relaxation
            alpha_U_eff, alpha_p_eff = self._regime_relaxation(
                self._flow_regime, self.alpha_U, self.alpha_p,
            )

            # Quadratic Boussinesq
            if use_boussinesq:
                F_buoyancy = self._compute_quadratic_boussinesq(self.T, rho0)

            # SIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_simple_iteration(mu_eff=mu_eff)
            )

            # Buoyancy pressure correction
            self.p = self._buoyancy_pressure_correction(self.p, self.T)

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

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantSimpleFoamEnhanced3 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced3 completed without convergence")

        return last_convergence or ConvergenceData()
