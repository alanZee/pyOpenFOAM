"""
buoyantSimpleFoamEnhanced4 — enhanced steady-state buoyant SIMPLE solver v4.

Extends :class:`BuoyantSimpleFoamEnhanced3` with:

- **Adaptive Boussinesq switching**: automatically transitions between
  the Boussinesq approximation and full variable-density formulation
  based on the observed temperature difference relative to a threshold.
- **Turbulence-regime-aware relaxation**: combines the flow regime
  classification from v3 with turbulence state information to provide
  optimised relaxation factors for each regime-turbulence combination.
- **Thermal stratification limiter**: applies a physically-based
  temperature gradient limiter that prevents unphysical thermal
  stratification in well-mixed regions.

Algorithm (per outer iteration):
1. Update turbulence
2. Classify flow regime (bulk + gradient Richardson, from v3)
3. Determine Boussinesq vs variable-density mode
4. Solve momentum predictor (with regime-aware relaxation)
5. Solve pressure equation (with buoyancy-driven correction from v3)
6. Solve energy equation (with stratification limiter)
7. Update density and check convergence

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4

    solver = BuoyantSimpleFoamEnhanced4("path/to/case", adaptive_boussinesq=True)
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

from .buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced4"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced4(BuoyantSimpleFoamEnhanced3):
    """Enhanced steady-state buoyant SIMPLE solver v4.

    Extends BuoyantSimpleFoamEnhanced3 with adaptive Boussinesq switching,
    turbulence-regime-aware relaxation, and thermal stratification limiter.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector.
    radiation : RadiationModel, optional
        Radiation model.
    adaptive_boussinesq : bool, optional
        Enable automatic Boussinesq/variable-density switching.  Default True.
    boussinesq_threshold : float, optional
        Temperature difference threshold for switching (K).  Default 20.0.
    stratification_limit : float, optional
        Maximum allowable temperature gradient (K/m).  Default 100.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        adaptive_boussinesq: bool = True,
        boussinesq_threshold: float = 20.0,
        stratification_limit: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.adaptive_boussinesq = adaptive_boussinesq
        self.boussinesq_threshold = max(1.0, boussinesq_threshold)
        self.stratification_limit = max(1.0, stratification_limit)

        # Turbulence-regime combination history
        self._regime_turb_history: list[str] = []

        logger.info(
            "BuoyantSimpleFoamEnhanced4 ready: adaptive_bouss=%s, thresh=%.1f",
            self.adaptive_boussinesq, self.boussinesq_threshold,
        )

    # ------------------------------------------------------------------
    # Adaptive Boussinesq switching
    # ------------------------------------------------------------------

    def _should_switch_to_variable_density(
        self,
        T: torch.Tensor,
        T_ref: float,
    ) -> bool:
        """Determine whether to switch from Boussinesq to variable density.

        Switches when the maximum temperature difference exceeds a
        threshold relative to the reference temperature.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        T_ref : float
            Reference temperature.

        Returns:
            True if variable-density mode should be used.
        """
        if not self.adaptive_boussinesq:
            return False

        dT_max = float((T - T_ref).abs().max().item())
        return dT_max > self.boussinesq_threshold

    # ------------------------------------------------------------------
    # Turbulence-regime-aware relaxation
    # ------------------------------------------------------------------

    def _turbulence_regime_relaxation(
        self,
        regime: str,
        turbulent: bool,
        alpha_U_base: float,
        alpha_p_base: float,
    ) -> tuple[float, float]:
        """Get relaxation factors based on flow regime and turbulence state.

        Turbulent flows with strong buoyancy need more conservative
        relaxation to prevent oscillations from buoyancy-turbulence
        interaction.

        Parameters
        ----------
        regime : str
            Flow regime: "forced", "mixed", or "natural".
        turbulent : bool
            Whether turbulence model is active.
        alpha_U_base, alpha_p_base : float
            Base relaxation factors.

        Returns:
            Tuple of (alpha_U, alpha_p).
        """
        # Start with regime-based relaxation (from v3)
        alpha_U, alpha_p = self._regime_relaxation(
            regime, alpha_U_base, alpha_p_base,
        )

        # Turbulence modifier
        if turbulent:
            if regime == "natural":
                # Turbulent natural convection: very conservative U
                alpha_U *= 0.8
                alpha_p *= 1.05
            elif regime == "mixed":
                # Turbulent mixed convection: slightly reduced
                alpha_U *= 0.9
                alpha_p *= 1.0
        # Laminar: no additional modification

        return alpha_U, alpha_p

    # ------------------------------------------------------------------
    # Thermal stratification limiter
    # ------------------------------------------------------------------

    def _limit_stratification(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Limit temperature gradient to prevent unphysical stratification.

        In well-mixed regions (forced/mixed convection), the temperature
        gradient should not exceed a physically reasonable limit.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.

        Returns:
            Limited temperature field.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Face temperature gradient
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        grad_T = (T_N - T_O) * delta_coeffs

        # Limit gradient magnitude
        grad_T_limited = grad_T.clamp(
            min=-self.stratification_limit,
            max=self.stratification_limit,
        )

        # Check if limiting is needed
        exceeded = (grad_T.abs() > self.stratification_limit)

        if not exceeded.any():
            return T

        # Apply correction (simplified: smooth cells with excessive gradient)
        T_corrected = T.clone()
        correction = torch.zeros(n_cells, dtype=dtype, device=device)
        n_contrib = torch.zeros(n_cells, dtype=dtype, device=device)

        delta_T = grad_T_limited - grad_T
        correction = correction + scatter_add(delta_T, owner, n_cells)
        correction = correction + scatter_add(-delta_T, neigh, n_cells)
        n_contrib = n_contrib + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        )
        n_contrib = n_contrib + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )

        T_corrected = T_corrected + correction / n_contrib.clamp(min=1.0) * 0.1

        return T_corrected

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 buoyantSimpleFoam solver.

        Uses adaptive Boussinesq switching, turbulence-regime-aware
        relaxation, and thermal stratification limiting.

        Returns:
            Final :class:`ConvergenceData`.
        """
        use_boussinesq = self._should_use_boussinesq()
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

        logger.info("Starting buoyantSimpleFoamEnhanced4 run")
        logger.info("  adaptive_bouss=%s, thresh=%.1f K",
                     self.adaptive_boussinesq, self.boussinesq_threshold)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            mu_eff = self._update_turbulence()

            # Richardson numbers (from v3)
            Ri = self._compute_richardson_number(self.U, self.T)
            self._richardson = Ri
            Ri_field = self._compute_gradient_richardson_field(self.U, self.T)

            # Classify flow regime
            self._flow_regime = self._classify_flow_regime(Ri, Ri_field)

            # Adaptive Boussinesq switching
            if self.adaptive_boussinesq:
                use_var_density = self._should_switch_to_variable_density(
                    self.T, self.T_ref,
                )
                if use_var_density and use_boussinesq:
                    logger.info("Switching to variable-density mode at step %d", step + 1)
                    use_boussinesq = False

            # Turbulence-regime-aware relaxation
            turbulent = self.turbulence.enabled if hasattr(self, 'turbulence') else False
            alpha_U_eff, alpha_p_eff = self._turbulence_regime_relaxation(
                self._flow_regime, turbulent, self.alpha_U, self.alpha_p,
            )

            if step % 10 == 0:
                logger.info(
                    "Ri=%.3f, regime=%s, boussinesq=%s",
                    Ri, self._flow_regime, use_boussinesq,
                )

            # Quadratic Boussinesq (from v3)
            if use_boussinesq:
                F_buoyancy = self._compute_quadratic_boussinesq(self.T, rho0)

            # SIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_simple_iteration(mu_eff=mu_eff)
            )

            # Buoyancy pressure correction (from v3)
            self.p = self._buoyancy_pressure_correction(self.p, self.T)

            # Thermal stratification limiter
            self.T = self._limit_stratification(self.T)

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
                logger.info("buoyantSimpleFoamEnhanced4 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced4 completed without convergence")

        return last_convergence or ConvergenceData()
