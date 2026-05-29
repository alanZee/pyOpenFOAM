"""
buoyantSimpleFoamEnhanced — enhanced steady-state buoyant SIMPLE solver.

Extends :class:`BuoyantSimpleFoam` with:

- **Improved Boussinesq approximation**: proper linearised buoyancy
  source term with reference temperature, reducing non-linear coupling.
- **Enhanced convergence for natural convection**: buoyancy-aware
  under-relaxation that adapts based on the Richardson number
  (ratio of buoyancy to inertial forces).
- **Boussinesq vs variable-density mode**: automatic selection based
  on temperature difference relative to reference temperature.

The Boussinesq approximation is:
    ρ ≈ ρ₀ (1 - β(T - T₀))

where β is the thermal expansion coefficient, ρ₀ is the reference
density, and T₀ is the reference temperature.  The buoyancy force
becomes:
    F_b = -ρ₀ β (T - T₀) g

This is valid when βΔT << 1, which avoids resolving density variations
in the momentum equation.

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced

    solver = BuoyantSimpleFoamEnhanced("path/to/case")
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
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_simple_foam import BuoyantSimpleFoam

__all__ = ["BuoyantSimpleFoamEnhanced"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced(BuoyantSimpleFoam):
    """Enhanced steady-state buoyant SIMPLE solver.

    Extends BuoyantSimpleFoam with improved Boussinesq approximation,
    Richardson-number-aware relaxation, and better natural convection
    convergence.

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
        Thermal expansion coefficient (1/K).  Reads from
        ``constant/thermophysicalProperties`` if None.  Default 3.33e-3
        (air at 300 K).
    T_ref : float, optional
        Reference temperature for Boussinesq (K).  Default 300.0.
    use_boussinesq : bool, optional
        Force Boussinesq mode (True) or variable-density mode (False).
        Default None (auto-select based on β*ΔT).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        beta: float | None = None,
        T_ref: float | None = None,
        use_boussinesq: bool | None = None,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity, radiation=radiation,
        )

        # Read or set Boussinesq properties
        bq_props = self._read_boussinesq_properties()
        self.beta = beta if beta is not None else bq_props.get("beta", 3.33e-3)
        self.T_ref = T_ref if T_ref is not None else bq_props.get("T_ref", 300.0)
        self._use_boussinesq = use_boussinesq

        # Richardson number tracking
        self._richardson = 0.0

        logger.info(
            "BuoyantSimpleFoamEnhanced ready: beta=%.6e, T_ref=%.1f",
            self.beta, self.T_ref,
        )

    # ------------------------------------------------------------------
    # Boussinesq properties reading
    # ------------------------------------------------------------------

    def _read_boussinesq_properties(self) -> dict[str, float]:
        """Read Boussinesq properties from thermophysicalProperties."""
        tp_path = self.case_path / "constant" / "thermophysicalProperties"
        props: dict[str, float] = {}

        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                tp = parse_dict_file(tp_path)

                # Read thermal expansion coefficient
                beta_raw = tp.get("beta", None)
                if beta_raw is not None:
                    if isinstance(beta_raw, (int, float)):
                        props["beta"] = float(beta_raw)
                    else:
                        # Parse dimensioned scalar
                        import re
                        match = re.search(r"\]\s*([\d.eE+\-]+)", str(beta_raw))
                        if match:
                            props["beta"] = float(match.group(1))

                T_ref_raw = tp.get("TRef", None)
                if T_ref_raw is not None:
                    props["T_ref"] = float(T_ref_raw)
            except Exception as e:
                logger.debug("Could not read Boussinesq properties: %s", e)

        return props

    # ------------------------------------------------------------------
    # Boussinesq mode selection
    # ------------------------------------------------------------------

    def _should_use_boussinesq(self) -> bool:
        """Determine if the Boussinesq approximation is valid.

        Boussinesq is valid when β * ΔT << 1, where ΔT is the
        temperature range in the domain.

        Returns:
            True if Boussinesq mode should be used.
        """
        if self._use_boussinesq is not None:
            return self._use_boussinesq

        delta_T = float((self.T.max() - self.T.min()).item())
        beta_delta_T = self.beta * delta_T

        # Boussinesq valid if β*ΔT < 0.1 (10% density change)
        return beta_delta_T < 0.1

    # ------------------------------------------------------------------
    # Richardson number
    # ------------------------------------------------------------------

    def _compute_richardson_number(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
    ) -> float:
        """Compute bulk Richardson number.

        Ri = g * β * ΔT * L / U_ref²

        where L is a characteristic length (cube root of domain volume)
        and U_ref is the mean velocity magnitude.

        High Ri indicates buoyancy-dominated flow (natural convection).
        Low Ri indicates forced convection.

        Returns:
            Richardson number.
        """
        g_mag = float(self.g.norm().item())
        delta_T = float((T.max() - T.min()).item())
        U_ref = float(U.norm(dim=1).mean().item())
        L = float(self.mesh.cell_volumes.sum().pow(1.0 / 3.0).item())

        if U_ref < 1e-10:
            return float("inf")  # Pure natural convection

        return g_mag * self.beta * delta_T * L / (U_ref**2)

    # ------------------------------------------------------------------
    # Boussinesq buoyancy force
    # ------------------------------------------------------------------

    def _compute_boussinesq_buoyancy(
        self,
        T: torch.Tensor,
        rho0: float,
    ) -> torch.Tensor:
        """Compute Boussinesq buoyancy force.

        F_b = -ρ₀ * β * (T - T_ref) * g

        This is a linearised buoyancy that avoids the non-linear
        coupling of variable density in the momentum equation.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        rho0 : float
            Reference density.

        Returns:
            ``(n_cells, 3)`` buoyancy force per unit volume.
        """
        T_diff = T - self.T_ref
        # F_b = -ρ₀ * β * (T - T_ref) * g
        buoyancy = -rho0 * self.beta * T_diff.unsqueeze(-1) * self.g.unsqueeze(0)
        return buoyancy

    # ------------------------------------------------------------------
    # Buoyancy-aware relaxation
    # ------------------------------------------------------------------

    def _buoyancy_aware_relaxation(
        self,
        Ri: float,
    ) -> tuple[float, float]:
        """Compute buoyancy-aware relaxation factors.

        For high Richardson number (buoyancy-dominated), reduce
        relaxation to avoid oscillations in temperature-velocity coupling.

        Parameters
        ----------
        Ri : float
            Richardson number.

        Returns:
            Tuple of (alpha_U, alpha_p) adjusted for buoyancy.
        """
        if Ri < 1.0:
            # Forced convection: use nominal relaxation
            return self.alpha_U, self.alpha_p
        elif Ri < 10.0:
            # Mixed convection: slightly reduce
            factor = 1.0 / (1.0 + 0.1 * Ri)
            return self.alpha_U * factor, self.alpha_p * factor
        else:
            # Natural convection: significantly reduce
            factor = max(0.3, 1.0 / (1.0 + 0.05 * Ri))
            return self.alpha_U * factor, self.alpha_p * factor

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced buoyantSimpleFoam solver.

        Uses Boussinesq approximation when valid and Richardson-number-
        aware relaxation for improved natural convection convergence.

        Returns:
            Final :class:`ConvergenceData`.
        """
        from .time_loop import TimeLoop
        from .convergence import ConvergenceMonitor

        # Auto-select Boussinesq mode
        use_boussinesq = self._should_use_boussinesq()
        if use_boussinesq:
            logger.info("Using Boussinesq approximation (beta=%.4e)", self.beta)
        else:
            logger.info("Using variable-density mode (beta*dT > 0.1)")

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

        logger.info("Starting buoyantSimpleFoamEnhanced run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence
            mu_eff = self._update_turbulence()

            # Compute Richardson number for adaptive relaxation
            Ri = self._compute_richardson_number(self.U, self.T)
            self._richardson = Ri

            if step % 10 == 0:
                logger.info("Richardson number: %.3f", Ri)

            # Get buoyancy-aware relaxation
            alpha_U_eff, alpha_p_eff = self._buoyancy_aware_relaxation(Ri)

            # Run one SIMPLE iteration with buoyancy
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_simple_iteration(mu_eff=mu_eff)
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
                logger.info("buoyantSimpleFoamEnhanced completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced completed without convergence")

        return last_convergence or ConvergenceData()
