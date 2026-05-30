"""Enhanced JANAF thermodynamic model v10 — collision integral coupling, phase stability, and mixture flash.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_9.JanafMultiThermoEnhanced9`
with:

- Collision integral coupling for high-temperature transport-thermo consistency
- Phase stability analysis using spinodal decomposition criteria
- Multi-component mixture flash with Rachford-Rice formulation

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_10 import JanafMultiThermoEnhanced10
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced10(R=461.5, phases=phases, blend_width=5.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_9 import JanafMultiThermoEnhanced9
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced10"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)


class JanafMultiThermoEnhanced10(JanafMultiThermoEnhanced9):
    """Enhanced multi-phase JANAF v10 with collision integral coupling and phase stability.

    Extends :class:`JanafMultiThermoEnhanced9` with:

    - **Collision integral coupling**: links Omega(2,2) collision integrals to
      JANAF thermo for transport-thermo consistency at high temperatures.
    - **Phase stability analysis**: spinodal decomposition criterion to detect
      thermodynamic instability regions.
    - **Rachford-Rice flash**: multi-component VLE flash using the Rachford-Rice
      equation for improved convergence.

    Parameters
    ----------
    R, phases, Hf, blend_width, steepness, beta_coeffs, latent_sensible_fraction :
        See parent.
    S_ref, T_ref, S_ref_table, Cp_extrapolate_power, T_extrapolate_start :
        See parent.
    gibbs_max_iter, gibbs_tol : See parent.
    radiation_emissivity, radiation_area_factor, flash_max_iter, flash_tol : See parent.
    collision_integral_coeff : float
        Coefficient for collision integral correction. Default 0.0 (disabled).
    stability_check : bool
        Enable phase stability analysis. Default False.
    rachford_rice_max_iter : int
        Maximum iterations for Rachford-Rice flash. Default 30.
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
        blend_width: float = 0.0,
        steepness: float = 2.0,
        beta_coeffs: Sequence[float] | None = None,
        latent_sensible_fraction: float = 0.1,
        S_ref: float = 0.0,
        T_ref: float = 298.15,
        S_ref_table: dict[float, float] | None = None,
        Cp_extrapolate_power: float = 0.5,
        T_extrapolate_start: float = 6000.0,
        gibbs_max_iter: int = 50,
        gibbs_tol: float = 1e-6,
        radiation_emissivity: float = 0.0,
        radiation_area_factor: float = 1.0,
        flash_max_iter: int = 20,
        flash_tol: float = 1e-6,
        collision_integral_coeff: float = 0.0,
        stability_check: bool = False,
        rachford_rice_max_iter: int = 30,
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref, S_ref_table=S_ref_table,
            Cp_extrapolate_power=Cp_extrapolate_power,
            T_extrapolate_start=T_extrapolate_start,
            gibbs_max_iter=gibbs_max_iter, gibbs_tol=gibbs_tol,
            radiation_emissivity=radiation_emissivity,
            radiation_area_factor=radiation_area_factor,
            flash_max_iter=flash_max_iter, flash_tol=flash_tol,
        )
        self._collision_integral_coeff = max(0.0, collision_integral_coeff)
        self._stability_check = stability_check
        self._rr_max_iter = max(1, rachford_rice_max_iter)

    # ------------------------------------------------------------------
    # Collision integral coupling
    # ------------------------------------------------------------------

    def collision_integral_correction(self, T: float) -> float:
        """Collision integral correction factor for transport-thermo coupling.

        Omega(2,2)* ~ 1.0 + coeff * (T_ref / T)^0.25

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Correction factor (dimensionless).
        """
        if self._collision_integral_coeff < 1e-15:
            return 1.0
        T_ref = max(getattr(self, '_T_ref_entropy', 298.15), 1.0)
        T_safe = max(T, 1.0)
        return 1.0 + self._collision_integral_coeff * (T_ref / T_safe) ** 0.25

    # ------------------------------------------------------------------
    # Phase stability analysis
    # ------------------------------------------------------------------

    def is_stable(self, T: float, p: float = 101325.0) -> bool:
        """Check thermodynamic phase stability using spinodal criterion.

        A phase is unstable if d^2G/dT^2 < 0 (negative heat capacity curvature).

        Parameters
        ----------
        T : float
            Temperature (K).
        p : float
            Pressure (Pa). Default 101325.

        Returns
        -------
        bool
            True if phase is thermodynamically stable.
        """
        if not self._stability_check:
            return True

        dT = max(T * 0.001, 0.1)
        Cp_plus = float(self.Cp(T + dT).item()) if hasattr(self.Cp(T + dT), "item") else float(self.Cp(T + dT))
        Cp_minus = float(self.Cp(T - dT).item()) if hasattr(self.Cp(T - dT), "item") else float(self.Cp(T - dT))

        # dCp/dT ~ d^2G/dT^2 * T (simplified)
        dCp_dT = (Cp_plus - Cp_minus) / (2.0 * dT)
        return dCp_dT >= -abs(Cp_plus) * 0.1

    # ------------------------------------------------------------------
    # Rachford-Rice flash
    # ------------------------------------------------------------------

    def rachford_rice_flash(
        self,
        T: float,
        p: float,
        z: Sequence[float],
        K_values: Sequence[float],
    ) -> dict[str, float | list[float]]:
        """Rachford-Rice VLE flash calculation.

        Solves for vapor fraction (beta) and updated compositions.

        Parameters
        ----------
        T : float
            Temperature (K).
        p : float
            Pressure (Pa).
        z : sequence of float
            Overall mole fractions.
        K_values : sequence of float
            Equilibrium K-values for each species.

        Returns
        -------
        dict
            'beta': vapor fraction,
            'x': liquid mole fractions,
            'y': vapor mole fractions,
            'converged': bool.
        """
        n = len(z)
        K = list(K_values)
        z_list = list(z)

        # Find beta bounds
        beta_min = 0.0
        beta_max = 1.0
        for i in range(n):
            if K[i] > 1.0:
                beta_min = max(beta_min, (K[i] * z_list[i] - 1.0) / (K[i] - 1.0))

        beta = (beta_min + beta_max) / 2.0
        converged = False

        for _ in range(self._rr_max_iter):
            # Rachford-Rice function: f(beta) = sum_i z_i*(K_i-1)/(1+beta*(K_i-1))
            f_val = 0.0
            df_val = 0.0
            for i in range(n):
                denom = 1.0 + beta * (K[i] - 1.0)
                if abs(denom) < 1e-30:
                    denom = 1e-30 * math.copysign(1.0, denom)
                f_val += z_list[i] * (K[i] - 1.0) / denom
                df_val -= z_list[i] * (K[i] - 1.0) ** 2 / denom ** 2

            if abs(f_val) < self._flash_tol:
                converged = True
                break

            if abs(df_val) < 1e-30:
                break

            beta_new = beta - f_val / df_val
            beta_new = max(beta_min + 1e-10, min(beta_new, beta_max - 1e-10))
            beta = beta_new

        # Compute compositions
        x = []
        y = []
        for i in range(n):
            denom = 1.0 + beta * (K[i] - 1.0)
            if abs(denom) < 1e-30:
                denom = 1e-30
            x_i = z_list[i] / denom
            y_i = K[i] * x_i
            x.append(max(x_i, 0.0))
            y.append(max(y_i, 0.0))

        # Normalize
        sum_x = sum(x)
        sum_y = sum(y)
        if sum_x > 0:
            x = [xi / sum_x for xi in x]
        if sum_y > 0:
            y = [yi / sum_y for yi in y]

        return {"beta": beta, "x": x, "y": y, "converged": converged}

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        rad = f", eps={self._radiation_emissivity}" if self._radiation_emissivity > 0 else ""
        ci = f", ci_coeff={self._collision_integral_coeff}" if self._collision_integral_coeff > 0 else ""
        return (
            f"JanafMultiThermoEnhanced10(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, L_total={total_L:.0f}{rad}{ci})"
        )
