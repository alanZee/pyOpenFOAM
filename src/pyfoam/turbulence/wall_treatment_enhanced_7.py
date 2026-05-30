"""Enhanced wall treatment v7 — blended two-layer, heat flux decomposition, y+ adaptive switching.

Extends EnhancedWallTreatment6 with:
- Blended two-layer wall function for smooth viscous/log-law transition
- Wall heat flux decomposition into convective and radiative components
- Adaptive y+ regime switching with hysteresis

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
    wt = EnhancedWallTreatment7(nu=1.5e-5, Pr=0.71, n_species=3)
"""

from __future__ import annotations
import logging
import math
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_6 import (
    EnhancedWallTreatment6, RoughnessCorrelation,
)

__all__ = ["EnhancedWallTreatment7", "HeatFluxDecomposition"]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced7")
class EnhancedWallTreatment7(EnhancedWallTreatment6):
    """Enhanced wall treatment v7 with blended two-layer and heat flux decomposition.

    Extends :class:`EnhancedWallTreatment6` with:

    - **Blended two-layer wall function**: smooth blending between viscous
      sublayer (u+ = y+) and log-law region using a modified blending function.
    - **Heat flux decomposition**: separates wall heat flux into convective
      (turbulent) and conductive (molecular) contributions.
    - **Adaptive y+ switching**: uses hysteresis to prevent oscillation
      between viscous and log-law regimes.

    Parameters
    ----------
    nu, kappa, E, C_mu : see parent.
    y_plus_low, y_plus_high, ks : see parent.
    Pr, Pr_t, hysteresis_width : see parent.
    Le, van_driest_A : see parent.
    k_solid, solid_thickness, y_plus_ema_alpha : see parent.
    n_species, Sc, roughness_constant : see parent.
    blend_exponent : float
        Exponent for two-layer blending function. Default 4.0.
    heat_flux_decomposition : bool
        Enable heat flux decomposition. Default False.
    adaptive_switching : bool
        Enable adaptive y+ regime switching. Default False.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
        y_plus_low: float = 5.0,
        y_plus_high: float = 30.0,
        ks: float = 0.0,
        Pr: float = 0.71,
        Pr_t: float = 0.85,
        hysteresis_width: float = 2.0,
        Le: float = 1.0,
        van_driest_A: float = 26.0,
        k_solid: float = 50.0,
        solid_thickness: float = 0.001,
        y_plus_ema_alpha: float = 0.3,
        n_species: int = 1,
        Sc: float = 0.7,
        roughness_constant: float = 0.5,
        blend_exponent: float = 4.0,
        heat_flux_decomposition: bool = False,
        adaptive_switching: bool = False,
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
            ks=ks, Pr=Pr, Pr_t=Pr_t, hysteresis_width=hysteresis_width,
            Le=Le, van_driest_A=van_driest_A,
            k_solid=k_solid, solid_thickness=solid_thickness,
            y_plus_ema_alpha=y_plus_ema_alpha,
            n_species=n_species, Sc=Sc,
            roughness_constant=roughness_constant,
        )
        self._blend_exp = max(1.0, blend_exponent)
        self._heat_flux_decomp = heat_flux_decomposition
        self._adaptive_switching = adaptive_switching
        self._current_regime: str = "unknown"
        self._hysteresis_width_val = max(0.0, hysteresis_width)

    @property
    def heat_flux_decomposition_enabled(self) -> bool:
        return self._heat_flux_decomp

    # ------------------------------------------------------------------
    # Blended two-layer wall function
    # ------------------------------------------------------------------

    def blended_u_plus(self, y_plus: float) -> float:
        """Blended two-layer u+ law.

        u+ = y+ * exp(-Gamma) + (1/kappa * ln(E * y+)) * exp(-1/Gamma)

        where Gamma = (y_plus / y_plus_blend)^n

        Parameters
        ----------
        y_plus : float
            Dimensionless wall distance.

        Returns
        -------
        float
            Dimensionless velocity u+.
        """
        y_p = max(y_plus, 0.01)
        y_blend = (self.y_plus_low + self.y_plus_high) / 2.0
        Gamma = (y_p / max(y_blend, 1.0)) ** self._blend_exp

        exp_neg_G = math.exp(-min(Gamma, 50.0))
        exp_neg_invG = math.exp(-min(1.0 / max(Gamma, 1e-10), 50.0))

        u_visc = y_p * exp_neg_G
        u_log = (1.0 / self.kappa) * math.log(max(self.E * y_p, 1.1)) * exp_neg_invG

        return u_visc + u_log

    def blended_nut(self, y_plus: float, u_tau: float) -> float:
        """Blended turbulent viscosity from two-layer model.

        Parameters
        ----------
        y_plus : float
            Dimensionless wall distance.
        u_tau : float
            Friction velocity (m/s).

        Returns
        -------
        float
            Blended turbulent viscosity (m^2/s).
        """
        y_p = max(y_plus, 0.01)
        y = y_p * self.nu / max(u_tau, 1e-10)
        kappa = self.kappa
        nu = self.nu

        # Viscous sublayer: nut ~ y^2
        nu_visc = nu * y_p ** 2 * 0.001

        # Log-law: nut = kappa * y * u_tau
        nu_log = kappa * y * u_tau

        # Blend
        y_blend = (self.y_plus_low + self.y_plus_high) / 2.0
        w = 1.0 / (1.0 + math.exp(-self._blend_exp * (y_p - y_blend) / max(y_blend, 1.0)))
        return (1.0 - w) * nu_visc + w * nu_log

    # ------------------------------------------------------------------
    # Heat flux decomposition
    # ------------------------------------------------------------------

    def wall_heat_flux_decomposition(
        self,
        T_wall: float,
        T_fluid: float,
        y: float,
    ) -> dict[str, float]:
        """Decompose wall heat flux into convective and conductive components.

        Parameters
        ----------
        T_wall : float
            Wall temperature (K).
        T_fluid : float
            Near-wall fluid temperature (K).
        y : float
            Wall-normal distance (m).

        Returns
        -------
        dict
            'q_convective': turbulent heat flux (W/m^2),
            'q_conductive': molecular heat flux (W/m^2),
            'q_total': total heat flux.
        """
        dT = T_wall - T_fluid
        y_safe = max(y, 1e-10)

        # Conductive (molecular): q = -k * dT/dy ~ -k * dT/y
        # k ~ rho * Cp * nu / Pr
        rho = 1.0  # Approximate
        Cp = 1005.0
        k_mol = rho * Cp * self.nu / max(self._Pr, 0.01)
        q_cond = k_mol * abs(dT) / y_safe

        # Convective (turbulent): q_turb = rho * Cp * alpha_t * dT/dy
        alpha_t = self.nu / max(self._Pr_t, 0.01)
        q_conv = rho * Cp * alpha_t * abs(dT) / y_safe

        return {
            "q_convective": q_conv,
            "q_conductive": q_cond,
            "q_total": q_conv + q_cond,
        }

    # ------------------------------------------------------------------
    # Adaptive y+ switching
    # ------------------------------------------------------------------

    def adaptive_regime(self, y_plus: float) -> str:
        """Determine wall regime with hysteresis.

        Parameters
        ----------
        y_plus : float
            Dimensionless wall distance.

        Returns
        -------
        str
            Regime: "viscous", "buffer", or "log_law".
        """
        if not self._adaptive_switching:
            if y_plus < self.y_plus_low:
                return "viscous"
            elif y_plus < self.y_plus_high:
                return "buffer"
            else:
                return "log_law"

        hw = self._hysteresis_width_val
        regime = self._current_regime

        if regime == "viscous" and y_plus > self.y_plus_low + hw:
            regime = "buffer"
        elif regime == "buffer":
            if y_plus < self.y_plus_low:
                regime = "viscous"
            elif y_plus > self.y_plus_high:
                regime = "log_law"
        elif regime == "log_law" and y_plus < self.y_plus_high - hw:
            regime = "buffer"
        elif regime == "unknown":
            if y_plus < self.y_plus_low:
                regime = "viscous"
            elif y_plus < self.y_plus_high:
                regime = "buffer"
            else:
                regime = "log_law"

        self._current_regime = regime
        return regime

    def __repr__(self) -> str:
        return (
            f"EnhancedWallTreatment7(nu={self.nu}, Pr={self._Pr}, "
            f"Le={self._Le}, n_species={self._n_species}, "
            f"blend_exp={self._blend_exp})"
        )


class HeatFluxDecomposition:
    """Standalone heat flux decomposition calculator.

    Parameters
    ----------
    Pr : float
        Prandtl number. Default 0.71.
    Pr_t : float
        Turbulent Prandtl number. Default 0.85.
    """

    def __init__(self, Pr: float = 0.71, Pr_t: float = 0.85) -> None:
        self._Pr = max(0.01, Pr)
        self._Pr_t = max(0.01, Pr_t)

    def turbulent_conductivity(self, nu_t: float, rho: float = 1.0, Cp: float = 1005.0) -> float:
        """Turbulent thermal conductivity.

        k_t = rho * Cp * nu_t / Pr_t

        Parameters
        ----------
        nu_t : float
            Turbulent viscosity (m^2/s).
        rho : float
            Density (kg/m^3).
        Cp : float
            Specific heat (J/(kg*K)).

        Returns
        -------
        float
            Turbulent conductivity (W/(m*K)).
        """
        return rho * Cp * nu_t / self._Pr_t

    def __repr__(self) -> str:
        return f"HeatFluxDecomposition(Pr={self._Pr}, Pr_t={self._Pr_t})"
