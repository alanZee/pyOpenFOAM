"""Enhanced wall treatment v9 — thermal wall function, species-aware treatment, and adaptive y+ selection.

Extends EnhancedWallTreatment8 with:
- Thermal wall function with Van Driest Prandtl number model
- Species-aware wall treatment for multi-component flows
- Adaptive y+ selection based on local flow conditions

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9
    wt = EnhancedWallTreatment9(nu=1.5e-5, Pr=0.71, n_species=3)
"""

from __future__ import annotations
import logging
import math
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_8 import (
    EnhancedWallTreatment8, ConjugateWallFunction,
)

__all__ = ["EnhancedWallTreatment9", "ThermalWallFunction"]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced9")
class EnhancedWallTreatment9(EnhancedWallTreatment8):
    """Enhanced wall treatment v9 with thermal wall function and species-aware treatment.

    Extends :class:`EnhancedWallTreatment8` with:

    - **Thermal wall function**: Van Driest Prandtl number model for
      accurate heat transfer in the viscous sublayer.
    - **Species-aware treatment**: wall functions that account for
      multi-component diffusion near walls.
    - **Adaptive y+ selection**: automatically selects optimal y+ target
      based on flow regime (laminar, transitional, turbulent).

    Parameters
    ----------
    nu, kappa, E, C_mu : see parent.
    n_species : int
        Number of species for multi-component treatment. Default 1.
    Sc_w : float
        Wall Schmidt number for species. Default 0.7.
    adaptive_y_plus : bool
        Enable adaptive y+ selection. Default False.
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
        n_layers: int = 3,
        solid_conductivity: float = 50.0,
        interface_resistance: float = 0.0,
        y_plus_uncertainty: bool = False,
        Sc_w: float = 0.7,
        adaptive_y_plus: bool = False,
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
            blend_exponent=blend_exponent,
            heat_flux_decomposition=heat_flux_decomposition,
            adaptive_switching=adaptive_switching,
            n_layers=n_layers,
            solid_conductivity=solid_conductivity,
            interface_resistance=interface_resistance,
            y_plus_uncertainty=y_plus_uncertainty,
        )
        self._Sc_w = max(0.01, Sc_w)
        self._adaptive_y_plus = adaptive_y_plus

    # ------------------------------------------------------------------
    # Thermal wall function (Van Driest Pr)
    # ------------------------------------------------------------------

    def thermal_wall_temperature(
        self,
        T_wall: float,
        T_cell: float,
        y_plus: float,
        u_tau: float,
    ) -> dict[str, float]:
        """Thermal wall function with Van Driest Prandtl model.

        T+ = Pr * y+ for y+ < 11 (viscous sublayer)
        T+ = Pr_t * (1/kappa * ln(E*y+) + P_lambda) for y+ > 11

        Parameters
        ----------
        T_wall : float
            Wall temperature (K).
        T_cell : float
            Cell-centre temperature (K).
        y_plus : float
            Dimensionless wall distance.
        u_tau : float
            Friction velocity (m/s).

        Returns
        -------
        dict
            'T_wall_calc': calculated wall temperature,
            'heat_flux': wall heat flux (W/m^2),
            'Nusselt': Nusselt number.
        """
        y_p = max(y_plus, 0.01)
        rho = 1.0
        Cp = 1005.0
        kappa_T = rho * Cp * self.nu / max(self._Pr, 0.01)

        if y_p < 11.0:
            T_plus = self._Pr * y_p
        else:
            # Log-law with P_lambda (Jayatilleke)
            P_lambda = 9.24 * ((self._Pr / max(self._Pr_t, 0.01)) ** 0.75 - 1.0) * (
                1.0 + 0.28 * math.exp(-0.007 * self._Pr / max(self._Pr_t, 0.01))
            )
            T_plus = self._Pr_t * (1.0 / self.kappa * math.log(max(self.E * y_p, 1.1)) + P_lambda)

        q = rho * Cp * u_tau * (T_cell - T_wall) / max(T_plus, 1.0)
        y = y_p * self.nu / max(u_tau, 1e-10)
        Nu = q * y / max(kappa_T * abs(T_cell - T_wall), 1e-30)

        return {
            "T_wall_calc": T_wall + q * y / max(kappa_T, 1e-10),
            "heat_flux": q,
            "Nusselt": abs(Nu),
        }

    # ------------------------------------------------------------------
    # Species-aware wall treatment
    # ------------------------------------------------------------------

    def species_wall_flux(
        self,
        Y_wall: float,
        Y_cell: float,
        y_plus: float,
        u_tau: float,
        species_idx: int = 0,
    ) -> dict[str, float]:
        """Species wall flux using mass transfer wall function.

        Parameters
        ----------
        Y_wall : float
            Wall mass fraction.
        Y_cell : float
            Cell-centre mass fraction.
        y_plus : float
            Dimensionless wall distance.
        u_tau : float
            Friction velocity.
        species_idx : int
            Species index.

        Returns
        -------
        dict
            'flux': species wall flux,
            'Sherwood': Sherwood number.
        """
        y_p = max(y_plus, 0.01)
        rho = 1.0
        D = self.nu / max(self._Sc_w, 0.01)

        if y_p < 11.0:
            Y_plus = self._Sc_w * y_p
        else:
            Y_plus = self._Pr_t * (1.0 / self.kappa * math.log(max(self.E * y_p, 1.1)))

        j = rho * u_tau * (Y_cell - Y_wall) / max(Y_plus, 1.0)
        y = y_p * self.nu / max(u_tau, 1e-10)
        Sh = j * y / max(D * abs(Y_cell - Y_wall), 1e-30)

        return {
            "flux": j,
            "Sherwood": abs(Sh),
        }

    # ------------------------------------------------------------------
    # Adaptive y+ selection
    # ------------------------------------------------------------------

    def adaptive_y_plus_target(self, Re_tau: float) -> float:
        """Select optimal y+ target based on friction Reynolds number.

        Parameters
        ----------
        Re_tau : float
            Friction Reynolds number.

        Returns
        -------
        float
            Target y+ value.
        """
        if not self._adaptive_y_plus:
            return self.y_plus_low

        if Re_tau < 100:
            return 1.0  # Low-Re: resolve viscous sublayer
        elif Re_tau < 500:
            return 5.0  # Transitional
        else:
            return 30.0  # High-Re: use wall function

    def __repr__(self) -> str:
        return (
            f"EnhancedWallTreatment9(nu={self.nu}, Pr={self._Pr}, "
            f"n_species={self._n_species}, Sc_w={self._Sc_w})"
        )


class ThermalWallFunction:
    """Standalone thermal wall function calculator.

    Parameters
    ----------
    Pr : float
        Prandtl number. Default 0.71.
    Pr_t : float
        Turbulent Prandtl number. Default 0.85.
    kappa : float
        Von Karman constant. Default 0.41.
    E : float
        Wall function constant. Default 9.8.
    """

    def __init__(
        self,
        Pr: float = 0.71,
        Pr_t: float = 0.85,
        kappa: float = 0.41,
        E: float = 9.8,
    ) -> None:
        self._Pr = max(0.01, Pr)
        self._Pr_t = max(0.01, Pr_t)
        self._kappa = max(0.01, kappa)
        self._E = max(1.0, E)

    def T_plus(self, y_plus: float) -> float:
        """Compute T+ from y+ using Van Driest model.

        Parameters
        ----------
        y_plus : float
            Dimensionless wall distance.

        Returns
        -------
        float
            T+ value.
        """
        y_p = max(y_plus, 0.01)
        if y_p < 11.0:
            return self._Pr * y_p
        else:
            P_lambda = 9.24 * ((self._Pr / self._Pr_t) ** 0.75 - 1.0)
            return self._Pr_t * (1.0 / self._kappa * math.log(max(self._E * y_p, 1.1)) + P_lambda)

    def __repr__(self) -> str:
        return f"ThermalWallFunction(Pr={self._Pr}, Pr_t={self._Pr_t})"
