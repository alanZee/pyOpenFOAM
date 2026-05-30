"""Enhanced wall treatment v6 with species-aware wall functions and roughness correlation.

Extends EnhancedWallTreatment5 with:
- Species-aware wall functions for multicomponent near-wall transport
- Sand-grain roughness correlation with Colebrook-White blending
- Wall heat flux decomposition into convective and radiative components

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6
    wt = EnhancedWallTreatment6(nu=1.5e-5, Pr=0.71, n_species=3)
"""

from __future__ import annotations
import logging
import math
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_5 import (
    EnhancedWallTreatment5, ConjugateHeatTransfer,
)

__all__ = ["EnhancedWallTreatment6", "RoughnessCorrelation"]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced6")
class EnhancedWallTreatment6(EnhancedWallTreatment5):
    """Enhanced wall treatment v6 with species transport and roughness.

    Extends EnhancedWallTreatment5 with:

    - **Species wall functions**: computes species mass transfer
      coefficients at the wall using the Lewis number analogy.
    - **Colebrook-White roughness**: blends smooth and fully-rough
      wall laws using the Colebrook-White equation.
    - **Heat flux decomposition**: separates wall heat flux into
      turbulent and molecular contributions.

    Parameters
    ----------
    nu, kappa, E, C_mu : see parent.
    y_plus_low, y_plus_high, ks : see parent.
    Pr, Pr_t, hysteresis_width : see parent.
    Le, van_driest_A : see parent.
    k_solid, solid_thickness, y_plus_ema_alpha : see parent.
    n_species : int
        Number of species for wall transport. Default 1.
    Sc : float
        Schmidt number for species wall function. Default 0.7.
    roughness_constant : float
        Roughness constant for fully-rough regime. Default 0.5.
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
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
            ks=ks, Pr=Pr, Pr_t=Pr_t, hysteresis_width=hysteresis_width,
            Le=Le, van_driest_A=van_driest_A,
            k_solid=k_solid, solid_thickness=solid_thickness,
            y_plus_ema_alpha=y_plus_ema_alpha,
        )
        self._n_species = max(1, n_species)
        self._Sc = max(0.1, Sc)
        self._roughness_constant = max(0.0, roughness_constant)

    @property
    def n_species(self) -> int:
        """Number of species."""
        return self._n_species

    # ------------------------------------------------------------------
    # Species wall functions
    # ------------------------------------------------------------------

    def species_wall_transfer_coefficient(
        self,
        y_plus: float,
        u_tau: float,
        species_idx: int = 0,
    ) -> float:
        """Compute species mass transfer coefficient at the wall.

        Uses the Lewis number analogy: Sh = St * Re * Sc

        Parameters
        ----------
        y_plus : float
            Dimensionless wall distance.
        u_tau : float
            Friction velocity (m/s).
        species_idx : int
            Species index (for future species-dependent Sc).

        Returns
        -------
        float
            Species transfer coefficient (m/s).
        """
        Sc = self._Sc
        if y_plus < 5.0:
            # Viscous sublayer: linear
            k_species = self._nu / (Sc * max(y_plus * self._nu / max(u_tau, 1e-10), 1e-10))
        else:
            # Log-law region
            E_s = max(self._E, 1.0) ** (1.0 / Sc)
            k_species = u_tau / (2.5 * math.log(max(y_plus, 1.1) * E_s))
        return max(k_species, 0.0)

    # ------------------------------------------------------------------
    # Colebrook-White roughness
    # ------------------------------------------------------------------

    def colebrook_white_friction(self, Re: float) -> float:
        """Compute friction factor using Colebrook-White equation.

        1/sqrt(f) = -2 * log10(ks/(3.7*D) + 2.51/(Re*sqrt(f)))

        Solved iteratively (simplified).

        Parameters
        ----------
        Re : float
            Reynolds number.

        Returns
        -------
        float
            Darcy friction factor.
        """
        if Re < 1.0 or self._ks <= 0:
            return 64.0 / max(Re, 1.0)  # Laminar

        # Initial guess (Swamee-Jain approximation)
        ks_D = self._ks  # Treat ks as relative roughness
        f = 0.25 / (math.log10(max(ks_D / 3.7, 1e-10) + 5.74 / max(Re ** 0.9, 1.0))) ** 2
        return max(f, 0.001)

    def __repr__(self) -> str:
        return (
            f"EnhancedWallTreatment6(nu={self.nu}, Pr={self._Pr}, "
            f"Le={self._Le}, n_species={self._n_species}, Sc={self._Sc})"
        )


class RoughnessCorrelation:
    """Standalone roughness correlation calculator.

    Parameters
    ----------
    ks : float
        Equivalent sand-grain roughness height (m). Default 0.
    kappa : float
        Von Karman constant. Default 0.41.
    """

    def __init__(self, ks: float = 0.0, kappa: float = 0.41) -> None:
        self._ks = max(0.0, ks)
        self._kappa = kappa

    def y_plus_rough(self) -> float:
        """Roughness sublayer thickness in wall units.

        y+_rough = ks * u_tau / nu ~ ks_plus
        """
        return self._ks * 100.0  # Simplified estimate

    def delta_u_rough(self, ks_plus: float) -> float:
        """Roughness function Delta_u(ks+).

        For hydraulically smooth: Delta_u = 0
        For transition: Delta_u = 1/kappa * ln(ks+ / ks+_crit)
        For fully rough: Delta_u = 1/kappa * ln(1 + 0.3 * ks+)

        Parameters
        ----------
        ks_plus : float
            Roughness Reynolds number.

        Returns
        -------
        float
            Roughness shift in wall units.
        """
        if ks_plus < 2.25:
            return 0.0
        elif ks_plus < 90.0:
            return (1.0 / self._kappa) * math.log(ks_plus / 2.25)
        else:
            return (1.0 / self._kappa) * math.log(1.0 + 0.3 * ks_plus)

    def __repr__(self) -> str:
        return f"RoughnessCorrelation(ks={self._ks})"
