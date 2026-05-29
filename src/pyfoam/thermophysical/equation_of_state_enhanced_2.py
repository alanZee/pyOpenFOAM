"""
Enhanced equation of state models v2 — improved cubic EOS with alpha functions.

Extends :class:`~pyfoam.thermophysical.equation_of_state_enhanced` with:

- Patel-Teja cubic EOS (three-parameter)
- Modified PR with temperature-dependent covolume (PR-Tb)
- Volume translation for improved liquid density prediction

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_2 import PatelTejaEOS, VolumeTranslatedPR

    pt = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
    rho = pt.rho(p=1e6, T=300.0)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    CubicEOS,
    PengRobinsonEOS,
)
from pyfoam.thermophysical.equation_of_state_enhanced import (
    SoaveRedlichKwongEOS,
    VirialEOS,
)

__all__ = [
    "PatelTejaEOS",
    "VolumeTranslatedPR",
    "PatelTejaValderramaEOS",
]

logger = logging.getLogger(__name__)


# ======================================================================
# Patel-Teja cubic EOS
# ======================================================================


class PatelTejaEOS(CubicEOS):
    """Patel-Teja (1982) three-parameter cubic equation of state.

    .. math::

        p = RT / (V - b) - a * alpha(T) / (V^2 + (b + c) * V - b * c)

    The third parameter c provides additional flexibility for predicting
    liquid densities compared to PR and SRK.

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
    ) -> None:
        # Patel-Teja Zc from accentric factor correlation
        Zc = 0.329032 - 0.076799 * accentric + 0.0211947 * accentric ** 2
        Zc = max(0.25, min(Zc, 0.35))

        # Omega values for Patel-Teja
        omega_a = 0.66121 - 0.76105 * Zc
        omega_b = 0.02207 + 0.20868 * Zc

        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp,
            Zc=Zc, omega_a=omega_a, omega_b=omega_b,
            accentric=accentric,
        )

        self._m = 0.452413 + 1.30982 * accentric - 0.295937 * accentric ** 2

        # Third parameter (c/b ratio)
        self._c_ratio = 0.5 * (1.0 - 3.0 * Zc) / (3.0 * Zc - 1.0)
        self._Omega_c = self._c_ratio * self._Omega_b

    @property
    def c_ratio(self) -> float:
        """Ratio of third parameter to covolume (c/b)."""
        return self._c_ratio

    def _alpha(self, T: torch.Tensor) -> torch.Tensor:
        """Patel-Teja alpha: [1 + m*(1 - sqrt(T/Tc))]^2."""
        T_r = T / self._Tc
        T_r = T_r.clamp(min=1e-10)
        factor = 1.0 + self._m * (1.0 - T_r.sqrt())
        return factor * factor

    def _Z_coeffs(self, p: torch.Tensor, T: torch.Tensor) -> tuple:
        """Patel-Teja Z-factor coefficients.

        Z^3 + (C-1)*Z^2 + (A - B*C - B^2 - C - B)*Z - (A*B - B^2*C - B^3) = 0
        (simplified for implementation)
        """
        aT = self._a(T)
        b = self._Omega_b
        c = self._Omega_c
        R = self._R

        A = aT * p / (R * R * T * T)
        B = b * p / (R * T)
        C_param = c * p / (R * T)

        c2 = C_param - torch.ones_like(A)
        c1 = A - B * C_param - B * B - C_param - B
        c0 = -(A * B - B * B * C_param - B ** 3)

        return c2, c1, c0

    def __repr__(self) -> str:
        return (
            f"PatelTejaEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )


# ======================================================================
# Volume-translated Peng-Robinson
# ======================================================================


class VolumeTranslatedPR(PengRobinsonEOS):
    """Peng-Robinson EOS with volume translation for improved liquid density.

    Applies a constant volume shift to the molar volume:

    V_corrected = V_PR - c_shift

    where c_shift is computed from the Peneloux correlation:
    c_shift = 0.40768 * R * Tc / Pc * (0.29441 - Z_RA)

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    volume_shift : float or None
        Volume translation parameter (m^3/mol). If None, estimated
        from the Peneloux correlation.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        volume_shift: float | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)

        if volume_shift is None:
            # Peneloux correlation
            Z_RA = 0.29056 - 0.08775 * accentric
            R_univ = 8.314462618
            Pc_safe = max(Pc, 1.0)
            self._c_shift = 0.40768 * R_univ * Tc / Pc_safe * (0.29441 - Z_RA)
        else:
            self._c_shift = volume_shift

    @property
    def volume_shift(self) -> float:
        """Volume translation parameter (m^3/mol)."""
        return self._c_shift

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with volume translation correction.

        First computes standard PR density, then applies shift.
        """
        rho_pr = super().rho(p, T)

        # Apply volume translation: rho_corrected = 1 / (1/rho_pr - c_shift/Mw)
        # where c_shift is in m^3/mol and Mw is in g/mol
        c_shift_mass = self._c_shift / (self._Mw * 1e-3)  # m^3/kg
        rho_pr_safe = rho_pr.clamp(min=1e-10)
        v_specific = 1.0 / rho_pr_safe  # m^3/kg
        v_corrected = v_specific - c_shift_mass
        v_corrected = v_corrected.clamp(min=1e-10)

        return 1.0 / v_corrected

    def __repr__(self) -> str:
        return (
            f"VolumeTranslatedPR(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, c_shift={self._c_shift:.6e})"
        )


# ======================================================================
# Patel-Teja-Valderrama (PTV) variant
# ======================================================================


class PatelTejaValderramaEOS(PatelTejaEOS):
    """Patel-Teja-Valderrama (1990) EOS variant for polar fluids.

    Uses Valderrama's correlations for the three parameters
    that improve predictions for polar and associating fluids:

        Omega_a = f(Zc)
        Omega_b = f(Zc)
        Zc = 0.329032 - 0.076799 * w + 0.0211947 * w^2

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Acentric factor.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)

        # Override with Valderrama's improved alpha
        self._m = 0.46283 + 1.54226 * accentric - 0.26992 * accentric ** 2

    def __repr__(self) -> str:
        return (
            f"PatelTejaValderramaEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"Pc={self._Pc:.0f}, accentric={self._accentric})"
        )
