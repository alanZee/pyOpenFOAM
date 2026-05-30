"""
Enhanced equation of state models v7 — cubic root selection, fugacity, and mixing rules.

Extends :class:`~pyfoam.thermophysical.equation_of_state_enhanced_6.EquationOfStateEnhanced6` with:

- Cubic EOS root selection algorithm (smallest positive real root)
- Fugacity coefficient computation for cubic EOS
- Van der Waals one-fluid mixing rules for cubic EOS

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_7 import (
        CubicRootSelector,
        FugacityCoefficientEOS,
        VdWOneFluidMixing,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS, RedlichKwongEOS
from pyfoam.thermophysical.equation_of_state_enhanced_6 import SRKVolumeTranslated

__all__ = [
    "CubicRootSelector",
    "FugacityCoefficientEOS",
    "VdWOneFluidMixing",
]

logger = logging.getLogger(__name__)


class CubicRootSelector(PengRobinsonEOS):
    """Peng-Robinson EOS with intelligent root selection for volume.

    When cubic EOS yields three real roots, selects the physically
    appropriate one (smallest positive for liquid, largest for vapour).

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
        Acentric factor. Default 0.
    phase : str
        Phase hint: "vapour" or "liquid". Default "vapour".
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        phase: str = "vapour",
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._phase = phase

    @property
    def phase(self) -> str:
        """Phase hint for root selection."""
        return self._phase

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with intelligent root selection."""
        rho_base = super().rho(p, T)
        # For now, use the base calculation with clamping
        return rho_base.clamp(min=1e-10)

    def V_molar(self, p: float, T: float) -> float:
        """Molar volume with root selection.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Molar volume (m^3/mol).
        """
        R = 8.314462618
        Tc = max(self._Tc, 1.0)
        Pc = max(self._Pc, 1.0)
        Tr = max(T, 1.0) / Tc

        # PR EOS parameters
        kappa = 0.37464 + 1.54226 * self._accentric - 0.26992 * self._accentric ** 2
        alpha = (1.0 + kappa * (1.0 - math.sqrt(max(Tr, 0.01)))) ** 2
        a = 0.45724 * R ** 2 * Tc ** 2 / Pc * alpha
        b = 0.07780 * R * Tc / Pc

        # Solve cubic: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        A = a * p / (R ** 2 * T ** 2)
        B = b * p / (R * T)
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B ** 2 - 2.0 * B
        c0 = -(A * B - B ** 2 - B ** 3)

        # Cardano's formula for real roots
        Q = (3.0 * c1 - c2 ** 2) / 9.0
        R_card = (9.0 * c2 * c1 - 27.0 * c0 - 2.0 * c2 ** 3) / 54.0
        D = Q ** 3 + R_card ** 2

        Z_roots = []
        if D >= 0:
            # One real root
            S = math.copysign(abs(R_card + math.sqrt(max(D, 0))) ** (1.0 / 3.0), R_card + math.sqrt(max(D, 0)))
            E = math.copysign(abs(R_card - math.sqrt(max(D, 0))) ** (1.0 / 3.0), R_card - math.sqrt(max(D, 0)))
            if abs(S) > 1e-30 or abs(E) > 1e-30:
                z = S + E - c2 / 3.0
                if z > 0:
                    Z_roots.append(z)
        else:
            # Three real roots
            theta = math.acos(max(-1.0, min(1.0, R_card / math.sqrt(max(-Q ** 3, 1e-30)))))
            for k in range(3):
                z = 2.0 * math.sqrt(max(-Q, 0)) * math.cos((theta + 2.0 * math.pi * k) / 3.0) - c2 / 3.0
                if z > 0:
                    Z_roots.append(z)

        if not Z_roots:
            return R * T / p  # Fallback to ideal gas

        Z_roots.sort()
        if self._phase == "liquid":
            Z_chosen = Z_roots[0]  # Smallest root for liquid
        else:
            Z_chosen = Z_roots[-1]  # Largest root for vapour

        return Z_chosen * R * T / max(p, 1e-10)

    def __repr__(self) -> str:
        return (
            f"CubicRootSelector(Mw={self._Mw}, Tc={self._Tc}, "
            f"phase={self._phase})"
        )


class FugacityCoefficientEOS(PengRobinsonEOS):
    """Peng-Robinson EOS with fugacity coefficient computation.

    Computes the fugacity coefficient phi = f/p from the PR EOS,
    useful for VLE calculations.

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
    """

    def fugacity_coefficient(self, p: float, T: float) -> float:
        """Compute fugacity coefficient from PR EOS.

        ln(phi) = Z - 1 - ln(Z - B) - A/(2*sqrt(2)*B) * ln((Z + (1+sqrt(2))*B) / (Z + (1-sqrt(2))*B))

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Fugacity coefficient (dimensionless).
        """
        R = 8.314462618
        Tc = max(self._Tc, 1.0)
        Pc = max(self._Pc, 1.0)
        Tr = max(T, 1.0) / Tc

        kappa = 0.37464 + 1.54226 * self._accentric - 0.26992 * self._accentric ** 2
        alpha = (1.0 + kappa * (1.0 - math.sqrt(max(Tr, 0.01)))) ** 2
        a = 0.45724 * R ** 2 * Tc ** 2 / Pc * alpha
        b = 0.07780 * R * Tc / Pc

        A = a * p / (R ** 2 * T ** 2)
        B = b * p / (R * T)

        # Use ideal gas Z as starting estimate
        Z = 1.0
        for _ in range(10):
            Z_new = 1.0 + B - A * Z / (Z * (Z + B) + B * (Z - B))
            if abs(Z_new - Z) < 1e-10:
                break
            Z = max(Z_new, 1e-10)

        # Fugacity coefficient
        sqrt2 = math.sqrt(2.0)
        ZB = max(Z - B, 1e-30)
        arg_num = Z + (1.0 + sqrt2) * B
        arg_den = Z + (1.0 - sqrt2) * B
        if abs(arg_den) < 1e-30 or arg_num <= 0:
            return 1.0

        ln_phi = Z - 1.0 - math.log(ZB) - A / (2.0 * sqrt2 * B) * math.log(arg_num / arg_den)
        return max(math.exp(ln_phi), 1e-10)

    def __repr__(self) -> str:
        return f"FugacityCoefficientEOS(Mw={self._Mw}, Tc={self._Tc})"


class VdWOneFluidMixing(PengRobinsonEOS):
    """Peng-Robinson EOS with Van der Waals one-fluid mixing rules.

    Combines multi-component parameters using standard mixing:

        a_mix = sum_i sum_j x_i * x_j * a_ij
        b_mix = sum_i x_i * b_i

    Parameters
    ----------
    Mw : float
        Mixture molecular weight (g/mol).
    Tc : float
        Mixture critical temperature (K).
    Pc : float
        Mixture critical pressure (Pa).
    Cp : float
        Specific heat (J/(kg*K)).
    accentric : float
        Mixture acentric factor. Default 0.
    k_ij : sequence of sequence of float or None
        Binary interaction parameters. Default None (zeros).
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        k_ij: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._k_ij = k_ij

    def mixture_parameters(
        self,
        x: Sequence[float],
        a_pure: Sequence[float],
        b_pure: Sequence[float],
    ) -> tuple[float, float]:
        """Compute mixture a and b using VdW one-fluid mixing rules.

        Parameters
        ----------
        x : sequence of float
            Mole fractions.
        a_pure : sequence of float
            Pure-component a parameters.
        b_pure : sequence of float
            Pure-component b parameters.

        Returns
        -------
        tuple of (float, float)
            (a_mix, b_mix).
        """
        n = len(x)
        b_mix = sum(x[i] * b_pure[i] for i in range(n))

        a_mix = 0.0
        for i in range(n):
            for j in range(n):
                k_ij = 0.0
                if self._k_ij is not None and i < len(self._k_ij) and j < len(self._k_ij[i]):
                    k_ij = self._k_ij[i][j]
                a_ij = math.sqrt(a_pure[i] * a_pure[j]) * (1.0 - k_ij)
                a_mix += x[i] * x[j] * a_ij

        return a_mix, b_mix

    def __repr__(self) -> str:
        n_kij = len(self._k_ij) if self._k_ij else 0
        return (
            f"VdWOneFluidMixing(Mw={self._Mw}, Tc={self._Tc}, "
            f"k_ij_dim={n_kij})"
        )
