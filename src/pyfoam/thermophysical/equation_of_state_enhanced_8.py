"""Enhanced equation of state models v8 — critical scaling, multi-component departure, and density initialization.

Extends v7 EOS models with:

- Critical scaling exponent EOS near the critical point
- Multi-component departure function for mixture EOS
- Robust density initialization for CFD solvers

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_8 import (
        CriticalScalingEOS,
        MultiComponentDepartureEOS,
        RobustDensityInitEOS,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS, PerfectGas
from pyfoam.thermophysical.equation_of_state_enhanced_7 import CubicRootSelector

__all__ = [
    "CriticalScalingEOS",
    "MultiComponentDepartureEOS",
    "RobustDensityInitEOS",
]

logger = logging.getLogger(__name__)


class CriticalScalingEOS(PengRobinsonEOS):
    """Peng-Robinson EOS with critical scaling corrections near Tc.

    Near the critical point, density follows a scaling law:

        rho ~ rho_c * (1 + A * |1 - T/Tc|^beta)

    where beta = 0.325 (3D Ising universality class).

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
    scaling_amplitude : float
        Amplitude A for critical scaling. Default 1.5.
    scaling_beta : float
        Critical exponent beta. Default 0.325.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        scaling_amplitude: float = 1.5,
        scaling_beta: float = 0.325,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._A_scale = scaling_amplitude
        self._beta = scaling_beta

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with critical scaling correction."""
        rho_base = super().rho(p, T)

        T_val = float(T) if isinstance(T, (int, float)) else float(T.item())
        Tc = max(self._Tc, 1.0)
        Tr = abs(1.0 - T_val / Tc)

        if Tr < 0.1:
            # Apply critical scaling
            correction = 1.0 + self._A_scale * Tr ** self._beta
            return rho_base * correction

        return rho_base.clamp(min=1e-10)

    def __repr__(self) -> str:
        return (
            f"CriticalScalingEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"beta={self._beta})"
        )


class MultiComponentDepartureEOS(PengRobinsonEOS):
    """Multi-component departure function for Peng-Robinson EOS.

    Computes departure function for mixtures:

        a_dep = sum_i sum_j x_i * x_j * (a_i * a_j)^0.5 * (1 - k_ij)
        b_dep = sum_i x_i * b_i

    with temperature-dependent k_ij.

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
    k_ij : list of list of float
        Binary interaction parameters.
    k_ij_T_coeff : list of list of float or None
        Temperature coefficient for k_ij(T) = k_ij + k_ij_T * (T - T_ref).
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        k_ij: Sequence[Sequence[float]] | None = None,
        k_ij_T_coeff: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._k_ij = k_ij
        self._k_ij_T = k_ij_T_coeff

    def departure_helmholtz(self, T: float, x: Sequence[float]) -> dict[str, float]:
        """Compute departure Helmholtz energy.

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.

        Returns
        -------
        dict
            'a_dep': departure attraction parameter,
            'b_dep': departure co-volume,
            'A_departure': dimensionless departure Helmholtz energy.
        """
        R = 8.314462618
        n = len(x)

        # Pure component parameters
        a_pure = []
        b_pure = []
        for i in range(n):
            Tc_i = max(self._Tc, 1.0)
            Pc_i = max(self._Pc, 1.0)
            kappa_i = 0.37464 + 1.54226 * self._accentric - 0.26992 * self._accentric ** 2
            Tr = max(T, 1.0) / Tc_i
            alpha = (1.0 + kappa_i * (1.0 - math.sqrt(max(Tr, 0.01)))) ** 2
            a_i = 0.45724 * R ** 2 * Tc_i ** 2 / Pc_i * alpha
            b_i = 0.07780 * R * Tc_i / Pc_i
            a_pure.append(a_i)
            b_pure.append(b_i)

        b_dep = sum(x[i] * b_pure[i] for i in range(n))

        a_dep = 0.0
        T_ref = 298.15
        for i in range(n):
            for j in range(n):
                k = 0.0
                k_T = 0.0
                if self._k_ij and i < len(self._k_ij) and j < len(self._k_ij[i]):
                    k = self._k_ij[i][j]
                if self._k_ij_T and i < len(self._k_ij_T) and j < len(self._k_ij_T[i]):
                    k_T = self._k_ij_T[i][j]
                k_eff = k + k_T * (T - T_ref)
                a_ij = math.sqrt(a_pure[i] * a_pure[j]) * (1.0 - k_eff)
                a_dep += x[i] * x[j] * a_ij

        V_mol = max(b_dep, 1e-30) * 1.5  # Approximate
        A_dep = -a_dep / (R * T * V_mol) if (R * T * V_mol) > 1e-30 else 0.0

        return {"a_dep": a_dep, "b_dep": b_dep, "A_departure": A_dep}

    def __repr__(self) -> str:
        n_kij = len(self._k_ij) if self._k_ij else 0
        return (
            f"MultiComponentDepartureEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"k_ij_dim={n_kij})"
        )


class RobustDensityInitEOS(PerfectGas):
    """Robust density initialization for CFD solvers.

    Provides safe initial density estimates that respect physical bounds
    even with extreme initial pressure/temperature guesses.

    Parameters
    ----------
    Mw : float
        Molecular weight (g/mol).
    rho_min : float
        Minimum density bound (kg/m^3). Default 0.001.
    rho_max : float
        Maximum density bound (kg/m^3). Default 10000.
    """

    def __init__(
        self,
        Mw: float = 28.97,
        rho_min: float = 0.001,
        rho_max: float = 10000.0,
    ) -> None:
        R_specific = 8.314462618 / max(Mw * 1e-3, 1e-10)
        super().__init__(R=R_specific, Cp=1005.0)
        self._Mw = Mw
        self._rho_min = max(1e-10, rho_min)
        self._rho_max = max(rho_min, rho_max)

    def rho_init(
        self,
        p: float,
        T: float,
    ) -> float:
        """Compute robust initial density.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Bounded density (kg/m^3).
        """
        R_specific = 8.314462618 / max(self._Mw * 1e-3, 1e-10)
        T_safe = max(T, 1.0)
        p_safe = max(p, 1.0)

        rho = p_safe / (R_specific * T_safe)
        return max(self._rho_min, min(rho, self._rho_max))

    def __repr__(self) -> str:
        return (
            f"RobustDensityInitEOS(Mw={self._Mw}, "
            f"rho_range=[{self._rho_min}, {self._rho_max}])"
        )
