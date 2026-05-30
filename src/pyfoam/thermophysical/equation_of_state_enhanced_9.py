"""Enhanced equation of state models v9 — mixture speed of sound, thermal pressure coefficient, and Joule-Thomson.

Extends v8 EOS models with:

- Mixture speed of sound computation
- Thermal pressure coefficient (beta = (dP/dT)_V)
- Joule-Thomson coefficient estimation

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_9 import (
        MixtureSpeedOfSoundEOS,
        ThermalPressureCoeffEOS,
        JouleThomsonEOS,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS, PerfectGas

__all__ = [
    "MixtureSpeedOfSoundEOS",
    "ThermalPressureCoeffEOS",
    "JouleThomsonEOS",
]

logger = logging.getLogger(__name__)


class MixtureSpeedOfSoundEOS(PengRobinsonEOS):
    """Mixture speed of sound from Peng-Robinson EOS.

    Computes speed of sound using:

        c^2 = (dP/drho)_s = gamma * P / rho

    with mixture gamma from mass-fraction-weighted species gammas.

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
    gamma_species : list of float
        Specific heat ratios for each species. Default [1.4].
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        gamma_species: Sequence[float] | None = None,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._gamma_species = list(gamma_species) if gamma_species else [1.4]

    def speed_of_sound(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
        Y: Sequence[float] | None = None,
    ) -> float:
        """Compute mixture speed of sound.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).
        Y : sequence of float or None
            Mass fractions. Default: single-species.

        Returns
        -------
        float
            Speed of sound (m/s).
        """
        rho = self.rho(p, T)
        rho_val = float(rho.item()) if hasattr(rho, "item") else float(rho)

        # Mixture gamma
        if Y is not None and len(Y) == len(self._gamma_species):
            gamma_mix = sum(Y[i] * self._gamma_species[i] for i in range(len(Y)))
        else:
            gamma_mix = self._gamma_species[0]

        p_val = float(p) if isinstance(p, (int, float)) else float(p.item())
        rho_safe = max(rho_val, 1e-10)

        c2 = gamma_mix * p_val / rho_safe
        return math.sqrt(max(c2, 0.0))

    def __repr__(self) -> str:
        return (
            f"MixtureSpeedOfSoundEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"n_species={len(self._gamma_species)})"
        )


class ThermalPressureCoeffEOS(PengRobinsonEOS):
    """Thermal pressure coefficient (beta) from Peng-Robinson EOS.

    beta = (dP/dT)_V = R / (V_mol - b) for ideal gas component.

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
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

    def thermal_pressure_coeff(self, T: float, rho: float) -> float:
        """Compute thermal pressure coefficient.

        Parameters
        ----------
        T : float
            Temperature (K).
        rho : float
            Density (kg/m^3).

        Returns
        -------
        float
            beta = (dP/dT)_V in Pa/K.
        """
        R = 8.314462618
        Mw_safe = max(self._Mw * 1e-3, 1e-10)
        rho_safe = max(rho, 1e-10)
        V_mol = Mw_safe / rho_safe  # m^3/mol

        # PR co-volume
        Tc = max(self._Tc, 1.0)
        Pc = max(self._Pc, 1.0)
        b = 0.07780 * R * Tc / Pc

        denom = max(V_mol - b, 1e-30)
        return R / denom

    def __repr__(self) -> str:
        return f"ThermalPressureCoeffEOS(Mw={self._Mw}, Tc={self._Tc})"


class JouleThomsonEOS(PengRobinsonEOS):
    """Joule-Thomson coefficient estimation from Peng-Robinson EOS.

    mu_JT = (1/Cp) * [T * (dV/dT)_P - V]

    Parameters
    ----------
    Mw, Tc, Pc, Cp, accentric : see PengRobinsonEOS.
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

    def joule_thomson_coeff(self, T: float, p: float) -> float:
        """Estimate Joule-Thomson coefficient.

        Parameters
        ----------
        T : float
            Temperature (K).
        p : float
            Pressure (Pa).

        Returns
        -------
        float
            mu_JT in K/Pa.
        """
        R = 8.314462618
        Cp = max(self._Cp, 1.0)
        T_safe = max(T, 1.0)
        p_safe = max(p, 1.0)

        # Ideal gas: V = RT/p, (dV/dT)_P = R/p
        V_mol = R * T_safe / p_safe
        dV_dT = R / p_safe

        mu_JT = (T_safe * dV_dT - V_mol) / Cp
        return mu_JT

    def inversion_temperature(self, p: float = 101325.0) -> float:
        """Estimate Joule-Thomson inversion temperature.

        At inversion: T * (dV/dT)_P = V, i.e., T_inv = V / (dV/dT)_P

        Parameters
        ----------
        p : float
            Pressure (Pa).

        Returns
        -------
        float
            Inversion temperature (K).
        """
        R = 8.314462618
        # For ideal gas: T_inv is infinite; for PR, estimate from acentric
        omega = self._accentric
        Tc = max(self._Tc, 1.0)
        # Approximate: T_inv ~ 6 * Tc / (1 + omega)
        return 6.0 * Tc / max(1.0 + omega, 0.1)

    def __repr__(self) -> str:
        return f"JouleThomsonEOS(Mw={self._Mw}, Tc={self._Tc})"
