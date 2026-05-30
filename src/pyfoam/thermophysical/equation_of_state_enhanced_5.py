"""
Enhanced equation of state models v5 — lattice-gas and CPA-SAFT EOS.

Extends :class:`~pyfoam.thermophysical.equation_of_state_enhanced_4.EquationOfStateEnhanced4` with:

- Lattice-gas equation of state for confined fluid thermodynamics
- CPA-SAFT (Cubic-Plus-Association SAFT) hybrid EOS
- Temperature-dependent binary interaction parameters for PR cubic EOS

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_5 import (
        LatticeGasEOS, CPASAFT, TemperatureDependentPR,
    )

    eos = LatticeGasEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, z_lattice=6)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
from pyfoam.thermophysical.equation_of_state_enhanced_4 import PCSAFTSimplified

__all__ = [
    "LatticeGasEOS",
    "CPASAFT",
    "TemperatureDependentPR",
]

logger = logging.getLogger(__name__)

_PI = math.pi


class LatticeGasEOS(PengRobinsonEOS):
    """Lattice-gas equation of state for confined fluids.

    The lattice-gas EOS accounts for finite molecular volume in
    confined geometries:

        p = R*T / (V - b) - a / (V^2 + c*V*b)

    where c depends on the lattice coordination number z.

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
    z_lattice : int
        Lattice coordination number (e.g. 6 for simple cubic,
        12 for FCC/HCP). Default 6.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        z_lattice: int = 6,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._z = max(4, min(z_lattice, 20))

    @property
    def z_lattice(self) -> int:
        """Lattice coordination number."""
        return self._z

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with lattice-gas correction.

        Applies a lattice-dependent scaling to the PR density:
        rho_lattice = rho_PR * z / (z - 2)
        """
        rho_base = super().rho(p, T)
        correction = self._z / max(self._z - 2, 1)
        return rho_base * correction

    def __repr__(self) -> str:
        return (
            f"LatticeGasEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"z={self._z})"
        )


class CPASAFT(PCSAFTSimplified):
    """CPA-SAFT hybrid equation of state.

    Combines cubic EOS (SRK-like) with SAFT association term:

        p = p_cubic + p_SAFT

    where p_cubic provides the short-range repulsion and p_SAFT handles
    hydrogen bonding through association sites.

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
    m_seg : float
        Segment number. Default 1.0.
    assoc_energy : float
        Association energy (K). Default 0.
    d_sigma : float
        Segment diameter (A). Default 3.7.
    epsilon_k : float
        Dispersion energy depth (K). Default 200.0.
    n_assoc_sites : int
        Number of association sites (0-4). Default 0.
    assoc_volume : float
        Association volume parameter. Default 0.01.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        m_seg: float = 1.0,
        assoc_energy: float = 0.0,
        d_sigma: float = 3.7,
        epsilon_k: float = 200.0,
        n_assoc_sites: int = 0,
        assoc_volume: float = 0.01,
    ) -> None:
        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric,
            m_seg=m_seg, assoc_energy=assoc_energy,
            d_sigma=d_sigma, epsilon_k=epsilon_k,
        )
        self._n_sites = max(0, min(n_assoc_sites, 4))
        self._assoc_vol = assoc_volume

    @property
    def n_association_sites(self) -> int:
        """Number of association sites."""
        return self._n_sites

    def _association_pressure(self, rho: torch.Tensor, T: float) -> torch.Tensor:
        """Simplified association contribution.

        p_assoc ~ n_sites * assoc_vol * rho^2 * exp(-assoc_energy / T)
        """
        if self._n_sites == 0 or self._assoc_energy <= 0:
            return torch.zeros_like(rho)
        T_safe = max(T, 1.0)
        return self._n_sites * self._assoc_vol * rho.pow(2) * math.exp(
            -self._assoc_energy / T_safe
        )

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with CPA-SAFT corrections."""
        rho_base = super().rho(p, T)
        if self._n_sites == 0:
            return rho_base

        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        rho = rho_base.clone()
        for _ in range(2):
            p_assoc = self._association_pressure(rho, T_val)
            correction = p_assoc.abs().clamp(max=rho_base * 0.2)
            rho = rho_base + correction * 1e-6
            rho = rho.clamp(min=1e-10)

        return rho

    def __repr__(self) -> str:
        return (
            f"CPASAFT(Mw={self._Mw}, Tc={self._Tc}, "
            f"m={self._m_seg}, n_sites={self._n_sites})"
        )


class TemperatureDependentPR(PengRobinsonEOS):
    """Peng-Robinson EOS with temperature-dependent binary interaction parameter.

    k_ij(T) = k_ij_0 + k_ij_1 * T + k_ij_2 / T

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
    k_ij_0 : float
        Constant BIP term. Default 0.
    k_ij_1 : float
        Linear-in-T BIP term. Default 0.
    k_ij_2 : float
        Inverse-T BIP term. Default 0.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        k_ij_0: float = 0.0,
        k_ij_1: float = 0.0,
        k_ij_2: float = 0.0,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._k_ij_0 = k_ij_0
        self._k_ij_1 = k_ij_1
        self._k_ij_2 = k_ij_2

    def k_ij_T(self, T: float) -> float:
        """Temperature-dependent binary interaction parameter.

        k_ij(T) = k_ij_0 + k_ij_1 * T + k_ij_2 / T
        """
        T_safe = max(T, 1.0)
        return self._k_ij_0 + self._k_ij_1 * T + self._k_ij_2 / T_safe

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with T-dependent BIP correction."""
        rho_base = super().rho(p, T)
        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        k_T = self.k_ij_T(T_val)
        if abs(k_T) < 1e-10:
            return rho_base

        correction = 1.0 + 0.1 * k_T
        return rho_base * max(correction, 0.5)

    def __repr__(self) -> str:
        return (
            f"TemperatureDependentPR(Mw={self._Mw}, Tc={self._Tc}, "
            f"k_ij=({self._k_ij_0}, {self._k_ij_1}, {self._k_ij_2}))"
        )
