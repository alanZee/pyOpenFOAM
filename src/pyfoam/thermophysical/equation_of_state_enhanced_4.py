"""
Enhanced equation of state models v4 — PC-SAFT and multi-fluid approximation.

Extends :class:`~pyfoam.thermophysical.equation_of_state_enhanced_3.EquationOfStateEnhanced3` with:

- Simplified PC-SAFT (Perturbed-Chain Statistical Associating Fluid Theory)
- Multi-fluid EOS approximation with departure functions
- Extended Corresponding States (ECS) model

Usage::

    from pyfoam.thermophysical.equation_of_state_enhanced_4 import PCSAFTSimplified, MultiFluidEOS

    pcsaft = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, m_seg=2.0)
    rho = pcsaft.rho(p=1e6, T=300.0)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    PengRobinsonEOS,
)
from pyfoam.thermophysical.equation_of_state_enhanced_3 import SAFTVRSimplified

__all__ = [
    "PCSAFTSimplified",
    "MultiFluidEOS",
    "ExtendedCorrespondingStatesEOS",
]

logger = logging.getLogger(__name__)

_PI = math.pi


# ======================================================================
# Simplified PC-SAFT
# ======================================================================


class PCSAFTSimplified(SAFTVRSimplified):
    """Simplified PC-SAFT equation of state.

    Perturbed-Chain SAFT with reduced parameterisation:

        p = p_hs_chain + p_disp + p_assoc

    where p_hs_chain is the hard-chain reference, p_disp is the
    dispersion contribution, and p_assoc is the association term.

    This implementation uses a simplified form suitable for engineering
    calculations with parameters fitted to critical properties.

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
        Association energy parameter (K). Default 0.
    d_sigma : float
        Temperature-independent segment diameter (Angstrom). Default 3.7.
    epsilon_k : float
        Dispersion energy depth (K). Default 200.0.
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
    ) -> None:
        super().__init__(
            Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric,
            m_seg=m_seg, assoc_energy=assoc_energy,
        )
        self._d_sigma = d_sigma
        self._epsilon_k = epsilon_k

    @property
    def segment_diameter(self) -> float:
        """Segment diameter (Angstrom)."""
        return self._d_sigma

    @property
    def dispersion_energy(self) -> float:
        """Dispersion energy depth epsilon/k_B (K)."""
        return self._epsilon_k

    def _packing_fraction(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute packing fraction eta.

        eta = pi/6 * rho * m * d^3 / Mw * N_A

        Simplified: eta ~ rho * m * d^3 * C_eta
        """
        C_eta = _PI / 6.0 * 1.0e-30 * 6.022e23 / max(self._Mw, 1.0)
        eta = rho * self._m_seg * self._d_sigma ** 3 * C_eta
        return eta.clamp(min=1e-10, max=0.5)

    def _dispersion_contribution(self, rho: torch.Tensor, T: float) -> torch.Tensor:
        """Simplified dispersion contribution to pressure.

        p_disp ~ -2 * pi * m^2 * eps_k * d^3 * rho^2 * I(T)

        where I(T) = a1 + a2/T* with T* = kT/eps.
        """
        T_safe = max(T, 1.0)
        T_star = T_safe / max(self._epsilon_k, 1.0)

        a1, a2 = -0.5, 0.3
        I_T = a1 + a2 / max(T_star, 0.01)

        coeff = -2.0 * _PI * self._m_seg ** 2 * self._epsilon_k * self._d_sigma ** 3
        return coeff * rho.pow(2) * I_T * 1e-30

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with PC-SAFT chain and dispersion corrections."""
        rho_pr = super().rho(p, T)

        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        # Apply dispersion correction
        rho = rho_pr.clone()
        for _ in range(3):
            p_disp = self._dispersion_contribution(rho, T_val)
            p_correction = p_disp.abs().clamp(max=rho_pr * 0.3)
            rho = rho_pr + p_correction * 1e-6
            rho = rho.clamp(min=1e-10)

        return rho

    def __repr__(self) -> str:
        return (
            f"PCSAFTSimplified(Mw={self._Mw}, Tc={self._Tc}, "
            f"m={self._m_seg}, d={self._d_sigma}, eps_k={self._epsilon_k})"
        )


# ======================================================================
# Multi-Fluid EOS
# ======================================================================


class MultiFluidEOS(PengRobinsonEOS):
    """Multi-fluid EOS with departure function.

    Combines a cubic EOS (PR) with a departure function that accounts
    for mixture non-ideality beyond simple mixing rules:

        a_mix = sum_i sum_j(x_i * x_j * a_ij * (1 - k_ij(T)))

    where k_ij(T) = k_ij_0 + k_ij_1 / T is temperature-dependent.

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
        Binary interaction parameter (constant part). Default 0.
    k_ij_1 : float
        Binary interaction parameter (temperature part). Default 0.
    departure_coeff : float
        Departure function coefficient. Default 0.
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
        departure_coeff: float = 0.0,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._k_ij_0 = k_ij_0
        self._k_ij_1 = k_ij_1
        self._departure_coeff = departure_coeff

    @property
    def k_ij(self) -> float:
        """Constant part of binary interaction parameter."""
        return self._k_ij_0

    def _k_ij_T(self, T: float) -> float:
        """Temperature-dependent binary interaction parameter.

        k_ij(T) = k_ij_0 + k_ij_1 / T
        """
        T_safe = max(T, 1.0)
        return self._k_ij_0 + self._k_ij_1 / T_safe

    def _departure_pressure(self, rho: torch.Tensor, T: float) -> torch.Tensor:
        """Departure function contribution to pressure.

        p_dep = departure_coeff * rho^2 * T / Tc
        """
        if abs(self._departure_coeff) < 1e-30:
            return torch.zeros_like(rho)

        return self._departure_coeff * rho.pow(2) * T / max(self._Tc, 1.0)

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with departure function correction."""
        rho_base = super().rho(p, T)

        if isinstance(T, (int, float)):
            T_val = float(T)
        else:
            T_val = float(T.item())

        if abs(self._departure_coeff) < 1e-30:
            return rho_base

        # Iterative correction
        rho = rho_base.clone()
        for _ in range(3):
            p_dep = self._departure_pressure(rho, T_val)
            rho = rho_base + p_dep.abs() * 1e-6
            rho = rho.clamp(min=1e-10)

        return rho

    def __repr__(self) -> str:
        return (
            f"MultiFluidEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"k_ij=({self._k_ij_0}, {self._k_ij_1}))"
        )


# ======================================================================
# Extended Corresponding States (ECS)
# ======================================================================


class ExtendedCorrespondingStatesEOS(PengRobinsonEOS):
    """Extended Corresponding States EOS.

    Uses shape factors and a reference fluid to predict thermodynamic
    properties via corresponding states:

        p = p_ref(rho_r, T_r) * f(shape)

    where shape factors account for molecular shape differences between
    the target fluid and the reference fluid.

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
    shape_theta : float
        Shape factor theta. Default 1.0 (simple fluid).
    shape_phi : float
        Shape factor phi. Default 1.0.
    """

    def __init__(
        self,
        Mw: float,
        Tc: float,
        Pc: float,
        Cp: float,
        accentric: float = 0.0,
        shape_theta: float = 1.0,
        shape_phi: float = 1.0,
    ) -> None:
        super().__init__(Mw=Mw, Tc=Tc, Pc=Pc, Cp=Cp, accentric=accentric)
        self._theta = max(shape_theta, 0.1)
        self._phi = max(shape_phi, 0.1)

    @property
    def shape_theta(self) -> float:
        """Shape factor theta."""
        return self._theta

    @property
    def shape_phi(self) -> float:
        """Shape factor phi."""
        return self._phi

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Density with ECS shape-factor correction.

        Applies shape factors to the PR density:
        rho_ECS = rho_PR * phi / theta
        """
        rho_base = super().rho(p, T)
        return rho_base * self._phi / self._theta

    def __repr__(self) -> str:
        return (
            f"ExtendedCorrespondingStatesEOS(Mw={self._Mw}, Tc={self._Tc}, "
            f"theta={self._theta}, phi={self._phi})"
        )
