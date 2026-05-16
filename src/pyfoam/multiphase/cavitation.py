"""
Cavitation models for multiphase flows.

Implements mass transfer between liquid and vapor phases due to
cavitation (pressure dropping below vapor pressure).

Models:
- Schnerr-Sauer: Based on bubble dynamics (Rayleigh-Plesset)
- Merkle: Empirical model
- Zwart-Gerber-Belamri: Modified Rayleigh-Plesset

Based on OpenFOAM's cavitationModels.

Usage::

    from pyfoam.multiphase.cavitation import SchnerrSauer

    model = SchnerrSauer(n_b=1e13, p_v=2300.0)
    m_dot = model.compute_mass_transfer(alpha, p, rho_l, rho_v)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["SchnerrSauer", "Merkle", "ZGB"]

logger = logging.getLogger(__name__)


class SchnerrSauer:
    """Schnerr-Sauer cavitation model.

    Based on the Rayleigh-Plesset equation for bubble dynamics:

        m_dot = (3 * rho_v * alpha * (1 - alpha) / R_b)
                * sign(p - p_v) * sqrt(2/3 * |p - p_v| / rho_l)

    where:
        R_b = (3 * alpha / (4 * pi * n_b))^(1/3) is the bubble radius
        n_b is the bubble number density
        p_v is the vapor pressure

    Parameters
    ----------
    n_b : float
        Bubble number density (m^-3). Default 1e13.
    p_v : float
        Vapor pressure (Pa). Default 2300.0 (water at 20°C).
    """

    def __init__(self, n_b: float = 1e13, p_v: float = 2300.0) -> None:
        self.n_b = n_b
        self.p_v = p_v

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute mass transfer rate from liquid to vapor.

        Args:
            alpha: Vapor volume fraction ``(n_cells,)``.
            p: Pressure field ``(n_cells,)``.
            rho_l: Liquid density (kg/m³).
            rho_v: Vapor density (kg/m³).

        Returns:
            ``(n_cells,)`` — mass transfer rate (positive = evaporation).
        """
        # Clamp alpha to avoid singularities
        alpha = alpha.clamp(min=1e-10, max=1.0 - 1e-10)

        # Bubble radius
        R_b = (3.0 * alpha / (4.0 * math.pi * self.n_b)).pow(1.0 / 3.0)
        R_b = R_b.clamp(min=1e-10)

        # Pressure difference
        p_diff = p - self.p_v

        # Mass transfer rate
        # When p < p_v: evaporation, m_dot > 0
        # When p > p_v: condensation, m_dot < 0
        sign = -torch.sign(p_diff)
        m_dot = (
            3.0 * rho_v * alpha * (1.0 - alpha) / R_b
            * sign * (2.0 / 3.0 * p_diff.abs() / rho_l).sqrt()
        )

        return m_dot


class Merkle:
    """Merkle empirical cavitation model.

    m_dot = C_evap * rho_v * min(p - p_v, 0) / (0.5 * rho_l * U_inf² * t_inf)
           + C_cond * rho_v * max(p - p_v, 0) / (0.5 * rho_l * U_inf² * t_inf)

    Parameters
    ----------
    C_evap : float
        Evaporation coefficient. Default 1.0.
    C_cond : float
        Condensation coefficient. Default 1.0.
    p_v : float
        Vapor pressure (Pa). Default 2300.0.
    U_inf : float
        Reference velocity (m/s). Default 1.0.
    t_inf : float
        Reference time scale (s). Default 1.0.
    """

    def __init__(
        self,
        C_evap: float = 1.0,
        C_cond: float = 1.0,
        p_v: float = 2300.0,
        U_inf: float = 1.0,
        t_inf: float = 1.0,
    ) -> None:
        self.C_evap = C_evap
        self.C_cond = C_cond
        self.p_v = p_v
        self.U_inf = U_inf
        self.t_inf = t_inf

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute mass transfer rate.

        Positive = evaporation (p < p_v), negative = condensation (p > p_v).
        """
        p_diff = p - self.p_v
        ref = 0.5 * rho_l * self.U_inf ** 2 * self.t_inf

        # Evaporation when p < p_v (p_diff < 0)
        evap = self.C_evap * rho_v * torch.min(p_diff, torch.zeros_like(p_diff)) / ref
        # Condensation when p > p_v (p_diff > 0) — negative m_dot
        cond = self.C_cond * rho_v * torch.max(p_diff, torch.zeros_like(p_diff)) / ref

        # evap is negative (p_diff < 0), so negate for positive evaporation
        return -evap - cond


class ZGB:
    """Zwart-Gerber-Belamri cavitation model.

    Modified Rayleigh-Plesset model with empirical coefficients.

    m_dot = C_evap * 3 * rho_v * alpha_nuc * (1 - alpha) / R_b
            * sqrt(2/3 * max(p_v - p, 0) / rho_l)
          - C_cond * 3 * rho_v * alpha / R_b
            * sqrt(2/3 * max(p - p_v, 0) / rho_l)

    Parameters
    ----------
    C_evap : float
        Evaporation coefficient. Default 0.02.
    C_cond : float
        Condensation coefficient. Default 0.01.
    alpha_nuc : float
        Nucleation site volume fraction. Default 5e-4.
    p_v : float
        Vapor pressure (Pa). Default 2300.0.
    R_b : float
        Bubble radius (m). Default 1e-6.
    """

    def __init__(
        self,
        C_evap: float = 0.02,
        C_cond: float = 0.01,
        alpha_nuc: float = 5e-4,
        p_v: float = 2300.0,
        R_b: float = 1e-6,
    ) -> None:
        self.C_evap = C_evap
        self.C_cond = C_cond
        self.alpha_nuc = alpha_nuc
        self.p_v = p_v
        self.R_b = R_b

    def compute_mass_transfer(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        rho_l: float,
        rho_v: float,
    ) -> torch.Tensor:
        """Compute mass transfer rate."""
        p_diff = p - self.p_v

        evap = (
            self.C_evap * 3.0 * rho_v * self.alpha_nuc * (1.0 - alpha) / self.R_b
            * (2.0 / 3.0 * torch.max(-p_diff, torch.zeros_like(p_diff)) / rho_l).sqrt()
        )

        cond = (
            self.C_cond * 3.0 * rho_v * alpha / self.R_b
            * (2.0 / 3.0 * torch.max(p_diff, torch.zeros_like(p_diff)) / rho_l).sqrt()
        )

        return evap - cond
