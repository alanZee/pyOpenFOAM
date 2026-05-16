"""
Interphase momentum exchange models for Euler-Euler multiphase flows.

Implements drag, lift, and virtual mass forces for coupling
multiple phases in Euler-Euler formulations.

Based on OpenFOAM's twoPhaseEulerInterphaseModels.

Usage::

    from pyfoam.multiphase.interphase_models import (
        SchillerNaumannDrag,
        TomiyamaLift,
        VirtualMassForce,
    )

    drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
    K = drag.compute(alpha2, U_rel)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "SchillerNaumannDrag",
    "WenYuDrag",
    "GidaspowDrag",
    "TomiyamaLift",
    "VirtualMassForce",
]

logger = logging.getLogger(__name__)


class SchillerNaumannDrag:
    """Schiller-Naumann drag model for spherical particles.

    Cd = max(24/Re * (1 + 0.15 * Re^0.687), 0.44)
    K = 0.75 * Cd * rho_c * |U_rel| / d * alpha_dispersed

    Parameters
    ----------
    d : float
        Particle/bubble diameter (m).
    rho_c : float
        Continuous phase density (kg/m³).
    mu_c : float
        Continuous phase dynamic viscosity (Pa·s).
    """

    def __init__(self, d: float, rho_c: float, mu_c: float) -> None:
        self.d = d
        self.rho_c = rho_c
        self.mu_c = mu_c

    def compute(self, alpha: torch.Tensor, U_rel: torch.Tensor) -> torch.Tensor:
        """Compute drag coefficient K.

        Args:
            alpha: Dispersed phase volume fraction ``(n_cells,)``.
            U_rel: Relative velocity magnitude |U_d - U_c| ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` — drag coefficient K.
        """
        Re = self.rho_c * U_rel * self.d / self.mu_c
        Re = Re.clamp(min=1e-10)

        Cd = torch.where(
            Re < 1000,
            24.0 / Re * (1.0 + 0.15 * Re.pow(0.687)),
            torch.full_like(Re, 0.44),
        )

        K = 0.75 * Cd * self.rho_c * U_rel / self.d * alpha
        return K


class WenYuDrag:
    """Wen-Yu drag model for dilute dispersed flows.

    Suitable for alpha_d < 0.2.

    Parameters
    ----------
    d : float
        Particle/bubble diameter (m).
    rho_c : float
        Continuous phase density (kg/m³).
    mu_c : float
        Continuous phase dynamic viscosity (Pa·s).
    """

    def __init__(self, d: float, rho_c: float, mu_c: float) -> None:
        self.d = d
        self.rho_c = rho_c
        self.mu_c = mu_c

    def compute(self, alpha: torch.Tensor, U_rel: torch.Tensor) -> torch.Tensor:
        """Compute drag coefficient K."""
        Re = self.rho_c * U_rel * self.d / self.mu_c
        Re = Re.clamp(min=1e-10)

        Cd = torch.where(
            Re < 1000,
            24.0 / Re * (1.0 + 0.15 * Re.pow(0.687)),
            torch.full_like(Re, 0.44),
        )

        alpha_f = (1.0 - alpha).clamp(min=1e-10)
        K = 0.75 * Cd * self.rho_c * U_rel / self.d * alpha * alpha_f.pow(-2.65)
        return K


class GidaspowDrag:
    """Gidaspow drag model (Ergun + Wen-Yu).

    Combines Ergun equation for dense packing with Wen-Yu for dilute.

    Parameters
    ----------
    d : float
        Particle/bubble diameter (m).
    rho_c : float
        Continuous phase density (kg/m³).
    mu_c : float
        Continuous phase dynamic viscosity (Pa·s).
    """

    def __init__(self, d: float, rho_c: float, mu_c: float) -> None:
        self.d = d
        self.rho_c = rho_c
        self.mu_c = mu_c

    def compute(self, alpha: torch.Tensor, U_rel: torch.Tensor) -> torch.Tensor:
        """Compute drag coefficient K."""
        alpha_c = (1.0 - alpha).clamp(min=1e-10)

        # Wen-Yu part (dilute)
        Re = self.rho_c * U_rel * self.d / self.mu_c
        Re = Re.clamp(min=1e-10)

        Cd = torch.where(
            Re < 1000,
            24.0 / Re * (1.0 + 0.15 * Re.pow(0.687)),
            torch.full_like(Re, 0.44),
        )
        K_wy = 0.75 * Cd * self.rho_c * U_rel / self.d * alpha * alpha_c.pow(-2.65)

        # Ergun part (dense)
        K_ergun = (
            150.0 * alpha * self.mu_c / (self.d ** 2 * alpha_c)
            + 1.75 * self.rho_c * U_rel / self.d * alpha / alpha_c
        )

        # Switch based on alpha_c
        K = torch.where(alpha_c > 0.8, K_wy, K_ergun)
        return K


class TomiyamaLift:
    """Tomiyama lift force for bubbles.

    F_L = C_L * rho_c * alpha * (U_d - U_c) × curl(U_c)

    Parameters
    ----------
    d : float
        Bubble diameter (m).
    rho_c : float
        Continuous phase density (kg/m³).
    """

    def __init__(self, d: float, rho_c: float) -> None:
        self.d = d
        self.rho_c = rho_c

    def compute(
        self,
        alpha: torch.Tensor,
        U_rel: torch.Tensor,
        vorticity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lift force per unit volume.

        Args:
            alpha: Dispersed phase volume fraction ``(n_cells,)``.
            U_rel: Relative velocity ``(n_cells, 3)``.
            Vorticity: Continuous phase vorticity ``(n_cells, 3)``.

        Returns:
            ``(n_cells, 3)`` — lift force per unit volume.
        """
        Eo = self._eotvos_number()
        f_Eo = self._f_eotvos(Eo)

        C_L = 0.12 * f_Eo

        # F_L = C_L * rho_c * alpha * (U_rel × vorticity)
        cross = torch.cross(U_rel, vorticity, dim=1)
        F_L = C_L * self.rho_c * alpha.unsqueeze(-1) * cross

        return F_L

    def _eotvos_number(self) -> float:
        """Compute Eötvös number: Eo = g * delta_rho * d² / sigma."""
        g = 9.81
        delta_rho = abs(self.rho_c - 1.225)  # approximate
        sigma = 0.07  # default
        return g * delta_rho * self.d ** 2 / sigma

    def _f_eotvos(self, Eo: float) -> float:
        """Tomiyama's f(Eo) correlation."""
        if Eo < 4:
            return 1.0
        elif Eo < 10:
            return 1.0 - 0.001 * (Eo - 4) ** 2
        else:
            return 0.3


class VirtualMassForce:
    """Virtual mass force for accelerating dispersed phase.

    F_vm = C_vm * rho_c * alpha * D(U_d - U_c)/Dt

    Parameters
    ----------
    C_vm : float
        Virtual mass coefficient (default 0.5 for spheres).
    """

    def __init__(self, C_vm: float = 0.5) -> None:
        self.C_vm = C_vm

    def compute(
        self,
        alpha: torch.Tensor,
        rho_c: float,
        Ddt_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute virtual mass force per unit volume.

        Args:
            alpha: Dispersed phase volume fraction ``(n_cells,)``.
            rho_c: Continuous phase density.
            Ddt_rel: Material derivative of relative velocity
                ``(n_cells, 3)`` (approximated).

        Returns:
            ``(n_cells, 3)`` — virtual mass force per unit volume.
        """
        return self.C_vm * rho_c * alpha.unsqueeze(-1) * Ddt_rel
