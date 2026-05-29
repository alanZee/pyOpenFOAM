"""
Dense particle-laden flow model (Eulerian two-fluid).

Models dense gas-solid or liquid-solid flows where the particle phase
is treated as a continuum with its own momentum equation. Accounts for
particle-particle interactions via the granular kinetic theory
(gas-solid) or empirical packing corrections (liquid-solid).

Key physics:

    - **Drag**: Gidaspow blend of Ergun (packed bed) and Wen-Yu
      (dilute) drag laws.
    - **Packing limit**: volume fraction is bounded to prevent
      unphysical overpacking.
    - **Solids pressure**: empirical gradient-based solids pressure to
      prevent particle phase collapse.
    - **Granular temperature**: optional granular temperature transport
      for kinetic-theory-based closures.

Governing equations (simplified two-fluid):

    Particle phase:
        d(alpha_p * rho_p)/dt + div(alpha_p * rho_p U_p) = 0
        d(alpha_p * rho_p U_p)/dt + div(alpha_p * rho_p U_p U_p)
            = -alpha_p * grad(p) - grad(p_s) + div(tau_p)
              + alpha_p * rho_p * g + F_drag

    where:
        alpha_p = particle volume fraction
        rho_p   = particle density
        p_s     = solids pressure
        F_drag  = interphase drag force

Parameters:
    d_p       : particle diameter (m)
    rho_p     : particle material density (kg/m3)
    alpha_max : maximum packing volume fraction (alpha_p ≤ alpha_max)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["DenseParticleFluid"]

logger = logging.getLogger(__name__)


class DenseParticleFluid:
    """Dense particle-laden Eulerian model.

    Provides closures for drag, solids pressure, and packing corrections
    in dense particle-laden flows. Uses the Gidaspow drag blend and
    empirical solids pressure.

    Parameters
    ----------
    d_p : float
        Particle diameter (m). Default: 1e-4 (100 µm).
    rho_p : float
        Particle material density (kg/m^3). Default: 2500.0.
    rho_f : float
        Fluid density (kg/m^3). Default: 1.225.
    mu_f : float
        Fluid dynamic viscosity (Pa·s). Default: 1.8e-5.
    alpha_max : float
        Maximum packing volume fraction. Default: 0.63.
    e_p : float
        Particle-particle restitution coefficient. Default: 0.9.
    """

    def __init__(
        self,
        d_p: float = 1e-4,
        rho_p: float = 2500.0,
        rho_f: float = 1.225,
        mu_f: float = 1.8e-5,
        alpha_max: float = 0.63,
        e_p: float = 0.9,
    ) -> None:
        if d_p <= 0:
            raise ValueError(f"Particle diameter must be positive, got {d_p}")
        if alpha_max <= 0 or alpha_max > 1:
            raise ValueError(f"alpha_max must be in (0, 1], got {alpha_max}")

        self.d_p = d_p
        self.rho_p = rho_p
        self.rho_f = rho_f
        self.mu_f = mu_f
        self.alpha_max = alpha_max
        self.e_p = e_p

    # ------------------------------------------------------------------
    # Drag force (Gidaspow blend)
    # ------------------------------------------------------------------

    def _drag_coefficient_wen_yu(self, Re_p: torch.Tensor) -> torch.Tensor:
        """Wen-Yu drag coefficient for dilute regime.

        C_D = (24/Re) * (1 + 0.15 * Re^0.687) for Re < 1000
        C_D = 0.44 for Re >= 1000
        """
        Re_safe = Re_p.clamp(min=1e-10)
        Cd_low = 24.0 / Re_safe * (1.0 + 0.15 * Re_safe.pow(0.687))
        Cd_high = torch.full_like(Re_safe, 0.44)
        return torch.where(Re_safe < 1000.0, Cd_low, Cd_high)

    def drag_coefficient(
        self,
        alpha_p: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag coefficient using Gidaspow blend.

        For alpha_p < 0.2 (dilute): Wen-Yu drag.
        For alpha_p >= 0.2 (dense): Ergun equation.

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.
        U_slip : torch.Tensor
            ``(n_cells, 3)`` slip velocity magnitude is used.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interphase drag coefficient.
        """
        alpha_f = (1.0 - alpha_p).clamp(min=1e-6)
        slip_mag = U_slip.norm(dim=1)
        zero_slip = slip_mag < 1e-15
        slip_mag_safe = slip_mag.clamp(min=1e-10)
        Re_p = self.rho_f * slip_mag_safe * self.d_p / max(self.mu_f, 1e-30)

        # Wen-Yu (dilute): beta = 0.75 * Cd * Re * mu_f * alpha_f / (d^2 * alpha_f^2.65)
        Cd = self._drag_coefficient_wen_yu(Re_p)
        beta_dilute = (
            0.75 * Cd * Re_p * self.mu_f * alpha_p
            / (self.d_p ** 2 * alpha_f.pow(2.65))
        )

        # Ergun (dense): beta = 150 * alpha_p^2 * mu_f / (alpha_f * d^2)
        #                     + 1.75 * rho_f * alpha_p * |U_slip| / d
        beta_dense = (
            150.0 * alpha_p.pow(2) * self.mu_f / (alpha_f * self.d_p ** 2)
            + 1.75 * self.rho_f * alpha_p * slip_mag_safe / self.d_p
        )

        # Gidaspow blend; zero where slip velocity is zero (no drag without relative motion)
        beta = torch.where(alpha_p < 0.2, beta_dilute, beta_dense)
        return beta.clamp(min=0.0) * (~zero_slip).to(beta.dtype)

    def compute_drag_force(
        self,
        alpha_p: torch.Tensor,
        U_p: torch.Tensor,
        U_f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interphase drag force per unit volume.

        F_drag = beta * (U_f - U_p)

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.
        U_p : torch.Tensor
            ``(n_cells, 3)`` particle velocity.
        U_f : torch.Tensor
            ``(n_cells, 3)`` fluid velocity.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` drag force on the particle phase.
        """
        U_slip = U_f - U_p
        beta = self.drag_coefficient(alpha_p, U_slip)
        return beta.unsqueeze(-1) * U_slip

    # ------------------------------------------------------------------
    # Solids pressure
    # ------------------------------------------------------------------

    def solids_pressure(
        self,
        alpha_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute empirical solids pressure.

        Uses the Syamlal-Rogers-O'Brien (SRO) correlation:

            p_s = rho_p * alpha_p^2 * g_0 * Theta

        where g_0 is the radial distribution function and Theta is
        the granular temperature. The simplified algebraic form used
        here is:

            p_s = 10 * alpha_p^2 / (alpha_max - alpha_p)^2

        This provides a stiff repulsive pressure near packing.

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` solids pressure (Pa).
        """
        alpha_safe = alpha_p.clamp(min=0.0, max=self.alpha_max - 1e-6)
        gap = (self.alpha_max - alpha_safe).clamp(min=1e-6)
        return 10.0 * alpha_safe.pow(2) / gap.pow(2)

    def radial_distribution(self, alpha_p: torch.Tensor) -> torch.Tensor:
        """Compute radial distribution function g_0.

        Carnahan-Starling type:

            g_0 = (2 - alpha_p) / (2 * (1 - alpha_p)^3)

        Clamped near packing to avoid singularity.

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` radial distribution function value.
        """
        alpha_safe = alpha_p.clamp(min=0.0, max=self.alpha_max - 1e-6)
        denom = (1.0 - alpha_safe).pow(3).clamp(min=1e-10)
        return (2.0 - alpha_safe) / (2.0 * denom)

    # ------------------------------------------------------------------
    # Packing correction
    # ------------------------------------------------------------------

    def correct_packing(
        self,
        alpha_p: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp particle volume fraction to [0, alpha_max].

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` clamped volume fraction.
        """
        return alpha_p.clamp(min=0.0, max=self.alpha_max)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def solve_timestep(
        self,
        alpha_p: torch.Tensor,
        U_p: torch.Tensor,
        U_f: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single-step solve: compute drag, pressure, and packing.

        Parameters
        ----------
        alpha_p : torch.Tensor
            ``(n_cells,)`` particle volume fraction.
        U_p : torch.Tensor
            ``(n_cells, 3)`` particle velocity.
        U_f : torch.Tensor
            ``(n_cells, 3)`` fluid velocity.

        Returns
        -------
        dict
            Keys: ``F_drag``, ``p_s``, ``g_0``, ``alpha_corrected``.
        """
        alpha_c = self.correct_packing(alpha_p)
        F_drag = self.compute_drag_force(alpha_c, U_p, U_f)
        p_s = self.solids_pressure(alpha_c)
        g_0 = self.radial_distribution(alpha_c)
        return {
            "F_drag": F_drag,
            "p_s": p_s,
            "g_0": g_0,
            "alpha_corrected": alpha_c,
        }

    def __repr__(self) -> str:
        return (
            f"DenseParticleFluid(d_p={self.d_p}, rho_p={self.rho_p}, "
            f"rho_f={self.rho_f}, alpha_max={self.alpha_max})"
        )
