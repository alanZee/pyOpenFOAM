"""
Thin film flow model for multiphase simulations.

Models thin liquid film on walls or surfaces, solving for film
thickness and film velocity. Used in Lagrangian film tracking
(``surfaceFilmModels`` in OpenFOAM) or Eulerian film formulations.

Governing equations:

    Film thickness (mass conservation):
        d(delta)/dt + div(delta * U_film) = m_dot / rho_film

    Film momentum:
        d(delta * U_film)/dt + div(delta * U_film U_film)
            = -delta * grad(p)/rho + g * delta * sin(theta)
              + tau_wall * delta / mu
              + sigma * delta * div(n) / rho

    where:
        delta    = film thickness
        U_film   = film velocity (2D tangential + normal)
        sigma    = surface tension coefficient
        theta    = contact angle

Parameters:
    sigma      : surface tension (N/m)
    contact_angle : wall contact angle (rad)
    rho_film   : film density (kg/m3)
    mu_film    : film dynamic viscosity (Pa·s)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["FilmModel"]

logger = logging.getLogger(__name__)


class FilmModel:
    """Eulerian thin film flow model.

    Solves simplified film thickness and velocity transport equations
    for thin liquid films on walls. Includes surface tension, contact
    angle effects, and gravity-driven spreading.

    Parameters
    ----------
    rho_film : float
        Film density (kg/m^3). Default: 998.0 (water).
    mu_film : float
        Film dynamic viscosity (Pa·s). Default: 1.002e-3.
    sigma : float
        Surface tension coefficient (N/m). Default: 0.072 (water-air).
    contact_angle : float
        Wall contact angle (degrees). Default: 90.0.
    delta_min : float
        Minimum film thickness (m). Default: 1e-7.
    """

    def __init__(
        self,
        rho_film: float = 998.0,
        mu_film: float = 1.002e-3,
        sigma: float = 0.072,
        contact_angle: float = 90.0,
        delta_min: float = 1e-7,
    ) -> None:
        self.rho_film = rho_film
        self.mu_film = mu_film
        self.sigma = sigma
        self.contact_angle = contact_angle
        self.delta_min = delta_min

    @property
    def contact_angle_rad(self) -> float:
        """Contact angle in radians."""
        return math.radians(self.contact_angle)

    @property
    def cos_theta(self) -> float:
        """Cosine of the contact angle."""
        return math.cos(self.contact_angle_rad)

    # ------------------------------------------------------------------
    # Film thickness evolution
    # ------------------------------------------------------------------

    def film_pressure(
        self,
        delta: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute film pressure from curvature and surface tension.

        p_film = -sigma * kappa * cos(theta) / delta

        where kappa is the wall curvature and theta is the contact
        angle. This capillary pressure drives film spreading on
        curved surfaces.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        curvature : torch.Tensor
            ``(n_cells,)`` wall curvature (1/m).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` film pressure (Pa).
        """
        delta_safe = delta.clamp(min=self.delta_min)
        return -self.sigma * curvature * self.cos_theta / delta_safe

    def gravity_source(
        self,
        delta: torch.Tensor,
        wall_normal: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gravity-driven film flow source term.

        S_grav = delta * rho * g_tangent

        where g_tangent is the tangential component of gravity along
        the wall surface.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        wall_normal : torch.Tensor
            ``(n_cells, 3)`` wall unit normal vectors.
        gravity : torch.Tensor
            ``(3,)`` gravity vector.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` gravity source term (tangential).
        """
        g = gravity.to(dtype=delta.dtype, device=delta.device)

        # Tangential gravity = g - (g . n_hat) * n_hat
        g_dot_n = (g.unsqueeze(0) * wall_normal).sum(dim=1, keepdim=True)
        g_tangent = g.unsqueeze(0) - g_dot_n * wall_normal

        return delta.unsqueeze(-1) * self.rho_film * g_tangent

    # ------------------------------------------------------------------
    # Surface tension effects
    # ------------------------------------------------------------------

    def capillary_pressure(
        self,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute capillary pressure from film thickness gradient.

        p_cap = sigma * cos(theta) / delta

        This creates a pressure gradient that opposes film rupture
        and promotes wetting.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` capillary pressure (Pa).
        """
        delta_safe = delta.clamp(min=self.delta_min)
        return self.sigma * self.cos_theta / delta_safe

    # ------------------------------------------------------------------
    # Wall friction
    # ------------------------------------------------------------------

    def wall_shear_stress(
        self,
        delta: torch.Tensor,
        U_film: torch.Tensor,
    ) -> torch.Tensor:
        """Compute wall shear stress for laminar film.

        tau_w = mu * U_film / delta  (Nusselt film theory)

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        U_film : torch.Tensor
            ``(n_cells, 3)`` film velocity.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` wall shear stress (Pa).
        """
        delta_safe = delta.clamp(min=self.delta_min)
        return self.mu_film * U_film / delta_safe.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Film properties
    # ------------------------------------------------------------------

    def film_reynolds_number(
        self,
        delta: torch.Tensor,
        U_film: torch.Tensor,
    ) -> torch.Tensor:
        """Compute film Reynolds number: Re_film = rho * |U| * delta / mu.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        U_film : torch.Tensor
            ``(n_cells, 3)`` film velocity.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` film Reynolds number.
        """
        delta_safe = delta.clamp(min=self.delta_min)
        U_mag = U_film.norm(dim=1)
        return self.rho_film * U_mag * delta_safe / max(self.mu_film, 1e-30)

    def film_weber_number(
        self,
        delta: torch.Tensor,
        U_film: torch.Tensor,
    ) -> torch.Tensor:
        """Compute film Weber number: We = rho * |U|^2 * delta / sigma.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        U_film : torch.Tensor
            ``(n_cells, 3)`` film velocity.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` film Weber number.
        """
        delta_safe = delta.clamp(min=self.delta_min)
        U_mag = U_film.norm(dim=1)
        return self.rho_film * U_mag.pow(2) * delta_safe / max(self.sigma, 1e-30)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def solve_timestep(
        self,
        delta: torch.Tensor,
        U_film: torch.Tensor,
        wall_normal: torch.Tensor,
        gravity: torch.Tensor,
        curvature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single-step: compute all film quantities.

        Parameters
        ----------
        delta : torch.Tensor
            ``(n_cells,)`` film thickness (m).
        U_film : torch.Tensor
            ``(n_cells, 3)`` film velocity.
        wall_normal : torch.Tensor
            ``(n_cells, 3)`` wall unit normal vectors.
        gravity : torch.Tensor
            ``(3,)`` gravity vector.
        curvature : torch.Tensor
            ``(n_cells,)`` wall curvature (1/m).

        Returns
        -------
        dict
            Keys: ``p_film``, ``p_cap``, ``S_grav``, ``tau_w``,
            ``Re_film``, ``We_film``.
        """
        return {
            "p_film": self.film_pressure(delta, curvature),
            "p_cap": self.capillary_pressure(delta),
            "S_grav": self.gravity_source(delta, wall_normal, gravity),
            "tau_w": self.wall_shear_stress(delta, U_film),
            "Re_film": self.film_reynolds_number(delta, U_film),
            "We_film": self.film_weber_number(delta, U_film),
        }

    def __repr__(self) -> str:
        return (
            f"FilmModel(rho={self.rho_film}, mu={self.mu_film}, "
            f"sigma={self.sigma}, theta={self.contact_angle}°)"
        )
