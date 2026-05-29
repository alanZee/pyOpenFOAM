"""
Xi-based premixed combustion model with flame surface density.

Implements the B-Xi two-equation model for premixed turbulent
combustion. The model transport equations describe:

    1. **Flame wrinkling (Xi)**: measures the ratio of wrinkled flame
       area to planar flame area.
    2. **Reaction progress (b)**: ranges from 0 (unburnt) to 1 (burnt).

These are coupled with the turbulence model to predict the turbulent
flame speed and burning rate.

Governing equations:

    Xi equation:
        d(rho Xi)/dt + div(rho U Xi) = div(rho D_Xi grad(Xi))
            + rho * P_Xi     (production by turbulence)
            - rho * D_Xi     (destruction / relaxation)
            + rho * S_L / E_y * |grad(Xi)|  (propagation)

    b-equation:
        d(rho b)/dt + div(rho U b) = div(rho D_b grad(b))
            + rho * omega_b  (reaction source)

    where:
        Xi  = flame wrinkling factor (= A_wrinkled / A_planar >= 1)
        b   = reaction progress variable (0 = unburnt, 1 = burnt)
        S_L = laminar flame speed
        E_y = flame efficiency function (Meneveau & Poinsot)

Reference:
    Weller, H.G. et al. (1998). "Implementation of premixed combustion
    models for transport equation flame tracking." CFDS, Technical Report.

OpenFOAM: ``XiModels/`` in combustion model framework.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["XiFluid"]

logger = logging.getLogger(__name__)


class XiFluid:
    """B-Xi two-equation premixed combustion model.

    Solves transport equations for flame wrinkling Xi and reaction
    progress b. Provides source terms for coupling with the flow
    solver.

    Parameters
    ----------
    S_L : float
        Laminar flame speed (m/s). Default: 0.3.
    sigma_y : float
        Flame stretch rate coefficient. Default: 2.0.
    Xi_min : float
        Minimum flame wrinkling. Default: 1.0 (planar flame).
    Xi_max : float
        Maximum flame wrinkling. Default: 10.0.
    b_min : float
        Minimum progress variable (burnt fraction floor). Default: 0.0.
    D_Xi : float
        Xi turbulent diffusivity multiplier. Default: 1.0.
    """

    def __init__(
        self,
        S_L: float = 0.3,
        sigma_y: float = 2.0,
        Xi_min: float = 1.0,
        Xi_max: float = 10.0,
        b_min: float = 0.0,
        D_Xi: float = 1.0,
    ) -> None:
        self.S_L = S_L
        self.sigma_y = sigma_y
        self.Xi_min = Xi_min
        self.Xi_max = Xi_max
        self.b_min = b_min
        self.D_Xi = D_Xi

    # ------------------------------------------------------------------
    # Flame efficiency function (Meneveau & Poinsot)
    # ------------------------------------------------------------------

    def flame_efficiency(
        self,
        Xi: torch.Tensor,
        u_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flame efficiency function E(Xi, u').

        Meneveau & Poinsot (1991):

            E = (Xi^2 - 1) * u'^2 / S_L^2 + Xi^2

        The efficiency function relates the flame surface density
        production to the turbulence intensity.

        Parameters
        ----------
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling factor.
        u_prime : torch.Tensor
            ``(n_cells,)`` turbulent velocity fluctuation (m/s).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` flame efficiency function.
        """
        S_L_safe = max(self.S_L, 1e-10)
        u_ratio = (u_prime / S_L_safe).pow(2)
        return (Xi.pow(2) - 1.0) * u_ratio + Xi.pow(2)

    # ------------------------------------------------------------------
    # Xi source terms
    # ------------------------------------------------------------------

    def xi_production(
        self,
        Xi: torch.Tensor,
        u_prime: torch.Tensor,
        l_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Xi production term.

        P_Xi = sigma_y * u'^2 / (S_L * l_t) * (Xi - 1) / Xi

        where l_t is the turbulent integral length scale.

        Parameters
        ----------
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling.
        u_prime : torch.Tensor
            ``(n_cells,)`` turbulent velocity fluctuation.
        l_t : torch.Tensor
            ``(n_cells,)`` turbulent length scale (m).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` Xi production rate (1/s).
        """
        S_L_safe = max(self.S_L, 1e-10)
        l_safe = l_t.clamp(min=1e-10)
        Xi_safe = Xi.clamp(min=self.Xi_min)

        return self.sigma_y * u_prime.pow(2) / (S_L_safe * l_safe) * (Xi_safe - 1.0) / Xi_safe

    def xi_destruction(
        self,
        Xi: torch.Tensor,
        u_prime: torch.Tensor,
        l_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Xi destruction (relaxation) term.

        D_Xi = sigma_y * S_L / l_t * (Xi - 1) / Xi * Xi_coeff

        Parameters
        ----------
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling.
        u_prime : torch.Tensor
            ``(n_cells,)`` turbulent velocity fluctuation.
        l_t : torch.Tensor
            ``(n_cells,)`` turbulent length scale (m).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` Xi destruction rate (1/s).
        """
        S_L_safe = max(self.S_L, 1e-10)
        l_safe = l_t.clamp(min=1e-10)
        Xi_safe = Xi.clamp(min=self.Xi_min)

        return (
            self.D_Xi * self.sigma_y * S_L_safe / l_safe
            * (Xi_safe - 1.0) / Xi_safe
        )

    # ------------------------------------------------------------------
    # Reaction progress source
    # ------------------------------------------------------------------

    def b_source(
        self,
        b: torch.Tensor,
        Xi: torch.Tensor,
        rho: torch.Tensor,
        grad_Xi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reaction progress (b) source term.

        omega_b = rho * S_L * Xi * |grad(b)|

        where the gradient of b is related to Xi through:
            |grad(b)| ~ (1 - b) * Xi / delta_L

        Simplified:
            omega_b = rho * S_L * Xi * (1 - b) / delta_flame

        Parameters
        ----------
        b : torch.Tensor
            ``(n_cells,)`` reaction progress (0 = unburnt, 1 = burnt).
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling.
        rho : torch.Tensor
            ``(n_cells,)`` density (kg/m^3).
        grad_Xi : torch.Tensor, optional
            ``(n_cells, 3)`` gradient of Xi (unused in simplified form,
            but available for more detailed models).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` reaction source (kg/m^3/s).
        """
        b_safe = b.clamp(min=self.b_min, max=1.0)
        Xi_safe = Xi.clamp(min=self.Xi_min)

        # Turbulent flame speed
        S_t = self.S_L * Xi_safe

        # Source: omega_b = rho * S_t * (1 - b)
        omega_b = rho * S_t * (1.0 - b_safe)

        return omega_b

    # ------------------------------------------------------------------
    # Turbulent flame speed
    # ------------------------------------------------------------------

    def turbulent_flame_speed(
        self,
        Xi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent flame speed: S_t = S_L * Xi.

        Parameters
        ----------
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling factor.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent flame speed (m/s).
        """
        return self.S_L * Xi.clamp(min=self.Xi_min)

    # ------------------------------------------------------------------
    # Clamping
    # ------------------------------------------------------------------

    def clamp_xi(self, Xi: torch.Tensor) -> torch.Tensor:
        """Clamp Xi to [Xi_min, Xi_max].

        Parameters
        ----------
        Xi : torch.Tensor
            Flame wrinkling field.

        Returns
        -------
        torch.Tensor
            Clamped Xi.
        """
        return Xi.clamp(min=self.Xi_min, max=self.Xi_max)

    def clamp_b(self, b: torch.Tensor) -> torch.Tensor:
        """Clamp b to [b_min, 1].

        Parameters
        ----------
        b : torch.Tensor
            Reaction progress field.

        Returns
        -------
        torch.Tensor
            Clamped b.
        """
        return b.clamp(min=self.b_min, max=1.0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def solve_timestep(
        self,
        Xi: torch.Tensor,
        b: torch.Tensor,
        rho: torch.Tensor,
        u_prime: torch.Tensor,
        l_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single-step: compute all Xi-fluid quantities.

        Parameters
        ----------
        Xi : torch.Tensor
            ``(n_cells,)`` flame wrinkling.
        b : torch.Tensor
            ``(n_cells,)`` reaction progress.
        rho : torch.Tensor
            ``(n_cells,)`` density.
        u_prime : torch.Tensor
            ``(n_cells,)`` turbulent velocity fluctuation.
        l_t : torch.Tensor
            ``(n_cells,)`` turbulent length scale.

        Returns
        -------
        dict
            Keys: ``Xi_clamped``, ``b_clamped``, ``P_Xi``, ``D_Xi``,
            ``omega_b``, ``S_t``, ``E``.
        """
        Xi_c = self.clamp_xi(Xi)
        b_c = self.clamp_b(b)
        return {
            "Xi_clamped": Xi_c,
            "b_clamped": b_c,
            "P_Xi": self.xi_production(Xi_c, u_prime, l_t),
            "D_Xi": self.xi_destruction(Xi_c, u_prime, l_t),
            "omega_b": self.b_source(b_c, Xi_c, rho),
            "S_t": self.turbulent_flame_speed(Xi_c),
            "E": self.flame_efficiency(Xi_c, u_prime),
        }

    def __repr__(self) -> str:
        return (
            f"XiFluid(S_L={self.S_L}, sigma_y={self.sigma_y}, "
            f"Xi=[{self.Xi_min}, {self.Xi_max}])"
        )
