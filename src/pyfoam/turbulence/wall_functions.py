"""
Turbulence wall functions for RANS models.

Implements wall-function computations for turbulent viscosity, turbulent
kinetic energy, and specific dissipation rate at wall boundaries.  These
are used by the boundary condition classes in ``pyfoam.boundary.wall_function``
but can also be used standalone.

Wall functions bridge the viscous sublayer (y⁺ < 5) and log-law region
(y⁺ > 30) using analytical solutions:

- Viscous sublayer: u⁺ = y⁺
- Log-law: u⁺ = (1/κ) ln(E y⁺)

where y⁺ = u_τ y / ν and u_τ = √(τ_w / ρ).

Functions:
    compute_nut_wall: Turbulent viscosity at wall faces (nutkWallFunction).
    compute_k_wall: Turbulent kinetic energy at wall faces (kqRWallFunction).
    compute_omega_wall: Specific dissipation rate at wall faces.
    compute_epsilon_wall: Dissipation rate at wall faces.
    compute_y_plus: Non-dimensional wall distance.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "compute_nut_wall",
    "compute_nut_low_re_wall",
    "compute_k_wall",
    "compute_omega_wall",
    "compute_epsilon_wall",
    "compute_y_plus",
]


# Physical constants
_KAPPA: float = 0.41  # Von Karman constant
_E: float = 9.8       # Log-law constant
_C_MU: float = 0.09   # k-ε model constant


def compute_y_plus(
    u_tau: torch.Tensor,
    y: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compute non-dimensional wall distance y⁺.

    y⁺ = u_τ y / ν

    Args:
        u_tau: Friction velocity at wall faces, shape ``(n_faces,)``.
        y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity.

    Returns:
        y⁺ at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    u_tau = u_tau.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    y_plus = u_tau * y / max(nu, 1e-30)
    return y_plus.clamp(min=1e-4)


def compute_nut_wall(
    k: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    kappa: float = _KAPPA,
    E: float = _E,
    C_mu: float = _C_MU,
) -> torch.Tensor:
    """Compute turbulent viscosity at wall faces (nutkWallFunction).

    Uses the k-equation wall function approach:

        u_τ = C_μ^{1/4} √k
        y⁺ = u_τ y / ν
        ν_t = κ u_τ y / ln(E y⁺)

    Args:
        k: Turbulent kinetic energy at wall-adjacent cells,
            shape ``(n_faces,)``.
        y: Wall-normal distance from cell centre to face,
            shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity.
        kappa: Von Karman constant (default 0.41).
        E: Log-law constant (default 9.8).
        C_mu: k-ε model constant (default 0.09).

    Returns:
        ν_t at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    k = k.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    # Friction velocity: u_τ = C_μ^{1/4} √k
    u_tau = C_mu**0.25 * torch.sqrt(k.clamp(min=1e-16))

    # y⁺
    y_plus = compute_y_plus(u_tau, y, nu)

    # ν_t = κ u_τ y / ln(E y⁺)
    nut = kappa * u_tau * y / torch.log(E * y_plus)

    # Ensure non-negative
    return nut.clamp(min=0.0)


def compute_k_wall(
    u_tau: torch.Tensor,
    C_mu: float = _C_MU,
) -> torch.Tensor:
    """Compute turbulent kinetic energy at wall faces (kqRWallFunction).

    Under local equilibrium assumption:

        k = u_τ² / √C_μ

    Args:
        u_tau: Friction velocity at wall faces, shape ``(n_faces,)``.
        C_mu: k-ε model constant (default 0.09).

    Returns:
        k at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    u_tau = u_tau.to(device=device, dtype=dtype)
    return u_tau**2 / math.sqrt(C_mu)


def compute_omega_wall(
    k: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    kappa: float = _KAPPA,
    C_mu: float = _C_MU,
) -> torch.Tensor:
    """Compute specific dissipation rate at wall faces.

    Near-wall ω based on Wilcox (2006):

        ω = √k / (C_μ^{1/4} κ y)

    This provides the correct asymptotic behaviour ω → ∞ as y → 0.

    Args:
        k: Turbulent kinetic energy at wall-adjacent cells,
            shape ``(n_faces,)``.
        y: Wall-normal distance from cell centre to face,
            shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity (unused, kept for API consistency).
        kappa: Von Karman constant (default 0.41).
        C_mu: k-ε model constant (default 0.09).

    Returns:
        ω at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    k = k.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    # ω = √k / (C_μ^{1/4} κ y)
    omega = torch.sqrt(k.clamp(min=1e-16)) / (C_mu**0.25 * kappa * y.clamp(min=1e-10))

    return omega.clamp(min=1e-10)


def compute_epsilon_wall(
    k: torch.Tensor,
    y: torch.Tensor,
    kappa: float = _KAPPA,
    C_mu: float = _C_MU,
) -> torch.Tensor:
    """Compute dissipation rate at wall faces.

    Near-wall ε based on local equilibrium:

        ε = C_μ^{3/4} k^{3/2} / (κ y)

    Args:
        k: Turbulent kinetic energy at wall-adjacent cells,
            shape ``(n_faces,)``.
        y: Wall-normal distance from cell centre to face,
            shape ``(n_faces,)``.
        kappa: Von Karman constant (default 0.41).
        C_mu: k-ε model constant (default 0.09).

    Returns:
        ε at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    k = k.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    # ε = C_μ^{3/4} k^{3/2} / (κ y)
    epsilon = C_mu**0.75 * k.clamp(min=1e-16) ** 1.5 / (kappa * y.clamp(min=1e-10))

    return epsilon.clamp(min=1e-10)


def compute_nut_low_re_wall(
    nu: float,
) -> torch.Tensor:
    """Compute turbulent viscosity for low-Re wall function.

    For low-Reynolds-number models, the wall function sets ν_t = 0
    at the wall (no wall function applied, viscous sublayer resolved).

    This is used by nutLowReWallFunction boundary condition.

    Args:
        nu: Molecular kinematic viscosity (unused, kept for API consistency).

    Returns:
        ν_t = 0 at each wall face (scalar, to be broadcast).
    """
    # Low-Re wall function: ν_t = 0 at wall
    # The wall-adjacent cell resolves the viscous sublayer
    return torch.tensor(0.0)
