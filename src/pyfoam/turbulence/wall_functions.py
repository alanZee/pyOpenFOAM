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
    "compute_nut_u_wall",
    "compute_nut_u_rough_wall",
    "compute_nut_u_spalding_wall",
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


def compute_nut_u_wall(
    U: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    kappa: float = _KAPPA,
    E: float = _E,
) -> torch.Tensor:
    """Compute turbulent viscosity from velocity (nutUWallFunction).

    Velocity-based nut wall function that computes u_tau from the
    velocity magnitude at the wall-adjacent cell using an iterative
    Newton-Raphson approach.

    Given |U| at the cell centre, the wall shear stress is found from:

        |U| = u_tau * [y⁺ + u_tau * y / ν * ...]  (implicit)

    which simplifies to solving:

        f(u_tau) = u_tau * (|U_parallel| / u_tau - y⁺ + f_visc(y⁺)) = 0

    For the standard log-law region:

        |U_parallel| = (u_tau / kappa) * ln(E * y_plus)

    So: u_tau = kappa * |U_parallel| / ln(E * u_tau * y / nu)

    Solved iteratively via Newton-Raphson.

    Args:
        U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
        y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity.
        kappa: Von Karman constant (default 0.41).
        E: Log-law constant (default 9.8).

    Returns:
        ν_t at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    U = U.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
    y = y.clamp(min=1e-10)
    nu_safe = max(nu, 1e-30)

    # Initial guess: u_tau = sqrt(nu * U_mag / y)
    u_tau = torch.sqrt(nu_safe * U_mag / y)

    # Newton-Raphson iterations
    for _ in range(20):
        y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
        ln_Ey = torch.log(E * y_plus).clamp(min=1e-4)

        # f(u_tau) = u_tau / kappa * ln(E * y_plus) - |U|
        f_val = u_tau / kappa * ln_Ey - U_mag

        # f'(u_tau) = (1/kappa) * [ln(E * y_plus) + 1]
        df_val = (ln_Ey + 1.0) / kappa

        delta = f_val / df_val.clamp(min=1e-10)
        u_tau = (u_tau - delta).clamp(min=1e-10)

        if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
            break

    y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
    nut = kappa * u_tau * y / torch.log(E * y_plus)

    return nut.clamp(min=0.0)


def compute_nut_u_rough_wall(
    U: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    Ks: float = 0.0,
    Cs: float = 0.5,
    kappa: float = _KAPPA,
) -> torch.Tensor:
    """Compute turbulent viscosity for rough walls (nutURoughWallFunction).

    Rough-wall variant of the velocity-based nut wall function.
    The log-law is modified for roughness:

        |U_parallel| = (u_tau / kappa) * ln(y / (Ks * Cs + y_0))

    where Ks is the equivalent sand-grain roughness height and Cs is
    the roughness constant.  When Ks = 0, reduces to smooth-wall.

    Args:
        U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
        y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity.
        Ks: Equivalent sand-grain roughness height (default 0).
        Cs: Roughness constant (default 0.5).
        kappa: Von Karman constant (default 0.41).

    Returns:
        ν_t at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    U = U.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
    y = y.clamp(min=1e-10)
    nu_safe = max(nu, 1e-30)

    # Effective roughness length: y0 = max(nu / u_tau, Ks * Cs)
    # For simplicity, use Ks * Cs + nu_0 where nu_0 is a small offset
    rough_length = Ks * Cs + nu_safe * 0.01

    # Initial guess
    u_tau = torch.sqrt(nu_safe * U_mag / y)

    for _ in range(20):
        # Effective wall distance from roughness
        y_eff = y + rough_length
        y_over_y0 = (y / rough_length).clamp(min=1.0 + 1e-6)
        ln_ratio = torch.log(y_over_y0).clamp(min=1e-4)

        # f(u_tau) = u_tau / kappa * ln(y / y0) - |U|
        f_val = u_tau / kappa * ln_ratio - U_mag
        df_val = ln_ratio / kappa

        delta = f_val / df_val.clamp(min=1e-10)
        u_tau = (u_tau - delta).clamp(min=1e-10)

        if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
            break

    # nut = kappa * u_tau * y (log-law relation)
    nut = kappa * u_tau * y

    return nut.clamp(min=0.0)


def compute_nut_u_spalding_wall(
    U: torch.Tensor,
    y: torch.Tensor,
    nu: float,
    kappa: float = _KAPPA,
    E: float = _E,
) -> torch.Tensor:
    """Compute turbulent viscosity using Spalding's unified wall function.

    Spalding's law-of-the-wall provides a single continuous expression
    valid from the viscous sublayer through the log-law region:

        y⁺ = u⁺ + exp(-κB) * [exp(κu⁺) - 1 - κu⁺ - (κu⁺)²/2 - (κu⁺)³/6]

    where B = ln(E) / κ and u⁺ = |U_parallel| / u_tau.

    This avoids the discontinuity between viscous and log-law sublayers.

    Solved iteratively via Newton-Raphson on u_tau.

    Args:
        U: Velocity vector at wall-adjacent cells, shape ``(n_faces, 3)``.
        y: Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
        nu: Molecular kinematic viscosity.
        kappa: Von Karman constant (default 0.41).
        E: Log-law constant (default 9.8).

    Returns:
        ν_t at each wall face, shape ``(n_faces,)``.
    """
    device = get_device()
    dtype = get_default_dtype()
    U = U.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    U_mag = torch.sqrt((U * U).sum(dim=-1)).clamp(min=1e-10)
    y = y.clamp(min=1e-10)
    nu_safe = max(nu, 1e-30)
    B = math.log(E) / kappa
    exp_neg_kB = math.exp(-kappa * B)

    # Initial guess
    u_tau = torch.sqrt(nu_safe * U_mag / y)

    for _ in range(30):
        u_plus = (U_mag / u_tau).clamp(min=1e-4, max=200.0)
        y_plus = u_tau * y / nu_safe

        # Spalding function: f(y_plus) = y_plus
        # g(u_plus) = u_plus + exp_neg_kB * (exp(kappa * u_plus) - 1 - kappa*u_plus - (kappa*u_plus)^2/2 - (kappa*u_plus)^3/6)
        ku = kappa * u_plus
        ku = ku.clamp(max=50.0)  # 防止溢出
        exp_ku = torch.exp(ku)
        taylor = 1.0 + ku + ku**2 / 2.0 + ku**3 / 6.0
        spalding = u_plus + exp_neg_kB * (exp_ku - taylor)

        # Residual: y_plus(u_tau) - spalding(u_plus) = 0
        residual = y_plus - spalding

        # d(y_plus)/d(u_tau) = y / nu
        dy_plus_dutau = y / nu_safe

        # d(u_plus)/d(u_tau) = -U_mag / u_tau^2
        du_plus_dutau = -U_mag / (u_tau * u_tau)

        # d(spalding)/d(u_tau) = d(spalding)/d(u_plus) * du_plus/dutau
        dspalding_du_plus = 1.0 + exp_neg_kB * (kappa * exp_ku - kappa - kappa**2 * u_plus - kappa**3 * u_plus**2 / 2.0)
        dspalding_dutau = dspalding_du_plus * du_plus_dutau

        df = dy_plus_dutau - dspalding_dutau
        df = df.clamp(min=1e-20)

        delta = residual / df
        u_tau = (u_tau - delta).clamp(min=1e-10)

        if torch.max(torch.abs(delta)) < 1e-8 * torch.max(u_tau):
            break

    y_plus = (u_tau * y / nu_safe).clamp(min=1e-4)
    nut = kappa * u_tau * y / torch.log(E * y_plus)

    return nut.clamp(min=0.0)
