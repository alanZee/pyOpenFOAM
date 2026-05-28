"""
Additional compressible wall functions for epsilon and omega.

Extends :mod:`compressible_wall_functions` with wall-function implementations
for the turbulent dissipation rate (epsilon) and specific dissipation rate
(omega) in variable-density compressible flows.

Models:

- :class:`CompressibleEpsilonWallFunction` — compressible epsilon wall function
- :class:`CompressibleOmegaWallFunction` — compressible omega wall function

These correspond to OpenFOAM's ``compressible::epsilonWallFunction`` and
``compressible::omegaWallFunction`` respectively.

In compressible flows, the wall functions must account for the local
density and viscosity variations near the wall.
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "CompressibleEpsilonWallFunction",
    "CompressibleOmegaWallFunction",
]

# Physical constants
_KAPPA: float = 0.41   # Von Karman constant
_E: float = 9.8        # Log-law constant
_C_MU: float = 0.09    # k-epsilon model constant
_BETA_1: float = 0.075 # k-omega SST beta_1


class CompressibleEpsilonWallFunction:
    """Compressible epsilon wall function.

    Computes the turbulent dissipation rate at wall faces for
    compressible flows using the local-equilibrium assumption:

        epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)

    where:
    - C_mu = 0.09 (k-epsilon model constant)
    - kappa = 0.41 (von Karman constant)
    - k is the turbulent kinetic energy at the wall-adjacent cell
    - y is the wall-normal distance from cell centre to face

    This is the compressible variant that uses the local density to
    compute the friction velocity and y+:

        u_tau = C_mu^{1/4} * sqrt(k)
        y+ = rho * u_tau * y / mu

    For y+ in the viscous sublayer (y+ < 11.225), epsilon is set to
    the molecular diffusion value:

        epsilon = 2 * nu * k / y^2

    This corresponds to OpenFOAM's ``compressible::epsilonWallFunction``.

    Parameters
    ----------
    kappa : float
        Von Karman constant. Default: 0.41.
    E : float
        Log-law constant. Default: 9.8.
    C_mu : float
        k-epsilon model constant. Default: 0.09.
    y_plus_visc : float
        Viscous sublayer y+ threshold. Default: 11.225.
    """

    def __init__(
        self,
        kappa: float = _KAPPA,
        E: float = _E,
        C_mu: float = _C_MU,
        y_plus_visc: float = 11.225,
    ) -> None:
        self.kappa = kappa
        self.E = E
        self.C_mu = C_mu
        self.y_plus_visc = y_plus_visc

    def compute(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressible epsilon at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy at wall-adjacent cells ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance from cell centre to face ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity at wall faces ``(n_faces,)``.
        rho : torch.Tensor
            Density at wall-adjacent cells ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            epsilon at each wall face ``(n_faces,)``.
        """
        device = k.device
        dtype = k.dtype

        k = k.to(device=device, dtype=dtype).clamp(min=1e-16)
        y = y.to(device=device, dtype=dtype).clamp(min=1e-30)
        mu = mu.to(device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)

        # Kinematic viscosity
        nu = mu / rho.clamp(min=1e-30)

        # Friction velocity: u_tau = C_mu^{1/4} * sqrt(k)
        u_tau = self.C_mu ** 0.25 * torch.sqrt(k)

        # Compressible y+
        y_p = rho * u_tau * y / mu.clamp(min=1e-30)
        y_p = y_p.clamp(min=1e-4)

        # Log-law epsilon: epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)
        eps_log = self.C_mu ** 0.75 * k ** 1.5 / (self.kappa * y)

        # Viscous sublayer: epsilon = 2 * nu * k / y^2
        eps_visc = 2.0 * nu * k / (y ** 2)

        # Switch based on y+
        in_viscous = y_p < self.y_plus_visc
        eps = torch.where(in_viscous, eps_visc, eps_log)

        return eps.clamp(min=1e-30)


class CompressibleOmegaWallFunction:
    """Compressible omega wall function.

    Computes the specific dissipation rate at wall faces for
    compressible flows.

    For the log-law region (y+ > viscous sublayer threshold):

        omega = u_tau / (C_mu^{1/4} * kappa * y)

    or equivalently:

        omega = sqrt(k) / (C_mu^{1/4} * kappa * y)

    For the viscous sublayer (y+ <= threshold):

        omega = 6 * nu / (beta_1 * y^2)

    where beta_1 = 0.075 is the k-omega SST model constant.

    The compressible variant uses local density to compute y+:

        y+ = rho * u_tau * y / mu

    This corresponds to OpenFOAM's ``compressible::omegaWallFunction``.

    Parameters
    ----------
    kappa : float
        Von Karman constant. Default: 0.41.
    C_mu : float
        k-epsilon model constant. Default: 0.09.
    beta_1 : float
        k-omega SST beta_1 constant. Default: 0.075.
    y_plus_visc : float
        Viscous sublayer y+ threshold. Default: 11.225.
    """

    def __init__(
        self,
        kappa: float = _KAPPA,
        C_mu: float = _C_MU,
        beta_1: float = _BETA_1,
        y_plus_visc: float = 11.225,
    ) -> None:
        self.kappa = kappa
        self.C_mu = C_mu
        self.beta_1 = beta_1
        self.y_plus_visc = y_plus_visc

    def compute(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressible omega at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy at wall-adjacent cells ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance from cell centre to face ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity at wall faces ``(n_faces,)``.
        rho : torch.Tensor
            Density at wall-adjacent cells ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            omega at each wall face ``(n_faces,)``.
        """
        device = k.device
        dtype = k.dtype

        k = k.to(device=device, dtype=dtype).clamp(min=1e-16)
        y = y.to(device=device, dtype=dtype).clamp(min=1e-30)
        mu = mu.to(device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)

        # Kinematic viscosity
        nu = mu / rho.clamp(min=1e-30)

        # Friction velocity
        u_tau = self.C_mu ** 0.25 * torch.sqrt(k)

        # Compressible y+
        y_p = rho * u_tau * y / mu.clamp(min=1e-30)
        y_p = y_p.clamp(min=1e-4)

        # Log-law omega: omega = u_tau / (C_mu^{1/4} * kappa * y)
        omega_log = u_tau / (self.C_mu ** 0.25 * self.kappa * y)

        # Viscous sublayer: omega = 6 * nu / (beta_1 * y^2)
        omega_visc = 6.0 * nu / (self.beta_1 * y ** 2)

        # Switch based on y+
        in_viscous = y_p < self.y_plus_visc
        omega = torch.where(in_viscous, omega_visc, omega_log)

        return omega.clamp(min=1e-10)
