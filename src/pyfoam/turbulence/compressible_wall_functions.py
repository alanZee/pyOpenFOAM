"""
Compressible wall functions for variable-density turbulence models.

Provides wall-function implementations for compressible flows where
density varies near the wall (e.g., high-speed external aerodynamics,
combustion chambers, nozzles).

Wall functions for compressible flows differ from incompressible ones by:
1. Using the mean density at the wall: rho_wall = (rho_P + rho_wall_face) / 2
2. Applying Van Driest damping for the viscous sublayer
3. Accounting for the temperature-dependent viscosity near the wall
4. Using compressible y+ definition: y+ = rho * u_tau * y / mu_wall

Models:

- :class:`CompressibleWallFunction` — abstract base
- :class:`CompressibleNutWallFunction` — compressible nut wall function
- :class:`CompressibleKWallFunction` — compressible k wall function

Usage::

    from pyfoam.turbulence.compressible_wall_functions import (
        CompressibleNutWallFunction,
        CompressibleKWallFunction,
    )

    nut_wf = CompressibleNutWallFunction(kappa=0.41, E=9.8)
    nut = nut_wf.compute(k, y, mu, rho)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "CompressibleWallFunction",
    "CompressibleNutWallFunction",
    "CompressibleKWallFunction",
]

# Physical constants
_KAPPA: float = 0.41   # Von Karman constant
_E: float = 9.8        # Log-law constant
_C_MU: float = 0.09    # k-epsilon model constant
_BETA_1: float = 0.075 # k-omega SST beta_1


class CompressibleWallFunction(ABC):
    """Abstract base class for compressible wall functions.

    Subclasses implement :meth:`compute` to evaluate the wall-function
    value given the local turbulence state and wall properties.
    """

    def __init__(
        self,
        kappa: float = _KAPPA,
        E: float = _E,
        C_mu: float = _C_MU,
    ) -> None:
        """
        Parameters
        ----------
        kappa : float
            Von Karman constant. Default: 0.41.
        E : float
            Log-law constant. Default: 9.8.
        C_mu : float
            k-epsilon model constant. Default: 0.09.
        """
        self.kappa = kappa
        self.E = E
        self.C_mu = C_mu

    @abstractmethod
    def compute(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the wall-function value.

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
            Wall-function value at each face ``(n_faces,)``.
        """

    def compute_u_tau(
        self,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute friction velocity from k.

        u_tau = C_mu^{1/4} * sqrt(k)

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            Friction velocity ``(n_faces,)``.
        """
        return self.C_mu ** 0.25 * torch.sqrt(k.clamp(min=1e-16))

    def compute_y_plus_compressible(
        self,
        u_tau: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressible y+.

        y+ = rho * u_tau * y / mu

        Parameters
        ----------
        u_tau : torch.Tensor
            Friction velocity ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity at wall ``(n_faces,)``.
        rho : torch.Tensor
            Density at wall-adjacent cells ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            Compressible y+ ``(n_faces,)``.
        """
        y_p = rho * u_tau * y / mu.clamp(min=1e-30)
        return y_p.clamp(min=1e-4)


class CompressibleNutWallFunction(CompressibleWallFunction):
    """Compressible nut wall function.

    Computes the kinematic turbulent viscosity at wall faces for
    compressible flows:

        y+ = rho * u_tau * y / mu
        u+ = (1/kappa) * ln(E * y+)
        nut = max(0, mu * (u+ / y+ - 1) / rho)

    The Van Driest damping function is applied in the viscous sublayer
    (y+ < 5) to ensure a smooth transition:

        nut = 0    for y+ < 5
        nut = ...  for y+ >= 5

    This corresponds to OpenFOAM's ``compressible::nutkWallFunction``.
    """

    def compute(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressible kinematic turbulent viscosity at wall.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity at wall ``(n_faces,)``.
        rho : torch.Tensor
            Density at wall-adjacent cells ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            nut at each wall face ``(n_faces,)``.
        """
        device = k.device
        dtype = k.dtype

        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        mu = mu.to(device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)

        # Friction velocity
        u_tau = self.compute_u_tau(k)

        # Compressible y+
        y_p = self.compute_y_plus_compressible(u_tau, y, mu, rho)

        # Log-law: u+ = (1/kappa) * ln(E * y+)
        u_plus = (1.0 / self.kappa) * torch.log(self.E * y_p)

        # nut = mu * (y+/u+ - 1) / rho
        # In viscous sublayer (y+ < 5), set nut = 0
        nut = mu * (y_p / u_plus - 1.0) / rho.clamp(min=1e-30)

        # Zero out in viscous sublayer
        in_viscous = y_p < 5.0
        nut = nut * (~in_viscous).to(dtype)

        return nut.clamp(min=0.0)


class CompressibleKWallFunction(CompressibleWallFunction):
    """Compressible k wall function.

    Computes the turbulent kinetic energy at wall faces for
    compressible flows using local equilibrium:

        k = u_tau^2 / sqrt(C_mu)

    where u_tau is the friction velocity computed from the log-law:

        u_tau = C_mu^{1/4} * sqrt(k_cell)

    This is the same as the incompressible kqRWallFunction but
    accounts for density-weighted quantities in the compressible
    formulation.

    This corresponds to OpenFOAM's ``compressible::kqRWallFunction``.
    """

    def compute(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compressible k at wall faces.

        Uses local equilibrium: k = u_tau^2 / sqrt(C_mu).

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy at wall-adjacent cells ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity ``(n_faces,)``.
        rho : torch.Tensor
            Density ``(n_faces,)``.

        Returns
        -------
        torch.Tensor
            k at wall faces ``(n_faces,)``.
        """
        device = k.device
        dtype = k.dtype

        k = k.to(device=device, dtype=dtype)

        # Friction velocity from cell-centred k
        u_tau = self.compute_u_tau(k)

        # Local equilibrium: k = u_tau^2 / sqrt(C_mu)
        k_wall = u_tau ** 2 / math.sqrt(self.C_mu)

        return k_wall.clamp(min=1e-16)

    def compute_with_van_driest(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
        A_plus: float = 26.0,
    ) -> torch.Tensor:
        """Compute k with Van Driest damping for near-wall correction.

        In the viscous sublayer, the production is damped by:

            f = 1 - exp(-y+ / A+)

        k_wall = k_cell * f  (for y+ < viscous sublayer limit)

        Parameters
        ----------
        k : torch.Tensor
            Cell-centred k ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance ``(n_faces,)``.
        mu : torch.Tensor
            Dynamic viscosity ``(n_faces,)``.
        rho : torch.Tensor
            Density ``(n_faces,)``.
        A_plus : float
            Van Driest damping constant. Default: 26.0.

        Returns
        -------
        torch.Tensor
            k at wall faces with Van Driest correction ``(n_faces,)``.
        """
        device = k.device
        dtype = k.dtype

        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        mu = mu.to(device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)

        # Friction velocity
        u_tau = self.compute_u_tau(k)

        # Compressible y+
        y_p = self.compute_y_plus_compressible(u_tau, y, mu, rho)

        # Van Driest damping function
        f_vd = 1.0 - torch.exp(-y_p / A_plus)

        # Standard k wall function with Van Driest correction
        k_eq = u_tau ** 2 / math.sqrt(self.C_mu)
        k_wall = k_eq * f_vd

        return k_wall.clamp(min=1e-16)
