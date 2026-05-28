"""
Turbulence time scale models for particle/droplet tracking and multiphase.

Provides time-scale estimates from turbulence fields for use in:
- Lagrangian particle tracking (stochastic dispersion)
- Multiphase interphase coupling (turbulent dispersion)
- Spray/breakup/coalescence models

Based on:
- Kolmogorov time scale: tau_eta = sqrt(nu / epsilon)
- Integral time scale: tau_T = k / epsilon

Usage::

    from pyfoam.turbulence.turbulence_time_scale import (
        KolmogorovTimeScale,
        IntegralTimeScale,
    )

    model = KolmogorovTimeScale(nu=1e-5)
    tau = model.compute(k, epsilon)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

__all__ = [
    "TurbulenceTimeScale",
    "KolmogorovTimeScale",
    "IntegralTimeScale",
]

logger = logging.getLogger(__name__)


class TurbulenceTimeScale(ABC):
    """Abstract base class for turbulence time scale models.

    Subclasses must implement :meth:`compute` which returns the
    characteristic time scale of turbulence.
    """

    @abstractmethod
    def compute(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the turbulence time scale.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)`` [m^2/s^2].
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)`` [m^2/s^3].

        Returns
        -------
        torch.Tensor
            Time scale ``(n_cells,)`` [s].
        """


class KolmogorovTimeScale(TurbulenceTimeScale):
    """Kolmogorov (micro) time scale.

    The Kolmogorov time scale is the smallest time scale in turbulent
    flow, characterising the dissipation-range eddies:

        tau_eta = sqrt(nu / epsilon)

    where nu is the kinematic viscosity and epsilon is the turbulent
    dissipation rate.

    This time scale governs the fastest turbulent fluctuations and is
    relevant for small-particle tracking and micro-mixing models.

    Parameters
    ----------
    nu : float
        Kinematic viscosity (m^2/s).  Default: 1e-5 (water at 20 C).
    """

    def __init__(self, nu: float = 1e-5) -> None:
        self.nu = nu

    def compute(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Kolmogorov time scale.

        tau_eta = sqrt(nu / epsilon)

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)`` [m^2/s^2].
            Not directly used but required for interface compatibility.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)`` [m^2/s^3].

        Returns
        -------
        torch.Tensor
            Kolmogorov time scale ``(n_cells,)`` [s].
        """
        eps_safe = epsilon.clamp(min=1e-30)
        return torch.sqrt(torch.tensor(self.nu, dtype=k.dtype, device=k.device) / eps_safe)

    def length_scale(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Kolmogorov length scale.

        eta = (nu^3 / epsilon)^(1/4)

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Kolmogorov length scale ``(n_cells,)`` [m].
        """
        eps_safe = epsilon.clamp(min=1e-30)
        nu = torch.tensor(self.nu, dtype=k.dtype, device=k.device)
        return (nu.pow(3) / eps_safe).pow(0.25)


class IntegralTimeScale(TurbulenceTimeScale):
    """Integral (large-eddy) time scale.

    The integral time scale characterises the largest, energy-containing
    eddies:

        tau_T = k / epsilon

    This is the most commonly used turbulence time scale in RANS models
    for interphase coupling, stochastic dispersion, and breakup/coalescence
    models.

    Parameters
    ----------
    C_T : float
        Model constant (default 1.0).  Some applications use C_T = 0.2
        for shortening the effective time scale.
    """

    def __init__(self, C_T: float = 1.0) -> None:
        self.C_T = C_T

    def compute(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute integral time scale.

        tau_T = C_T * k / epsilon

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)`` [m^2/s^2].
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)`` [m^2/s^3].

        Returns
        -------
        torch.Tensor
            Integral time scale ``(n_cells,)`` [s].
        """
        eps_safe = epsilon.clamp(min=1e-30)
        return self.C_T * k / eps_safe

    def length_scale(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute integral length scale.

        L = k^(3/2) / epsilon

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Integral length scale ``(n_cells,)`` [m].
        """
        eps_safe = epsilon.clamp(min=1e-30)
        return k.pow(1.5) / eps_safe

    def velocity_scale(
        self,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbulent velocity scale.

        u' = sqrt(2/3 * k)

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Velocity scale ``(n_cells,)`` [m/s].
        """
        return torch.sqrt(2.0 / 3.0 * k.clamp(min=0.0))
