"""
Transport models for viscosity.

Provides temperature-dependent dynamic viscosity models:

- **ConstantViscosity**: μ = const (isothermal flows)
- **Sutherland**: μ = μ_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)

These models are used by compressible solvers to compute the
viscous stress tensor and thermal diffusivity.

Usage::

    from pyfoam.thermophysical.transport_model import Sutherland

    transport = Sutherland(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
    mu = transport.mu(T=300.0)  # dynamic viscosity at 300 K
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TransportModel",
    "ConstantViscosity",
    "Sutherland",
]

logger = logging.getLogger(__name__)


class TransportModel(ABC):
    """Abstract base class for transport (viscosity) models.

    Subclasses must implement :meth:`mu` to return dynamic viscosity.
    """

    @abstractmethod
    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute dynamic viscosity.

        Args:
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Dynamic viscosity (Pa·s) — same shape as input.
        """

    def nu(
        self,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute kinematic viscosity: ν = μ / ρ.

        Args:
            T: Temperature (K).
            rho: Density (kg/m³).

        Returns:
            Kinematic viscosity (m²/s).
        """
        return self.mu(T) / rho


class ConstantViscosity(TransportModel):
    """Constant dynamic viscosity model: μ = μ_0.

    Parameters
    ----------
    mu : float
        Constant dynamic viscosity (Pa·s). Default 1.8e-5 (air at STP).

    Examples::

        transport = ConstantViscosity(mu=1.8e-5)
        mu = transport.mu(T=300.0)  # always 1.8e-5
    """

    def __init__(self, mu: float = 1.8e-5) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self._mu = mu

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Return constant dynamic viscosity.

        Args:
            T: Temperature (K) — ignored.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if isinstance(T, torch.Tensor):
            return torch.full_like(T, self._mu)
        return torch.tensor(self._mu, dtype=dtype, device=device)

    def __repr__(self) -> str:
        return f"ConstantViscosity(mu={self._mu})"


class Sutherland(TransportModel):
    """Sutherland's law for temperature-dependent viscosity.

    μ = μ_ref * (T / T_ref)^(3/2) * (T_ref + S) / (T + S)

    Valid for air from ~100 K to ~2000 K.

    Parameters
    ----------
    mu_ref : float
        Reference dynamic viscosity (Pa·s) at T_ref.
        Default 1.716e-5 (air at 273.15 K).
    T_ref : float
        Reference temperature (K). Default 273.15.
    S : float
        Sutherland constant (K). Default 110.4 for air.

    Examples::

        sutherland = Sutherland()
        mu = sutherland.mu(T=300.0)  # ~1.846e-5 Pa·s
    """

    def __init__(
        self,
        mu_ref: float = 1.716e-5,
        T_ref: float = 273.15,
        S: float = 110.4,
    ) -> None:
        if mu_ref <= 0:
            raise ValueError(f"mu_ref must be positive, got {mu_ref}")
        if T_ref <= 0:
            raise ValueError(f"T_ref must be positive, got {T_ref}")
        if S <= 0:
            raise ValueError(f"S must be positive, got {S}")

        self._mu_ref = mu_ref
        self._T_ref = T_ref
        self._S = S

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute dynamic viscosity using Sutherland's law.

        μ = μ_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)

        Args:
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)  # prevent division by zero

        T_ratio = T_safe / self._T_ref
        return (
            self._mu_ref
            * T_ratio.pow(1.5)
            * (self._T_ref + self._S)
            / (T_safe + self._S)
        )

    def __repr__(self) -> str:
        return (
            f"Sutherland(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S})"
        )
