"""
Turbulence damping models for multiphase free-surface flows.

Damps turbulence quantities (k, epsilon/omega) near phase interfaces to
prevent unphysical turbulence production at the free surface.

Models:

- :class:`TurbulenceDampingModel` — abstract base with RTS registry
- :class:`InterfaceDamping` — damps k and epsilon near the interface
  based on the local volume fraction gradient (alpha * (1 - alpha) proxy)

In OpenFOAM, this corresponds to the ``interfaceCompression`` turbulence
damping often used with VOF multiphase solvers.

Usage::

    from pyfoam.multiphase.turbulence_damping import InterfaceDamping

    model = InterfaceDamping(damping_coeff=10.0, alpha_min=0.01, alpha_max=0.99)
    k_damped = model.damp_k(alpha, k)
    epsilon_damped = model.damp_epsilon(alpha, epsilon)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceDampingModel",
    "InterfaceDamping",
]

logger = logging.getLogger(__name__)


class TurbulenceDampingModel(ABC):
    """Abstract base class for turbulence damping near interfaces.

    Subclasses implement :meth:`damp_k` and :meth:`damp_epsilon` (or
    :meth:`damp_omega`) to suppress turbulence near the phase interface.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDampingModel"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        damping_coeff : float
            Damping strength coefficient (default: 10.0).
        alpha_min : float
            Lower alpha threshold for damping region (default: 0.01).
        alpha_max : float
            Upper alpha threshold for damping region (default: 0.99).
        """
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDampingModel]) -> Type[TurbulenceDampingModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence damping model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDampingModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence damping model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    def compute_damping_factor(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the scalar damping factor from volume fraction.

        Uses the interface indicator: f = 4 * alpha * (1 - alpha), which
        peaks at 1.0 when alpha = 0.5 (the interface) and is zero when
        alpha = 0 or 1 (pure phases).  The factor is scaled by the
        damping coefficient.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)`` in [0, damping_coeff].
        """
        # Clamp alpha to [0, 1]
        alpha_c = alpha.clamp(0.0, 1.0)
        # Interface indicator: peaks at alpha = 0.5
        indicator = 4.0 * alpha_c * (1.0 - alpha_c)
        # Only apply damping in the interface region
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        factor = self.damping_coeff * indicator
        return factor * in_interface.to(factor.dtype)

    @abstractmethod
    def damp_k(self, alpha: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Apply damping to turbulent kinetic energy.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Damped turbulent kinetic energy.
        """

    @abstractmethod
    def damp_epsilon(
        self, alpha: torch.Tensor, epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Apply damping to turbulent dissipation rate.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Damped dissipation rate.
        """


@TurbulenceDampingModel.register("interfaceDamping")
class InterfaceDamping(TurbulenceDampingModel):
    """Interface turbulence damping model.

    Damps k and epsilon near the phase interface using:

        k_damped = k * exp(-damping_factor)
        epsilon_damped = epsilon * exp(-damping_factor)

    where damping_factor = damping_coeff * 4 * alpha * (1 - alpha) in the
    interface region (alpha_min < alpha < alpha_max).

    This exponential decay smoothly reduces turbulence to near zero at the
    interface while leaving the bulk phases unaffected.
    """

    def damp_k(self, alpha: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Damp k: k_damped = k * exp(-f)."""
        f = self.compute_damping_factor(alpha)
        return k * torch.exp(-f)

    def damp_epsilon(
        self, alpha: torch.Tensor, epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Damp epsilon: eps_damped = eps * exp(-f)."""
        f = self.compute_damping_factor(alpha)
        return epsilon * torch.exp(-f)

    def damp_omega(
        self, alpha: torch.Tensor, omega: torch.Tensor,
    ) -> torch.Tensor:
        """Damp omega (for omega-based models): omega_damped = omega * exp(-f).

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        omega : torch.Tensor
            Specific dissipation rate ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Damped specific dissipation rate.
        """
        f = self.compute_damping_factor(alpha)
        return omega * torch.exp(-f)
