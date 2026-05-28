"""
Enhanced turbulence damping models for multiphase near-wall flows.

Provides additional damping models beyond the basic interface damping:

- :class:`TurbulenceDamping2Model` — enhanced base with y+ awareness
- :class:`WolfhardtDamping` — Wolfhardt model for near-wall turbulence
  damping in multiphase flows

The Wolfhardt model damps turbulence in the near-wall region based on
the distance from the wall (y+) and the local volume fraction:

    damping = exp(-(y+ / A)^2) * alpha * (1 - alpha)

where A is a model constant controlling the damping range.

This is particularly important for multiphase wall-bounded flows where
the standard wall functions over-predict turbulence production when
the interface passes near the wall.

Usage::

    from pyfoam.multiphase.turbulence_damping_2 import WolfhardtDamping

    model = WolfhardtDamping(damping_coeff=5.0, y_plus_ref=50.0)
    k_damped = model.damp_k(alpha, k, y_plus=y_plus_field)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceDamping2Model",
    "WolfhardtDamping",
]

logger = logging.getLogger(__name__)


class TurbulenceDamping2Model(ABC):
    """Enhanced abstract base for turbulence damping with y+ awareness.

    Extends the basic TurbulenceDampingModel interface to include
    wall-distance (y+) information, enabling near-wall damping in
    multiphase flows.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping2Model"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 5.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        y_plus_ref: float = 50.0,
    ) -> None:
        """
        Parameters
        ----------
        damping_coeff : float
            Damping strength coefficient. Default: 5.0.
        alpha_min : float
            Lower alpha threshold for interface region. Default: 0.01.
        alpha_max : float
            Upper alpha threshold for interface region. Default: 0.99.
        y_plus_ref : float
            Reference y+ for damping range. Default: 50.0.
        """
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.y_plus_ref = y_plus_ref

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping2Model]) -> Type[TurbulenceDamping2Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Turbulence damping2 model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping2Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown turbulence damping2 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    def compute_interface_indicator(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute interface indicator: 4 * alpha * (1 - alpha).

        Peaks at 1.0 when alpha = 0.5, zero when alpha = 0 or 1.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)`` in [0, 1].

        Returns
        -------
        torch.Tensor
            Interface indicator ``(n_cells,)``.
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        return 4.0 * alpha_c * (1.0 - alpha_c)

    @abstractmethod
    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply damping to turbulent kinetic energy.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        y_plus : torch.Tensor, optional
            Non-dimensional wall distance ``(n_cells,)``.
        """

    @abstractmethod
    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply damping to turbulent dissipation rate.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.
        y_plus : torch.Tensor, optional
            Non-dimensional wall distance ``(n_cells,)``.
        """


@TurbulenceDamping2Model.register("wolfhardtDamping")
class WolfhardtDamping(TurbulenceDamping2Model):
    """Wolfhardt model for near-wall turbulence damping in multiphase flows.

    Combines interface damping (alpha-based) with near-wall y+ damping:

        f_interface = 4 * alpha * (1 - alpha)
        f_wall = exp(-(y+ / y_plus_ref)^2)
        f_total = damping_coeff * f_interface * f_wall
        k_damped = k * exp(-f_total)

    This ensures turbulence is damped both:
    1. Near the phase interface (alpha ~ 0.5)
    2. Near the wall (y+ < y_plus_ref)

    The exponential wall damping function smoothly transitions from
    full damping at y+ = 0 to no damping for y+ >> y_plus_ref.

    Parameters
    ----------
    damping_coeff : float
        Overall damping strength. Default: 5.0.
    y_plus_ref : float
        Reference y+ controlling damping range. Default: 50.0.
    alpha_min : float
        Lower alpha threshold. Default: 0.01.
    alpha_max : float
        Upper alpha threshold. Default: 0.99.
    """

    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp k using Wolfhardt model.

        k_damped = k * exp(-f_total)

        where f_total = damping_coeff * f_interface * f_wall.
        """
        device = k.device
        dtype = k.dtype

        # Interface indicator
        f_interface = self.compute_interface_indicator(alpha)

        # Wall damping
        if y_plus is not None:
            y_plus_t = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            f_wall = torch.exp(-(y_plus_t / max(self.y_plus_ref, 1e-6)) ** 2)
        else:
            f_wall = torch.ones_like(k)

        # Interface region filter
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        f_total = self.damping_coeff * f_interface * f_wall
        f_total = f_total * in_interface.to(dtype)

        return k * torch.exp(-f_total)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp epsilon using Wolfhardt model.

        epsilon_damped = epsilon * exp(-f_total)
        """
        device = epsilon.device
        dtype = epsilon.dtype

        f_interface = self.compute_interface_indicator(alpha)

        if y_plus is not None:
            y_plus_t = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            f_wall = torch.exp(-(y_plus_t / max(self.y_plus_ref, 1e-6)) ** 2)
        else:
            f_wall = torch.ones_like(epsilon)

        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        f_total = self.damping_coeff * f_interface * f_wall
        f_total = f_total * in_interface.to(dtype)

        return epsilon * torch.exp(-f_total)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        y_plus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Damp omega using Wolfhardt model.

        omega_damped = omega * exp(-f_total)
        """
        device = omega.device
        dtype = omega.dtype

        f_interface = self.compute_interface_indicator(alpha)

        if y_plus is not None:
            y_plus_t = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            f_wall = torch.exp(-(y_plus_t / max(self.y_plus_ref, 1e-6)) ** 2)
        else:
            f_wall = torch.ones_like(omega)

        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        f_total = self.damping_coeff * f_interface * f_wall
        f_total = f_total * in_interface.to(dtype)

        return omega * torch.exp(-f_total)
