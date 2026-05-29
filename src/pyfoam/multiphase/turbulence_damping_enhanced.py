"""
Enhanced turbulence damping models for multiphase flows.

Provides additional damping models beyond the basic and y+-aware models:

- :class:`TurbulenceDampingEnhancedModel` — abstract base with gradient-based
  and blended damping strategies
- :class:`GradientDamping` — damps turbulence using the magnitude of the
  volume fraction gradient (not just alpha itself)
- :class:`ExponentialBlendedDamping` — smooth blending between bulk and
  interface turbulence using multiple alpha thresholds

These models address limitations of the basic alpha-based damping in
complex multiphase geometries where the interface is not well-resolved
or where the alpha field is diffuse.

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced import GradientDamping

    model = GradientDamping(damping_coeff=8.0, grad_threshold=0.1)
    k_damped = model.damp_k(alpha, k, grad_alpha=grad_alpha)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceDampingEnhancedModel",
    "GradientDamping",
    "ExponentialBlendedDamping",
]

logger = logging.getLogger(__name__)


class TurbulenceDampingEnhancedModel(ABC):
    """Enhanced abstract base for turbulence damping.

    Supports gradient-based damping indicators and multi-threshold
    blending in addition to the standard alpha-based approach.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDampingEnhancedModel"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDampingEnhancedModel]) -> Type[TurbulenceDampingEnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Enhanced damping model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDampingEnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown enhanced damping model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abstractmethod
    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply damping to turbulent kinetic energy.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k : torch.Tensor
            Turbulent kinetic energy ``(n_cells,)``.
        **kwargs
            Optional: ``grad_alpha``, ``y_plus``.
        """

    @abstractmethod
    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply damping to turbulent dissipation rate.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        epsilon : torch.Tensor
            Turbulent dissipation rate ``(n_cells,)``.
        **kwargs
            Optional: ``grad_alpha``, ``y_plus``.
        """


@TurbulenceDampingEnhancedModel.register("gradientDamping")
class GradientDamping(TurbulenceDampingEnhancedModel):
    """Gradient-based turbulence damping model.

    Damps turbulence using the magnitude of the volume fraction
    gradient rather than alpha alone. This is more physically
    appropriate for diffuse interfaces where alpha alone cannot
    reliably identify the interface.

    The damping factor is:

        f = damping_coeff * min(|grad(alpha)|, grad_max) / grad_max
            * (1 if alpha_min < alpha < alpha_max else 0)

    Parameters
    ----------
    damping_coeff : float
        Damping strength. Default: 10.0.
    alpha_min : float
        Lower alpha threshold. Default: 0.01.
    alpha_max : float
        Upper alpha threshold. Default: 0.99.
    grad_threshold : float
        Gradient magnitude threshold for full damping. Default: 0.1.
    grad_max : float
        Maximum gradient for normalisation. Default: 1.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        grad_threshold: float = 0.1,
        grad_max: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._grad_threshold = grad_threshold
        self._grad_max = max(grad_max, 1e-10)

    @property
    def grad_threshold(self) -> float:
        """Gradient threshold for non-zero damping."""
        return self._grad_threshold

    @property
    def grad_max(self) -> float:
        """Gradient magnitude for full damping."""
        return self._grad_max

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        grad_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute gradient-based damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        grad_alpha : torch.Tensor, optional
            ``(n_cells, 3)`` gradient of alpha. If None, uses the
            alpha*(1-alpha) proxy.

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)`` in [0, damping_coeff].
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        # Interface region filter
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        if grad_alpha is not None:
            grad_mag = grad_alpha.to(device=device, dtype=dtype).norm(dim=-1)
            # Normalise gradient magnitude
            grad_indicator = (grad_mag / self._grad_max).clamp(0.0, 1.0)
            # Only apply above threshold
            above_threshold = grad_mag >= self._grad_threshold
            indicator = grad_indicator * above_threshold.to(dtype)
        else:
            # Fallback: use alpha-based proxy
            indicator = 4.0 * alpha_c * (1.0 - alpha_c)

        factor = self.damping_coeff * indicator
        return factor * in_interface.to(dtype)

    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp k: k_damped = k * exp(-f)."""
        grad_alpha = kwargs.get("grad_alpha", None)
        f = self.compute_damping_factor(alpha, grad_alpha)
        return k * torch.exp(-f)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp epsilon: eps_damped = eps * exp(-f)."""
        grad_alpha = kwargs.get("grad_alpha", None)
        f = self.compute_damping_factor(alpha, grad_alpha)
        return epsilon * torch.exp(-f)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp omega: omega_damped = omega * exp(-f)."""
        grad_alpha = kwargs.get("grad_alpha", None)
        f = self.compute_damping_factor(alpha, grad_alpha)
        return omega * torch.exp(-f)


@TurbulenceDampingEnhancedModel.register("exponentialBlendedDamping")
class ExponentialBlendedDamping(TurbulenceDampingEnhancedModel):
    """Multi-threshold exponential damping model.

    Uses multiple alpha thresholds to create a smooth damping profile
    that transitions from no damping in the bulk to full damping at
    the interface.  The damping is the product of two exponential
    sigmoid functions:

        f = C * exp(-((alpha - alpha_min) / width)^2)
                 * exp(-((alpha_max - alpha) / width)^2)

    This creates a bell-shaped damping profile that peaks at the
    interface (alpha ~ 0.5) and smoothly decays to zero in both
    bulk phases.

    Parameters
    ----------
    damping_coeff : float
        Maximum damping strength. Default: 10.0.
    alpha_min : float
        Lower alpha threshold. Default: 0.01.
    alpha_max : float
        Upper alpha threshold. Default: 0.99.
    width : float
        Width of the damping transition. Default: 0.1.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        width: float = 0.1,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._width = max(width, 1e-10)

    @property
    def width(self) -> float:
        """Damping transition width."""
        return self._width

    def compute_damping_factor(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute bell-shaped damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)``.
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        w = self._width

        # Bell-shaped profile: peaks at interface, decays to zero
        exp_low = torch.exp(-((alpha_c - self.alpha_min) / w).pow(2))
        exp_high = torch.exp(-((self.alpha_max - alpha_c) / w).pow(2))

        # Product creates a profile that is non-zero only between
        # alpha_min and alpha_max
        bell = exp_low * exp_high

        return self.damping_coeff * bell

    def damp_k(
        self,
        alpha: torch.Tensor,
        k: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp k: k_damped = k * exp(-f)."""
        f = self.compute_damping_factor(alpha)
        return k * torch.exp(-f)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp epsilon: eps_damped = eps * exp(-f)."""
        f = self.compute_damping_factor(alpha)
        return epsilon * torch.exp(-f)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp omega: omega_damped = omega * exp(-f)."""
        f = self.compute_damping_factor(alpha)
        return omega * torch.exp(-f)
