"""
Non-linear viscosity models for generalised-Newtonian fluids.

Provides a formal ABC hierarchy with RTS (Run-Time Selection) registry
for non-Newtonian viscosity constitutive laws.

Models:

- :class:`NonLinearViscosityModel` — abstract base with RTS registry
- :class:`PowerLawViscosity` — power-law: mu = K * |gamma_dot|^(n-1)
- :class:`BirdCarreauViscosity` — Bird-Carreau four-parameter model
- :class:`CrossPowerLawViscosity` — Cross power-law model

All models implement :meth:`mu` to compute the apparent dynamic
viscosity from the local shear-strain rate magnitude.

Usage::

    from pyfoam.turbulence.non_linear_viscosity import NonLinearViscosityModel

    # Factory creation
    model = NonLinearViscosityModel.create("powerLaw", K=0.01, n=0.5)
    mu = model.mu(gamma_dot_tensor)

    # Direct creation
    from pyfoam.turbulence.non_linear_viscosity import BirdCarreauViscosity
    model = BirdCarreauViscosity(mu_0=0.05, mu_inf=0.001, lambda_=1.0, n=0.4)
    mu = model.mu(gamma_dot_tensor)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "NonLinearViscosityModel",
    "PowerLawViscosity",
    "BirdCarreauViscosity",
    "CrossPowerLawViscosity",
]

logger = logging.getLogger(__name__)

# Small value to prevent division by zero
_EPS = 1e-30


class NonLinearViscosityModel(ABC):
    """Abstract base class for non-linear viscosity constitutive laws.

    Subclasses implement :meth:`mu` to compute the apparent dynamic
    viscosity from the shear-strain rate magnitude.

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = NonLinearViscosityModel.create("powerLaw", K=0.01, n=0.5)
    """

    _registry: ClassVar[dict[str, Type["NonLinearViscosityModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a viscosity model under *name*."""

        def decorator(model_cls: Type[NonLinearViscosityModel]) -> Type[NonLinearViscosityModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Non-linear viscosity model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "NonLinearViscosityModel":
        """Factory: create a model by registered *name*.

        Args:
            name: Registered model name (e.g. ``"powerLaw"``).
            **kwargs: Model parameters.

        Returns:
            Instantiated viscosity model.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown non-linear viscosity model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute apparent viscosity from shear-strain rate magnitude.

        Args:
            gamma_dot: |gamma_dot| = sqrt(2 * S_ij * S_ij), ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` apparent dynamic viscosity.
        """

    @abstractmethod
    def mu_zero(self) -> float:
        """Return the zero-shear-rate viscosity limit.

        For models without a well-defined zero-shear limit, returns
        the viscosity at a very low strain rate.
        """

    @abstractmethod
    def mu_inf(self) -> float:
        """Return the infinite-shear-rate viscosity limit.

        For models without a well-defined infinite-shear limit, returns
        the viscosity at a very high strain rate.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Power-law model
# ---------------------------------------------------------------------------


@NonLinearViscosityModel.register("powerLaw")
class PowerLawViscosity(NonLinearViscosityModel):
    """Power-law viscosity model.

    mu = K * |gamma_dot|^(n - 1)

    Parameters
    ----------
    K : float
        Consistency index (Pa*s^n).  Default 0.01.
    n : float
        Power-law index.
        - n < 1: shear-thinning (pseudoplastic)
        - n = 1: Newtonian
        - n > 1: shear-thickening (dilatant)
    mu_min : float
        Lower bound on viscosity (default 1e-6).
    mu_max : float
        Upper bound on viscosity (default 1e4).
    """

    def __init__(
        self,
        K: float = 0.01,
        n: float = 0.5,
        mu_min: float = 1e-6,
        mu_max: float = 1e4,
    ) -> None:
        self.K = K
        self.n = n
        self.mu_min = mu_min
        self.mu_max = mu_max

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute mu = K * |gamma_dot|^(n-1), clamped to [mu_min, mu_max]."""
        gd = gamma_dot.clamp(min=_EPS)
        result = self.K * gd.pow(self.n - 1.0)
        return result.clamp(min=self.mu_min, max=self.mu_max)

    def mu_zero(self) -> float:
        """Zero-shear limit: mu -> mu_max (or infinity, clamped)."""
        if self.n < 1:
            return self.mu_max
        elif self.n > 1:
            return self.mu_min
        return self.K

    def mu_inf(self) -> float:
        """Infinite-shear limit: mu -> mu_min (or infinity, clamped)."""
        if self.n < 1:
            return self.mu_min
        elif self.n > 1:
            return self.mu_max
        return self.K

    def __repr__(self) -> str:
        return f"PowerLawViscosity(K={self.K}, n={self.n})"


# ---------------------------------------------------------------------------
# Bird-Carreau model
# ---------------------------------------------------------------------------


@NonLinearViscosityModel.register("BirdCarreau")
class BirdCarreauViscosity(NonLinearViscosityModel):
    """Bird-Carreau four-parameter viscosity model.

    mu = mu_inf_val + (mu_0 - mu_inf_val) * (1 + (lambda * |gamma_dot|)^2)^((n-1)/2)

    This model smoothly transitions between a zero-shear viscosity
    ``mu_0`` and an infinite-shear viscosity ``mu_inf_val``.

    Parameters
    ----------
    mu_0 : float
        Zero-shear viscosity (Pa*s).  Default 0.05.
    mu_inf_val : float
        Infinite-shear viscosity (Pa*s).  Default 0.001.
    lambda_ : float
        Time constant / relaxation time (s).  Default 1.0.
    n : float
        Power-law index.  Default 0.4 (shear-thinning).
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf_val: float = 0.001,
        lambda_: float = 1.0,
        n: float = 0.4,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf_val = mu_inf_val
        self.lambda_ = lambda_
        self.n = n

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute Bird-Carreau viscosity."""
        gd = gamma_dot.clamp(min=0.0)
        factor = (1.0 + (self.lambda_ * gd).pow(2)).pow((self.n - 1.0) / 2.0)
        return self.mu_inf_val + (self.mu_0 - self.mu_inf_val) * factor

    def mu_zero(self) -> float:
        """Zero-shear viscosity: mu_0."""
        return self.mu_0

    def mu_inf(self) -> float:
        """Infinite-shear viscosity: mu_inf_val."""
        return self.mu_inf_val

    def __repr__(self) -> str:
        return (
            f"BirdCarreauViscosity(mu_0={self.mu_0}, "
            f"mu_inf_val={self.mu_inf_val}, "
            f"lambda_={self.lambda_}, n={self.n})"
        )


# ---------------------------------------------------------------------------
# Cross power-law model
# ---------------------------------------------------------------------------


@NonLinearViscosityModel.register("CrossPowerLaw")
class CrossPowerLawViscosity(NonLinearViscosityModel):
    """Cross power-law viscosity model.

    mu = mu_inf_val + (mu_0 - mu_inf_val) / (1 + (lambda * |gamma_dot|)^m)

    This model is commonly used for polymer solutions and melts.

    Parameters
    ----------
    mu_0 : float
        Zero-shear viscosity (Pa*s).  Default 0.05.
    mu_inf_val : float
        Infinite-shear viscosity (Pa*s).  Default 0.001.
    lambda_ : float
        Time constant (s).  Default 1.0.
    m : float
        Cross exponent.  Default 1.0.
    """

    def __init__(
        self,
        mu_0: float = 0.05,
        mu_inf_val: float = 0.001,
        lambda_: float = 1.0,
        m: float = 1.0,
    ) -> None:
        self.mu_0 = mu_0
        self.mu_inf_val = mu_inf_val
        self.lambda_ = lambda_
        self.m = m

    def mu(self, gamma_dot: torch.Tensor) -> torch.Tensor:
        """Compute Cross power-law viscosity."""
        gd = gamma_dot.clamp(min=0.0)
        denom = 1.0 + (self.lambda_ * gd).pow(self.m)
        return self.mu_inf_val + (self.mu_0 - self.mu_inf_val) / denom

    def mu_zero(self) -> float:
        """Zero-shear viscosity: mu_0."""
        return self.mu_0

    def mu_inf(self) -> float:
        """Infinite-shear viscosity: mu_inf_val."""
        return self.mu_inf_val

    def __repr__(self) -> str:
        return (
            f"CrossPowerLawViscosity(mu_0={self.mu_0}, "
            f"mu_inf_val={self.mu_inf_val}, "
            f"lambda_={self.lambda_}, m={self.m})"
        )
