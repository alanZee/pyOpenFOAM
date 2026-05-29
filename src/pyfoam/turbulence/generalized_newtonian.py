"""
Generalised Newtonian viscosity models with RTS (Run-Time Selection) registry.

Provides yield-stress and composition-dependent viscosity constitutive laws
beyond the standard power-law / Bird-Carreau / Cross models.

Models:

- :class:`GeneralizedNewtonianViscosity` — abstract base with RTS registry
- :class:`CassonModel` — Casson yield-stress model
- :class:`HerschelBulkleyModel` — Herschel-Bulkley yield-stress + power-law
- :class:`BinghamModel` — Bingham plastic (special case of Herschel-Bulkley)
- :class:`QuemadaModel` — Quemada suspension model (volume-fraction dependent)
- :class:`StrainRateFunctionModel` — user-defined arbitrary viscosity function

Usage::

    from pyfoam.turbulence.generalized_newtonian import GeneralizedNewtonianViscosity

    # Factory creation
    model = GeneralizedNewtonianViscosity.create("Casson", tau_y=0.1, mu_inf=0.001)
    mu = model.mu(shear_rate_tensor, mu_inf=0.001, mu_0=1.0)

    # Direct creation
    from pyfoam.turbulence.generalized_newtonian import HerschelBulkleyModel
    model = HerschelBulkleyModel(tau_y=0.1, K=0.5, n=0.7)
    mu = model.mu(shear_rate_tensor)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "GeneralizedNewtonianViscosity",
    "CassonModel",
    "HerschelBulkleyModel",
    "BinghamModel",
    "QuemadaModel",
    "StrainRateFunctionModel",
]

logger = logging.getLogger(__name__)

# Small value to prevent division by zero
_EPS = 1e-30


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class GeneralizedNewtonianViscosity(ABC):
    """Abstract base class for generalised Newtonian viscosity models.

    Subclasses implement :meth:`mu` to compute the apparent dynamic
    viscosity from the local shear-strain rate.

    RTS (Run-Time Selection) registry allows string-based lookup::

        model = GeneralizedNewtonianViscosity.create("Casson", tau_y=0.1)
    """

    _registry: ClassVar[dict[str, Type["GeneralizedNewtonianViscosity"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a viscosity model under *name*."""

        def decorator(
            model_cls: Type[GeneralizedNewtonianViscosity],
        ) -> Type[GeneralizedNewtonianViscosity]:
            if name in cls._registry:
                raise ValueError(
                    f"Generalised Newtonian viscosity model '{name}' is already "
                    f"registered to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "GeneralizedNewtonianViscosity":
        """Factory: create a model by registered *name*.

        Args:
            name: Registered model name (e.g. ``"Casson"``).
            **kwargs: Model parameters.

        Returns:
            Instantiated viscosity model.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown generalised Newtonian viscosity model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model type names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute apparent dynamic viscosity.

        Args:
            shear_rate: |gamma_dot| = sqrt(2 * S_ij * S_ij), ``(n_cells,)``.
            mu_inf: Infinite-shear viscosity limit (Pa*s).  Used by models
                that need an explicit infinite-shear reference.
            mu_0: Zero-shear viscosity limit (Pa*s).  Used by models that
                need an explicit zero-shear reference.

        Returns:
            ``(n_cells,)`` apparent dynamic viscosity (Pa*s).
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
# Casson model
# ---------------------------------------------------------------------------


@GeneralizedNewtonianViscosity.register("Casson")
class CassonModel(GeneralizedNewtonianViscosity):
    """Casson yield-stress viscosity model.

    .. math::

        \\sqrt{\\mu} = \\sqrt{\\tau_y / \\dot{\\gamma}} + \\sqrt{\\mu_{\\infty}}

    when :math:`\\tau_y / \\dot{\\gamma} > 0`, and
    :math:`\\mu = \\mu_{\\infty}` when the shear rate is zero.

    Parameters
    ----------
    tau_y : float
        Yield stress (Pa).  Default 0.01.
    mu_inf : float
        Infinite-shear-rate / plastic viscosity (Pa*s).  Default 0.001.
    """

    def __init__(
        self,
        tau_y: float = 0.01,
        mu_inf: float = 0.001,
    ) -> None:
        self.tau_y = tau_y
        self._mu_inf = mu_inf

    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute Casson viscosity.

        When shear rate > 0: mu = (sqrt(tau_y / |gd|) + sqrt(mu_inf))^2.
        When shear rate ~ 0: mu = mu_inf (the infinite-shear limit acts as
        the plastic viscosity floor).
        """
        mu_inf_eff = self._mu_inf
        gd = shear_rate.clamp(min=_EPS)
        term = torch.sqrt(self.tau_y / gd) + mu_inf_eff ** 0.5
        return term.pow(2)

    def mu_zero(self) -> float:
        """Zero-shear limit: yields a very large viscosity (tau_y / EPS)."""
        return self.tau_y / _EPS

    def mu_inf(self) -> float:
        """Infinite-shear limit: mu_inf (plastic viscosity)."""
        return self._mu_inf

    def __repr__(self) -> str:
        return f"CassonModel(tau_y={self.tau_y}, mu_inf={self._mu_inf})"


# ---------------------------------------------------------------------------
# Herschel-Bulkley model
# ---------------------------------------------------------------------------


@GeneralizedNewtonianViscosity.register("HerschelBulkley")
class HerschelBulkleyModel(GeneralizedNewtonianViscosity):
    """Herschel-Bulkley yield-stress + power-law viscosity model.

    .. math::

        \\tau = \\tau_y + K \\dot{\\gamma}^n
        \\quad (\\tau > \\tau_y)

    When :math:`\\tau \\le \\tau_y` the fluid behaves as a rigid body
    (:math:`\\dot{\\gamma} = 0`).

    The effective viscosity is:

    .. math::

        \\mu = \\tau_y / \\dot{\\gamma} + K \\dot{\\gamma}^{n-1}

    Parameters
    ----------
    tau_y : float
        Yield stress (Pa).  Default 0.01.
    K : float
        Consistency index (Pa*s^n).  Default 0.01.
    n : float
        Power-law index.  n < 1: shear-thinning; n > 1: shear-thickening.
        Default 0.5.
    """

    def __init__(
        self,
        tau_y: float = 0.01,
        K: float = 0.01,
        n: float = 0.5,
    ) -> None:
        self.tau_y = tau_y
        self.K = K
        self.n = n

    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute Herschel-Bulkley effective viscosity.

        mu = tau_y / |gd| + K * |gd|^(n-1)

        Regularised for |gd| -> 0 by clamping shear rate to _EPS.
        """
        gd = shear_rate.clamp(min=_EPS)
        return self.tau_y / gd + self.K * gd.pow(self.n - 1.0)

    def mu_zero(self) -> float:
        """Zero-shear limit: very large (yield stress dominates)."""
        return self.tau_y / _EPS

    def mu_inf(self) -> float:
        """Infinite-shear limit.

        For n < 1: K * |gd|^(n-1) -> 0, so mu -> 0 (yield stress
        contribution vanishes at infinite shear).  Return a small value.
        For n = 1: mu -> K (Bingham limit).
        For n > 1: mu -> infinity (clamped).
        """
        if self.n < 1:
            return 1e-6
        elif self.n == 1:
            return self.K
        return 1e6

    def __repr__(self) -> str:
        return (
            f"HerschelBulkleyModel(tau_y={self.tau_y}, "
            f"K={self.K}, n={self.n})"
        )


# ---------------------------------------------------------------------------
# Bingham model
# ---------------------------------------------------------------------------


@GeneralizedNewtonianViscosity.register("Bingham")
class BinghamModel(GeneralizedNewtonianViscosity):
    """Bingham plastic viscosity model.

    A special case of the Herschel-Bulkley model with :math:`n = 1`:

    .. math::

        \\tau = \\tau_y + \\mu_{\\infty} \\dot{\\gamma}
        \\quad (\\tau > \\tau_y)

    The effective viscosity is:

    .. math::

        \\mu = \\tau_y / \\dot{\\gamma} + \\mu_{\\infty}

    Parameters
    ----------
    tau_y : float
        Yield stress (Pa).  Default 0.01.
    mu_inf : float
        Plastic (infinite-shear) viscosity (Pa*s).  Default 0.001.
    """

    def __init__(
        self,
        tau_y: float = 0.01,
        mu_inf: float = 0.001,
    ) -> None:
        self.tau_y = tau_y
        self._mu_inf = mu_inf

    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute Bingham effective viscosity.

        mu = tau_y / |gd| + mu_inf

        Regularised for |gd| -> 0 by clamping shear rate to _EPS.
        """
        gd = shear_rate.clamp(min=_EPS)
        return self.tau_y / gd + self._mu_inf

    def mu_zero(self) -> float:
        """Zero-shear limit: very large (yield stress dominates)."""
        return self.tau_y / _EPS

    def mu_inf(self) -> float:
        """Infinite-shear limit: mu_inf (plastic viscosity)."""
        return self._mu_inf

    def __repr__(self) -> str:
        return f"BinghamModel(tau_y={self.tau_y}, mu_inf={self._mu_inf})"


# ---------------------------------------------------------------------------
# Quemada model
# ---------------------------------------------------------------------------


@GeneralizedNewtonianViscosity.register("Quemada")
class QuemadaModel(GeneralizedNewtonianViscosity):
    """Quemada suspension viscosity model.

    Models the viscosity of suspensions as a function of volume fraction
    and shear rate.

    .. math::

        \\mu = \\mu_{\\infty} (1 - 0.5 k \\phi)^{-2}

    where the structural parameter :math:`k` depends on shear rate:

    .. math::

        k = k_0 + k_{\\infty} (\\dot{\\gamma} / \\dot{\\gamma}_{\\text{ref}})^{0.5}

    Parameters
    ----------
    phi : float
        Volume fraction of suspended phase (dimensionless, 0-1).
        Default 0.3.
    k0 : float
        Zero-shear structural parameter.  Default 2.5.
    k_inf : float
        Infinite-shear structural parameter offset.  Default 0.5.
    gamma_dot_ref : float
        Reference shear rate for the structural parameter (1/s).
        Default 1.0.
    mu_inf : float
        Solvent / matrix viscosity (Pa*s).  Default 0.001.
    """

    def __init__(
        self,
        phi: float = 0.3,
        k0: float = 2.5,
        k_inf: float = 0.5,
        gamma_dot_ref: float = 1.0,
        mu_inf: float = 0.001,
    ) -> None:
        self.phi = phi
        self.k0 = k0
        self.k_inf = k_inf
        self.gamma_dot_ref = gamma_dot_ref
        self._mu_inf = mu_inf

    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute Quemada suspension viscosity.

        mu = mu_inf * (1 - 0.5 * k * phi)^(-2)

        where k = k0 + k_inf * (|gd| / gamma_dot_ref)^0.5.
        """
        gd = shear_rate.clamp(min=_EPS)
        k = self.k0 + self.k_inf * (gd / self.gamma_dot_ref).sqrt()
        denom = 1.0 - 0.5 * k * self.phi
        # Clamp denominator to avoid singularities at very high phi
        denom = denom.clamp(min=1e-6)
        return self._mu_inf * denom.pow(-2)

    def mu_zero(self) -> float:
        """Zero-shear limit: mu_inf * (1 - 0.5 * k0 * phi)^(-2)."""
        k_zero = self.k0
        denom = 1.0 - 0.5 * k_zero * self.phi
        if denom <= 0:
            return float("inf")
        return self._mu_inf * denom ** (-2)

    def mu_inf(self) -> float:
        """Infinite-shear limit: k -> k0 + k_inf * large -> very large k.

        At infinite shear: k -> infinity, so mu -> infinity
        (suspension locks up).  This is a physical limitation of the model.
        Return a large representative value.
        """
        # Use a representative high shear rate
        k_high = self.k0 + self.k_inf * 1e3
        denom = 1.0 - 0.5 * k_high * self.phi
        if denom <= 1e-6:
            return self._mu_inf * 1e12
        return self._mu_inf * denom ** (-2)

    def __repr__(self) -> str:
        return (
            f"QuemadaModel(phi={self.phi}, k0={self.k0}, "
            f"k_inf={self.k_inf}, gamma_dot_ref={self.gamma_dot_ref}, "
            f"mu_inf={self._mu_inf})"
        )


# ---------------------------------------------------------------------------
# Strain-rate function model (user-defined)
# ---------------------------------------------------------------------------


@GeneralizedNewtonianViscosity.register("strainRateFunction")
class StrainRateFunctionModel(GeneralizedNewtonianViscosity):
    """User-defined viscosity as a function of shear rate.

    Wraps an arbitrary callable ``f(gamma_dot) -> mu`` so it can be used
    anywhere the generalised Newtonian viscosity interface is expected.

    Parameters
    ----------
    func : Callable[[torch.Tensor], torch.Tensor]
        Function mapping shear-rate magnitude to apparent viscosity.
    mu_zero_val : float
        Zero-shear viscosity for :meth:`mu_zero`.  Default 1e-1.
    mu_inf_val : float
        Infinite-shear viscosity for :meth:`mu_inf`.  Default 1e-3.
    """

    def __init__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        mu_zero_val: float = 1e-1,
        mu_inf_val: float = 1e-3,
    ) -> None:
        self._func = func
        self._mu_zero_val = mu_zero_val
        self._mu_inf_val = mu_inf_val

    def mu(
        self,
        shear_rate: torch.Tensor,
        mu_inf: float = 1e-3,
        mu_0: float = 1e-1,
    ) -> torch.Tensor:
        """Compute viscosity via the user-defined function."""
        return self._func(shear_rate)

    def mu_zero(self) -> float:
        """Return the user-specified zero-shear viscosity."""
        return self._mu_zero_val

    def mu_inf(self) -> float:
        """Return the user-specified infinite-shear viscosity."""
        return self._mu_inf_val

    def __repr__(self) -> str:
        return (
            f"StrainRateFunctionModel("
            f"func={self._func.__name__ if hasattr(self._func, '__name__') else '<lambda>'}, "
            f"mu_zero_val={self._mu_zero_val}, "
            f"mu_inf_val={self._mu_inf_val})"
        )
