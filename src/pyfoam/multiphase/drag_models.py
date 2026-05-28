"""
Drag force models for multiphase Euler-Euler flows.

Provides an abstract base class and three standard drag correlations
used in gas-solid and gas-liquid multiphase simulations.

Models:

- :class:`DragModel` — abstract base with RTS registry
- :class:`SchillerNaumannDrag` — Schiller-Naumann drag for spherical particles
- :class:`WenYuDrag` — Wen-Yu drag for dilute gas-solid flows
- :class:`GidaspowDrag` — Gidaspow drag (Ergun + Wen-Yu) for packed beds

All models register themselves via ``@DragModel.register(name)``
and can be instantiated at run-time via ``DragModel.create(name, ...)``.

Usage::

    from pyfoam.multiphase.drag_models import DragModel

    drag = DragModel.create("schillerNaumann", d=1e-3, rho_c=1.225, mu_c=1.8e-5)
    K = drag.compute(alpha, U_rel)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "DragModel",
    "SchillerNaumannDrag",
    "WenYuDrag",
    "GidaspowDrag",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DragModel(ABC):
    """Abstract base class for drag force models.

    Subclasses must implement :meth:`compute` which returns the drag
    coefficient *K* (momentum exchange coefficient) per unit volume.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @DragModel.register("schillerNaumann")
        class SchillerNaumannDrag(DragModel):
            ...

        drag = DragModel.create("schillerNaumann", d=1e-3, rho_c=1.225, mu_c=1.8e-5)
    """

    _registry: ClassVar[dict[str, Type[DragModel]]] = {}

    def __init__(self, d: float, rho_c: float, mu_c: float) -> None:
        """
        Parameters
        ----------
        d : float
            Particle/bubble diameter (m).
        rho_c : float
            Continuous phase density (kg/m^3).
        mu_c : float
            Continuous phase dynamic viscosity (Pa*s).
        """
        self.d = d
        self.rho_c = rho_c
        self.mu_c = mu_c

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a drag model class under *name*."""

        def decorator(model_cls: Type[DragModel]) -> Type[DragModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Drag model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DragModel:
        """Factory: create a drag model instance by registered *name*.

        Parameters
        ----------
        name : str
            Registered model type name.
        **kwargs
            Constructor arguments forwarded to the model class.

        Returns
        -------
        DragModel
            Instantiated drag model.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown drag model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered drag model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self, alpha: torch.Tensor, U_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag coefficient K.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity magnitude |U_d - U_c| ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Drag coefficient K ``(n_cells,)`` [kg/(m^3 s)].
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reynolds(self, U_rel: torch.Tensor) -> torch.Tensor:
        """Compute particle Reynolds number: Re = rho_c * |U_rel| * d / mu_c."""
        return (self.rho_c * U_rel * self.d / self.mu_c).clamp(min=1e-10)

    def _schiller_naumann_cd(self, Re: torch.Tensor) -> torch.Tensor:
        """Schiller-Naumann drag coefficient.

        Cd = max(24/Re * (1 + 0.15 * Re^0.687), 0.44)
        """
        return torch.where(
            Re < 1000,
            24.0 / Re * (1.0 + 0.15 * Re.pow(0.687)),
            torch.full_like(Re, 0.44),
        )


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------


@DragModel.register("schillerNaumann")
class SchillerNaumannDrag(DragModel):
    """Schiller-Naumann drag model for spherical particles.

    Standard drag correlation for isolated spheres covering the full
    Reynolds number range up to Re ~ 1000::

        Cd = max(24/Re * (1 + 0.15 * Re^0.687), 0.44)
        K  = 0.75 * Cd * rho_c * |U_rel| / d * alpha
    """

    def compute(
        self, alpha: torch.Tensor, U_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag coefficient K.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity magnitude ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Drag coefficient K ``(n_cells,)``.
        """
        Re = self._reynolds(U_rel)
        Cd = self._schiller_naumann_cd(Re)
        return 0.75 * Cd * self.rho_c * U_rel / self.d * alpha


@DragModel.register("wenYu")
class WenYuDrag(DragModel):
    """Wen-Yu drag model for dilute dispersed flows.

    Suitable for dilute suspensions where the dispersed phase volume
    fraction is low (alpha < 0.2).  Includes void-fraction correction::

        Cd = max(24/Re * (1 + 0.15 * Re^0.687), 0.44)
        K  = 0.75 * Cd * rho_c * |U_rel| / d * alpha * (1-alpha)^(-2.65)
    """

    def compute(
        self, alpha: torch.Tensor, U_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag coefficient K.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity magnitude ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Drag coefficient K ``(n_cells,)``.
        """
        Re = self._reynolds(U_rel)
        Cd = self._schiller_naumann_cd(Re)
        alpha_f = (1.0 - alpha).clamp(min=1e-10)
        return 0.75 * Cd * self.rho_c * U_rel / self.d * alpha * alpha_f.pow(-2.65)


@DragModel.register("gidaspow")
class GidaspowDrag(DragModel):
    """Gidaspow drag model (Ergun + Wen-Yu).

    Combines the Ergun equation for dense packing (alpha_c <= 0.8)
    with the Wen-Yu correlation for dilute flow (alpha_c > 0.8).

    Dense (Ergun)::

        K = 150 * alpha * mu_c / (d^2 * alpha_c)
            + 1.75 * rho_c * |U_rel| / d * alpha / alpha_c

    Dilute (Wen-Yu)::

        K = 0.75 * Cd * rho_c * |U_rel| / d * alpha * alpha_c^(-2.65)
    """

    def compute(
        self, alpha: torch.Tensor, U_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag coefficient K.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_rel : torch.Tensor
            Relative velocity magnitude ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Drag coefficient K ``(n_cells,)``.
        """
        alpha_c = (1.0 - alpha).clamp(min=1e-10)

        # Wen-Yu (dilute)
        Re = self._reynolds(U_rel)
        Cd = self._schiller_naumann_cd(Re)
        K_wy = 0.75 * Cd * self.rho_c * U_rel / self.d * alpha * alpha_c.pow(-2.65)

        # Ergun (dense)
        K_ergun = (
            150.0 * alpha * self.mu_c / (self.d ** 2 * alpha_c)
            + 1.75 * self.rho_c * U_rel / self.d * alpha / alpha_c
        )

        # Switch based on continuous-phase fraction
        return torch.where(alpha_c > 0.8, K_wy, K_ergun)
