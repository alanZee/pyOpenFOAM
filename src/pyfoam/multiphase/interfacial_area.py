"""
Interfacial area density models for multiphase Euler-Euler simulations.

Provides models for computing the interfacial area concentration (A_i/V)
between phases, which governs interphase mass, momentum, and heat transfer:

    Transfer rate ~ a_i * driving_force

where a_i is the interfacial area density (m^2/m^3 = 1/m).

Models:
    - :class:`InterfacialAreaModel` — abstract base with RTS registry
    - :class:`ConstantInterfacialArea` — constant interfacial area density
    - :class:`VariableInterfacialArea` — alpha-dependent interfacial area density

In OpenFOAM, interfacial area models are used within the ``twoPhaseSystem``
and ``multiphaseSystem`` frameworks to close interphase transfer terms::

    interfacialAreaModel  variable;

    variableCoeffs
    {
        d0      3e-3;       // reference diameter (m)
        alphaMin 1e-4;      // minimum alpha for non-zero area
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "InterfacialAreaModel",
    "ConstantInterfacialArea",
    "VariableInterfacialArea",
]


class InterfacialAreaModel(ABC):
    """Abstract base class for interfacial area density models.

    Subclasses implement :meth:`compute` to return the local interfacial
    area density (A_i / V) given the current phase configuration.

    Provides an RTS (Run-Time Selection) registry consistent with
    :class:`~pyfoam.boundary.boundary_condition.BoundaryCondition` and
    :class:`~pyfoam.multiphase.bubble_models.BubbleModel`.
    """

    _registry: ClassVar[dict[str, Type["InterfacialAreaModel"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an interfacial area model under *name*."""

        def decorator(model_cls: Type[InterfacialAreaModel]) -> Type[InterfacialAreaModel]:
            if name in cls._registry:
                raise ValueError(
                    f"InterfacialAreaModel '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialAreaModel":
        """Factory: create an interfacial area model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown InterfacialAreaModel '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density for each cell.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interfacial area density (1/m).
        """
        ...


@InterfacialAreaModel.register("constant")
class ConstantInterfacialArea(InterfacialAreaModel):
    """Constant interfacial area density model.

    The simplest model: the interfacial area density is a fixed
    constant regardless of local flow conditions:

        a_i = a_i0

    Parameters
    ----------
    a_i0 : float
        Constant interfacial area density (1/m).  Default: 1000.0.
    """

    def __init__(self, a_i0: float = 1000.0) -> None:
        self._a_i0 = a_i0

    @property
    def a_i0(self) -> float:
        """Constant interfacial area density (1/m)."""
        return self._a_i0

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return constant interfacial area density for all cells.

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction (unused).
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` tensor filled with ``a_i0``.
        """
        device = get_device()
        dtype = get_default_dtype()
        return torch.full((n_cells,), self._a_i0, device=device, dtype=dtype)


@InterfacialAreaModel.register("variable")
class VariableInterfacialArea(InterfacialAreaModel):
    """Alpha-dependent interfacial area density model.

    Computes the interfacial area density from the local volume
    fraction and a reference Sauter mean diameter:

        a_i = 6 * alpha * (1 - alpha) / d_0

    The ``alpha * (1 - alpha)`` factor ensures zero interfacial area
    at pure phases (alpha = 0 or 1) and maximum area at alpha = 0.5.
    The factor of 6 comes from the sphere surface-to-volume ratio.

    For dilute systems (alpha << 1), this simplifies to:

        a_i ≈ 6 * alpha / d_0

    which is the standard dilute bubble/droplet correlation.

    Parameters
    ----------
    d0 : float
        Reference Sauter mean diameter (m).  Default: 3e-3 (3 mm).
    alpha_min : float
        Minimum volume fraction for non-zero area.  Default: 1e-4.
    """

    def __init__(
        self,
        d0: float = 3e-3,
        alpha_min: float = 1e-4,
    ) -> None:
        self._d0 = d0
        self._alpha_min = alpha_min

    @property
    def d0(self) -> float:
        """Reference Sauter mean diameter (m)."""
        return self._d0

    @property
    def alpha_min(self) -> float:
        """Minimum alpha for non-zero area."""
        return self._alpha_min

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute alpha-dependent interfacial area density.

        a_i = 6 * alpha * (1 - alpha) / d_0

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed-phase volume fraction ``(n_cells,)``.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interfacial area density (1/m).
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        # Below alpha_min, interfacial area is zero
        mask = alpha_dev >= self._alpha_min

        # a_i = 6 * alpha * (1 - alpha) / d_0
        a_i = 6.0 * alpha_dev * (1.0 - alpha_dev) / max(self._d0, 1e-20)
        a_i = torch.where(mask, a_i, torch.zeros_like(a_i))

        return a_i.clamp(min=0.0)
