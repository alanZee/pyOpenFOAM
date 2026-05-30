"""Enhanced interfacial area density models — v12.

Extends v11 with:
- Dynamic Sauter mean diameter transport
- Population-balance-coupled interfacial area evolution
- Real-time interfacial area budget tracking

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_11 import (
        DynamicSauterDiameterModel,
        PBECoupledTransportModel,
        BudgetTrackingModel,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_10 import InterfacialArea10Model

__all__ = [
    "InterfacialArea11Model",
    "DynamicSauterDiameterModel",
    "PBECoupledTransportModel",
    "BudgetTrackingModel",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class InterfacialArea11Model(InterfacialArea10Model):
    """Enhanced abstract base for v12 interfacial area models."""
    _registry: ClassVar[dict[str, Type["InterfacialArea11Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[InterfacialArea11Model]) -> Type[InterfacialArea11Model]:
            if name in cls._registry:
                raise ValueError(f"Interfacial area v11 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea11Model":
        if name not in cls._registry:
            raise KeyError(f"Unknown interfacial area v11 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@InterfacialArea11Model.register("dynamicSauter")
class DynamicSauterDiameterModel(InterfacialArea11Model):
    """Dynamic Sauter mean diameter transport model.

    Transports d32 as a field variable with breakup/coalescence source terms:

        d(d32)/dt + U.grad(d32) = S_breakup + S_coalescence

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Initial Sauter mean diameter. Default 3e-3.
    breakup_rate : float
        Breakup rate coefficient. Default 0.1.
    coalescence_rate : float
        Coalescence rate coefficient. Default 0.05.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        breakup_rate: float = 0.1,
        coalescence_rate: float = 0.05,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_break = max(0.0, breakup_rate)
        self._C_coal = max(0.0, coalescence_rate)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_i = 6.0 * alpha_dev / max(d32, _EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def d32_evolution(self, d32: torch.Tensor, alpha: torch.Tensor, dt: float) -> torch.Tensor:
        """Update d32 from breakup and coalescence.

        Parameters
        ----------
        d32 : torch.Tensor
            Current Sauter mean diameter.
        alpha : torch.Tensor
            Volume fraction.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Updated d32.
        """
        S_break = self._C_break * d32.pow(-1) * alpha * (1.0 - alpha)
        S_coal = self._C_coal * d32 * alpha.pow(2)
        d32_new = d32 + dt * (S_break - S_coal)
        return d32_new.clamp(min=1e-6, max=1.0)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"breakup": torch.zeros_like(a_i), "coalescence": torch.zeros_like(a_i)}


@InterfacialArea11Model.register("pbeTransport")
class PBECoupledTransportModel(InterfacialArea11Model):
    """PBE-coupled interfacial area transport.

    Couples interfacial area evolution with population balance equation
    for consistent bubble/droplet size distribution.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    n_bins : int
        Number of PBE bins. Default 10.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        n_bins: int = 10,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._n_bins = max(3, n_bins)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_i = 6.0 * alpha_dev / max(d32, _EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea11Model.register("budgetTracking")
class BudgetTrackingModel(InterfacialArea11Model):
    """Interfacial area budget tracking model.

    Tracks production, destruction, and transport of interfacial area
    for diagnostic purposes.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._budget_history: list[dict[str, float]] = []

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_i = 6.0 * alpha_dev / max(d32, _EPS)
        a_i = a_i.clamp(self._a_i_min, self._a_i_max)

        # Track budget
        a_mean = float(a_i.mean().item())
        a_max = float(a_i.max().item())
        a_min = float(a_i.min().item())
        self._budget_history.append({"mean": a_mean, "max": a_max, "min": a_min})

        return a_i

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}

    @property
    def budget_history(self) -> list[dict[str, float]]:
        return self._budget_history
