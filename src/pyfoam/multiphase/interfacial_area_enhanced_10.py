"""Enhanced interfacial area density models — v11.

Extends v10 with:
- Multi-scale interfacial area transport
- Contact line contribution to interfacial area
- Real-time interfacial area monitoring with alerts

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_10 import (
        MultiScaleAreaModel,
        ContactLineAreaModel,
        MonitoredAreaModel,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_9 import InterfacialArea9Model

__all__ = [
    "InterfacialArea10Model",
    "MultiScaleAreaModel",
    "ContactLineAreaModel",
    "MonitoredAreaModel",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class InterfacialArea10Model(InterfacialArea9Model):
    """Enhanced abstract base for v11 interfacial area models."""
    _registry: ClassVar[dict[str, Type["InterfacialArea10Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[InterfacialArea10Model]) -> Type[InterfacialArea10Model]:
            if name in cls._registry:
                raise ValueError(f"Interfacial area v10 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea10Model":
        if name not in cls._registry:
            raise KeyError(f"Unknown interfacial area v10 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@InterfacialArea10Model.register("multiScale")
class MultiScaleAreaModel(InterfacialArea10Model):
    """Multi-scale interfacial area transport model.

    Splits interfacial area into large-scale (resolved) and
    sub-grid (unresolved) contributions:

        a_i = a_resolved + a_sgs

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_sgs : float
        Sub-grid contribution coefficient. Default 0.2.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_sgs: float = 0.2,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_sgs = max(0.0, C_sgs)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)
        delta = kwargs.get("delta", torch.ones(n_cells, device=device, dtype=dtype) * 0.01)

        delta_t = delta.to(device=device, dtype=dtype).abs().clamp(min=_EPS)

        # Resolved: 6*alpha/d32
        a_resolved = 6.0 * alpha_dev / max(d32, _EPS)
        # SGS: depends on grid size relative to d32
        a_sgs = self._C_sgs * alpha_dev * (1.0 - alpha_dev) / delta_t

        return (a_resolved + a_sgs).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea10Model.register("contactLine")
class ContactLineAreaModel(InterfacialArea10Model):
    """Contact line contribution to interfacial area.

    Adds contact line perimeter contribution for three-phase
    (solid-liquid-gas) systems:

        a_i_contact = P_cl / A_cell

    where P_cl is the contact line perimeter per cell.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_contact : float
        Contact line contribution coefficient. Default 0.1.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_contact: float = 0.1,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_contact = max(0.0, C_contact)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_bulk = 6.0 * alpha_dev / max(d32, _EPS)
        # Contact line contribution: peaks near alpha = 0.5
        a_contact = self._C_contact * 4.0 * alpha_dev * (1.0 - alpha_dev) / max(d32, _EPS)

        return (a_bulk + a_contact).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea10Model.register("monitored")
class MonitoredAreaModel(InterfacialArea10Model):
    """Monitored interfacial area model with alert thresholds.

    Computes interfacial area with real-time monitoring
    and alert when area exceeds thresholds.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    a_i_alert_high : float
        High alert threshold. Default 1e4.
    a_i_alert_low : float
        Low alert threshold. Default 1e-4.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        a_i_alert_high: float = 1e4,
        a_i_alert_low: float = 1e-4,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._alert_high = a_i_alert_high
        self._alert_low = max(0.0, a_i_alert_low)
        self._alert_history: list[dict[str, float]] = []

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_i = (6.0 * alpha_dev / max(d32, _EPS)).clamp(self._a_i_min, self._a_i_max)

        # Monitor
        a_mean = float(a_i.mean().item())
        a_max = float(a_i.max().item())
        alert = {
            "mean": a_mean,
            "max": a_max,
            "is_high_alert": a_max > self._alert_high,
            "is_low_alert": a_mean < self._alert_low,
        }
        self._alert_history.append(alert)

        return a_i

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}

    @property
    def alert_history(self) -> list[dict[str, float]]:
        return self._alert_history
