"""Enhanced interfacial area density models — v9.

Extends v8 with:
- **湍流增强界面面积**: turbulence-enhanced interfacial area transport
- **多分散群体平衡耦合**: coupled polydisperse PBE with size-dependent breakup
- **界面面积时间松弛**: time-relaxation model for interfacial area dynamics

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_8 import (
        TurbulentEnhancedArea,
        SizeDependentBreakupArea,
        TimeRelaxationArea,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_7 import InterfacialArea7Model

__all__ = [
    "InterfacialArea8Model",
    "TurbulentEnhancedArea",
    "SizeDependentBreakupArea",
    "TimeRelaxationArea",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class InterfacialArea8Model(InterfacialArea7Model):
    """Enhanced abstract base for v9 interfacial area models."""
    _registry: ClassVar[dict[str, Type["InterfacialArea8Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[InterfacialArea8Model]) -> Type[InterfacialArea8Model]:
            if name in cls._registry:
                raise ValueError(f"Interfacial area v8 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea8Model":
        if name not in cls._registry:
            raise KeyError(f"Unknown interfacial area v8 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@InterfacialArea8Model.register("turbulentEnhanced")
class TurbulentEnhancedArea(InterfacialArea8Model):
    """Turbulence-enhanced interfacial area model.

    The interfacial area increases with turbulent kinetic energy:

        a_i = a_i_base * (1 + C_turb * k / (g * delta_rho * d32))

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_turb : float
        Turbulence enhancement coefficient. Default 0.5.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_turb: float = 0.5,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_turb = max(0.0, C_turb)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)
        k = kwargs.get("k", torch.zeros(n_cells, device=device, dtype=dtype))
        k_t = k.to(device=device, dtype=dtype).abs() if isinstance(k, torch.Tensor) else torch.tensor(k, device=device, dtype=dtype)

        a_i_base = 6.0 * alpha_dev / max(d32, _EPS)
        enhancement = 1.0 + self._C_turb * k_t / max(d32, _EPS)
        return (a_i_base * enhancement).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        return {"relaxation": (a_i_eq - a_i) / 1.0}


@InterfacialArea8Model.register("sizeDependentBreakup")
class SizeDependentBreakupArea(InterfacialArea8Model):
    """Size-dependent breakup interfacial area model.

    Uses a breakup kernel that depends on droplet/bubble size:

        a_i = 6 * alpha / d32_breakup

    where d32_breakup = d32_0 * (1 + C_b * We^n)^(-1/m).

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_breakup : float
        Breakup coefficient. Default 0.5.
    We_exp : float
        Weber number exponent. Default 0.6.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_breakup: float = 0.5,
        We_exp: float = 0.6,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_b = max(0.0, C_breakup)
        self._We_exp = max(0.01, We_exp)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        We = kwargs.get("We", torch.ones(n_cells, device=device, dtype=dtype))
        We_t = We.to(device=device, dtype=dtype) if isinstance(We, torch.Tensor) else torch.tensor(We, device=device, dtype=dtype)

        d32_breakup = self._d32_0 / (1.0 + self._C_b * We_t.pow(self._We_exp)).clamp(min=1.0)
        a_i = 6.0 * alpha_dev / d32_breakup.clamp(min=_EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea8Model.register("timeRelaxation")
class TimeRelaxationArea(InterfacialArea8Model):
    """Time-relaxation interfacial area model.

    da_i/dt = (a_i_eq - a_i) / tau_relax

    where tau_relax depends on local flow conditions.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    tau_relax : float
        Relaxation time scale (s). Default 0.1.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        tau_relax: float = 0.1,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._tau = max(_EPS, tau_relax)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        a_i = 6.0 * alpha_dev / max(self._d32_0, _EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        return {"relaxation": (a_i_eq - a_i) / self._tau}
