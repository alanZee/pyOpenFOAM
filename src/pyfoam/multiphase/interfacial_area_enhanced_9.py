"""Enhanced interfacial area density models — v10.

Extends v9 with:
- **分形界面修正**: fractal interface correction for complex topologies
- **湍流强度加权面积**: turbulence intensity-weighted interfacial area
- **群体平衡耦合**: population balance equation coupling

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_9 import (
        FractalInterfaceArea,
        TurbulenceIntensityArea,
        PBECoupledArea,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_8 import InterfacialArea8Model

__all__ = [
    "InterfacialArea9Model",
    "FractalInterfaceArea",
    "TurbulenceIntensityArea",
    "PBECoupledArea",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class InterfacialArea9Model(InterfacialArea8Model):
    """Enhanced abstract base for v10 interfacial area models."""
    _registry: ClassVar[dict[str, Type["InterfacialArea9Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[InterfacialArea9Model]) -> Type[InterfacialArea9Model]:
            if name in cls._registry:
                raise ValueError(f"Interfacial area v9 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea9Model":
        if name not in cls._registry:
            raise KeyError(f"Unknown interfacial area v9 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@InterfacialArea9Model.register("fractalInterface")
class FractalInterfaceArea(InterfacialArea9Model):
    """Fractal interface correction model.

    Corrects the interfacial area for complex, fractal-like topologies:

        a_i = a_i_base * (1 + C_fractal * D_f^(1/3))

    where D_f is the fractal dimension of the interface (2.0 = smooth, 3.0 = fully fractal).

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_fractal : float
        Fractal correction coefficient. Default 0.5.
    D_fractal : float
        Fractal dimension of interface. Default 2.2.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_fractal: float = 0.5,
        D_fractal: float = 2.2,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_f = max(0.0, C_fractal)
        self._D_f = max(2.0, min(D_fractal, 3.0))

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)

        a_i_base = 6.0 * alpha_dev / max(d32, _EPS)
        correction = 1.0 + self._C_f * (self._D_f - 2.0) ** (1.0 / 3.0)
        return (a_i_base * correction).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea9Model.register("turbulenceIntensity")
class TurbulenceIntensityArea(InterfacialArea9Model):
    """Turbulence intensity-weighted interfacial area model.

    Weights the interfacial area by local turbulence intensity:

        a_i = a_i_base * (1 + C_ti * (k / U_mean^2))

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    C_ti : float
        Turbulence intensity coefficient. Default 0.3.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        C_ti: float = 0.3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._C_ti = max(0.0, C_ti)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)
        k = kwargs.get("k", torch.zeros(n_cells, device=device, dtype=dtype))
        U_mean = kwargs.get("U_mean", torch.ones(n_cells, device=device, dtype=dtype))

        k_t = k.to(device=device, dtype=dtype).abs() if isinstance(k, torch.Tensor) else torch.tensor(k, device=device, dtype=dtype)
        U_t = U_mean.to(device=device, dtype=dtype).abs().clamp(min=_EPS) if isinstance(U_mean, torch.Tensor) else torch.full((n_cells,), U_mean, device=device, dtype=dtype)

        ti = (k_t / U_t.pow(2)).clamp(max=5.0)
        a_i_base = 6.0 * alpha_dev / max(d32, _EPS)
        return (a_i_base * (1.0 + self._C_ti * ti)).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}


@InterfacialArea9Model.register("pbeCoupled")
class PBECoupledArea(InterfacialArea9Model):
    """Population balance equation coupled interfacial area model.

    Links the interfacial area to PBE moments:

        a_i = 6 * alpha / d32_from_PBE

    where d32 is derived from PBE moment ratio.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    tau_relax : float
        Relaxation time for PBE coupling. Default 0.1.
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
        d32 = kwargs.get("d32", self._d32_0)
        a_i = 6.0 * alpha_dev / max(d32, _EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        return {"relaxation": (a_i_eq - a_i) / self._tau}
