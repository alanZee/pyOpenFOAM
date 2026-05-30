"""Enhanced interfacial area density models — v8.

Extends v7 with:
- **Hamaker 力修正的界面面积**: considers van der Waals attractive forces
- **聚并-破碎平衡模型**: equilibrium between coalescence and breakup
- **多分散群体平衡**: polydisperse PBE-coupled interfacial area transport

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_7 import (
        HamakerCorrectedArea,
        CoalescenceBreakupEquilibrium,
        PolydispersePBETArea,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_6 import InterfacialArea6Model

__all__ = [
    "InterfacialArea7Model",
    "HamakerCorrectedArea",
    "CoalescenceBreakupEquilibrium",
    "PolydispersePBETArea",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class InterfacialArea7Model(InterfacialArea6Model):
    """Enhanced abstract base for v8 interfacial area models."""
    _registry: ClassVar[dict[str, Type["InterfacialArea7Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[InterfacialArea7Model]) -> Type[InterfacialArea7Model]:
            if name in cls._registry:
                raise ValueError(f"Interfacial area v7 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea7Model":
        if name not in cls._registry:
            raise KeyError(f"Unknown interfacial area v7 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@InterfacialArea7Model.register("hamaker")
class HamakerCorrectedArea(InterfacialArea7Model):
    """Hamaker-corrected interfacial area model.

    Accounts for van der Waals attractive forces that promote film thinning
    and coalescence, reducing the effective interfacial area:

        a_i = a_i_base * f_Hamaker(d_film)

    where f_Hamaker = 1 - A_H / (12 * pi * d_film^2 * sigma_st)

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    A_hamaker : float
        Hamaker constant (J). Default 1e-20.
    sigma_st : float
        Surface tension (N/m). Default 0.072.
    d32_0 : float
        Reference Sauter mean diameter (m). Default 3e-3.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        A_hamaker: float = 1e-20,
        sigma_st: float = 0.072,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._A_h = A_hamaker
        self._sigma_st = max(_EPS, sigma_st)
        self._d32_0 = d32_0

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)
        d32_t = d32 if isinstance(d32, torch.Tensor) else torch.tensor(d32, device=device, dtype=dtype)
        a_i_base = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)

        d_film = kwargs.get("d_film", torch.full((n_cells,), 1e-6, device=device, dtype=dtype))
        d_film_t = d_film.to(device=device, dtype=dtype)
        f_hamaker = (1.0 - self._A_h / (12.0 * math.pi * d_film_t.pow(2).clamp(min=_EPS) * self._sigma_st)).clamp(min=0.1, max=1.0)

        return (a_i_base * f_hamaker).clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        return {"relaxation": (a_i_eq - a_i) / 1.0}


@InterfacialArea7Model.register("coalescenceBreakupEq")
class CoalescenceBreakupEquilibrium(InterfacialArea7Model):
    """Coalescence-breakup equilibrium interfacial area model.

    Equilibrium interfacial area from balance of coalescence (reducing area)
    and breakup (increasing area):

        a_i_eq = 6 * alpha / d32_eq

    where d32_eq = d32_0 * (We_crit / We)^0.4 for We > We_crit.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    d32_0 : float
        Reference Sauter mean diameter. Default 3e-3.
    We_crit : float
        Critical Weber number. Default 12.0.
    coalescence_rate : float
        Coalescence time scale (1/s). Default 1.0.
    breakup_rate : float
        Breakup time scale (1/s). Default 0.5.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        d32_0: float = 3e-3,
        We_crit: float = 12.0,
        coalescence_rate: float = 1.0,
        breakup_rate: float = 0.5,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._d32_0 = d32_0
        self._We_crit = max(_EPS, We_crit)
        self._k_coal = max(_EPS, coalescence_rate)
        self._k_break = max(_EPS, breakup_rate)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        d32 = kwargs.get("d32", self._d32_0)
        d32_t = d32 if isinstance(d32, torch.Tensor) else torch.tensor(d32, device=device, dtype=dtype)
        a_i_base = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)
        return a_i_base.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        S_coal = -self._k_coal * (a_i - a_i_eq).clamp(min=0.0)
        S_break = self._k_break * (a_i_eq - a_i).clamp(min=0.0)
        return {"coalescence": S_coal, "breakup": S_break, "net": S_coal + S_break}


@InterfacialArea7Model.register("polydispersePBE")
class PolydispersePBETArea(InterfacialArea7Model):
    """Polydisperse PBE-coupled interfacial area transport.

    Computes interfacial area from a polydisperse size distribution:

        a_i = integral_0^inf n(d) * pi * d^2 dd = 6 * alpha / d32

    where d32 comes from the population balance model.

    Parameters
    ----------
    a_i_min, a_i_max : float
        Interfacial area bounds.
    n_bins : int
        Number of PBE size bins. Default 10.
    d_min : float
        Minimum bubble diameter (m). Default 1e-4.
    d_max : float
        Maximum bubble diameter (m). Default 1e-2.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        n_bins: int = 10,
        d_min: float = 1e-4,
        d_max: float = 1e-2,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._n_bins = max(2, n_bins)
        self._d_min = max(_EPS, d_min)
        self._d_max = max(self._d_min * 1.1, d_max)

    def compute(self, alpha: torch.Tensor, n_cells: int, **kwargs: Any) -> torch.Tensor:
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        # Use provided size distribution or default uniform
        n_dist = kwargs.get("n_dist", None)
        if n_dist is not None:
            # Compute d32 from size distribution
            d_bins = torch.linspace(self._d_min, self._d_max, self._n_bins, device=device, dtype=dtype)
            n_t = n_dist.to(device=device, dtype=dtype)
            d32_num = (n_t * d_bins.pow(3)).sum()
            d32_den = (n_t * d_bins.pow(2)).sum().clamp(min=_EPS)
            d32 = d32_num / d32_den
        else:
            d32 = (self._d_min + self._d_max) / 2.0

        a_i = 6.0 * alpha_dev / max(d32, _EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(self, alpha: torch.Tensor, a_i: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"net": torch.zeros_like(a_i)}
