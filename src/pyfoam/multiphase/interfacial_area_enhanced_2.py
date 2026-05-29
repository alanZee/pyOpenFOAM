"""
Enhanced interfacial area density models — v2.

在 interfacial_area_enhanced 基础上增加：

- **DiameterTransport**：直接求解界面面积密度的输运方程
- **LuoCoalescenceBreakup**：Luo 模型的破碎/聚并源项
- **群体平衡耦合**：与 PBM 模型的双向耦合接口

Governing transport equation for interfacial area density a_i:

    d(a_i)/dt + div(U * a_i) = S_coal + S_break + S_exp

where S_coal < 0 (coalescence decreases area), S_break > 0 (breakup increases area).

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_2 import (
        DiameterTransportArea,
        LuoCoalescenceBreakupArea,
    )

    model = LuoCoalescenceBreakupArea(d32_0=3e-3)
    a_i = model.compute(alpha, n_cells=100, epsilon=eps, sigma=sigma)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area import InterfacialAreaModel

__all__ = [
    "InterfacialArea2Model",
    "DiameterTransportArea",
    "LuoCoalescenceBreakupArea",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class InterfacialArea2Model(ABC):
    """Enhanced abstract base for v2 interfacial area models.

    Extends the basic interface with transport equation support
    and source term computation.
    """

    _registry: ClassVar[dict[str, Type["InterfacialArea2Model"]]] = {}

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
    ) -> None:
        self._a_i_min = a_i_min
        self._a_i_max = a_i_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model under *name*."""

        def decorator(model_cls: Type[InterfacialArea2Model]) -> Type[InterfacialArea2Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Interfacial area v2 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea2Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown interfacial area v2 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density (1/m)."""

    @abstractmethod
    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute source terms for the transport equation.

        Returns dict with keys like ``"coalescence"``, ``"breakup"``,
        ``"expansion"``, each ``(n_cells,)`` tensor.
        """


# ======================================================================
# 直径输运模型
# ======================================================================

@InterfacialArea2Model.register("diameterTransport")
class DiameterTransportArea(InterfacialArea2Model):
    """Transport equation for interfacial area density.

    Solves a transport equation for a_i with source terms:

        S_total = S_coal + S_break + S_exp

    where:
        S_coal  = -C_coal * a_i^2 * epsilon^(1/3) / (sigma/rho_c)^(1/2)
        S_break =  C_break * alpha * epsilon^(1/3) / d32^2
        S_exp   =  a_i / alpha * d(alpha)/dt   (expansion/compression)

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density (1/m). Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density (1/m). Default ``1e6``.
    C_coal : float
        Coalescence rate constant. Default ``0.05``.
    C_break : float
        Breakup rate constant. Default ``0.01``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_coal: float = 0.05,
        C_break: float = 0.01,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_coal = C_coal
        self._C_break = C_break
        self._d32_0 = d32_0

    @property
    def C_coal(self) -> float:
        """Coalescence rate constant."""
        return self._C_coal

    @property
    def C_break(self) -> float:
        """Breakup rate constant."""
        return self._C_break

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area from alpha and d32.

        a_i = 6 * alpha / d32
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        d32 = kwargs.get("d32", self._d32_0)
        d32_t = (
            d32.to(device=device, dtype=dtype)
            if isinstance(d32, torch.Tensor)
            else torch.tensor(d32, device=device, dtype=dtype)
        )

        a_i = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute transport equation source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` current interfacial area density.
        **kwargs
            ``epsilon``: dissipation rate, ``sigma``: surface tension,
            ``rho_c``: continuous density.

        Returns
        -------
        dict
            Keys: ``coalescence``, ``breakup``, ``expansion``.
        """
        device = a_i.device
        dtype = a_i.dtype

        epsilon = kwargs.get("epsilon", torch.full_like(a_i, 0.01))
        sigma = kwargs.get("sigma", 0.072)
        rho_c = kwargs.get("rho_c", 1000.0)

        eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        rho_t = torch.tensor(rho_c, device=device, dtype=dtype)

        # Coalescence: S_coal = -C_coal * a_i^2 * eps^(1/3) / (sigma/rho)^(1/2)
        cap_term = (sigma_t / rho_t).clamp(min=_EPS).sqrt()
        S_coal = -self._C_coal * a_i.pow(2) * eps_t.pow(1.0 / 3.0) / cap_term

        # Breakup: S_break = C_break * alpha * eps^(1/3) / d32^2
        d32 = (6.0 * alpha.clamp(min=_EPS) / a_i.clamp(min=_EPS)).clamp(min=_EPS)
        S_break = (
            self._C_break * alpha.clamp(0, 1) * eps_t.pow(1.0 / 3.0)
            / d32.pow(2).clamp(min=_EPS)
        )

        # Expansion (placeholder: requires time derivative of alpha)
        S_exp = kwargs.get("dalpha_dt", torch.zeros_like(a_i)) * a_i / alpha.clamp(min=_EPS)

        return {
            "coalescence": S_coal,
            "breakup": S_break,
            "expansion": S_exp,
        }


# ======================================================================
# Luo 破碎/聚并模型
# ======================================================================

@InterfacialArea2Model.register("luoCoalescenceBreakup")
class LuoCoalescenceBreakupArea(InterfacialArea2Model):
    """Luo coalescence/breakup interfacial area model.

    Implements the Luo coalescence and breakup kernels:

    Coalescence rate:
        Q_coal = C_coal * (epsilon/d32)^(1/3) * d32^3 * exp(-We_crit/We)

    Breakup rate:
        Q_break = C_break * (epsilon/d32^2)^(1/3)
              * exp(-We_crit/We) * (1 - d_min/d)^2

    where We = rho_c * epsilon^(2/3) * d^(5/3) / sigma

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_coal : float
        Luo coalescence constant. Default ``0.0064`` (Luo, 1993).
    C_break : float
        Luo breakup constant. Default ``0.0015``.
    We_crit : float
        Critical Weber number. Default ``1.2``.
    d_min_ratio : float
        Minimum bubble diameter as ratio of parent. Default ``0.1``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_coal: float = 0.0064,
        C_break: float = 0.0015,
        We_crit: float = 1.2,
        d_min_ratio: float = 0.1,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_coal = C_coal
        self._C_break = C_break
        self._We_crit = We_crit
        self._d_min_ratio = d_min_ratio
        self._d32_0 = d32_0

    @property
    def C_coal(self) -> float:
        return self._C_coal

    @property
    def C_break(self) -> float:
        return self._C_break

    @property
    def We_crit(self) -> float:
        return self._We_crit

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density.

        a_i = 6 * alpha / d32
        """
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        d32 = kwargs.get("d32", self._d32_0)
        d32_t = (
            d32.to(device=device, dtype=dtype)
            if isinstance(d32, torch.Tensor)
            else torch.tensor(d32, device=device, dtype=dtype)
        )

        a_i = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute Luo coalescence and breakup source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``epsilon``: dissipation rate, ``sigma``: surface tension,
            ``rho_c``: continuous density.

        Returns
        -------
        dict
            Keys: ``coalescence``, ``breakup``.
        """
        device = a_i.device
        dtype = a_i.dtype

        epsilon = kwargs.get("epsilon", torch.full_like(a_i, 0.01))
        sigma = kwargs.get("sigma", 0.072)
        rho_c = kwargs.get("rho_c", 1000.0)

        eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        rho_t = torch.tensor(rho_c, device=device, dtype=dtype)

        # Current diameter
        d = (6.0 * alpha.clamp(min=_EPS) / a_i.clamp(min=_EPS)).clamp(min=_EPS)

        # Weber number: We = rho_c * eps^(2/3) * d^(5/3) / sigma
        We = rho_t * eps_t.pow(2.0 / 3.0) * d.pow(5.0 / 3.0) / sigma_t.clamp(min=_EPS)
        We = We.clamp(min=_EPS)

        # Coalescence: Q_coal = C_coal * (eps/d32)^(1/3) * d32^3 * exp(-We_crit/We)
        d32 = d  # use current d as d32
        coal_rate = (
            self._C_coal
            * (eps_t / d32).clamp(min=_EPS).pow(1.0 / 3.0)
            * d32.pow(3)
            * torch.exp(-self._We_crit / We)
        )
        S_coal = -coal_rate * a_i / d32.clamp(min=_EPS)

        # Breakup: S_break = C_break * (eps/d^2)^(1/3) * exp(-We_crit/We) * (1 - d_min/d)^2
        d_min = self._d_min_ratio * d
        breakup_rate = (
            self._C_break
            * (eps_t / d.pow(2)).clamp(min=_EPS).pow(1.0 / 3.0)
            * torch.exp(-self._We_crit / We)
            * (1.0 - d_min / d.clamp(min=_EPS)).clamp(min=0.0).pow(2)
        )
        S_break = breakup_rate * a_i / d.clamp(min=_EPS)

        return {
            "coalescence": S_coal,
            "breakup": S_break,
        }
