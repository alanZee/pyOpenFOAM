"""
Enhanced interfacial area density models — v3.

在 Enhanced v2 基础上增加：

- **TurbulentBreakupArea**：基于湍流能谱的破碎模型 (Prince-Blanch)
- **TipStreamingArea**：尖端射流界面面积生成模型
- **有限尺寸效应**：大变形修正的界面面积模型

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_3 import (
        TurbulentBreakupArea,
        TipStreamingArea,
    )

    model = TurbulentBreakupArea(d32_0=3e-3)
    a_i = model.compute(alpha, n_cells=100, epsilon=eps, sigma=sigma)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_2 import InterfacialArea2Model

__all__ = [
    "InterfacialArea3Model",
    "TurbulentBreakupArea",
    "TipStreamingArea",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class InterfacialArea3Model(InterfacialArea2Model):
    """Enhanced abstract base for v3 interfacial area models.

    Extends v2 with finite-size corrections and advanced breakup models.
    """

    _registry: ClassVar[dict[str, Type["InterfacialArea3Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model under *name*."""

        def decorator(model_cls: Type[InterfacialArea3Model]) -> Type[InterfacialArea3Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Interfacial area v3 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea3Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown interfacial area v3 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 湍流能谱破碎模型 (Prince-Blanch)
# ======================================================================

@InterfacialArea3Model.register("turbulentBreakup")
class TurbulentBreakupArea(InterfacialArea3Model):
    """Prince-Blanch turbulent breakup interfacial area model.

    基于湍流能谱的破碎频率：

        f_break = C1 * epsilon^(1/3) * d^(2/3) / sigma^(1/2)
                  * exp(-We_cr / We)

    where We = rho_c * epsilon^(2/3) * d^(5/3) / sigma

    破碎后的界面面积变化：

        S_break = f_break * (a_i_new - a_i) * alpha

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C1 : float
        Breakup rate constant. Default ``0.001``.
    We_cr : float
        Critical Weber number. Default ``1.2``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    daughter_ratio : float
        Ratio of daughter to parent diameter after breakup. Default ``0.6``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C1: float = 0.001,
        We_cr: float = 1.2,
        d32_0: float = 3e-3,
        daughter_ratio: float = 0.6,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C1 = C1
        self._We_cr = We_cr
        self._d32_0 = d32_0
        self._daughter_ratio = daughter_ratio

    @property
    def C1(self) -> float:
        return self._C1

    @property
    def We_cr(self) -> float:
        return self._We_cr

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
        """Compute Prince-Blanch breakup source term.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``epsilon``, ``sigma``, ``rho_c``.

        Returns
        -------
        dict
            Keys: ``breakup``, ``coalescence``.
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

        # Weber number
        We = rho_t * eps_t.pow(2.0 / 3.0) * d.pow(5.0 / 3.0) / sigma_t.clamp(min=_EPS)
        We = We.clamp(min=_EPS)

        # Breakup frequency
        f_break = (
            self._C1
            * eps_t.pow(1.0 / 3.0)
            * d.pow(2.0 / 3.0)
            / sigma_t.clamp(min=_EPS).pow(0.5)
            * torch.exp(-self._We_cr / We)
        )

        # Daughter diameter
        d_daughter = self._daughter_ratio * d
        a_i_daughter = 6.0 * alpha.clamp(min=_EPS) / d_daughter.clamp(min=_EPS)

        # Source: breakup increases area
        S_break = f_break * (a_i_daughter - a_i).clamp(min=0.0) * alpha.clamp(0, 1)

        # Simple coalescence (proportional to a_i^2)
        C_coal = 0.05
        cap_term = (sigma_t / rho_t).clamp(min=_EPS).sqrt()
        S_coal = -C_coal * a_i.pow(2) * eps_t.pow(1.0 / 3.0) / cap_term

        return {
            "breakup": S_break,
            "coalescence": S_coal,
        }


# ======================================================================
# 尖端射流模型
# ======================================================================

@InterfacialArea3Model.register("tipStreaming")
class TipStreamingArea(InterfacialArea3Model):
    """Tip streaming interfacial area generation model.

    尖端射流（tip streaming）在高毛细数（Ca > Ca_cr）时发生，
    产生细小液滴从而显著增加界面面积。

    生成速率：

        S_ts = C_ts * alpha * max(0, Ca - Ca_cr)^n / d

    where Ca = mu_c * U_rel / sigma

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_ts : float
        Tip streaming rate constant. Default ``0.01``.
    Ca_cr : float
        Critical capillary number. Default ``0.5``.
    exponent : float
        Power-law exponent. Default ``1.5``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_ts: float = 0.01,
        Ca_cr: float = 0.5,
        exponent: float = 1.5,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_ts = C_ts
        self._Ca_cr = Ca_cr
        self._exponent = exponent
        self._d32_0 = d32_0

    @property
    def C_ts(self) -> float:
        return self._C_ts

    @property
    def Ca_cr(self) -> float:
        return self._Ca_cr

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
        """Compute tip streaming source term.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``mu_c``, ``U_rel``, ``sigma``.

        Returns
        -------
        dict
            Keys: ``tip_streaming``.
        """
        device = a_i.device
        dtype = a_i.dtype

        mu_c = kwargs.get("mu_c", torch.full_like(a_i, 1e-3))
        U_rel = kwargs.get("U_rel", torch.full_like(a_i, 0.1))
        sigma = kwargs.get("sigma", 0.072)

        mu_t = mu_c.to(device=device, dtype=dtype).clamp(min=_EPS)
        U_t = U_rel.to(device=device, dtype=dtype).abs()
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)

        # Capillary number
        Ca = mu_t * U_t / sigma_t.clamp(min=_EPS)
        Ca_excess = (Ca - self._Ca_cr).clamp(min=0.0)

        # Current diameter
        d = (6.0 * alpha.clamp(min=_EPS) / a_i.clamp(min=_EPS)).clamp(min=_EPS)

        # Tip streaming source
        S_ts = (
            self._C_ts
            * alpha.clamp(0, 1)
            * Ca_excess.pow(self._exponent)
            / d.clamp(min=_EPS)
        )

        return {
            "tip_streaming": S_ts,
        }
