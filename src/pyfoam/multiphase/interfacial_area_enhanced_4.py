"""
Enhanced interfacial area density models — v4.

在 Enhanced v3 基础上增加：

- **WaveBreakupArea**：基于界面不稳定波的破碎模型
- **StretchRateArea**：基于应变率的界面面积生成模型
- **统一框架模型**：统一破碎/聚并/拉伸的综合框架

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_4 import (
        WaveBreakupArea,
        StretchRateArea,
        UnifiedBreakupCoalescenceArea,
    )

    model = WaveBreakupArea(d32_0=3e-3)
    a_i = model.compute(alpha, n_cells=100)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_3 import InterfacialArea3Model

__all__ = [
    "InterfacialArea4Model",
    "WaveBreakupArea",
    "StretchRateArea",
    "UnifiedBreakupCoalescenceArea",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class InterfacialArea4Model(InterfacialArea3Model):
    """Enhanced abstract base for v4 interfacial area models.

    Extends v3 with wave breakup, stretch rate, and unified framework models.
    """

    _registry: ClassVar[dict[str, Type["InterfacialArea4Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model under *name*."""

        def decorator(model_cls: Type[InterfacialArea4Model]) -> Type[InterfacialArea4Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Interfacial area v4 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea4Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown interfacial area v4 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 界面波破碎模型
# ======================================================================

@InterfacialArea4Model.register("waveBreakup")
class WaveBreakupArea(InterfacialArea4Model):
    """Wave-driven breakup interfacial area model.

    基于 Rayleigh-Taylor 和 Kelvin-Helmholtz 不稳定性的界面破碎：

        lambda_RT = 2*pi * sqrt(sigma / (g * delta_rho))
        f_break = C_w * U_rel / lambda_RT
        d_break = C_d * lambda_RT

    界面面积生成：

        S_wave = f_break * (6 * alpha / d_break - a_i) * alpha

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_w : float
        Wave breakup rate constant. Default ``0.1``.
    C_d : float
        Daughter droplet diameter ratio. Default ``0.3``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_w: float = 0.1,
        C_d: float = 0.3,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_w = C_w
        self._C_d = C_d
        self._d32_0 = d32_0

    @property
    def C_w(self) -> float:
        return self._C_w

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density: a_i = 6 * alpha / d32."""
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
        """Compute wave breakup source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``sigma``, ``delta_rho``, ``g``, ``U_rel``.

        Returns
        -------
        dict
            Keys: ``wave_breakup``.
        """
        device = a_i.device
        dtype = a_i.dtype

        sigma = kwargs.get("sigma", 0.072)
        delta_rho = kwargs.get("delta_rho", 997.0)
        g = kwargs.get("g", 9.81)
        U_rel = kwargs.get("U_rel", torch.full_like(a_i, 0.1))

        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        drho_t = torch.tensor(delta_rho, device=device, dtype=dtype)
        g_t = torch.tensor(g, device=device, dtype=dtype)
        U_t = U_rel.to(device=device, dtype=dtype).abs()

        # RT wavelength
        lambda_RT = 2.0 * 3.14159265 * torch.sqrt(
            sigma_t / (g_t * drho_t).clamp(min=_EPS)
        )

        # Breakup frequency
        f_break = self._C_w * U_t / lambda_RT.clamp(min=_EPS)

        # Daughter droplet diameter
        d_break = self._C_d * lambda_RT
        a_i_break = 6.0 * alpha.clamp(min=_EPS) / d_break.clamp(min=_EPS)

        # Source: wave breakup increases area
        S_wave = f_break * (a_i_break - a_i).clamp(min=0.0) * alpha.clamp(0, 1)

        return {"wave_breakup": S_wave}


# ======================================================================
# 应变率模型
# ======================================================================

@InterfacialArea4Model.register("stretchRate")
class StretchRateArea(InterfacialArea4Model):
    """Stretch-rate-driven interfacial area generation model.

    界面面积受应变率拉伸：

        S_stretch = C_s * a_i * |S_ij|

    where S_ij = 0.5 * (dU_i/dx_j + dU_j/dx_i) is the strain rate tensor,
    and |S_ij| = sqrt(2 * S_ij * S_ij).

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_s : float
        Stretch rate coefficient. Default ``0.5``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_s: float = 0.5,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_s = C_s
        self._d32_0 = d32_0

    @property
    def C_s(self) -> float:
        return self._C_s

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density: a_i = 6 * alpha / d32."""
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
        """Compute stretch rate source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``strain_rate``: ``(n_cells,)`` magnitude of strain rate tensor.

        Returns
        -------
        dict
            Keys: ``stretch``, ``relaxation``.
        """
        device = a_i.device
        dtype = a_i.dtype

        strain_rate = kwargs.get("strain_rate", torch.full_like(a_i, 1.0))
        S_mag = strain_rate.to(device=device, dtype=dtype).abs()

        # Stretch generation: a_i grows with strain rate
        S_stretch = self._C_s * a_i * S_mag * alpha.clamp(0, 1)

        # Relaxation: interface area tends to equilibrium
        # a_i_eq = 6 * alpha / d32_eq
        d32_eq = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32_eq
        tau_relax = 1.0  # Relaxation time scale (s)
        S_relax = (a_i_eq - a_i).clamp(min=0.0) / tau_relax

        return {
            "stretch": S_stretch,
            "relaxation": S_relax,
        }


# ======================================================================
# 统一破碎/聚并/拉伸框架
# ======================================================================

@InterfacialArea4Model.register("unified")
class UnifiedBreakupCoalescenceArea(InterfacialArea4Model):
    """Unified breakup/coalescence/stretch framework model.

    统一框架整合三种界面面积变化机制：

        da_i/dt = S_breakup + S_coalescence + S_stretch

    - 破碎：基于 Weber 数的临界破碎
    - 聚并：基于碰撞频率的液滴聚并
    - 拉伸：基于应变率的界面拉伸

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_break : float
        Breakup coefficient. Default ``0.01``.
    C_coal : float
        Coalescence coefficient. Default ``0.05``.
    C_stretch : float
        Stretch coefficient. Default ``0.1``.
    We_cr : float
        Critical Weber number. Default ``1.2``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_break: float = 0.01,
        C_coal: float = 0.05,
        C_stretch: float = 0.1,
        We_cr: float = 1.2,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_break = C_break
        self._C_coal = C_coal
        self._C_stretch = C_stretch
        self._We_cr = We_cr
        self._d32_0 = d32_0

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area density: a_i = 6 * alpha / d32."""
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
        """Compute unified source terms (breakup + coalescence + stretch).

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``epsilon``, ``sigma``, ``rho_c``, ``strain_rate``.

        Returns
        -------
        dict
            Keys: ``breakup``, ``coalescence``, ``stretch``, ``net``.
        """
        device = a_i.device
        dtype = a_i.dtype

        epsilon = kwargs.get("epsilon", torch.full_like(a_i, 0.01))
        sigma = kwargs.get("sigma", 0.072)
        rho_c = kwargs.get("rho_c", 1000.0)
        strain_rate = kwargs.get("strain_rate", torch.full_like(a_i, 1.0))

        eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        rho_t = torch.tensor(rho_c, device=device, dtype=dtype)
        S_mag = strain_rate.to(device=device, dtype=dtype).abs()

        # Current diameter
        d = (6.0 * alpha.clamp(min=_EPS) / a_i.clamp(min=_EPS)).clamp(min=_EPS)

        # Breakup: Weber number criterion
        We = rho_t * eps_t.pow(2.0 / 3.0) * d.pow(5.0 / 3.0) / sigma_t.clamp(min=_EPS)
        f_break = self._C_break * eps_t.pow(1.0 / 3.0) * torch.exp(-self._We_cr / We.clamp(min=_EPS))
        d_daughter = 0.6 * d
        a_i_daughter = 6.0 * alpha.clamp(min=_EPS) / d_daughter.clamp(min=_EPS)
        S_breakup = f_break * (a_i_daughter - a_i).clamp(min=0.0) * alpha.clamp(0, 1)

        # Coalescence: proportional to a_i^2
        cap_term = (sigma_t / rho_t).clamp(min=_EPS).sqrt()
        S_coal = -self._C_coal * a_i.pow(2) * eps_t.pow(1.0 / 3.0) / cap_term

        # Stretch: proportional to strain rate
        S_stretch = self._C_stretch * a_i * S_mag * alpha.clamp(0, 1)

        # Net source
        S_net = S_breakup + S_coal + S_stretch

        return {
            "breakup": S_breakup,
            "coalescence": S_coal,
            "stretch": S_stretch,
            "net": S_net,
        }
