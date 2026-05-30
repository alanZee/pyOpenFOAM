"""
Enhanced interfacial area density models — v7.

在 Enhanced v6 基础上增加：

- **随机界面波动模型**：考虑湍流引起的界面面积随机波动
- **Weber 数修正**：根据局部 Weber 数修正界面面积产生和耗散
- **核化界面生成**：壁面核化沸腾产生的界面面积源项

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_6 import (
        StochasticInterfacialArea,
        WeberCorrectedArea,
        NucleationAreaGeneration,
    )

    model = StochasticInterfacialArea(d32_0=3e-3)
    a_i = model.compute(alpha, n_cells=100)
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_5 import InterfacialArea5Model

__all__ = [
    "InterfacialArea6Model",
    "StochasticInterfacialArea",
    "WeberCorrectedArea",
    "NucleationAreaGeneration",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class InterfacialArea6Model(InterfacialArea5Model):
    """Enhanced abstract base for v7 interfacial area models.

    Extends v6 with stochastic, Weber-corrected, and nucleation models.
    """

    _registry: ClassVar[dict[str, Type["InterfacialArea6Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model under *name*."""

        def decorator(model_cls: Type[InterfacialArea6Model]) -> Type[InterfacialArea6Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Interfacial area v6 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea6Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown interfacial area v6 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 随机界面波动
# ======================================================================

@InterfacialArea6Model.register("stochastic")
class StochasticInterfacialArea(InterfacialArea6Model):
    """Stochastic interfacial area fluctuation model.

    考虑湍流引起的界面面积随机波动：

        a_i = a_i_mean * (1 + epsilon_fluct)

    其中 epsilon_fluct ~ N(0, sigma_fluct^2) 由湍流强度决定。

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    sigma_fluct : float
        Fluctuation intensity (standard deviation). Default ``0.1``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        sigma_fluct: float = 0.1,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._sigma_fluct = max(0.0, min(sigma_fluct, 1.0))
        self._d32_0 = d32_0

    @property
    def sigma_fluct(self) -> float:
        """Fluctuation intensity."""
        return self._sigma_fluct

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute stochastically fluctuating interfacial area density.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        n_cells : int
            Number of cells.
        **kwargs
            ``d32``: Sauter mean diameter override.
            ``turb_intensity``: ``(n_cells,)`` local turbulence intensity.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` interfacial area density with fluctuations.
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

        # Base area
        a_i_base = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)

        # Fluctuation amplitude scaled by turbulence intensity
        turb = kwargs.get("turb_intensity", None)
        if turb is not None:
            turb_t = turb.to(device=device, dtype=dtype)
            sigma = self._sigma_fluct * turb_t.clamp(min=0.0, max=3.0)
        else:
            sigma = self._sigma_fluct

        # Stochastic fluctuation
        epsilon = sigma * torch.randn(n_cells, device=device, dtype=dtype)
        a_i = a_i_base * (1.0 + epsilon).clamp(min=0.5, max=2.0)

        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute stochastic source terms: fluctuation dissipation."""
        device = a_i.device
        dtype = a_i.dtype

        # Relaxation toward equilibrium
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        tau_relax = 2.0
        S_relax = (a_i_eq - a_i) / tau_relax

        # Fluctuation source
        S_fluct = self._sigma_fluct * a_i * torch.randn_like(a_i) * 0.1

        return {
            "relaxation": S_relax,
            "fluctuation": S_fluct,
            "net": S_relax + S_fluct,
        }


# ======================================================================
# Weber 数修正
# ======================================================================

@InterfacialArea6Model.register("weberCorrected")
class WeberCorrectedArea(InterfacialArea6Model):
    """Weber-number-corrected interfacial area model.

    根据局部 Weber 数修正界面面积：

        We = rho * U_rel^2 * d32 / sigma_st

    - We < We_crit: 稳定界面，面积由 alpha/d32 决定
    - We > We_crit: 破碎增强，面积增大

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    We_crit : float
        Critical Weber number for breakup. Default ``12.0``.
    C_weber : float
        Weber correction coefficient. Default ``0.5``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    sigma_st : float
        Surface tension coefficient (N/m). Default ``0.072``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        We_crit: float = 12.0,
        C_weber: float = 0.5,
        d32_0: float = 3e-3,
        sigma_st: float = 0.072,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._We_crit = max(_EPS, We_crit)
        self._C_weber = max(0.0, C_weber)
        self._d32_0 = d32_0
        self._sigma_st = max(_EPS, sigma_st)

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Weber-corrected interfacial area density."""
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        d32 = kwargs.get("d32", self._d32_0)
        d32_t = (
            d32.to(device=device, dtype=dtype)
            if isinstance(d32, torch.Tensor)
            else torch.tensor(d32, device=device, dtype=dtype)
        )

        a_i_base = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)

        # Weber number
        rho = kwargs.get("rho", 1000.0)
        U_rel = kwargs.get("U_rel", torch.zeros(n_cells, device=device, dtype=dtype))
        U_rel_t = U_rel.to(device=device, dtype=dtype) if isinstance(U_rel, torch.Tensor) \
            else torch.full((n_cells,), U_rel, device=device, dtype=dtype)

        We = rho * U_rel_t.pow(2) * d32_t / self._sigma_st

        # Correction: enhancement when We > We_crit
        excess = (We - self._We_crit).clamp(min=0.0)
        correction = 1.0 + self._C_weber * (excess / self._We_crit).clamp(max=5.0)

        a_i = a_i_base * correction
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute Weber-dependent source terms."""
        device = a_i.device
        dtype = a_i.dtype

        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        tau_relax = 1.5
        S_relax = (a_i_eq - a_i) / tau_relax

        return {"relaxation": S_relax}


# ======================================================================
# 核化界面生成
# ======================================================================

@InterfacialArea6Model.register("nucleation")
class NucleationAreaGeneration(InterfacialArea6Model):
    """Nucleation-induced interfacial area generation model.

    壁面核化沸腾产生的界面面积源项：

        S_nuc = f_nuc * A_w * (pi * d_b^2 / 6) * n_b

    其中 n_b 是核化位点密度，d_b 是气泡脱离直径。

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_nuc : float
        Nucleation rate coefficient. Default ``0.01``.
    d_bubble : float
        Departure bubble diameter (m). Default ``1e-3``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_nuc: float = 0.01,
        d_bubble: float = 1e-3,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_nuc = max(0.0, C_nuc)
        self._d_bubble = max(_EPS, d_bubble)
        self._d32_0 = d32_0

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute interfacial area including nucleation contribution."""
        device = get_device()
        dtype = get_default_dtype()
        alpha_dev = alpha.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        d32 = kwargs.get("d32", self._d32_0)
        d32_t = (
            d32.to(device=device, dtype=dtype)
            if isinstance(d32, torch.Tensor)
            else torch.tensor(d32, device=device, dtype=dtype)
        )

        # Base area
        a_i_base = 6.0 * alpha_dev / d32_t.clamp(min=_EPS)

        # Nucleation contribution: increases area in near-wall regions
        # Proxy: high alpha gradient indicates near-wall/boiling
        wall_factor = kwargs.get("wall_factor", torch.ones(n_cells, device=device, dtype=dtype))
        wall_t = wall_factor.to(device=device, dtype=dtype)

        # Single bubble area
        A_bubble = math.pi * self._d_bubble ** 2
        a_nuc = self._C_nuc * A_bubble * wall_t

        a_i = a_i_base + a_nuc
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute nucleation source terms."""
        device = a_i.device
        dtype = a_i.dtype

        # Nucleation source
        T_wall = kwargs.get("T_wall", torch.full_like(a_i, 373.15))
        T_sat = kwargs.get("T_sat", 373.15)
        dT = (T_wall.to(device=device, dtype=dtype) - T_sat).clamp(min=0.0)

        S_nuc = self._C_nuc * dT / max(self._d_bubble, _EPS)

        # Condensation sink
        d32 = self._d32_0
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32
        tau_cond = 0.5
        S_cond = -(a_i - a_i_eq).clamp(min=0.0) / tau_cond

        return {
            "nucleation": S_nuc,
            "condensation": S_cond,
            "net": S_nuc + S_cond,
        }
