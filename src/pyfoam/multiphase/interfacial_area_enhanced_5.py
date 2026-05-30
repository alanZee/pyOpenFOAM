"""
Enhanced interfacial area density models — v6.

在 Enhanced v5 基础上增加：

- **拓扑界面追踪**：基于拓扑连接性的界面面积变化率
- **分形维数修正**：考虑界面粗糙度的分形修正
- **相感知面积输运**：不同相的独立面积输运方程

Usage::

    from pyfoam.multiphase.interfacial_area_enhanced_5 import (
        TopologicalInterfaceArea,
        FractalDimensionArea,
        PhaseAwareAreaTransport,
    )

    model = TopologicalInterfaceArea(d32_0=3e-3)
    a_i = model.compute(alpha, n_cells=100)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.interfacial_area_enhanced_4 import InterfacialArea4Model

__all__ = [
    "InterfacialArea5Model",
    "TopologicalInterfaceArea",
    "FractalDimensionArea",
    "PhaseAwareAreaTransport",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class InterfacialArea5Model(InterfacialArea4Model):
    """Enhanced abstract base for v6 interfacial area models.

    Extends v5 with topological, fractal, and phase-aware transport models.
    """

    _registry: ClassVar[dict[str, Type["InterfacialArea5Model"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a model under *name*."""

        def decorator(model_cls: Type[InterfacialArea5Model]) -> Type[InterfacialArea5Model]:
            if name in cls._registry:
                raise ValueError(
                    f"Interfacial area v5 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "InterfacialArea5Model":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown interfacial area v5 model '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 拓扑界面追踪
# ======================================================================

@InterfacialArea5Model.register("topological")
class TopologicalInterfaceArea(InterfacialArea5Model):
    """Topology-driven interfacial area model.

    基于界面拓扑连接性的面积变化率：

        dA/dt = S_connect + S_disconnect

    - connect: 新的界面连接生成（合并中的液滴形成桥接）
    - disconnect: 界面连接断裂（液滴分裂）

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_connect : float
        Connection rate coefficient. Default ``0.1``.
    C_disconnect : float
        Disconnection rate coefficient. Default ``0.05``.
    connectivity_threshold : float
        Threshold for connectivity detection. Default ``0.3``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_connect: float = 0.1,
        C_disconnect: float = 0.05,
        connectivity_threshold: float = 0.3,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_connect = C_connect
        self._C_disconnect = C_disconnect
        self._connectivity_threshold = connectivity_threshold
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
        """Compute topological source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``grad_alpha_mag``: ``(n_cells,)`` gradient magnitude.

        Returns
        -------
        dict
            Keys: ``connect``, ``disconnect``, ``net``.
        """
        device = a_i.device
        dtype = a_i.dtype

        grad_alpha = kwargs.get("grad_alpha_mag", torch.full_like(a_i, 0.1))
        grad_t = grad_alpha.to(device=device, dtype=dtype).abs()

        # Connectivity indicator: high gradient means interface present
        connectivity = torch.tanh(grad_t / max(self._connectivity_threshold, _EPS))

        # Connection: area generation where alpha is moderate (merging)
        interface_flag = 4.0 * alpha.clamp(0, 1) * (1.0 - alpha.clamp(0, 1))
        S_connect = self._C_connect * connectivity * interface_flag * a_i

        # Disconnection: area reduction where alpha approaches 0 or 1
        isolation = 1.0 - interface_flag
        S_disconnect = -self._C_disconnect * isolation * a_i

        return {
            "connect": S_connect,
            "disconnect": S_disconnect,
            "net": S_connect + S_disconnect,
        }


# ======================================================================
# 分形维数修正
# ======================================================================

@InterfacialArea5Model.register("fractal")
class FractalDimensionArea(InterfacialArea5Model):
    """Fractal-dimension-corrected interfacial area model.

    考虑界面粗糙度的分形修正：

        a_i_eff = a_i * (d / d0)^(D_f - 2)

    其中 D_f 是分形维数 (2 <= D_f <= 3)，d0 是参考直径。
    光滑界面 D_f = 2，完全粗糙 D_f -> 3。

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    D_f : float
        Fractal dimension. Default ``2.2``.
    d0 : float
        Reference length scale (m). Default ``1e-3``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        D_f: float = 2.2,
        d0: float = 1e-3,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._D_f = max(2.0, min(D_f, 3.0))
        self._d0 = max(_EPS, d0)
        self._d32_0 = d32_0

    @property
    def D_f(self) -> float:
        """Fractal dimension."""
        return self._D_f

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute fractal-corrected interfacial area density.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        n_cells : int
            Number of cells.
        **kwargs
            ``d32``: Sauter mean diameter override.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` effective interfacial area density.
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

        # Fractal correction
        d0_t = torch.tensor(self._d0, device=device, dtype=dtype)
        fractal_ratio = (d32_t / d0_t).clamp(min=_EPS)
        fractal_exponent = self._D_f - 2.0
        correction = fractal_ratio.pow(fractal_exponent)

        a_i = a_i_base * correction
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute fractal source terms.

        Returns ``relaxation`` term that drives a_i toward fractal equilibrium.
        """
        device = a_i.device
        dtype = a_i.dtype

        # Equilibrium from current d32
        d32 = kwargs.get("d32", self._d32_0)
        d32_t = (
            d32.to(device=device, dtype=dtype)
            if isinstance(d32, torch.Tensor)
            else torch.tensor(d32, device=device, dtype=dtype)
        )
        d0_t = torch.tensor(self._d0, device=device, dtype=dtype)
        fractal_ratio = (d32_t / d0_t).clamp(min=_EPS)
        correction = fractal_ratio.pow(self._D_f - 2.0)
        a_i_eq = 6.0 * alpha.clamp(min=_EPS) / d32_t.clamp(min=_EPS) * correction

        tau_relax = 1.0
        S_relax = (a_i_eq - a_i) / tau_relax

        return {"relaxation": S_relax}


# ======================================================================
# 相感知面积输运
# ======================================================================

@InterfacialArea5Model.register("phaseAware")
class PhaseAwareAreaTransport(InterfacialArea5Model):
    """Phase-aware interfacial area transport model.

    对每个相分别追踪界面面积，支持多相系统中不同界面的独立演化：

        d(a_ij)/dt + div(U * a_ij) = S_ij

    其中 i, j 是相邻相编号。

    Parameters
    ----------
    a_i_min : float
        Minimum interfacial area density. Default ``1e-6``.
    a_i_max : float
        Maximum interfacial area density. Default ``1e6``.
    C_transport : float
        Area transport coefficient. Default ``0.2``.
    d32_0 : float
        Reference Sauter mean diameter (m). Default ``3e-3``.
    """

    def __init__(
        self,
        a_i_min: float = 1e-6,
        a_i_max: float = 1e6,
        C_transport: float = 0.2,
        d32_0: float = 3e-3,
    ) -> None:
        super().__init__(a_i_min, a_i_max)
        self._C_transport = C_transport
        self._d32_0 = d32_0

    def compute(
        self,
        alpha: torch.Tensor,
        n_cells: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute phase-aware interfacial area density.

        For a binary system, a_i = 6 * alpha * (1-alpha) / d32.
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

        # Phase-aware: area peaks at alpha = 0.5
        a_i = 6.0 * alpha_dev * (1.0 - alpha_dev) / d32_t.clamp(min=_EPS)
        return a_i.clamp(self._a_i_min, self._a_i_max)

    def source_terms(
        self,
        alpha: torch.Tensor,
        a_i: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute transport source terms.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` volume fraction.
        a_i : torch.Tensor
            ``(n_cells,)`` interfacial area density.
        **kwargs
            ``alpha_rate``: ``(n_cells,)`` time derivative of alpha.

        Returns
        -------
        dict
            Keys: ``transport``, ``equilibrium``.
        """
        device = a_i.device
        dtype = a_i.dtype

        alpha_rate = kwargs.get("alpha_rate", torch.zeros_like(a_i))
        dalpha_dt = alpha_rate.to(device=device, dtype=dtype)

        d32 = self._d32_0
        # Equilibrium: a_i_eq = 6 * alpha * (1-alpha) / d32
        a_i_eq = 6.0 * alpha.clamp(0, 1) * (1.0 - alpha.clamp(0, 1)) / d32

        # Transport: driven by alpha evolution
        S_transport = self._C_transport * dalpha_dt.abs() / d32

        # Equilibrium relaxation
        tau_relax = 0.5
        S_eq = (a_i_eq - a_i).clamp(min=0.0) / tau_relax

        return {
            "transport": S_transport,
            "equilibrium": S_eq,
        }
