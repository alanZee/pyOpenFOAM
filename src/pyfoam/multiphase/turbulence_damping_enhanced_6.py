"""
Enhanced turbulence damping models for multiphase flows — v6.

在 v5 基础上增加：

- **LES 感知阻尼**：针对大涡模拟滤波尺度的阻尼策略
- **动态系数调节**：根据局部湍流状态自适应调节阻尼系数
- **界面拓扑感知阻尼**：考虑界面拓扑结构（连通/离散）的阻尼

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_6 import (
        LESAwareDamping,
        DynamicCoefficientDamping,
        TopologyAwareDamping,
    )

    model = LESAwareDamping(damping_coeff=10.0)
    k_damped = model.damp_k(alpha, k, delta=grid_delta)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_5 import (
    TurbulenceDamping5EnhancedModel,
)

__all__ = [
    "TurbulenceDamping6EnhancedModel",
    "LESAwareDamping",
    "DynamicCoefficientDamping",
    "TopologyAwareDamping",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping6EnhancedModel(TurbulenceDamping5EnhancedModel):
    """Enhanced abstract base for v6 turbulence damping in multiphase.

    Extends v5 with LES-aware, dynamic coefficient, and topology-aware strategies.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping6EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping6EnhancedModel]) -> Type[TurbulenceDamping6EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping v6 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping6EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping v6 model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# LES 感知阻尼
# ======================================================================

@TurbulenceDamping6EnhancedModel.register("lesAware")
class LESAwareDamping(TurbulenceDamping6EnhancedModel):
    """LES-aware turbulence damping model.

    针对大涡模拟中滤波尺度的阻尼策略：

    - 滤波尺度 delta 与界面厚度的关系
    - 次网格尺度阻尼增强
    - 亚格子模型感知的阻尼系数

    f_damp = C * alpha*(1-alpha) * H(delta/delta_interface) * g(SGS)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    delta_interface : float
        Interface thickness scale (m). Default ``1e-3``.
    sgs_weight : float
        Sub-grid scale damping weight. Default ``0.5``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        delta_interface: float = 1e-3,
        sgs_weight: float = 0.5,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._delta_interface = max(_EPS, delta_interface)
        self._sgs_weight = max(0.0, min(sgs_weight, 1.0))

    @property
    def delta_interface(self) -> float:
        return self._delta_interface

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute LES-aware damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``delta``: ``(n_cells,)`` grid filter width.
            ``sgs_energy``: ``(n_cells,)`` sub-grid turbulent energy.

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)``.
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Grid filter width ratio: H(delta/delta_interface)
        delta = kwargs.get("delta", None)
        if delta is not None:
            delta_t = delta.to(device=device, dtype=dtype).clamp(min=_EPS)
            ratio = delta_t / self._delta_interface
            # Smooth step: increases damping when delta > delta_interface
            grid_factor = torch.tanh(ratio.pow(2))
        else:
            grid_factor = torch.ones_like(alpha_c)

        # SGS energy factor: g(SGS) = 1 + sgs_weight * (k_SGS / k_ref)
        sgs_energy = kwargs.get("sgs_energy", None)
        if sgs_energy is not None:
            sgs_t = sgs_energy.to(device=device, dtype=dtype).clamp(min=_EPS)
            k_ref = sgs_t.mean().clamp(min=_EPS)
            sgs_factor = 1.0 + self._sgs_weight * (sgs_t / k_ref).clamp(max=3.0)
        else:
            sgs_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * grid_factor * sgs_factor * in_interface.to(dtype)


# ======================================================================
# 动态系数调节
# ======================================================================

@TurbulenceDamping6EnhancedModel.register("dynamicCoefficient")
class DynamicCoefficientDamping(TurbulenceDamping6EnhancedModel):
    """Dynamic coefficient turbulence damping model.

    根据局部湍流状态自适应调节阻尼系数：

    C_eff = C_base * f(Re_t) * g(k_ratio)

    其中 Re_t 是湍流 Reynolds 数，k_ratio = k / k_eq 是湍动能比值。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    Re_t_ref : float
        Reference turbulent Reynolds number. Default ``100.0``.
    k_ratio_relax : float
        Relaxation factor for k_ratio correction. Default ``0.3``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        Re_t_ref: float = 100.0,
        k_ratio_relax: float = 0.3,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._Re_t_ref = max(_EPS, Re_t_ref)
        self._k_ratio_relax = max(0.0, min(k_ratio_relax, 1.0))

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute dynamically adjusted damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``k``: ``(n_cells,)`` turbulent kinetic energy.
            ``epsilon``: ``(n_cells,)`` dissipation rate.
            ``nu``: kinematic viscosity (scalar).
            ``k_eq``: ``(n_cells,)`` equilibrium TKE.

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)``.
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Re_t adjustment
        k = kwargs.get("k", None)
        epsilon = kwargs.get("epsilon", None)
        nu = kwargs.get("nu", 1e-6)

        if k is not None and epsilon is not None:
            k_t = k.to(device=device, dtype=dtype).clamp(min=_EPS)
            eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
            nu_t = torch.tensor(nu, device=device, dtype=dtype)

            # Turbulent Reynolds number: Re_t = k^2 / (nu * epsilon)
            Re_t = k_t.pow(2) / (nu_t * eps_t).clamp(min=_EPS)
            # Normalized: increases damping for low Re_t
            Re_factor = torch.exp(-Re_t / self._Re_t_ref)
            Re_factor = (1.0 + Re_factor).clamp(max=3.0)
        else:
            Re_factor = torch.ones_like(alpha_c)

        # k_ratio adjustment
        k_eq = kwargs.get("k_eq", None)
        if k is not None and k_eq is not None:
            k_t = k.to(device=device, dtype=dtype).clamp(min=_EPS)
            k_eq_t = k_eq.to(device=device, dtype=dtype).clamp(min=_EPS)
            k_ratio = k_t / k_eq_t.clamp(min=_EPS)
            # Reduce damping when k >> k_eq (already damped enough)
            k_factor = (1.0 / k_ratio.clamp(min=_EPS)).clamp(max=2.0)
            k_factor = 1.0 + self._k_ratio_relax * (k_factor - 1.0)
        else:
            k_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * Re_factor * k_factor * in_interface.to(dtype)


# ======================================================================
# 界面拓扑感知阻尼
# ======================================================================

@TurbulenceDamping6EnhancedModel.register("topologyAware")
class TopologyAwareDamping(TurbulenceDamping6EnhancedModel):
    """Interface topology-aware turbulence damping model.

    根据界面拓扑结构（连通 vs 离散）选择不同阻尼策略：

    - 连续界面：标准阻尼（大尺度界面结构）
    - 离散液滴/气泡：增强阻尼（小尺度界面破碎）

    拓扑指标：
        T = |grad(alpha)| / (a_i * d32)

    T < 1 => 连续界面；T > 1 => 离散相

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    continuous_weight : float
        Damping weight for continuous interface. Default ``1.0``.
    dispersed_weight : float
        Damping weight for dispersed phase. Default ``2.0``.
    topology_threshold : float
        Topology transition threshold. Default ``1.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        continuous_weight: float = 1.0,
        dispersed_weight: float = 2.0,
        topology_threshold: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._continuous_weight = max(0.0, continuous_weight)
        self._dispersed_weight = max(0.0, dispersed_weight)
        self._topology_threshold = max(_EPS, topology_threshold)

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute topology-aware damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``grad_alpha_mag``: ``(n_cells,)`` gradient magnitude.
            ``a_i``: ``(n_cells,)`` interfacial area density.
            ``d32``: Sauter mean diameter (scalar or tensor).

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)``.
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Topology indicator
        grad_alpha = kwargs.get("grad_alpha_mag", None)
        a_i = kwargs.get("a_i", None)
        d32 = kwargs.get("d32", 3e-3)

        if grad_alpha is not None and a_i is not None:
            grad_t = grad_alpha.to(device=device, dtype=dtype).abs().clamp(min=_EPS)
            a_i_t = a_i.to(device=device, dtype=dtype).abs().clamp(min=_EPS)
            d32_t = (
                d32.to(device=device, dtype=dtype)
                if isinstance(d32, torch.Tensor)
                else torch.tensor(d32, device=device, dtype=dtype)
            )

            # T = |grad(alpha)| / (a_i * d32)
            topology = grad_t / (a_i_t * d32_t).clamp(min=_EPS)

            # Blend: continuous (T < threshold) vs dispersed (T > threshold)
            blend = torch.tanh(topology / self._topology_threshold)
            # -1 => continuous, +1 => dispersed
            weight = self._continuous_weight * (1.0 - blend) * 0.5 + \
                     self._dispersed_weight * (1.0 + blend) * 0.5
        else:
            weight = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * weight * in_interface.to(dtype)
