"""
Enhanced turbulence damping models for multiphase flows — v4.

在 v3 (近壁面处理) 基础上增加：

- **ReynoldsAdaptiveDamping**：基于局部 Re 自适应调整阻尼强度
- **TwoLayerDamping**：两层模型（内层+外层），不同阻尼策略
- **AlphaGradientLimiter**：基于 alpha 梯度的阻尼限制器

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_4 import (
        ReynoldsAdaptiveDamping,
        TwoLayerDamping,
        AlphaGradientLimiter,
    )

    model = ReynoldsAdaptiveDamping(damping_coeff=10.0)
    k_damped = model.damp_k(alpha, k, Re_loc=Re, y_plus=yp)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_3 import (
    TurbulenceDamping3EnhancedModel,
)

__all__ = [
    "TurbulenceDamping4EnhancedModel",
    "ReynoldsAdaptiveDamping",
    "TwoLayerDamping",
    "AlphaGradientLimiter",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping4EnhancedModel(TurbulenceDamping3EnhancedModel):
    """Enhanced abstract base for v4 turbulence damping in multiphase.

    Extends v3 with Reynolds-adaptive and two-layer strategies.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping4EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping4EnhancedModel]) -> Type[TurbulenceDamping4EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping v4 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping4EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping v4 model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# Reynolds 自适应阻尼
# ======================================================================

@TurbulenceDamping4EnhancedModel.register("reynoldsAdaptive")
class ReynoldsAdaptiveDamping(TurbulenceDamping4EnhancedModel):
    """Reynolds number-adaptive turbulence damping model.

    根据局部 Reynolds 数自适应调整阻尼强度：

        f_damp = C * alpha * (1-alpha) * g(Re_loc) * h(y+)

    其中 g(Re_loc) 是 Reynolds 数调节函数：
        g(Re) = 1.0                             if Re < Re_low
        g(Re) = 0.5 * (1 + tanh(slope*(Re-Re_mid)))  if Re_low <= Re <= Re_high
        g(Re) = g_max                            if Re > Re_high

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    Re_low : float
        Low Reynolds number threshold. Default ``100``.
    Re_high : float
        High Reynolds number threshold. Default ``10000``.
    g_max : float
        Maximum Reynolds factor. Default ``0.5``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        Re_low: float = 100.0,
        Re_high: float = 10000.0,
        g_max: float = 0.5,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._Re_low = max(Re_low, _EPS)
        self._Re_high = max(Re_high, self._Re_low + _EPS)
        self._g_max = max(g_max, _EPS)

    @property
    def Re_low(self) -> float:
        return self._Re_low

    @property
    def Re_high(self) -> float:
        return self._Re_high

    @property
    def g_max(self) -> float:
        return self._g_max

    def _reynolds_factor(self, Re_loc: torch.Tensor) -> torch.Tensor:
        """Reynolds number modulation factor."""
        Re = Re_loc.clamp(min=0.0)
        slope = 4.0 / max(self._Re_high - self._Re_low, _EPS)
        Re_mid = 0.5 * (self._Re_low + self._Re_high)
        return self._g_max * 0.5 * (1.0 + torch.tanh(slope * (Re - Re_mid)))

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Reynolds-adaptive damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``Re_loc``: local Reynolds number ``(n_cells,)``.
            ``y_plus``: y+ values ``(n_cells,)``.

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

        # Reynolds factor
        Re_loc = kwargs.get("Re_loc", None)
        if Re_loc is not None:
            g_Re = self._reynolds_factor(
                Re_loc.to(device=device, dtype=dtype),
            )
        else:
            g_Re = torch.ones_like(alpha_c)

        # Wall factor (optional, simplified)
        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            wall_factor = torch.exp(-yp / 30.0)
        else:
            wall_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * g_Re * wall_factor * in_interface.to(dtype)


# ======================================================================
# 两层阻尼模型
# ======================================================================

@TurbulenceDamping4EnhancedModel.register("twoLayer")
class TwoLayerDamping(TurbulenceDamping4EnhancedModel):
    """Two-layer turbulence damping model.

    将阻尼分为内层（近壁面）和外层（自由面）两个区域：

    - 内层 (y+ < y+_switch): f_inner = C_wall * exp(-y+^2 / (2*y+_s^2))
    - 外层 (y+ >= y+_switch): f_outer = C * alpha * (1-alpha)

    混合方式：在 y+_switch 附近平滑过渡。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient (outer layer). Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    y_plus_switch : float
        y+ threshold for inner/outer transition. Default ``11.0``.
    wall_damping_coeff : float
        Inner layer (wall) damping intensity. Default ``8.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        y_plus_switch: float = 11.0,
        wall_damping_coeff: float = 8.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._y_plus_switch = max(y_plus_switch, _EPS)
        self._wall_damping_coeff = wall_damping_coeff

    @property
    def y_plus_switch(self) -> float:
        return self._y_plus_switch

    @property
    def wall_damping_coeff(self) -> float:
        return self._wall_damping_coeff

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute two-layer damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``y_plus``: ``(n_cells,)`` y+ values.

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

        # Outer layer: standard interface damping
        f_outer = self.damping_coeff * alpha_damping * in_interface.to(dtype)

        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)

            # Inner layer: wall-based damping
            f_inner = self._wall_damping_coeff * torch.exp(
                -0.5 * (yp / self._y_plus_switch).pow(2)
            )

            # Smooth blending: sigmoid transition at y_plus_switch
            # blend=0 at wall (inner), blend=1 far from wall (outer)
            blend = torch.sigmoid((yp - self._y_plus_switch) / (0.1 * self._y_plus_switch))
            f = blend * f_outer + (1.0 - blend) * f_inner
        else:
            f = f_outer

        return f


# ======================================================================
# Alpha 梯度限制器
# ======================================================================

@TurbulenceDamping4EnhancedModel.register("alphaGradientLimiter")
class AlphaGradientLimiter(TurbulenceDamping4EnhancedModel):
    """Alpha gradient-limited turbulence damping model.

    使用 alpha 梯度幅值来限制阻尼范围，避免在 alpha 均匀区域过度阻尼：

        f = C * min(|grad(alpha)|, grad_max) / grad_max * alpha * (1-alpha)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    grad_max : float
        Maximum gradient for normalization. Default ``1.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        grad_max: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._grad_max = max(grad_max, _EPS)

    @property
    def grad_max(self) -> float:
        return self._grad_max

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute gradient-limited damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``grad_alpha``: ``(n_cells, 3)`` gradient of alpha, or
            ``grad_alpha_mag``: ``(n_cells,)`` gradient magnitude.

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

        # Gradient magnitude
        grad_alpha = kwargs.get("grad_alpha", None)
        grad_alpha_mag = kwargs.get("grad_alpha_mag", None)

        if grad_alpha is not None:
            grad_mag = grad_alpha.to(device=device, dtype=dtype).norm(dim=-1)
        elif grad_alpha_mag is not None:
            grad_mag = grad_alpha_mag.to(device=device, dtype=dtype)
        else:
            # Fallback: use alpha-based proxy
            grad_mag = alpha_damping

        # Limit gradient
        grad_limited = torch.minimum(grad_mag, torch.tensor(self._grad_max, device=device, dtype=dtype))
        grad_norm = grad_limited / self._grad_max

        return self.damping_coeff * grad_norm * alpha_damping * in_interface.to(dtype)
