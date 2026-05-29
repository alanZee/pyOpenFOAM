"""
Enhanced turbulence damping models for multiphase flows — v3.

在 Phase 1 (gradient, exponential) 和 Phase 2 (Lopez de Bertodano, Kataoka)
基础上增加 **近壁面处理** 增强：

- **WallDampedDamping**：基于 y+ 的壁面感知阻尼，结合标准阻尼和壁面阻尼
- **SpaldingDamping**：基于 Spalding 壁面律的阻尼模型
- **BlendedWallInterfaceDamping**：壁面-界面耦合阻尼，处理近壁面自由面

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_3 import (
        WallDampedDamping,
        SpaldingDamping,
        BlendedWallInterfaceDamping,
    )

    model = WallDampedDamping(damping_coeff=10.0, y_plus_switch=30.0)
    k_damped = model.damp_k(alpha, k, y_plus=yp, wall_distance=d)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceDamping3EnhancedModel",
    "WallDampedDamping",
    "SpaldingDamping",
    "BlendedWallInterfaceDamping",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping3EnhancedModel(ABC):
    """Enhanced abstract base for near-wall turbulence damping in multiphase.

    Subclasses implement wall-aware damping strategies that combine
    free-surface damping with near-wall turbulence models.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping3EnhancedModel"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping3EnhancedModel]) -> Type[TurbulenceDamping3EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping v3 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping3EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping v3 model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abstractmethod
    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute damping factor ``(n_cells,)``."""

    def damp_k(
        self,
        alpha: torch.Tensor,
        k_field: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp turbulent kinetic energy: k_damped = k * exp(-f)."""
        f = self.compute_damping_factor(alpha, **kwargs)
        return k_field * torch.exp(-f)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon_field: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp dissipation rate: eps_damped = eps * exp(-f)."""
        f = self.compute_damping_factor(alpha, **kwargs)
        return epsilon_field * torch.exp(-f)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega_field: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp specific dissipation: omega_damped = omega * exp(-f)."""
        f = self.compute_damping_factor(alpha, **kwargs)
        return omega_field * torch.exp(-f)


# ======================================================================
# 壁面感知阻尼模型
# ======================================================================

@TurbulenceDamping3EnhancedModel.register("wallDampedDamping")
class WallDampedDamping(TurbulenceDamping3EnhancedModel):
    """Wall-aware turbulence damping model.

    结合自由面阻尼和近壁面阻尼：

        f = f_interface * f_wall

    其中:
    - f_interface: 基于 alpha 的自由面阻尼（bell-shaped）
    - f_wall: 基于 y+ 的壁面阻尼（viscous sublayer 加强阻尼）

    壁面阻尼因子:

        f_wall = 1.0                                        if y+ > y_plus_switch
        f_wall = exp(-y+^2 / (2 * y_plus_switch^2))        if y+ <= y_plus_switch

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    y_plus_switch : float
        y+ threshold for wall damping transition. Default ``30.0``.
    wall_damping_coeff : float
        Wall damping intensity. Default ``5.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        y_plus_switch: float = 30.0,
        wall_damping_coeff: float = 5.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._y_plus_switch = max(y_plus_switch, _EPS)
        self._wall_damping_coeff = wall_damping_coeff

    @property
    def y_plus_switch(self) -> float:
        """y+ threshold for wall damping transition."""
        return self._y_plus_switch

    @property
    def wall_damping_coeff(self) -> float:
        """Wall damping intensity."""
        return self._wall_damping_coeff

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute wall-aware damping factor.

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

        # Interface damping: bell-shaped between alpha_min and alpha_max
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        interface_factor = self.damping_coeff * 4.0 * alpha_c * (1.0 - alpha_c)
        f_interface = interface_factor * in_interface.to(dtype)

        # Wall damping
        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            # Gaussian decay: stronger damping at small y+
            f_wall = self._wall_damping_coeff * torch.exp(
                -0.5 * (yp / self._y_plus_switch).pow(2)
            )
        else:
            f_wall = torch.zeros_like(alpha_c)

        # Combined: product of interface and wall damping
        return f_interface + f_wall


# ======================================================================
# Spalding 壁面律阻尼
# ======================================================================

@TurbulenceDamping3EnhancedModel.register("spaldingDamping")
class SpaldingDamping(TurbulenceDamping3EnhancedModel):
    """Spalding wall-function-based damping model.

    使用 Spalding 壁面律确定湍流阻尼强度：

        y+ = u+ + exp(-kappa*B) * [exp(kappa*u+) - 1 - kappa*u+
             - (kappa*u+)^2/2 - (kappa*u+)^3/6]

    阻尼因子基于 u+ 与 y+ 的关系：

        f = C * alpha * (1 - alpha) * f_spalding(y+)

    where f_spalding interpolates between viscous-sublayer and log-law regimes.

    Parameters
    ----------
    damping_coeff : float
        Damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    kappa : float
        von Karman constant. Default ``0.41``.
    B : float
        Spalding constant. Default ``5.5``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        kappa: float = 0.41,
        B: float = 5.5,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._kappa = kappa
        self._B = B

    @property
    def kappa(self) -> float:
        """von Karman constant."""
        return self._kappa

    @property
    def B(self) -> float:
        """Spalding constant."""
        return self._B

    def _spalding_f(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Spalding wall function damping shape.

        Returns a value in [0, 1] where:
        - 1 = strong damping (viscous sublayer, y+ < 5)
        - 0 = no damping (log-law region, y+ > 30)
        """
        yp = y_plus.clamp(min=_EPS)

        # Smooth transition between viscous and log-law
        # f_spalding = exp(-y+ / y+_switch)
        y_switch = 11.0  # Transition at y+ ~ 11 (buffer layer)
        f = torch.exp(-yp / y_switch)

        return f.clamp(0.0, 1.0)

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Spalding wall-law damping factor.

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

        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype)
            spalding_f = self._spalding_f(yp)
            factor = self.damping_coeff * alpha_damping * spalding_f
        else:
            # No wall info: use standard alpha-based damping only
            factor = self.damping_coeff * alpha_damping * 0.5

        return factor * in_interface.to(dtype)


# ======================================================================
# 壁面-界面耦合阻尼
# ======================================================================

@TurbulenceDamping3EnhancedModel.register("blendedWallInterface")
class BlendedWallInterfaceDamping(TurbulenceDamping3EnhancedModel):
    """Blended wall-interface damping model.

    处理近壁面自由面（如沿壁面上升的气泡）的湍流阻尼。

    将阻尼分解为两个正交分量：
    1. 界面阻尼：基于 |grad(alpha)| 的自由面阻尼
    2. 壁面阻尼：基于 wall_distance 的壁面阻尼

    混合权重：

        w_wall = exp(-d_wall / d_ref)
        f = (1 - w_wall) * f_interface + w_wall * f_wall

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    d_ref : float
        Reference wall distance for blending (m). Default ``1e-3``.
    wall_damping_coeff : float
        Wall damping strength. Default ``8.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        d_ref: float = 1e-3,
        wall_damping_coeff: float = 8.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._d_ref = max(d_ref, _EPS)
        self._wall_damping_coeff = wall_damping_coeff

    @property
    def d_ref(self) -> float:
        """Reference wall distance (m)."""
        return self._d_ref

    @property
    def wall_damping_coeff(self) -> float:
        """Wall damping strength."""
        return self._wall_damping_coeff

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute blended wall-interface damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``wall_distance``: ``(n_cells,)`` distance to nearest wall (m).
            ``grad_alpha``: ``(n_cells, 3)`` gradient of alpha.

        Returns
        -------
        torch.Tensor
            Damping factor ``(n_cells,)``.
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        # Interface damping: gradient-based
        grad_alpha = kwargs.get("grad_alpha", None)
        if grad_alpha is not None:
            grad_mag = grad_alpha.to(device=device, dtype=dtype).norm(dim=-1)
            f_interface = self.damping_coeff * grad_mag.clamp(0.0, 1.0)
        else:
            # Fallback: alpha-based proxy
            f_interface = self.damping_coeff * 4.0 * alpha_c * (1.0 - alpha_c)
        f_interface = f_interface * in_interface.to(dtype)

        # Wall damping
        wall_distance = kwargs.get("wall_distance", None)
        if wall_distance is not None:
            d = wall_distance.to(device=device, dtype=dtype).clamp(min=0.0)
            w_wall = torch.exp(-d / self._d_ref)
            f_wall = self._wall_damping_coeff * w_wall
        else:
            w_wall = torch.zeros_like(alpha_c)
            f_wall = torch.zeros_like(alpha_c)

        # Blended
        return (1.0 - w_wall) * f_interface + w_wall * f_wall
