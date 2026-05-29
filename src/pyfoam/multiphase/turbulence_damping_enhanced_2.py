"""
Enhanced turbulence damping models for multiphase flows — Phase 2.

在 Phase 1 的梯度阻尼和指数混合阻尼基础上，提供基于物理关联的阻尼模型：

- **Lopez de Bertodano model**: 基于湍流相互作用时间尺度的阻尼
- **Kataoka model**: 基于湍流长度尺度与界面特征尺度比的阻尼

Provides:

- :class:`LopezDeBertodanoDamping` — Lopez de Bertodano turbulence damping
- :class:`KataokaDamping` — Kataoka turbulence damping

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_2 import (
        LopezDeBertodanoDamping, KataokaDamping,
    )

    model = LopezDeBertodanoDamping(
        C_td=1.0, alpha_max=0.3,
        d_bubble=1e-3, rho_c=1000.0, mu_c=1e-3,
    )
    k_damped = model.damp_k(alpha, k)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "TurbulenceDamping2EnhancedModel",
    "LopezDeBertodanoDamping",
    "KataokaDamping",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping2EnhancedModel(ABC):
    """Enhanced abstract base for physics-based turbulence damping.

    Subclasses implement physically motivated damping strategies
    based on interphase transfer correlations.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping2EnhancedModel"]]] = {}

    def __init__(
        self,
        damping_coeff: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        self.damping_coeff = damping_coeff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping2EnhancedModel]) -> Type[TurbulenceDamping2EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping2EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping model '{name}'. Available: {available}"
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
        """Damp turbulent kinetic energy: k_damped = k_field * exp(-f).

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        k_field : torch.Tensor
            Turbulent kinetic energy field ``(n_cells,)``.
        **kwargs
            Optional: ``tke`` for turbulence kinetic energy input to
            damping factor computation, ``epsilon`` for dissipation rate.
        """
        f = self.compute_damping_factor(alpha, **kwargs)
        return k_field * torch.exp(-f)

    def damp_epsilon(
        self,
        alpha: torch.Tensor,
        epsilon_field: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp dissipation rate: eps_damped = eps * exp(-f).

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        epsilon_field : torch.Tensor
            Dissipation rate field ``(n_cells,)``.
        **kwargs
            Optional: ``tke``, ``epsilon`` for damping factor computation.
        """
        f = self.compute_damping_factor(alpha, **kwargs)
        return epsilon_field * torch.exp(-f)

    def damp_omega(
        self,
        alpha: torch.Tensor,
        omega_field: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Damp specific dissipation: omega_damped = omega * exp(-f).

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        omega_field : torch.Tensor
            Specific dissipation field ``(n_cells,)``.
        **kwargs
            Optional: ``tke``, ``epsilon`` for damping factor computation.
        """
        f = self.compute_damping_factor(alpha, **kwargs)
        return omega_field * torch.exp(-f)


# ======================================================================
# Lopez de Bertodano 阻尼模型
# ======================================================================

@TurbulenceDamping2EnhancedModel.register("lopezDeBertodano")
class LopezDeBertodanoDamping(TurbulenceDamping2EnhancedModel):
    """Lopez de Bertodano turbulence damping model.

    基于湍流相互作用时间尺度的阻尼模型 (Lopez de Bertodano, 1994)。
    阻尼因子基于颗粒/气泡与湍流涡旋的时间尺度比：

    .. math::

        f = C_{td} \\cdot \\alpha \\cdot \\frac{\\tau_t}{\\tau_t + \\tau_p}

    其中:
    - :math:`C_{td}` 是阻尼系数
    - :math:`\\tau_t = k / \\epsilon` 是湍流时间尺度
    - :math:`\\tau_p = d_b / (6 \\nu_c)^{1/2}` 是颗粒时间尺度

    当颗粒时间尺度远大于湍流时间尺度时 (大颗粒/大涡)，
    颗粒对湍流的影响最大，阻尼最强。

    Parameters
    ----------
    damping_coeff : float
        Turbulence damping coefficient C_td. Default ``1.0``.
    alpha_min : float
        Lower alpha threshold for damping. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    d_bubble : float
        Bubble/particle diameter (m). Default ``1e-3``.
    rho_c : float
        Continuous phase density (kg/m3). Default ``1000.0``.
    mu_c : float
        Continuous phase dynamic viscosity (Pa*s). Default ``1e-3``.
    """

    def __init__(
        self,
        damping_coeff: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        d_bubble: float = 1e-3,
        rho_c: float = 1000.0,
        mu_c: float = 1e-3,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        if d_bubble <= 0.0:
            raise ValueError(f"d_bubble must be positive, got {d_bubble}")
        if rho_c <= 0.0:
            raise ValueError(f"rho_c must be positive, got {rho_c}")
        if mu_c <= 0.0:
            raise ValueError(f"mu_c must be positive, got {mu_c}")

        self.d_bubble = d_bubble
        self.rho_c = rho_c
        self.mu_c = mu_c

        # 颗粒时间尺度: tau_p = d_b / sqrt(6 * nu_c)
        nu_c = mu_c / rho_c
        self._tau_p = d_bubble / max((6.0 * nu_c) ** 0.5, _EPS)

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Lopez de Bertodano damping factor.

        Accepts optional kwargs:
        - ``tke``: turbulent kinetic energy ``(n_cells,)``
        - ``epsilon``: dissipation rate ``(n_cells,)``
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        # Interface region filter
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        # 湍流时间尺度
        k_tke = kwargs.get("tke", None)
        epsilon = kwargs.get("epsilon", None)

        if k_tke is not None and epsilon is not None:
            k_t = k_tke.to(device=device, dtype=dtype)
            eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
            tau_t = k_t / eps_t
        else:
            # 默认湍流时间尺度（假设中等湍流强度）
            tau_t = torch.full_like(alpha_c, 0.01)

        tau_p = torch.tensor(
            self._tau_p, device=device, dtype=dtype,
        )

        # f = C_td * alpha * tau_t / (tau_t + tau_p)
        time_ratio = tau_t / (tau_t + tau_p + _EPS)
        factor = self.damping_coeff * alpha_c * time_ratio

        return factor * in_interface.to(dtype)

    def __repr__(self) -> str:
        return (
            f"LopezDeBertodanoDamping("
            f"C_td={self.damping_coeff}, "
            f"d_bubble={self.d_bubble})"
        )


# ======================================================================
# Kataoka 阻尼模型
# ======================================================================

@TurbulenceDamping2EnhancedModel.register("kataoka")
class KataokaDamping(TurbulenceDamping2EnhancedModel):
    """Kataoka turbulence damping model.

    基于湍流长度尺度与界面特征尺度比的阻尼模型 (Kataoka et al., 2001)。
    当湍流涡旋尺度远大于气泡/颗粒尺度时，涡旋不受界面影响，
    阻尼较弱；当两者尺度接近时，阻尼最强。

    .. math::

        f = C_{td} \\cdot \\alpha (1 - \\alpha) \\cdot
            \\exp\\left(-\\left(\\frac{L_t}{d_b}\\right)^2 / 2\\right)

    其中:
    - :math:`L_t = (k^{3/2} / \\epsilon)` 是湍流积分长度尺度
    - :math:`d_b` 是气泡直径

    Parameters
    ----------
    damping_coeff : float
        Turbulence damping coefficient C_td. Default ``1.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    d_bubble : float
        Bubble/particle diameter (m). Default ``1e-3``.
    """

    def __init__(
        self,
        damping_coeff: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        d_bubble: float = 1e-3,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        if d_bubble <= 0.0:
            raise ValueError(f"d_bubble must be positive, got {d_bubble}")
        self.d_bubble = d_bubble

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Kataoka damping factor.

        Accepts optional kwargs:
        - ``tke``: turbulent kinetic energy ``(n_cells,)``
        - ``epsilon``: dissipation rate ``(n_cells,)``
        """
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)

        # Interface region filter
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)

        # 湍流积分长度尺度
        k_tke = kwargs.get("tke", None)
        epsilon = kwargs.get("epsilon", None)

        if k_tke is not None and epsilon is not None:
            k_t = k_tke.to(device=device, dtype=dtype).clamp(min=0.0)
            eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
            L_t = k_t.pow(1.5) / eps_t
        else:
            # 默认湍流长度尺度
            L_t = torch.full_like(alpha_c, 1e-3)

        # 长度尺度比: L_t / d_b
        ratio = L_t / self.d_bubble

        # f = C_td * alpha*(1-alpha) * exp(-(L_t/d_b)^2 / 2)
        size_damping = torch.exp(-0.5 * ratio.pow(2))
        factor = self.damping_coeff * alpha_c * (1.0 - alpha_c) * size_damping

        return factor * in_interface.to(dtype)

    def __repr__(self) -> str:
        return (
            f"KataokaDamping("
            f"C_td={self.damping_coeff}, "
            f"d_bubble={self.d_bubble})"
        )
