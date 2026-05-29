"""
Enhanced turbulence damping models for multiphase flows — v5.

在 v4 基础上增加：

- **NearWallAnisotropicDamping**：考虑近壁面湍流各向异性的阻尼模型
- **BetaDampedModel**：基于 beta 分布的阻尼模型，平滑过渡
- **MultiScaleDamping**：多尺度阻尼，不同湍流尺度用不同阻尼策略

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_5 import (
        NearWallAnisotropicDamping,
        BetaDampedModel,
        MultiScaleDamping,
    )

    model = NearWallAnisotropicDamping(damping_coeff=10.0)
    k_damped = model.damp_k(alpha, k, y_plus=yp, aniso_ratio=ar)
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_4 import (
    TurbulenceDamping4EnhancedModel,
)

__all__ = [
    "TurbulenceDamping5EnhancedModel",
    "NearWallAnisotropicDamping",
    "BetaDampedModel",
    "MultiScaleDamping",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping5EnhancedModel(TurbulenceDamping4EnhancedModel):
    """Enhanced abstract base for v5 turbulence damping in multiphase.

    Extends v4 with anisotropic, beta-distributed, and multi-scale strategies.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping5EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping5EnhancedModel]) -> Type[TurbulenceDamping5EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping v5 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping5EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping v5 model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 近壁面各向异性阻尼
# ======================================================================

@TurbulenceDamping5EnhancedModel.register("nearWallAnisotropic")
class NearWallAnisotropicDamping(TurbulenceDamping5EnhancedModel):
    """Near-wall anisotropic turbulence damping model.

    考虑近壁面湍流的各向异性特性：

        f_damp = C * alpha * (1-alpha) * A(aniso) * h(y+)

    其中 A(aniso) 是各向异性调节函数：
        A = (1 - f_aniso * (1 - k_2/k_1))

    k_1, k_2 是主方向和次方向的湍动能分量。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    aniso_factor : float
        Anisotropy weighting factor. Default ``0.3``.
    y_plus_visc : float
        y+ threshold for viscous sublayer. Default ``5.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        aniso_factor: float = 0.3,
        y_plus_visc: float = 5.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._aniso_factor = max(0.0, min(aniso_factor, 1.0))
        self._y_plus_visc = max(y_plus_visc, _EPS)

    @property
    def aniso_factor(self) -> float:
        return self._aniso_factor

    @property
    def y_plus_visc(self) -> float:
        return self._y_plus_visc

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute anisotropic near-wall damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``y_plus``: ``(n_cells,)`` y+ values.
            ``aniso_ratio``: ``(n_cells,)`` k_minor/k_major ratio (0-1).

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

        # Anisotropy factor
        aniso_ratio = kwargs.get("aniso_ratio", None)
        if aniso_ratio is not None:
            ar = aniso_ratio.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            A = 1.0 - self._aniso_factor * (1.0 - ar)
        else:
            A = torch.ones_like(alpha_c)

        # Wall factor: enhanced near-wall damping
        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            # Stronger in viscous sublayer, decay in log-law
            wall_factor = torch.exp(-yp.pow(2) / (2.0 * self._y_plus_visc.pow(2)))
        else:
            wall_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * A * wall_factor * in_interface.to(dtype)


# ======================================================================
# Beta 分布阻尼
# ======================================================================

@TurbulenceDamping5EnhancedModel.register("betaDamped")
class BetaDampedModel(TurbulenceDamping5EnhancedModel):
    """Beta-distribution-based turbulence damping model.

    使用 beta 分布平滑处理界面区域的阻尼强度，避免不连续：

        f_damp = C * B(alpha; a, b) * h(y+)

    其中 B(alpha; a, b) = alpha^(a-1) * (1-alpha)^(b-1) / B(a,b)
    归一化后使得峰值阻尼在 alpha = a/(a+b) 处。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    beta_a : float
        Beta distribution shape parameter a. Default ``2.0``.
    beta_b : float
        Beta distribution shape parameter b. Default ``2.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        beta_a: float = 2.0,
        beta_b: float = 2.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._beta_a = max(beta_a, _EPS)
        self._beta_b = max(beta_b, _EPS)
        # Pre-compute normalization constant
        self._beta_norm = (
            math.gamma(self._beta_a) * math.gamma(self._beta_b)
            / math.gamma(self._beta_a + self._beta_b)
        )

    @property
    def beta_a(self) -> float:
        return self._beta_a

    @property
    def beta_b(self) -> float:
        return self._beta_b

    def _beta_kernel(self, alpha: torch.Tensor) -> torch.Tensor:
        """Evaluate the unnormalized beta distribution kernel.

        B(alpha; a, b) ~ alpha^(a-1) * (1-alpha)^(b-1)
        """
        a = self._beta_a
        b = self._beta_b
        alpha_c = alpha.clamp(_EPS, 1.0 - _EPS)

        kernel = alpha_c.pow(a - 1.0) * (1.0 - alpha_c).pow(b - 1.0)
        # Normalize so peak is 1.0
        peak_alpha = (a - 1.0) / max(a + b - 2.0, _EPS) if (a + b) > 2.0 else 0.5
        peak_alpha = max(_EPS, min(peak_alpha, 1.0 - _EPS))
        peak_val = peak_alpha.pow(a - 1.0) * (1.0 - peak_alpha).pow(b - 1.0)

        return kernel / max(peak_val, _EPS)

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute beta-distributed damping factor.

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

        # Beta kernel for smooth interface damping
        beta_kernel = self._beta_kernel(alpha_c)

        # Wall factor (optional)
        y_plus = kwargs.get("y_plus", None)
        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            wall_factor = torch.exp(-yp / 30.0)
        else:
            wall_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * beta_kernel * wall_factor * in_interface.to(dtype)


# ======================================================================
# 多尺度阻尼
# ======================================================================

@TurbulenceDamping5EnhancedModel.register("multiScale")
class MultiScaleDamping(TurbulenceDamping5EnhancedModel):
    """Multi-scale turbulence damping model.

    不同湍流尺度使用不同阻尼策略：

    - 大尺度（含能区）：基于 alpha 的标准阻尼
    - 中尺度（惯性子区）：基于 epsilon 和 |grad(alpha)| 的增强阻尼
    - 小尺度（耗散区）：基于 Kolmogorov 尺度的壁面增强阻尼

    f_damp = f_large + f_medium + f_small

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    large_scale_weight : float
        Weight for large-scale damping. Default ``0.5``.
    medium_scale_weight : float
        Weight for medium-scale damping. Default ``0.3``.
    small_scale_weight : float
        Weight for small-scale damping. Default ``0.2``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        large_scale_weight: float = 0.5,
        medium_scale_weight: float = 0.3,
        small_scale_weight: float = 0.2,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        # Normalize weights
        w_total = large_scale_weight + medium_scale_weight + small_scale_weight
        w_total = max(w_total, _EPS)
        self._w_large = large_scale_weight / w_total
        self._w_medium = medium_scale_weight / w_total
        self._w_small = small_scale_weight / w_total

    @property
    def weights(self) -> tuple[float, float, float]:
        """Scale weights (large, medium, small)."""
        return self._w_large, self._w_medium, self._w_small

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute multi-scale damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``epsilon``: ``(n_cells,)`` dissipation rate.
            ``grad_alpha_mag``: ``(n_cells,)`` gradient magnitude.
            ``y_plus``: ``(n_cells,)`` y+ values.
            ``nu``: kinematic viscosity (scalar).

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

        # Large-scale: standard alpha damping
        f_large = self._w_large * alpha_damping

        # Medium-scale: enhanced where epsilon and gradient are high
        epsilon = kwargs.get("epsilon", None)
        grad_alpha_mag = kwargs.get("grad_alpha_mag", None)

        if epsilon is not None and grad_alpha_mag is not None:
            eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
            grad_t = grad_alpha_mag.to(device=device, dtype=dtype).abs()
            # Normalize by reference values
            eps_norm = (eps_t / eps_t.mean().clamp(min=_EPS)).clamp(max=3.0)
            grad_norm = (grad_t / grad_t.mean().clamp(min=_EPS)).clamp(max=3.0)
            f_medium = self._w_medium * alpha_damping * (eps_norm * grad_norm).sqrt()
        else:
            f_medium = self._w_medium * alpha_damping

        # Small-scale: wall-enhanced using y+ and Kolmogorov
        y_plus = kwargs.get("y_plus", None)
        nu = kwargs.get("nu", 1e-6)

        if y_plus is not None:
            yp = y_plus.to(device=device, dtype=dtype).clamp(min=0.0)
            nu_t = torch.tensor(nu, device=device, dtype=dtype)

            if epsilon is not None:
                eps_t = epsilon.to(device=device, dtype=dtype).clamp(min=_EPS)
                # Kolmogorov length scale: eta = (nu^3/epsilon)^(1/4)
                eta = (nu_t.pow(3) / eps_t).pow(0.25)
                # Enhanced damping at small scales (eta small)
                scale_factor = (1.0 / eta.clamp(min=_EPS)).clamp(max=10.0)
                scale_factor = scale_factor / scale_factor.mean().clamp(min=_EPS)
            else:
                scale_factor = torch.ones_like(alpha_c)

            wall_factor = torch.exp(-yp.pow(2) / 200.0)
            f_small = self._w_small * alpha_damping * wall_factor * scale_factor
        else:
            f_small = self._w_small * alpha_damping

        f_total = (f_large + f_medium + f_small) * in_interface.to(dtype)

        return self.damping_coeff * f_total
