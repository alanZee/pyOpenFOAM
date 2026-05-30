"""Enhanced turbulence damping models for multiphase flows — v12.

Extends v11 with:
- Gradient-aware interface damping with smooth transition
- Scale-dependent damping for LES/RANS hybrid
- Monitoring and adaptive coefficient adjustment

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_11 import (
        GradientAwareDamping,
        ScaleDependentDamping,
        AdaptiveMonitoringDamping,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_10 import TurbulenceDamping10EnhancedModel

__all__ = [
    "TurbulenceDamping11EnhancedModel",
    "GradientAwareDamping",
    "ScaleDependentDamping",
    "AdaptiveMonitoringDamping",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class TurbulenceDamping11EnhancedModel(TurbulenceDamping10EnhancedModel):
    """Enhanced abstract base for v11 turbulence damping in multiphase."""
    _registry: ClassVar[dict[str, Type["TurbulenceDamping11EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[TurbulenceDamping11EnhancedModel]) -> Type[TurbulenceDamping11EnhancedModel]:
            if name in cls._registry:
                raise ValueError(f"Damping v11 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping11EnhancedModel":
        if name not in cls._registry:
            raise KeyError(f"Unknown damping v11 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@TurbulenceDamping11EnhancedModel.register("gradientAware")
class GradientAwareDamping(TurbulenceDamping11EnhancedModel):
    """Gradient-aware interface damping model.

    Weights the damping by the alpha gradient magnitude to
    concentrate damping at sharp interfaces:

        D = C_base * |grad(alpha)| * alpha*(1-alpha)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Gradient magnitude (use finite differences as proxy)
        grad_alpha = kwargs.get("grad_alpha", torch.ones_like(alpha_c))
        grad_mag = grad_alpha.to(device=device, dtype=dtype).abs().clamp(max=10.0)
        f_grad = (grad_mag / (grad_mag + 1.0)).clamp(0.0, 1.0)

        return self.damping_coeff * alpha_damping * f_grad * in_interface.to(dtype)


@TurbulenceDamping11EnhancedModel.register("scaleDependent")
class ScaleDependentDamping(TurbulenceDamping11EnhancedModel):
    """Scale-dependent damping model for LES/RANS hybrid.

    Applies different damping coefficients at different turbulence
    scales (integral, Taylor, Kolmogorov):

        D = C_base * alpha*(1-alpha) * f(scale_ratio)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    scale_ratio_coeff : float
        Scale ratio weighting coefficient. Default 1.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        scale_ratio_coeff: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_scale = max(0.0, scale_ratio_coeff)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Scale ratio: ratio of grid scale to integral scale
        delta = kwargs.get("delta", torch.ones_like(alpha_c) * 0.01)
        L_int = kwargs.get("L_integral", torch.ones_like(alpha_c) * 0.1)
        delta_t = delta.to(device=device, dtype=dtype).abs().clamp(min=_EPS)
        L_t = L_int.to(device=device, dtype=dtype).abs().clamp(min=_EPS)
        scale_ratio = (delta_t / L_t).clamp(0.0, 1.0)

        f_scale = (1.0 + self._C_scale * scale_ratio).clamp(max=3.0)

        return self.damping_coeff * alpha_damping * f_scale * in_interface.to(dtype)


@TurbulenceDamping11EnhancedModel.register("adaptiveMonitoring")
class AdaptiveMonitoringDamping(TurbulenceDamping11EnhancedModel):
    """Adaptive monitoring damping model.

    Adjusts damping coefficient based on real-time monitoring
    of interface behavior with alert thresholds:

        D = C_adapted * alpha*(1-alpha)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    adaptation_rate : float
        Rate of coefficient adaptation. Default 0.1.
    alert_threshold : float
        Damping factor alert threshold. Default 5.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        adaptation_rate: float = 0.1,
        alert_threshold: float = 5.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._adapt_rate = max(0.0, min(adaptation_rate, 1.0))
        self._alert_thresh = max(0.0, alert_threshold)
        self._current_coeff = damping_coeff
        self._alert_history: list[dict[str, float]] = []

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        d_mean = float((self._current_coeff * alpha_damping * in_interface.to(dtype)).mean().item())

        # Adapt coefficient
        if d_mean > self._alert_thresh:
            self._current_coeff *= (1.0 - self._adapt_rate)
        else:
            self._current_coeff = self._current_coeff * (1.0 - self._adapt_rate) + \
                                  self.damping_coeff * self._adapt_rate

        self._alert_history.append({"mean_damping": d_mean, "coeff": self._current_coeff})

        return self._current_coeff * alpha_damping * in_interface.to(dtype)

    @property
    def alert_history(self) -> list[dict[str, float]]:
        return self._alert_history
