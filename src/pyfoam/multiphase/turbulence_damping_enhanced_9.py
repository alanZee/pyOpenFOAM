"""Enhanced turbulence damping models for multiphase flows — v9.

Extends v8 with:
- **自适应阻尼系数**: adaptive damping coefficient based on local flow conditions
- **湍流动能产生项阻尼**: TKE production-aware damping
- **分层流阻尼**: stratified flow damping model

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_9 import (
        AdaptiveCoefficientDamping,
        TKEProductionDamping,
        StratifiedFlowDamping,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_8 import TurbulenceDamping8EnhancedModel

__all__ = [
    "TurbulenceDamping9EnhancedModel",
    "AdaptiveCoefficientDamping",
    "TKEProductionDamping",
    "StratifiedFlowDamping",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class TurbulenceDamping9EnhancedModel(TurbulenceDamping8EnhancedModel):
    """Enhanced abstract base for v9 turbulence damping in multiphase."""
    _registry: ClassVar[dict[str, Type["TurbulenceDamping9EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[TurbulenceDamping9EnhancedModel]) -> Type[TurbulenceDamping9EnhancedModel]:
            if name in cls._registry:
                raise ValueError(f"Damping v9 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping9EnhancedModel":
        if name not in cls._registry:
            raise KeyError(f"Unknown damping v9 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@TurbulenceDamping9EnhancedModel.register("adaptiveCoefficient")
class AdaptiveCoefficientDamping(TurbulenceDamping9EnhancedModel):
    """Adaptive coefficient damping model.

    Adjusts the damping coefficient based on the local volume fraction
    gradient magnitude and turbulence intensity:

        C_adapt = C_base * (1 + C_grad * |grad(alpha)|) / (1 + C_turb * k)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds. Default 0.01, 0.99.
    C_grad : float
        Gradient enhancement coefficient. Default 1.0.
    C_turb : float
        Turbulence attenuation coefficient. Default 0.5.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_grad: float = 1.0,
        C_turb: float = 0.5,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_grad = max(0.0, C_grad)
        self._C_turb = max(0.0, C_turb)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        grad_alpha = kwargs.get("grad_alpha_mag", torch.zeros_like(alpha_c))
        k = kwargs.get("k", torch.zeros_like(alpha_c))
        grad_t = grad_alpha.to(device=device, dtype=dtype).abs()
        k_t = k.to(device=device, dtype=dtype).abs()

        C_adapt = self.damping_coeff * (1.0 + self._C_grad * grad_t.clamp(max=5.0))
        C_adapt = C_adapt / (1.0 + self._C_turb * k_t.sqrt().clamp(max=10.0))

        return C_adapt * alpha_damping * in_interface.to(dtype)


@TurbulenceDamping9EnhancedModel.register("tkeProduction")
class TKEProductionDamping(TurbulenceDamping9EnhancedModel):
    """TKE production-aware damping model.

    Reduces damping when TKE production is high to avoid
    over-damping turbulence generation at the interface:

        D = C_base * alpha*(1-alpha) * max(0, 1 - P_k / (C_lim * epsilon))

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    C_limit : float
        Production limit ratio. Default 2.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_limit: float = 2.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_limit = max(0.1, C_limit)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        P_k = kwargs.get("P_k", torch.zeros_like(alpha_c))
        eps = kwargs.get("epsilon", torch.ones_like(alpha_c))
        P_k_t = P_k.to(device=device, dtype=dtype).abs()
        eps_t = eps.to(device=device, dtype=dtype).abs().clamp(min=_EPS)

        production_ratio = (P_k_t / (self._C_limit * eps_t)).clamp(max=1.0)
        f_prod = (1.0 - production_ratio).clamp(min=0.0, max=1.0)

        return self.damping_coeff * alpha_damping * f_prod * in_interface.to(dtype)


@TurbulenceDamping9EnhancedModel.register("stratifiedFlow")
class StratifiedFlowDamping(TurbulenceDamping9EnhancedModel):
    """Stratified flow damping model.

    Applies enhanced damping for stratified flows where gravity-driven
    mixing suppression is important:

        D = C_base * alpha*(1-alpha) * (1 + Ri / (1 + Ri))

    where Ri is the local Richardson number.

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    g : float
        Gravitational acceleration (m/s^2). Default 9.81.
    rho_ref : float
        Reference density difference (kg/m^3). Default 500.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        g: float = 9.81,
        rho_ref: float = 500.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._g = max(0.0, g)
        self._rho_ref = max(_EPS, rho_ref)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        S_mag = kwargs.get("S_mag", torch.ones_like(alpha_c))
        L_char = kwargs.get("L_char", torch.full_like(alpha_c, 0.01))
        S_t = S_mag.to(device=device, dtype=dtype).abs().clamp(min=_EPS)
        L_t = L_char.to(device=device, dtype=dtype).abs().clamp(min=_EPS)

        # Richardson number: Ri = g * delta_rho * L / (rho_ref * S^2)
        Ri = (self._g * self._rho_ref * L_t / (S_t.pow(2))).clamp(max=100.0)
        f_ri = (1.0 + Ri / (1.0 + Ri)).clamp(max=3.0)

        return self.damping_coeff * alpha_damping * f_ri * in_interface.to(dtype)
