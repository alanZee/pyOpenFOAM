"""Enhanced turbulence damping models for multiphase flows — v10.

Extends v9 with:
- **距离加权阻尼**: distance-weighted damping for near-wall and near-interface regions
- **湍动能预算修正**: TKE budget correction to avoid over-damping production
- **界面湍流反馈**: interface-turbulence feedback coupling

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_10 import (
        DistanceWeightedDamping,
        TKEBudgetCorrectionDamping,
        InterfaceTurbulenceFeedbackDamping,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_9 import TurbulenceDamping9EnhancedModel

__all__ = [
    "TurbulenceDamping10EnhancedModel",
    "DistanceWeightedDamping",
    "TKEBudgetCorrectionDamping",
    "InterfaceTurbulenceFeedbackDamping",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class TurbulenceDamping10EnhancedModel(TurbulenceDamping9EnhancedModel):
    """Enhanced abstract base for v10 turbulence damping in multiphase."""
    _registry: ClassVar[dict[str, Type["TurbulenceDamping10EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[TurbulenceDamping10EnhancedModel]) -> Type[TurbulenceDamping10EnhancedModel]:
            if name in cls._registry:
                raise ValueError(f"Damping v10 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping10EnhancedModel":
        if name not in cls._registry:
            raise KeyError(f"Unknown damping v10 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@TurbulenceDamping10EnhancedModel.register("distanceWeighted")
class DistanceWeightedDamping(TurbulenceDamping10EnhancedModel):
    """Distance-weighted damping model.

    Weights the damping factor by the distance from the interface:

        D = C_base * alpha*(1-alpha) * exp(-d/d_ref)

    where d is the distance from the interface center.

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    d_ref : float
        Reference distance for exponential decay (m). Default 0.01.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        d_ref: float = 0.01,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._d_ref = max(_EPS, d_ref)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        distance = kwargs.get("distance", torch.zeros_like(alpha_c))
        d_t = distance.to(device=device, dtype=dtype).abs()
        f_dist = torch.exp(-d_t / self._d_ref).clamp(min=0.0, max=1.0)

        return self.damping_coeff * alpha_damping * f_dist * in_interface.to(dtype)


@TurbulenceDamping10EnhancedModel.register("tkeBudgetCorrection")
class TKEBudgetCorrectionDamping(TurbulenceDamping10EnhancedModel):
    """TKE budget correction damping model.

    Corrects damping to preserve the TKE budget balance,
    avoiding over-damping of production near the interface:

        D = C_base * alpha*(1-alpha) * min(1, epsilon / (P_k + epsilon))

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    C_budget : float
        Budget correction coefficient. Default 1.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_budget: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_budget = max(0.0, C_budget)

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

        budget_ratio = (eps_t / (P_k_t + eps_t)).clamp(min=0.0, max=1.0)
        f_budget = self._C_budget * budget_ratio

        return self.damping_coeff * alpha_damping * f_budget * in_interface.to(dtype)


@TurbulenceDamping10EnhancedModel.register("interfaceTurbulenceFeedback")
class InterfaceTurbulenceFeedbackDamping(TurbulenceDamping10EnhancedModel):
    """Interface-turbulence feedback damping model.

    Couples the damping to the interface curvature and
    turbulence anisotropy for mutual feedback:

        D = C_base * alpha*(1-alpha) * (1 + C_curv * |kappa|) / (1 + C_a * a_ratio)

    where kappa is interface curvature and a_ratio is the Reynolds stress anisotropy ratio.

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    C_curvature : float
        Curvature coupling coefficient. Default 0.5.
    C_anisotropy : float
        Anisotropy coupling coefficient. Default 0.3.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_curvature: float = 0.5,
        C_anisotropy: float = 0.3,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_curv = max(0.0, C_curvature)
        self._C_a = max(0.0, C_anisotropy)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        curvature = kwargs.get("curvature", torch.zeros_like(alpha_c))
        anisotropy_ratio = kwargs.get("anisotropy_ratio", torch.ones_like(alpha_c))
        kappa_t = curvature.to(device=device, dtype=dtype).abs()
        a_ratio_t = anisotropy_ratio.to(device=device, dtype=dtype).abs()

        f_curv = (1.0 + self._C_curv * kappa_t.clamp(max=5.0))
        f_a = 1.0 / (1.0 + self._C_a * a_ratio_t.clamp(max=5.0))

        return self.damping_coeff * alpha_damping * f_curv * f_a * in_interface.to(dtype)
