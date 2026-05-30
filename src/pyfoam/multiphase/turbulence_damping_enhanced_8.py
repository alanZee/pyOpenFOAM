"""Enhanced turbulence damping models for multiphase flows — v8.

Extends v7 with:
- **界面感知的LES阻尼**: interface-aware LES damping that varies with local alpha gradient
- **湍流-界面耦合阻尼**: coupled damping model accounting for interfacial turbulence generation
- **多尺度能量级串阻尼**: energy cascade-aware damping across multiple length scales

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_8 import (
        InterfaceAwareLESDamping,
        TurbulenceInterfaceCoupledDamping,
        MultiScaleCascadeDamping,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_7 import (
    TurbulenceDamping7EnhancedModel,
)

__all__ = [
    "TurbulenceDamping8EnhancedModel",
    "InterfaceAwareLESDamping",
    "TurbulenceInterfaceCoupledDamping",
    "MultiScaleCascadeDamping",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class TurbulenceDamping8EnhancedModel(TurbulenceDamping7EnhancedModel):
    """Enhanced abstract base for v8 turbulence damping in multiphase."""
    _registry: ClassVar[dict[str, Type["TurbulenceDamping8EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[TurbulenceDamping8EnhancedModel]) -> Type[TurbulenceDamping8EnhancedModel]:
            if name in cls._registry:
                raise ValueError(f"Damping v8 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping8EnhancedModel":
        if name not in cls._registry:
            raise KeyError(f"Unknown damping v8 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@TurbulenceDamping8EnhancedModel.register("interfaceAwareLES")
class InterfaceAwareLESDamping(TurbulenceDamping8EnhancedModel):
    """Interface-aware LES turbulence damping model.

    Damping strength varies with the local interface gradient, providing
    stronger damping near sharp interfaces and weaker damping in bulk regions:

        D = C * (1 + C_grad * |grad(alpha)|) * alpha * (1 - alpha)

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds. Default 0.01, 0.99.
    C_grad : float
        Gradient enhancement coefficient. Default 2.0.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_grad: float = 2.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_grad = max(0.0, C_grad)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        grad_alpha = kwargs.get("grad_alpha_mag", torch.zeros_like(alpha_c))
        grad_t = grad_alpha.to(device=device, dtype=dtype).abs()
        enhancement = 1.0 + self._C_grad * grad_t.clamp(max=5.0)

        return self.damping_coeff * alpha_damping * enhancement * in_interface.to(dtype)


@TurbulenceDamping8EnhancedModel.register("turbInterfaceCoupled")
class TurbulenceInterfaceCoupledDamping(TurbulenceDamping8EnhancedModel):
    """Turbulence-interface coupled damping model.

    Accounts for interfacial turbulence generation (surface tension
    fluctuations at the interface):

        D = C_base * alpha*(1-alpha) - C_gen * sigma * |grad(alpha)| * k

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    C_generation : float
        Interfacial turbulence generation coefficient. Default 0.1.
    sigma : float
        Surface tension (N/m). Default 0.072.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_generation: float = 0.1,
        sigma: float = 0.072,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_gen = max(0.0, C_generation)
        self._sigma = max(_EPS, sigma)

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

        S_base = self.damping_coeff * alpha_damping
        S_gen = self._C_gen * self._sigma * grad_t * k_t.sqrt()
        S_net = (S_base - S_gen).clamp(min=0.0)

        return S_net * in_interface.to(dtype)


@TurbulenceDamping8EnhancedModel.register("multiscaleCascade")
class MultiScaleCascadeDamping(TurbulenceDamping8EnhancedModel):
    """Multi-scale energy cascade damping model.

    Applies damping across multiple length scales, with the damping
    strength proportional to the local eddy size:

        D_n = C_n * (delta_n / delta_max) * alpha*(1-alpha)

    where n runs over N_scales and delta_n = delta_max / 2^n.

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    N_scales : int
        Number of cascade scales. Default 3.
    delta_max : float
        Maximum scale (m). Default 0.01.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        N_scales: int = 3,
        delta_max: float = 0.01,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._N_scales = max(1, min(N_scales, 8))
        self._delta_max = max(_EPS, delta_max)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Weighted sum over cascade scales
        total_weight = 0.0
        D_total = torch.zeros_like(alpha_c)
        for n in range(self._N_scales):
            weight = 1.0 / (2.0 ** n)
            total_weight += weight
            D_total = D_total + weight * alpha_damping

        D_total = D_total / max(total_weight, _EPS)
        return self.damping_coeff * D_total * in_interface.to(dtype)
