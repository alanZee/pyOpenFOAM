"""Enhanced turbulence damping models for multiphase flows — v13.

Extends v12 with:
- Interface-normal damping with directional selectivity
- Turbulence kinetic energy budget correction at interface
- Time-averaged damping coefficient adaptation

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_12 import (
        InterfaceNormalDamping,
        TKEBudgetDamping,
        TimeAveragedDamping,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Any, ClassVar, Type
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_11 import TurbulenceDamping11EnhancedModel

__all__ = [
    "TurbulenceDamping12EnhancedModel",
    "InterfaceNormalDamping",
    "TKEBudgetDamping",
    "TimeAveragedDamping",
]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class TurbulenceDamping12EnhancedModel(TurbulenceDamping11EnhancedModel):
    """Enhanced abstract base for v12 turbulence damping in multiphase."""
    _registry: ClassVar[dict[str, Type["TurbulenceDamping12EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_cls: Type[TurbulenceDamping12EnhancedModel]) -> Type[TurbulenceDamping12EnhancedModel]:
            if name in cls._registry:
                raise ValueError(f"Damping v12 model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping12EnhancedModel":
        if name not in cls._registry:
            raise KeyError(f"Unknown damping v12 model '{name}'")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


@TurbulenceDamping12EnhancedModel.register("interfaceNormal")
class InterfaceNormalDamping(TurbulenceDamping12EnhancedModel):
    """Interface-normal directional damping model.

    Applies damping preferentially in the direction normal to the
    interface, preserving tangential turbulence:

        D = C_base * alpha*(1-alpha) * (n . e_i)^2

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

        # Interface normal direction (use grad_alpha as proxy)
        grad_alpha = kwargs.get("grad_alpha", None)
        if grad_alpha is not None:
            grad_a = grad_alpha.to(device=device, dtype=dtype)
            # Normal component weighting: |grad(alpha)| / (|grad(alpha)| + epsilon)
            n_weight = (grad_a.abs() / (grad_a.abs() + 1.0)).clamp(0.0, 1.0)
        else:
            n_weight = torch.ones_like(alpha_c) * 0.5

        return self.damping_coeff * alpha_damping * n_weight * in_interface.to(dtype)


@TurbulenceDamping12EnhancedModel.register("tkeBudget")
class TKEBudgetDamping(TurbulenceDamping12EnhancedModel):
    """TKE budget correction damping model.

    Adds damping source/sink terms to TKE transport equation
    at the interface for budget closure:

        S_k = -C * alpha*(1-alpha) * k / tau_t

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    tau_t_ref : float
        Reference turbulent time scale. Default 0.01.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        tau_t_ref: float = 0.01,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._tau_t = max(1e-6, tau_t_ref)

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # TKE weighting
        k = kwargs.get("k", torch.ones_like(alpha_c))
        k_dev = k.to(device=device, dtype=dtype).clamp(min=0.0)
        tau = max(self._tau_t, _EPS)

        return self.damping_coeff * alpha_damping * (k_dev / tau) * in_interface.to(dtype)


@TurbulenceDamping12EnhancedModel.register("timeAveraged")
class TimeAveragedDamping(TurbulenceDamping12EnhancedModel):
    """Time-averaged adaptive damping model.

    Maintains a running average of the damping coefficient and
    adapts based on time history for stability:

        C_n+1 = alpha_adapt * C_n + (1 - alpha_adapt) * C_base

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default 10.0.
    alpha_min, alpha_max : float
        Alpha thresholds.
    averaging_coeff : float
        Exponential moving average coefficient. Default 0.1.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        averaging_coeff: float = 0.1,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._avg_coeff = max(0.01, min(averaging_coeff, 1.0))
        self._running_coeff = damping_coeff
        self._step_count = 0

    def compute_damping_factor(self, alpha: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        device = alpha.device
        dtype = alpha.dtype
        alpha_c = alpha.clamp(0.0, 1.0)
        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c)

        # Time-averaged coefficient
        d_current = float((self.damping_coeff * alpha_damping * in_interface.to(dtype)).mean().item())
        self._running_coeff = (1.0 - self._avg_coeff) * self._running_coeff + self._avg_coeff * d_current
        self._step_count += 1

        return self._running_coeff * alpha_damping * in_interface.to(dtype)

    @property
    def running_coefficient(self) -> float:
        return self._running_coeff
