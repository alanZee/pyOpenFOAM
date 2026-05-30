"""
Enhanced turbulence damping models for multiphase flows — v7.

在 v6 基础上增加：

- **机器学习辅助阻尼**：基于代理模型的阻尼系数优化
- **各向异性张量阻尼**：阻尼不再是标量系数而是各向异性张量
- **相界面剪切层阻尼**：专门针对相界面剪切层的增强阻尼模型

Usage::

    from pyfoam.multiphase.turbulence_damping_enhanced_7 import (
        MLAssistedDamping,
        AnisotropicTensorDamping,
        ShearLayerDamping,
    )

    model = MLAssistedDamping(damping_coeff=10.0)
    k_damped = model.damp_k(alpha, k, delta=grid_delta)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.turbulence_damping_enhanced_6 import (
    TurbulenceDamping6EnhancedModel,
)

__all__ = [
    "TurbulenceDamping7EnhancedModel",
    "MLAssistedDamping",
    "AnisotropicTensorDamping",
    "ShearLayerDamping",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ======================================================================
# 抽象基类
# ======================================================================

class TurbulenceDamping7EnhancedModel(TurbulenceDamping6EnhancedModel):
    """Enhanced abstract base for v7 turbulence damping in multiphase.

    Extends v6 with ML-assisted, anisotropic tensor, and shear layer strategies.
    """

    _registry: ClassVar[dict[str, Type["TurbulenceDamping7EnhancedModel"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a damping model under *name*."""

        def decorator(model_cls: Type[TurbulenceDamping7EnhancedModel]) -> Type[TurbulenceDamping7EnhancedModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Damping v7 model '{name}' already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "TurbulenceDamping7EnhancedModel":
        """Factory: create a model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown damping v7 model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ======================================================================
# 机器学习辅助阻尼
# ======================================================================

@TurbulenceDamping7EnhancedModel.register("mlAssisted")
class MLAssistedDamping(TurbulenceDamping7EnhancedModel):
    """ML-assisted turbulence damping model.

    使用代理模型（surrogate）的多项式特征映射来预测最优阻尼系数：

        C_eff = w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x1*x2 + w5*x2^2

    其中 x1 = alpha*(1-alpha)，x2 = |grad(alpha)| 为界面特征。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    weights : list of float
        Polynomial feature weights (length 6). Default is heuristic.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        if weights is not None and len(weights) == 6:
            self._weights = weights
        else:
            # Heuristic weights
            self._weights = [1.0, 2.0, 0.5, -1.0, 0.3, -0.2]

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute ML-assisted damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
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

        # Feature x1: interface indicator
        x1 = 4.0 * alpha_c * (1.0 - alpha_c)

        # Feature x2: gradient magnitude
        grad_alpha = kwargs.get("grad_alpha_mag", torch.zeros_like(alpha_c))
        x2 = grad_alpha.to(device=device, dtype=dtype).abs().clamp(max=5.0)

        # Polynomial feature mapping
        w = self._weights
        C_eff = (
            w[0] + w[1] * x1 + w[2] * x2
            + w[3] * x1.pow(2) + w[4] * x1 * x2 + w[5] * x2.pow(2)
        ).clamp(min=0.1, max=5.0)

        return self.damping_coeff * C_eff * x1 * in_interface.to(dtype)


# ======================================================================
# 各向异性张量阻尼
# ======================================================================

@TurbulenceDamping7EnhancedModel.register("anisotropicTensor")
class AnisotropicTensorDamping(TurbulenceDamping7EnhancedModel):
    """Anisotropic tensor turbulence damping model.

    阻尼不再是标量系数而是各向异性张量：

        D_ij = C * alpha*(1-alpha) * (delta_ij + beta * n_i * n_j)

    其中 n 是界面法线方向，beta 是各向异性强度。

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    anisotropy_beta : float
        Anisotropy strength. Default ``0.5``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        anisotropy_beta: float = 0.5,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._beta = max(0.0, min(anisotropy_beta, 2.0))

    def compute_damping_tensor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute anisotropic damping tensor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``normal``: ``(n_cells, 3)`` interface normal.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3, 3)`` damping tensor.
        """
        device = alpha.device
        dtype = alpha.dtype
        n_cells = alpha.numel()
        alpha_c = alpha.clamp(0.0, 1.0)

        in_interface = (alpha_c > self.alpha_min) & (alpha_c < self.alpha_max)
        alpha_damping = 4.0 * alpha_c * (1.0 - alpha_c) * in_interface.to(dtype)

        # Base isotropic part
        D = torch.zeros(n_cells, 3, 3, device=device, dtype=dtype)
        for d in range(3):
            D[:, d, d] = self.damping_coeff * alpha_damping

        # Anisotropic part: beta * n_i * n_j
        normal = kwargs.get("normal", None)
        if normal is not None:
            n = normal.to(device=device, dtype=dtype)
            # Normalise
            n_mag = n.norm(dim=1, keepdim=True).clamp(min=_EPS)
            n = n / n_mag

            for i in range(3):
                for j in range(3):
                    D[:, i, j] = D[:, i, j] + (
                        self.damping_coeff * self._beta * alpha_damping
                        * n[:, i] * n[:, j]
                    )

        return D

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute scalar damping factor (trace / 3 of tensor)."""
        D = self.compute_damping_tensor(alpha, **kwargs)
        # Trace / 3
        trace = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]
        return trace / 3.0


# ======================================================================
# 相界面剪切层阻尼
# ======================================================================

@TurbulenceDamping7EnhancedModel.register("shearLayer")
class ShearLayerDamping(TurbulenceDamping7EnhancedModel):
    """Interface shear layer turbulence damping model.

    专门针对相界面剪切层的增强阻尼：

    - 剪切层位置由 |grad(alpha)| 检测
    - 阻尼强度由局部剪切率 S = |grad(U)| 调节
    - 支持剪切层厚度自适应

    Parameters
    ----------
    damping_coeff : float
        Base damping coefficient. Default ``10.0``.
    alpha_min : float
        Lower alpha threshold. Default ``0.01``.
    alpha_max : float
        Upper alpha threshold. Default ``0.99``.
    C_shear : float
        Shear rate modulation coefficient. Default ``0.3``.
    shear_threshold : float
        Minimum shear rate for enhanced damping. Default ``1.0``.
    """

    def __init__(
        self,
        damping_coeff: float = 10.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        C_shear: float = 0.3,
        shear_threshold: float = 1.0,
    ) -> None:
        super().__init__(damping_coeff, alpha_min, alpha_max)
        self._C_shear = max(0.0, C_shear)
        self._shear_threshold = max(_EPS, shear_threshold)

    def compute_damping_factor(
        self,
        alpha: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute shear-layer-aware damping factor.

        Parameters
        ----------
        alpha : torch.Tensor
            Volume fraction ``(n_cells,)``.
        **kwargs
            ``grad_alpha_mag``: ``(n_cells,)`` gradient magnitude.
            ``shear_rate``: ``(n_cells,)`` local shear rate magnitude.

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

        # Shear layer detection via gradient
        grad_alpha = kwargs.get("grad_alpha_mag", torch.zeros_like(alpha_c))
        grad_t = grad_alpha.to(device=device, dtype=dtype).abs()
        in_shear = torch.tanh(grad_t / self._shear_threshold)

        # Shear rate modulation
        shear = kwargs.get("shear_rate", None)
        if shear is not None:
            S = shear.to(device=device, dtype=dtype).abs()
            # Enhancement factor: higher shear -> stronger damping
            S_ratio = (S / self._shear_threshold).clamp(max=5.0)
            shear_factor = 1.0 + self._C_shear * S_ratio
        else:
            shear_factor = torch.ones_like(alpha_c)

        return self.damping_coeff * alpha_damping * in_shear * shear_factor * in_interface.to(dtype)
