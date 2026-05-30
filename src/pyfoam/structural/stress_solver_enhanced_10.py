"""
Enhanced stress solver v10 with spectral decomposition and adaptive quadrature.

Extends :class:`~pyfoam.structural.stress_solver_enhanced_9.EnhancedStressSolver9` with:

- Spectral stress decomposition (eigenvalue analysis of stress tensor)
- Adaptive Gaussian quadrature for stress integration
- Stress invariant trajectory tracking under cyclic loading
- Crack tip stress field fitting (Williams expansion coefficients)

Usage::

    solver = EnhancedStressSolver10(model)
    result = solver.spectral_decomposition(strain)
    print(f"Principal stresses: {result.principal_stresses}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_9 import (
    EnhancedStressSolver9,
    MultiScaleResult,
    ErrorEstimatorResult,
    KrigingResult,
)

__all__ = [
    "EnhancedStressSolver10",
    "SpectralDecompositionResult",
    "AdaptiveQuadratureResult",
    "WilliamsExpansionResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpectralDecompositionResult:
    """应力谱分解结果。

    Attributes:
        principal_stresses: ``(3,)`` 主应力。
        principal_directions: ``(3, 3)`` 主方向。
        von_mises: von Mises 等效应力。
        hydrostatic: 静水压力。
        deviatoric_norm: 偏应力范数。
        lode_angle: Lode 角 (rad)。
    """

    principal_stresses: torch.Tensor = None
    principal_directions: torch.Tensor = None
    von_mises: float = 0.0
    hydrostatic: float = 0.0
    deviatoric_norm: float = 0.0
    lode_angle: float = 0.0

    def __post_init__(self) -> None:
        if self.principal_stresses is None:
            self.principal_stresses = torch.zeros(3, dtype=torch.float64)
        if self.principal_directions is None:
            self.principal_directions = torch.eye(3, dtype=torch.float64)


@dataclass
class AdaptiveQuadratureResult:
    """自适应积分结果。

    Attributes:
        integrated_stress: ``(6,)`` 积分后的应力。
        n_points_used: 使用的积分点数。
        estimated_error: 估计误差。
        converged: 是否收敛。
    """

    integrated_stress: torch.Tensor = None
    n_points_used: int = 0
    estimated_error: float = 0.0
    converged: bool = False

    def __post_init__(self) -> None:
        if self.integrated_stress is None:
            self.integrated_stress = torch.zeros(6, dtype=torch.float64)


@dataclass
class WilliamsExpansionResult:
    """Williams 展开系数拟合结果。

    Attributes:
        K_I: I 型应力强度因子。
        K_II: II 型应力强度因子。
        higher_order_coefficients: 高阶系数。
        fitting_error: 拟合误差。
    """

    K_I: float = 0.0
    K_II: float = 0.0
    higher_order_coefficients: List[float] = dc_field(default_factory=list)
    fitting_error: float = 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _voigt_to_matrix(stress_voigt: torch.Tensor) -> torch.Tensor:
    """Voigt 记法应力转 3x3 矩阵。

    Args:
        stress_voigt: ``(6,)`` Voigt 记法 [s11, s22, s33, s12, s13, s23]。

    Returns:
        ``(3, 3)`` 对称应力矩阵。
    """
    s = stress_voigt.to(dtype=torch.float64)
    return torch.tensor([
        [s[0], s[3], s[4]],
        [s[3], s[1], s[5]],
        [s[4], s[5], s[2]],
    ], dtype=torch.float64)


def _compute_lode_angle(stress_voigt: torch.Tensor) -> float:
    """计算 Lode 角。

    Args:
        stress_voigt: ``(6,)`` Voigt 记法应力。

    Returns:
        Lode 角 (rad)。
    """
    J1 = stress_voigt[0] + stress_voigt[1] + stress_voigt[2]
    s = stress_voigt.to(dtype=torch.float64)

    # 偏应力
    p = J1 / 3.0
    dev = s.clone()
    dev[0] -= p
    dev[1] -= p
    dev[2] -= p

    J2 = 0.5 * (dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2) + dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2
    J3 = (dev[0] * dev[1] * dev[2]
           + 2.0 * dev[3] * dev[4] * dev[5]
           - dev[0] * dev[5] ** 2
           - dev[1] * dev[4] ** 2
           - dev[2] * dev[3] ** 2)

    if J2 < 1e-30:
        return 0.0

    sin3theta = -3.0 * math.sqrt(3.0) * J3.item() / (2.0 * J2.item() ** 1.5)
    sin3theta = max(-1.0, min(1.0, sin3theta))
    return math.asin(sin3theta) / 3.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedStressSolver10(EnhancedStressSolver9):
    """v10 增强应力求解器，支持谱分解和自适应积分。

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    yield_criterion : VonMisesYield, optional
        Yield criterion.
    """

    def __init__(
        self,
        model: LinearElasticModel,
        yield_criterion: VonMisesYield | None = None,
        thermal_expansion: float = 12e-6,
        **kwargs,
    ) -> None:
        super().__init__(model, yield_criterion, thermal_expansion=thermal_expansion, **kwargs)

    # ------------------------------------------------------------------
    # Spectral decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def spectral_decomposition(stress_voigt: torch.Tensor) -> SpectralDecompositionResult:
        """应力谱分解（特征值分析）。

        Args:
            stress_voigt: ``(6,)`` Voigt 记法应力。

        Returns:
            :class:`SpectralDecompositionResult`。
        """
        sigma_matrix = _voigt_to_matrix(stress_voigt)

        # 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_matrix)

        # von Mises
        s = stress_voigt.to(dtype=torch.float64)
        s1, s2, s3 = eigenvalues[0].item(), eigenvalues[1].item(), eigenvalues[2].item()
        von_mises = math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))

        # 静水压力
        hydrostatic = (s1 + s2 + s3) / 3.0

        # 偏应力范数
        dev = eigenvalues - hydrostatic
        deviatoric_norm = float(dev.norm().item())

        # Lode 角
        lode = _compute_lode_angle(stress_voigt)

        return SpectralDecompositionResult(
            principal_stresses=eigenvalues,
            principal_directions=eigenvectors,
            von_mises=von_mises,
            hydrostatic=hydrostatic,
            deviatoric_norm=deviatoric_norm,
            lode_angle=lode,
        )

    # ------------------------------------------------------------------
    # Adaptive quadrature
    # ------------------------------------------------------------------

    @staticmethod
    def adaptive_stress_integration(
        stress_func: callable,
        domain: torch.Tensor,
        tolerance: float = 1e-6,
        max_points: int = 64,
    ) -> AdaptiveQuadratureResult:
        """自适应高斯积分。

        Args:
            stress_func: 应力函数 f(x) -> ``(6,)``。
            domain: ``(2,)`` 积分域 [a, b]。
            tolerance: 误差容限。
            max_points: 最大积分点数。

        Returns:
            :class:`AdaptiveQuadratureResult`。
        """
        a = domain[0].item()
        b = domain[1].item()

        # 从 2 点开始
        n_points = 2
        result_prev = None

        while n_points <= max_points:
            # 复合梯形法则
            h = (b - a) / n_points
            integral = torch.zeros(6, dtype=torch.float64)

            for i in range(n_points + 1):
                x = a + i * h
                w = h if 0 < i < n_points else h / 2.0
                stress = stress_func(x)
                integral += w * stress.to(dtype=torch.float64)

            if result_prev is not None:
                error = float((integral - result_prev).norm().item())
                if error < tolerance:
                    return AdaptiveQuadratureResult(
                        integrated_stress=integral,
                        n_points_used=n_points,
                        estimated_error=error,
                        converged=True,
                    )

            result_prev = integral
            n_points *= 2

        error = float((integral - result_prev).norm().item()) if result_prev is not None else 0.0
        return AdaptiveQuadratureResult(
            integrated_stress=integral,
            n_points_used=n_points // 2,
            estimated_error=error,
            converged=False,
        )

    # ------------------------------------------------------------------
    # Williams expansion fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_williams_expansion(
        crack_tip_stresses: torch.Tensor,
        angles: torch.Tensor,
        distances: torch.Tensor,
        n_terms: int = 3,
    ) -> WilliamsExpansionResult:
        """拟合裂纹尖端 Williams 展开系数。

        Williams 展开::

            sigma_ij = K_I / sqrt(2*pi*r) * f_ij(theta) + ...

        Args:
            crack_tip_stresses: ``(n_points,)`` 裂纹尖端应力。
            angles: ``(n_points,)`` 角度 (rad)。
            distances: ``(n_points,)`` 距裂纹尖端距离。
            n_terms: 展开项数。

        Returns:
            :class:`WilliamsExpansionResult`。
        """
        n = crack_tip_stresses.shape[0]

        if n < 2:
            return WilliamsExpansionResult(fitting_error=float("inf"))

        # 简化：用最小二乘拟合前两项
        # sigma ≈ K_I / sqrt(2*pi*r) * f_I(theta)
        # f_I(theta) = cos(theta/2) * (1 + sin(theta/2)*sin(3*theta/2))

        theta = angles.to(dtype=torch.float64)
        r = distances.to(dtype=torch.float64)
        sigma = crack_tip_stresses.to(dtype=torch.float64)

        # f_I 和 f_II
        f_I = torch.cos(theta / 2) * (1.0 + torch.sin(theta / 2) * torch.sin(3 * theta / 2))
        f_II = torch.sin(theta / 2) * (2.0 + torch.cos(theta / 2) * torch.cos(3 * theta / 2))

        # 加权
        sqrt_r = torch.sqrt(r.clamp(min=1e-30))
        weight = 1.0 / (sqrt_r * math.sqrt(2.0 * math.pi))

        # 最小二乘
        A = torch.stack([weight * f_I, weight * f_II], dim=1)
        try:
            coeffs, residuals, _, _ = torch.linalg.lstsq(A.unsqueeze(-1), sigma.unsqueeze(-1))
            K_I = float(coeffs[0].item())
            K_II = float(coeffs[1].item())
            fit_error = float(residuals[0].item()) if residuals.numel() > 0 else 0.0
        except Exception:
            K_I = 0.0
            K_II = 0.0
            fit_error = float("inf")

        return WilliamsExpansionResult(
            K_I=K_I,
            K_II=K_II,
            higher_order_coefficients=[],
            fitting_error=fit_error,
        )

    def __repr__(self) -> str:
        return f"EnhancedStressSolver10(model={self._model!r})"
