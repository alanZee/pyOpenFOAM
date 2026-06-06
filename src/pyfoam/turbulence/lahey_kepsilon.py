"""
LaheyKEpsilon — Lahey k-ε 湍流模型。

对应 OpenFOAM-13 的 MomentumTransportModels/RAS/kEpsilon/LaheyKEpsilon.H。
用于沸腾两相流中气泡诱导湍流的修正 k-ε 模型。

在标准 k-ε 基础上添加：
1. 气泡产生项：P_b = C_b * alpha * |g| * d_b
2. 湍流粘度修正：nut = C_mu * k²/epsilon * f(alpha)
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.turbulence.k_epsilon import KEpsilonModel


class LaheyKEpsilon(KEpsilonModel):
    """Lahey k-ε 沸腾两相流湍流模型。

    在标准 k-ε 基础上添加气泡诱导湍流效应。
    """

    # 气泡诱导湍流常数
    C_b = 0.25  # 气泡产生系数
    C_mu = 0.09  # 标准 C_mu

    def __init__(
        self,
        mesh,
        bubble_diameter: float = 1e-3,
        g_mag: float = 9.81,
        alpha_max: float = 0.3,
    ):
        """初始化。

        Args:
            mesh: FvMesh 实例。
            bubble_diameter: 气泡直径 (m)。
            g_mag: 重力加速度大小 (m/s²)。
            alpha_max: 最大体积分数。
        """
        super().__init__(mesh)
        self._d_b = bubble_diameter
        self._g_mag = g_mag
        self._alpha_max = alpha_max

    def bubble_production(self, alpha: torch.Tensor) -> torch.Tensor:
        """气泡诱导湍流产生项。

        P_b = C_b * alpha * g * d_b

        Args:
            alpha: 气相体积分数 ``(n_cells,)``。

        Returns:
            气泡产生项 ``(n_cells,)``。
        """
        return self.C_b * alpha.clamp(max=self._alpha_max) * self._g_mag * self._d_b

    def turbulent_viscosity_correction(
        self,
        k: torch.Tensor,
        epsilon: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """修正的湍流粘度。

        nut = C_mu * k²/epsilon * (1 + C_alpha * alpha)

        Args:
            k: 湍流动能 ``(n_cells,)``。
            epsilon: 耗散率 ``(n_cells,)``。
            alpha: 气相体积分数 ``(n_cells,)``。

        Returns:
            湍流粘度 ``(n_cells,)``。
        """
        nut_base = self.C_mu * k.pow(2) / epsilon.clamp(min=1e-30)
        return nut_base * (1 + 2.5 * alpha.clamp(max=self._alpha_max))

    @property
    def bubble_diameter(self) -> float:
        return self._d_b
