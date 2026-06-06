"""
buoyantKEpsilon — 浮力 k-ε 湍流模型。

对应 OpenFOAM-13 的 MomentumTransportModels/RAS/kEpsilon/kEpsilon.H
（带 buoyancy production 项）。

在标准 k-ε 模型基础上添加浮力源项：
  P_b = -beta * g * (nut/SigmaT) * grad(T)
  k 方程：额外 + P_b
  epsilon 方程：额外 + C3e * epsilon/k * P_b
"""
from __future__ import annotations

from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.turbulence.k_epsilon import KEpsilonModel


class BuoyantKEpsilon(KEpsilonModel):
    """浮力 k-ε 模型。

    在标准 k-ε 基础上添加浮力源项，用于自然对流和混合对流模拟。
    """

    # 模型常数
    C3e = 1.44  # 浮力 epsilon 系数
    SigmaT = 0.85  # 热扩散 Prandtl 数

    def __init__(self, mesh, g: Optional[torch.Tensor] = None, beta: float = 3.43e-3):
        """初始化。

        Args:
            mesh: FvMesh 实例。
            g: 重力加速度向量 ``(3,)``。默认 (0, -9.81, 0)。
            beta: 热膨胀系数 (1/K)。
        """
        super().__init__(mesh)
        if g is None:
            g = torch.tensor([0, -9.81, 0], dtype=CFD_DTYPE)
        self._g = g
        self._beta = beta

    def compute_buoyancy_production(
        self,
        grad_T: torch.Tensor,
        nut: torch.Tensor,
    ) -> torch.Tensor:
        """计算浮力产生项 P_b = -beta * g_i * (nut/SigmaT) * dT/dx_i。

        Args:
            grad_T: 温度梯度 ``(n_cells, 3)``。
            nut: 湍流粘度 ``(n_cells,)``。

        Returns:
            浮力产生项 ``(n_cells,)``。
        """
        # g · grad(T)
        g_dot_gradT = (self._g.unsqueeze(0) * grad_T).sum(dim=1)
        return -self._beta * g_dot_gradT * nut / self.SigmaT

    @property
    def gravity(self) -> torch.Tensor:
        """重力加速度向量。"""
        return self._g

    @property
    def beta(self) -> float:
        """热膨胀系数。"""
        return self._beta
