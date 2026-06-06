"""
kOmegaSSTSato — 带气泡诱导湍流的 k-ω SST 模型。

对应 OpenFOAM-13 的 MomentumTransportModels/RAS/kOmegaSST/kOmegaSSTSAS.H
（Sato 气泡诱导湍流模型变体）。

在标准 k-ω SST 基础上添加气泡诱导粘度：
  nut_bubble = C_mu * d_b * |U_slip|
  nut_total = nut_sst + nut_bubble
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel as KOmegaSST


class KOmegaSSTSato(KOmegaSST):
    """带 Sato 气泡诱导湍流的 k-ω SST 模型。

    用于气液两相流中气泡对湍流的影响。
    """

    # Sato 常数
    C_mu_bubble = 0.6

    def __init__(
        self,
        mesh,
        bubble_diameter: float = 1e-3,
        alpha_max: float = 0.3,
    ):
        """初始化。

        Args:
            mesh: FvMesh 实例。
            bubble_diameter: 气泡直径 (m)。
            alpha_max: 最大体积分数。
        """
        super().__init__(mesh)
        self._d_b = bubble_diameter
        self._alpha_max = alpha_max

    def bubble_viscosity(
        self,
        alpha: torch.Tensor,
        slip_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """计算气泡诱导粘度。

        nut_bubble = C_mu * alpha * d_b * |U_slip|

        Args:
            alpha: 气相体积分数 ``(n_cells,)``。
            slip_velocity: 滑移速度 ``(n_cells, 3)``。

        Returns:
            气泡诱导粘度 ``(n_cells,)``。
        """
        U_slip_mag = slip_velocity.norm(dim=1)
        return self.C_mu_bubble * alpha.clamp(max=self._alpha_max) * self._d_b * U_slip_mag

    @property
    def bubble_diameter(self) -> float:
        """气泡直径。"""
        return self._d_b
