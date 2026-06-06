"""
SpecieTransferModel — 组分传输模型框架。

对应 OpenFOAM-13 的 specieTransfer/。
处理多相流中相间组分传输（蒸发、溶解、化学反应等）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class SpecieTransferModel(ABC):
    """组分传输模型基类。

    对应 OpenFOAM-13 的 specieTransfer/interfaceComposition/interfaceCompositionModel。
    """

    @abstractmethod
    def mass_flux(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """计算组分传输质量通量。

        Args:
            Y: 组分质量分数 ``(n_cells,)``。
            T: 温度 ``(n_cells,)``。
            alpha: 相体积分数 ``(n_cells,)``。

        Returns:
            质量通量 ``(n_cells,)``。
        """
        ...

    @abstractmethod
    def heat_flux(
        self,
        T: torch.Tensor,
        mass_flux: torch.Tensor,
    ) -> torch.Tensor:
        """计算相变潜热通量。

        Args:
            T: 温度 ``(n_cells,)``。
            mass_flux: 质量通量 ``(n_cells,)``。

        Returns:
            热通量 ``(n_cells,)``。
        """
        ...


class SimpleDiffusionModel(SpecieTransferModel):
    """简单扩散组分传输模型。

    使用 Fick 扩散定律：
      J = -D_eff * grad(Y)
      D_eff = alpha * D_mass + (1 - alpha) * D_turb
    """

    def __init__(
        self,
        D_mass: float = 1e-5,
        D_turb: float = 1e-3,
        latent_heat: float = 2.26e6,
    ):
        """初始化。

        Args:
            D_mass: 分子扩散系数 (m²/s)。
            D_turb: 湍流扩散系数 (m²/s)。
            latent_heat: 相变潜热 (J/kg)。
        """
        self._D_mass = D_mass
        self._D_turb = D_turb
        self._L = latent_heat

    def mass_flux(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Fick 扩散质量通量（简化：使用梯度大小估计）。"""
        D_eff = alpha * self._D_mass + (1 - alpha) * self._D_turb
        # 简化：使用 Y 与平衡值的差
        Y_eq = self._equilibrium(T)
        return D_eff * (Y_eq - Y)

    def heat_flux(
        self,
        T: torch.Tensor,
        mass_flux: torch.Tensor,
    ) -> torch.Tensor:
        """潜热通量 = L * dm/dt。"""
        return self._L * mass_flux

    def _equilibrium(self, T: torch.Tensor) -> torch.Tensor:
        """简化平衡浓度（Clausius-Clapeyron 近似）。"""
        T_ref = 373.15
        return 0.1 * torch.exp(-self._L / 461.0 * (1 / T - 1 / T_ref))

    @property
    def molecular_diffusivity(self) -> float:
        return self._D_mass

    @property
    def latent_heat(self) -> float:
        return self._L
