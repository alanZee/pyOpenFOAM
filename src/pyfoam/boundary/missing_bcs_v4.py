"""
缺失边界条件补全（第四部分）。

实现剩余速度 BC（fluxCorrectedVelocity, interstitialInletVelocity）。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class FluxCorrectedVelocityBC:
    """通量校正速度边界条件。

    对应 OpenFOAM-13 的 fluxCorrectedVelocity。
    校正面通量以确保质量守恒。

    U_corrected = U * (phi_target / phi_current)
    """

    def __init__(self, phi_target: float = 0.0):
        self._phi_target = phi_target

    def correct(
        self,
        U: torch.Tensor,
        face_normal: torch.Tensor,
        face_area: torch.Tensor,
        phi_current: torch.Tensor,
    ) -> torch.Tensor:
        """校正速度以匹配目标通量。

        Args:
            U: 速度 ``(n_faces, 3)``。
            face_normal: 面法向 ``(n_faces, 3)``。
            face_area: 面面积 ``(n_faces,)``。
            phi_current: 当前通量 ``(n_faces,)``。

        Returns:
            校正后的速度。
        """
        phi_ratio = self._phi_target / phi_current.clamp(min=1e-30)
        return U * phi_ratio.unsqueeze(1)

    @property
    def target_flux(self) -> float:
        return self._phi_target


class InterstitialInletVelocityBC:
    """间隙入口速度边界条件。

    对应 OpenFOAM-13 的 interstitialInletVelocity。
    用于多相流中考虑体积分数的入口速度。

    U_interstitial = U_superficial / alpha
    """

    def __init__(self, U_superficial: tuple = (1.0, 0.0, 0.0)):
        self._U_super = torch.tensor(U_superficial, dtype=CFD_DTYPE)

    def evaluate(self, alpha: torch.Tensor) -> torch.Tensor:
        """计算间隙速度。

        Args:
            alpha: 体积分数 ``(n_faces,)``。

        Returns:
            间隙速度 ``(n_faces, 3)``。
        """
        n = alpha.shape[0]
        U = self._U_super.unsqueeze(0).expand(n, -1).clone()
        return U / alpha.clamp(min=1e-3).unsqueeze(1)

    @property
    def superficial_velocity(self) -> torch.Tensor:
        return self._U_super


class CyclicSlipBC:
    """周期滑移边界条件。

    对应 OpenFOAM-13 的 cyclicSlip。
    在周期边界间允许滑移（切向速度不连续）。
    """

    def __init__(self, owner_patch: str = "", neighbour_patch: str = ""):
        self._owner = owner_patch
        self._neighbour = neighbour_patch

    def apply(
        self,
        U_owner: torch.Tensor,
        U_neighbour: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """应用滑移周期条件。

        Args:
            U_owner: 所有者侧速度 ``(n_faces, 3)``。
            U_neighbour: 邻居侧速度 ``(n_faces, 3)``。
            normal: 面法向 ``(n_faces, 3)``。

        Returns:
            邻居侧修正后的速度。
        """
        # 法向分量匹配，切向分量保持
        U_normal = (U_owner * normal).sum(dim=1, keepdim=True) * normal
        U_tangent = U_neighbour - (U_neighbour * normal).sum(dim=1, keepdim=True) * normal
        return U_normal + U_tangent

    @property
    def owner_patch(self) -> str:
        return self._owner

    @property
    def neighbour_patch(self) -> str:
        return self._neighbour
