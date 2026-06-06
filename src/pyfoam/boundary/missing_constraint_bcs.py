"""
缺失约束边界条件。

对应 OpenFOAM-13 的 finiteVolume/fields/fvPatchFields/constraint/。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class JumpCyclicBC:
    """跳跃周期边界条件。

    对应 OpenFOAM-13 的 jumpCyclic。
    在周期边界间施加固定跳跃值（用于压力、温度等）。
    """

    def __init__(self, jump: float = 0.0):
        self._jump = jump

    def apply(
        self,
        field_owner: torch.Tensor,
        field_neighbour: torch.Tensor,
    ) -> torch.Tensor:
        """应用跳跃。

        Args:
            field_owner: 所有者侧场值。
            field_neighbour: 邻居侧场值。

        Returns:
            跳跃后的邻居侧值。
        """
        return field_owner + self._jump

    @property
    def jump(self) -> float:
        return self._jump

    @jump.setter
    def jump(self, value: float):
        self._jump = value


class NonConformalCyclicBC:
    """非共形周期边界条件。

    对应 OpenFOAM-13 的 nonConformalCyclic。
    处理非匹配网格间的周期连接。
    """

    def __init__(self, owner_patch: str, neighbour_patch: str):
        self._owner_patch = owner_patch
        self._neighbour_patch = neighbour_patch

    @property
    def owner_patch(self) -> str:
        return self._owner_patch

    @property
    def neighbour_patch(self) -> str:
        return self._neighbour_patch


class NonConformalErrorBC:
    """非共形错误边界条件。

    对应 OpenFOAM-13 的 nonConformalError。
    当非共形界面未被正确处理时触发错误。
    """

    def __init__(self, message: str = "Non-conformal interface not handled"):
        self._message = message

    def check(self, has_ncc: bool) -> None:
        if not has_ncc:
            raise RuntimeError(self._message)


class FixedMeanBC:
    """固定均值边界条件。

    对应 OpenFOAM-13 的 fixedMean。
    调整边界值使得面积加权平均等于指定值。
    """

    def __init__(self, target_mean: float = 0.0):
        self._target = target_mean

    def correct(
        self,
        field: torch.Tensor,
        areas: torch.Tensor,
    ) -> torch.Tensor:
        """修正场使得面积加权平均等于目标值。

        Args:
            field: 边界场值 ``(n_faces,)``。
            areas: 面面积 ``(n_faces,)``。

        Returns:
            修正后的场值。
        """
        current_mean = (field * areas).sum() / areas.sum().clamp(min=1e-30)
        correction = self._target - current_mean
        return field + correction

    @property
    def target_mean(self) -> float:
        return self._target


class PartialSlipBC:
    """部分滑移边界条件。

    对应 OpenFOAM-13 的 partialSlip。
    在完全滑移和无滑移之间插值。

    U_bc = blend * U_interior + (1 - blend) * U_wall
    blend = 0 → 无滑移，blend = 1 → 完全滑移
    """

    def __init__(self, blend: float = 0.5):
        self._blend = blend

    def apply(
        self,
        U_interior: torch.Tensor,
        U_wall: torch.Tensor,
    ) -> torch.Tensor:
        """应用部分滑移。

        Args:
            U_interior: 内部单元速度 ``(n_faces, 3)``。
            U_wall: 壁面速度 ``(n_faces, 3)``。

        Returns:
            边界速度。
        """
        return self._blend * U_interior + (1 - self._blend) * U_wall

    @property
    def blend(self) -> float:
        return self._blend
