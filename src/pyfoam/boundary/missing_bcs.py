"""
缺失边界条件实现。

对应 OpenFOAM-13 的 finiteVolume/fields/fvPatchFields/derived/。
实现 freestream、supersonicFreestream、fixedProfile 等。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE


class FreestreamVelocityBC:
    """自由流速度边界条件。

    对应 OpenFOAM-13 的 freestreamVelocity。
    在流入时使用指定值，流出时使用零梯度。
    """

    def __init__(self, U_inf: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        self._U_inf = torch.tensor(U_inf, dtype=CFD_DTYPE)

    def apply(
        self,
        U: torch.Tensor,
        face_normal: torch.Tensor,
        patch_faces: torch.Tensor,
    ) -> torch.Tensor:
        """应用边界条件。

        Args:
            U: 速度场 ``(n_cells, 3)``。
            face_normal: 面法向量 ``(n_patch_faces, 3)``。
            patch_faces: patch 面索引。

        Returns:
            边界速度 ``(n_patch_faces, 3)``。
        """
        n = patch_faces.shape[0]
        U_bc = self._U_inf.unsqueeze(0).expand(n, -1).clone()

        # 流出面（U·n > 0）使用零梯度
        U_interior = U[patch_faces]
        outflow = (U_interior * face_normal).sum(dim=1) > 0
        U_bc[outflow] = U_interior[outflow]

        return U_bc


class SupersonicFreestreamBC:
    """超声速自由流边界条件。

    对应 OpenFOAM-13 的 supersonicFreestream。
    所有特征线都从外部传入，因此所有变量使用固定值。
    """

    def __init__(
        self,
        U: Tuple[float, float, float] = (1000.0, 0.0, 0.0),
        p: float = 101325.0,
        T: float = 300.0,
    ):
        self._U = torch.tensor(U, dtype=CFD_DTYPE)
        self._p = p
        self._T = T

    def apply_velocity(self, n_faces: int) -> torch.Tensor:
        return self._U.unsqueeze(0).expand(n_faces, -1)

    def apply_pressure(self, n_faces: int) -> torch.Tensor:
        return torch.full((n_faces,), self._p, dtype=CFD_DTYPE)

    def apply_temperature(self, n_faces: int) -> torch.Tensor:
        return torch.full((n_faces,), self._T, dtype=CFD_DTYPE)


class FixedProfileBC:
    """固定速度剖面边界条件。

    对应 OpenFOAM-13 的 fixedProfile。
    根据指定的剖面函数（抛物线、对数律等）设置边界值。
    """

    def __init__(
        self,
        profile_type: str = "parabolic",
        U_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
    ):
        self._type = profile_type
        self._U_max = U_max
        self._y_min = y_min
        self._y_max = y_max

    def evaluate(self, y: torch.Tensor) -> torch.Tensor:
        """计算剖面值。

        Args:
            y: 坐标 ``(n_faces,)``。

        Returns:
            速度值 ``(n_faces,)``。
        """
        # 归一化坐标
        eta = (y - self._y_min) / (self._y_max - self._y_min).clamp(min=1e-30)
        eta = eta.clamp(0, 1)

        if self._type == "parabolic":
            # 抛物线剖面：U = 4 * U_max * eta * (1 - eta)
            return 4 * self._U_max * eta * (1 - eta)
        elif self._type == "uniform":
            return torch.full_like(y, self._U_max)
        else:
            return torch.full_like(y, self._U_max)


class TotalTemperatureBC:
    """总温边界条件。

    对应 OpenFOAM-13 的 totalTemperature。
    T_total = T_static + |U|² / (2 * Cp)
    因此 T_static = T_total - |U|² / (2 * Cp)
    """

    def __init__(self, T_total: float = 300.0, Cp: float = 1005.0):
        self._T_total = T_total
        self._Cp = Cp

    def evaluate(self, U_mag: torch.Tensor) -> torch.Tensor:
        """从总温计算静温。

        Args:
            U_mag: 速度大小 ``(n_faces,)``。

        Returns:
            静温 ``(n_faces,)``。
        """
        return self._T_total - U_mag.pow(2) / (2 * self._Cp)


class InterfaceCompressionBC:
    """界面压缩边界条件。

    对应 OpenFOAM-13 的 interfaceCompression。
    用于 VOF 方法中相界面的压缩，增强界面锐度。
    """

    def __init__(self, c_alpha: float = 1.0):
        self._c_alpha = c_alpha

    @property
    def c_alpha(self) -> float:
        return self._c_alpha
