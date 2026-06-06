"""
MeshMovers — 网格运动框架。

对应 OpenFOAM-13 的 fvMeshMovers/。
管理网格点位移和变形。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class MeshMover(ABC):
    """网格运动基类。

    对应 OpenFOAM-13 的 fvMeshMover。
    """

    @abstractmethod
    def move_points(self, displacement: torch.Tensor) -> None:
        """施加点位移。

        Args:
            displacement: 位移向量 ``(n_points, 3)``。
        """
        ...

    @abstractmethod
    def update_motion(self, dt: float) -> None:
        """推进运动一个时间步。

        Args:
            dt: 时间步长。
        """
        ...


class DeformingMeshMover(MeshMover):
    """变形网格运动器。

    根据指定的点位移场移动网格点，并重新计算几何量。
    """

    def __init__(self, mesh):
        """初始化。

        Args:
            mesh: FvMesh 实例。
        """
        self._mesh = mesh
        self._displacement = torch.zeros(mesh.n_points, 3, dtype=CFD_DTYPE)

    def move_points(self, displacement: torch.Tensor) -> None:
        """施加点位移并更新网格。"""
        self._displacement = displacement.to(dtype=CFD_DTYPE)
        self._mesh._points = self._mesh._points + self._displacement
        self._mesh.compute_geometry()

    def update_motion(self, dt: float) -> None:
        """基于时间步更新运动（默认无操作）。"""
        pass

    @property
    def displacement(self) -> torch.Tensor:
        """当前位移场。"""
        return self._displacement
