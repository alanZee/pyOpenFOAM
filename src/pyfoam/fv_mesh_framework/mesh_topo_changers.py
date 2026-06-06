"""
MeshTopoChanger — 网格拓扑变更框架。

对应 OpenFOAM-13 的 fvMeshTopoChangers/。
管理网格拓扑结构的动态变化（如滑移网格、层添加/移除）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class MeshTopoChanger(ABC):
    """网格拓扑变更器基类。

    对应 OpenFOAM-13 的 fvMeshTopoChanger。
    """

    @abstractmethod
    def update_topology(self) -> bool:
        """更新网格拓扑。

        Returns:
            True 如果拓扑发生了变化。
        """
        ...


class LayerAdditionRemoval(MeshTopoChanger):
    """层添加/移除拓扑变更器。

    用于壁面适应，在边界层区域添加或移除网格层。
    """

    def __init__(self, mesh, patch_name: str, min_layers: int = 1, max_layers: int = 10):
        self._mesh = mesh
        self._patch = patch_name
        self._min_layers = min_layers
        self._max_layers = max_layers

    def update_topology(self) -> bool:
        """检查并执行层添加/移除。"""
        # 当前实现为占位
        return False

    @property
    def patch_name(self) -> str:
        return self._patch


class SlidingInterface(MeshTopoChanger):
    """滑移界面拓扑变更器。

    处理旋转机械中的滑移网格界面。
    """

    def __init__(self, mesh, interface_name: str):
        self._mesh = mesh
        self._interface = interface_name

    def update_topology(self) -> bool:
        """更新滑移界面拓扑。"""
        return False

    @property
    def interface_name(self) -> str:
        return self._interface
