"""
TopoSet — 拓扑集合定义与源。

对应 OpenFOAM-13 的 topoSetSources/。
定义 cellSet、faceSet、pointSet 及其选择源。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE


class TopoSet:
    """拓扑集合。

    对应 OpenFOAM-13 的 topoSet/cellSet/faceSet/pointSet。
    存储一组单元/面/点的索引。
    """

    def __init__(self, name: str, element_type: str = "cell"):
        """初始化。

        Args:
            name: 集合名称。
            element_type: 元素类型 ('cell', 'face', 'point')。
        """
        self._name = name
        self._type = element_type
        self._indices: Set[int] = set()

    def add(self, index: int) -> None:
        self._indices.add(index)

    def add_range(self, start: int, end: int) -> None:
        self._indices.update(range(start, end))

    def remove(self, index: int) -> None:
        self._indices.discard(index)

    def invert(self, n_total: int) -> None:
        """反转集合。"""
        self._indices = set(range(n_total)) - self._indices

    def __contains__(self, index: int) -> bool:
        return index in self._indices

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def name(self) -> str:
        return self._name

    @property
    def element_type(self) -> str:
        return self._type

    @property
    def indices(self) -> Set[int]:
        return self._indices.copy()

    def to_tensor(self) -> torch.Tensor:
        """转换为排序后的张量。"""
        if not self._indices:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(sorted(self._indices), dtype=torch.long)


class TopoSetSource(ABC):
    """拓扑集合源基类。

    对应 OpenFOAM-13 的 topoSetSources/topoSetSource。
    """

    @abstractmethod
    def select(self, topo_set: TopoSet) -> None:
        """向集合中添加符合条件的元素。"""
        ...


class BoxToCell(TopoSetSource):
    """盒形区域选择源。

    选择位于指定盒形区域内的所有单元。
    """

    def __init__(self, min_point: Tuple[float, float, float], max_point: Tuple[float, float, float]):
        self._min = torch.tensor(min_point, dtype=CFD_DTYPE)
        self._max = torch.tensor(max_point, dtype=CFD_DTYPE)

    def select(self, topo_set: TopoSet) -> None:
        """需要 cell_centres 数据来选择。"""
        pass

    def select_from_centres(self, topo_set: TopoSet, centres: torch.Tensor) -> None:
        """从单元中心坐标选择。

        Args:
            topo_set: 目标集合。
            centres: 单元中心 ``(n_cells, 3)``。
        """
        mask = ((centres >= self._min.unsqueeze(0)) & (centres <= self._max.unsqueeze(0))).all(dim=1)
        for i in mask.nonzero(as_tuple=True)[0]:
            topo_set.add(i.item())

    @property
    def min_point(self) -> torch.Tensor:
        return self._min

    @property
    def max_point(self) -> torch.Tensor:
        return self._max


class CylinderToCell:
    """圆柱形区域选择源。"""

    def __init__(
        self,
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        radius: float,
    ):
        self._p1 = torch.tensor(point1, dtype=CFD_DTYPE)
        self._p2 = torch.tensor(point2, dtype=CFD_DTYPE)
        self._radius = radius

    def select_from_centres(self, topo_set: TopoSet, centres: torch.Tensor) -> None:
        """选择圆柱内的单元。"""
        axis = self._p2 - self._p1
        axis_len = axis.norm()
        axis_dir = axis / axis_len.clamp(min=1e-30)

        # 投影到轴线
        dp = centres - self._p1.unsqueeze(0)
        proj = (dp * axis_dir.unsqueeze(0)).sum(dim=1)
        # 径向距离
        radial = dp - proj.unsqueeze(1) * axis_dir.unsqueeze(0)
        r = radial.norm(dim=1)

        mask = (proj >= 0) & (proj <= axis_len) & (r <= self._radius)
        for i in mask.nonzero(as_tuple=True)[0]:
            topo_set.add(i.item())
