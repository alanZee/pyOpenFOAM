"""
PolyTopoChange — 网格拓扑修改操作。

对应 OpenFOAM-13 的 polyTopoChange/polyTopoChange/polyTopoChange.H。
管理网格拓扑的批量修改（添加/删除面、单元、点等）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TopoChangeType(Enum):
    """拓扑变更类型。"""
    ADD_CELL = auto()
    REMOVE_CELL = auto()
    ADD_FACE = auto()
    REMOVE_FACE = auto()
    ADD_POINT = auto()
    REMOVE_POINT = auto()
    MODIFY_FACE = auto()


@dataclass
class TopoAction:
    """单个拓扑变更操作。"""
    action_type: TopoChangeType
    index: int = -1
    data: dict = field(default_factory=dict)


class PolyTopoChange:
    """网格拓扑修改管理器。

    对应 OpenFOAM-13 的 polyTopoChange。
    收集并批量执行网格拓扑变更操作。

    Examples:
        >>> topo = PolyTopoChange(n_cells=100, n_faces=200, n_points=300)
        >>> topo.add_cell(points=[0, 1, 2, 3, 4, 5, 6, 7])
        >>> topo.remove_cell(50)
        >>> changes = topo.get_changes()
    """

    def __init__(self, n_cells: int = 0, n_faces: int = 0, n_points: int = 0):
        self._n_cells = n_cells
        self._n_faces = n_faces
        self._n_points = n_points
        self._actions: List[TopoAction] = []

    def add_cell(self, points: List[int], owner_face: int = -1) -> int:
        """添加单元。

        Args:
            points: 单元顶点索引列表。
            owner_face: 所属面索引。

        Returns:
            新单元索引。
        """
        idx = self._n_cells
        self._actions.append(TopoAction(
            TopoChangeType.ADD_CELL,
            index=idx,
            data={"points": points, "owner_face": owner_face},
        ))
        self._n_cells += 1
        return idx

    def remove_cell(self, cell_id: int) -> None:
        """标记单元待删除。"""
        self._actions.append(TopoAction(
            TopoChangeType.REMOVE_CELL,
            index=cell_id,
        ))

    def add_face(
        self,
        vertices: List[int],
        owner: int,
        neighbour: int = -1,
    ) -> int:
        """添加面。

        Args:
            vertices: 面顶点索引。
            owner: 所有者单元。
            neighbour: 邻居单元（-1 表示边界）。

        Returns:
            新面索引。
        """
        idx = self._n_faces
        self._actions.append(TopoAction(
            TopoChangeType.ADD_FACE,
            index=idx,
            data={"vertices": vertices, "owner": owner, "neighbour": neighbour},
        ))
        self._n_faces += 1
        return idx

    def remove_face(self, face_id: int) -> None:
        """标记面待删除。"""
        self._actions.append(TopoAction(
            TopoChangeType.REMOVE_FACE,
            index=face_id,
        ))

    def modify_face(
        self,
        face_id: int,
        new_vertices: Optional[List[int]] = None,
        new_owner: int = -1,
        new_neighbour: int = -1,
    ) -> None:
        """修改现有面。"""
        self._actions.append(TopoAction(
            TopoChangeType.MODIFY_FACE,
            index=face_id,
            data={
                "vertices": new_vertices,
                "owner": new_owner,
                "neighbour": new_neighbour,
            },
        ))

    def get_changes(self) -> List[TopoAction]:
        """获取所有待执行的变更。"""
        return self._actions.copy()

    def clear(self) -> None:
        """清除所有待执行的变更。"""
        self._actions.clear()

    @property
    def n_pending(self) -> int:
        """待执行变更数。"""
        return len(self._actions)

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def n_faces(self) -> int:
        return self._n_faces

    @property
    def n_points(self) -> int:
        return self._n_points
