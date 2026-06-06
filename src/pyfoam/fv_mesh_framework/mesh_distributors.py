"""
MeshDistributor — 网格分区分布框架。

对应 OpenFOAM-13 的 fvMeshDistributors/。
管理并行计算中的网格分区和重新分布。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class MeshDistributor(ABC):
    """网格分区分布器基类。

    对应 OpenFOAM-13 的 fvMeshDistributor。
    """

    @abstractmethod
    def distribute(self, n_proc: int) -> List[List[int]]:
        """将网格分区到多个处理器。

        Args:
            n_proc: 处理器数量。

        Returns:
            每个处理器的单元索引列表。
        """
        ...


class SimpleDistributor(MeshDistributor):
    """简单均匀分区分布器。

    将单元均匀分配到各处理器。
    """

    def __init__(self, mesh):
        self._mesh = mesh

    def distribute(self, n_proc: int) -> List[List[int]]:
        """均匀分区。

        Args:
            n_proc: 处理器数量。

        Returns:
            每个处理器的单元索引列表。
        """
        n_cells = self._mesh.n_cells
        cells_per_proc = n_cells // n_proc
        remainder = n_cells % n_proc

        partitions: List[List[int]] = []
        start = 0
        for i in range(n_proc):
            count = cells_per_proc + (1 if i < remainder else 0)
            partitions.append(list(range(start, start + count)))
            start += count

        return partitions


class ScotchDistributor(MeshDistributor):
    """SCOTCH 图分区分布器。

    使用图分区算法最小化通信量。
    当前为简化实现（使用均匀分区作为后备）。
    """

    def __init__(self, mesh):
        self._mesh = mesh

    def distribute(self, n_proc: int) -> List[List[int]]:
        """使用图分区（当前简化为均匀分区）。"""
        # 完整实现需要 SCOTCH 库或 Metis
        simple = SimpleDistributor(self._mesh)
        return simple.distribute(n_proc)
