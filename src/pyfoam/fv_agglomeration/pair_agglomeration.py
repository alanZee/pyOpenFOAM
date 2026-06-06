"""
PairGamgAgglomeration — 配对 GAMG 网格粗化。

对应 OpenFOAM-13 的 fvAgglomeration/methods/pairGAMGAgglomeration/。
实现 GAMG 多重网格求解器的网格粗化策略。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class PairGamgAgglomeration:
    """配对 GAMG 网格粗化。

    通过配对面连接的相邻单元来构建粗化级别。
    每个细级别单元对映射到一个粗级别单元。

    Examples:
        >>> agg = PairGamgAgglomeration(n_cells=100, owner=owner, neighbour=neighbour)
        >>> levels = agg.build_levels(n_levels=4)
        >>> len(levels)  # 粗化级别数
        4
    """

    def __init__(
        self,
        n_cells: int,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
    ):
        """初始化。

        Args:
            n_cells: 单元数。
            owner: 面的所有者单元索引。
            neighbour: 面的邻居单元索引（仅内部面）。
        """
        self._n_cells = n_cells
        self._owner = owner
        self._neighbour = neighbour
        self._levels: List[dict] = []

    def build_levels(self, n_levels: int = 5) -> List[dict]:
        """构建粗化级别。

        Args:
            n_levels: 最大粗化级别数。

        Returns:
            每个级别的信息字典列表。
        """
        levels = []
        n_coarse = self._n_cells
        cell_map = torch.arange(n_coarse, dtype=INDEX_DTYPE)

        for level in range(n_levels):
            if n_coarse <= 4:
                break

            # 配对粗化：为每个未配对的单元找一个邻居配对
            new_map = self._pair_cells(n_coarse, cell_map)
            if new_map is None:
                break

            # 计算粗化后的单元数
            unique = torch.unique(new_map)
            n_coarse_new = len(unique)

            # 如果粗化效果不好（<10% 减少），停止
            if n_coarse_new > n_coarse * 0.9:
                break

            levels.append({
                "level": level,
                "n_cells_fine": n_coarse,
                "n_cells_coarse": n_coarse_new,
                "cell_map": new_map,
            })

            cell_map = new_map
            n_coarse = n_coarse_new

        self._levels = levels
        return levels

    def _pair_cells(
        self,
        n_cells: int,
        cell_map: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """执行配对粗化。

        Args:
            n_cells: 当前单元数。
            cell_map: 当前到最细级别的单元映射。

        Returns:
            粗化后的单元映射，或 None 如果无法粗化。
        """
        # 构建邻居关系
        n_internal = len(self._neighbour)
        paired = torch.zeros(n_cells, dtype=torch.bool)
        new_map = cell_map.clone()

        for i in range(n_internal):
            c0 = cell_map[self._owner[i]].item()
            c1 = cell_map[self._neighbour[i]].item()
            if c0 == c1:
                continue
            if paired[c0] or paired[c1]:
                continue
            # 配对 c0 和 c1
            new_map[new_map == c1] = c0
            paired[c0] = True
            paired[c1] = True

        if paired.sum() == 0:
            return None

        # 重新编号为连续索引
        unique_vals = torch.unique(new_map)
        remap = {v.item(): i for i, v in enumerate(unique_vals)}
        result = torch.zeros_like(new_map)
        for i in range(len(new_map)):
            result[i] = remap[new_map[i].item()]

        return result

    @property
    def n_levels(self) -> int:
        """粗化级别数。"""
        return len(self._levels)

    @property
    def levels(self) -> List[dict]:
        """粗化级别列表。"""
        return self._levels
