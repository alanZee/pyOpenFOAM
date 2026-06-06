"""
Tests for fv_agglomeration module.
"""
import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.fv_agglomeration.pair_agglomeration import PairGamgAgglomeration


class TestPairGamgAgglomeration:
    """配对 GAMG 粗化测试。"""

    def test_build_levels_1d(self):
        """1D 链网格粗化。"""
        n_cells = 16
        # 1D 链：0-1, 1-2, ..., 14-15
        owner = torch.arange(n_cells - 1, dtype=INDEX_DTYPE)
        neighbour = torch.arange(1, n_cells, dtype=INDEX_DTYPE)

        agg = PairGamgAgglomeration(n_cells, owner, neighbour)
        levels = agg.build_levels(n_levels=3)

        assert len(levels) > 0
        assert levels[0]["n_cells_fine"] == 16
        assert levels[0]["n_cells_coarse"] < 16

    def test_build_levels_small(self):
        """小网格不粗化。"""
        n_cells = 4
        owner = torch.tensor([0, 1, 2], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2, 3], dtype=INDEX_DTYPE)

        agg = PairGamgAgglomeration(n_cells, owner, neighbour)
        levels = agg.build_levels(n_levels=5)

        # 4 个单元不应粗化（<= 4 阈值）
        assert len(levels) == 0

    def test_n_levels(self):
        n_cells = 64
        owner = torch.arange(n_cells - 1, dtype=INDEX_DTYPE)
        neighbour = torch.arange(1, n_cells, dtype=INDEX_DTYPE)

        agg = PairGamgAgglomeration(n_cells, owner, neighbour)
        agg.build_levels(n_levels=5)
        assert agg.n_levels == len(agg.levels)
