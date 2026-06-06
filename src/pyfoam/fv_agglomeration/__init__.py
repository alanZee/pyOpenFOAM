"""fvAgglomeration — 网格粗化方法（用于 GAMG 求解器）。"""
from pyfoam.fv_agglomeration.pair_agglomeration import PairGamgAgglomeration

__all__ = [
    "PairGamgAgglomeration",
]
