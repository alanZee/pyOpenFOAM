"""
增强 fvModels v2 — SemiImplicitSource 变体。

提供:

- :class:`CellSetSemiImplicitSource` — 限定在命名 cellSet 内的半隐式源项
- :class:`PatchSemiImplicitSource` — 通过边界 patch 施加的源项
- :class:`ExplicitSource` — 纯显式体积源项（无隐式部分）

Usage::

    from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource

    model = CellSetSemiImplicitSource(Su=100.0, Sp=-0.5, cells=[0, 1, 2])
    model.apply(matrix, field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "CellSetSemiImplicitSource",
    "PatchSemiImplicitSource",
    "ExplicitSource",
]


# ---------------------------------------------------------------------------
# CellSetSemiImplicitSource
# ---------------------------------------------------------------------------


@FvModel.register("cellSetSemiImplicitSource")
class CellSetSemiImplicitSource(FvModel):
    """限定在命名 cellSet 内的半隐式源项。

    与基础 :class:`SemiImplicitSource` 类似，但要求显式指定
    cell 索引集合（不允许 ``cells=None`` 施加到所有单元），
    确保源项仅施加在指定区域内。

    对应 OpenFOAM 中基于 ``cellSet`` 选取的 ``semiImplicitSource``。

    Parameters
    ----------
    Su : float | torch.Tensor
        显式源项（常数部分）。
    Sp : float | torch.Tensor
        隐式源项系数（场量比例部分）。
    cells : list[int] | torch.Tensor
        施加源项的单元索引。**必填**。
    V : float
        区域总体积 [m^3]，用于将总源项归一化到单位体积。
        默认 ``1.0``（不归一化）。

    Examples::

        model = CellSetSemiImplicitSource(
            Su=50.0, Sp=-0.1, cells=[10, 11, 12, 13], V=0.001,
        )
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        Su: float | torch.Tensor = 0.0,
        Sp: float | torch.Tensor = 0.0,
        cells: list[int] | torch.Tensor,
        V: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(Su=Su, Sp=Sp, cells=cells, V=V, **kwargs)
        if V <= 0.0:
            raise ValueError(f"V must be > 0, got {V}")

        self._Su = Su
        self._Sp = Sp
        self._V = V
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells.to(dtype=torch.long)
        )

    @property
    def Su(self) -> float | torch.Tensor:
        """显式源项系数。"""
        return self._Su

    @property
    def Sp(self) -> float | torch.Tensor:
        """隐式源项系数。"""
        return self._Sp

    @property
    def cells(self) -> torch.Tensor:
        """施加源项的单元索引。"""
        return self._cells

    @property
    def V(self) -> float:
        """区域总体积 [m^3]。"""
        return self._V

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """将源项施加到指定 cellSet 内的单元。

        源项先按区域体积归一化: Su/V, Sp/V，再施加到矩阵。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        def _to_cells(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, (int, float)):
                return torch.full((n,), float(val) / self._V, device=device, dtype=dtype)
            return val.to(device=device, dtype=dtype) / self._V

        su = _to_cells(self._Su)
        sp = _to_cells(self._Sp)

        idx = self._cells.to(device=device)
        mask = torch.zeros(n, device=device, dtype=dtype)
        mask.scatter_(0, idx, 1.0)
        su = su * mask
        sp = sp * mask

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"CellSetSemiImplicitSource(Su={self._Su}, Sp={self._Sp}, "
            f"n_cells={len(self._cells)}, V={self._V})"
        )


# ---------------------------------------------------------------------------
# PatchSemiImplicitSource
# ---------------------------------------------------------------------------


@FvModel.register("patchSemiImplicitSource")
class PatchSemiImplicitSource(FvModel):
    """通过边界 patch 施加的半隐式源项。

    模拟 OpenFOAM 中基于 ``patch`` 选择的源项，将源项分布到
    与指定 patch 相邻的内部单元上。在 pyOpenFOAM 中，通过
    指定 patch 相邻单元索引来实现。

    适用于需要通过边界注入能量/动量的场景（如进气口加热、
    壁面热通量等）。

    对应 OpenFOAM 的 ``patchSemiImplicitSource``。

    Parameters
    ----------
    Su : float | torch.Tensor
        显式源项。
    Sp : float | torch.Tensor
        隐式源项系数。
    patch_cells : list[int] | torch.Tensor
        patch 相邻的内部单元索引。
    weight : float
        源项权重因子（可用来控制 patch 面积占比）。默认 ``1.0``。

    Examples::

        model = PatchSemiImplicitSource(
            Su=1000.0, Sp=-2.0, patch_cells=[0, 1, 2],
        )
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        Su: float | torch.Tensor = 0.0,
        Sp: float | torch.Tensor = 0.0,
        patch_cells: list[int] | torch.Tensor,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            Su=Su, Sp=Sp, patch_cells=patch_cells, weight=weight, **kwargs,
        )
        if weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {weight}")

        self._Su = Su
        self._Sp = Sp
        self._weight = weight
        self._patch_cells = (
            torch.tensor(patch_cells, dtype=torch.long)
            if isinstance(patch_cells, list)
            else patch_cells.to(dtype=torch.long)
        )

    @property
    def Su(self) -> float | torch.Tensor:
        """显式源项系数。"""
        return self._Su

    @property
    def Sp(self) -> float | torch.Tensor:
        """隐式源项系数。"""
        return self._Sp

    @property
    def patch_cells(self) -> torch.Tensor:
        """patch 相邻的内部单元索引。"""
        return self._patch_cells

    @property
    def weight(self) -> float:
        """源项权重因子。"""
        return self._weight

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """将源项施加到 patch 相邻单元。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells
        w = self._weight

        def _to_cells(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, (int, float)):
                return torch.full((n,), float(val) * w, device=device, dtype=dtype)
            return val.to(device=device, dtype=dtype) * w

        su = _to_cells(self._Su)
        sp = _to_cells(self._Sp)

        idx = self._patch_cells.to(device=device)
        mask = torch.zeros(n, device=device, dtype=dtype)
        mask.scatter_(0, idx, 1.0)
        su = su * mask
        sp = sp * mask

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"PatchSemiImplicitSource(Su={self._Su}, Sp={self._Sp}, "
            f"n_cells={len(self._patch_cells)}, weight={self._weight})"
        )


# ---------------------------------------------------------------------------
# ExplicitSource
# ---------------------------------------------------------------------------


@FvModel.register("explicitSource")
class ExplicitSource(FvModel):
    """纯显式体积源项，无隐式对角贡献。

    适用于不依赖场量的固定源项（如恒定加热率、恒定质量注入）。
    由于没有隐式部分，不改善对角占优，但实现简单直接。

    对应 OpenFOAM 中 ``explicit`` 类型的源项。

    Parameters
    ----------
    Su : float | torch.Tensor
        显式源项值。标量广播到所有单元，或逐单元张量。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` 表示所有单元。

    Examples::

        model = ExplicitSource(Su=500.0, cells=[5, 6, 7])
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        Su: float | torch.Tensor = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(Su=Su, cells=cells, **kwargs)
        self._Su = Su
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def Su(self) -> float | torch.Tensor:
        """显式源项值。"""
        return self._Su

    @property
    def cells(self) -> list[int] | torch.Tensor | None:
        """限定单元索引。"""
        return self._cells

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加纯显式源项（仅修改 source，不动 diag）。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        if isinstance(self._Su, (int, float)):
            su = torch.full((n,), float(self._Su), device=device, dtype=dtype)
        else:
            su = self._Su.to(device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            su = su * mask

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        cells_info = (
            f"all" if self._cells is None
            else f"n={len(self._cells)}"
        )
        return f"ExplicitSource(Su={self._Su}, cells={cells_info})"
