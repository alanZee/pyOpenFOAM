"""
增强 fvConstraints v8 — 温度/物种约束。

提供:

- :class:`MinTemperatureConstraint` — 最低温度约束
- :class:`MaxTemperatureConstraint` — 最高温度约束
- :class:`MassFractionLimitsConstraint` — 物种质量分数约束

Usage::

    from pyfoam.fv.enhanced_8 import MinTemperatureConstraint

    constraint = MinTemperatureConstraint(T_min=200.0)
    constraint.apply(T_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.fv.fv_constraints import FvConstraint

__all__ = [
    "MinTemperatureConstraint",
    "MaxTemperatureConstraint",
    "MassFractionLimitsConstraint",
]


# ---------------------------------------------------------------------------
# MinTemperatureConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("minTemperature")
class MinTemperatureConstraint(FvConstraint):
    """最低温度约束。

    确保温度不低于指定的物理下限，防止非物理的低温值。
    在辐射冷却或高度过冲等场景中尤为重要。

    对应 OpenFOAM 的 ``minTemperature`` fvConstraint。

    Parameters
    ----------
    T_min : float
        最低温度 [K]。默认 ``1.0``（避免 T=0 问题）。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        c = MinTemperatureConstraint(T_min=200.0)
        c.apply(T_field)
    """

    def __init__(
        self,
        *,
        T_min: float = 1.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(T_min=T_min, cells=cells, **kwargs)
        if T_min <= 0.0:
            raise ValueError(f"T_min must be > 0, got {T_min}")

        self._T_min = T_min
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def T_min(self) -> float:
        """最低温度 [K]。"""
        return self._T_min

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """将温度场下限钳位到 T_min。"""
        if self._cells is not None:
            sub = field[self._cells]
            sub.clamp_(min=self._T_min)
            field[self._cells] = sub
        else:
            field.clamp_(min=self._T_min)
        return field

    def __repr__(self) -> str:
        return f"MinTemperatureConstraint(T_min={self._T_min})"


# ---------------------------------------------------------------------------
# MaxTemperatureConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("maxTemperature")
class MaxTemperatureConstraint(FvConstraint):
    """最高温度约束。

    确保温度不超过指定的物理上限，防止数值过冲导致的
    非物理高温值。对于化学反应、电弧加热等高温场景至关重要。

    对应 OpenFOAM 的 ``maxTemperature`` fvConstraint。

    Parameters
    ----------
    T_max : float
        最高温度 [K]。默认 ``10000.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        c = MaxTemperatureConstraint(T_max=5000.0)
        c.apply(T_field)
    """

    def __init__(
        self,
        *,
        T_max: float = 10000.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(T_max=T_max, cells=cells, **kwargs)
        if T_max <= 0.0:
            raise ValueError(f"T_max must be > 0, got {T_max}")

        self._T_max = T_max
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def T_max(self) -> float:
        """最高温度 [K]。"""
        return self._T_max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """将温度场上限钳位到 T_max。"""
        if self._cells is not None:
            sub = field[self._cells]
            sub.clamp_(max=self._T_max)
            field[self._cells] = sub
        else:
            field.clamp_(max=self._T_max)
        return field

    def __repr__(self) -> str:
        return f"MaxTemperatureConstraint(T_max={self._T_max})"


# ---------------------------------------------------------------------------
# MassFractionLimitsConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("massFractionLimits")
class MassFractionLimitsConstraint(FvConstraint):
    """物种质量分数约束。

    确保各物种的质量分数在 [0, 1] 范围内，保证物理可实现性。
    在多组分燃烧模拟中，质量分数可能出现数值过冲/下冲。

    对应 OpenFOAM 的 ``massFractionLimits`` fvConstraint。

    Parameters
    ----------
    min : float
        最小质量分数。默认 ``0.0``。
    max : float
        最大质量分数。默认 ``1.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        c = MassFractionLimitsConstraint(min=0.0, max=1.0)
        c.apply(Y_field)
    """

    def __init__(
        self,
        *,
        min: float = 0.0,
        max: float = 1.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(min=min, max=max, cells=cells, **kwargs)
        if min < 0.0:
            raise ValueError(f"min must be >= 0, got {min}")
        if max > 1.0:
            raise ValueError(f"max must be <= 1.0, got {max}")
        if min > max:
            raise ValueError(f"min ({min}) must be <= max ({max})")

        self._min = min
        self._max = max
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def min(self) -> float:
        """最小质量分数。"""
        return self._min

    @property
    def max(self) -> float:
        """最大质量分数。"""
        return self._max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """将质量分数场钳位到 [min, max]。"""
        if self._cells is not None:
            sub = field[self._cells]
            sub.clamp_(min=self._min, max=self._max)
            field[self._cells] = sub
        else:
            field.clamp_(min=self._min, max=self._max)
        return field

    def __repr__(self) -> str:
        return (
            f"MassFractionLimitsConstraint(min={self._min}, max={self._max})"
        )
