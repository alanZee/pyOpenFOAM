"""
增强 fvConstraints v4 — 密度/速度约束变体。

提供:

- :class:`MinMaxConstraint` — 独立的 min/max 钳位约束
- :class:`RhoLimitsConstraint` — 密度上下限约束
- :class:`VelocityLimitsConstraint` — 速度幅值约束

Usage::

    from pyfoam.fv.enhanced_4 import RhoLimitsConstraint

    constraint = RhoLimitsConstraint(min=0.1, max=10.0)
    constraint.apply(rho_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.fv.fv_constraints import FvConstraint

__all__ = [
    "MinMaxConstraint",
    "RhoLimitsConstraint",
    "VelocityLimitsConstraint",
]


# ---------------------------------------------------------------------------
# MinMaxConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("minMax")
class MinMaxConstraint(FvConstraint):
    """独立的 min/max 钳位约束。

    与基础 ``BoundConstraint`` 类似，但额外支持 ``cells`` 限定，
    可对场的特定区域施加不同的约束范围。

    对应 OpenFOAM 中 ``fvConstraints`` 的 ``minMax`` 变体。

    Parameters
    ----------
    min : float | None
        下界。``None`` 表示无下界。
    max : float | None
        上界。``None`` 表示无上界。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        c = MinMaxConstraint(min=0.0, max=1.0, cells=[0, 1, 2])
        c.apply(field)
    """

    def __init__(
        self,
        *,
        min: float | None = None,
        max: float | None = None,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(min=min, max=max, cells=cells, **kwargs)
        if min is not None and max is not None and min > max:
            raise ValueError(
                f"min ({min}) must be <= max ({max})"
            )
        self._min = min
        self._max = max
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def min(self) -> float | None:
        """下界。"""
        return self._min

    @property
    def max(self) -> float | None:
        """上界。"""
        return self._max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """对指定单元施加 min/max 钳位。"""
        if self._cells is not None:
            sub = field[self._cells]
            if self._min is not None:
                sub = sub.clamp(min=self._min)
            if self._max is not None:
                sub = sub.clamp(max=self._max)
            field[self._cells] = sub
        else:
            if self._min is not None:
                field.clamp_(min=self._min)
            if self._max is not None:
                field.clamp_(max=self._max)
        return field

    def __repr__(self) -> str:
        cells_info = (
            "all" if self._cells is None
            else f"n={len(self._cells)}"
        )
        return f"MinMaxConstraint(min={self._min}, max={self._max}, cells={cells_info})"


# ---------------------------------------------------------------------------
# RhoLimitsConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("rhoLimits")
class RhoLimitsConstraint(FvConstraint):
    """密度场上下限约束。

    确保密度在物理合理的范围内，防止数值发散。
    典型的密度下限用于避免零密度或负密度。

    对应 OpenFOAM 的 ``rhoLimits`` fvConstraint。

    Parameters
    ----------
    min : float
        最小密度 [kg/m^3]。默认 ``1e-10``。
    max : float
        最大密度 [kg/m^3]。默认 ``100.0``。

    Examples::

        c = RhoLimitsConstraint(min=0.1, max=10.0)
        c.apply(rho_field)
    """

    def __init__(
        self,
        *,
        min: float = 1e-10,
        max: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(min=min, max=max, **kwargs)
        if min < 0.0:
            raise ValueError(f"min must be >= 0, got {min}")
        if max <= 0.0:
            raise ValueError(f"max must be > 0, got {max}")
        if min > max:
            raise ValueError(f"min ({min}) must be <= max ({max})")

        self._min = min
        self._max = max

    @property
    def min(self) -> float:
        """最小密度 [kg/m^3]。"""
        return self._min

    @property
    def max(self) -> float:
        """最大密度 [kg/m^3]。"""
        return self._max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """将密度场钳位到 [min, max]。"""
        field.clamp_(min=self._min, max=self._max)
        return field

    def __repr__(self) -> str:
        return f"RhoLimitsConstraint(min={self._min}, max={self._max})"


# ---------------------------------------------------------------------------
# VelocityLimitsConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("velocityLimits")
class VelocityLimitsConstraint(FvConstraint):
    """速度幅值约束。

    限制速度矢量的幅值 (|U|) 在指定范围内，防止数值
    过冲导致的速度非物理增长。

    当 |U| > max 时，将速度矢量缩放到 max 幅值。
    当 |U| < min 时，将速度矢量缩放到 min 幅值（除非为零向量）。

    注意: 本约束作用于速度幅值标量场 (|U|)，而非矢量分量。
    对于矢量场，请对每个分量分别施加。

    对应 OpenFOAM 的 ``velocityLimits`` fvConstraint。

    Parameters
    ----------
    min : float
        最小速度幅值 [m/s]。默认 ``0.0``。
    max : float
        最大速度幅值 [m/s]。默认 ``1000.0``。

    Examples::

        c = VelocityLimitsConstraint(min=0.0, max=100.0)
        U_mag = torch.norm(U_field, dim=1)
        c.apply(U_mag)
    """

    def __init__(
        self,
        *,
        min: float = 0.0,
        max: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(min=min, max=max, **kwargs)
        if min < 0.0:
            raise ValueError(f"min must be >= 0, got {min}")
        if max <= 0.0:
            raise ValueError(f"max must be > 0, got {max}")
        if min > max:
            raise ValueError(f"min ({min}) must be <= max ({max})")

        self._min = min
        self._max = max

    @property
    def min(self) -> float:
        """最小速度幅值 [m/s]。"""
        return self._min

    @property
    def max(self) -> float:
        """最大速度幅值 [m/s]。"""
        return self._max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """将速度幅值场钳位到 [min, max]。"""
        field.clamp_(min=self._min, max=self._max)
        return field

    def __repr__(self) -> str:
        return f"VelocityLimitsConstraint(min={self._min}, max={self._max})"
