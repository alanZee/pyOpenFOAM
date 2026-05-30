"""
增强 fvSources v10 — 旋转体与用户编码源项。

提供:

- :class:`RotatingDiskSource` — 旋转盘动量源项
- :class:`RotatingConeSource` — 旋转锥动量源项
- :class:`CodedSource` — 用户编码源项

Usage::

    from pyfoam.fv.enhanced_10 import RotatingDiskSource

    model = RotatingDiskSource(omega=314.16, R=0.5)
    model.apply(momentum_matrix, U_field)
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "RotatingDiskSource",
    "RotatingConeSource",
    "CodedSource",
]


# ---------------------------------------------------------------------------
# RotatingDiskSource
# ---------------------------------------------------------------------------


@FvModel.register("rotatingDiskSource")
class RotatingDiskSource(FvModel):
    """旋转盘动量源项。

    模拟旋转盘（如搅拌器、离心泵叶轮）对流体的动量传递::

        F = rho * C_d * omega^2 * r^3 / R^2

    其中:
    - ``C_d`` — 阻力系数 [-]
    - ``omega`` — 角速度 [rad/s]
    - ``r`` — 径向距离 [m]
    - ``R`` — 盘半径 [m]

    简化为常数源项::

        Su = rho * C_d * omega^2 * R   (基于盘边缘速度)

    对应 OpenFOAM 的 ``rotatingDisk`` fvModel。

    Parameters
    ----------
    omega : float
        角速度 [rad/s]。默认 ``100.0``。
    R : float
        盘半径 [m]。默认 ``0.5``。
    C_d : float
        阻力系数 [-]。默认 ``1.0``。
    rho : float
        流体密度 [kg/m^3]。默认 ``1.225``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = RotatingDiskSource(omega=314.16, R=0.3)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        omega: float = 100.0,
        R: float = 0.5,
        C_d: float = 1.0,
        rho: float = 1.225,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            omega=omega, R=R, C_d=C_d, rho=rho, cells=cells, **kwargs,
        )
        if R <= 0.0:
            raise ValueError(f"R must be > 0, got {R}")
        if C_d < 0.0:
            raise ValueError(f"C_d must be >= 0, got {C_d}")

        self._omega = omega
        self._R = R
        self._C_d = C_d
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def omega(self) -> float:
        """角速度 [rad/s]。"""
        return self._omega

    @property
    def R(self) -> float:
        """盘半径 [m]。"""
        return self._R

    @property
    def C_d(self) -> float:
        """阻力系数。"""
        return self._C_d

    @property
    def rho(self) -> float:
        """流体密度 [kg/m^3]。"""
        return self._rho

    @property
    def tip_speed(self) -> float:
        """盘边缘线速度 [m/s] = omega * R。"""
        return self._omega * self._R

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加旋转盘动量源项。

        field 解释为速度分量 U。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # F = rho * C_d * omega^2 * R
        su_val = self._rho * self._C_d * self._omega ** 2 * self._R

        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
        else:
            su[:] = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"RotatingDiskSource(omega={self._omega}, R={self._R}, "
            f"C_d={self._C_d}, rho={self._rho})"
        )


# ---------------------------------------------------------------------------
# RotatingConeSource
# ---------------------------------------------------------------------------


@FvModel.register("rotatingConeSource")
class RotatingConeSource(FvModel):
    """旋转锥动量源项。

    模拟锥形搅拌器或锥形旋转体对流体的动量传递。
    锥形几何使动量源沿轴向衰减::

        F(r, z) = rho * C_d * omega^2 * r * (1 - z/H)

    其中:
    - ``r`` — 径向距离
    - ``z`` — 轴向距离
    - ``H`` — 锥高度

    简化为常数形式::

        Su = rho * C_d * omega^2 * R_mean

    对应 OpenFOAM 的 ``rotatingCone`` fvModel。

    Parameters
    ----------
    omega : float
        角速度 [rad/s]。默认 ``100.0``。
    R_mean : float
        锥体平均半径 [m]。默认 ``0.3``。
    H : float
        锥体高度 [m]。默认 ``0.5``。
    C_d : float
        阻力系数 [-]。默认 ``1.0``。
    rho : float
        流体密度 [kg/m^3]。默认 ``1.225``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = RotatingConeSource(omega=200.0, R_mean=0.2, H=0.5)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        omega: float = 100.0,
        R_mean: float = 0.3,
        H: float = 0.5,
        C_d: float = 1.0,
        rho: float = 1.225,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            omega=omega, R_mean=R_mean, H=H, C_d=C_d, rho=rho,
            cells=cells, **kwargs,
        )
        if R_mean <= 0.0:
            raise ValueError(f"R_mean must be > 0, got {R_mean}")
        if H <= 0.0:
            raise ValueError(f"H must be > 0, got {H}")

        self._omega = omega
        self._R_mean = R_mean
        self._H = H
        self._C_d = C_d
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def omega(self) -> float:
        """角速度 [rad/s]。"""
        return self._omega

    @property
    def R_mean(self) -> float:
        """锥体平均半径 [m]。"""
        return self._R_mean

    @property
    def H(self) -> float:
        """锥体高度 [m]。"""
        return self._H

    @property
    def C_d(self) -> float:
        """阻力系数。"""
        return self._C_d

    @property
    def rho(self) -> float:
        """流体密度 [kg/m^3]。"""
        return self._rho

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加旋转锥动量源项。

        field 解释为速度分量 U。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # F = rho * C_d * omega^2 * R_mean
        su_val = self._rho * self._C_d * self._omega ** 2 * self._R_mean

        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
        else:
            su[:] = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"RotatingConeSource(omega={self._omega}, R_mean={self._R_mean}, "
            f"H={self._H}, C_d={self._C_d})"
        )


# ---------------------------------------------------------------------------
# CodedSource
# ---------------------------------------------------------------------------


@FvModel.register("codedSource")
class CodedSource(FvModel):
    """用户编码源项。

    允许通过用户提供的 Python 函数计算任意源项。
    与 :class:`CodedFvModel` 类似，但支持返回 Su 和 Sp
    的不同签名，更灵活::

        def my_source(field, t=0.0):
            Su = ...
            Sp = ...
            return Su, Sp

    对应 OpenFOAM 的 ``codedSource`` fvModel。

    Parameters
    ----------
    code : callable
        源项计算函数，签名 ``(field) -> (Su, Sp)`` 或
        ``(field) -> (Su, Sp, t)``。
    name : str
        描述性名称。默认 ``"codedSource"``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        def custom_source(field):
            Su = -0.1 * field ** 2
            Sp = -0.2 * field
            return Su, Sp

        model = CodedSource(code=custom_source, name="quadratic_sink")
        model.apply(matrix, field)
    """

    def __init__(
        self,
        *,
        code: Callable,
        name: str = "codedSource",
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(code=code, name=name, cells=cells, **kwargs)
        if not callable(code):
            raise TypeError("code must be a callable")

        self._code = code
        self._name = name
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def code(self) -> Callable:
        """用户提供的源项函数。"""
        return self._code

    @property
    def name(self) -> str:
        """描述性名称。"""
        return self._name

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """评估用户函数并施加源项到矩阵。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        f = field.to(device=device, dtype=dtype)

        result = self._code(f)
        Su, Sp = result[0], result[1]

        if isinstance(Su, (int, float)):
            Su = torch.full((n,), float(Su), device=device, dtype=dtype)
        else:
            Su = Su.to(device=device, dtype=dtype)

        if isinstance(Sp, (int, float)):
            Sp = torch.full((n,), float(Sp), device=device, dtype=dtype)
        else:
            Sp = Sp.to(device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            Su = Su * mask
            Sp = Sp * mask

        matrix._source = matrix._source + Su
        matrix._diag = matrix._diag + Sp

    def __repr__(self) -> str:
        return f"CodedSource(name='{self._name}')"
