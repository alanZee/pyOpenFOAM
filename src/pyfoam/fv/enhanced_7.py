"""
增强 fvSources v7 — 多参考系 (MRF) 与区域间传热源项。

提供:

- :class:`MRFSource` — 多参考系 (MRF) 动量源项
- :class:`MRFSolidBody` — MRF 固体区域旋转源项
- :class:`InterRegionHeatTransfer` — 区域间传热源项

Usage::

    from pyfoam.fv.enhanced_7 import MRFSource

    model = MRFSource(omega=100.0, axis=[0, 0, 1])
    model.apply(momentum_matrix, U_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "MRFSource",
    "MRFSolidBody",
    "InterRegionHeatTransfer",
]


# ---------------------------------------------------------------------------
# MRFSource
# ---------------------------------------------------------------------------


@FvModel.register("mrfSource")
class MRFSource(FvModel):
    """多参考系 (MRF) 动量源项。

    在 MRF 方法中，部分计算域被指定为旋转参考系。
    MRF 区域内的单元受到附加的 Coriolis 力和离心力::

        F_coriolis = -2 * rho * omega x U_rel
        F_centrifugal = rho * omega x (omega x r)

    简化标量形式（假设 z 轴旋转）::

        Su = omega^2 * r      (离心力，显式)
        Sp = -2 * omega        (Coriolis，隐式)

    对应 OpenFOAM 的 ``MRFZone`` / ``fv::MRFSource``。

    Parameters
    ----------
    omega : float
        角速度 [rad/s]。默认 ``100.0``。
    axis : list[float]
        旋转轴方向。默认 ``[0, 0, 1]``。
    rho : float
        流体密度 [kg/m^3]。默认 ``1.225``。
    coriolis_implicit : float
        Coriolis 力隐式线性化系数。默认 ``-2.0``。
    cells : list[int] | torch.Tensor | None
        MRF 区域的单元索引。

    Examples::

        model = MRFSource(omega=314.16, cells=mrf_cells)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        omega: float = 100.0,
        axis: list[float] | None = None,
        rho: float = 1.225,
        coriolis_implicit: float = -2.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if axis is None:
            axis = [0.0, 0.0, 1.0]
        super().__init__(
            omega=omega, axis=axis, rho=rho,
            coriolis_implicit=coriolis_implicit, cells=cells, **kwargs,
        )

        self._omega = omega
        self._axis = torch.tensor(axis, dtype=torch.float64)
        self._rho = rho
        self._coriolis_implicit = coriolis_implicit
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
    def axis(self) -> torch.Tensor:
        """旋转轴方向矢量。"""
        return self._axis

    @property
    def rho(self) -> float:
        """流体密度 [kg/m^3]。"""
        return self._rho

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 MRF 源项。

        field 解释为速度分量（标量）。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # 离心力: Su = rho * omega^2 (显式)
        su_val = self._rho * self._omega ** 2
        # Coriolis: Sp = rho * coriolis_implicit * omega
        sp_val = self._rho * self._coriolis_implicit * self._omega

        su = torch.zeros(n, device=device, dtype=dtype)
        sp = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
            sp.scatter_(0, idx, sp_val)
        else:
            su[:] = su_val
            sp[:] = sp_val

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"MRFSource(omega={self._omega}, rho={self._rho}, "
            f"axis={self._axis.tolist()})"
        )


# ---------------------------------------------------------------------------
# MRFSolidBody
# ---------------------------------------------------------------------------


@FvModel.register("mrfSolidBody")
class MRFSolidBody(FvModel):
    """MRF 固体区域旋转源项。

    用于固体（如叶轮叶片）在 MRF 框架下的处理。
    固体区域内的速度被强制等于旋转速度::

        U_solid = omega x r

    通过大的源项系数强制实现::

        Su = omega^2 * r * S_large  (强制目标速度)
        Sp = -S_large               (隐式惩罚)

    其中 S_large 是一个大数（惩罚参数）。

    对应 OpenFOAM 中 MRFZone 的 ``solidBodyMotion`` 功能。

    Parameters
    ----------
    omega : float
        角速度 [rad/s]。默认 ``100.0``。
    penalty : float
        惩罚系数（越大约束越强）。默认 ``1e6``。
    cells : list[int] | torch.Tensor | None
        固体区域的单元索引。

    Examples::

        model = MRFSolidBody(omega=314.16, penalty=1e8, cells=solid_cells)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        omega: float = 100.0,
        penalty: float = 1e6,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            omega=omega, penalty=penalty, cells=cells, **kwargs,
        )
        if penalty <= 0.0:
            raise ValueError(f"penalty must be > 0, got {penalty}")

        self._omega = omega
        self._penalty = penalty
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
    def penalty(self) -> float:
        """惩罚系数。"""
        return self._penalty

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加固体旋转惩罚源项。

        field 解释为当前速度分量 U。
        惩罚方法强制 U ≈ omega * r。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        # U_target = omega * r (简化为 omega)
        U_target = self._omega

        su_val = self._penalty * U_target
        sp_val = -self._penalty

        su = torch.zeros(n, device=device, dtype=dtype)
        sp = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
            sp.scatter_(0, idx, sp_val)
        else:
            su[:] = su_val
            sp[:] = sp_val

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"MRFSolidBody(omega={self._omega}, penalty={self._penalty})"
        )


# ---------------------------------------------------------------------------
# InterRegionHeatTransfer
# ---------------------------------------------------------------------------


@FvModel.register("interRegionHeatTransfer")
class InterRegionHeatTransfer(FvModel):
    """区域间传热源项。

    模拟两个相邻区域之间的热量传递（如固体-流体界面）::

        Q = h * A_s / V * (T_neighbor - T)

    其中:
    - ``h`` — 界面传热系数 [W/(m^2 K)]
    - ``A_s`` — 界面面积 [m^2]
    - ``V`` — 单元体积 [m^3]
    - ``T_neighbor`` — 邻域温度 [K]

    半隐式线性化::

        Su = h * A_s / V * T_neighbor   (显式)
        Sp = -h * A_s / V               (隐式，负值稳定)

    对应 OpenFOAM 的 ``interRegion`` 传热 fvModel。

    Parameters
    ----------
    h : float
        界面传热系数 [W/(m^2 K)]。默认 ``100.0``。
    A_s : float
        界面面积 [m^2]。默认 ``1.0``。
    V : float
        单元体积 [m^3]。默认 ``1.0``。
    T_neighbor : float
        邻域参考温度 [K]。默认 ``300.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = InterRegionHeatTransfer(
            h=500.0, A_s=0.01, V=1e-3, T_neighbor=350.0,
        )
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        h: float = 100.0,
        A_s: float = 1.0,
        V: float = 1.0,
        T_neighbor: float = 300.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            h=h, A_s=A_s, V=V, T_neighbor=T_neighbor, cells=cells, **kwargs,
        )
        if h < 0.0:
            raise ValueError(f"h must be >= 0, got {h}")
        if A_s <= 0.0:
            raise ValueError(f"A_s must be > 0, got {A_s}")
        if V <= 0.0:
            raise ValueError(f"V must be > 0, got {V}")

        self._h = h
        self._A_s = A_s
        self._V = V
        self._T_neighbor = T_neighbor
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def h(self) -> float:
        """界面传热系数 [W/(m^2 K)]。"""
        return self._h

    @property
    def A_s(self) -> float:
        """界面面积 [m^2]。"""
        return self._A_s

    @property
    def V(self) -> float:
        """单元体积 [m^3]。"""
        return self._V

    @property
    def T_neighbor(self) -> float:
        """邻域参考温度 [K]。"""
        return self._T_neighbor

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加区域间传热源项。

        field 解释为温度场 T。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        coeff = self._h * self._A_s / self._V

        su_val = coeff * self._T_neighbor
        sp_val = -coeff

        su = torch.zeros(n, device=device, dtype=dtype)
        sp = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
            sp.scatter_(0, idx, sp_val)
        else:
            su[:] = su_val
            sp[:] = sp_val

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"InterRegionHeatTransfer(h={self._h}, A_s={self._A_s}, "
            f"V={self._V}, T_neighbor={self._T_neighbor})"
        )
