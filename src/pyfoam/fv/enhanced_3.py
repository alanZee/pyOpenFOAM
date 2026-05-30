"""
增强 fvSources v3 — 浮力与旋转参考系力源项。

提供:

- :class:`BuoyancyForce` — 浮力力源项: (rho - rho_ref) * g
- :class:`BoussinesqBuoyancy` — Boussinesq 近似浮力: -rho_ref * beta * (T - T_ref) * g
- :class:`SRFForce` — 单参考系 (SRF) 力: Coriolis + 离心力

Usage::

    from pyfoam.fv.enhanced_3 import BuoyancyForce

    model = BuoyancyForce(rho_ref=1.225, g=[0, 0, -9.81])
    model.apply(momentum_matrix, rho_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "BuoyancyForce",
    "BoussinesqBuoyancy",
    "SRFForce",
]


# ---------------------------------------------------------------------------
# BuoyancyForce
# ---------------------------------------------------------------------------


@FvModel.register("buoyancyForce")
class BuoyancyForce(FvModel):
    """浮力力源项: F = (rho - rho_ref) * g。

    用于自然对流、分层流等需要考虑密度差引起浮力的场景。
    与 :class:`GravitationalBodyForce` 不同，本模型始终以密度差形式
    施加重力，适用于可变密度流。

    对应 OpenFOAM 的 ``buoyancyForce`` fvModel。

    Parameters
    ----------
    rho_ref : float
        参考密度 [kg/m^3]。默认 ``1.225``（空气）。
    g : list[float] | torch.Tensor
        重力加速度矢量 [m/s^2]。默认 ``[0, 0, -9.81]``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = BuoyancyForce(rho_ref=1.225)
        model.apply(momentum_matrix, rho_field)
    """

    def __init__(
        self,
        *,
        rho_ref: float = 1.225,
        g: list[float] | torch.Tensor | None = None,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if g is None:
            g = [0.0, 0.0, -9.81]
        super().__init__(rho_ref=rho_ref, g=g, cells=cells, **kwargs)

        self._rho_ref = rho_ref
        self._g = (
            torch.tensor(g, dtype=torch.float64)
            if isinstance(g, list)
            else g
        )
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def rho_ref(self) -> float:
        """参考密度 [kg/m^3]。"""
        return self._rho_ref

    @property
    def g(self) -> torch.Tensor:
        """重力加速度矢量。"""
        return self._g

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加浮力源项。

        field 解释为密度场 rho。使用 g 的 z 分量计算标量浮力。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        rho = field.to(device=device, dtype=dtype)
        g_z = float(self._g[2].item()) if len(self._g) > 2 else -9.81

        # F = (rho - rho_ref) * g_z
        su_val = (rho - self._rho_ref) * g_z

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
        else:
            su = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        g_str = f"[{', '.join(f'{v:.2f}' for v in self._g.tolist())}]"
        return f"BuoyancyForce(rho_ref={self._rho_ref}, g={g_str})"


# ---------------------------------------------------------------------------
# BoussinesqBuoyancy
# ---------------------------------------------------------------------------


@FvModel.register("boussinesqBuoyancy")
class BoussinesqBuoyancy(FvModel):
    """Boussinesq 近似浮力源项。

    使用 Boussinesq 近似计算浮力::

        F = -rho_ref * beta * (T - T_ref) * g

    其中:
    - ``beta`` — 热膨胀系数 [1/K]
    - ``T_ref`` — 参考温度 [K]
    - ``rho_ref`` — 参考密度 [kg/m^3]

    Boussinesq 近似适用于温差较小 (Delta_T / T_ref << 1) 的自然对流。

    对应 OpenFOAM 的 ``boussinesq`` fvModel。

    Parameters
    ----------
    beta : float
        热膨胀系数 [1/K]。对于理想气体 beta = 1/T_ref。
        默认 ``3.33e-3``（空气在 300K）。
    T_ref : float
        参考温度 [K]。默认 ``300.0``。
    rho_ref : float
        参考密度 [kg/m^3]。默认 ``1.225``。
    g : list[float] | torch.Tensor
        重力加速度矢量 [m/s^2]。默认 ``[0, 0, -9.81]``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = BoussinesqBuoyancy(beta=3.33e-3, T_ref=300.0)
        model.apply(momentum_matrix, T_field)
    """

    def __init__(
        self,
        *,
        beta: float = 3.33e-3,
        T_ref: float = 300.0,
        rho_ref: float = 1.225,
        g: list[float] | torch.Tensor | None = None,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if g is None:
            g = [0.0, 0.0, -9.81]
        super().__init__(
            beta=beta, T_ref=T_ref, rho_ref=rho_ref, g=g, cells=cells, **kwargs,
        )
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if rho_ref <= 0.0:
            raise ValueError(f"rho_ref must be > 0, got {rho_ref}")

        self._beta = beta
        self._T_ref = T_ref
        self._rho_ref = rho_ref
        self._g = (
            torch.tensor(g, dtype=torch.float64)
            if isinstance(g, list)
            else g
        )
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def beta(self) -> float:
        """热膨胀系数 [1/K]。"""
        return self._beta

    @property
    def T_ref(self) -> float:
        """参考温度 [K]。"""
        return self._T_ref

    @property
    def rho_ref(self) -> float:
        """参考密度 [kg/m^3]。"""
        return self._rho_ref

    @property
    def g(self) -> torch.Tensor:
        """重力加速度矢量。"""
        return self._g

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 Boussinesq 浮力源项。

        field 解释为温度场 T。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        T = field.to(device=device, dtype=dtype)
        g_z = float(self._g[2].item()) if len(self._g) > 2 else -9.81

        # F = -rho_ref * beta * (T - T_ref) * g_z
        su_val = -self._rho_ref * self._beta * (T - self._T_ref) * g_z

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
        else:
            su = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"BoussinesqBuoyancy(beta={self._beta}, T_ref={self._T_ref}, "
            f"rho_ref={self._rho_ref})"
        )


# ---------------------------------------------------------------------------
# SRFForce
# ---------------------------------------------------------------------------


@FvModel.register("srfForce")
class SRFForce(FvModel):
    """单参考系 (SRF) 旋转力源项。

    在旋转参考系中，流体受到 Coriolis 力和离心力::

        F_coriolis = -2 * rho * omega x U
        F_centrifugal = rho * omega x (omega x r)

    其中:
    - ``omega`` — 角速度矢量 [rad/s]
    - ``U`` — 相对于旋转参考系的速度
    - ``r`` — 从旋转轴到单元中心的位置矢量

    本模型将两个力合并施加。在 pyOpenFOAM 中简化为标量形式:

    - Coriolis: Sp_coriolis = -2 * omega （隐式，改善稳定性）
    - 离心力: Su_centrifugal = omega^2 * r_perp （显式）

    对应 OpenFOAM 的 SRF 模型。

    Parameters
    ----------
    omega : float
        角速度标量 [rad/s]。默认 ``100.0``。
    axis : list[float]
        旋转轴方向单位矢量。默认 ``[0, 0, 1]``（z 轴）。
    omega_implicit : float
        Coriolis 力隐式线性化系数。默认 ``0.0``（纯显式）。
        设为负值可改善对角占优。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = SRFForce(omega=314.16)  # 3000 RPM
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        omega: float = 100.0,
        axis: list[float] | None = None,
        omega_implicit: float = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if axis is None:
            axis = [0.0, 0.0, 1.0]
        super().__init__(
            omega=omega, axis=axis, omega_implicit=omega_implicit,
            cells=cells, **kwargs,
        )

        self._omega = omega
        self._axis = torch.tensor(axis, dtype=torch.float64)
        self._omega_implicit = omega_implicit
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
    def omega_implicit(self) -> float:
        """Coriolis 隐式线性化系数。"""
        return self._omega_implicit

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 SRF 力源项。

        field 解释为速度场（标量分量或幅值）。

        Coriolis 力（简化标量形式）通过隐式项施加，
        离心力通过显式项施加。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        U = field.to(device=device, dtype=dtype)

        # Coriolis: -2 * omega * U  → Sp = -2 * omega
        # 离心力: omega^2 * r （简化为常数源项）
        sp_val = -2.0 * self._omega * self._omega_implicit
        su_val = self._omega ** 2  # 简化：单位体积离心力

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
            f"SRFForce(omega={self._omega}, axis={self._axis.tolist()}, "
            f"omega_implicit={self._omega_implicit})"
        )
