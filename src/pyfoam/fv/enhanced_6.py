"""
增强 fvModels v6 — 燃烧与反应源项。

提供:

- :class:`XiEqModel` — 平衡火焰褶皱模型
- :class:`PaSRSource` — 部分搅拌反应器 (PaSR) 燃烧源项
- :class:`EDCSource` — 涡旋耗散概念 (EDC) 燃烧源项

Usage::

    from pyfoam.fv.enhanced_6 import PaSRSource

    model = PaSRSource(kappa=0.1, tau_mix=0.01)
    model.apply(energy_matrix, T_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "XiEqModel",
    "PaSRSource",
    "EDCSource",
]


# ---------------------------------------------------------------------------
# XiEqModel
# ---------------------------------------------------------------------------


@FvModel.register("xiEqModel")
class XiEqModel(FvModel):
    """平衡火焰褶皱因子模型。

    计算预混湍流燃烧中的平衡火焰褶皱因子 Xi_eq::

        Xi_eq = 1 + A * (u' / S_L)^0.5 * (Re_t)^0.25

    其中:
    - ``A`` — 模型常数（默认 0.7）
    - ``u'`` — 湍流脉动速度 [m/s]
    - ``S_L`` — 层流火焰速度 [m/s]
    - ``Re_t`` — 湍流雷诺数

    源项将 Xi_eq 施加到混合分数或反应进度方程中，
    通过半隐式线性化保证稳定性。

    对应 OpenFOAM 的 ``XiEq`` 模型。

    Parameters
    ----------
    A : float
        模型常数。默认 ``0.7``。
    u_turb : float
        湍流脉动速度 [m/s]。默认 ``1.0``。
    S_L : float
        层流火焰速度 [m/s]。默认 ``0.3``。
    alpha : float
        隐式线性化系数。默认 ``0.0``（纯显式）。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = XiEqModel(u_turb=2.0, S_L=0.3)
        model.apply(combustion_matrix, c_field)
    """

    def __init__(
        self,
        *,
        A: float = 0.7,
        u_turb: float = 1.0,
        S_L: float = 0.3,
        alpha: float = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            A=A, u_turb=u_turb, S_L=S_L, alpha=alpha, cells=cells, **kwargs,
        )
        if S_L <= 0.0:
            raise ValueError(f"S_L must be > 0, got {S_L}")
        if u_turb < 0.0:
            raise ValueError(f"u_turb must be >= 0, got {u_turb}")

        self._A = A
        self._u_turb = u_turb
        self._S_L = S_L
        self._alpha = alpha
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def A(self) -> float:
        """模型常数。"""
        return self._A

    @property
    def u_turb(self) -> float:
        """湍流脉动速度 [m/s]。"""
        return self._u_turb

    @property
    def S_L(self) -> float:
        """层流火焰速度 [m/s]。"""
        return self._S_L

    @property
    def Xi_eq(self) -> float:
        """平衡火焰褶皱因子。"""
        return 1.0 + self._A * (self._u_turb / self._S_L) ** 0.5

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 Xi_eq 源项。

        field 解释为反应进度变量 c。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        xi = self.Xi_eq

        # Su = xi * (1 - alpha), Sp = -alpha * xi (负值改善稳定性)
        su_val = xi * (1.0 - self._alpha)
        sp_val = -self._alpha * xi

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
            f"XiEqModel(A={self._A}, u_turb={self._u_turb}, "
            f"S_L={self._S_L}, Xi_eq={self.Xi_eq:.3f})"
        )


# ---------------------------------------------------------------------------
# PaSRSource
# ---------------------------------------------------------------------------


@FvModel.register("pasrSource")
class PaSRSource(FvModel):
    """部分搅拌反应器 (PaSR) 燃烧源项。

    PaSR 模型将计算域分为反应区和非反应区::

        Q = kappa * rho * Y_F * omega_F

    其中:
    - ``kappa`` — 反应区体积分数 [-]
    - ``Y_F`` — 燃料质量分数
    - ``omega_F`` — 燃料消耗率 [1/s]

    kappa 由混合时间和化学时间尺度决定::

        kappa = tau_c / (tau_c + tau_mix)

    源项半隐式线性化: Sp = -kappa * omega_F（负值改善稳定性）。

    对应 OpenFOAM 的 ``PaSR`` 模型。

    Parameters
    ----------
    kappa : float
        反应区体积分数 [-]。默认 ``0.1``。
    omega : float
        特征反应速率 [1/s]。默认 ``10.0``。
    rho : float
        流体密度 [kg/m^3]。默认 ``1.225``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = PaSRSource(kappa=0.5, omega=100.0)
        model.apply(species_matrix, Y_field)
    """

    def __init__(
        self,
        *,
        kappa: float = 0.1,
        omega: float = 10.0,
        rho: float = 1.225,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kappa=kappa, omega=omega, rho=rho, cells=cells, **kwargs,
        )
        if kappa < 0.0 or kappa > 1.0:
            raise ValueError(f"kappa must be in [0, 1], got {kappa}")
        if omega < 0.0:
            raise ValueError(f"omega must be >= 0, got {omega}")

        self._kappa = kappa
        self._omega = omega
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def kappa(self) -> float:
        """反应区体积分数。"""
        return self._kappa

    @property
    def omega(self) -> float:
        """特征反应速率 [1/s]。"""
        return self._omega

    @property
    def rho(self) -> float:
        """流体密度 [kg/m^3]。"""
        return self._rho

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 PaSR 燃烧源项。

        field 解释为燃料质量分数 Y_F。
        源项: Su = kappa * rho * omega * Y_F (显式贡献)
              Sp = -kappa * rho * omega (隐式，负值稳定)
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        Y = field.to(device=device, dtype=dtype)

        rate = self._kappa * self._rho * self._omega
        su_val = rate * Y  # 随场量变化
        sp_val = -rate     # 恒定隐式系数

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            sp = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
            sp.scatter_(0, idx, sp_val)
        else:
            su = su_val
            sp = torch.full((n,), sp_val, device=device, dtype=dtype)

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"PaSRSource(kappa={self._kappa}, omega={self._omega}, "
            f"rho={self._rho})"
        )


# ---------------------------------------------------------------------------
# EDCSource
# ---------------------------------------------------------------------------


@FvModel.register("edcSource")
class EDCSource(FvModel):
    """涡旋耗散概念 (EDC) 燃烧源项。

    EDC 模型假设化学反应发生在湍流最小涡旋尺度::

        Q = rho * gamma^2 / tau * min(Y_F, Y_O / s)

    其中:
    - ``gamma`` — 尺度分数常数（典型 2.1377）
    - ``tau`` — 湍流特征时间尺度 [s]
    - ``Y_F`` — 燃料质量分数
    - ``Y_O`` — 氧化剂质量分数
    - ``s`` — 化学计量比

    简化形式::

        Su = C_edc / tau * rho * Y_F  (显式)
        Sp = -C_edc / tau * rho       (隐式，稳定)

    对应 OpenFOAM 的 ``EDC`` 燃烧模型。

    Parameters
    ----------
    C_edc : float
        EDC 常数（gamma^2 的近似）。默认 ``4.57``。
    tau : float
        湍流特征时间尺度 [s]。默认 ``0.01``。
    rho : float
        流体密度 [kg/m^3]。默认 ``1.225``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = EDCSource(C_edc=4.57, tau=0.01)
        model.apply(species_matrix, Y_F_field)
    """

    def __init__(
        self,
        *,
        C_edc: float = 4.57,
        tau: float = 0.01,
        rho: float = 1.225,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            C_edc=C_edc, tau=tau, rho=rho, cells=cells, **kwargs,
        )
        if C_edc <= 0.0:
            raise ValueError(f"C_edc must be > 0, got {C_edc}")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")

        self._C_edc = C_edc
        self._tau = tau
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def C_edc(self) -> float:
        """EDC 常数。"""
        return self._C_edc

    @property
    def tau(self) -> float:
        """湍流特征时间尺度 [s]。"""
        return self._tau

    @property
    def rho(self) -> float:
        """流体密度 [kg/m^3]。"""
        return self._rho

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 EDC 燃烧源项。

        field 解释为燃料质量分数 Y_F。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        Y = field.to(device=device, dtype=dtype)

        rate = self._C_edc / self._tau * self._rho

        # Su = rate * Y (显式)
        su_val = rate * Y
        # Sp = -rate (隐式)
        sp_val = -rate

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            sp = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
            sp.scatter_(0, idx, sp_val)
        else:
            su = su_val
            sp = torch.full((n,), sp_val, device=device, dtype=dtype)

        matrix._source = matrix._source + su
        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"EDCSource(C_edc={self._C_edc}, tau={self._tau}, "
            f"rho={self._rho})"
        )
