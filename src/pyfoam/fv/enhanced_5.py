"""
增强 fvModels v5 — 辐射与多相流源项。

提供:

- :class:`FvDOMRadiationSource` — 有限体积离散坐标法 (FvDOM) 辐射源项
- :class:`SolarLoadSource` — 太阳辐射热负荷源项
- :class:`InterPhaseChangeModel` — 相间传质（空化/沸腾）模型

Usage::

    from pyfoam.fv.enhanced_5 import FvDOMRadiationSource

    model = FvDOMRadiationSource(a=0.5, T_ref=300.0)
    model.apply(energy_matrix, T_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "FvDOMRadiationSource",
    "SolarLoadSource",
    "InterPhaseChangeModel",
]


# ---------------------------------------------------------------------------
# FvDOMRadiationSource
# ---------------------------------------------------------------------------


@FvModel.register("fvDOMRadiationSource")
class FvDOMRadiationSource(FvModel):
    """有限体积离散坐标法 (FvDOM) 辐射源项。

    模拟参与介质中的辐射换热。源项由吸收和发射两部分组成::

        Q_rad = a * (4 * sigma * T^4 - G)

    其中:
    - ``a`` — 吸收系数 [1/m]
    - ``sigma`` — Stefan-Boltzmann 常数 [W/(m^2 K^4)]
    - ``G`` — 入射辐射强度 [W/m^2]（简化处理为参数）
    - ``T`` — 局部温度 [K]

    为改善收敛性，源项被半隐式线性化::

        Su = a * (4 * sigma * T_ref^4 - G)    (显式，围绕参考温度)
        Sp = 16 * a * sigma * T_ref^3          (隐式，关于 T 的导数)

    对应 OpenFOAM 的 ``fvDOM`` 辐射 fvModel。

    Parameters
    ----------
    a : float
        吸收系数 [1/m]。默认 ``0.5``。
    sigma_sb : float
        Stefan-Boltzmann 常数。默认 ``5.67e-8``。
    G : float
        入射辐射强度 [W/m^2]。默认 ``1000.0``。
    T_ref : float
        参考温度（线性化中心）[K]。默认 ``300.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = FvDOMRadiationSource(a=0.5, G=500.0, T_ref=400.0)
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        a: float = 0.5,
        sigma_sb: float = 5.67e-8,
        G: float = 1000.0,
        T_ref: float = 300.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            a=a, sigma_sb=sigma_sb, G=G, T_ref=T_ref, cells=cells, **kwargs,
        )
        if a < 0.0:
            raise ValueError(f"a must be >= 0, got {a}")
        if sigma_sb <= 0.0:
            raise ValueError(f"sigma_sb must be > 0, got {sigma_sb}")

        self._a = a
        self._sigma_sb = sigma_sb
        self._G = G
        self._T_ref = T_ref
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def a(self) -> float:
        """吸收系数 [1/m]。"""
        return self._a

    @property
    def sigma_sb(self) -> float:
        """Stefan-Boltzmann 常数。"""
        return self._sigma_sb

    @property
    def G(self) -> float:
        """入射辐射强度 [W/m^2]。"""
        return self._G

    @property
    def T_ref(self) -> float:
        """参考温度 [K]。"""
        return self._T_ref

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加 FvDOM 辐射源项。

        field 解释为温度场 T。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        T4_ref = self._T_ref ** 4

        # Su = a * (4 * sigma * T_ref^4 - G)
        su_val = self._a * (4.0 * self._sigma_sb * T4_ref - self._G)
        # Sp = 16 * a * sigma * T_ref^3
        sp_val = 16.0 * self._a * self._sigma_sb * self._T_ref ** 3

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
            f"FvDOMRadiationSource(a={self._a}, G={self._G}, "
            f"T_ref={self._T_ref})"
        )


# ---------------------------------------------------------------------------
# SolarLoadSource
# ---------------------------------------------------------------------------


@FvModel.register("solarLoadSource")
class SolarLoadSource(FvModel):
    """太阳辐射热负荷源项。

    模拟太阳辐射在流体域内的体积加热效应::

        Q_solar = eta * I_solar * f_vol

    其中:
    - ``eta`` — 吸收效率 [-] (0~1)
    - ``I_solar`` — 太阳辐射强度 [W/m^2]
    - ``f_vol`` — 体积分配因子 [1/m]

    源项纯显式施加，因为太阳辐射不依赖局部温度。

    对应 OpenFOAM 的 ``solarLoad`` fvModel。

    Parameters
    ----------
    eta : float
        太阳辐射吸收效率 [-]。默认 ``0.5``。
    I_solar : float
        太阳辐射强度 [W/m^2]。默认 ``1000.0``。
    f_vol : float
        体积分配因子 [1/m]。默认 ``1.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = SolarLoadSource(eta=0.8, I_solar=1000.0)
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        eta: float = 0.5,
        I_solar: float = 1000.0,
        f_vol: float = 1.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eta=eta, I_solar=I_solar, f_vol=f_vol, cells=cells, **kwargs,
        )
        if eta < 0.0 or eta > 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        if I_solar < 0.0:
            raise ValueError(f"I_solar must be >= 0, got {I_solar}")

        self._eta = eta
        self._I_solar = I_solar
        self._f_vol = f_vol
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def eta(self) -> float:
        """吸收效率 [-]。"""
        return self._eta

    @property
    def I_solar(self) -> float:
        """太阳辐射强度 [W/m^2]。"""
        return self._I_solar

    @property
    def f_vol(self) -> float:
        """体积分配因子 [1/m]。"""
        return self._f_vol

    @property
    def Q_solar(self) -> float:
        """太阳辐射体积加热率 [W/m^3] = eta * I_solar * f_vol。"""
        return self._eta * self._I_solar * self._f_vol

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加太阳辐射热负荷（纯显式）。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        q = self.Q_solar
        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, q)
        else:
            su[:] = q

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"SolarLoadSource(eta={self._eta}, I_solar={self._I_solar}, "
            f"f_vol={self._f_vol})"
        )


# ---------------------------------------------------------------------------
# InterPhaseChangeModel
# ---------------------------------------------------------------------------


@FvModel.register("interPhaseChange")
class InterPhaseChangeModel(FvModel):
    """相间传质（空化/沸腾）模型。

    模拟两相流中由于压力变化引起的相间质量传递::

        m_dot = -rho_v * (p - p_sat) / (0.5 * rho_l * U_inf^2)
                  * tanh(alpha / alpha_cav)

    简化形式（蒸气产生/凝结）::

        当 p < p_sat: m_dot > 0 (蒸发/空化)
        当 p > p_sat: m_dot < 0 (凝结)

    其中:
    - ``p_sat`` — 饱和压力 [Pa]
    - ``rho_v`` — 蒸气密度 [kg/m^3]
    - ``rho_l`` — 液体密度 [kg/m^3]
    - ``U_inf`` — 参考速度 [m/s]

    对应 OpenFOAM 的 ``interPhaseChange`` fvModel。

    Parameters
    ----------
    p_sat : float
        饱和压力 [Pa]。默认 ``2340.0`` (水在 20°C)。
    rho_v : float
        蒸气密度 [kg/m^3]。默认 ``0.0256``。
    rho_l : float
        液体密度 [kg/m^3]。默认 ``998.0``。
    U_inf : float
        参考速度 [m/s]。默认 ``1.0``。
    m_dot_max : float
        最大传质率（防止数值发散）[kg/(m^3 s)]。默认 ``1000.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。``None`` = 所有单元。

    Examples::

        model = InterPhaseChangeModel(p_sat=2340.0)
        model.apply(continuity_matrix, p_field)
    """

    def __init__(
        self,
        *,
        p_sat: float = 2340.0,
        rho_v: float = 0.0256,
        rho_l: float = 998.0,
        U_inf: float = 1.0,
        m_dot_max: float = 1000.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            p_sat=p_sat, rho_v=rho_v, rho_l=rho_l, U_inf=U_inf,
            m_dot_max=m_dot_max, cells=cells, **kwargs,
        )
        if rho_v <= 0.0:
            raise ValueError(f"rho_v must be > 0, got {rho_v}")
        if rho_l <= 0.0:
            raise ValueError(f"rho_l must be > 0, got {rho_l}")
        if m_dot_max <= 0.0:
            raise ValueError(f"m_dot_max must be > 0, got {m_dot_max}")

        self._p_sat = p_sat
        self._rho_v = rho_v
        self._rho_l = rho_l
        self._U_inf = U_inf
        self._m_dot_max = m_dot_max
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def p_sat(self) -> float:
        """饱和压力 [Pa]。"""
        return self._p_sat

    @property
    def rho_v(self) -> float:
        """蒸气密度 [kg/m^3]。"""
        return self._rho_v

    @property
    def rho_l(self) -> float:
        """液体密度 [kg/m^3]。"""
        return self._rho_l

    @property
    def U_inf(self) -> float:
        """参考速度 [m/s]。"""
        return self._U_inf

    @property
    def m_dot_max(self) -> float:
        """最大传质率。"""
        return self._m_dot_max

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加相间传质源项。

        field 解释为压力场 p。当 p < p_sat 时产生蒸气（质量源），
        当 p > p_sat 时凝结（质量汇）。
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        p = field.to(device=device, dtype=dtype)
        p_diff = (self._p_sat - p)

        # m_dot = rho_v * (p_sat - p) / (0.5 * rho_l * U_inf^2)
        denom = 0.5 * self._rho_l * self._U_inf ** 2
        denom_safe = max(denom, 1e-30)
        m_dot = self._rho_v * p_diff / denom_safe

        # 钳位到 [-m_dot_max, m_dot_max]
        m_dot = m_dot.clamp(-self._m_dot_max, self._m_dot_max)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, m_dot.gather(0, idx))
        else:
            su = m_dot

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"InterPhaseChangeModel(p_sat={self._p_sat}, "
            f"rho_v={self._rho_v}, rho_l={self._rho_l})"
        )
