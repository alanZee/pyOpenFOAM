"""
Chemistry models for reactive flow simulations.

Provides an abstract chemistry model framework and concrete ODE-based
and simplified reaction mechanism (SRM) solvers for integrating
chemical source terms in reacting-flow CFD.

**Models:**

- :class:`ChemistryModel` — abstract base with RTS registry
- :class:`ODEChemistrySolver` — ODE-based stiff chemistry integration
- :class:`SRMChemistrySolver` — Simplified Reaction Mechanism (tabulated/progress variable)

These models manage a set of species, reactions, and thermodynamic
data, providing the chemical source terms ``dY/dt`` and ``dT/dt``
needed by reacting-flow solvers.

Usage::

    from pyfoam.thermophysical.chemistry import ChemistryModel

    # ODE-based chemistry
    chem = ChemistryModel.create("ODE",
        species=["O2", "N2", "CH4"],
        reactions=[...],
        dt_max=1e-4,
    )
    dYdt, dTdt = chem.source(Y, T, p, rho, dt)

    # SRM chemistry (progress variable approach)
    chem = ChemistryModel.create("SRM",
        species=["fuel", "products"],
        progress_variable=1.0,
        table_resolution=50,
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.reaction import (
    ArrheniusReaction,
    ReactionRateModel,
    R_UNIVERSAL,
)

__all__ = [
    "ChemistryModel",
    "ODEChemistrySolver",
    "SRMChemistrySolver",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _to_tensor(
    value: torch.Tensor | float,
    ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """将标量转为张量，若已是张量则直接返回。"""
    if isinstance(value, torch.Tensor):
        return value
    device = ref.device if ref is not None else get_device()
    dtype = ref.dtype if ref is not None else get_default_dtype()
    return torch.tensor(value, dtype=dtype, device=device)


# ===================================================================
# 化学模型抽象基类
# ===================================================================


class ChemistryModel(ABC):
    """化学模型抽象基类。

    管理组分、反应和热力学数据，提供化学源项计算接口。
    子类必须实现 :meth:`source` 方法。

    RTS (Run-Time Selection) 注册表::

        @ChemistryModel.register("ODE")
        class ODEChemistrySolver(ChemistryModel):
            ...

        chem = ChemistryModel.create("ODE", species=["O2", "N2"])
    """

    _registry: ClassVar[dict[str, Type[ChemistryModel]]] = {}

    # ------------------------------------------------------------------
    # RTS 注册表
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """装饰器：将化学模型类注册到 *name*。"""

        def decorator(model_cls: Type[ChemistryModel]) -> Type[ChemistryModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Chemistry model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ChemistryModel:
        """工厂：根据注册名创建化学模型实例。

        Parameters
        ----------
        name : str
            注册的模型类型名。
        **kwargs
            转发给模型构造函数的参数。

        Raises
        ------
        KeyError
            如果 *name* 不在注册表中。
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown chemistry model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """返回已注册的模型类型名列表（排序）。"""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def source(
        self,
        Y: torch.Tensor,
        T: torch.Tensor | float,
        p: torch.Tensor | float,
        rho: torch.Tensor | float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算化学源项。

        Parameters
        ----------
        Y : torch.Tensor
            组分质量分数 ``(n_cells, n_species)``。
        T : torch.Tensor | float
            温度 (K) — 标量或 ``(n_cells,)`` 张量。
        p : torch.Tensor | float
            压力 (Pa)。
        rho : torch.Tensor | float
            密度 (kg/m³)。
        dt : float
            化学时间步长 (s)。

        Returns
        -------
        dYdt : torch.Tensor
            组分源项 ``(n_cells, n_species)`` (1/s)。
        dTdt : torch.Tensor
            温度源项 ``(n_cells,)`` (K/s)。
        """

    @property
    def species(self) -> list[str]:
        """返回组分名称列表。"""
        return []


# ===================================================================
# ODE 化学求解器
# ===================================================================


@ChemistryModel.register("ODE")
class ODEChemistrySolver(ChemistryModel):
    """ODE-based stiff chemistry solver.

    Integrates chemical kinetics using a stiff ODE solver.  Each
    reaction is modelled with an Arrhenius rate expression, and the
    system of ODEs for species mass fractions and temperature is
    integrated over the given time step.

    The source terms are computed as::

        dY_i/dt = sum_j (nu_ij * omega_j) / rho
        dT/dt   = -sum_j (Q_j * omega_j) / (rho * Cp)

    where ``omega_j`` is the reaction rate of reaction *j* and ``Q_j``
    is its heat release.

    For simplicity, this implementation uses a quasi-steady explicit
    integration with sub-cycling (repeated small steps) for stiff
    chemistry.

    Parameters
    ----------
    species : list[str]
        Species names (e.g. ``["O2", "N2", "CH4", "CO2"]``).
    reactions : list[dict], optional
        Reaction definitions, each with ``A``, ``Ea``, ``b``,
        ``reactants``, ``products``, and optional ``Q`` (heat release).
    dt_max : float
        Maximum sub-cycle step size (s). Default ``1e-5``.
    Cp : float
        Specific heat capacity [J/(kg K)]. Default ``1005.0``.

    Examples::

        chem = ODEChemistrySolver(
            species=["CH4", "O2", "CO2", "H2O"],
            reactions=[{
                "A": 1e10, "Ea": 8e4, "b": 0.0,
                "reactants": {"CH4": 1, "O2": 2},
                "products": {"CO2": 1, "H2O": 2},
                "Q": 8e5,
            }],
        )
        dYdt, dTdt = chem.source(Y, T, p, rho, dt)
    """

    def __init__(
        self,
        *,
        species: list[str] | None = None,
        reactions: list[dict[str, Any]] | None = None,
        dt_max: float = 1e-5,
        Cp: float = 1005.0,
        **kwargs: Any,
    ) -> None:
        self._species = species or ["A", "B"]
        self._dt_max = dt_max
        self._Cp = Cp

        # 构建反应列表
        self._reactions: list[dict[str, Any]] = []
        for rxn_def in (reactions or []):
            rate_model = ArrheniusReaction(
                A=rxn_def.get("A", 1.0),
                b=rxn_def.get("b", 0.0),
                Ea=rxn_def.get("Ea", 0.0),
            )
            self._reactions.append({
                "rate": rate_model,
                "reactants": rxn_def.get("reactants", {}),
                "products": rxn_def.get("products", {}),
                "Q": rxn_def.get("Q", 0.0),
            })

        # 组分名称到索引映射
        self._species_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._species)
        }

    @property
    def species(self) -> list[str]:
        """返回组分名称列表。"""
        return list(self._species)

    @property
    def n_species(self) -> int:
        """组分数。"""
        return len(self._species)

    @property
    def n_reactions(self) -> int:
        """反应数。"""
        return len(self._reactions)

    def source(
        self,
        Y: torch.Tensor,
        T: torch.Tensor | float,
        p: torch.Tensor | float,
        rho: torch.Tensor | float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 ODE 化学源项。

        使用亚循环（sub-cycling）显式积分化学动力学 ODE。

        Parameters
        ----------
        Y : torch.Tensor
            组分质量分数 ``(n_cells, n_species)``。
        T : torch.Tensor | float
            温度 (K)。
        p : torch.Tensor | float
            压力 (Pa) — 当前模型中未使用（恒压假设）。
        rho : torch.Tensor | float
            密度 (kg/m³)。
        dt : float
            化学时间步长 (s)。

        Returns
        -------
        dYdt : torch.Tensor
            组分源项 ``(n_cells, n_species)`` (1/s)。
        dTdt : torch.Tensor
            温度源项 ``(n_cells,)`` (K/s)。
        """
        T_t = _to_tensor(T)
        rho_t = _to_tensor(rho, ref=T_t)

        if T_t.dim() == 0:
            T_t = T_t.unsqueeze(0)
        if rho_t.dim() == 0:
            rho_t = rho_t.expand_as(T_t)

        n_cells = T_t.shape[0]
        n_sp = self.n_species
        device = T_t.device
        dtype = T_t.dtype

        # 初始化
        Y_cur = Y.to(device=device, dtype=dtype).clone()
        if Y_cur.dim() == 1:
            Y_cur = Y_cur.unsqueeze(0)
        T_cur = T_t.clone()

        Y_init = Y_cur.clone()
        T_init = T_cur.clone()

        # 亚循环积分
        remaining = dt
        while remaining > 1e-30:
            sub_dt = min(remaining, self._dt_max)

            # 计算每个反应的速率
            dY = torch.zeros_like(Y_cur)
            dT = torch.zeros_like(T_cur)

            for rxn in self._reactions:
                rate_model = rxn["rate"]
                reactants = rxn["reactants"]
                products = rxn["products"]
                Q = rxn["Q"]

                # 计算反应速率常数 k(T)
                k = rate_model.rate(T_cur)  # (n_cells,)

                # 计算反应物消耗率: omega = k * prod(Y_i^nu_i)
                omega = k.clone()
                for sp_name, stoich in reactants.items():
                    if sp_name in self._species_idx:
                        idx = self._species_idx[sp_name]
                        Y_sp = Y_cur[:, idx].clamp(min=0.0)
                        omega = omega * Y_sp.pow(stoich)

                # 更新组分
                for sp_name, stoich in reactants.items():
                    if sp_name in self._species_idx:
                        idx = self._species_idx[sp_name]
                        dY[:, idx] = dY[:, idx] - stoich * omega

                for sp_name, stoich in products.items():
                    if sp_name in self._species_idx:
                        idx = self._species_idx[sp_name]
                        dY[:, idx] = dY[:, idx] + stoich * omega

                # 温度源项: dT/dt = Q * omega / (rho * Cp)
                # 放热反应 (Q > 0) → 温度上升
                dT = dT + Q * omega / (rho_t * self._Cp)

            # 前向 Euler 子步
            Y_cur = Y_cur + sub_dt * dY
            T_cur = T_cur + sub_dt * dT

            # 钳制质量分数到 [0, 1]
            Y_cur = Y_cur.clamp(0.0, 1.0)
            T_cur = T_cur.clamp(min=200.0)

            remaining -= sub_dt

        # 返回平均源项
        dYdt = (Y_cur - Y_init) / dt
        dTdt = (T_cur - T_init) / dt

        return dYdt, dTdt

    def __repr__(self) -> str:
        return (
            f"ODEChemistrySolver(n_species={self.n_species}, "
            f"n_reactions={self.n_reactions}, dt_max={self._dt_max})"
        )


# ===================================================================
# SRM 化学求解器
# ===================================================================


@ChemistryModel.register("SRM")
class SRMChemistrySolver(ChemistryModel):
    """Simplified Reaction Mechanism (SRM) chemistry solver.

    Uses a progress variable approach to tabulate chemical source terms.
    Instead of integrating stiff ODEs, the SRM approach pre-computes
    the relationship between a progress variable (e.g. product mass
    fraction) and the chemical source terms, then looks up values
    during the simulation.

    The tabulation is done on a uniform grid of progress variable
    values.  Source terms are linearly interpolated during lookup.

    This is much faster than ODE integration for large mechanisms
    but requires an initial pre-processing step.

    The source terms for each species are computed as::

        dY_i/dt = table_Y_i(c) / tau_chem
        dT/dt   = table_T(c) / tau_chem

    where ``c`` is the progress variable (0 = unburned, 1 = burned)
    and ``tau_chem`` is the chemical time scale.

    Parameters
    ----------
    species : list[str]
        Species names.
    progress_variable : float
        Initial progress variable value (0 to 1). Default ``0.0``.
    table_resolution : int
        Number of tabulation points. Default ``100``.
    tau_chem : float
        Chemical time scale (s). Default ``1e-3``.
    T_unburned : float
        Unburned gas temperature (K). Default ``300.0``.
    T_burned : float
        Burned gas temperature (K). Default ``2000.0``.

    Examples::

        chem = SRMChemistrySolver(
            species=["fuel", "O2", "products"],
            tau_chem=1e-3,
            T_unburned=300.0,
            T_burned=2000.0,
        )
        dYdt, dTdt = chem.source(Y, T, p, rho, dt)
    """

    def __init__(
        self,
        *,
        species: list[str] | None = None,
        progress_variable: float = 0.0,
        table_resolution: int = 100,
        tau_chem: float = 1e-3,
        T_unburned: float = 300.0,
        T_burned: float = 2000.0,
        **kwargs: Any,
    ) -> None:
        self._species = species or ["fuel", "products"]
        self._progress_variable = progress_variable
        self._table_resolution = table_resolution
        self._tau_chem = tau_chem
        self._T_unburned = T_unburned
        self._T_burned = T_burned

        # 构建查找表
        self._build_table()

    def _build_table(self) -> None:
        """构建进度变量 → 源项查找表。"""
        n = self._table_resolution
        n_sp = len(self._species)

        # 进度变量网格 [0, 1]
        self._c_grid = torch.linspace(0.0, 1.0, n)

        # 简化的源项表：假设 "fuel" 组分消耗、"products" 组分生成
        # 其余组分不参与进度变量映射
        self._dY_table = torch.zeros(n, n_sp)
        self._dT_table = torch.zeros(n)

        fuel_idx = None
        product_idx = None
        for i, name in enumerate(self._species):
            if name.lower() in ("fuel", "ch4", "c8h18", "hydrogen", "h2"):
                fuel_idx = i
            elif name.lower() in ("products", "product", "prod", "co2", "h2o"):
                product_idx = i

        for j in range(n):
            c = self._c_grid[j].item()
            # Sigmoid-like progress from unburned (c=0) to burned (c=1)
            # Source term drives c toward 1
            dc_dt = c * (1.0 - c) / self._tau_chem
            self._dT_table[j] = (self._T_burned - self._T_unburned) * dc_dt

            if fuel_idx is not None:
                self._dY_table[j, fuel_idx] = -dc_dt
            if product_idx is not None:
                self._dY_table[j, product_idx] = dc_dt

    @property
    def species(self) -> list[str]:
        """返回组分名称列表。"""
        return list(self._species)

    @property
    def tau_chem(self) -> float:
        """化学时间尺度 (s)。"""
        return self._tau_chem

    @property
    def table_resolution(self) -> int:
        """查找表分辨率。"""
        return self._table_resolution

    def _lookup(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """线性插值查找源项。

        Parameters
        ----------
        c : torch.Tensor
            进度变量值 ``(n_cells,)``，截断到 [0, 1]。

        Returns
        -------
        dY : torch.Tensor
            组分源项 ``(n_cells, n_species)``。
        dT : torch.Tensor
            温度源项 ``(n_cells,)``。
        """
        c_clamped = c.clamp(0.0, 1.0)

        # 索引到查找表
        n = self._table_resolution
        idx_f = c_clamped * (n - 1)
        idx_lo = idx_f.long().clamp(0, n - 2)
        idx_hi = idx_lo + 1
        w = (idx_f - idx_lo.float()).clamp(0.0, 1.0)

        # 线性插值
        dY_lo = self._dY_table[idx_lo]  # (n_cells, n_species)
        dY_hi = self._dY_table[idx_hi]
        dY = dY_lo + w.unsqueeze(-1) * (dY_hi - dY_lo)

        dT_lo = self._dT_table[idx_lo]
        dT_hi = self._dT_table[idx_hi]
        dT = dT_lo + w * (dT_hi - dT_lo)

        return dY, dT

    def source(
        self,
        Y: torch.Tensor,
        T: torch.Tensor | float,
        p: torch.Tensor | float,
        rho: torch.Tensor | float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 SRM 化学源项（查找表插值）。

        进度变量 *c* 从当前温度推断::

            c = (T - T_unburned) / (T_burned - T_unburned)

        Parameters
        ----------
        Y : torch.Tensor
            组分质量分数 ``(n_cells, n_species)``。
        T : torch.Tensor | float
            温度 (K)。
        p : torch.Tensor | float
            压力 (Pa) — 未使用。
        rho : torch.Tensor | float
            密度 (kg/m³) — 未使用。
        dt : float
            时间步长 (s) — 未使用（源项已包含时间尺度）。

        Returns
        -------
        dYdt : torch.Tensor
            组分源项 ``(n_cells, n_species)`` (1/s)。
        dTdt : torch.Tensor
            温度源项 ``(n_cells,)`` (K/s)。
        """
        T_t = _to_tensor(T)
        if T_t.dim() == 0:
            T_t = T_t.unsqueeze(0)

        # 从温度推断进度变量
        T_range = self._T_burned - self._T_unburned
        if abs(T_range) < 1e-10:
            c = torch.zeros_like(T_t)
        else:
            c = (T_t - self._T_unburned) / T_range
            c = c.clamp(0.0, 1.0)

        dYdt, dTdt = self._lookup(c)

        # 确保输出形状正确
        device = T_t.device
        dtype = T_t.dtype
        dYdt = dYdt.to(device=device, dtype=dtype)
        dTdt = dTdt.to(device=device, dtype=dtype)

        return dYdt, dTdt

    def __repr__(self) -> str:
        return (
            f"SRMChemistrySolver(n_species={len(self._species)}, "
            f"tau_chem={self._tau_chem}, "
            f"table_resolution={self._table_resolution})"
        )
