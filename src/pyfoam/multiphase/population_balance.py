"""
群体平衡方程 (Population Balance Equation, PBE) 求解器。

使用离散类方法（Method of Classes）求解液滴/气泡的尺寸分布演化。
将连续尺寸空间离散为 N 个尺寸区间（bins），每个 bin 追踪数密度 n_i。

群体平衡方程：

    ∂n(v)/∂t + ∇·(U n(v)) = B_coal - D_coal + B_break - D_break

其中：
    - n(v): 数密度分布函数
    - B_coal, D_coal: 聚并的生成和消亡项
    - B_break, D_break: 破碎的生成和消亡项

离散类方法将尺寸空间 [v_min, v_max] 离散为 N 个区间，
每个区间 i 的代表体积为 v_i，数密度为 n_i。

支持的子模型：
    - **聚并**: 常数聚并核、Shear 聚并核（与剪切率成正比）
    - **破碎**: 常数破碎率、Weber 破碎模型

Usage::

    from pyfoam.multiphase.population_balance import (
        PopulationBalanceModel, PBEBin, ConstantCoalescence, ShearBreakup,
    )

    # 定义尺寸区间
    bins = PBEBin.create_geometric_bins(
        v_min=1e-12, v_max=1e-6, n_bins=10, ratio=2.0,
    )

    # 创建 PBE 模型
    pbe = PopulationBalanceModel(
        mesh, bins,
        coalescence=ConstantCoalescence(C_coal=1e-3),
        breakup=ShearBreakup(C_break=0.1),
    )

    pbe.advance(dt=1e-4)
    d32 = pbe.sauter_mean_diameter()  # Sauter 平均直径
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "PopulationBalanceModel",
    "PBEBin",
    "CoalescenceModel",
    "BreakupModel",
    "ConstantCoalescence",
    "ShearCoalescence",
    "ConstantBreakup",
    "WeberBreakup",
    "ShearBreakup",
]

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ---------------------------------------------------------------------------
# 尺寸区间定义
# ---------------------------------------------------------------------------


@dataclass
class PBEBin:
    """群体平衡方程的离散尺寸区间。

    Attributes:
        v_center: 区间中心体积 (m³).
        v_lower: 区间下界体积 (m³).
        v_upper: 区间上界体积 (m³).
        d: 区间代表直径 (m)，d = (6 v / π)^(1/3).
        dv: 区间宽度 (m³).
    """

    v_center: float
    v_lower: float
    v_upper: float
    d: float
    dv: float

    @staticmethod
    def create_geometric_bins(
        v_min: float,
        v_max: float,
        n_bins: int,
        ratio: float = 2.0,
    ) -> list[PBEBin]:
        """创建几何级数尺寸区间。

        区间边界按等比级数划分：v_{i+1} / v_i = ratio。

        Parameters
        ----------
        v_min : float
            最小体积 (m³).
        v_max : float
            最大体积 (m³).
        n_bins : int
            区间数量.
        ratio : float
            相邻区间体积比. 默认 2.0.

        Returns:
            PBEBin 列表.
        """
        bins: list[PBEBin] = []
        v_lower = v_min
        for i in range(n_bins):
            v_upper = v_lower * ratio
            if i == n_bins - 1:
                v_upper = v_max
            v_center = (v_lower + v_upper) / 2.0
            d = (6.0 * v_center / 3.141592653589793) ** (1.0 / 3.0)
            bins.append(PBEBin(
                v_center=v_center,
                v_lower=v_lower,
                v_upper=v_upper,
                d=d,
                dv=v_upper - v_lower,
            ))
            v_lower = v_upper
        return bins


# ---------------------------------------------------------------------------
# 聚并和破碎子模型协议
# ---------------------------------------------------------------------------


@runtime_checkable
class CoalescenceModel(Protocol):
    """聚并子模型协议."""

    def coalescence_rate(
        self,
        v_i: float,
        v_j: float,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """计算 bin i 和 bin j 之间的聚并速率.

        Parameters
        ----------
        v_i, v_j : float
            bin 体积 (m³).
        n_i, n_j : torch.Tensor
            两个 bin 的数密度 ``(n_cells,)``.
        gamma : torch.Tensor
            剪切率 ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` 聚并速率.
        """
        ...


@runtime_checkable
class BreakupModel(Protocol):
    """破碎子模型协议."""

    def breakup_rate(
        self,
        v_i: float,
        n_i: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """计算 bin i 的破碎速率.

        Parameters
        ----------
        v_i : float
            bin 体积 (m³).
        n_i : torch.Tensor
            bin 数密度 ``(n_cells,)``.
        gamma : torch.Tensor
            剪切率 ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` 破碎速率.
        """
        ...

    def daughter_distribution(
        self,
        v_parent: float,
        v_daughter: float,
    ) -> float:
        """子液滴/气泡体积分布函数.

        Parameters
        ----------
        v_parent : float
            母体体积.
        v_daughter : float
            子体体积.

        Returns:
            分布权重.
        """
        ...


# ---------------------------------------------------------------------------
# 聚并模型实现
# ---------------------------------------------------------------------------


class ConstantCoalescence:
    """常数聚并核.

    β(v_i, v_j) = C_coal

    Parameters
    ----------
    C_coal : float
        聚并核常数 (m³/s).  默认 1e-3.
    """

    def __init__(self, C_coal: float = 1e-3) -> None:
        self.C_coal = C_coal

    def coalescence_rate(
        self,
        v_i: float,
        v_j: float,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """常数聚并核速率: β = C_coal."""
        return self.C_coal * n_i * n_j


class ShearCoalescence:
    """剪切聚并核.

    β(v_i, v_j) = C_coal * γ * (r_i + r_j)³

    其中 r_i 为 bin i 的等效半径。

    Parameters
    ----------
    C_coal : float
        聚并核系数.  默认 1.0.
    """

    def __init__(self, C_coal: float = 1.0) -> None:
        self.C_coal = C_coal

    def coalescence_rate(
        self,
        v_i: float,
        v_j: float,
        n_i: torch.Tensor,
        n_j: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """剪切聚并核速率."""
        r_i = (3.0 * v_i / (4.0 * 3.141592653589793)) ** (1.0 / 3.0)
        r_j = (3.0 * v_j / (4.0 * 3.141592653589793)) ** (1.0 / 3.0)
        beta = self.C_coal * gamma * (r_i + r_j) ** 3
        return beta * n_i * n_j


# ---------------------------------------------------------------------------
# 破碎模型实现
# ---------------------------------------------------------------------------


class ConstantBreakup:
    """常数破碎率.

    Parameters
    ----------
    C_break : float
        破碎率常数 (1/s).  默认 0.1.
    n_daughters : int
        每次破碎产生的子体数量.  默认 2.
    """

    def __init__(self, C_break: float = 0.1, n_daughters: int = 2) -> None:
        self.C_break = C_break
        self.n_daughters = n_daughters

    def breakup_rate(
        self,
        v_i: float,
        n_i: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """常数破碎率: g = C_break."""
        return self.C_break * n_i

    def daughter_distribution(
        self,
        v_parent: float,
        v_daughter: float,
    ) -> float:
        """均匀子体分布: 每个子体获得 1/n_daughters 的母体体积."""
        v_each = v_parent / self.n_daughters
        if abs(v_daughter - v_each) < v_each * 0.01:
            return 1.0 / self.n_daughters
        return 0.0


class WeberBreakup:
    """Weber 数破碎模型.

    破碎率与局部 Weber 数相关:

        g = C_break * γ * (We / We_cr - 1)^0.5    if We > We_cr
        g = 0                                       otherwise

    We = ρ γ² d³ / σ

    Parameters
    ----------
    C_break : float
        破碎率系数.  默认 0.1.
    rho : float
        连续相密度 (kg/m³).  默认 1000.0.
    sigma : float
        表面张力 (N/m).  默认 0.07.
    We_cr : float
        临界 Weber 数.  默认 12.0.
    n_daughters : int
        每次破碎产生的子体数量.  默认 2.
    """

    def __init__(
        self,
        C_break: float = 0.1,
        rho: float = 1000.0,
        sigma: float = 0.07,
        We_cr: float = 12.0,
        n_daughters: int = 2,
    ) -> None:
        self.C_break = C_break
        self.rho = rho
        self.sigma = sigma
        self.We_cr = We_cr
        self.n_daughters = n_daughters

    def breakup_rate(
        self,
        v_i: float,
        n_i: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Weber 数破碎率."""
        d = (6.0 * v_i / 3.141592653589793) ** (1.0 / 3.0)
        We = self.rho * gamma.pow(2) * d ** 3 / max(self.sigma, _EPS)
        we_ratio = (We / self.We_cr - 1.0).clamp(min=0.0)
        rate = self.C_break * gamma * torch.sqrt(we_ratio)
        return rate * n_i

    def daughter_distribution(
        self,
        v_parent: float,
        v_daughter: float,
    ) -> float:
        """均匀子体分布."""
        v_each = v_parent / self.n_daughters
        if abs(v_daughter - v_each) < v_each * 0.01:
            return 1.0 / self.n_daughters
        return 0.0


class ShearBreakup:
    """剪切破碎模型.

    破碎率与剪切率成正比:

        g = C_break * γ

    Parameters
    ----------
    C_break : float
        破碎率系数.  默认 0.1.
    n_daughters : int
        每次破碎产生的子体数量.  默认 2.
    """

    def __init__(self, C_break: float = 0.1, n_daughters: int = 2) -> None:
        self.C_break = C_break
        self.n_daughters = n_daughters

    def breakup_rate(
        self,
        v_i: float,
        n_i: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """剪切破碎率: g = C_break * γ."""
        return self.C_break * gamma * n_i

    def daughter_distribution(
        self,
        v_parent: float,
        v_daughter: float,
    ) -> float:
        """均匀子体分布."""
        v_each = v_parent / self.n_daughters
        if abs(v_daughter - v_each) < v_each * 0.01:
            return 1.0 / self.n_daughters
        return 0.0


# ---------------------------------------------------------------------------
# 群体平衡方程求解器
# ---------------------------------------------------------------------------


class PopulationBalanceModel:
    """群体平衡方程求解器（离散类方法）.

    使用离散类方法 (Method of Classes) 求解液滴/气泡的尺寸分布演化。
    将连续尺寸空间离散为 N 个 bin，每个 bin 追踪数密度 n_i。

    Parameters
    ----------
    mesh : Any
        有限体积网格.
    bins : list[PBEBin]
        尺寸区间列表.
    coalescence : CoalescenceModel | None
        聚并子模型.  默认 None (无聚并).
    breakup : BreakupModel | None
        破碎子模型.  默认 None (无破碎).
    alpha : torch.Tensor | None
        初始体积分数 ``(n_cells,)``，用于归一化数密度。
        默认 None（假设为 1）.

    Attributes
    ----------
    n_fields : list[torch.Tensor]
        各 bin 的数密度场 ``[(n_cells,), ...]``.
    """

    def __init__(
        self,
        mesh: Any,
        bins: list[PBEBin],
        coalescence: CoalescenceModel | None = None,
        breakup: BreakupModel | None = None,
        alpha: torch.Tensor | None = None,
    ) -> None:
        self._mesh = mesh
        self._bins = bins
        self._n_bins = len(bins)
        self._device = get_device()
        self._dtype = get_default_dtype()

        if self._n_bins < 1:
            raise ValueError("至少需要一个尺寸区间")

        self._coalescence = coalescence
        self._breakup = breakup

        n_cells = mesh.n_cells

        # 初始化数密度场
        self._n_fields: list[torch.Tensor] = []
        for i, b in enumerate(bins):
            # 均匀分布的初始数密度
            n0 = 1.0 / (self._n_bins * max(b.dv, _EPS))
            nf = torch.full(
                (n_cells,), n0, device=self._device, dtype=self._dtype,
            )
            self._n_fields.append(nf)

        # 剪切率场（默认零，由外部设置）
        self._gamma = torch.zeros(
            n_cells, device=self._device, dtype=self._dtype,
        )

        # 体积分数（可选）
        if alpha is not None:
            self._alpha = alpha.to(device=self._device, dtype=self._dtype)
        else:
            self._alpha = torch.ones(
                n_cells, device=self._device, dtype=self._dtype,
            )

        # 时间步长计数
        self._time = 0.0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def bins(self) -> list[PBEBin]:
        """尺寸区间列表."""
        return self._bins

    @property
    def n_bins(self) -> int:
        """区间数量."""
        return self._n_bins

    @property
    def n_fields(self) -> list[torch.Tensor]:
        """各 bin 的数密度场."""
        return self._n_fields

    @property
    def gamma(self) -> torch.Tensor:
        """剪切率场 ``(n_cells,)``."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor) -> None:
        self._gamma = value.to(device=self._device, dtype=self._dtype)

    @property
    def time(self) -> float:
        """当前模拟时间."""
        return self._time

    # ------------------------------------------------------------------
    # 求解方法
    # ------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """推进群体平衡方程一个时间步.

        使用显式 Euler 方法推进所有 bin 的数密度场，
        同时考虑聚并和破碎的源项。

        Parameters
        ----------
        dt : float
            时间步长 (s).
        """
        n_cells = self._mesh.n_cells

        # 源项：每个 bin 的净变化率
        sources = [
            torch.zeros(n_cells, device=self._device, dtype=self._dtype)
            for _ in range(self._n_bins)
        ]

        # --- 聚并源项 ---
        if self._coalescence is not None:
            self._compute_coalescence_sources(sources)

        # --- 破碎源项 ---
        if self._breakup is not None:
            self._compute_breakup_sources(sources)

        # --- 显式 Euler 推进 ---
        for i in range(self._n_bins):
            self._n_fields[i] = (self._n_fields[i] + dt * sources[i]).clamp(
                min=0.0,
            )

        self._time += dt

    def _compute_coalescence_sources(
        self,
        sources: list[torch.Tensor],
    ) -> None:
        """计算聚并源项.

        聚并导致:
        - bin i 的数密度减少: -∑_j β(v_i, v_j) n_i n_j
        - 产物 bin k 的数密度增加: +β(v_i, v_j) n_i n_j（若 v_k = v_i + v_j）
        """
        coal = self._coalescence
        gamma = self._gamma

        for i in range(self._n_bins):
            for j in range(i, self._n_bins):
                rate_ij = coal.coalescence_rate(
                    self._bins[i].v_center,
                    self._bins[j].v_center,
                    self._n_fields[i],
                    self._n_fields[j],
                    gamma,
                )

                # 消亡项: bin i 和 bin j 各自减少
                if i == j:
                    sources[i] -= rate_ij
                else:
                    sources[i] -= rate_ij
                    sources[j] -= rate_ij

                # 生成项: 聚并产物 v_k = v_i + v_j
                v_sum = self._bins[i].v_center + self._bins[j].v_center
                k = self._find_bin(v_sum)
                if k is not None:
                    sources[k] += rate_ij

    def _compute_breakup_sources(
        self,
        sources: list[torch.Tensor],
    ) -> None:
        """计算破碎源项.

        破碎导致:
        - bin i 的数密度减少: -g(v_i) n_i
        - 子体 bin j 的数密度增加: +g(v_i) n_i * b(v_i, v_j) * n_daughters
        """
        breakup = self._breakup
        gamma = self._gamma

        for i in range(self._n_bins):
            v_i = self._bins[i].v_center

            # 消亡项
            breakup_i = breakup.breakup_rate(v_i, self._n_fields[i], gamma)
            sources[i] -= breakup_i

            # 生成项: 分配到子体 bin
            for j in range(self._n_bins):
                v_j = self._bins[j].v_center
                b_ij = breakup.daughter_distribution(v_i, v_j)
                if b_ij > 0:
                    sources[j] += breakup_i * b_ij

    def _find_bin(self, v: float) -> int | None:
        """找到包含体积 v 的 bin 索引.

        如果 v 超出最大 bin 范围，分配到最大 bin。
        如果 v 小于最小 bin 范围，返回 None。
        """
        for k, b in enumerate(self._bins):
            if b.v_lower <= v < b.v_upper:
                return k
        # 超出范围：分配到最大 bin
        if v >= self._bins[-1].v_upper:
            return self._n_bins - 1
        return None

    # ------------------------------------------------------------------
    # 统计量
    # ------------------------------------------------------------------

    def total_number_density(self) -> torch.Tensor:
        """总数密度 N₀ = Σ n_i ``(n_cells,)``."""
        result = torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )
        for i in range(self._n_bins):
            result += self._n_fields[i]
        return result

    def total_volume_fraction(self) -> torch.Tensor:
        """总分散相体积分数 α = Σ v_i n_i ``(n_cells,)``."""
        result = torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )
        for i, b in enumerate(self._bins):
            result += b.v_center * self._n_fields[i]
        return result

    def mean_diameter(self) -> torch.Tensor:
        """数密度加权平均直径 d₁₀ = Σ d_i n_i / Σ n_i ``(n_cells,)``."""
        N0 = self.total_number_density().clamp(min=_EPS)
        d_sum = torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )
        for i, b in enumerate(self._bins):
            d_sum += b.d * self._n_fields[i]
        return d_sum / N0

    def sauter_mean_diameter(self) -> torch.Tensor:
        """Sauter 平均直径 d₃₂ = Σ v_i n_i / Σ (v_i/d_i) n_i ``(n_cells,)``.

        Sauter 平均直径定义为面积平均直径:
            d₃₂ = (Σ d_i³ n_i) / (Σ d_i² n_i)
        """
        num = torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )
        den = torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )
        for i, b in enumerate(self._bins):
            d = b.d
            num += d ** 3 * self._n_fields[i]
            den += d ** 2 * self._n_fields[i]
        return num / den.clamp(min=_EPS)

    def size_distribution(self) -> tuple[list[float], torch.Tensor]:
        """返回尺寸分布.

        Returns:
            (diameters, distribution)：
            - diameters: 各 bin 代表直径列表.
            - distribution: ``(n_bins, n_cells)`` 各 bin 的数密度.
        """
        diameters = [b.d for b in self._bins]
        dist = torch.stack(self._n_fields, dim=0)
        return diameters, dist

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        coal_type = type(self._coalescence).__name__ if self._coalescence else "None"
        break_type = type(self._breakup).__name__ if self._breakup else "None"
        return (
            f"PopulationBalanceModel("
            f"n_bins={self._n_bins}, "
            f"coalescence={coal_type}, "
            f"breakup={break_type}, "
            f"n_cells={self._mesh.n_cells})"
        )
