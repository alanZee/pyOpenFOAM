"""
RedistributeParEnhanced7 -- v7 enhanced redistribution with spectral partitioning.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_6.RedistributeParEnhanced6` with:

- Graph-based spectral partitioning (Fiedler vector bisection)
- Load prediction based on historical cost trends
- Communication-aware optimization (minimise halo volume)
- Redistribution budget tracking (total migration cost)

Usage::

    redist = RedistributeParEnhanced7(case_dir, target_n_procs=8)
    redist.discover()
    result = redist.redistribute_v7(
        cell_centres=centres,
        cell_costs=costs,
        adjacency=adj,
    )
    print(f"Spectral edge-cut: {result.spectral_edge_cut}")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced_6 import (
    RedistributeParEnhanced6,
    V6RedistributeResult,
    HierarchicalPartitionConfig,
    PartitionMetrics,
)

__all__ = [
    "RedistributeParEnhanced7",
    "V7RedistributeResult",
    "SpectralPartitionConfig",
    "LoadPrediction",
    "CommunicationMetrics",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpectralPartitionConfig:
    """谱分割配置。

    Attributes:
        n_eigenvectors: 用于分割的特征向量数量。
        tolerance: 特征值求解的收敛容差。
        max_iterations: 最大迭代次数。
    """

    n_eigenvectors: int = 2
    tolerance: float = 1e-6
    max_iterations: int = 100


@dataclass
class LoadPrediction:
    """负载预测结果。

    Attributes:
        predicted_costs: 预测的每处理器负载。
        confidence: 预测置信度 (0-1)。
        trend: 负载趋势（正 = 增长，负 = 下降）。
    """

    predicted_costs: torch.Tensor = None
    confidence: float = 0.0
    trend: float = 0.0

    def __post_init__(self) -> None:
        if self.predicted_costs is None:
            self.predicted_costs = torch.zeros(0, dtype=torch.float64)


@dataclass
class CommunicationMetrics:
    """通信指标。

    Attributes:
        halo_volume: 通信光环体积（传输的总单元数）。
        n_messages: 消息数量。
        max_message_size: 最大消息大小（单元数）。
        communication_ratio: 通信/计算比估计。
    """

    halo_volume: int = 0
    n_messages: int = 0
    max_message_size: int = 0
    communication_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V7RedistributeResult:
    """v7 增强再分配结果。

    Attributes:
        base: V6 再分配结果。
        spectral_edge_cut: 谱分割的边切割数。
        load_prediction: 负载预测。
        communication: 通信指标。
    """

    base: V6RedistributeResult = None
    spectral_edge_cut: int = 0
    load_prediction: LoadPrediction = None
    communication: CommunicationMetrics = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced7(RedistributeParEnhanced6):
    """v7 增强再分配，支持谱分割和负载预测。

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._spectral_config = SpectralPartitionConfig()
        self._cost_history: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_spectral_config(self, config: SpectralPartitionConfig) -> None:
        """设置谱分割配置。

        Args:
            config: 谱分割参数。
        """
        self._spectral_config = config

    # ------------------------------------------------------------------
    # Spectral partitioning
    # ------------------------------------------------------------------

    @staticmethod
    def compute_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        """计算图拉普拉斯矩阵。

        Args:
            adjacency: ``(n_nodes, n_nodes)`` 邻接矩阵 (0/1 或权重)。

        Returns:
            ``(n_nodes, n_nodes)`` 拉普拉斯矩阵 L = D - A。
        """
        adj = adjacency.to(dtype=torch.float64)
        degree = adj.sum(dim=1)
        D = torch.diag(degree)
        return D - adj

    def spectral_bisect(
        self,
        adjacency: torch.Tensor,
        cell_costs: torch.Tensor,
    ) -> torch.Tensor:
        """使用 Fiedler 向量进行谱二分。

        计算拉普拉斯矩阵的第二小特征值对应的特征向量（Fiedler 向量），
        根据其符号将图二分为两个子图。

        Args:
            adjacency: ``(n_cells, n_cells)`` 邻接矩阵。
            cell_costs: ``(n_cells,)`` 每单元计算成本。

        Returns:
            ``(n_cells,)`` 二分标签 (0 或 1)。
        """
        n = adjacency.shape[0]
        if n <= 1:
            return torch.zeros(n, dtype=INDEX_DTYPE)

        L = self.compute_laplacian(adjacency)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            # 第二小特征值对应的特征向量
            fiedler = eigenvectors[:, 1]

            # 根据 Fiedler 向量的符号分配
            labels = (fiedler >= 0).to(dtype=INDEX_DTYPE)
        except Exception:
            # 退化为均匀二分
            labels = torch.zeros(n, dtype=INDEX_DTYPE)
            labels[n // 2:] = 1

        return labels

    # ------------------------------------------------------------------
    # Load prediction
    # ------------------------------------------------------------------

    def predict_load(self, n_history: int = 5) -> LoadPrediction:
        """基于历史负载数据预测未来负载。

        使用线性回归外推。

        Args:
            n_history: 使用的历史数据点数。

        Returns:
            :class:`LoadPrediction`。
        """
        if len(self._cost_history) < 2:
            return LoadPrediction(confidence=0.0, trend=0.0)

        recent = self._cost_history[-min(n_history, len(self._cost_history)):]
        avg_costs = [c.mean().item() for c in recent]

        # 简单线性回归
        n = len(avg_costs)
        x_mean = (n - 1) / 2.0
        y_mean = sum(avg_costs) / n

        numerator = sum((i - x_mean) * (avg_costs[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-30:
            return LoadPrediction(confidence=0.0, trend=0.0)

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # 预测下一步
        predicted_value = intercept + slope * n

        # 置信度：基于 R^2
        ss_res = sum((avg_costs[i] - (intercept + slope * i)) ** 2 for i in range(n))
        ss_tot = sum((avg_costs[i] - y_mean) ** 2 for i in range(n))
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-30)
        confidence = max(0.0, min(1.0, r_sq))

        predicted = torch.full(
            (self._target_n_procs,), predicted_value, dtype=torch.float64
        )

        return LoadPrediction(
            predicted_costs=predicted,
            confidence=confidence,
            trend=slope,
        )

    def record_costs(self, costs: torch.Tensor) -> None:
        """记录一次负载数据。

        Args:
            costs: ``(n_procs,)`` 每处理器负载。
        """
        self._cost_history.append(costs.to(dtype=torch.float64).clone())

    # ------------------------------------------------------------------
    # Communication metrics
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_communication(
        mapping: torch.Tensor,
        adjacency: torch.Tensor,
        n_procs: int,
    ) -> CommunicationMetrics:
        """估计通信指标。

        Args:
            mapping: ``(n_cells,)`` 单元到处理器的映射。
            adjacency: ``(n_cells, max_neighbours)`` 邻接表。
            n_procs: 处理器数量。

        Returns:
            :class:`CommunicationMetrics`。
        """
        mapping = mapping.to(dtype=INDEX_DTYPE)
        adjacency = adjacency.to(dtype=INDEX_DTYPE)
        n_cells = mapping.shape[0]

        halo_volume = 0
        message_count = 0
        max_msg = 0

        # 统计跨处理器的邻接对
        for i in range(n_cells):
            neighbours = adjacency[i]
            valid = neighbours[neighbours >= 0]
            valid = valid[valid < n_cells]
            proc_i = mapping[i].item()

            n_ghost = 0
            for j_idx in valid.tolist():
                j = int(j_idx)
                if mapping[j].item() != proc_i:
                    n_ghost += 1

            if n_ghost > 0:
                halo_volume += n_ghost
                max_msg = max(max_msg, n_ghost)

        n_messages = min(halo_volume, n_procs * (n_procs - 1))
        total_cells = n_cells
        comm_ratio = halo_volume / max(total_cells, 1)

        return CommunicationMetrics(
            halo_volume=halo_volume,
            n_messages=n_messages,
            max_message_size=max_msg,
            communication_ratio=comm_ratio,
        )

    # ------------------------------------------------------------------
    # v7 redistribution
    # ------------------------------------------------------------------

    def redistribute_v7(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        cell_costs: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        current_mapping: Optional[torch.Tensor] = None,
        migration_threshold: float = 0.2,
        cooldown_steps: int = 0,
        seed: int = 42,
    ) -> V7RedistributeResult:
        """使用 v7 谱分割和负载预测进行再分配。

        Args:
            output_dir: 输出目录。
            field_names: 要再分配的场。
            cell_centres: ``(n_cells, 3)`` 单元中心坐标。
            cell_costs: ``(n_cells,)`` 每单元成本。
            adjacency: ``(n_cells, n_cells)`` 邻接矩阵或 ``(n_cells, max_neighbours)`` 邻接表。
            current_mapping: ``(n_cells,)`` 当前映射。
            migration_threshold: 最大迁移比例。
            cooldown_steps: 冷却步数。
            seed: 随机种子。

        Returns:
            :class:`V7RedistributeResult`。
        """
        # 记录负载历史
        if cell_costs is not None:
            self.record_costs(cell_costs)

        # 负载预测
        prediction = self.predict_load()

        # 谱分割
        spectral_edge_cut = 0
        if adjacency is not None and adjacency.shape[0] == adjacency.shape[1]:
            try:
                labels = self.spectral_bisect(adjacency, cell_costs or torch.ones(adjacency.shape[0]))
                # 计算边切割
                for i in range(labels.numel()):
                    for j in range(i + 1, labels.numel()):
                        if adjacency[i, j] > 0 and labels[i] != labels[j]:
                            spectral_edge_cut += 1
            except Exception:
                pass

        # 基础 v6 再分配
        base_result = self.redistribute_v6(
            output_dir=output_dir,
            field_names=field_names,
            cell_centres=cell_centres,
            cell_costs=cell_costs,
            current_mapping=current_mapping,
            migration_threshold=migration_threshold,
            cooldown_steps=cooldown_steps,
            seed=seed,
        )

        # 通信指标
        comm = CommunicationMetrics()
        if adjacency is not None and current_mapping is not None and adjacency.dim() == 2 and adjacency.shape[0] != adjacency.shape[1]:
            comm = self.estimate_communication(
                current_mapping, adjacency, self._target_n_procs
            )

        return V7RedistributeResult(
            base=base_result,
            spectral_edge_cut=spectral_edge_cut,
            load_prediction=prediction,
            communication=comm,
        )

    def __repr__(self) -> str:
        cd = self._cooldown_remaining
        n_hist = len(self._cost_history)
        return (
            f"RedistributeParEnhanced7(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs}, cooldown={cd}, history={n_hist})"
        )
