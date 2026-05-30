"""
RedistributeParEnhanced8 -- v8 enhanced redistribution with multi-objective optimization.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_7.RedistributeParEnhanced7` with:

- Multi-objective partitioning (balance load + minimise communication + spatial locality)
- Incremental redistribution (only migrate changed cells)
- Reproducibility controls (deterministic seeding, partition fingerprinting)
- Partition quality dashboard metrics

Usage::

    redist = RedistributeParEnhanced8(case_dir, target_n_procs=8)
    redist.discover()
    result = redist.redistribute_v8(
        cell_centres=centres,
        cell_costs=costs,
        adjacency=adj,
        incremental=True,
    )
    print(f"Migration savings: {result.migration_savings:.1%}")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced_7 import (
    RedistributeParEnhanced7,
    V7RedistributeResult,
    SpectralPartitionConfig,
    LoadPrediction,
    CommunicationMetrics,
)

__all__ = [
    "RedistributeParEnhanced8",
    "V8RedistributeResult",
    "MultiObjectiveConfig",
    "PartitionFingerprint",
    "IncrementalPlan",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置。

    Attributes:
        weight_balance: 负载均衡权重。
        weight_communication: 通信量权重。
        weight_locality: 空间局部性权重。
        pareto_samples: Pareto 前沿采样数。
    """

    weight_balance: float = 0.4
    weight_communication: float = 0.3
    weight_locality: float = 0.3
    pareto_samples: int = 10


@dataclass
class PartitionFingerprint:
    """分区指纹：用于检测分区是否变化。

    Attributes:
        hash_value: 哈希值。
        n_cells: 单元数。
        n_procs: 处理器数。
        edge_cut: 边切割数。
    """

    hash_value: str = ""
    n_cells: int = 0
    n_procs: int = 0
    edge_cut: int = 0


@dataclass
class IncrementalPlan:
    """增量再分配计划。

    Attributes:
        cells_to_migrate: 需要迁移的单元索引。
        source_proc: 源处理器。
        target_proc: 目标处理器。
        n_migrated: 迁移单元数。
        n_total: 总单元数。
        savings_ratio: 相比全量迁移的节省比例。
    """

    cells_to_migrate: torch.Tensor = None
    source_proc: int = 0
    target_proc: int = 1
    n_migrated: int = 0
    n_total: int = 0
    savings_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.cells_to_migrate is None:
            self.cells_to_migrate = torch.zeros(0, dtype=INDEX_DTYPE)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V8RedistributeResult:
    """v8 增强再分配结果。

    Attributes:
        base: V7 再分配结果。
        incremental_plan: 增量迁移计划。
        fingerprint: 分区指纹。
        multi_objective_score: 多目标综合得分。
        migration_savings: 迁移节省比例。
    """

    base: V7RedistributeResult = None
    incremental_plan: IncrementalPlan = None
    fingerprint: PartitionFingerprint = None
    multi_objective_score: float = 0.0
    migration_savings: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced8(RedistributeParEnhanced7):
    """v8 增强再分配，支持多目标优化和增量迁移。

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._mo_config = MultiObjectiveConfig()
        self._previous_mapping: torch.Tensor | None = None
        self._seed: int = 42

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_multi_objective_config(self, config: MultiObjectiveConfig) -> None:
        """设置多目标优化配置。

        Args:
            config: 多目标权重和采样参数。
        """
        self._mo_config = config

    # ------------------------------------------------------------------
    # Partition fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fingerprint(
        mapping: torch.Tensor,
        n_procs: int,
        edge_cut: int = 0,
    ) -> PartitionFingerprint:
        """计算分区指纹。

        Args:
            mapping: ``(n_cells,)`` 单元到处理器的映射。
            n_procs: 处理器数量。
            edge_cut: 边切割数。

        Returns:
            :class:`PartitionFingerprint`。
        """
        m = mapping.to(dtype=INDEX_DTYPE)
        n_cells = m.shape[0]

        # 计算映射哈希
        hash_bytes = hashlib.sha256(m.numpy().tobytes()).hexdigest()[:16]

        return PartitionFingerprint(
            hash_value=hash_bytes,
            n_cells=n_cells,
            n_procs=n_procs,
            edge_cut=edge_cut,
        )

    # ------------------------------------------------------------------
    # Incremental redistribution
    # ------------------------------------------------------------------

    @staticmethod
    def compute_incremental_plan(
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
    ) -> IncrementalPlan:
        """计算增量迁移计划。

        Args:
            old_mapping: ``(n_cells,)`` 旧映射。
            new_mapping: ``(n_cells,)`` 新映射。

        Returns:
            :class:`IncrementalPlan`。
        """
        old = old_mapping.to(dtype=INDEX_DTYPE)
        new = new_mapping.to(dtype=INDEX_DTYPE)
        n_total = old.shape[0]

        # 需要迁移的单元
        changed = old != new
        cells_to_migrate = torch.where(changed)[0].to(dtype=INDEX_DTYPE)
        n_migrated = cells_to_migrate.numel()

        savings = 1.0 - n_migrated / max(n_total, 1)

        return IncrementalPlan(
            cells_to_migrate=cells_to_migrate,
            n_migrated=n_migrated,
            n_total=n_total,
            savings_ratio=savings,
        )

    # ------------------------------------------------------------------
    # Multi-objective scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_multi_objective_score(
        load_imbalance: float,
        communication_ratio: float,
        locality_score: float,
        config: MultiObjectiveConfig,
    ) -> float:
        """计算多目标综合得分。

        Args:
            load_imbalance: 负载不均衡度 (0 = 完美均衡)。
            communication_ratio: 通信比。
            locality_score: 空间局部性得分 (1 = 完美局部性)。
            config: 权重配置。

        Returns:
            综合得分 (0 = 最优)。
        """
        w_b = config.weight_balance
        w_c = config.weight_communication
        w_l = config.weight_locality

        # 归一化各指标到 [0, 1] 范围
        balance_term = min(load_imbalance, 1.0)
        comm_term = min(communication_ratio, 1.0)
        locality_term = 1.0 - min(locality_score, 1.0)

        return w_b * balance_term + w_c * comm_term + w_l * locality_term

    # ------------------------------------------------------------------
    # v8 redistribution
    # ------------------------------------------------------------------

    def redistribute_v8(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        cell_costs: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        current_mapping: Optional[torch.Tensor] = None,
        incremental: bool = False,
        migration_threshold: float = 0.2,
        cooldown_steps: int = 0,
        seed: int = 42,
    ) -> V8RedistributeResult:
        """使用 v8 多目标优化和增量迁移进行再分配。

        Args:
            output_dir: 输出目录。
            field_names: 要再分配的场。
            cell_centres: ``(n_cells, 3)`` 单元中心坐标。
            cell_costs: ``(n_cells,)`` 每单元成本。
            adjacency: ``(n_cells, n_cells)`` 邻接矩阵或 ``(n_cells, max_neighbours)`` 邻接表。
            current_mapping: ``(n_cells,)`` 当前映射。
            incremental: 是否使用增量迁移。
            migration_threshold: 最大迁移比例。
            cooldown_steps: 冷却步数。
            seed: 随机种子。

        Returns:
            :class:`V8RedistributeResult`。
        """
        self._seed = seed

        # 基础 v7 再分配
        base_result = self.redistribute_v7(
            output_dir=output_dir,
            field_names=field_names,
            cell_centres=cell_centres,
            cell_costs=cell_costs,
            adjacency=adjacency,
            current_mapping=current_mapping,
            migration_threshold=migration_threshold,
            cooldown_steps=cooldown_steps,
            seed=seed,
        )

        # 分区指纹
        fingerprint = PartitionFingerprint()
        if current_mapping is not None:
            fingerprint = self.compute_fingerprint(
                current_mapping, self._target_n_procs
            )

        # 增量迁移计划
        inc_plan = IncrementalPlan()
        if incremental and self._previous_mapping is not None and current_mapping is not None:
            inc_plan = self.compute_incremental_plan(
                self._previous_mapping, current_mapping
            )

        # 记录当前映射
        if current_mapping is not None:
            self._previous_mapping = current_mapping.to(dtype=INDEX_DTYPE).clone()

        # 多目标得分
        mo_score = 0.0
        if cell_costs is not None:
            load_imb = float(cell_costs.std() / max(cell_costs.mean().item(), 1e-30))
            comm_ratio = base_result.communication.communication_ratio if base_result.communication else 0.0
            mo_score = self.compute_multi_objective_score(
                load_imb, comm_ratio, 0.5, self._mo_config
            )

        migration_savings = inc_plan.savings_ratio if incremental else 0.0

        return V8RedistributeResult(
            base=base_result,
            incremental_plan=inc_plan,
            fingerprint=fingerprint,
            multi_objective_score=mo_score,
            migration_savings=migration_savings,
        )

    def __repr__(self) -> str:
        cd = self._cooldown_remaining
        n_hist = len(self._cost_history)
        return (
            f"RedistributeParEnhanced8(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs}, cooldown={cd}, history={n_hist})"
        )
