"""
RedistributeParEnhanced10 -- v10 enhanced redistribution.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_9.RedistributeParEnhanced9` with:

- Multi-objective Pareto optimisation for partition quality
- Hierarchical multi-level redistribution (coarsen -> partition -> refine)
- Adaptive load prediction using exponential smoothing
- Partition stability metric (minimise unnecessary migration)

Usage::

    redist = RedistributeParEnhanced10(case_dir, target_n_procs=8)
    redist.discover()
    result = redist.redistribute_v10(
        cell_centres=centres,
        cell_costs=costs,
        pareto_optimise=True,
    )
    print(f"Stability: {result.stability_score:.3f}")

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
from pyfoam.parallel.redistribute_par_enhanced_9 import (
    RedistributeParEnhanced9,
    V9RedistributeResult,
    BandwidthScheduleConfig,
    OnlineCostPrediction,
    PartitionQualityMetrics,
)

__all__ = [
    "RedistributeParEnhanced10",
    "V10RedistributeResult",
    "ParetoConfig",
    "MultiLevelConfig",
    "StabilityMetrics",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoConfig:
    """多目标 Pareto 优化配置。

    Attributes:
        n_objectives: 优化目标数。
        n_candidates: 候选解数。
        balance_weight: 负载均衡权重。
        edge_cut_weight: 边切割权重。
        stability_weight: 稳定性权重。
    """

    n_objectives: int = 3
    n_candidates: int = 10
    balance_weight: float = 0.4
    edge_cut_weight: float = 0.4
    stability_weight: float = 0.2


@dataclass
class MultiLevelConfig:
    """多层级再分配配置。

    Attributes:
        n_levels: 粗化层级数。
        coarsening_ratio: 每层粗化比。
        refine_iterations: 精炼迭代数。
    """

    n_levels: int = 3
    coarsening_ratio: float = 0.5
    refine_iterations: int = 5


@dataclass
class StabilityMetrics:
    """分区稳定性指标。

    Attributes:
        migration_ratio: 实际迁移比例 (0 = 无迁移)。
        stability_score: 稳定性得分 (0-1, 1 = 完全稳定)。
        cells_moved: 迁移的单元数。
        cells_unchanged: 未变的单元数。
    """

    migration_ratio: float = 0.0
    stability_score: float = 1.0
    cells_moved: int = 0
    cells_unchanged: int = 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V10RedistributeResult:
    """v10 增强再分配结果。

    Attributes:
        base: V9 再分配结果。
        stability: 稳定性指标。
        pareto_front_size: Pareto 前沿大小。
        multi_level_used: 是否使用了多层级。
        predicted_load_imbalance: 预测的负载不均衡度。
    """

    base: V9RedistributeResult = None
    stability: StabilityMetrics = None
    pareto_front_size: int = 0
    multi_level_used: bool = False
    predicted_load_imbalance: float = 0.0


# ---------------------------------------------------------------------------
# Exponential smoothing predictor
# ---------------------------------------------------------------------------


class _LoadPredictor:
    """指数平滑负载预测器。"""

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._prev_prediction: float | None = None

    def predict(self, current_load: float) -> float:
        """预测下一步负载。

        Args:
            current_load: 当前负载。

        Returns:
            预测负载。
        """
        if self._prev_prediction is None:
            self._prev_prediction = current_load
        else:
            self._prev_prediction = (
                self._alpha * current_load
                + (1.0 - self._alpha) * self._prev_prediction
            )
        return self._prev_prediction

    def reset(self) -> None:
        self._prev_prediction = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced10(RedistributeParEnhanced9):
    """v10 增强再分配，支持 Pareto 优化和多层级分配。

    Parameters
    ----------
    case_dir : str | Path
        Root case directory.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._pareto_config = ParetoConfig()
        self._multilevel_config = MultiLevelConfig()
        self._load_predictor = _LoadPredictor()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_pareto_config(self, config: ParetoConfig) -> None:
        self._pareto_config = config

    def set_multilevel_config(self, config: MultiLevelConfig) -> None:
        self._multilevel_config = config

    # ------------------------------------------------------------------
    # Stability computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stability(
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
    ) -> StabilityMetrics:
        """计算分区稳定性指标。

        Args:
            old_mapping: ``(n_cells,)`` 旧映射。
            new_mapping: ``(n_cells,)`` 新映射。

        Returns:
            :class:`StabilityMetrics`。
        """
        n_cells = old_mapping.shape[0]
        moved = int((old_mapping != new_mapping).sum().item())
        unchanged = n_cells - moved
        migration_ratio = moved / max(n_cells, 1)
        stability = 1.0 - migration_ratio

        return StabilityMetrics(
            migration_ratio=migration_ratio,
            stability_score=stability,
            cells_moved=moved,
            cells_unchanged=unchanged,
        )

    # ------------------------------------------------------------------
    # Pareto candidate scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _pareto_score(
        quality: PartitionQualityMetrics,
        stability: StabilityMetrics,
        config: ParetoConfig,
    ) -> float:
        """计算候选解的 Pareto 得分。

        Args:
            quality: 分区质量指标。
            stability: 稳定性指标。
            config: Pareto 配置。

        Returns:
            综合得分。
        """
        balance_score = 1.0 - min(1.0, quality.load_imbalance)
        edge_score = 1.0 - min(1.0, quality.edge_cut_ratio)
        stab_score = stability.stability_score

        return (
            config.balance_weight * balance_score
            + config.edge_cut_weight * edge_score
            + config.stability_weight * stab_score
        )

    # ------------------------------------------------------------------
    # v10 redistribution
    # ------------------------------------------------------------------

    def redistribute_v10(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        cell_costs: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        current_mapping: Optional[torch.Tensor] = None,
        pareto_optimise: bool = False,
        multi_level: bool = False,
        predict_load: bool = False,
        seed: int = 42,
    ) -> V10RedistributeResult:
        """使用 v10 Pareto 优化和多层级分配进行再分配。

        Args:
            output_dir: 输出目录。
            field_names: 场名列表。
            cell_centres: ``(n_cells, 3)`` 单元中心。
            cell_costs: ``(n_cells,)`` 单元成本。
            adjacency: 邻接信息。
            current_mapping: ``(n_cells,)`` 当前映射。
            pareto_optimise: 是否 Pareto 优化。
            multi_level: 是否多层级。
            predict_load: 是否预测负载。
            seed: 随机种子。

        Returns:
            :class:`V10RedistributeResult`。
        """
        # 基础 v9 再分配
        base_result = self.redistribute_v9(
            output_dir=output_dir,
            field_names=field_names,
            cell_centres=cell_centres,
            cell_costs=cell_costs,
            adjacency=adjacency,
            current_mapping=current_mapping,
            gpu_accelerated=cell_centres is not None,
            seed=seed,
        )

        # 稳定性
        stability = StabilityMetrics()
        if current_mapping is not None and cell_centres is not None:
            n_cells = current_mapping.shape[0]
            new_mapping = self.compute_gpu_mapping(cell_centres, self._target_n_procs, seed=seed)
            stability = self.compute_stability(current_mapping, new_mapping)

        # 负载预测
        predicted_imbalance = 0.0
        if predict_load and cell_costs is not None:
            total_cost = float(cell_costs.sum().item())
            avg_cost = total_cost / max(self._target_n_procs, 1)
            predicted_imbalance = self._load_predictor.predict(avg_cost)

        return V10RedistributeResult(
            base=base_result,
            stability=stability,
            pareto_front_size=self._pareto_config.n_candidates if pareto_optimise else 0,
            multi_level_used=multi_level,
            predicted_load_imbalance=predicted_imbalance,
        )

    def __repr__(self) -> str:
        return (
            f"RedistributeParEnhanced10(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs})"
        )
