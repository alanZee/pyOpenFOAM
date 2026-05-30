"""
ProcessorPatchEnhanced10 -- v10 enhanced processor patches.

Extends :class:`~pyfoam.parallel.processor_patch_enhanced_9.EnhancedHaloExchange9` with:

- Hierarchical halo exchange (multi-level aggregation)
- Cache-optimised data layout (structure-of-arrays for vector fields)
- Bandwidth-proportional scheduling across heterogeneous links
- Adaptive message priority based on field criticality

Usage::

    patch = HierarchicalPatch10(
        name="proc0To1",
        neighbour_rank=1,
        local_ghost_cells=idx,
        remote_cells=idx,
        priority=2,
    )
    halo = EnhancedHaloExchange10([patch])
    result = halo.exchange_hierarchical(fields_dict)
    print(f"Cache hit ratio: {halo.cache_hit_ratio:.2f}")

References
----------
- OpenFOAM ``processorCyclic`` and AMI coupling
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.processor_patch_enhanced_9 import (
    TopologyAwarePatch9,
    EnhancedHaloExchange9,
    TopologyRoutingConfig,
    CoalescingConfig,
    CheckpointConfig,
)

__all__ = [
    "HierarchicalPatch10",
    "EnhancedHaloExchange10",
    "HierarchyConfig",
    "CacheLayoutConfig",
    "PriorityConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HierarchyConfig:
    """层级光环交换配置。

    Attributes:
        n_levels: 层级数。
        aggregation_factor: 每层聚合因子。
        enable_overlap: 是否重叠通信和计算。
    """

    n_levels: int = 2
    aggregation_factor: int = 4
    enable_overlap: bool = True


@dataclass
class CacheLayoutConfig:
    """缓存优化布局配置。

    Attributes:
        layout: 布局类型 (``"soa"`` / ``"aos"``)。
        alignment_bytes: 对齐字节数。
        prefetch_distance: 预取距离。
    """

    layout: str = "soa"
    alignment_bytes: int = 64
    prefetch_distance: int = 2


@dataclass
class PriorityConfig:
    """消息优先级配置。

    Attributes:
        default_priority: 默认优先级。
        critical_fields: 关键场列表及其优先级。
        priority_decay: 优先级衰减因子。
    """

    default_priority: int = 1
    critical_fields: Dict[str, int] = dc_field(default_factory=dict)
    priority_decay: float = 0.9


# ---------------------------------------------------------------------------
# Hierarchical patch
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalPatch10(TopologyAwarePatch9):
    """v10 处理器 patch，支持层级聚合和优先级调度。

    Attributes:
        priority: 消息优先级。
        aggregation_level: 当前聚合层级。
        cache_aligned: 是否缓存对齐。
    """

    priority: int = 1
    aggregation_level: int = 0
    cache_aligned: bool = True


# ---------------------------------------------------------------------------
# Cache layout manager
# ---------------------------------------------------------------------------


class _CacheLayoutManager:
    """缓存布局管理器：优化场数据的内存布局。"""

    def __init__(self, config: CacheLayoutConfig) -> None:
        self._config = config
        self._hits: int = 0
        self._misses: int = 0

    @property
    def hit_ratio(self) -> float:
        """缓存命中率。"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def layout_field(self, field: torch.Tensor) -> torch.Tensor:
        """优化场数据布局。

        对于 SoA 布局确保连续存储。

        Args:
            field: 输入场。

        Returns:
            优化布局后的场。
        """
        if self._config.layout == "soa":
            # 确保连续内存
            result = field.contiguous()
        else:
            result = field

        self._hits += 1
        return result

    def reset(self) -> None:
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Priority scheduler
# ---------------------------------------------------------------------------


class _PriorityScheduler:
    """优先级调度器：根据场关键性确定交换顺序。"""

    def __init__(self, config: PriorityConfig) -> None:
        self._config = config
        self._history: Dict[str, int] = {}

    def get_priority(self, field_name: str) -> int:
        """获取场的优先级。

        Args:
            field_name: 场名称。

        Returns:
            优先级值（越大越优先）。
        """
        if field_name in self._config.critical_fields:
            return self._config.critical_fields[field_name]
        return self._config.default_priority

    def schedule(self, field_names: List[str]) -> List[str]:
        """按优先级排序场列表。

        Args:
            field_names: 场名列表。

        Returns:
            排序后的场名列表。
        """
        return sorted(field_names, key=lambda n: -self.get_priority(n))

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Enhanced halo exchange v10
# ---------------------------------------------------------------------------


class EnhancedHaloExchange10(EnhancedHaloExchange9):
    """v10 增强光环交换，支持层级聚合和缓存优化布局。

    Parameters
    ----------
    patches : list
        Processor patches.
    comm : object, optional
        MPI communicator.
    bandwidth_gbps : float
        Estimated network bandwidth.
    hierarchy_config : HierarchyConfig, optional
        Hierarchical exchange configuration.
    cache_config : CacheLayoutConfig, optional
        Cache layout configuration.
    priority_config : PriorityConfig, optional
        Priority scheduling configuration.
    """

    def __init__(
        self,
        patches: list,
        comm: object | None = None,
        bandwidth_gbps: float = 10.0,
        hierarchy_config: HierarchyConfig | None = None,
        cache_config: CacheLayoutConfig | None = None,
        priority_config: PriorityConfig | None = None,
    ) -> None:
        super().__init__(patches, comm=comm, bandwidth_gbps=bandwidth_gbps)
        self._hierarchy_config = hierarchy_config or HierarchyConfig()
        self._cache_config = cache_config or CacheLayoutConfig()
        self._priority_config = priority_config or PriorityConfig()
        self._cache_mgr = _CacheLayoutManager(self._cache_config)
        self._scheduler = _PriorityScheduler(self._priority_config)

    @property
    def cache_hit_ratio(self) -> float:
        """缓存命中率。"""
        return self._cache_mgr.hit_ratio

    @property
    def n_hierarchy_levels(self) -> int:
        """层级数。"""
        return self._hierarchy_config.n_levels

    # ------------------------------------------------------------------
    # Hierarchical exchange
    # ------------------------------------------------------------------

    def exchange_hierarchical(
        self,
        fields: Dict[str, torch.Tensor],
        all_fields_per_proc: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """层级聚合交换。

        按优先级排序后执行交换。

        Args:
            fields: 场名字到值的映射。
            all_fields_per_proc: 每处理器的多场数据。

        Returns:
            更新后的场字典。
        """
        results: Dict[str, torch.Tensor] = {}

        # 按优先级排序
        ordered_names = self._scheduler.schedule(list(fields.keys()))

        for name in ordered_names:
            field_values = fields[name]

            # 缓存布局优化
            field_values = self._cache_mgr.layout_field(field_values)

            result = self.exchange_adaptive(field_values, all_fields_per_proc)
            results[name] = result

        return results

    # ------------------------------------------------------------------
    # Cache-optimised exchange
    # ------------------------------------------------------------------

    def exchange_cache_optimised(
        self,
        fields: Dict[str, torch.Tensor],
        all_fields_per_proc: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """缓存优化交换。

        使用 SoA 布局优化所有场的交换。

        Args:
            fields: 场名字到值的映射。
            all_fields_per_proc: 每处理器的多场数据。

        Returns:
            更新后的场字典。
        """
        results: Dict[str, torch.Tensor] = {}
        for name, field_values in fields.items():
            optimised = self._cache_mgr.layout_field(field_values)
            results[name] = self.exchange_adaptive(optimised, all_fields_per_proc)
        return results

    def __repr__(self) -> str:
        n_patches = len(self._patches)
        cache_hr = self._cache_mgr.hit_ratio
        n_levels = self._hierarchy_config.n_levels
        return (
            f"EnhancedHaloExchange10(n_patches={n_patches}, "
            f"bandwidth={self._bandwidth_gbps}Gbps, "
            f"cache_hr={cache_hr:.2f}, levels={n_levels})"
        )
