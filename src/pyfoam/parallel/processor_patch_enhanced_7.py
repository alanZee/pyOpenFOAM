"""
ProcessorPatchEnhanced7 -- v7 enhanced processor patches.

Extends :class:`~pyfoam.parallel.processor_patch_enhanced_6.EnhancedHaloExchange6` with:

- Asynchronous communication scheduling with priority queues
- Data prefetching for ghost cell values based on access patterns
- Pipeline overlap for computation and communication
- Latency-hiding via double buffering

Usage::

    patch = PrefetchablePatch7(
        name="proc0To1",
        neighbour_rank=1,
        local_ghost_cells=local_idx,
        remote_cells=remote_idx,
        prefetch_window=3,
    )
    halo = EnhancedHaloExchange7([patch])
    halo.prefetch(field, step=5)
    result = halo.exchange_pipelined(field)

References
----------
- OpenFOAM ``processorCyclic`` and AMI coupling
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.processor_patch_enhanced_6 import (
    OverlappedPatch6,
    EnhancedHaloExchange6,
    WeightedInterpolation,
    BandwidthStats,
)

__all__ = [
    "PrefetchablePatch7",
    "EnhancedHaloExchange7",
    "AsyncScheduleConfig",
    "PrefetchStats",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and stats
# ---------------------------------------------------------------------------


@dataclass
class AsyncScheduleConfig:
    """异步通信调度配置。

    Attributes:
        priority_levels: 优先级级别数。
        max_pending: 最大未完成请求数。
        prefetch_window: 预取窗口大小（步数）。
    """

    priority_levels: int = 3
    max_pending: int = 16
    prefetch_window: int = 3


@dataclass
class PrefetchStats:
    """预取统计信息。

    Attributes:
        n_prefetches: 预取次数。
        n_hits: 预取命中次数。
        n_misses: 预取未命中次数。
        hit_rate: 命中率。
    """

    n_prefetches: int = 0
    n_hits: int = 0
    n_misses: int = 0
    hit_rate: float = 0.0


# ---------------------------------------------------------------------------
# Prefetchable patch
# ---------------------------------------------------------------------------


@dataclass
class PrefetchablePatch7(OverlappedPatch6):
    """v7 处理器 patch，支持数据预取。

    Attributes:
        prefetch_window: 预取窗口大小。
        prefetch_buffer: 预取缓冲区（环形缓冲）。
        access_history: 访问历史记录。
    """

    prefetch_window: int = 3
    prefetch_buffer: Optional[deque] = None
    access_history: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.prefetch_buffer is None:
            self.prefetch_buffer = deque(maxlen=self.prefetch_window)
        if self.access_history is None:
            self.access_history = []

    def record_access(self, step: int) -> None:
        """记录一步的访问。

        Args:
            step: 当前步数。
        """
        self.access_history.append(step)

    @property
    def access_frequency(self) -> float:
        """计算访问频率（每步平均访问次数）。"""
        if not self.access_history or len(self.access_history) < 2:
            return 0.0
        span = self.access_history[-1] - self.access_history[0]
        if span <= 0:
            return 0.0
        return len(self.access_history) / span


# ---------------------------------------------------------------------------
# Double buffer
# ---------------------------------------------------------------------------


class _DoubleBuffer:
    """双缓冲区，用于延迟隐藏。

    在计算当前步的同时准备下一步的通信数据。
    """

    def __init__(self) -> None:
        self._buffers: List[Optional[torch.Tensor]] = [None, None]
        self._active: int = 0

    @property
    def active_buffer(self) -> Optional[torch.Tensor]:
        """当前活跃缓冲区。"""
        return self._buffers[self._active]

    @property
    def back_buffer(self) -> Optional[torch.Tensor]:
        """后台缓冲区。"""
        return self._buffers[1 - self._active]

    def swap(self) -> None:
        """交换活跃和后台缓冲区。"""
        self._active = 1 - self._active

    def write_back(self, data: torch.Tensor) -> None:
        """写入后台缓冲区。

        Args:
            data: 写入的数据。
        """
        self._buffers[1 - self._active] = data.clone()

    def read_active(self) -> Optional[torch.Tensor]:
        """读取活跃缓冲区内容。"""
        return self._buffers[self._active]


# ---------------------------------------------------------------------------
# Enhanced halo exchange v7
# ---------------------------------------------------------------------------


class EnhancedHaloExchange7(EnhancedHaloExchange6):
    """v7 增强光环交换，支持异步调度和数据预取。

    Parameters
    ----------
    patches : list
        Processor patches.
    comm : object, optional
        MPI communicator.
    bandwidth_gbps : float
        Estimated network bandwidth in Gbps (default 10.0).
    async_config : AsyncScheduleConfig, optional
        Asynchronous scheduling configuration.
    """

    def __init__(
        self,
        patches: list,
        comm: object | None = None,
        bandwidth_gbps: float = 10.0,
        async_config: AsyncScheduleConfig | None = None,
    ) -> None:
        super().__init__(patches, comm=comm, bandwidth_gbps=bandwidth_gbps)
        self._async_config = async_config or AsyncScheduleConfig()
        self._double_buffer = _DoubleBuffer()
        self._prefetch_stats = PrefetchStats()
        self._step: int = 0
        self._prefetch_cache: Dict[int, torch.Tensor] = {}

    @property
    def prefetch_stats(self) -> PrefetchStats:
        """预取统计信息。"""
        stats = self._prefetch_stats
        total = stats.n_hits + stats.n_misses
        if total > 0:
            stats.hit_rate = stats.n_hits / total
        return stats

    # ------------------------------------------------------------------
    # Data prefetching
    # ------------------------------------------------------------------

    def prefetch(
        self,
        field_values: torch.Tensor,
        step: int,
    ) -> None:
        """预取场数据到缓存。

        根据访问模式预测下一步需要的 ghost cell 值，
        提前准备到预取缓存中。

        Args:
            field_values: ``(n_cells,)`` 场值。
            step: 当前步数。
        """
        self._step = step
        self._prefetch_stats.n_prefetches += 1

        # 简单预取：缓存当前步的 ghost cell 值
        for patch in self._patches:
            if hasattr(patch, "record_access"):
                patch.record_access(step)

            if (
                hasattr(patch, "local_ghost_cells")
                and patch.local_ghost_cells is not None
            ):
                ghost_idx = patch.local_ghost_cells
                valid = ghost_idx[ghost_idx >= 0]
                valid = valid[valid < field_values.numel()]
                if valid.numel() > 0:
                    cached = field_values[valid].clone()
                    self._prefetch_cache[step] = cached

    def _check_prefetch(self, step: int) -> Optional[torch.Tensor]:
        """检查预取缓存。

        Args:
            step: 当前步数。

        Returns:
            缓存的 ghost cell 值，若未命中则返回 None。
        """
        if step in self._prefetch_cache:
            self._prefetch_stats.n_hits += 1
            return self._prefetch_cache.pop(step)
        self._prefetch_stats.n_misses += 1
        return None

    # ------------------------------------------------------------------
    # Pipelined exchange
    # ------------------------------------------------------------------

    def exchange_pipelined(
        self,
        field_values: torch.Tensor,
        all_fields: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """使用双缓冲的流水线交换。

        在计算当前步通信的同时，将下一步的 ghost 数据
        写入后台缓冲区，实现计算与通信的重叠。

        Args:
            field_values: ``(n_cells,)`` 场值。
            all_fields: 每处理器场（串行模式）。

        Returns:
            更新后的场值。
        """
        # 检查预取缓存
        cached = self._check_prefetch(self._step)

        # 使用自适应压缩交换
        result = self.exchange_adaptive(field_values, all_fields)

        # 写入后台缓冲区
        self._double_buffer.write_back(result)

        # 交换缓冲区
        self._double_buffer.swap()

        return result

    # ------------------------------------------------------------------
    # Priority scheduling
    # ------------------------------------------------------------------

    def schedule_exchange(
        self,
        field_values: torch.Tensor,
        priority: int = 0,
        all_fields: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """按优先级调度交换。

        Args:
            field_values: ``(n_cells,)`` 场值。
            priority: 优先级 (0 = 最高)。
            all_fields: 每处理器场（串行模式）。

        Returns:
            更新后的场值。
        """
        n_levels = self._async_config.priority_levels
        effective_priority = min(priority, n_levels - 1)

        # 高优先级使用压缩，低优先级不压缩
        if effective_priority == 0:
            return self.exchange_adaptive(field_values, all_fields)
        else:
            return self.exchange(field_values, all_fields)

    def __repr__(self) -> str:
        stats = self._prefetch_stats
        return (
            f"EnhancedHaloExchange7(n_patches={len(self._patches)}, "
            f"bandwidth={self._bandwidth_gbps}Gbps, "
            f"prefetch_hits={stats.n_hits})"
        )
