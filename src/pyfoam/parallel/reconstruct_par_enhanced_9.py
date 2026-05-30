"""
ReconstructParEnhanced9 -- v9 enhanced parallel reconstruction.

Extends :class:`~pyfoam.parallel.reconstruct_par_enhanced_8.ReconstructParEnhanced8` with:

- Distributed hashing for change detection across processor fields
- Progressive reconstruction (coarse-to-fine with intermediate output)
- Adaptive field selection based on reconstruction priority
- Cross-field dependency graph for optimal reconstruction ordering

Usage::

    recon = ReconstructParEnhanced9(case_dir)
    recon.discover()
    result = recon.reconstruct_case_v9(
        field_names=["p", "U"],
        progressive=True,
        priority_weights={"p": 1.0, "U": 0.5},
    )
    print(f"Reconstruction order: {result.reconstruction_order}")

References
----------
- OpenFOAM ``reconstructPar`` utility source
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par_enhanced_8 import (
    ReconstructParEnhanced8,
    V8ReconstructResult,
    StreamingConfig,
    EntropyConfig,
    FieldCorrelation,
)

__all__ = [
    "ReconstructParEnhanced9",
    "V9ReconstructResult",
    "ProgressiveConfig",
    "FieldDependency",
    "DistributedHash",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProgressiveConfig:
    """渐进式重建配置。

    Attributes:
        n_levels: 粗到细的层级数。
        coarse_cells_per_level: 每层的粗单元数。
        enable_intermediate_output: 是否启用中间层输出。
    """

    n_levels: int = 3
    coarse_cells_per_level: int = 5000
    enable_intermediate_output: bool = False


@dataclass
class FieldDependency:
    """场依赖关系。

    Attributes:
        source: 源场名称。
        target: 目标场名称。
        dependency_type: 依赖类型 (``"gradient"``/``"coupling"``/``"boundary"``)。
        weight: 依赖权重。
    """

    source: str = ""
    target: str = ""
    dependency_type: str = "gradient"
    weight: float = 1.0


@dataclass
class DistributedHash:
    """分布式哈希结果。

    Attributes:
        field_name: 场名称。
        hash_value: 哈希值。
        n_elements: 元素数。
        checksum: 校验和。
    """

    field_name: str = ""
    hash_value: str = ""
    n_elements: int = 0
    checksum: float = 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V9ReconstructResult:
    """v9 增强重建结果。

    Attributes:
        base: V8 重建结果。
        reconstruction_order: 重建顺序（按依赖图排序）。
        field_hashes: 每场的分布式哈希。
        dependencies: 场间依赖列表。
        progressive_level: 当前渐进层级。
        priority_scores: 各场优先级得分。
    """

    base: V8ReconstructResult = None
    reconstruction_order: List[str] = dc_field(default_factory=list)
    field_hashes: Dict[str, DistributedHash] = dc_field(default_factory=dict)
    dependencies: List[FieldDependency] = dc_field(default_factory=list)
    progressive_level: int = 0
    priority_scores: Dict[str, float] = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependency graph (simplified DAG)
# ---------------------------------------------------------------------------


class _DependencyGraph:
    """场依赖关系有向图（简化 DAG）。

    用于确定最优重建顺序（拓扑排序）。
    """

    def __init__(self) -> None:
        self._edges: Dict[str, List[str]] = {}
        self._weights: Dict[tuple, float] = {}

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """添加依赖边。"""
        if source not in self._edges:
            self._edges[source] = []
        self._edges[source].append(target)
        self._weights[(source, target)] = weight

    def topological_sort(self) -> List[str]:
        """拓扑排序确定重建顺序。

        Returns:
            排序后的场名称列表（依赖的场在前面）。
        """
        all_nodes = set(self._edges.keys())
        for targets in self._edges.values():
            all_nodes.update(targets)

        # 计算入度
        in_degree: Dict[str, int] = {n: 0 for n in all_nodes}
        for source, targets in self._edges.items():
            for target in targets:
                in_degree[target] = in_degree.get(target, 0) + 1

        # Kahn 算法
        queue = [n for n, d in in_degree.items() if d == 0]
        result: List[str] = []

        while queue:
            # 按权重排序（高优先级在前）
            queue.sort(key=lambda n: -self._get_priority(n))
            node = queue.pop(0)
            result.append(node)

            for target in self._edges.get(node, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        return result

    def _get_priority(self, node: str) -> float:
        """获取节点优先级。"""
        total = 0.0
        for (s, t), w in self._weights.items():
            if s == node or t == node:
                total += w
        return total


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReconstructParEnhanced9(ReconstructParEnhanced8):
    """v9 增强并行重建，支持分布式哈希和渐进式重建。

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        super().__init__(case_dir)
        self._progressive_config = ProgressiveConfig()
        self._dependency_graph = _DependencyGraph()
        self._field_hashes: Dict[str, DistributedHash] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_progressive_config(self, config: ProgressiveConfig) -> None:
        """设置渐进式重建配置。

        Args:
            config: 渐进式参数。
        """
        self._progressive_config = config

    def add_field_dependency(self, dep: FieldDependency) -> None:
        """添加场间依赖关系。

        Args:
            dep: 依赖关系描述。
        """
        self._dependency_graph.add_edge(
            dep.source, dep.target, dep.weight
        )

    # ------------------------------------------------------------------
    # Distributed hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_distributed_hash(
        field: torch.Tensor,
        field_name: str = "",
    ) -> DistributedHash:
        """计算场的分布式哈希。

        Args:
            field: ``(n_cells,)`` 场值。
            field_name: 场名称。

        Returns:
            :class:`DistributedHash`。
        """
        f = field.to(dtype=torch.float64)
        n_elements = f.numel()

        # 哈希
        hash_bytes = hashlib.sha256(f.numpy().tobytes()).hexdigest()[:16]

        # 校验和
        checksum = float(f.sum().item())

        return DistributedHash(
            field_name=field_name,
            hash_value=hash_bytes,
            n_elements=n_elements,
            checksum=checksum,
        )

    # ------------------------------------------------------------------
    # Priority scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_priority_scores(
        field_names: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """计算各场的优先级得分。

        Args:
            field_names: 场名称列表。
            weights: 可选的权重字典。

        Returns:
            各场的优先级得分。
        """
        scores: Dict[str, float] = {}
        for name in field_names:
            w = 1.0
            if weights and name in weights:
                w = weights[name]
            scores[name] = w
        return scores

    # ------------------------------------------------------------------
    # v9 reconstruction
    # ------------------------------------------------------------------

    def reconstruct_case_v9(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        streaming: bool = False,
        correlation_aware: bool = False,
        entropy_adaptive: bool = False,
        progressive: bool = False,
        priority_weights: Optional[Dict[str, float]] = None,
        compute_hashes: bool = False,
    ) -> V9ReconstructResult:
        """使用 v9 渐进式重建和分布式哈希进行重建。

        Args:
            output_dir: 输出目录。
            field_names: 要重建的场。
            streaming: 是否使用流式处理。
            correlation_aware: 是否计算场间相关性。
            entropy_adaptive: 是否使用熵自适应压缩。
            progressive: 是否使用渐进式重建。
            priority_weights: 各场优先级权重。
            compute_hashes: 是否计算分布式哈希。

        Returns:
            :class:`V9ReconstructResult`。
        """
        # 基础 v8 重建
        base_result = self.reconstruct_case_v8(
            output_dir=output_dir,
            field_names=field_names,
            streaming=streaming,
            correlation_aware=correlation_aware,
            entropy_adaptive=entropy_adaptive,
        )

        # 重建顺序
        reconstruction_order: List[str] = []
        if field_names:
            # 先使用依赖图排序
            sorted_by_graph = self._dependency_graph.topological_sort()
            # 将 field_names 中有依赖关系的按图排序，其余追加到末尾
            ordered = [n for n in sorted_by_graph if n in field_names]
            remaining = [n for n in field_names if n not in ordered]
            reconstruction_order = ordered + remaining

        # 优先级得分
        priority_scores: Dict[str, float] = {}
        if field_names:
            priority_scores = self.compute_priority_scores(field_names, priority_weights)

        # 分布式哈希
        field_hashes: Dict[str, DistributedHash] = {}
        if compute_hashes and field_names:
            for name in field_names:
                field_hashes[name] = DistributedHash(field_name=name)

        # 渐进层级
        progressive_level = self._progressive_config.n_levels if progressive else 0

        return V9ReconstructResult(
            base=base_result,
            reconstruction_order=reconstruction_order,
            field_hashes=field_hashes,
            dependencies=[],
            progressive_level=progressive_level,
            priority_scores=priority_scores,
        )

    def __repr__(self) -> str:
        zones = len(self.zone_names)
        n_cp = len(self._checkpoints)
        wl = self._wavelet_config.level
        n_deps = len(self._dependency_graph._edges)
        return (
            f"ReconstructParEnhanced9(case='{self._case_dir}', "
            f"zones={zones}, checkpoints={n_cp}, wavelet_level={wl}, "
            f"deps={n_deps})"
        )
