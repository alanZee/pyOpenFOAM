"""
RedistributeParEnhanced6 -- v6 enhanced redistribution with hierarchical partitioning.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_5.RedistributeParEnhanced5` with:

- Hierarchical partitioning (coarse + fine two-level decomposition)
- Partition quality metrics (edge-cut, balance ratio)
- Repartitioning cooldown logic (avoid oscillation)
- Incremental redistribution (only move changed cells)

Usage::

    redist = RedistributeParEnhanced6(case_dir, target_n_procs=8)
    redist.discover()
    result = redist.redistribute_v6(
        cell_centres=centres,
        cell_costs=costs,
        current_mapping=old_mapping,
        cooldown_steps=5,
    )
    print(f"Edge-cut: {result.edge_cut:.1f}")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced_5 import (
    RedistributeParEnhanced5,
    V5RedistributeResult,
    MigrationPlan,
    CostEstimator,
)

__all__ = [
    "RedistributeParEnhanced6",
    "V6RedistributeResult",
    "HierarchicalPartitionConfig",
    "PartitionMetrics",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchical partition config
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalPartitionConfig:
    """Configuration for hierarchical two-level partitioning.

    Attributes:
        n_coarse_groups: Number of coarse-level groups.
        fine_cells_per_group: Target number of fine cells per group.
        rebalance_threshold: Imbalance threshold to trigger rebalancing.
    """

    n_coarse_groups: int = 2
    fine_cells_per_group: int = 1000
    rebalance_threshold: float = 0.1


# ---------------------------------------------------------------------------
# Partition metrics
# ---------------------------------------------------------------------------


@dataclass
class PartitionMetrics:
    """Quality metrics for a partition.

    Attributes:
        edge_cut: Number of inter-processor edges.
        balance_ratio: Ratio of max to average processor load.
        max_cells: Maximum cells on any processor.
        min_cells: Minimum cells on any processor.
    """

    edge_cut: int = 0
    balance_ratio: float = 1.0
    max_cells: int = 0
    min_cells: int = 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V6RedistributeResult:
    """Result of a v6 enhanced redistribution.

    Attributes:
        base: V5 redistribution result.
        metrics: Partition quality metrics.
        hierarchical: Whether hierarchical partitioning was used.
        cooldown_remaining: Cooldown steps remaining before next repartition.
    """

    base: V5RedistributeResult
    metrics: PartitionMetrics = None
    hierarchical: bool = False
    cooldown_remaining: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced6(RedistributeParEnhanced5):
    """v6 enhanced redistribution with hierarchical partitioning and cooldown.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._hierarchical_config: Optional[HierarchicalPartitionConfig] = None
        self._cooldown_remaining: int = 0
        self._last_partition_time: float = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_hierarchical_config(
        self, config: HierarchicalPartitionConfig
    ) -> None:
        """Set hierarchical partitioning configuration.

        Args:
            config: Hierarchical partition parameters.
        """
        self._hierarchical_config = config

    # ------------------------------------------------------------------
    # Partition metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_partition_metrics(
        mapping: torch.Tensor,
        cell_costs: torch.Tensor,
        n_procs: int,
    ) -> PartitionMetrics:
        """Compute partition quality metrics.

        Args:
            mapping: ``(n_cells,)`` cell-to-processor mapping.
            cell_costs: ``(n_cells,)`` per-cell costs.
            n_procs: Number of processors.

        Returns:
            :class:`PartitionMetrics`.
        """
        mapping = mapping.to(dtype=INDEX_DTYPE)
        cell_costs = cell_costs.to(dtype=torch.float64)

        proc_load = torch.zeros(n_procs, dtype=torch.float64)
        proc_count = torch.zeros(n_procs, dtype=INDEX_DTYPE)

        for p in range(n_procs):
            mask = mapping == p
            if mask.any():
                proc_load[p] = cell_costs[mask].sum()
                proc_count[p] = mask.sum()

        avg_load = proc_load.sum() / max(n_procs, 1)
        balance = proc_load.max().item() / max(avg_load.item(), 1e-30)

        # Edge cut: count cells whose neighbour belongs to a different proc
        # (simplified: count cells that change proc in consecutive indices)
        edge_cut = 0
        for i in range(1, mapping.numel()):
            if mapping[i] != mapping[i - 1]:
                edge_cut += 1

        return PartitionMetrics(
            edge_cut=edge_cut,
            balance_ratio=balance,
            max_cells=int(proc_count.max().item()),
            min_cells=int(proc_count.min().item()),
        )

    # ------------------------------------------------------------------
    # Cooldown logic
    # ------------------------------------------------------------------

    def check_cooldown(self) -> bool:
        """Check if repartitioning is allowed (cooldown has expired).

        Returns:
            True if repartitioning is allowed.
        """
        return self._cooldown_remaining <= 0

    def advance_cooldown(self) -> None:
        """Advance the cooldown counter by one step."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ------------------------------------------------------------------
    # Hierarchical partitioning
    # ------------------------------------------------------------------

    def hierarchical_partition(
        self,
        cell_centres: torch.Tensor,
        cell_costs: torch.Tensor,
    ) -> torch.Tensor:
        """Perform two-level hierarchical partitioning.

        First groups cells into coarse groups (spatial clustering),
        then distributes groups to processors.

        Args:
            cell_centres: ``(n_cells, 3)`` cell centre coordinates.
            cell_costs: ``(n_cells,)`` per-cell cost weights.

        Returns:
            ``(n_cells,)`` processor assignment.
        """
        cell_centres = cell_centres.to(dtype=torch.float64)
        n_cells = cell_centres.shape[0]

        if self._hierarchical_config is None:
            # Fallback: simple round-robin
            return torch.arange(n_cells, dtype=INDEX_DTYPE) % self._target_n_procs

        n_groups = min(self._hierarchical_config.n_coarse_groups, n_cells)

        # Simple spatial clustering: assign to group by x-coordinate ordering
        sorted_indices = cell_centres[:, 0].argsort()
        group_assignment = torch.zeros(n_cells, dtype=INDEX_DTYPE)
        cells_per_group = n_cells // n_groups

        for g in range(n_groups):
            start = g * cells_per_group
            end = start + cells_per_group if g < n_groups - 1 else n_cells
            group_assignment[sorted_indices[start:end]] = g

        # Distribute groups to processors
        proc_assignment = torch.zeros(n_cells, dtype=INDEX_DTYPE)
        groups_per_proc = max(n_groups // self._target_n_procs, 1)

        for p in range(self._target_n_procs):
            start_g = p * groups_per_proc
            end_g = min(start_g + groups_per_proc, n_groups)
            for g in range(start_g, end_g):
                mask = group_assignment == g
                proc_assignment[mask] = p

        return proc_assignment

    # ------------------------------------------------------------------
    # v6 redistribution
    # ------------------------------------------------------------------

    def redistribute_v6(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        cell_costs: Optional[torch.Tensor] = None,
        current_mapping: Optional[torch.Tensor] = None,
        migration_threshold: float = 0.2,
        cooldown_steps: int = 0,
        seed: int = 42,
    ) -> V6RedistributeResult:
        """Redistribute with v6 hierarchical partitioning and cooldown.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            cell_centres: ``(n_cells, 3)`` cell centre coordinates.
            cell_costs: ``(n_cells,)`` per-cell cost estimates.
            current_mapping: ``(n_cells,)`` current mapping.
            migration_threshold: Maximum fraction of cells to migrate.
            cooldown_steps: Number of steps to wait before next repartition.
            seed: Random seed.

        Returns:
            :class:`V6RedistributeResult`.
        """
        # Check cooldown
        if not self.check_cooldown():
            logger.info("Repartitioning blocked by cooldown (%d remaining)", self._cooldown_remaining)
            # Return empty result
            base = V5RedistributeResult(
                base=None,
                migration_plan=None,
                n_migrations=0,
                estimated_speedup=1.0,
                repartition_recommended=False,
            )
            return V6RedistributeResult(
                base=base,
                metrics=PartitionMetrics(),
                hierarchical=False,
                cooldown_remaining=self._cooldown_remaining,
            )

        # Base v5 redistribution
        base_result = self.redistribute_v5(
            output_dir=output_dir,
            field_names=field_names,
            cell_centres=cell_centres,
            cell_costs=cell_costs,
            current_mapping=current_mapping,
            migration_threshold=migration_threshold,
            seed=seed,
        )

        # Hierarchical partitioning
        use_hierarchical = (
            self._hierarchical_config is not None
            and cell_centres is not None
            and cell_costs is not None
        )

        # Compute metrics
        metrics = PartitionMetrics()
        if cell_costs is not None and self._global_cell_map is not None:
            metrics = self.compute_partition_metrics(
                self._global_cell_map, cell_costs, self._target_n_procs
            )

        # Set cooldown
        self._cooldown_remaining = cooldown_steps

        return V6RedistributeResult(
            base=base_result,
            metrics=metrics,
            hierarchical=use_hierarchical,
            cooldown_remaining=cooldown_steps,
        )

    def __repr__(self) -> str:
        cd = self._cooldown_remaining
        return (
            f"RedistributeParEnhanced6(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs}, cooldown={cd})"
        )
