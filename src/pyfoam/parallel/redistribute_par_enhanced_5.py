"""
RedistributeParEnhanced5 — v5 enhanced redistribution with dynamic repartitioning.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_4.RedistributeParEnhanced4` with:

- Dynamic repartitioning based on runtime cost estimation
- Partition migration minimisation (reduce data movement)
- Cost prediction from mesh statistics and solver type
- Repartitioning decision logic (only repartition when beneficial)

Usage::

    redist = RedistributeParEnhanced5(case_dir, target_n_procs=8)
    redist.discover()
    result = redist.redistribute_v5(
        cell_centres=centres,
        cell_costs=costs,
        current_mapping=old_mapping,
        migration_threshold=0.2,
    )
    print(f"Migrations: {result.n_migrations}, benefit: {result.estimated_speedup:.2f}")

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
from pyfoam.parallel.redistribute_par_enhanced_4 import (
    RedistributeParEnhanced4,
    V4RedistributeResult,
    PartitionQualityMetrics,
)

__all__ = [
    "RedistributeParEnhanced5",
    "V5RedistributeResult",
    "MigrationPlan",
    "CostEstimator",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migration planning
# ---------------------------------------------------------------------------


@dataclass
class MigrationPlan:
    """Plan for migrating cells between processors.

    Attributes:
        cell_indices: Indices of cells to migrate.
        source_proc: Source processor for each cell.
        target_proc: Target processor for each cell.
        n_migrations: Total number of cell migrations.
    """

    cell_indices: torch.Tensor
    source_proc: torch.Tensor
    target_proc: torch.Tensor
    n_migrations: int = 0


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class CostEstimator:
    """Estimates computational cost per cell from mesh statistics.

    Uses cell volume, face count, and a solver-type multiplier to
    produce per-cell cost weights.

    Args:
        solver_type: Type of solver (``"incompressible"``,
            ``"compressible"``, ``"multiphase"``).
        base_cost: Base cost per cell.
    """

    SOLVER_MULTIPLIERS = {
        "incompressible": 1.0,
        "compressible": 1.5,
        "multiphase": 2.0,
        "les": 1.8,
    }

    def __init__(
        self,
        solver_type: str = "incompressible",
        base_cost: float = 1.0,
    ) -> None:
        self._solver_type = solver_type
        self._base_cost = base_cost
        self._multiplier = self.SOLVER_MULTIPLIERS.get(
            solver_type, 1.0
        )

    def estimate_costs(
        self,
        cell_volumes: Optional[torch.Tensor] = None,
        n_faces_per_cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate per-cell computational cost.

        Args:
            cell_volumes: ``(n_cells,)`` cell volumes.
            n_faces_per_cell: ``(n_cells,)`` face count per cell.

        Returns:
            ``(n_cells,)`` estimated costs.
        """
        if cell_volumes is not None:
            n_cells = cell_volumes.shape[0]
            costs = self._base_cost * self._multiplier * torch.ones(
                n_cells, dtype=torch.float64
            )
            # Scale by relative volume
            vol_mean = cell_volumes.to(dtype=torch.float64).mean()
            if vol_mean > 1e-30:
                costs *= cell_volumes.to(dtype=torch.float64) / vol_mean
        elif n_faces_per_cell is not None:
            n_cells = n_faces_per_cell.shape[0]
            costs = self._base_cost * self._multiplier * torch.ones(
                n_cells, dtype=torch.float64
            )
            # Scale by face count (more faces = more work)
            face_mean = n_faces_per_cell.to(dtype=torch.float64).mean()
            if face_mean > 1e-30:
                costs *= n_faces_per_cell.to(dtype=torch.float64) / face_mean
        else:
            costs = torch.tensor(
                [self._base_cost * self._multiplier],
                dtype=torch.float64,
            )

        return costs

    @property
    def solver_type(self) -> str:
        return self._solver_type


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V5RedistributeResult:
    """Result of a v5 enhanced redistribution.

    Attributes:
        base: V4 redistribution result.
        migration_plan: Details of cell migrations.
        n_migrations: Number of cells migrated.
        estimated_speedup: Estimated speedup from repartitioning.
        repartition_recommended: Whether repartitioning was recommended.
    """

    base: V4RedistributeResult
    migration_plan: MigrationPlan = None
    n_migrations: int = 0
    estimated_speedup: float = 1.0
    repartition_recommended: bool = True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced5(RedistributeParEnhanced4):
    """v5 enhanced redistribution with dynamic repartitioning.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._cost_estimator: Optional[CostEstimator] = None
        self._current_mapping: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_cost_estimator(self, estimator: CostEstimator) -> None:
        """Set cost estimator for dynamic load balancing.

        Args:
            estimator: CostEstimator instance.
        """
        self._cost_estimator = estimator

    # ------------------------------------------------------------------
    # Migration minimisation
    # ------------------------------------------------------------------

    def compute_migration_plan(
        self,
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
    ) -> MigrationPlan:
        """Compute migration plan minimising data movement.

        Identifies cells that change processor and builds a migration
        plan. Only cells that actually move are included.

        Args:
            old_mapping: ``(n_cells,)`` current cell-to-processor mapping.
            new_mapping: ``(n_cells,)`` desired cell-to-processor mapping.

        Returns:
            :class:`MigrationPlan`.
        """
        old_mapping = old_mapping.to(dtype=INDEX_DTYPE)
        new_mapping = new_mapping.to(dtype=INDEX_DTYPE)

        change_mask = old_mapping != new_mapping
        migration_indices = change_mask.nonzero(as_tuple=True)[0]

        return MigrationPlan(
            cell_indices=migration_indices,
            source_proc=old_mapping[migration_indices],
            target_proc=new_mapping[migration_indices],
            n_migrations=migration_indices.numel(),
        )

    # ------------------------------------------------------------------
    # Repartitioning decision
    # ------------------------------------------------------------------

    def should_repartition(
        self,
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
        cell_costs: torch.Tensor,
        migration_threshold: float = 0.2,
    ) -> tuple[bool, float]:
        """Decide whether repartitioning is beneficial.

        Compares the load imbalance improvement against the migration
        cost. Returns True if the estimated benefit exceeds the threshold.

        Args:
            old_mapping: Current mapping.
            new_mapping: Proposed mapping.
            cell_costs: Per-cell costs.
            migration_threshold: Maximum fraction of cells to migrate.

        Returns:
            Tuple of (recommended, estimated_speedup).
        """
        cell_costs = cell_costs.to(dtype=torch.float64)
        n_cells = cell_costs.shape[0]

        # Compute load imbalance for old and new
        n_procs = self._target_n_procs

        def imbalance(mapping: torch.Tensor) -> float:
            proc_load = torch.zeros(n_procs, dtype=torch.float64)
            for p in range(n_procs):
                mask = mapping == p
                if mask.any():
                    proc_load[p] = cell_costs[mask].sum()
            avg = proc_load.sum() / n_procs
            if avg < 1e-30:
                return 1.0
            return proc_load.max().item() / avg.item()

        old_imb = imbalance(old_mapping)
        new_imb = imbalance(new_mapping)

        # Migration cost
        change_mask = old_mapping != new_mapping
        migration_frac = change_mask.sum().item() / max(n_cells, 1)

        if migration_frac > migration_threshold:
            return False, 1.0

        speedup = old_imb / max(new_imb, 1e-10)
        return speedup > 1.01, speedup

    # ------------------------------------------------------------------
    # v5 redistribution
    # ------------------------------------------------------------------

    def redistribute_v5(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        cell_costs: Optional[torch.Tensor] = None,
        current_mapping: Optional[torch.Tensor] = None,
        migration_threshold: float = 0.2,
        seed: int = 42,
    ) -> V5RedistributeResult:
        """Redistribute with v5 dynamic repartitioning.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            cell_centres: ``(n_cells, 3)`` cell centre coordinates.
            cell_costs: ``(n_cells,)`` per-cell cost estimates.
            current_mapping: ``(n_cells,)`` current cell-to-processor
                mapping (for migration minimisation).
            migration_threshold: Maximum fraction of cells to migrate.
            seed: Random seed.

        Returns:
            :class:`V5RedistributeResult`.
        """
        # Estimate costs if estimator is set and costs not provided
        if cell_costs is None and self._cost_estimator is not None:
            cell_costs = self._cost_estimator.estimate_costs()

        # Base v4 redistribution
        base_result = self.redistribute_v4(
            output_dir=output_dir,
            field_names=field_names,
            cell_centres=cell_centres,
            cell_weights=cell_costs,
            seed=seed,
        )

        # Migration analysis
        n_migrations = 0
        speedup = 1.0
        recommended = True
        plan = None

        if current_mapping is not None and self._global_cell_map is not None:
            plan = self.compute_migration_plan(
                current_mapping, self._global_cell_map
            )
            n_migrations = plan.n_migrations

            if cell_costs is not None:
                recommended, speedup = self.should_repartition(
                    current_mapping,
                    self._global_cell_map,
                    cell_costs,
                    migration_threshold=migration_threshold,
                )

        return V5RedistributeResult(
            base=base_result,
            migration_plan=plan,
            n_migrations=n_migrations,
            estimated_speedup=speedup,
            repartition_recommended=recommended,
        )

    def __repr__(self) -> str:
        return (
            f"RedistributeParEnhanced5(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs})"
        )
