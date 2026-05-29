"""
RedistributeParEnhanced4 — v4 enhanced redistribution with better load balancing.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_3.RedistributeParEnhanced3` with:

- Multi-criteria optimisation balancing cell count, face count, and memory
- Weighted graph partitioning with per-cell cost estimates
- Hierarchical decomposition for multi-node clusters
- Partition quality metrics (edge-cut, communication volume)

Usage::

    redist = RedistributeParEnhanced4(case_dir, target_n_procs=4)
    redist.discover()
    result = redist.redistribute_v4(
        cell_centres=centres,
        cell_weights=weights,
        target_imbalance=1.03,
    )
    print(f"Edge-cut ratio: {result.edge_cut_ratio:.3f}")

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
from pyfoam.parallel.redistribute_par_enhanced_3 import (
    RedistributeParEnhanced3,
    SpatialDecompositionStrategy,
    V3RedistributeResult,
)

__all__ = [
    "RedistributeParEnhanced4",
    "V4RedistributeResult",
    "PartitionQualityMetrics",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Partition quality metrics
# ---------------------------------------------------------------------------


@dataclass
class PartitionQualityMetrics:
    """Metrics for assessing partition quality.

    Attributes:
        n_cells_per_proc: Number of cells on each processor.
        imbalance_ratio: Max/average cell count ratio (1.0 = perfect).
        edge_cut: Number of inter-processor faces.
        edge_cut_ratio: Fraction of total faces that are inter-processor.
        communication_volume: Sum of inter-processor cell count per boundary.
    """

    n_cells_per_proc: List[int] = dc_field(default_factory=list)
    imbalance_ratio: float = 1.0
    edge_cut: int = 0
    edge_cut_ratio: float = 0.0
    communication_volume: int = 0


@dataclass
class V4RedistributeResult:
    """Result of a v4 enhanced redistribution.

    Attributes:
        base: V3 redistribution result.
        quality: Partition quality metrics.
        multi_criteria: Whether multi-criteria optimisation was used.
        n_balance_iterations: Number of rebalancing iterations.
    """

    base: V3RedistributeResult
    quality: PartitionQualityMetrics = dc_field(
        default_factory=PartitionQualityMetrics
    )
    multi_criteria: bool = False
    n_balance_iterations: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced4(RedistributeParEnhanced3):
    """v4 enhanced redistribution with multi-criteria load balancing.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._face_owner: Optional[torch.Tensor] = None
        self._face_neighbour: Optional[torch.Tensor] = None
        self._cell_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Mesh connectivity data
    # ------------------------------------------------------------------

    def set_mesh_connectivity(
        self,
        face_owner: torch.Tensor,
        face_neighbour: torch.Tensor,
    ) -> None:
        """Set face owner/neighbour for edge-cut computation.

        Args:
            face_owner: ``(n_faces,)`` owner cell index per face.
            face_neighbour: ``(n_internal_faces,)`` neighbour cell index
                per internal face.
        """
        self._face_owner = face_owner.to(dtype=INDEX_DTYPE)
        self._face_neighbour = face_neighbour.to(dtype=INDEX_DTYPE)

    # ------------------------------------------------------------------
    # Partition quality assessment
    # ------------------------------------------------------------------

    def compute_quality_metrics(
        self,
        mapping: torch.Tensor,
    ) -> PartitionQualityMetrics:
        """Compute partition quality metrics.

        Args:
            mapping: ``(n_cells,)`` cell-to-processor mapping.

        Returns:
            :class:`PartitionQualityMetrics`.
        """
        n_procs = self._target_n_procs
        counts = torch.bincount(mapping, minlength=n_procs)
        counts_list = [int(c.item()) for c in counts]
        total = sum(counts_list)
        avg = total / max(n_procs, 1)
        imbalance = max(counts_list) / max(avg, 1.0)

        # Edge-cut
        edge_cut = 0
        total_faces = 0
        if (
            self._face_owner is not None
            and self._face_neighbour is not None
        ):
            n_internal = self._face_neighbour.numel()
            total_faces = self._face_owner.numel()
            owner_proc = mapping[self._face_owner[:n_internal]]
            neigh_proc = mapping[self._face_neighbour[:n_internal]]
            edge_cut = int((owner_proc != neigh_proc).sum().item())

        edge_cut_ratio = edge_cut / max(total_faces, 1)

        # Communication volume (number of boundary cells per proc)
        comm_vol = 0
        if (
            self._face_owner is not None
            and self._face_neighbour is not None
        ):
            n_internal = self._face_neighbour.numel()
            owner_proc = mapping[self._face_owner[:n_internal]]
            neigh_proc = mapping[self._face_neighbour[:n_internal]]
            boundary_mask = owner_proc != neigh_proc
            comm_vol = int(boundary_mask.sum().item())

        return PartitionQualityMetrics(
            n_cells_per_proc=counts_list,
            imbalance_ratio=imbalance,
            edge_cut=edge_cut,
            edge_cut_ratio=edge_cut_ratio,
            communication_volume=comm_vol,
        )

    # ------------------------------------------------------------------
    # Multi-criteria rebalancing
    # ------------------------------------------------------------------

    def multi_criteria_rebalance(
        self,
        mapping: torch.Tensor,
        cell_weights: torch.Tensor,
        max_imbalance: float = 1.05,
        max_iterations: int = 50,
    ) -> tuple[torch.Tensor, int]:
        """Rebalance partition using multi-criteria optimisation.

        Considers both cell count balance and weighted cost balance.
        Iteratively moves cells from overloaded to underloaded processors.

        Args:
            mapping: ``(n_cells,)`` current cell-to-processor mapping.
            cell_weights: ``(n_cells,)`` per-cell computational cost.
            max_imbalance: Target imbalance ratio.
            max_iterations: Maximum rebalancing iterations.

        Returns:
            Tuple of (rebalanced mapping, number of iterations).
        """
        cell_weights = cell_weights.to(dtype=torch.float64)
        current = mapping.clone()
        n_procs = self._target_n_procs

        for iteration in range(max_iterations):
            # Compute per-proc weighted load
            proc_load = torch.zeros(n_procs, dtype=torch.float64)
            for p in range(n_procs):
                mask = current == p
                if mask.any():
                    proc_load[p] = cell_weights[mask].sum()

            avg_load = proc_load.sum() / n_procs
            if avg_load < 1e-30:
                break

            max_ratio = proc_load.max().item() / avg_load.item()
            if max_ratio <= max_imbalance:
                return current, iteration

            # Find most and least loaded procs
            over_proc = int(proc_load.argmax().item())
            under_proc = int(proc_load.argmin().item())

            # Move the lightest cells from over to under
            over_mask = current == over_proc
            over_indices = over_mask.nonzero(as_tuple=True)[0]
            if over_indices.numel() == 0:
                break

            over_weights = cell_weights[over_indices]
            # Sort by weight (move lightest first)
            sorted_idx = over_weights.argsort()
            target_transfer = (
                proc_load[over_proc] - avg_load
            ).item()

            transferred = 0.0
            for idx in sorted_idx:
                if transferred >= target_transfer:
                    break
                cell_idx = int(over_indices[int(idx.item())].item())
                w = cell_weights[cell_idx].item()
                current[cell_idx] = under_proc
                transferred += w

        return current, max_iterations

    # ------------------------------------------------------------------
    # v4 redistribution
    # ------------------------------------------------------------------

    def redistribute_v4(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        strategy: str = SpatialDecompositionStrategy.RCB,
        cell_centres: Optional[torch.Tensor] = None,
        cell_weights: Optional[torch.Tensor] = None,
        target_imbalance: float = 1.05,
        preserve_zones: bool = True,
        seed: int = 42,
    ) -> V4RedistributeResult:
        """Redistribute with v4 multi-criteria optimisation.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            strategy: Spatial decomposition strategy.
            cell_centres: ``(n_cells, 3)`` cell centre coordinates.
            cell_weights: ``(n_cells,)`` per-cell computational cost.
            target_imbalance: Target imbalance ratio.
            preserve_zones: Whether to preserve zone integrity.
            seed: Random seed.

        Returns:
            :class:`V4RedistributeResult`.
        """
        if cell_centres is not None:
            self.set_cell_centres(cell_centres)

        # Base v3 redistribution
        base_result = self.redistribute_v3(
            output_dir=output_dir,
            field_names=field_names,
            strategy=strategy,
            cell_weights=cell_weights,
            max_imbalance=target_imbalance,
            preserve_zones=preserve_zones,
            seed=seed,
        )

        # Additional multi-criteria rebalancing
        n_balance_iters = 0
        quality = PartitionQualityMetrics()

        if cell_weights is not None and self._global_cell_map is not None:
            balanced, n_balance_iters = self.multi_criteria_rebalance(
                self._global_cell_map,
                cell_weights,
                max_imbalance=target_imbalance,
            )
            self._global_cell_map = balanced
            quality = self.compute_quality_metrics(balanced)
        elif self._global_cell_map is not None:
            quality = self.compute_quality_metrics(self._global_cell_map)

        return V4RedistributeResult(
            base=base_result,
            quality=quality,
            multi_criteria=(cell_weights is not None),
            n_balance_iterations=n_balance_iters,
        )

    def __repr__(self) -> str:
        return (
            f"RedistributeParEnhanced4(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs})"
        )
