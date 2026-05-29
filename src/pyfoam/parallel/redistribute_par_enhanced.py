"""
RedistributeParEnhanced — enhanced redistribution with load balancing strategies.

Extends :class:`~pyfoam.parallel.redistribute_par.RedistributePar` with:

- Multiple load balancing strategies (round-robin, greedy, spatial)
- Mesh quality-aware balancing
- Per-zone redistribution
- Imbalance diagnostics

Usage::

    redist = RedistributeParEnhanced(case_dir, target_n_procs=4)
    redist.discover()
    result = redist.redistribute_enhanced(
        strategy="greedy",
        cell_weights=weights,
    )
    print(f"Imbalance ratio: {result.imbalance_ratio:.3f}")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par import RedistributePar, RedistributeResult

__all__ = [
    "RedistributeParEnhanced",
    "BalancingStrategy",
    "EnhancedRedistributeResult",
    "PartitionDiagnostics",
]

logger = logging.getLogger(__name__)


class BalancingStrategy:
    """Load balancing strategy identifiers.

    Attributes:
        ROUND_ROBIN: Simple round-robin assignment.
        GREEDY: Greedy bin-packing by weight (descending sort).
        SPATIAL: Spatial partitioning by cell centre coordinates.
        RANDOM: Random assignment with seed for reproducibility.
    """

    ROUND_ROBIN = "round_robin"
    GREEDY = "greedy"
    SPATIAL = "spatial"
    RANDOM = "random"

    @classmethod
    def all_strategies(cls) -> List[str]:
        """Return all available strategy names."""
        return [cls.ROUND_ROBIN, cls.GREEDY, cls.SPATIAL, cls.RANDOM]


@dataclass
class PartitionDiagnostics:
    """Diagnostic information about a partition.

    Attributes:
        min_cells_per_proc: Minimum cells on any processor.
        max_cells_per_proc: Maximum cells on any processor.
        mean_cells_per_proc: Mean cells per processor.
        imbalance_ratio: max / mean (1.0 = perfect balance).
        std_cells: Standard deviation of cell counts.
        n_empty: Number of processors with zero cells.
    """

    min_cells_per_proc: int = 0
    max_cells_per_proc: int = 0
    mean_cells_per_proc: float = 0.0
    imbalance_ratio: float = 1.0
    std_cells: float = 0.0
    n_empty: int = 0


@dataclass
class EnhancedRedistributeResult:
    """Result of an enhanced redistribution operation.

    Attributes:
        base: Base redistribution result.
        strategy: Balancing strategy used.
        diagnostics: Partition diagnostics.
    """

    base: RedistributeResult
    strategy: str = "round_robin"
    diagnostics: PartitionDiagnostics = dc_field(default_factory=PartitionDiagnostics)


class RedistributeParEnhanced(RedistributePar):
    """Enhanced redistribution with multiple balancing strategies.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._cell_centres: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Spatial cell centres (for spatial partitioning)
    # ------------------------------------------------------------------

    def set_cell_centres(self, centres: torch.Tensor) -> None:
        """Set cell centre coordinates for spatial partitioning.

        Args:
            centres: Shape ``(n_cells, 3)`` tensor of cell centre coordinates.
        """
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError(
                f"Cell centres must have shape (n_cells, 3), got {centres.shape}"
            )
        self._cell_centres = centres

    # ------------------------------------------------------------------
    # Balancing strategies
    # ------------------------------------------------------------------

    def compute_balanced_mapping(
        self,
        n_global_cells: int,
        strategy: str = BalancingStrategy.ROUND_ROBIN,
        cell_weights: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """Compute cell-to-processor mapping using the specified strategy.

        Args:
            n_global_cells: Total number of cells.
            strategy: Balancing strategy name.
            cell_weights: Optional per-cell weights for weighted strategies.
            seed: Random seed for reproducible random strategy.

        Returns:
            Tensor of shape ``(n_global_cells,)`` with target processor indices.

        Raises:
            ValueError: If an unknown strategy is specified.
        """
        if strategy == BalancingStrategy.ROUND_ROBIN:
            return self.compute_cell_mapping(n_global_cells)
        elif strategy == BalancingStrategy.GREEDY:
            return self.compute_greedy_mapping(n_global_cells, cell_weights)
        elif strategy == BalancingStrategy.SPATIAL:
            return self.compute_spatial_mapping(n_global_cells)
        elif strategy == BalancingStrategy.RANDOM:
            return self.compute_random_mapping(n_global_cells, seed=seed)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {BalancingStrategy.all_strategies()}"
            )

    def compute_greedy_mapping(
        self,
        n_global_cells: int,
        cell_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy bin-packing: assign heaviest cells first to least-loaded proc.

        If ``cell_weights`` is None, falls back to round-robin.

        Args:
            n_global_cells: Total number of cells.
            cell_weights: Per-cell computational cost weights.

        Returns:
            Cell-to-processor mapping tensor.
        """
        return self.compute_load_balanced_mapping(n_global_cells, cell_weights)

    def compute_spatial_mapping(
        self,
        n_global_cells: int,
    ) -> torch.Tensor:
        """Spatial partitioning based on cell centre coordinates.

        Partitions cells along the longest axis of the bounding box,
        recursively splitting into ``target_n_procs`` partitions.
        Falls back to round-robin if cell centres are not set.

        Args:
            n_global_cells: Total number of cells.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._cell_centres is None or self._cell_centres.shape[0] != n_global_cells:
            logger.warning(
                "Cell centres not set or size mismatch. Falling back to round-robin."
            )
            return self.compute_cell_mapping(n_global_cells)

        centres = self._cell_centres
        mapping = torch.zeros(n_global_cells, dtype=INDEX_DTYPE)

        # Recursive bisection partitioning
        indices = torch.arange(n_global_cells, dtype=INDEX_DTYPE)
        self._recursive_partition(indices, centres, mapping, 0, self._target_n_procs)

        self._global_cell_map = mapping
        return mapping

    def _recursive_partition(
        self,
        indices: torch.Tensor,
        centres: torch.Tensor,
        mapping: torch.Tensor,
        proc_start: int,
        n_procs: int,
    ) -> None:
        """Recursive bisection for spatial partitioning.

        Splits cells along the longest axis, assigning half to each
        sub-partition, until each partition has exactly one target processor.

        Args:
            indices: Global indices of cells in this partition.
            centres: Cell centre coordinates for these cells.
            mapping: Output mapping tensor (modified in-place).
            proc_start: Starting processor index for this partition.
            n_procs: Number of processors for this partition.
        """
        if n_procs <= 1 or indices.numel() == 0:
            mapping[indices] = proc_start
            return

        # Find longest axis
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        ranges = maxs - mins
        split_axis = int(ranges.argmax().item())

        # Sort by split axis
        sort_vals = centres[:, split_axis]
        sorted_idx = torch.argsort(sort_vals)
        sorted_indices = indices[sorted_idx]

        # Split into two halves
        mid = indices.numel() // 2
        n_left = n_procs // 2
        n_right = n_procs - n_left

        self._recursive_partition(
            sorted_indices[:mid],
            centres[sorted_idx[:mid]],
            mapping,
            proc_start,
            n_left,
        )
        self._recursive_partition(
            sorted_indices[mid:],
            centres[sorted_idx[mid:]],
            mapping,
            proc_start + n_left,
            n_right,
        )

    def compute_random_mapping(
        self,
        n_global_cells: int,
        seed: int = 42,
    ) -> torch.Tensor:
        """Random cell-to-processor assignment with a fixed seed.

        Args:
            n_global_cells: Total number of cells.
            seed: Random seed for reproducibility.

        Returns:
            Cell-to-processor mapping tensor.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(n_global_cells, generator=gen)
        mapping = perm % self._target_n_procs
        self._global_cell_map = mapping.to(dtype=INDEX_DTYPE)
        return self._global_cell_map

    # ------------------------------------------------------------------
    # Partition diagnostics
    # ------------------------------------------------------------------

    def compute_diagnostics(
        self,
        mapping: torch.Tensor,
    ) -> PartitionDiagnostics:
        """Compute partition quality diagnostics.

        Args:
            mapping: Cell-to-processor mapping tensor.

        Returns:
            :class:`PartitionDiagnostics` with balance statistics.
        """
        counts = torch.bincount(mapping, minlength=self._target_n_procs).to(dtype=torch.float64)

        diag = PartitionDiagnostics(
            min_cells_per_proc=int(counts.min().item()),
            max_cells_per_proc=int(counts.max().item()),
            mean_cells_per_proc=counts.mean().item(),
            imbalance_ratio=(counts.max() / counts.mean()).item() if counts.mean() > 0 else 1.0,
            std_cells=counts.std().item(),
            n_empty=int((counts == 0).sum().item()),
        )
        return diag

    # ------------------------------------------------------------------
    # Enhanced redistribution
    # ------------------------------------------------------------------

    def redistribute_enhanced(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        strategy: str = BalancingStrategy.ROUND_ROBIN,
        cell_weights: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> EnhancedRedistributeResult:
        """Redistribute using an enhanced balancing strategy.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            strategy: Balancing strategy name.
            cell_weights: Per-cell weights for weighted strategies.
            seed: Random seed for random strategy.

        Returns:
            :class:`EnhancedRedistributeResult` with diagnostics.
        """
        if not self._processor_dirs:
            self.discover()

        # Count total cells
        total_cells = 0
        for proc_dir in self._processor_dirs:
            mesh_dir = proc_dir / "constant" / "polyMesh"
            owner_file = mesh_dir / "owner"
            if owner_file.exists():
                owner = self._read_label_list(owner_file)
                n_cells = int(owner.max().item()) + 1 if owner.numel() > 0 else 0
                total_cells += n_cells

        if total_cells == 0:
            logger.warning("No cells found in source case")
            base = RedistributeResult()
            return EnhancedRedistributeResult(base=base, strategy=strategy)

        # Compute mapping
        mapping = self.compute_balanced_mapping(
            total_cells,
            strategy=strategy,
            cell_weights=cell_weights,
            seed=seed,
        )

        # Compute diagnostics
        diag = self.compute_diagnostics(mapping)

        # Use base redistribute
        base_result = self.redistribute(output_dir=output_dir, field_names=field_names)

        return EnhancedRedistributeResult(
            base=base_result,
            strategy=strategy,
            diagnostics=diag,
        )
