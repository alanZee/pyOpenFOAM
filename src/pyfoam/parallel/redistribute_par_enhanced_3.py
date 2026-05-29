"""
RedistributeParEnhanced3 — v3 enhanced redistribution with spatial decomposition.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced_2.RedistributeParEnhanced2` with:

- Spatial (coordinate-based) decomposition for geometric locality
- Recursive Coordinate Bisection (RCB) partitioning
- Zone-aware redistribution preserving zone integrity
- Multi-criteria optimisation balancing cell count, face count, and memory

Usage::

    redist = RedistributeParEnhanced3(case_dir, target_n_procs=4)
    redist.discover()
    result = redist.redistribute_v3(
        strategy="spatial",
        cell_centres=centres,
        max_imbalance=1.05,
    )
    print(f"Final imbalance: {result.final_diagnostics.imbalance_ratio:.3f}")

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
from pyfoam.parallel.redistribute_par_enhanced_2 import (
    RedistributeParEnhanced2,
    V2RedistributeResult,
    GraphPartitionStrategy,
)
from pyfoam.parallel.redistribute_par_enhanced import (
    EnhancedRedistributeResult,
    PartitionDiagnostics,
    BalancingStrategy,
)
from pyfoam.parallel.redistribute_par import RedistributeResult

__all__ = [
    "RedistributeParEnhanced3",
    "SpatialDecompositionStrategy",
    "V3RedistributeResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial decomposition strategy
# ---------------------------------------------------------------------------


class SpatialDecompositionStrategy:
    """Spatial decomposition strategy identifiers.

    Attributes:
        RCB: Recursive Coordinate Bisection.
        RIB: Recursive Inertial Bisection (uses inertia tensor).
        SCATTER: Scatter cells evenly in spatial blocks.
    """

    RCB = "spatial_rcb"
    RIB = "spatial_rib"
    SCATTER = "spatial_scatter"

    @classmethod
    def all_strategies(cls) -> List[str]:
        return [cls.RCB, cls.RIB, cls.SCATTER]


@dataclass
class V3RedistributeResult:
    """Result of a v3 enhanced redistribution.

    Attributes:
        base: V2 redistribution result.
        spatial_strategy: Spatial strategy used.
        zone_preserved: Whether zone integrity was maintained.
        n_zones_preserved: Number of zones whose cells stayed on same proc.
    """

    base: V2RedistributeResult
    spatial_strategy: str = ""
    zone_preserved: bool = False
    n_zones_preserved: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced3(RedistributeParEnhanced2):
    """v3 enhanced redistribution with spatial decomposition.

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
        self._zone_map: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Cell centre data
    # ------------------------------------------------------------------

    def set_cell_centres(self, centres: torch.Tensor) -> None:
        """Set per-cell centre coordinates for spatial decomposition.

        Args:
            centres: ``(n_cells, 3)`` cell centre coordinates.
        """
        self._cell_centres = centres.to(dtype=torch.float64)

    def set_zone_map(self, zone_map: Dict[str, torch.Tensor]) -> None:
        """Set the global cell-to-zone mapping for zone-aware redistribution.

        Args:
            zone_map: Dict mapping zone name to ``(n_cells_in_zone,)``
                tensor of global cell indices.
        """
        self._zone_map = zone_map

    # ------------------------------------------------------------------
    # Recursive Coordinate Bisection (RCB)
    # ------------------------------------------------------------------

    def compute_rcb_mapping(
        self,
        n_global_cells: int,
    ) -> torch.Tensor:
        """Compute cell-to-processor mapping using RCB.

        Recursively bisects along the longest dimension of the bounding
        box, splitting cells into two equal halves at the median
        coordinate.

        Args:
            n_global_cells: Total number of cells.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._cell_centres is None:
            logger.warning("Cell centres not set. Falling back to round-robin.")
            return self.compute_cell_mapping(n_global_cells)

        mapping = torch.zeros(n_global_cells, dtype=INDEX_DTYPE)
        indices = torch.arange(n_global_cells, dtype=INDEX_DTYPE)
        self._rcb_partition(indices, mapping, 0, self._target_n_procs)
        self._global_cell_map = mapping
        return mapping

    def _rcb_partition(
        self,
        indices: torch.Tensor,
        mapping: torch.Tensor,
        proc_start: int,
        n_procs: int,
    ) -> None:
        """Recursive coordinate bisection helper."""
        if n_procs <= 1 or indices.numel() == 0:
            mapping[indices] = proc_start
            return

        # Find longest dimension
        centres = self._cell_centres[indices]
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        ranges = maxs - mins
        split_dim = int(ranges.argmax().item())

        # Split at median
        coords = centres[:, split_dim]
        median_val = coords.median()
        mask_left = coords <= median_val

        left_indices = indices[mask_left]
        right_indices = indices[~mask_left]

        # Handle degenerate cases
        if left_indices.numel() == 0:
            mapping[right_indices] = proc_start
            return
        if right_indices.numel() == 0:
            mapping[left_indices] = proc_start
            return

        n_left = n_procs // 2
        n_right = n_procs - n_left

        self._rcb_partition(left_indices, mapping, proc_start, n_left)
        self._rcb_partition(right_indices, mapping, proc_start + n_left, n_right)

    # ------------------------------------------------------------------
    # Recursive Inertial Bisection (RIB)
    # ------------------------------------------------------------------

    def compute_rib_mapping(
        self,
        n_global_cells: int,
    ) -> torch.Tensor:
        """Compute cell-to-processor mapping using RIB.

        Uses the inertia tensor of cell positions to find the axis
        of least inertia, then bisects along that axis.

        Args:
            n_global_cells: Total number of cells.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._cell_centres is None:
            logger.warning("Cell centres not set. Falling back to round-robin.")
            return self.compute_cell_mapping(n_global_cells)

        mapping = torch.zeros(n_global_cells, dtype=INDEX_DTYPE)
        indices = torch.arange(n_global_cells, dtype=INDEX_DTYPE)
        self._rib_partition(indices, mapping, 0, self._target_n_procs)
        self._global_cell_map = mapping
        return mapping

    def _rib_partition(
        self,
        indices: torch.Tensor,
        mapping: torch.Tensor,
        proc_start: int,
        n_procs: int,
    ) -> None:
        """Recursive inertial bisection helper."""
        if n_procs <= 1 or indices.numel() == 0:
            mapping[indices] = proc_start
            return

        centres = self._cell_centres[indices]
        centroid = centres.mean(dim=0, keepdim=True)
        shifted = centres - centroid

        # Inertia tensor: J = sum(r_i * r_i^T) for each cell
        # For 3D: J is 3x3 symmetric
        J = shifted.T @ shifted

        # Eigenvector of smallest eigenvalue = axis of least inertia
        try:
            eigvals, eigvecs = torch.linalg.eigh(J)
            split_axis = eigvecs[:, 0]  # smallest eigenvalue
        except Exception:
            # Fallback to RCB logic
            self._rcb_partition(indices, mapping, proc_start, n_procs)
            return

        # Project cells onto split axis and bisect at median
        projections = shifted @ split_axis
        median_val = projections.median()
        mask_left = projections <= median_val

        left_indices = indices[mask_left]
        right_indices = indices[~mask_left]

        if left_indices.numel() == 0:
            mapping[right_indices] = proc_start
            return
        if right_indices.numel() == 0:
            mapping[left_indices] = proc_start
            return

        n_left = n_procs // 2
        n_right = n_procs - n_left

        self._rib_partition(left_indices, mapping, proc_start, n_left)
        self._rib_partition(right_indices, mapping, proc_start + n_left, n_right)

    # ------------------------------------------------------------------
    # Spatial scatter decomposition
    # ------------------------------------------------------------------

    def compute_scatter_mapping(
        self,
        n_global_cells: int,
    ) -> torch.Tensor:
        """Compute cell-to-processor mapping using spatial scatter.

        Divides the bounding box into ``target_n_procs`` equal blocks
        and assigns cells to the block they fall in.

        Args:
            n_global_cells: Total number of cells.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._cell_centres is None:
            logger.warning("Cell centres not set. Falling back to round-robin.")
            return self.compute_cell_mapping(n_global_cells)

        centres = self._cell_centres
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        ranges = maxs - mins + 1e-15

        # Normalise to [0, 1]
        normalised = (centres - mins) / ranges

        # Encode 3D position to a single scalar using Hilbert-like ordering
        # Simple approach: z-order (Morton code) approximation
        n_per_axis = max(2, int(self._target_n_procs ** (1.0 / 3.0)))
        grid_idx = torch.clamp(
            (normalised * n_per_axis).long(), 0, n_per_axis - 1
        )

        # Map 3D grid index to linear index
        linear = (
            grid_idx[:, 0]
            + grid_idx[:, 1] * n_per_axis
            + grid_idx[:, 2] * n_per_axis * n_per_axis
        )

        # Distribute linear indices to processors
        mapping = (linear % self._target_n_procs).to(dtype=INDEX_DTYPE)
        self._global_cell_map = mapping
        return mapping

    # ------------------------------------------------------------------
    # Zone-aware redistribution
    # ------------------------------------------------------------------

    def preserve_zones(
        self,
        mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Adjust mapping to keep zone cells together where possible.

        For each zone, if cells are split across many processors,
        consolidate to the processor that already owns the most cells.

        Args:
            mapping: Current cell-to-processor mapping.

        Returns:
            Tuple of (updated mapping, n_zones_preserved).
        """
        if self._zone_map is None:
            return mapping, 0

        current = mapping.clone()
        n_preserved = 0

        for zone_name, cell_indices in self._zone_map.items():
            if cell_indices.numel() == 0:
                continue

            zone_procs = current[cell_indices]
            proc_counts = torch.bincount(
                zone_procs, minlength=self._target_n_procs
            ).to(dtype=torch.float64)

            # If zone is already mostly on one proc, consolidate
            dominant_proc = int(proc_counts.argmax().item())
            dominant_frac = proc_counts[dominant_proc] / cell_indices.numel()

            if dominant_frac > 0.5:
                # Move all zone cells to the dominant processor
                current[cell_indices] = dominant_proc
                n_preserved += 1

        return current, n_preserved

    # ------------------------------------------------------------------
    # v3 redistribution
    # ------------------------------------------------------------------

    def redistribute_v3(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        strategy: str = SpatialDecompositionStrategy.RCB,
        cell_weights: Optional[torch.Tensor] = None,
        max_imbalance: float = 1.05,
        preserve_zones: bool = True,
        seed: int = 42,
    ) -> V3RedistributeResult:
        """Redistribute with v3 spatial decomposition.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            strategy: Spatial decomposition strategy.
            cell_weights: Per-cell weights.
            max_imbalance: Target imbalance ratio.
            preserve_zones: Whether to preserve zone integrity.
            seed: Random seed.

        Returns:
            :class:`V3RedistributeResult`.
        """
        try:
            if not self._processor_dirs:
                self.discover()
        except FileNotFoundError:
            pass

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
            enhanced = EnhancedRedistributeResult(base=base, strategy=strategy)
            v2 = V2RedistributeResult(base=enhanced)
            return V3RedistributeResult(base=v2, spatial_strategy=strategy)

        # Compute mapping based on strategy
        if strategy == SpatialDecompositionStrategy.RCB:
            mapping = self.compute_rcb_mapping(total_cells)
        elif strategy == SpatialDecompositionStrategy.RIB:
            mapping = self.compute_rib_mapping(total_cells)
        elif strategy == SpatialDecompositionStrategy.SCATTER:
            mapping = self.compute_scatter_mapping(total_cells)
        else:
            # Fall back to v2 graph-based
            return V3RedistributeResult(
                base=super().redistribute_v2(
                    output_dir=output_dir,
                    field_names=field_names,
                    strategy=strategy,
                    cell_weights=cell_weights,
                    max_imbalance=max_imbalance,
                    seed=seed,
                ),
                spatial_strategy=strategy,
            )

        initial_diag = self.compute_diagnostics(mapping)

        # Adaptive rebalancing
        balanced, n_iters, converged = self.adaptive_rebalance(
            mapping, max_imbalance=max_imbalance
        )

        # Zone preservation
        n_zones_preserved = 0
        if preserve_zones:
            balanced, n_zones_preserved = self.preserve_zones(balanced)

        final_diag = self.compute_diagnostics(balanced)
        self._global_cell_map = balanced

        # Base redistribution
        base_result = self.redistribute(
            output_dir=output_dir, field_names=field_names
        )

        enhanced = EnhancedRedistributeResult(
            base=base_result,
            strategy=strategy,
            diagnostics=final_diag,
        )

        v2_result = V2RedistributeResult(
            base=enhanced,
            initial_diagnostics=initial_diag,
            final_diagnostics=final_diag,
            n_iterations=n_iters,
            converged=converged,
        )

        return V3RedistributeResult(
            base=v2_result,
            spatial_strategy=strategy,
            zone_preserved=preserve_zones,
            n_zones_preserved=n_zones_preserved,
        )

    def __repr__(self) -> str:
        return (
            f"RedistributeParEnhanced3(case='{self._case_dir}', "
            f"n_procs={self._target_n_procs})"
        )
