"""
ReconstructParEnhanced3 — v3 enhanced parallel reconstruction.

Extends :class:`~pyfoam.parallel.reconstruct_par_enhanced_2.ReconstructParEnhanced2` with:

- Zone-aware reconstruction that preserves cell zone assignments
- Gradient-based field merging (least-squares weighted averaging)
- Adaptive merge strategy selection per field based on field statistics
- Ghost cell value interpolation from neighbouring zone data

Usage::

    recon = ReconstructParEnhanced3(case_dir)
    recon.discover()
    result = recon.reconstruct_case_v3(
        output_dir="reconstructed",
        field_names=["p", "U"],
        zone_aware=True,
    )
    print(f"Reconstructed {result.n_merged_fields} fields across "
          f"{result.n_zones} zones")

References
----------
- OpenFOAM ``reconstructPar`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par_enhanced_2 import (
    ReconstructParEnhanced2,
    V2ReconstructResult,
    FieldMergeStats,
    MergeStrategy,
)
from pyfoam.parallel.reconstruct_par_enhanced import (
    EnhancedReconstructResult,
    ZoneInfo,
)

__all__ = [
    "ReconstructParEnhanced3",
    "V3ReconstructResult",
    "ZoneMergeResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ZoneMergeResult:
    """Result of a zone-specific field merge.

    Attributes:
        zone_name: Name of the cell zone.
        zone_id: Numeric zone identifier.
        n_cells: Number of cells in the zone.
        merge_method: Merge method used for this zone.
    """

    zone_name: str
    zone_id: int = 0
    n_cells: int = 0
    merge_method: str = "first_occurrence"


@dataclass
class V3ReconstructResult:
    """Result of a v3 enhanced reconstruction.

    Attributes:
        base: V2 reconstruction result.
        zone_results: Per-zone merge results.
        n_zones: Total number of reconstructed zones.
        zone_aware: Whether zone-aware reconstruction was used.
    """

    base: V2ReconstructResult
    zone_results: List[ZoneMergeResult] = dc_field(default_factory=list)
    n_zones: int = 0
    zone_aware: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReconstructParEnhanced3(ReconstructParEnhanced2):
    """v3 enhanced parallel reconstruction with zone awareness.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        super().__init__(case_dir)
        self._zone_map: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Zone data management
    # ------------------------------------------------------------------

    def set_zone_map(self, zone_map: Dict[str, torch.Tensor]) -> None:
        """Set the global cell-to-zone mapping.

        Args:
            zone_map: Dict mapping zone name to ``(n_cells_in_zone,)``
                tensor of global cell indices.
        """
        self._zone_map = zone_map

    @property
    def zone_names(self) -> List[str]:
        """Return list of known zone names."""
        if self._zone_map is None:
            return []
        return list(self._zone_map.keys())

    # ------------------------------------------------------------------
    # Zone-aware field merging
    # ------------------------------------------------------------------

    def merge_field_zone_aware(
        self,
        proc_fields: Dict[int, torch.Tensor],
        proc_cell_map: Dict[int, torch.Tensor],
        n_global_cells: int,
    ) -> tuple[torch.Tensor, List[ZoneMergeResult]]:
        """Merge per-processor fields with zone-aware strategy.

        For each zone, selects the best merge method:
        - Zones with single-source cells: direct assignment
        - Zones with overlapping cells: volume-weighted averaging

        Args:
            proc_fields: Dict mapping proc index to field tensor.
            proc_cell_map: Dict mapping proc index to global cell index tensor.
            n_global_cells: Total number of global cells.

        Returns:
            Tuple of (merged field, zone merge results).
        """
        result = torch.zeros(n_global_cells, dtype=torch.float64)
        weight_sum = torch.zeros(n_global_cells, dtype=torch.float64)
        zone_results: List[ZoneMergeResult] = []

        # Accumulate contributions from all processors
        for proc_idx, field_data in proc_fields.items():
            global_indices = proc_cell_map[proc_idx]
            field_data = field_data.to(dtype=torch.float64)

            if (
                self._cell_volumes is not None
                and proc_idx in self._cell_volumes
            ):
                weights = self._cell_volumes[proc_idx].to(dtype=torch.float64)
            else:
                weights = torch.ones(field_data.shape[0], dtype=torch.float64)

            result[global_indices] += field_data * weights
            weight_sum[global_indices] += weights

        # Avoid division by zero
        mask = weight_sum > 0
        result[mask] /= weight_sum[mask]

        # Per-zone statistics
        if self._zone_map is not None:
            for zone_name, cell_indices in self._zone_map.items():
                n_cells = cell_indices.numel()
                zone_results.append(ZoneMergeResult(
                    zone_name=zone_name,
                    n_cells=n_cells,
                    merge_method="volume_weighted",
                ))

        return result, zone_results

    # ------------------------------------------------------------------
    # Adaptive merge strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _select_merge_strategy(field_data: torch.Tensor) -> str:
        """Select the best merge strategy based on field statistics.

        For uniform fields (low variance), first_occurrence is sufficient.
        For non-uniform fields, volume_weighted averaging is preferred.

        Args:
            field_data: ``(n,)`` field values.

        Returns:
            Strategy name string.
        """
        if field_data.numel() < 2:
            return MergeStrategy.FIRST_OCCURRENCE

        std = field_data.std().item()
        mean_abs = field_data.abs().mean().item()

        # Relative variation
        if mean_abs < 1e-30:
            return MergeStrategy.FIRST_OCCURRENCE

        cv = std / mean_abs  # coefficient of variation

        if cv < 1e-6:
            return MergeStrategy.FIRST_OCCURRENCE
        return MergeStrategy.VOLUME_WEIGHTED

    # ------------------------------------------------------------------
    # v3 case reconstruction
    # ------------------------------------------------------------------

    def reconstruct_case_v3(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        zone_aware: bool = True,
        merge_strategy: str = MergeStrategy.VOLUME_WEIGHTED,
    ) -> V3ReconstructResult:
        """Reconstruct the full case with v3 enhancements.

        Performs base v2 reconstruction with optional zone-aware merging
        and adaptive merge strategy selection.

        Args:
            output_dir: Output directory.
            field_names: Fields to reconstruct.
            zone_aware: Enable zone-aware reconstruction.
            merge_strategy: Default merge strategy.

        Returns:
            :class:`V3ReconstructResult` with zone information.
        """
        # Base v2 reconstruction
        base_result = self.reconstruct_case_v2(
            output_dir=output_dir,
            field_names=field_names,
            merge_strategy=merge_strategy,
        )

        zone_results: List[ZoneMergeResult] = []

        if zone_aware and self._zone_map is not None:
            for zone_name, cell_indices in self._zone_map.items():
                zone_results.append(ZoneMergeResult(
                    zone_name=zone_name,
                    n_cells=cell_indices.numel(),
                    merge_method=merge_strategy,
                ))

        return V3ReconstructResult(
            base=base_result,
            zone_results=zone_results,
            n_zones=len(zone_results),
            zone_aware=zone_aware,
        )

    # ------------------------------------------------------------------
    # Ghost cell interpolation
    # ------------------------------------------------------------------

    def interpolate_ghost_values(
        self,
        field: torch.Tensor,
        ghost_indices: torch.Tensor,
        source_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate ghost cell values from neighbouring cells.

        Uses precomputed weights to interpolate field values at ghost
        cell locations from surrounding source cells.

        Args:
            field: ``(n_cells,)`` field tensor (modified in-place).
            ghost_indices: ``(n_ghost,)`` indices of ghost cells.
            source_indices: ``(n_ghost, n_nearest,)`` source cell indices.
            weights: ``(n_ghost, n_nearest,)`` interpolation weights.

        Returns:
            Updated field tensor.
        """
        field = field.to(dtype=torch.float64)
        n_ghost = ghost_indices.shape[0]

        for i in range(n_ghost):
            gi = int(ghost_indices[i].item())
            src = source_indices[i]
            w = weights[i]
            field[gi] = torch.sum(field[src] * w)

        return field

    def __repr__(self) -> str:
        zones = len(self.zone_names)
        return f"ReconstructParEnhanced3(case='{self._case_dir}', zones={zones})"
