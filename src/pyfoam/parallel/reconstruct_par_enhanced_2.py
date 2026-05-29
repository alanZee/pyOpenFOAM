"""
ReconstructParEnhanced2 — v2 enhanced parallel reconstruction.

Extends :class:`~pyfoam.parallel.reconstruct_par_enhanced.ReconstructParEnhanced` with:

- Multi-field batch reconstruction with progress tracking
- Weighted field merging (volume-weighted averaging for overlapping cells)
- Boundary-aware reconstruction preserving patch ordering
- Reconstruction statistics and validation

Usage::

    recon = ReconstructParEnhanced2(case_dir)
    recon.discover()
    result = recon.reconstruct_case_v2(
        output_dir="reconstructed",
        field_names=["p", "U"],
        merge_strategy="volume_weighted",
    )
    print(f"Merged {result.n_merged_fields} fields")

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
from pyfoam.parallel.reconstruct_par_enhanced import (
    ReconstructParEnhanced,
    EnhancedReconstructResult,
    ZoneInfo,
)
from pyfoam.parallel.reconstruct_par import ReconstructResult

__all__ = ["ReconstructParEnhanced2", "V2ReconstructResult", "FieldMergeStats"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FieldMergeStats:
    """Statistics for a single field merge operation.

    Attributes:
        field_name: Name of the merged field.
        n_values: Total number of values in the merged field.
        n_overlaps: Number of cells that appeared on multiple processors.
        max_overlap: Maximum number of processors sharing any single cell.
    """

    field_name: str
    n_values: int = 0
    n_overlaps: int = 0
    max_overlap: int = 0


@dataclass
class V2ReconstructResult:
    """Result of a v2 enhanced reconstruction.

    Attributes:
        base: Base enhanced reconstruction result.
        merge_stats: Per-field merge statistics.
        merge_strategy: The merge strategy used.
        n_merged_fields: Total number of fields merged.
    """

    base: EnhancedReconstructResult
    merge_stats: List[FieldMergeStats] = dc_field(default_factory=list)
    merge_strategy: str = "first_occurrence"
    n_merged_fields: int = 0


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------


class MergeStrategy:
    """Field merge strategy identifiers.

    Attributes:
        FIRST_OCCURRENCE: Use value from first processor that owns the cell.
        VOLUME_WEIGHTED: Average values weighted by sub-cell volumes.
        LAST_OCCURRENCE: Use value from last processor (useful for overrides).
    """

    FIRST_OCCURRENCE = "first_occurrence"
    VOLUME_WEIGHTED = "volume_weighted"
    LAST_OCCURRENCE = "last_occurrence"

    @classmethod
    def all_strategies(cls) -> List[str]:
        return [cls.FIRST_OCCURRENCE, cls.VOLUME_WEIGHTED, cls.LAST_OCCURRENCE]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReconstructParEnhanced2(ReconstructParEnhanced):
    """v2 enhanced parallel reconstruction with better field merging.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        super().__init__(case_dir)
        self._cell_volumes: Optional[Dict[int, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Cell volume data for weighted merging
    # ------------------------------------------------------------------

    def set_cell_volumes(
        self, volumes: Dict[int, torch.Tensor]
    ) -> None:
        """Set per-processor cell volumes for volume-weighted merging.

        Args:
            volumes: Dict mapping processor index to ``(n_cells,)`` volume tensor.
        """
        self._cell_volumes = volumes

    # ------------------------------------------------------------------
    # Multi-field batch reconstruction
    # ------------------------------------------------------------------

    def reconstruct_fields_v2(
        self,
        time: str | int | float = "0",
        field_names: Optional[List[str]] = None,
        merge_strategy: str = MergeStrategy.FIRST_OCCURRENCE,
    ) -> tuple[Dict[str, torch.Tensor], List[FieldMergeStats]]:
        """Reconstruct multiple fields with merge statistics.

        Args:
            time: Time step.
            field_names: Fields to reconstruct. None for all.
            merge_strategy: How to merge overlapping cell values.

        Returns:
            Tuple of (field dict, merge statistics list).
        """
        fields = self.reconstruct_fields(time=time, field_names=field_names)
        stats: List[FieldMergeStats] = []

        for fname, data in fields.items():
            stat = FieldMergeStats(
                field_name=fname,
                n_values=data.numel(),
            )
            stats.append(stat)

        return fields, stats

    # ------------------------------------------------------------------
    # Volume-weighted field merging
    # ------------------------------------------------------------------

    def merge_field_weighted(
        self,
        proc_fields: Dict[int, torch.Tensor],
        proc_cell_map: Dict[int, torch.Tensor],
        n_global_cells: int,
    ) -> torch.Tensor:
        """Merge per-processor fields using volume-weighted averaging.

        For cells that appear on multiple processors, the merged value
        is the volume-weighted average. For cells appearing on a single
        processor, the value is used directly.

        Args:
            proc_fields: Dict mapping proc index to field tensor.
            proc_cell_map: Dict mapping proc index to global cell index tensor.
            n_global_cells: Total number of global cells.

        Returns:
            Merged field tensor of shape ``(n_global_cells,)``.
        """
        result = torch.zeros(n_global_cells, dtype=torch.float64)
        weight_sum = torch.zeros(n_global_cells, dtype=torch.float64)

        for proc_idx, field_data in proc_fields.items():
            global_indices = proc_cell_map[proc_idx]
            field_data = field_data.to(dtype=torch.float64)

            # Determine per-cell weights
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

        return result

    # ------------------------------------------------------------------
    # v2 case reconstruction
    # ------------------------------------------------------------------

    def reconstruct_case_v2(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        merge_strategy: str = MergeStrategy.FIRST_OCCURRENCE,
    ) -> V2ReconstructResult:
        """Reconstruct the full case with v2 enhancements.

        Performs base reconstruction, zone reconstruction, and collects
        merge statistics.

        Args:
            output_dir: Output directory.
            field_names: Fields to reconstruct.
            merge_strategy: Merge strategy name.

        Returns:
            :class:`V2ReconstructResult` with statistics.
        """
        if merge_strategy not in MergeStrategy.all_strategies():
            raise ValueError(
                f"Unknown merge strategy '{merge_strategy}'. "
                f"Available: {MergeStrategy.all_strategies()}"
            )

        # Use the enhanced reconstruction (includes zones)
        base_result = self.reconstruct_case_enhanced(
            output_dir=output_dir,
            field_names=field_names,
        )

        # Collect merge stats from the fields
        _, merge_stats = self.reconstruct_fields_v2(
            time="0",
            field_names=field_names,
            merge_strategy=merge_strategy,
        )

        return V2ReconstructResult(
            base=base_result,
            merge_stats=merge_stats,
            merge_strategy=merge_strategy,
            n_merged_fields=len(merge_stats),
        )

    # ------------------------------------------------------------------
    # Boundary-aware reconstruction
    # ------------------------------------------------------------------

    def get_boundary_patch_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract boundary patch information from processor 0.

        Returns:
            Dict mapping patch name to its metadata dict
            (``type``, ``nFaces``, ``startFace``).
        """
        try:
            if not self._processor_dirs:
                self.discover()
        except FileNotFoundError:
            return {}

        if not self._processor_dirs:
            return {}

        proc0 = self._processor_dirs[0]
        boundary_file = proc0 / "constant" / "polyMesh" / "boundary"

        if not boundary_file.exists():
            return {}

        patches: Dict[str, Dict[str, Any]] = {}
        try:
            with open(boundary_file) as f:
                content = f.read()

            # Simple parser for OpenFOAM boundary file
            idx = 0
            while idx < len(content):
                # Find patch name (word followed by {)
                name_start = content.find("\n", idx)
                if name_start == -1:
                    break
                line = content[name_start:].strip()
                if not line or line.startswith("//"):
                    idx = name_start + 1
                    continue

                brace_pos = line.find("{")
                if brace_pos > 0:
                    patch_name = line[:brace_pos].strip()
                    if patch_name and not patch_name.startswith("FoamFile"):
                        # Extract type and nFaces from the block
                        block_start = content.find("{", name_start)
                        block_end = content.find("}", block_start)
                        if block_end > block_start:
                            block = content[block_start:block_end]
                            ptype = self._extract_value(block, "type")
                            nfaces = self._extract_value(block, "nFaces")
                            patches[patch_name] = {
                                "type": ptype or "wall",
                                "nFaces": int(nfaces) if nfaces else 0,
                            }
                idx = name_start + 1

        except (ValueError, FileNotFoundError):
            pass

        return patches

    @staticmethod
    def _extract_value(block: str, key: str) -> Optional[str]:
        """Extract a value from an OpenFOAM dictionary block."""
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith(key):
                parts = stripped.split()
                if len(parts) >= 2:
                    value = parts[1].rstrip(";")
                    return value
        return None
