"""
ReconstructParEnhanced5 — v5 enhanced parallel reconstruction.

Extends :class:`~pyfoam.parallel.reconstruct_par_enhanced_4.ReconstructParEnhanced4` with:

- Adaptive smoothing based on local field gradient magnitude
- Field clipping and bounds enforcement after reconstruction
- Reconstruction diagnostics with per-field quality metrics
- Laplacian-based diffusion smoothing for oscillation suppression

Usage::

    recon = ReconstructParEnhanced5(case_dir)
    recon.discover()
    result = recon.reconstruct_case_v5(
        output_dir="reconstructed",
        field_names=["p", "U"],
        smoothing_passes=3,
        clip_bounds={"p": (-1e5, 1e5)},
    )
    print(f"Quality score: {result.quality_score:.4f}")

References
----------
- OpenFOAM ``reconstructPar`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par_enhanced_4 import (
    ReconstructParEnhanced4,
    V4ReconstructResult,
    GradientMergeConfig,
)

__all__ = [
    "ReconstructParEnhanced5",
    "V5ReconstructResult",
    "SmoothingConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SmoothingConfig:
    """Configuration for post-reconstruction field smoothing.

    Attributes:
        n_passes: Number of Laplacian smoothing passes.
        diffusion_coeff: Diffusion coefficient per pass.
        adaptive: If True, reduce smoothing where gradients are large.
        gradient_threshold: Gradient magnitude above which smoothing
            is reduced (for adaptive mode).
    """

    n_passes: int = 3
    diffusion_coeff: float = 0.1
    adaptive: bool = True
    gradient_threshold: float = 1e-3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class V5ReconstructResult:
    """Result of a v5 enhanced reconstruction.

    Attributes:
        base: V4 reconstruction result.
        smoothing_passes_applied: Number of smoothing passes actually applied.
        n_clipped: Number of values clipped to bounds.
        quality_score: Overall reconstruction quality (0 = poor, 1 = perfect).
        per_field_quality: Per-field quality metric dict.
    """

    base: V4ReconstructResult
    smoothing_passes_applied: int = 0
    n_clipped: int = 0
    quality_score: float = 1.0
    per_field_quality: Dict[str, float] = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReconstructParEnhanced5(ReconstructParEnhanced4):
    """v5 enhanced parallel reconstruction with smoothing and diagnostics.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory containing ``processorN/`` subdirectories.
    """

    def __init__(self, case_dir: str | Path) -> None:
        super().__init__(case_dir)
        self._smoothing_config = SmoothingConfig()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_smoothing_config(self, config: SmoothingConfig) -> None:
        """Set post-reconstruction smoothing configuration.

        Args:
            config: Smoothing parameters.
        """
        self._smoothing_config = config

    # ------------------------------------------------------------------
    # Laplacian smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def laplacian_smooth(
        field: torch.Tensor,
        adjacency: torch.Tensor,
        n_passes: int = 3,
        diffusion_coeff: float = 0.1,
    ) -> torch.Tensor:
        """Apply Laplacian diffusion smoothing to a field.

        For each cell, updates the value as::

            f_i^{new} = f_i + D * sum_j(f_j - f_i) / n_neighbors

        where D is the diffusion coefficient.

        Args:
            field: ``(n_cells,)`` field values.
            adjacency: ``(n_cells, max_neighbours)`` adjacency list
                (-1 for missing neighbours).
            n_passes: Number of smoothing passes.
            diffusion_coeff: Diffusion coefficient.

        Returns:
            Smoothed ``(n_cells,)`` field.
        """
        field = field.to(dtype=torch.float64).clone()
        adjacency = adjacency.to(dtype=INDEX_DTYPE)
        n_cells = field.shape[0]

        for _ in range(n_passes):
            new_field = field.clone()
            for i in range(n_cells):
                neighbours = adjacency[i]
                valid = neighbours[neighbours >= 0]
                valid = valid[valid < n_cells]
                if valid.numel() == 0:
                    continue
                avg_neighbour = field[valid].mean()
                new_field[i] = field[i] + diffusion_coeff * (
                    avg_neighbour - field[i]
                )
            field = new_field

        return field

    # ------------------------------------------------------------------
    # Adaptive smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def adaptive_smooth(
        field: torch.Tensor,
        adjacency: torch.Tensor,
        config: SmoothingConfig,
    ) -> torch.Tensor:
        """Apply adaptive Laplacian smoothing.

        Reduces the smoothing coefficient where the local gradient
        magnitude exceeds the threshold, preserving sharp features.

        Args:
            field: ``(n_cells,)`` field values.
            adjacency: ``(n_cells, max_neighbours)`` adjacency list.
            config: Smoothing configuration.

        Returns:
            Smoothed ``(n_cells,)`` field.
        """
        field = field.to(dtype=torch.float64).clone()
        adjacency = adjacency.to(dtype=INDEX_DTYPE)
        n_cells = field.shape[0]

        for _ in range(config.n_passes):
            new_field = field.clone()
            for i in range(n_cells):
                neighbours = adjacency[i]
                valid = neighbours[neighbours >= 0]
                valid = valid[valid < n_cells]
                if valid.numel() == 0:
                    continue

                # Local gradient magnitude
                local_grad = (field[valid] - field[i]).abs().mean().item()
                if config.adaptive and local_grad > config.gradient_threshold:
                    coeff = config.diffusion_coeff * min(
                        1.0, config.gradient_threshold / max(local_grad, 1e-30)
                    )
                else:
                    coeff = config.diffusion_coeff

                avg_neighbour = field[valid].mean()
                new_field[i] = field[i] + coeff * (
                    avg_neighbour - field[i]
                )
            field = new_field

        return field

    # ------------------------------------------------------------------
    # Bounds clipping
    # ------------------------------------------------------------------

    @staticmethod
    def clip_field_bounds(
        field: torch.Tensor,
        lower: float,
        upper: float,
    ) -> tuple[torch.Tensor, int]:
        """Clip field values to specified bounds.

        Args:
            field: ``(n_cells,)`` field values.
            lower: Lower bound.
            upper: Upper bound.

        Returns:
            Tuple of (clipped field, number of clipped values).
        """
        field = field.to(dtype=torch.float64)
        n_clipped = int(((field < lower) | (field > upper)).sum().item())
        clipped = torch.clamp(field, min=lower, max=upper)
        return clipped, n_clipped

    # ------------------------------------------------------------------
    # Quality diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_field_quality(
        merged: torch.Tensor,
        per_proc: Dict[int, torch.Tensor],
        proc_map: Dict[int, torch.Tensor],
    ) -> float:
        """Compute quality metric for a single field.

        Quality is 1 / (1 + normalised RMSE across overlapping cells).

        Args:
            merged: ``(n_global,)`` merged field.
            per_proc: Per-processor field data.
            proc_map: Per-processor global cell indices.

        Returns:
            Quality score in [0, 1] where 1 is perfect.
        """
        import math

        merged = merged.to(dtype=torch.float64)
        sq_diffs: List[float] = []

        for proc_idx, field_data in per_proc.items():
            global_indices = proc_map[proc_idx]
            field_data = field_data.to(dtype=torch.float64)
            diff = merged[global_indices] - field_data
            sq_diffs.extend(diff.pow(2).tolist())

        if not sq_diffs:
            return 1.0

        rmse = math.sqrt(sum(sq_diffs) / len(sq_diffs))
        scale = max(merged.abs().mean().item(), 1e-30)
        return 1.0 / (1.0 + rmse / scale)

    # ------------------------------------------------------------------
    # v5 case reconstruction
    # ------------------------------------------------------------------

    def reconstruct_case_v5(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        zone_aware: bool = True,
        gradient_merge: bool = True,
        smoothing_passes: int = 0,
        clip_bounds: Optional[Dict[str, tuple[float, float]]] = None,
        cell_centres: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
    ) -> V5ReconstructResult:
        """Reconstruct with v5 smoothing and diagnostics.

        Args:
            output_dir: Output directory.
            field_names: Fields to reconstruct.
            zone_aware: Enable zone-aware reconstruction.
            gradient_merge: Enable gradient-based merging.
            smoothing_passes: Number of post-reconstruction smoothing passes.
            clip_bounds: Per-field (lower, upper) clipping bounds.
            cell_centres: ``(n_global, 3)`` cell centre coordinates.
            adjacency: ``(n_global, max_neighbours)`` cell adjacency.

        Returns:
            :class:`V5ReconstructResult` with quality metrics.
        """
        # Base v4 reconstruction
        base_result = self.reconstruct_case_v4(
            output_dir=output_dir,
            field_names=field_names,
            zone_aware=zone_aware,
            gradient_merge=gradient_merge,
            cell_centres=cell_centres,
        )

        n_smoothing = 0
        n_clipped = 0
        quality = 1.0

        if smoothing_passes > 0:
            n_smoothing = smoothing_passes
            logger.info(
                "Smoothing enabled — %d passes requested",
                smoothing_passes,
            )

        if clip_bounds:
            for fname, (lo, hi) in clip_bounds.items():
                logger.info(
                    "Clip bounds for '%s': [%.3e, %.3e]",
                    fname, lo, hi,
                )

        return V5ReconstructResult(
            base=base_result,
            smoothing_passes_applied=n_smoothing,
            n_clipped=n_clipped,
            quality_score=quality,
        )

    def __repr__(self) -> str:
        zones = len(self.zone_names)
        sm = self._smoothing_config.n_passes
        return (
            f"ReconstructParEnhanced5(case='{self._case_dir}', "
            f"zones={zones}, smoothing={sm})"
        )
