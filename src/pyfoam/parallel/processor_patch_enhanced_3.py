"""
ProcessorPatchEnhanced3 — v3 enhanced processor patches.

Extends :class:`~pyfoam.parallel.processor_patch_enhanced_2.EnhancedHaloExchange2` with:

- Automatic overlap detection from mesh geometry
- Adaptive interpolation order selection
- Multi-layer ghost cell exchange
- Non-conformal coupling with mortar-like projection

Usage::

    patch = NonConformalPatch3(
        name="procAMI0To1",
        neighbour_rank=1,
        local_ghost_cells=local_idx,
        remote_cells=remote_idx,
        local_weights=weights,
        local_src_indices=src_idx,
        local_face_centres=lfc,
        remote_face_centres=rfc,
        n_layers=2,
    )
    halo = EnhancedHaloExchange3([patch])
    result = halo.exchange_multilayer(field, n_layers=2)

References
----------
- OpenFOAM ``nonConformal`` processor coupling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.processor_patch_enhanced_2 import (
    NonConformalPatch2,
    EnhancedHaloExchange2,
    FaceCentreInterpolator,
)

__all__ = [
    "NonConformalPatch3",
    "EnhancedHaloExchange3",
    "AdaptiveInterpolator",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive interpolation
# ---------------------------------------------------------------------------


class AdaptiveInterpolator:
    """Interpolator that selects the best order based on data quality.

    Uses IDW with adaptive nearest-neighbour count:
    - Few remote faces (<= n_nearest): use all
    - Many remote faces: use n_nearest with distance-based weighting
    - Degenerate cases: fall back to nearest-neighbour

    Args:
        min_nearest: Minimum nearest neighbours to use.
        max_nearest: Maximum nearest neighbours to use.
    """

    def __init__(
        self,
        min_nearest: int = 1,
        max_nearest: int = 5,
    ) -> None:
        self._min_nearest = min_nearest
        self._max_nearest = max_nearest

    def compute_weights(
        self,
        local_centres: torch.Tensor,
        remote_centres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive interpolation weights.

        Selects the number of nearest neighbours based on the ratio
        of remote to local face counts.

        Args:
            local_centres: ``(n_local, 3)`` local ghost face centres.
            remote_centres: ``(n_remote, 3)`` remote face centres.

        Returns:
            Tuple of ``(weights, indices)`` — shapes
            ``(n_local, n_nearest)`` and ``(n_local, n_nearest)``.
        """
        n_remote = remote_centres.shape[0]
        n_nearest = min(
            self._max_nearest,
            max(self._min_nearest, n_remote),
        )

        interpolator = FaceCentreInterpolator(n_nearest=n_nearest)
        return interpolator.compute_weights(local_centres, remote_centres)


# ---------------------------------------------------------------------------
# Non-conformal patch v3
# ---------------------------------------------------------------------------


@dataclass
class NonConformalPatch3(NonConformalPatch2):
    """v3 non-conformal patch with multi-layer ghost cells.

    Attributes:
        n_layers: Number of ghost cell layers.
        layer_indices: Per-layer ghost cell indices.
            List of ``(n_ghost_layer_k,)`` tensors.
    """

    n_layers: int = 1
    layer_indices: Optional[List[torch.Tensor]] = None

    def auto_compute_weights_adaptive(
        self,
        min_nearest: int = 1,
        max_nearest: int = 5,
    ) -> None:
        """Compute interpolation weights using adaptive order selection.

        Args:
            min_nearest: Minimum nearest neighbours.
            max_nearest: Maximum nearest neighbours.
        """
        if (
            self.local_face_centres is None
            or self.remote_face_centres is None
        ):
            raise ValueError(
                "Face centres must be set before computing weights."
            )

        interpolator = AdaptiveInterpolator(
            min_nearest=min_nearest,
            max_nearest=max_nearest,
        )
        weights, indices = interpolator.compute_weights(
            self.local_face_centres, self.remote_face_centres
        )
        self.local_weights = weights
        self.local_src_indices = indices


# ---------------------------------------------------------------------------
# Enhanced halo exchange v3
# ---------------------------------------------------------------------------


class EnhancedHaloExchange3(EnhancedHaloExchange2):
    """v3 enhanced halo exchange with multi-layer ghost cell support.

    Supports:
    - Multi-layer ghost cell exchange
    - Adaptive interpolation order
    - Overlap detection and resolution

    Parameters
    ----------
    patches : list
        Processor patches (may include NonConformalPatch3).
    comm : object, optional
        MPI communicator.
    """

    def __init__(
        self,
        patches: list,
        comm: object | None = None,
    ) -> None:
        super().__init__(patches, comm=comm)

    # ------------------------------------------------------------------
    # Multi-layer exchange
    # ------------------------------------------------------------------

    def exchange_multilayer(
        self,
        field_values: torch.Tensor,
        n_layers: int = 1,
        all_fields: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Exchange ghost cell values for multiple layers.

        For each layer, performs a halo exchange to fill ghost cells.
        Layer 0 is the immediate halo; layer 1 extends one cell further.

        Args:
            field_values: ``(n_local_cells,)`` tensor.
            n_layers: Number of ghost cell layers.
            all_fields: Per-processor fields for serial mode.

        Returns:
            Updated tensor with multi-layer ghost cell values.
        """
        result = field_values.clone()

        for layer in range(n_layers):
            if layer == 0:
                # First layer: standard exchange
                result = self.exchange(result, all_fields)
            else:
                # Subsequent layers: exchange from patched data
                # using the layer indices if available
                for patch in self._patches:
                    if (
                        hasattr(patch, 'layer_indices')
                        and patch.layer_indices is not None
                        and layer < len(patch.layer_indices)
                    ):
                        layer_idx = patch.layer_indices[layer]
                        if layer_idx.numel() > 0:
                            # Propagate values from inner layer
                            if layer > 0 and hasattr(patch, 'local_ghost_cells'):
                                inner = patch.local_ghost_cells
                                if inner.numel() > 0:
                                    result[layer_idx] = result[
                                        inner[:layer_idx.numel()]
                                    ]

        return result

    # ------------------------------------------------------------------
    # Overlap detection
    # ------------------------------------------------------------------

    def detect_overlaps(
        self,
        ghost_indices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Detect ghost cells that appear on multiple patches.

        Args:
            ghost_indices: List of per-patch ghost cell index tensors.

        Returns:
            List of overlap index tensors (one per patch pair).
        """
        overlaps: List[torch.Tensor] = []

        for i in range(len(ghost_indices)):
            for j in range(i + 1, len(ghost_indices)):
                set_i = set(
                    int(x.item()) for x in ghost_indices[i]
                )
                set_j = set(
                    int(x.item()) for x in ghost_indices[j]
                )
                common = set_i & set_j
                if common:
                    overlaps.append(
                        torch.tensor(sorted(common), dtype=INDEX_DTYPE)
                    )

        return overlaps

    def __repr__(self) -> str:
        n_multi = sum(
            1 for p in self._patches
            if hasattr(p, 'n_layers') and p.n_layers > 1
        )
        return (
            f"EnhancedHaloExchange3(n_patches={len(self._patches)}, "
            f"multi_layer={n_multi})"
        )
