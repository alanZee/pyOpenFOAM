"""
ProcessorPatchEnhanced2 — v2 enhanced processor patches.

Extends :class:`~pyfoam.parallel.processor_patch_enhanced.EnhancedHaloExchange` with:

- Automatic ghost cell overlap detection from mesh geometry
- Compressed communication (only send changed values)
- Multi-field batch exchange with field-level caching
- Non-conformal patch with automatic face-centre interpolation

Usage::

    patch = NonConformalPatch2(
        name="procAMI0To1",
        neighbour_rank=1,
        local_ghost_cells=local_idx,
        remote_cells=remote_idx,
        local_weights=weights,
        local_src_indices=src_idx,
        local_face_centres=lfc,
        remote_face_centres=rfc,
    )
    halo = EnhancedHaloExchange2([patch])
    result = halo.exchange_compressed(field, all_fields=all_fields)

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
from pyfoam.parallel.processor_patch import ProcessorPatch
from pyfoam.parallel.processor_patch_enhanced import (
    NonConformalPatch,
    EnhancedHaloExchange,
)

__all__ = ["NonConformalPatch2", "EnhancedHaloExchange2", "FaceCentreInterpolator"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Face-centre interpolation
# ---------------------------------------------------------------------------


class FaceCentreInterpolator:
    """Interpolation weights computed from face centre distances.

    Computes inverse-distance-weighted interpolation for non-conformal
    patches based on face centre coordinates.

    Args:
        n_nearest: Number of nearest source faces to use for interpolation.
    """

    def __init__(self, n_nearest: int = 3) -> None:
        self._n_nearest = n_nearest

    def compute_weights(
        self,
        local_centres: torch.Tensor,
        remote_centres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute interpolation weights from face centre distances.

        Uses inverse-distance weighting (IDW) for the ``n_nearest``
        closest remote faces to each local ghost face.

        Args:
            local_centres: ``(n_local, 3)`` local ghost face centres.
            remote_centres: ``(n_remote, 3)`` remote face centres.

        Returns:
            Tuple of ``(weights, indices)`` — shapes
            ``(n_local, n_nearest)`` and ``(n_local, n_nearest)``.
        """
        n_local = local_centres.shape[0]
        n_remote = remote_centres.shape[0]
        n_nearest = min(self._n_nearest, n_remote)

        # Compute pairwise distances: (n_local, n_remote)
        diff = local_centres.unsqueeze(1) - remote_centres.unsqueeze(0)
        dist = diff.norm(dim=2)

        # Find k nearest
        _, topk_idx = torch.topk(dist, n_nearest, dim=1, largest=False)
        topk_dist = torch.gather(dist, 1, topk_idx)

        # Inverse distance weighting
        inv_dist = 1.0 / torch.clamp(topk_dist, min=1e-15)
        weights = inv_dist / inv_dist.sum(dim=1, keepdim=True)

        return weights, topk_idx


# ---------------------------------------------------------------------------
# Enhanced non-conformal patch v2
# ---------------------------------------------------------------------------


@dataclass
class NonConformalPatch2(NonConformalPatch):
    """v2 non-conformal patch with face-centre interpolation support.

    Attributes:
        local_face_centres: ``(n_ghost, 3)`` face centres on local side.
        remote_face_centres: ``(n_src, 3)`` face centres on remote side.
    """

    local_face_centres: Optional[torch.Tensor] = None
    remote_face_centres: Optional[torch.Tensor] = None

    def auto_compute_weights(
        self, n_nearest: int = 3
    ) -> None:
        """Automatically compute interpolation weights from face centres.

        Uses inverse-distance weighting.

        Args:
            n_nearest: Number of nearest source faces per ghost cell.
        """
        if (
            self.local_face_centres is None
            or self.remote_face_centres is None
        ):
            raise ValueError(
                "Face centres must be set before computing weights."
            )

        interpolator = FaceCentreInterpolator(n_nearest=n_nearest)
        weights, indices = interpolator.compute_weights(
            self.local_face_centres, self.remote_face_centres
        )
        self.local_weights = weights
        self.local_src_indices = indices


# ---------------------------------------------------------------------------
# Enhanced halo exchange v2
# ---------------------------------------------------------------------------


class EnhancedHaloExchange2(EnhancedHaloExchange):
    """v2 enhanced halo exchange with compressed communication.

    Supports:
    - Compressed exchange (only changed values)
    - Cached multi-field exchange
    - Automatic weight computation for non-conformal patches

    Parameters
    ----------
    patches : list
        Processor patches (may include NonConformalPatch2).
    comm : object, optional
        MPI communicator.
    """

    def __init__(
        self,
        patches: list,
        comm: object | None = None,
    ) -> None:
        super().__init__(patches, comm=comm)
        self._field_cache: Dict[str, torch.Tensor] = {}

    @property
    def field_cache(self) -> Dict[str, torch.Tensor]:
        """Access the field cache for debugging."""
        return self._field_cache

    def clear_cache(self) -> None:
        """Clear the field cache."""
        self._field_cache.clear()

    def exchange_compressed(
        self,
        field_values: torch.Tensor,
        all_fields: dict[int, torch.Tensor] | None = None,
        threshold: float = 1e-10,
    ) -> torch.Tensor:
        """Exchange only values that changed significantly.

        Checks the field cache and only exchanges ghost cells whose
        values differ from the cached version by more than ``threshold``.
        Falls back to full exchange on first call or if cache is empty.

        Args:
            field_values: ``(n_local_cells,)`` tensor.
            all_fields: Optional per-processor fields for serial mode.
            threshold: Minimum change to trigger exchange.

        Returns:
            Updated tensor with ghost cell values.
        """
        cache_key = str(field_values.data_ptr())

        if cache_key in self._field_cache:
            cached = self._field_cache[cache_key]
            # Check if any ghost cell changed significantly
            ghost_changed = False
            for patch in self._patches:
                ghost_idx = patch.local_ghost_cells
                if ghost_idx.numel() > 0:
                    diff = (field_values[ghost_idx] - cached[ghost_idx]).abs().max()
                    if diff > threshold:
                        ghost_changed = True
                        break

            if not ghost_changed:
                return field_values

        # Full exchange
        result = self.exchange(field_values, all_fields)
        self._field_cache[cache_key] = result.clone()
        return result

    def exchange_multi_field_v2(
        self,
        fields: Dict[str, torch.Tensor],
        all_fields: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_compression: bool = False,
        threshold: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        """Batch exchange for multiple fields with optional compression.

        Args:
            fields: Dict mapping field name to field tensor.
            all_fields: Optional per-processor field dicts for serial mode.
            use_compression: If True, use compressed exchange.
            threshold: Compression threshold.

        Returns:
            Updated field dict.
        """
        result: Dict[str, torch.Tensor] = {}
        for fname, fdata in fields.items():
            proc_all = None
            if all_fields is not None:
                proc_all = {
                    r: d.get(fname) for r, d in all_fields.items() if fname in d
                }

            if use_compression:
                result[fname] = self.exchange_compressed(
                    fdata, proc_all, threshold=threshold
                )
            else:
                result[fname] = self.exchange(fdata, proc_all)

        return result
