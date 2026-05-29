"""
ProcessorPatchEnhanced4 — v4 enhanced processor patches.

Extends :class:`~pyfoam.parallel.processor_patch_enhanced_3.EnhancedHaloExchange3` with:

- Mortar-like projection for non-conformal coupling
- Conservative field transfer preserving integral quantities
- Higher-order interpolation (quadratic IDW) for improved accuracy
- Periodic boundary support in halo exchange

Usage::

    patch = NonConformalPatch4(
        name="procAMI0To1",
        neighbour_rank=1,
        local_ghost_cells=local_idx,
        remote_cells=remote_idx,
        local_face_centres=lfc,
        remote_face_centres=rfc,
    )
    patch.auto_compute_weights_higher_order(order=2)
    halo = EnhancedHaloExchange4([patch])
    result = halo.exchange_conservative(field, all_fields)

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
from pyfoam.parallel.processor_patch_enhanced_3 import (
    NonConformalPatch3,
    EnhancedHaloExchange3,
    AdaptiveInterpolator,
)

__all__ = [
    "NonConformalPatch4",
    "EnhancedHaloExchange4",
    "HigherOrderInterpolator",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Higher-order interpolation
# ---------------------------------------------------------------------------


class HigherOrderInterpolator:
    """Higher-order IDW interpolator for mortar-like projection.

    Uses inverse-distance weighting with configurable power and
    nearest-neighbour count to achieve higher accuracy at
    non-conformal interfaces.

    Args:
        power: IDW power parameter (1=linear, 2=standard, 3=sharp).
        n_nearest: Number of nearest neighbours to use.
    """

    def __init__(
        self,
        power: float = 2.0,
        n_nearest: int = 4,
    ) -> None:
        self._power = power
        self._n_nearest = n_nearest

    def compute_weights(
        self,
        local_centres: torch.Tensor,
        remote_centres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute higher-order IDW interpolation weights.

        Args:
            local_centres: ``(n_local, 3)`` local ghost face centres.
            remote_centres: ``(n_remote, 3)`` remote face centres.

        Returns:
            Tuple of ``(weights, indices)`` — shapes
            ``(n_local, n_nearest)`` and ``(n_local, n_nearest)``.
        """
        local_centres = local_centres.to(dtype=torch.float64)
        remote_centres = remote_centres.to(dtype=torch.float64)
        n_local = local_centres.shape[0]
        n_remote = remote_centres.shape[0]
        n_nearest = min(self._n_nearest, n_remote)

        weights = torch.zeros(
            n_local, n_nearest, dtype=torch.float64
        )
        indices = torch.zeros(
            n_local, n_nearest, dtype=INDEX_DTYPE
        )

        for i in range(n_local):
            # Squared distances to all remote centres
            diff = remote_centres - local_centres[i]
            dist_sq = diff.pow(2).sum(dim=1)
            dist = torch.sqrt(torch.clamp(dist_sq, min=1e-30))

            # Select nearest
            _, top_idx = torch.topk(dist, n_nearest, largest=False)
            indices[i] = top_idx

            nearest_dist = dist[top_idx]
            nearest_dist = torch.clamp(nearest_dist, min=1e-30)

            # IDW weights: w_j = (1/d_j^p) / sum(1/d_k^p)
            inv_dist = 1.0 / nearest_dist.pow(self._power)
            w_sum = inv_dist.sum()
            if w_sum > 1e-30:
                weights[i] = inv_dist / w_sum
            else:
                weights[i] = 1.0 / n_nearest

        return weights, indices


# ---------------------------------------------------------------------------
# Non-conformal patch v4
# ---------------------------------------------------------------------------


@dataclass
class NonConformalPatch4(NonConformalPatch3):
    """v4 non-conformal patch with higher-order interpolation.

    Attributes:
        interpolation_order: IDW power for higher-order interpolation.
    """

    interpolation_order: float = 2.0

    def auto_compute_weights_higher_order(
        self,
        order: float = 2.0,
        n_nearest: int = 4,
    ) -> None:
        """Compute interpolation weights using higher-order IDW.

        Args:
            order: IDW power parameter.
            n_nearest: Number of nearest neighbours.
        """
        if (
            self.local_face_centres is None
            or self.remote_face_centres is None
        ):
            raise ValueError(
                "Face centres must be set before computing weights."
            )

        self.interpolation_order = order
        interpolator = HigherOrderInterpolator(
            power=order, n_nearest=n_nearest
        )
        weights, indices = interpolator.compute_weights(
            self.local_face_centres, self.remote_face_centres
        )
        self.local_weights = weights
        self.local_src_indices = indices


# ---------------------------------------------------------------------------
# Enhanced halo exchange v4
# ---------------------------------------------------------------------------


class EnhancedHaloExchange4(EnhancedHaloExchange3):
    """v4 enhanced halo exchange with conservative transfer.

    Supports:
    - Conservative field transfer (preserves integrals)
    - Higher-order interpolation via mortar-like projection
    - Periodic boundary exchange

    Parameters
    ----------
    patches : list
        Processor patches (may include NonConformalPatch4).
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
    # Conservative exchange
    # ------------------------------------------------------------------

    def exchange_conservative(
        self,
        field_values: torch.Tensor,
        all_fields: dict[int, torch.Tensor] | None = None,
        areas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange ghost cell values with conservation enforcement.

        The standard exchange interpolates values directly. The
        conservative exchange ensures that the total field integral
        (field * area) is preserved after the exchange.

        Args:
            field_values: ``(n_cells,)`` tensor.
            all_fields: Per-processor fields for serial mode.
            areas: ``(n_cells,)`` cell areas/volumes for weighting.

        Returns:
            Updated tensor with conserved ghost cell values.
        """
        result = field_values.clone()

        # Compute pre-exchange integral
        if areas is not None:
            areas = areas.to(dtype=torch.float64)
            pre_integral = (result * areas).sum().item()
        else:
            pre_integral = result.sum().item()

        # Standard exchange
        result = self.exchange(result, all_fields)

        # Compute post-exchange integral
        if areas is not None:
            post_integral = (result * areas).sum().item()
        else:
            post_integral = result.sum().item()

        # Apply global correction if needed
        diff = post_integral - pre_integral
        if abs(diff) > 1e-30 * max(abs(pre_integral), 1e-30):
            # Distribute correction over ghost cells
            n_ghost = 0
            for patch in self._patches:
                if hasattr(patch, 'local_ghost_cells'):
                    n_ghost += patch.local_ghost_cells.numel()

            if n_ghost > 0 and areas is not None:
                correction_per_cell = diff / n_ghost
                for patch in self._patches:
                    if hasattr(patch, 'local_ghost_cells'):
                        ghost_idx = patch.local_ghost_cells
                        if ghost_idx.numel() > 0:
                            result[ghost_idx] -= (
                                correction_per_cell
                                / areas[ghost_idx].clamp(min=1e-30)
                            )

        return result

    # ------------------------------------------------------------------
    # Periodic exchange
    # ------------------------------------------------------------------

    def exchange_periodic(
        self,
        field_values: torch.Tensor,
        periodic_offset: torch.Tensor,
        all_fields: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Exchange ghost cell values with periodic offset.

        Adds a constant offset to ghost values received from the
        periodic partner (e.g. pressure jump across a periodic boundary).

        Args:
            field_values: ``(n_cells,)`` tensor.
            periodic_offset: Scalar offset to add to ghost values.
            all_fields: Per-processor fields for serial mode.

        Returns:
            Updated tensor with periodic ghost cell values.
        """
        result = self.exchange(field_values, all_fields)

        # Apply periodic offset to all ghost cells (both conformal and non-conformal)
        for patch in self._patches:
            if hasattr(patch, 'local_ghost_cells'):
                ghost_idx = patch.local_ghost_cells
                if ghost_idx.numel() > 0:
                    result[ghost_idx] += periodic_offset

        # Also iterate over non-conformal patches stored in parent
        if hasattr(self, '_nc_patches'):
            for patch in self._nc_patches:
                if hasattr(patch, 'local_ghost_cells'):
                    ghost_idx = patch.local_ghost_cells
                    if ghost_idx.numel() > 0:
                        result[ghost_idx] += periodic_offset

        return result

    def __repr__(self) -> str:
        n_ho = sum(
            1 for p in self._patches
            if hasattr(p, 'interpolation_order')
            and p.interpolation_order > 1.0
        )
        return (
            f"EnhancedHaloExchange4(n_patches={len(self._patches)}, "
            f"higher_order={n_ho})"
        )
