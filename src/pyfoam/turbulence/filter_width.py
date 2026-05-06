"""
Filter width computation for Large Eddy Simulation.

The filter width Δ characterises the spatial cutoff between resolved
and modelled (subgrid) scales.  For unstructured finite volume meshes
the standard definition is the cube root of the cell volume:

.. math::

    \\Delta = V^{1/3}

This module provides a single function that computes this quantity from
a mesh's cell volumes.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core`.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["compute_filter_width"]


def compute_filter_width(
    mesh: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute the LES filter width for each cell.

    The filter width is defined as the cube root of the cell volume:

        Δ = V^(1/3)

    This is the standard definition used by OpenFOAM and most LES codes
    for unstructured meshes.

    Args:
        mesh: An :class:`~pyfoam.mesh.fv_mesh.FvMesh` with computed
            geometry (``cell_volumes`` must be available).
        device: Target device.  If ``None``, uses the global default.
        dtype: Target dtype.  If ``None``, uses the global default.

    Returns:
        ``(n_cells,)`` tensor of filter widths.

    Raises:
        AttributeError: If the mesh does not have ``cell_volumes``.

    Examples::

        >>> from pyfoam.mesh import FvMesh
        >>> mesh = FvMesh(...)  # doctest: +SKIP
        >>> mesh.compute_geometry()  # doctest: +SKIP
        >>> delta = compute_filter_width(mesh)  # doctest: +SKIP
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)
    # Clamp to avoid issues with degenerate (zero-volume) cells
    safe_volumes = cell_volumes.clamp(min=1e-30)
    return safe_volumes.pow(1.0 / 3.0)
