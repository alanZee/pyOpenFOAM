"""
mapFields — map fields between different meshes.

Mirrors OpenFOAM's ``mapFields`` utility.  Maps cell-centred (vol) fields
from a *source* mesh to a *target* mesh using nearest-neighbour
interpolation: for each target cell centre, the value of the nearest
source cell is copied.

Usage::

    from pyfoam.tools.map_fields import map_fields

    mapped = map_fields(source_mesh, target_mesh, source_fields)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["map_fields"]


def map_fields(
    source_mesh: "FvMesh",
    target_mesh: "FvMesh",
    source_fields: Dict[str, np.ndarray],
    method: str = "nearest",
) -> Dict[str, np.ndarray]:
    """Map cell-centred fields from one mesh to another.

    For each target cell centre, the value of the nearest source cell is
    selected (nearest-neighbour interpolation).

    Parameters
    ----------
    source_mesh : FvMesh
        Mesh that the source fields are defined on.
    target_mesh : FvMesh
        Mesh to map the fields onto.
    source_fields : dict
        ``{field_name: numpy_array}`` with shape ``(n_source_cells,)`` for
        scalars or ``(n_source_cells, 3)`` for vectors.
    method : str, optional
        Interpolation method.  Currently only ``"nearest"`` is supported.

    Returns
    -------
    dict
        ``{field_name: numpy_array}`` with values defined on
        *target_mesh*.  Scalar arrays have shape ``(n_target_cells,)``,
        vector arrays ``(n_target_cells, 3)``.

    Raises
    ------
    ValueError
        If *method* is not ``"nearest"`` or field shapes are inconsistent
        with the source mesh.
    """
    if method != "nearest":
        raise ValueError(
            f"Unknown interpolation method: {method!r}.  Only 'nearest' is supported."
        )

    # Extract cell centres as numpy arrays
    src_centres = source_mesh.cell_centres.detach().cpu().numpy()  # (n_src, 3)
    tgt_centres = target_mesh.cell_centres.detach().cpu().numpy()  # (n_tgt, 3)

    n_src = src_centres.shape[0]
    n_tgt = tgt_centres.shape[0]

    # Build nearest-neighbour mapping: for each target cell, find the
    # closest source cell.  Use a vectorised distance computation with
    # chunking to avoid excessive memory for very large meshes.
    indices = _nearest_neighbour_indices(src_centres, tgt_centres)

    # Map each field
    mapped: Dict[str, np.ndarray] = {}
    for name, data in source_fields.items():
        if data.ndim == 1:
            if data.shape[0] != n_src:
                raise ValueError(
                    f"Field '{name}' has {data.shape[0]} values but source mesh "
                    f"has {n_src} cells."
                )
            mapped[name] = data[indices]
        elif data.ndim == 2 and data.shape[1] == 3:
            if data.shape[0] != n_src:
                raise ValueError(
                    f"Field '{name}' has {data.shape[0]} values but source mesh "
                    f"has {n_src} cells."
                )
            mapped[name] = data[indices]
        else:
            raise ValueError(
                f"Field '{name}' has unsupported shape {data.shape}.  "
                "Expected (n_cells,) or (n_cells, 3)."
            )

    return mapped


def _nearest_neighbour_indices(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Find index of nearest source point for each target point.

    Parameters
    ----------
    src_points : ndarray (n_src, 3)
    tgt_points : ndarray (n_tgt, 3)
    chunk_size : int
        Process target points in chunks to limit memory usage.

    Returns
    -------
    ndarray (n_tgt,) of int
        Index into *src_points* for each target point.
    """
    n_tgt = tgt_points.shape[0]
    indices = np.empty(n_tgt, dtype=np.int64)

    for start in range(0, n_tgt, chunk_size):
        end = min(start + chunk_size, n_tgt)
        chunk = tgt_points[start:end]  # (chunk, 3)

        # Squared distances: (chunk, n_src)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        a_sq = np.sum(chunk ** 2, axis=1, keepdims=True)     # (chunk, 1)
        b_sq = np.sum(src_points ** 2, axis=1, keepdims=True)  # (n_src, 1)
        dist_sq = a_sq + b_sq.T - 2.0 * chunk @ src_points.T  # (chunk, n_src)

        indices[start:end] = np.argmin(dist_sq, axis=1)

    return indices


def map_fields_from_case(
    source_case: Union[str, "FvMesh"],
    target_case: Union[str, "FvMesh"],
    source_fields: Dict[str, np.ndarray],
    time: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """High-level convenience: map fields between two case directories.

    This is a thin wrapper around :func:`map_fields` that accepts case
    directory paths.  When case paths are given, the meshes must be
    loaded externally and passed as *source_case* / *target_case*.

    Parameters
    ----------
    source_case : str or FvMesh
        Source case directory path or pre-loaded mesh.
    target_case : str or FvMesh
        Target case directory path or pre-loaded mesh.
    source_fields : dict
        Fields defined on the source mesh.
    time : float, optional
        Time value (currently unused; reserved for on-disk field loading).

    Returns
    -------
    dict
        Mapped fields on the target mesh.
    """
    # If strings are passed, we cannot load meshes from disk yet
    if isinstance(source_case, str) or isinstance(target_case, str):
        raise NotImplementedError(
            "On-disk mesh loading is not yet supported.  "
            "Pass FvMesh objects directly."
        )

    return map_fields(source_case, target_case, source_fields)
