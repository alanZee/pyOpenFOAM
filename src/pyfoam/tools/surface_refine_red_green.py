"""
surfaceRefineRedGreen — red-green refinement of triangulated surfaces.

Mirrors the surface refinement strategy used in OpenFOAM's meshing tools.
Provides two refinement strategies:

- **Red refinement**: Split each marked triangle into 4 congruent sub-triangles
  by inserting midpoints on all three edges (uniform 1-to-4 split).
- **Green refinement**: Split each marked triangle into 2 sub-triangles by
  inserting one midpoint on the longest edge (1-to-2 split, used as a
  transition to avoid excessive hanging nodes).

The combined red-green strategy applies red refinement to selected
triangles and green refinement to their neighbours to maintain
conformity (no hanging nodes).

Usage::

    from pyfoam.tools.surface_refine_red_green import surface_refine

    out_verts, out_faces = surface_refine(
        vertices, faces,
        refine_mask=mask,
        levels=1,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["RefineResult", "surface_refine"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RefineResult:
    """Result from surface refinement.

    Attributes
    ----------
    vertices : np.ndarray
        ``(n_points, 3)`` vertex coordinates of the refined mesh.
    faces : np.ndarray
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    n_input_faces : int
        Number of faces before refinement.
    n_output_faces : int
        Number of faces after refinement.
    n_red_splits : int
        Number of triangles that received red (1-to-4) refinement.
    n_green_splits : int
        Number of triangles that received green (1-to-2) refinement.
    levels : int
        Number of refinement levels applied.
    """

    vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    n_input_faces: int = 0
    n_output_faces: int = 0
    n_red_splits: int = 0
    n_green_splits: int = 0
    levels: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_refine(
    vertices: np.ndarray,
    faces: np.ndarray,
    refine_mask: Optional[np.ndarray] = None,
    levels: int = 1,
    angle_threshold: Optional[float] = None,
) -> RefineResult:
    """Refine a triangulated surface using red-green strategy.

    Parameters
    ----------
    vertices : np.ndarray
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    refine_mask : np.ndarray, optional
        Boolean array of shape ``(n_faces,)``.  ``True`` marks a face
        for red refinement.  If ``None`` and *angle_threshold* is given,
        faces are selected automatically based on the dihedral angle
        criterion.  If both are ``None``, all faces are refined.
    levels : int
        Number of successive refinement passes.  Default ``1``.
    angle_threshold : float, optional
        If given, automatically mark faces whose maximum edge length
        exceeds ``angle_threshold`` times the mean edge length of the
        mesh.  This is a proxy for curvature-based refinement.

    Returns
    -------
    RefineResult
        Refined mesh and statistics.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    facs = np.asarray(faces, dtype=np.int32)

    if facs.shape[0] == 0:
        raise ValueError("Mesh must have at least one face.")
    if facs.ndim != 2 or facs.shape[1] != 3:
        raise ValueError("Faces must be (n_faces, 3) array.")

    if refine_mask is not None:
        mask = np.asarray(refine_mask, dtype=bool)
    elif angle_threshold is not None:
        mask = _auto_select(verts, facs, angle_threshold)
    else:
        mask = np.ones(facs.shape[0], dtype=bool)

    n_red = 0
    n_green = 0

    for _level in range(levels):
        verts, facs, nr, ng = _refine_level(verts, facs, mask)
        n_red += nr
        n_green += ng
        # Recompute mask for next level: refine all new faces
        if _level < levels - 1:
            mask = np.ones(facs.shape[0], dtype=bool)

    return RefineResult(
        vertices=verts,
        faces=facs,
        n_input_faces=faces.shape[0],
        n_output_faces=facs.shape[0],
        n_red_splits=n_red,
        n_green_splits=n_green,
        levels=levels,
    )


# ---------------------------------------------------------------------------
# One refinement level
# ---------------------------------------------------------------------------


def _refine_level(
    verts: np.ndarray,
    facs: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Apply one level of red-green refinement.

    Returns (new_vertices, new_faces, n_red, n_green).
    """
    n_faces = facs.shape[0]
    # Edge midpoint cache: (vi, vj) -> midpoint index
    mid_cache: dict[tuple[int, int], int] = {}
    new_verts_list = list(verts)

    def _get_midpoint(vi: int, vj: int) -> int:
        key = (min(vi, vj), max(vi, vj))
        if key in mid_cache:
            return mid_cache[key]
        mid = 0.5 * (new_verts_list[vi] + new_verts_list[vj])
        idx = len(new_verts_list)
        new_verts_list.append(mid)
        mid_cache[key] = idx
        return idx

    new_faces: list[list[int]] = []
    n_red = 0
    n_green = 0

    # Track which edges of non-refined faces have midpoints (from
    # adjacent refined faces)
    # edge_has_midpoint[(vi, vj)] = midpoint_index
    edge_has_midpoint: dict[tuple[int, int], int] = {}

    # First pass: identify all midpoints from red-refined faces
    for fi in range(n_faces):
        if not mask[fi]:
            continue
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            vi, vj = int(tri[a]), int(tri[b])
            key = (min(vi, vj), max(vi, vj))
            if key not in mid_cache:
                mid = 0.5 * (new_verts_list[vi] + new_verts_list[vj])
                mid_idx = len(new_verts_list)
                new_verts_list.append(mid)
                mid_cache[key] = mid_idx
            edge_has_midpoint[key] = mid_cache[key]

    # Second pass: split faces
    for fi in range(n_faces):
        tri = facs[fi]
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])

        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        mids = [edge_has_midpoint.get(e) for e in edges]

        n_split_edges = sum(1 for m in mids if m is not None)

        if mask[fi]:
            # Red refinement: all 3 edges get midpoints -> 4 sub-triangles
            m01 = _get_midpoint(v0, v1)
            m12 = _get_midpoint(v1, v2)
            m20 = _get_midpoint(v2, v0)
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])
            n_red += 1
        elif n_split_edges == 1:
            # Green refinement: one edge has a midpoint -> 2 sub-triangles
            # Find which edge has the midpoint
            for i, (e, m) in enumerate(zip(edges, mids)):
                if m is not None:
                    mid_idx = m
                    if i == 0:
                        new_faces.append([v0, mid_idx, v2])
                        new_faces.append([mid_idx, v1, v2])
                    elif i == 1:
                        new_faces.append([v1, mid_idx, v0])
                        new_faces.append([mid_idx, v2, v0])
                    else:
                        new_faces.append([v2, mid_idx, v1])
                        new_faces.append([mid_idx, v0, v1])
                    break
            n_green += 1
        elif n_split_edges >= 2:
            # Too many hanging edges — apply red refinement for conformity
            m01 = _get_midpoint(v0, v1)
            m12 = _get_midpoint(v1, v2)
            m20 = _get_midpoint(v2, v0)
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])
            n_red += 1
        else:
            # No split edges — keep as is
            new_faces.append([v0, v1, v2])

    out_verts = np.array(new_verts_list, dtype=np.float64)
    out_faces = np.array(new_faces, dtype=np.int32)
    return out_verts, out_faces, n_red, n_green


# ---------------------------------------------------------------------------
# Automatic face selection
# ---------------------------------------------------------------------------


def _auto_select(
    verts: np.ndarray,
    facs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Select faces whose max edge length exceeds threshold * mean."""
    n_faces = facs.shape[0]
    edge_lens = np.zeros((n_faces, 3), dtype=np.float64)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
        edge_lens[:, i] = np.linalg.norm(
            verts[facs[:, b]] - verts[facs[:, a]], axis=1
        )
    mean_len = edge_lens.mean()
    max_len = edge_lens.max(axis=1)
    return max_len > threshold * mean_len
