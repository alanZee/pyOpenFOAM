"""
surfaceFeatures — extract feature edges from a triangulated surface.

Mirrors OpenFOAM's ``surfaceFeatures`` utility.  Identifies edges where
the angle between adjacent triangle normals exceeds a threshold (the
*included angle*) and returns them as feature edges suitable for meshing
with ``snappyHexMesh``.

Algorithm
---------
1. Build an edge-to-face adjacency map from the triangulated surface.
2. For each edge shared by two faces, compute the dihedral angle between
   the face normals.
3. Edges whose dihedral angle exceeds ``included_angle`` (or where only
   one face is present — boundary/open edges) are classified as features.
4. Return a structured result with feature-edge endpoints and angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceFeaturesResult", "surface_features"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurfaceFeaturesResult:
    """Structured result from :func:`surface_features`.

    Attributes
    ----------
    n_edges : int
        Total number of unique edges in the surface.
    n_features : int
        Number of feature edges detected.
    feature_points : np.ndarray
        ``(n_features, 2, 3)`` array of feature-edge endpoint coordinates.
        Each entry is ``[[x0, y0, z0], [x1, y1, z1]]``.
    feature_angles : np.ndarray
        ``(n_features,)`` array of dihedral angles (degrees) for each
        feature edge.  Open/boundary edges have angle ``180.0``.
    feature_edge_indices : list[tuple[int, int]]
        Vertex-index pairs ``(i, j)`` with ``i < j`` for each feature edge.
    """

    n_edges: int = 0
    n_features: int = 0
    feature_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2, 3), dtype=np.float64)
    )
    feature_angles: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    feature_edge_indices: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_features(
    surface_path: Union[str, Path],
    included_angle: float = 150.0,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> SurfaceFeaturesResult:
    """Extract feature edges from a triangulated surface.

    The surface can be provided either as a file path (STL/OBJ/VTK) or
    directly as vertex/face/normal arrays.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file (STL, OBJ, or VTK).  Ignored when
        *vertices* and *faces* are provided directly.
    included_angle : float
        Dihedral angle threshold in degrees.  Edges where the angle
        between adjacent face normals exceeds this value are classified
        as features.  Default ``150`` (i.e. edges sharper than 30 degrees
        are flagged).
    vertices : np.ndarray, optional
        ``(n_points, 3)`` vertex coordinates.  When provided together
        with *faces*, the file is not read.
    faces : np.ndarray, optional
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    normals : np.ndarray, optional
        ``(n_faces, 3)`` face normal vectors.  Computed automatically
        from vertices and faces when not provided.

    Returns
    -------
    SurfaceFeaturesResult
        Feature edge data.

    Raises
    ------
    FileNotFoundError
        If *surface_path* does not exist and no arrays are provided.
    ValueError
        If the surface has no faces.
    """
    # Obtain geometry
    if vertices is not None and faces is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = np.asarray(faces, dtype=np.int32)
        norms = (
            np.asarray(normals, dtype=np.float64)
            if normals is not None
            else _compute_normals(verts, facs)
        )
    else:
        from pyfoam.tools.surface_convert import _rs, _df

        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, norms, facs = _rs(p, fmt)

    if facs.shape[0] == 0:
        raise ValueError("Surface has no faces.")

    result = SurfaceFeaturesResult()
    n_faces = facs.shape[0]

    # Build edge → face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ei = (int(tri[a]), int(tri[b]))
            key = (min(ei), max(ei))
            edge_faces.setdefault(key, []).append(fi)

    result.n_edges = len(edge_faces)

    # Classify edges
    feat_pts = []
    feat_angles = []
    feat_indices = []
    cos_thresh = np.cos(np.radians(included_angle))

    for (vi, vj), adj_faces in edge_faces.items():
        if len(adj_faces) == 1:
            # Open/boundary edge — always a feature
            feat_pts.append([verts[vi], verts[vj]])
            feat_angles.append(180.0)
            feat_indices.append((vi, vj))
        elif len(adj_faces) == 2:
            n0 = norms[adj_faces[0]]
            n1 = norms[adj_faces[1]]
            cos_angle = np.dot(n0, n1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            # angle between normals; included angle = 180 - dihedral
            included = 180.0 - angle
            if included < included_angle:
                feat_pts.append([verts[vi], verts[vj]])
                feat_angles.append(angle)
                feat_indices.append((vi, vj))
        # edges with >2 adjacent faces are non-manifold; skip

    result.n_features = len(feat_indices)
    result.feature_points = (
        np.array(feat_pts, dtype=np.float64)
        if feat_pts
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    result.feature_angles = (
        np.array(feat_angles, dtype=np.float64)
        if feat_angles
        else np.empty(0, dtype=np.float64)
    )
    result.feature_edge_indices = feat_indices

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals from vertices and triangle faces."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe_norms
