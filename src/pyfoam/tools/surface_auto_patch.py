"""
surfaceAutoPatch — auto-patch surface based on feature angle.

Mirrors OpenFOAM's ``surfaceAutoPatch`` utility.  Groups connected
triangles into patches based on the dihedral angle between adjacent
faces.  Faces separated by angles larger than the threshold are
assigned to different patches.

This is useful for automatically decomposing a single STL region into
named patches suitable for boundary-condition assignment.

Algorithm
---------
1. Build face adjacency from shared edges.
2. Flood-fill connected components: two faces belong to the same patch
   if the dihedral angle between them is below *feature_angle*.
3. Assign a numeric patch ID to each connected component.
4. Optionally write a patched surface file.

Usage::

    from pyfoam.tools.surface_auto_patch import surface_auto_patch

    result = surface_auto_patch("body.stl", feature_angle=30.0)
    print(result.n_patches, result.patch_ids.shape)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchResult", "surface_auto_patch"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurfaceAutoPatchResult:
    """Structured result from :func:`surface_auto_patch`.

    Attributes
    ----------
    n_patches : int
        Number of patches detected.
    patch_ids : np.ndarray
        ``(n_faces,)`` integer patch ID for each face (0-based).
    patch_face_counts : dict[int, int]
        ``{patch_id: n_faces}`` count for each patch.
    vertices : np.ndarray
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray
        ``(n_faces, 3)`` triangle vertex indices.
    """

    n_patches: int = 0
    patch_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    patch_face_counts: dict = field(default_factory=dict)
    vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_auto_patch(
    surface_path: Union[str, Path] = "",
    feature_angle: float = 30.0,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> SurfaceAutoPatchResult:
    """Group surface triangles into patches by feature angle.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file.  Ignored when *vertices* and *faces*
        are provided directly.
    feature_angle : float
        Dihedral angle threshold in degrees.  Adjacent faces whose
        normals differ by more than this angle are placed in separate
        patches.  Default ``30.0``.
    vertices : np.ndarray, optional
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray, optional
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    normals : np.ndarray, optional
        ``(n_faces, 3)`` face normals.
    output_path : str or Path, optional
        If provided, write the patched surface as an STL file with
        per-patch solid blocks.

    Returns
    -------
    SurfaceAutoPatchResult
        Patch assignment and surface data.

    Raises
    ------
    FileNotFoundError
        If *surface_path* does not exist and no arrays are provided.
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

    n_faces = facs.shape[0]
    if n_faces == 0:
        return SurfaceAutoPatchResult(vertices=verts, faces=facs)

    # Build edge → face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    # Build face adjacency (neighbours sharing an edge with angle < threshold)
    cos_thresh = np.cos(np.radians(feature_angle))
    face_neighbours: list[list[int]] = [[] for _ in range(n_faces)]
    for adj_faces in edge_faces.values():
        if len(adj_faces) == 2:
            f0, f1 = adj_faces
            dot = np.dot(norms[f0], norms[f1])
            dot = np.clip(dot, -1.0, 1.0)
            # included angle = 180 - arccos(dot)
            # Two faces are in the same patch if included angle >= feature_angle
            # i.e. arccos(dot) <= 180 - feature_angle
            # i.e. dot >= cos(180 - feature_angle) = -cos(feature_angle)
            if dot >= -cos_thresh:
                face_neighbours[f0].append(f1)
                face_neighbours[f1].append(f0)

    # Flood-fill connected components
    patch_ids = np.full(n_faces, -1, dtype=np.int32)
    current_patch = 0
    for seed in range(n_faces):
        if patch_ids[seed] >= 0:
            continue
        # BFS
        queue = [seed]
        patch_ids[seed] = current_patch
        while queue:
            fi = queue.pop()
            for nbr in face_neighbours[fi]:
                if patch_ids[nbr] < 0:
                    patch_ids[nbr] = current_patch
                    queue.append(nbr)
        current_patch += 1

    # Build result
    result = SurfaceAutoPatchResult()
    result.n_patches = current_patch
    result.patch_ids = patch_ids
    result.vertices = verts
    result.faces = facs
    for pid in range(current_patch):
        result.patch_face_counts[pid] = int((patch_ids == pid).sum())

    # Optionally write output
    if output_path is not None:
        _write_patched_stl(Path(output_path), verts, facs, norms, patch_ids, current_patch)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _write_patched_stl(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    patch_ids: np.ndarray,
    n_patches: int,
) -> None:
    """Write a multi-solid ASCII STL with one solid per patch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pid in range(n_patches):
            mask = patch_ids == pid
            f.write(f"solid patch_{pid}\n")
            for fi in np.where(mask)[0]:
                n = normals[fi]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for vi in range(3):
                    pt = verts[faces[fi, vi]]
                    f.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
                f.write("    endloop\n  endfacet\n")
            f.write(f"endsolid patch_{pid}\n")
