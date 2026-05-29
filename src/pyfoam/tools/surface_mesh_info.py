"""
surfaceMeshInfo — compute detailed surface mesh statistics.

Mirrors OpenFOAM's ``surfaceMeshInfo`` utility.  Analyses a triangulated
surface and reports geometric and topological statistics:

- Point, face, and edge counts
- Bounding box dimensions
- Face area statistics (min, max, mean, total)
- Edge length statistics (min, max, mean)
- Vertex valence distribution
- Manifoldness and watertightness checks

Usage::

    from pyfoam.tools.surface_mesh_info import surface_mesh_info

    info = surface_mesh_info("body.stl")
    print(info.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["SurfaceMeshInfo", "surface_mesh_info"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMeshInfo:
    """Structured result from :func:`surface_mesh_info`.

    Attributes
    ----------
    n_points : int
        Number of unique vertices.
    n_faces : int
        Number of triangles.
    n_edges : int
        Number of unique edges.
    n_open_edges : int
        Edges shared by only one triangle.
    n_non_manifold_edges : int
        Edges shared by more than two triangles.
    is_manifold : bool
        ``True`` if no non-manifold edges.
    is_watertight : bool
        ``True`` if no open edges and manifold.
    bbox_min : np.ndarray
        ``(3,)`` bounding-box minimum corner.
    bbox_max : np.ndarray
        ``(3,)`` bounding-box maximum corner.
    bbox_size : np.ndarray
        ``(3,)`` bounding-box extent (max - min).
    min_face_area : float
        Smallest triangle area.
    max_face_area : float
        Largest triangle area.
    mean_face_area : float
        Mean triangle area.
    total_area : float
        Sum of all triangle areas.
    min_edge_length : float
        Shortest edge length.
    max_edge_length : float
        Longest edge length.
    mean_edge_length : float
        Mean edge length.
    min_valence : int
        Minimum number of edges meeting at a vertex.
    max_valence : int
        Maximum number of edges meeting at a vertex.
    mean_valence : float
        Mean vertex valence.
    n_degenerate_faces : int
        Triangles with near-zero area.
    genus : int
        Approximate topological genus (Euler characteristic based).
    """

    n_points: int = 0
    n_faces: int = 0
    n_edges: int = 0
    n_open_edges: int = 0
    n_non_manifold_edges: int = 0
    is_manifold: bool = True
    is_watertight: bool = True
    bbox_min: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    bbox_max: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    bbox_size: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    min_face_area: float = 0.0
    max_face_area: float = 0.0
    mean_face_area: float = 0.0
    total_area: float = 0.0
    min_edge_length: float = 0.0
    max_edge_length: float = 0.0
    mean_edge_length: float = 0.0
    min_valence: int = 0
    max_valence: int = 0
    mean_valence: float = 0.0
    n_degenerate_faces: int = 0
    genus: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Surface mesh info: {self.n_points} points, {self.n_faces} faces, "
            f"{self.n_edges} edges",
            f"  Bounding box: [{self.bbox_min[0]:.6g}, {self.bbox_min[1]:.6g}, "
            f"{self.bbox_min[2]:.6g}] -- [{self.bbox_max[0]:.6g}, "
            f"{self.bbox_max[1]:.6g}, {self.bbox_max[2]:.6g}]",
            f"  Bounding box size: [{self.bbox_size[0]:.6g}, "
            f"{self.bbox_size[1]:.6g}, {self.bbox_size[2]:.6g}]",
            f"  Face area: min={self.min_face_area:.6e}  max={self.max_face_area:.6e}  "
            f"mean={self.mean_face_area:.6e}  total={self.total_area:.6e}",
            f"  Edge length: min={self.min_edge_length:.6e}  max={self.max_edge_length:.6e}  "
            f"mean={self.mean_edge_length:.6e}",
            f"  Vertex valence: min={self.min_valence}  max={self.max_valence}  "
            f"mean={self.mean_valence:.2f}",
            f"  Open edges: {self.n_open_edges}",
            f"  Non-manifold edges: {self.n_non_manifold_edges}",
            f"  Degenerate faces: {self.n_degenerate_faces}",
            f"  Manifold: {self.is_manifold}",
            f"  Watertight: {self.is_watertight}",
            f"  Genus (approx): {self.genus}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_mesh_info(
    surface_path: Union[str, Path] = "",
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    area_tol: float = 1e-30,
) -> SurfaceMeshInfo:
    """Compute detailed statistics for a triangulated surface mesh.

    The surface can be provided either as a file path (STL/OBJ/VTK) or
    directly as vertex/face arrays.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file.  Ignored when *vertices* and *faces*
        are provided directly.
    vertices : np.ndarray, optional
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray, optional
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    area_tol : float
        Minimum face area threshold for degenerate-face detection.

    Returns
    -------
    SurfaceMeshInfo
        Mesh statistics.

    Raises
    ------
    FileNotFoundError
        If *surface_path* does not exist and no arrays are provided.
    """
    # Obtain geometry
    if vertices is not None and faces is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = np.asarray(faces, dtype=np.int32)
    else:
        from pyfoam.tools.surface_convert import _rs, _df

        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, _, facs = _rs(p, fmt)

    info = SurfaceMeshInfo()
    info.n_points = verts.shape[0]
    info.n_faces = facs.shape[0]

    if info.n_faces == 0:
        return info

    # Bounding box
    info.bbox_min = verts.min(axis=0)
    info.bbox_max = verts.max(axis=0)
    info.bbox_size = info.bbox_max - info.bbox_min

    # Face areas
    areas = _compute_areas(verts, facs)
    info.min_face_area = float(areas.min())
    info.max_face_area = float(areas.max())
    info.mean_face_area = float(areas.mean())
    info.total_area = float(areas.sum())
    info.n_degenerate_faces = int(np.sum(areas < area_tol))

    # Edge adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_lengths: dict[tuple[int, int], float] = {}
    vertex_edges: dict[int, set[int]] = {}

    for fi in range(info.n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            vi, vj = int(tri[a]), int(tri[b])
            key = (min(vi, vj), max(vi, vj))
            edge_faces.setdefault(key, []).append(fi)
            if key not in edge_lengths:
                edge_lengths[key] = float(np.linalg.norm(verts[vi] - verts[vj]))
            vertex_edges.setdefault(vi, set()).add(key)
            vertex_edges.setdefault(vj, set()).add(key)

    info.n_edges = len(edge_faces)

    # Open and non-manifold edges
    n_open = 0
    n_non_manifold = 0
    for adj in edge_faces.values():
        if len(adj) == 1:
            n_open += 1
        elif len(adj) > 2:
            n_non_manifold += 1

    info.n_open_edges = n_open
    info.n_non_manifold_edges = n_non_manifold
    info.is_manifold = n_non_manifold == 0
    info.is_watertight = (n_open == 0) and (n_non_manifold == 0)

    # Edge length statistics
    elens = np.array(list(edge_lengths.values()), dtype=np.float64)
    if len(elens) > 0:
        info.min_edge_length = float(elens.min())
        info.max_edge_length = float(elens.max())
        info.mean_edge_length = float(elens.mean())

    # Vertex valence (number of edges meeting at each vertex)
    if vertex_edges:
        valences = np.array([len(es) for es in vertex_edges.values()])
        info.min_valence = int(valences.min())
        info.max_valence = int(valences.max())
        info.mean_valence = float(valences.mean())

    # Euler characteristic: chi = V - E + F
    # For closed orientable surface: genus = (2 - chi) / 2
    chi = info.n_points - info.n_edges + info.n_faces
    if info.is_watertight:
        info.genus = max(0, (2 - chi) // 2)

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area of each triangle."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)
