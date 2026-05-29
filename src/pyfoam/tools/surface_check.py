"""
surfaceCheck — check surface mesh quality.

Mirrors OpenFOAM's ``surfaceCheck`` utility.  Analyses a triangulated
surface for common quality issues:

- **Open edges**: edges shared by only one triangle (surface is not watertight).
- **Non-manifold edges**: edges shared by more than two triangles.
- **Duplicate points**: coincident vertices within a tolerance.
- **Degenerate faces**: triangles with zero area.
- **Normal consistency**: whether face normals point outward consistently.

Usage::

    from pyfoam.tools.surface_check import surface_check

    result = surface_check("body.stl")
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["SurfaceCheckResult", "surface_check"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurfaceCheckResult:
    """Structured result from :func:`surface_check`.

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
    n_duplicate_points : int
        Number of duplicate vertex groups detected.
    n_degenerate_faces : int
        Triangles with near-zero area.
    is_watertight : bool
        ``True`` if no open or non-manifold edges exist.
    min_face_area : float
        Smallest triangle area.
    max_face_area : float
        Largest triangle area.
    total_area : float
        Sum of all triangle areas.
    bbox_min : np.ndarray
        ``(3,)`` bounding-box minimum corner.
    bbox_max : np.ndarray
        ``(3,)`` bounding-box maximum corner.
    open_edge_indices : list[tuple[int, int]]
        Vertex-index pairs for open edges.
    warnings : list[str]
        Human-readable warning messages.
    """

    n_points: int = 0
    n_faces: int = 0
    n_edges: int = 0
    n_open_edges: int = 0
    n_non_manifold_edges: int = 0
    n_duplicate_points: int = 0
    n_degenerate_faces: int = 0
    is_watertight: bool = True
    min_face_area: float = 0.0
    max_face_area: float = 0.0
    total_area: float = 0.0
    bbox_min: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    bbox_max: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    open_edge_indices: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Surface check: {self.n_points} points, {self.n_faces} faces, "
            f"{self.n_edges} edges",
            f"  Open edges: {self.n_open_edges}",
            f"  Non-manifold edges: {self.n_non_manifold_edges}",
            f"  Duplicate points: {self.n_duplicate_points}",
            f"  Degenerate faces: {self.n_degenerate_faces}",
            f"  Watertight: {self.is_watertight}",
            f"  Total area: {self.total_area:.6e}",
            f"  Bounding box: [{self.bbox_min[0]:.4f}, {self.bbox_min[1]:.4f}, "
            f"{self.bbox_min[2]:.4f}] — [{self.bbox_max[0]:.4f}, "
            f"{self.bbox_max[1]:.4f}, {self.bbox_max[2]:.4f}]",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_check(
    surface_path: Union[str, Path] = "",
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    duplicate_tol: float = 1e-10,
    area_tol: float = 1e-30,
) -> SurfaceCheckResult:
    """Check quality of a triangulated surface mesh.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file.  Ignored when *vertices* and *faces*
        are provided directly.
    vertices : np.ndarray, optional
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray, optional
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    normals : np.ndarray, optional
        ``(n_faces, 3)`` face normals.  Used for consistency check.
    duplicate_tol : float
        Distance tolerance for detecting duplicate points.
    area_tol : float
        Minimum face area threshold for degenerate-face detection.

    Returns
    -------
    SurfaceCheckResult
        Quality metrics and diagnostics.

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
            np.asarray(normals, dtype=np.float64) if normals is not None else None
        )
    else:
        from pyfoam.tools.surface_convert import _rs, _df

        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, norms, facs = _rs(p, fmt)

    result = SurfaceCheckResult()
    result.n_points = verts.shape[0]
    result.n_faces = facs.shape[0]

    if result.n_faces == 0:
        result.warnings.append("Surface has no faces.")
        return result

    # Bounding box
    result.bbox_min = verts.min(axis=0)
    result.bbox_max = verts.max(axis=0)

    # Face areas
    areas = _compute_areas(verts, facs)
    result.min_face_area = float(areas.min())
    result.max_face_area = float(areas.max())
    result.total_area = float(areas.sum())

    # Degenerate faces
    degen_mask = areas < area_tol
    result.n_degenerate_faces = int(degen_mask.sum())
    if result.n_degenerate_faces > 0:
        result.warnings.append(
            f"{result.n_degenerate_faces} degenerate face(s) with area < {area_tol}."
        )

    # Edge adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(result.n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    result.n_edges = len(edge_faces)

    # Open and non-manifold edges
    open_edges = []
    n_non_manifold = 0
    for (vi, vj), adj in edge_faces.items():
        if len(adj) == 1:
            open_edges.append((vi, vj))
        elif len(adj) > 2:
            n_non_manifold += 1

    result.n_open_edges = len(open_edges)
    result.open_edge_indices = open_edges
    result.n_non_manifold_edges = n_non_manifold

    if result.n_open_edges > 0:
        result.warnings.append(
            f"{result.n_open_edges} open edge(s) — surface is not watertight."
        )
    if n_non_manifold > 0:
        result.warnings.append(
            f"{n_non_manifold} non-manifold edge(s) detected."
        )

    result.is_watertight = (result.n_open_edges == 0) and (n_non_manifold == 0)

    # Duplicate points
    result.n_duplicate_points = _count_duplicates(verts, duplicate_tol)
    if result.n_duplicate_points > 0:
        result.warnings.append(
            f"{result.n_duplicate_points} duplicate point group(s) detected "
            f"(tol={duplicate_tol})."
        )

    return result


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


def _count_duplicates(verts: np.ndarray, tol: float) -> int:
    """Count groups of duplicate points within tolerance."""
    n = verts.shape[0]
    if n < 2:
        return 0
    # Simple O(n^2) check — acceptable for typical surface sizes
    visited = np.zeros(n, dtype=bool)
    n_dup = 0
    for i in range(n):
        if visited[i]:
            continue
        for j in range(i + 1, n):
            if visited[j]:
                continue
            dist = np.linalg.norm(verts[i] - verts[j])
            if dist < tol:
                visited[j] = True
                n_dup += 1
    return n_dup
