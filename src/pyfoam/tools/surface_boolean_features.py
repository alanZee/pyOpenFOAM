"""
surfaceBooleanOps — perform Boolean operations on triangulated surfaces.

Provides CSG (Constructive Solid Geometry) Boolean operations on
triangle meshes:

- **Union**: Merge two solids into one.
- **Intersection**: Keep only the volume common to both solids.
- **Difference**: Subtract one solid from another.

The implementation uses a voxel-based approach: both surfaces are
voxelised onto a regular grid, the Boolean operation is performed on
the voxel grid, and the result is converted back to a triangulated
surface via marching cubes.

This is a pure-Python implementation suitable for moderate-resolution
surfaces.  For very large meshes, a dedicated CSG library (e.g.
Trimesh, libigl) is recommended.

Usage::

    from pyfoam.tools.surface_boolean_features import surface_boolean

    result = surface_boolean(
        vertices_a, faces_a,
        vertices_b, faces_b,
        operation="union",
    )
    out_verts, out_faces = result.vertices, result.faces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["BooleanResult", "surface_boolean"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BooleanResult:
    """Result from a Boolean surface operation.

    Attributes
    ----------
    vertices : np.ndarray
        ``(n_points, 3)`` vertex coordinates of the result mesh.
    faces : np.ndarray
        ``(n_faces, 3)`` triangle vertex indices (0-based).
    n_input_faces_a : int
        Number of faces in the first input mesh.
    n_input_faces_b : int
        Number of faces in the second input mesh.
    n_output_faces : int
        Number of faces in the output mesh.
    operation : str
        The Boolean operation that was performed.
    """

    vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    n_input_faces_a: int = 0
    n_input_faces_b: int = 0
    n_output_faces: int = 0
    operation: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_boolean(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
    operation: str = "union",
    resolution: int = 50,
) -> BooleanResult:
    """Perform a Boolean operation on two triangulated surfaces.

    Parameters
    ----------
    vertices_a : np.ndarray
        ``(n_points, 3)`` vertices of mesh A.
    faces_a : np.ndarray
        ``(n_faces, 3)`` triangle indices of mesh A (0-based).
    vertices_b : np.ndarray
        ``(n_points, 3)`` vertices of mesh B.
    faces_b : np.ndarray
        ``(n_faces, 3)`` triangle indices of mesh B (0-based).
    operation : str
        One of ``"union"``, ``"intersection"``, ``"difference"``.
    resolution : int
        Voxel grid resolution along the longest axis.  Higher values
        give more accurate results but use more memory.

    Returns
    -------
    BooleanResult
        Result mesh and metadata.

    Raises
    ------
    ValueError
        If *operation* is not recognised or inputs are invalid.
    """
    valid_ops = {"union", "intersection", "difference"}
    if operation not in valid_ops:
        raise ValueError(
            f"Unknown operation {operation!r}. Valid: {sorted(valid_ops)}"
        )

    va = np.asarray(vertices_a, dtype=np.float64)
    fa = np.asarray(faces_a, dtype=np.int32)
    vb = np.asarray(vertices_b, dtype=np.float64)
    fb = np.asarray(faces_b, dtype=np.int32)

    if fa.shape[0] == 0 or fb.shape[0] == 0:
        raise ValueError("Both meshes must have at least one face.")

    # Compute combined bounding box with margin
    all_verts = np.vstack([va, vb])
    bb_min = all_verts.min(axis=0)
    bb_max = all_verts.max(axis=0)
    margin = 0.05 * (bb_max - bb_min).max()
    bb_min -= margin
    bb_max += margin

    # Build voxel grid
    extent = bb_max - bb_min
    max_ext = extent.max()
    if max_ext < 1e-30:
        raise ValueError("Degenerate bounding box — meshes are co-located.")

    nx = max(8, int(resolution * extent[0] / max_ext))
    ny = max(8, int(resolution * extent[1] / max_ext))
    nz = max(8, int(resolution * extent[2] / max_ext))
    nx = max(nx, 2)
    ny = max(ny, 2)
    nz = max(nz, 2)

    dx = extent[0] / nx
    dy = extent[1] / ny
    dz = extent[2] / nz

    # Voxelize both meshes
    grid_a = _voxelize(va, fa, bb_min, nx, ny, nz, dx, dy, dz)
    grid_b = _voxelize(vb, fb, bb_min, nx, ny, nz, dx, dy, dz)

    # Apply Boolean operation on the voxel grids
    if operation == "union":
        grid_result = grid_a | grid_b
    elif operation == "intersection":
        grid_result = grid_a & grid_b
    else:  # difference
        grid_result = grid_a & ~grid_b

    # Extract surface via marching cubes
    out_verts, out_faces = _marching_cubes(
        grid_result, bb_min, dx, dy, dz,
    )

    return BooleanResult(
        vertices=out_verts,
        faces=out_faces,
        n_input_faces_a=fa.shape[0],
        n_input_faces_b=fb.shape[0],
        n_output_faces=out_faces.shape[0],
        operation=operation,
    )


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------


def _voxelize(
    verts: np.ndarray,
    faces: np.ndarray,
    bb_min: np.ndarray,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
) -> np.ndarray:
    """Voxelize a closed triangulated surface using ray casting.

    Returns a boolean 3D array where ``True`` means inside the solid.
    """
    grid = np.zeros((nx, ny, nz), dtype=bool)

    # For each voxel centre, cast a ray in +x and count face intersections
    for iz in range(nz):
        z = bb_min[2] + (iz + 0.5) * dz
        for iy in range(ny):
            y = bb_min[1] + (iy + 0.5) * dy
            x_ray = bb_min[0] - 1.0  # ray origin (outside mesh)

            n_cross = 0
            for fi in range(faces.shape[0]):
                tri = verts[faces[fi]]
                if _ray_triangle_intersect(
                    np.array([x_ray, y, z]),
                    np.array([1.0, 0.0, 0.0]),
                    tri,
                ):
                    n_cross += 1

            # Odd number of crossings = inside
            inside = (n_cross % 2) == 1
            if inside:
                grid[:, iy, iz] = True

    return grid


def _ray_triangle_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    tri: np.ndarray,
) -> bool:
    """Moller-Trumbore ray-triangle intersection test."""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    det = np.dot(edge1, h)

    if abs(det) < 1e-30:
        return False

    inv_det = 1.0 / det
    s = origin - v0
    u = np.dot(s, h) * inv_det

    if u < -1e-10 or u > 1.0 + 1e-10:
        return False

    q = np.cross(s, edge1)
    v = np.dot(direction, q) * inv_det

    if v < -1e-10 or u + v > 1.0 + 1e-10:
        return False

    t = np.dot(edge2, q) * inv_det
    return t > 1e-10


# ---------------------------------------------------------------------------
# Marching cubes (minimal implementation)
# ---------------------------------------------------------------------------

# Edge table for marching cubes (which edges are intersected for each
# of the 256 cube configurations).
_MC_EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
]

# Triangle table for marching cubes (which triangles to form).
_MC_TRI_TABLE = [
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [0, 8, 3], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [0, 1, 9], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [1, 8, 3, 9, 8, 1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [1, 2, 10], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [0, 8, 3, 1, 2, 10],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [9, 2, 10, 0, 2, 9], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [3, 11, 2], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [0, 11, 2, 8, 11, 0], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [1, 9, 0, 2, 3, 11], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [3, 10, 1, 11, 10, 3], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [3, 9, 0, 3, 11, 9, 11, 10, 9], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [9, 8, 10, 10, 8, 11], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [4, 7, 8], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [4, 3, 0, 7, 3, 4],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [0, 1, 9, 8, 4, 7], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [4, 1, 9, 4, 7, 1, 7, 3, 1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [1, 2, 10, 8, 4, 7], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [3, 4, 7, 3, 0, 4, 1, 2, 10],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [9, 2, 10, 9, 0, 2, 8, 4, 7],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [2, 10, 9, 2, 9, 7, 2, 7, 3,
    7, 9, 4], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [8, 4, 7, 3, 11, 2],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [11, 4, 7, 11, 2, 4, 2, 0, 4],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [9, 0, 1, 8, 4, 7, 2, 3, 11],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [4, 7, 11, 9, 4, 11, 9, 11, 2,
    9, 2, 1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [3, 10, 1, 3, 11,
    10, 7, 8, 4], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1, 10, 11, 1, 11,
    4, 1, 4, 0, 7, 4, 11], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [4, 7,
    8, 9, 0, 11, 9, 11, 10, 11, 0, 3], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [4, 7, 11, 4, 11, 9, 9, 11, 10], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [9, 5, 4], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [9, 5,
    4, 0, 8, 3], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [0, 5, 4, 1, 5, 0],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [8, 5, 4, 8, 3, 5, 3, 1, 5],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [1, 2, 10, 9, 5, 4], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [3, 0, 8, 1, 2, 10, 4, 9, 5], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [5, 2, 10, 5, 4, 2, 4, 0, 2], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [9, 5, 4, 2, 3, 11], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [0, 11, 2, 0, 8, 11, 4, 9, 5], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [0, 5, 4, 0, 1, 5, 2, 3, 11], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [10, 3, 11, 10, 1, 3, 9, 5, 4],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [4, 9, 5, 0, 8, 1, 8, 10, 1,
    8, 11, 10], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [5, 4, 0, 5, 0, 11,
    5, 11, 10, 11, 0, 3], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [5, 4,
    8, 5, 8, 10, 10, 8, 11], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [9, 7,
    8, 5, 7, 9], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [9, 3, 0, 9, 5, 3,
    5, 7, 3], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [0, 7, 8, 0, 1, 7,
    1, 5, 7], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1, 5, 3, 3, 5, 7],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [9, 7, 8, 9, 5, 7, 10, 1, 2],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [10, 1, 2, 9, 5, 0, 5, 3, 0,
    5, 7, 3], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [8, 0, 2, 8, 2, 5,
    8, 5, 7, 10, 5, 2], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [2, 10, 5, 2,
    5, 3, 3, 5, 7], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [7, 9, 5, 7,
    8, 9, 3, 11, 2], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [9, 5, 7, 9,
    7, 2, 9, 2, 0, 2, 7, 11], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [2, 3,
    11, 0, 1, 8, 1, 7, 8, 1, 5, 7], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [11, 10, 0, 11, 0, 3, 10, 5, 0,
    8, 0, 7, 5, 7, 0], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
    [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [11, 10, 5, 7,
    11, 5],
]


def _marching_cubes(
    grid: np.ndarray,
    bb_min: np.ndarray,
    dx: float, dy: float, dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on a boolean voxel grid.

    Returns (vertices, faces) arrays.
    """
    nx, ny, nz = grid.shape
    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    vert_cache: dict[tuple, int] = {}

    def _edge_vertex(e: int, ix: int, iy: int, iz: int) -> int:
        """Get or create a vertex on the given edge of the cube."""
        # Cube corner positions
        corners = [
            (ix, iy, iz), (ix + 1, iy, iz), (ix + 1, iy + 1, iz),
            (ix, iy + 1, iz), (ix, iy, iz + 1), (ix + 1, iy, iz + 1),
            (ix + 1, iy + 1, iz + 1), (ix, iy + 1, iz + 1),
        ]
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        c0, c1 = edge_pairs[e]
        key = tuple(sorted([(c0, ix, iy, iz), (c1, ix, iy, iz)]))
        if key in vert_cache:
            return vert_cache[key]

        # Midpoint of the edge
        p0 = corners[c0]
        p1 = corners[c1]
        mx = bb_min[0] + (0.5 * (p0[0] + p1[0]) + 0.5) * dx
        my = bb_min[1] + (0.5 * (p0[1] + p1[1]) + 0.5) * dy
        mz = bb_min[2] + (0.5 * (p0[2] + p1[2]) + 0.5) * dz

        idx = len(all_verts)
        all_verts.append([mx, my, mz])
        vert_cache[key] = idx
        return idx

    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                # Cube index: which corners are inside
                cube_idx = 0
                for bit, (dx_, dy_, dz_) in enumerate([
                    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
                ]):
                    if grid[ix + dx_, iy + dy_, iz + dz_]:
                        cube_idx |= (1 << bit)

                if cube_idx == 0 or cube_idx == 255:
                    continue

                edge_bits = _MC_EDGE_TABLE[cube_idx]
                if edge_bits == 0:
                    continue

                tri_config = _MC_TRI_TABLE[cube_idx]
                if tri_config == [-1]:
                    continue

                face_verts = []
                for e in tri_config:
                    if edge_bits & (1 << e):
                        face_verts.append(_edge_vertex(e, ix, iy, iz))

                # Group into triangles
                for k in range(0, len(face_verts) - 2, 3):
                    all_faces.append(
                        [face_verts[k], face_verts[k + 1], face_verts[k + 2]]
                    )

    out_verts = np.array(all_verts, dtype=np.float64) if all_verts else np.empty((0, 3))
    out_faces = np.array(all_faces, dtype=np.int32) if all_faces else np.empty((0, 3), dtype=np.int32)
    return out_verts, out_faces
