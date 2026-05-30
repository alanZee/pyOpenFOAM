"""
surfaceCheck enhanced v3 — enhanced surface quality checking with radius
ratio, condition number, and per-face detail reporting (third generation).

Extends :func:`surface_check_enhanced_2` with:

- **Radius ratio**: Per-triangle circumradius-to-inradius ratio.
- **Condition number**: Jacobian condition number estimate per face.
- **Per-face detail**: Optional storage of per-face quality arrays.
- **Check summary string**: Machine-parseable summary of key metrics.

Usage::

    from pyfoam.tools.surface_check_enhanced_3 import surface_check_enhanced_3

    result = surface_check_enhanced_3(
        vertices=pts, faces=tris,
        check_self_intersection=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced3Result", "surface_check_enhanced_3"]


@dataclass
class SurfaceCheckEnhanced3Result:
    """Enhanced v3 surface check result.

    Attributes
    ----------
    n_points, n_faces, n_edges : int
    n_open_edges, n_non_manifold_edges, n_duplicate_points : int
    n_degenerate_faces, n_non_manifold_vertices : int
    is_watertight : bool
    min_face_area, max_face_area, total_area : float
    bbox_min, bbox_max : np.ndarray
    mean_aspect_ratio, max_aspect_ratio, mean_skewness : float
    min_angle_mean, max_angle_mean : float
    euler_characteristic : int
    n_connected_components : int
    genus : int
    face_grades : dict[str, int]
    face_areas, face_aspect_ratios, face_skewness : np.ndarray
    face_radius_ratios : np.ndarray
        Per-triangle circumradius-to-inradius ratio.
    face_condition_numbers : np.ndarray
        Per-triangle Jacobian condition number estimate.
    mean_radius_ratio : float
    mean_condition_number : float
    degenerate_face_indices : list[int]
    warnings : list[str]
    """

    n_points: int = 0
    n_faces: int = 0
    n_edges: int = 0
    n_open_edges: int = 0
    n_non_manifold_edges: int = 0
    n_duplicate_points: int = 0
    n_degenerate_faces: int = 0
    n_non_manifold_vertices: int = 0
    is_watertight: bool = True
    min_face_area: float = 0.0
    max_face_area: float = 0.0
    total_area: float = 0.0
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mean_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    mean_skewness: float = 0.0
    min_angle_mean: float = 0.0
    max_angle_mean: float = 0.0
    euler_characteristic: int = 0
    n_connected_components: int = 0
    genus: int = 0
    face_grades: Dict[str, int] = field(default_factory=dict)
    face_areas: np.ndarray = field(default_factory=lambda: np.empty(0))
    face_aspect_ratios: np.ndarray = field(default_factory=lambda: np.empty(0))
    face_skewness: np.ndarray = field(default_factory=lambda: np.empty(0))
    face_radius_ratios: np.ndarray = field(default_factory=lambda: np.empty(0))
    face_condition_numbers: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_radius_ratio: float = 0.0
    mean_condition_number: float = 0.0
    degenerate_face_indices: list = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v3): {self.n_points} points, "
            f"{self.n_faces} faces, {self.n_edges} edges",
            f"  Open edges: {self.n_open_edges}",
            f"  Non-manifold edges: {self.n_non_manifold_edges}",
            f"  Non-manifold vertices: {self.n_non_manifold_vertices}",
            f"  Duplicate points: {self.n_duplicate_points}",
            f"  Degenerate faces: {self.n_degenerate_faces}",
            f"  Watertight: {self.is_watertight}",
            f"  Total area: {self.total_area:.6e}",
            f"  Mean aspect ratio: {self.mean_aspect_ratio:.4f}",
            f"  Max aspect ratio: {self.max_aspect_ratio:.4f}",
            f"  Mean skewness: {self.mean_skewness:.4f}",
            f"  Mean radius ratio: {self.mean_radius_ratio:.4f}",
            f"  Mean condition number: {self.mean_condition_number:.4f}",
            f"  Euler characteristic: {self.euler_characteristic}",
            f"  Connected components: {self.n_connected_components}",
            f"  Genus: {self.genus}",
        ]
        if self.face_grades:
            lines.append(f"  Quality grades: {self.face_grades}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_3(
    surface_path: Union[str, Path] = "",
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    duplicate_tol: float = 1e-10,
    area_tol: float = 1e-30,
    check_self_intersection: bool = False,
    quality_thresholds: Optional[Dict[str, float]] = None,
) -> SurfaceCheckEnhanced3Result:
    """Check surface quality with radius ratio and condition number metrics.

    Parameters
    ----------
    surface_path, vertices, faces, normals
        Geometry input.
    duplicate_tol, area_tol : float
        Tolerances.
    check_self_intersection : bool
    quality_thresholds : dict, optional

    Returns
    -------
    SurfaceCheckEnhanced3Result
    """
    if vertices is not None and faces is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = np.asarray(faces, dtype=np.int32)
        norms = np.asarray(normals, dtype=np.float64) if normals is not None else None
    else:
        from pyfoam.tools.surface_convert import _rs, _df
        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, norms, facs = _rs(p, fmt)

    result = SurfaceCheckEnhanced3Result()
    result.n_points = verts.shape[0]
    result.n_faces = facs.shape[0]

    if result.n_faces == 0:
        result.warnings.append("Surface has no faces.")
        return result

    result.bbox_min = verts.min(axis=0)
    result.bbox_max = verts.max(axis=0)

    # Face areas
    areas = _compute_areas(verts, facs)
    result.face_areas = areas
    result.min_face_area = float(areas.min())
    result.max_face_area = float(areas.max())
    result.total_area = float(areas.sum())

    degen_mask = areas < area_tol
    result.n_degenerate_faces = int(degen_mask.sum())
    result.degenerate_face_indices = list(np.where(degen_mask)[0])
    if result.n_degenerate_faces > 0:
        result.warnings.append(f"{result.n_degenerate_faces} degenerate face(s).")

    # Aspect ratios
    ar = _compute_aspect_ratios(verts, facs)
    result.face_aspect_ratios = ar
    result.mean_aspect_ratio = float(np.mean(ar)) if ar.size > 0 else 0.0
    result.max_aspect_ratio = float(np.max(ar)) if ar.size > 0 else 0.0

    # Skewness and angles
    sk, min_angles, max_angles = _compute_skewness_and_angles(verts, facs)
    result.face_skewness = sk
    result.mean_skewness = float(np.mean(sk)) if sk.size > 0 else 0.0
    result.min_angle_mean = float(np.mean(min_angles)) if min_angles.size > 0 else 0.0
    result.max_angle_mean = float(np.mean(max_angles)) if max_angles.size > 0 else 0.0

    # Radius ratio (circumradius / inradius)
    rr = _compute_radius_ratio(verts, facs)
    result.face_radius_ratios = rr
    result.mean_radius_ratio = float(np.mean(rr)) if rr.size > 0 else 0.0

    # Condition number
    cn = _compute_condition_numbers(verts, facs)
    result.face_condition_numbers = cn
    result.mean_condition_number = float(np.mean(cn)) if cn.size > 0 else 0.0

    # Edge adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(result.n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    result.n_edges = len(edge_faces)
    open_edges = []
    n_non_manifold = 0
    for (vi, vj), adj in edge_faces.items():
        if len(adj) == 1:
            open_edges.append((vi, vj))
        elif len(adj) > 2:
            n_non_manifold += 1

    result.n_open_edges = len(open_edges)
    result.n_non_manifold_edges = n_non_manifold
    result.is_watertight = (result.n_open_edges == 0) and (n_non_manifold == 0)

    if result.n_open_edges > 0:
        result.warnings.append(f"{result.n_open_edges} open edge(s) — not watertight.")
    if n_non_manifold > 0:
        result.warnings.append(f"{n_non_manifold} non-manifold edge(s).")

    # Duplicate points
    result.n_duplicate_points = _count_duplicates(verts, duplicate_tol)
    if result.n_duplicate_points > 0:
        result.warnings.append(f"{result.n_duplicate_points} duplicate point group(s).")

    # Non-manifold vertices
    result.n_non_manifold_vertices = _count_non_manifold_vertices(facs, edge_faces)

    # Euler characteristic
    result.euler_characteristic = result.n_points - result.n_edges + result.n_faces

    # Connected components
    result.n_connected_components = _count_components(facs, result.n_points)

    # Genus
    if result.is_watertight:
        result.genus = max(0, (2 - result.euler_characteristic) // 2)

    # Quality grading
    default_thresh = {"ar_A": 1.5, "ar_B": 3.0, "ar_C": 10.0}
    if quality_thresholds:
        default_thresh.update(quality_thresholds)
    result.face_grades = _grade_faces(ar, default_thresh)

    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _compute_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _compute_aspect_ratios(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    safe_longest = np.where(longest > 1e-30, longest, 1.0)
    shortest_alt = 2.0 * areas / safe_longest
    safe_alt = np.where(shortest_alt > 1e-30, shortest_alt, 1e-30)
    return longest / safe_alt


def _compute_skewness_and_angles(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    a = v1 - v0
    b = v2 - v0
    c = v2 - v1

    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)

    safe = np.maximum(la * lb, 1e-30)
    cos_a = np.clip(np.sum(a * b, axis=1) / safe, -1, 1)
    safe = np.maximum(la * lc, 1e-30)
    cos_b = np.clip(-np.sum(a * c, axis=1) / safe, -1, 1)
    safe = np.maximum(lb * lc, 1e-30)
    cos_c = np.clip(np.sum(b * c, axis=1) / safe, -1, 1)

    angles = np.degrees(np.arccos(np.column_stack([cos_a, cos_b, cos_c])))
    min_angle = angles.min(axis=1)
    max_angle = angles.max(axis=1)
    skew = 1.0 - min_angle / 60.0
    return skew, min_angle, max_angle


def _compute_radius_ratio(verts, faces):
    """Circumradius-to-inradius ratio for each triangle."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    s = 0.5 * (e0 + e1 + e2)  # semi-perimeter
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Inradius: r = area / s
    safe_s = np.where(s > 1e-30, s, 1.0)
    inradius = area / safe_s

    # Circumradius: R = (e0 * e1 * e2) / (4 * area)
    safe_area = np.where(area > 1e-30, area, 1.0)
    circumradius = (e0 * e1 * e2) / (4.0 * safe_area)

    safe_inradius = np.where(inradius > 1e-30, inradius, 1e-30)
    return circumradius / safe_inradius


def _compute_condition_numbers(verts, faces):
    """Estimate Jacobian condition number for each triangle.

    Uses the ratio of longest to shortest edge as a proxy.
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest = np.minimum(np.minimum(e0, e1), e2)
    safe_shortest = np.where(shortest > 1e-30, shortest, 1e-30)
    return longest / safe_shortest


def _count_duplicates(verts, tol):
    n = verts.shape[0]
    if n < 2:
        return 0
    cell_size = max(tol * 2, 1e-12)
    hash_table: dict[tuple, int] = {}
    n_dup = 0
    for i in range(n):
        gx = int(np.floor(verts[i, 0] / cell_size))
        gy = int(np.floor(verts[i, 1] / cell_size))
        gz = int(np.floor(verts[i, 2] / cell_size))
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in hash_table:
                        j = hash_table[key]
                        if np.linalg.norm(verts[i] - verts[j]) < tol:
                            n_dup += 1
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            hash_table[(gx, gy, gz)] = i
    return n_dup


def _count_non_manifold_vertices(faces, edge_faces):
    vert_faces: dict[int, set[int]] = {}
    for fi in range(faces.shape[0]):
        for vi in faces[fi]:
            vert_faces.setdefault(int(vi), set()).add(fi)
    n_nm = 0
    for vi, adj_faces in vert_faces.items():
        if len(adj_faces) < 2:
            continue
        edge_count = 0
        for fi in adj_faces:
            tri = faces[fi]
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                ea, eb = int(tri[a]), int(tri[b])
                if ea == vi or eb == vi:
                    edge_count += 1
        if edge_count != len(adj_faces):
            n_nm += 1
    return n_nm


def _count_components(faces, n_points):
    parent = list(range(n_points))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for fi in range(faces.shape[0]):
        tri = faces[fi]
        union(int(tri[0]), int(tri[1]))
        union(int(tri[1]), int(tri[2]))

    roots = set(find(i) for i in range(n_points))
    return len(roots)


def _grade_faces(ar, thresholds):
    grades = {"A": 0, "B": 0, "C": 0, "F": 0}
    a_thresh = thresholds.get("ar_A", 1.5)
    b_thresh = thresholds.get("ar_B", 3.0)
    c_thresh = thresholds.get("ar_C", 10.0)
    for val in ar:
        if val < a_thresh:
            grades["A"] += 1
        elif val < b_thresh:
            grades["B"] += 1
        elif val < c_thresh:
            grades["C"] += 1
        else:
            grades["F"] += 1
    return grades
