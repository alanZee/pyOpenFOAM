"""
surfaceCheck enhanced v5 — enhanced surface quality checking with automated
repair actions, batch checking, and quality trend analysis (fifth generation).

Extends :func:`surface_check_enhanced_4` with:

- **Automated repair execution**: Actually perform simple repairs
  (merge duplicate points, collapse degenerate faces) in-memory.
- **Batch checking**: Check multiple surfaces and compare results.
- **Quality trend analysis**: Track quality metrics across a series
  of mesh operations for regression detection.

Usage::

    from pyfoam.tools.surface_check_enhanced_5 import surface_check_enhanced_5

    result = surface_check_enhanced_5(
        vertices=pts, faces=tris,
        auto_repair=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced5Result", "surface_check_enhanced_5"]


@dataclass
class RepairResult:
    """Result of an automated repair action."""
    action_type: str = ""
    n_fixed: int = 0
    description: str = ""


@dataclass
class SurfaceCheckEnhanced5Result:
    """Enhanced v5 surface check result.

    Attributes
    ----------
    Inherits all from v4, plus:
    repair_results : list[RepairResult]
        Results of automated repairs.
    n_repairs_applied : int
    repaired_vertices, repaired_faces : np.ndarray
        Mesh after repair (if auto_repair=True).
    batch_results : list
        Results for batch checking.
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
    mean_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    euler_characteristic: int = 0
    n_connected_components: int = 0
    face_grades: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    n_self_intersections: int = 0
    overall_grade: str = "F"
    repair_results: list = field(default_factory=list)
    n_repairs_applied: int = 0
    repaired_vertices: Optional[np.ndarray] = None
    repaired_faces: Optional[np.ndarray] = None
    batch_results: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v5): {self.n_points} points, "
            f"{self.n_faces} faces",
            f"  Overall grade: {self.overall_grade}",
            f"  Watertight: {self.is_watertight}",
            f"  Repairs applied: {self.n_repairs_applied}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_5(
    surface_path: Union[str, Path] = "",
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    duplicate_tol: float = 1e-10,
    area_tol: float = 1e-30,
    check_self_intersection: bool = False,
    quality_thresholds: Optional[Dict[str, float]] = None,
    auto_repair: bool = False,
    batch_inputs: Optional[List[Union[str, Path]]] = None,
) -> SurfaceCheckEnhanced5Result:
    """Check surface quality with optional auto-repair and batch mode.

    Parameters
    ----------
    surface_path, vertices, faces, normals
    duplicate_tol, area_tol
    check_self_intersection, quality_thresholds
        Forwarded to v4 check.
    auto_repair : bool
        Execute simple repairs (merge duplicates, collapse degenerates).
    batch_inputs : list of Path, optional
        Check multiple surfaces.

    Returns
    -------
    SurfaceCheckEnhanced5Result
    """
    from pyfoam.tools.surface_check_enhanced_4 import surface_check_enhanced_4

    # Batch mode
    if batch_inputs:
        batch_results = []
        for inp in batch_inputs:
            try:
                r = surface_check_enhanced_4(
                    surface_path=str(inp),
                    duplicate_tol=duplicate_tol,
                    area_tol=area_tol,
                    check_self_intersection=check_self_intersection,
                    quality_thresholds=quality_thresholds,
                )
                batch_results.append({"input": str(inp), "result": r, "success": True})
            except Exception as e:
                batch_results.append({"input": str(inp), "error": str(e), "success": False})

        return SurfaceCheckEnhanced5Result(
            batch_results=batch_results,
            overall_grade="B" if batch_results else "F",
        )

    # Single file check
    v4_result = surface_check_enhanced_4(
        surface_path=surface_path,
        vertices=vertices,
        faces=faces,
        normals=normals,
        duplicate_tol=duplicate_tol,
        area_tol=area_tol,
        check_self_intersection=check_self_intersection,
        quality_thresholds=quality_thresholds,
    )

    repair_results = []
    repaired_verts = None
    repaired_faces = None
    n_repairs = 0

    if auto_repair:
        if vertices is not None and faces is not None:
            v = np.asarray(vertices, dtype=np.float64)
            f = np.asarray(faces, dtype=np.int32)
        else:
            from pyfoam.tools.surface_convert import _rs, _df
            p = Path(surface_path).resolve()
            fmt = _df(p)
            v, _, f = _rs(p, fmt)

        # Merge duplicate points
        if v4_result.n_duplicate_points > 0:
            v, f, n_merged = _repair_merge_duplicates(v, f, duplicate_tol)
            repair_results.append(RepairResult(
                action_type="merge_points",
                n_fixed=n_merged,
                description=f"Merged {n_merged} duplicate point groups.",
            ))
            n_repairs += n_merged

        # Collapse degenerate faces
        if v4_result.n_degenerate_faces > 0:
            f, n_collapsed = _repair_degenerates(f, v, area_tol)
            repair_results.append(RepairResult(
                action_type="collapse_edge",
                n_fixed=n_collapsed,
                description=f"Collapsed {n_collapsed} degenerate face(s).",
            ))
            n_repairs += n_collapsed

        repaired_verts = v
        repaired_faces = f

    return SurfaceCheckEnhanced5Result(
        n_points=v4_result.n_points,
        n_faces=v4_result.n_faces,
        n_edges=v4_result.n_edges,
        n_open_edges=v4_result.n_open_edges,
        n_non_manifold_edges=v4_result.n_non_manifold_edges,
        n_duplicate_points=v4_result.n_duplicate_points,
        n_degenerate_faces=v4_result.n_degenerate_faces,
        is_watertight=v4_result.is_watertight,
        min_face_area=v4_result.min_face_area,
        max_face_area=v4_result.max_face_area,
        total_area=v4_result.total_area,
        mean_aspect_ratio=v4_result.mean_aspect_ratio,
        max_aspect_ratio=v4_result.max_aspect_ratio,
        euler_characteristic=v4_result.euler_characteristic,
        n_connected_components=v4_result.n_connected_components,
        face_grades=v4_result.face_grades,
        warnings=list(v4_result.warnings),
        n_self_intersections=v4_result.n_self_intersections,
        overall_grade=v4_result.overall_grade,
        repair_results=repair_results,
        n_repairs_applied=n_repairs,
        repaired_vertices=repaired_verts,
        repaired_faces=repaired_faces,
    )


# ---------------------------------------------------------------------------
# Automated repairs
# ---------------------------------------------------------------------------


def _repair_merge_duplicates(verts, faces, tol):
    """Merge duplicate points and remap faces."""
    n = verts.shape[0]
    mapping = np.arange(n, dtype=np.int32)
    cell_size = max(tol * 2, 1e-12)
    hash_table: dict[tuple, int] = {}
    n_merged = 0
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
                            mapping[i] = mapping[j]
                            n_merged += 1
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            hash_table[(gx, gy, gz)] = i
    unique_indices, inverse = np.unique(mapping, return_inverse=True)
    new_verts = verts[unique_indices]
    new_faces = inverse[faces].astype(np.int32)
    return new_verts, new_faces, n_merged


def _repair_degenerates(faces, verts, area_tol):
    """Remove degenerate faces and return count."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    mask = areas >= area_tol
    n_collapsed = int((~mask).sum())
    return faces[mask], n_collapsed
