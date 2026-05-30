"""
surfaceCheck enhanced v6 — enhanced surface quality checking with
differential checking, automated report generation, and repair
prioritisation (sixth generation).

Extends :func:`surface_check_enhanced_5` with:

- **Differential checking**: Compare two surface states and report
  only the changes (new degenerates, lost watertightness, etc.).
- **Automated report generation**: Produce a structured quality
  report in OpenFOAM dictionary format.
- **Repair prioritisation**: Rank repair actions by impact and
  suggest the optimal execution order.

Usage::

    from pyfoam.tools.surface_check_enhanced_6 import surface_check_enhanced_6

    result = surface_check_enhanced_6(
        vertices=pts, faces=tris,
        auto_repair=True,
        generate_report=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced6Result", "surface_check_enhanced_6"]


@dataclass
class RepairPriority:
    """Repair action with priority ranking."""
    action_type: str = ""
    priority: int = 0
    impact_score: float = 0.0
    description: str = ""


@dataclass
class DifferentialReport:
    """Summary of changes between two surface states."""
    n_new_degenerates: int = 0
    n_new_open_edges: int = 0
    watertightness_changed: bool = False
    grade_changed: bool = False
    previous_grade: str = ""
    current_grade: str = ""


@dataclass
class SurfaceCheckEnhanced6Result:
    """Enhanced v6 surface check result.

    Attributes
    ----------
    Inherits all from v5, plus:
    differential : DifferentialReport, optional
    repair_priorities : list[RepairPriority]
    report_text : str, optional
        OpenFOAM-formatted quality report.
    n_prioritised_repairs : int
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
    differential: Optional[DifferentialReport] = None
    repair_priorities: list = field(default_factory=list)
    report_text: Optional[str] = None
    n_prioritised_repairs: int = 0

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v6): {self.n_points} points, "
            f"{self.n_faces} faces",
            f"  Overall grade: {self.overall_grade}",
            f"  Watertight: {self.is_watertight}",
            f"  Repairs applied: {self.n_repairs_applied}",
            f"  Prioritised repairs: {self.n_prioritised_repairs}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_6(
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
    previous_result: Optional[object] = None,
    generate_report: bool = False,
    prioritize_repairs: bool = False,
) -> SurfaceCheckEnhanced6Result:
    """Check surface quality with differential checking and reporting.

    Parameters
    ----------
    surface_path, vertices, faces, normals,
    duplicate_tol, area_tol,
    check_self_intersection, quality_thresholds,
    auto_repair, batch_inputs
        Forwarded to v5 check.
    previous_result : SurfaceCheckEnhanced5Result, optional
        Previous check result for differential comparison.
    generate_report : bool
        Generate an OpenFOAM-formatted quality report.
    prioritize_repairs : bool
        Rank repair actions by impact.

    Returns
    -------
    SurfaceCheckEnhanced6Result
    """
    from pyfoam.tools.surface_check_enhanced_5 import surface_check_enhanced_5

    v5_result = surface_check_enhanced_5(
        surface_path=surface_path,
        vertices=vertices,
        faces=faces,
        normals=normals,
        duplicate_tol=duplicate_tol,
        area_tol=area_tol,
        check_self_intersection=check_self_intersection,
        quality_thresholds=quality_thresholds,
        auto_repair=auto_repair,
        batch_inputs=batch_inputs,
    )

    # Differential checking
    diff = None
    if previous_result is not None:
        diff = _compute_differential(previous_result, v5_result)

    # Repair prioritisation
    priorities = []
    if prioritize_repairs:
        priorities = _prioritise_repairs(v5_result)

    # Report generation
    report = None
    if generate_report:
        report = _generate_report(v5_result)

    return SurfaceCheckEnhanced6Result(
        n_points=v5_result.n_points,
        n_faces=v5_result.n_faces,
        n_edges=v5_result.n_edges,
        n_open_edges=v5_result.n_open_edges,
        n_non_manifold_edges=v5_result.n_non_manifold_edges,
        n_duplicate_points=v5_result.n_duplicate_points,
        n_degenerate_faces=v5_result.n_degenerate_faces,
        is_watertight=v5_result.is_watertight,
        min_face_area=v5_result.min_face_area,
        max_face_area=v5_result.max_face_area,
        total_area=v5_result.total_area,
        mean_aspect_ratio=v5_result.mean_aspect_ratio,
        max_aspect_ratio=v5_result.max_aspect_ratio,
        euler_characteristic=v5_result.euler_characteristic,
        n_connected_components=v5_result.n_connected_components,
        face_grades=v5_result.face_grades,
        warnings=list(v5_result.warnings),
        n_self_intersections=v5_result.n_self_intersections,
        overall_grade=v5_result.overall_grade,
        repair_results=v5_result.repair_results,
        n_repairs_applied=v5_result.n_repairs_applied,
        repaired_vertices=v5_result.repaired_vertices,
        repaired_faces=v5_result.repaired_faces,
        differential=diff,
        repair_priorities=priorities,
        report_text=report,
        n_prioritised_repairs=len(priorities),
    )


# ---------------------------------------------------------------------------
# Differential checking
# ---------------------------------------------------------------------------


def _compute_differential(prev, curr):
    """Compare two check results."""
    return DifferentialReport(
        n_new_degenerates=max(0, curr.n_degenerate_faces - prev.n_degenerate_faces),
        n_new_open_edges=max(0, curr.n_open_edges - prev.n_open_edges),
        watertightness_changed=prev.is_watertight != curr.is_watertight,
        grade_changed=prev.overall_grade != curr.overall_grade,
        previous_grade=prev.overall_grade,
        current_grade=curr.overall_grade,
    )


# ---------------------------------------------------------------------------
# Repair prioritisation
# ---------------------------------------------------------------------------


def _prioritise_repairs(result):
    """Rank repair actions by impact score."""
    priorities = []
    if result.n_degenerate_faces > 0:
        priorities.append(RepairPriority(
            action_type="collapse_edge",
            priority=1,
            impact_score=result.n_degenerate_faces / max(result.n_faces, 1),
            description=f"Fix {result.n_degenerate_faces} degenerate faces",
        ))
    if result.n_duplicate_points > 0:
        priorities.append(RepairPriority(
            action_type="merge_points",
            priority=2,
            impact_score=result.n_duplicate_points / max(result.n_points, 1),
            description=f"Merge {result.n_duplicate_points} duplicate points",
        ))
    if result.n_open_edges > 0:
        priorities.append(RepairPriority(
            action_type="close_surface",
            priority=3,
            impact_score=result.n_open_edges / max(result.n_edges, 1),
            description=f"Close {result.n_open_edges} open edges",
        ))
    priorities.sort(key=lambda p: p.priority)
    return priorities


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(result):
    """Generate OpenFOAM-formatted quality report."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      surfaceCheckReport;",
        "}",
        "",
        f"nPoints        {result.n_points};",
        f"nFaces         {result.n_faces};",
        f"nEdges         {result.n_edges};",
        f"overallGrade   \"{result.overall_grade}\";",
        f"isWatertight   {str(result.is_watertight).lower()};",
        f"nOpenEdges     {result.n_open_edges};",
        f"nDegenerates   {result.n_degenerate_faces};",
        f"nDuplicates    {result.n_duplicate_points};",
        f"totalArea      {result.total_area:.6g};",
        f"meanAR         {result.mean_aspect_ratio:.4f};",
    ]
    if result.warnings:
        lines.append("")
        lines.append("warnings")
        lines.append("(")
        for w in result.warnings:
            lines.append(f"    \"{w}\"")
        lines.append(")")
    return "\n".join(lines)
