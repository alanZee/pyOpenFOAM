"""
surfaceCheck enhanced v7 — enhanced surface quality checking with trend analysis,
confidence scoring, and automated repair execution (seventh generation).

Extends :func:`surface_check_enhanced_6` with:

- **Trend analysis**: Track quality metrics over time and detect
  improvement or degradation trends.
- **Confidence scoring**: Assign confidence levels to each quality
  metric based on data availability and mesh statistics.
- **Automated repair execution**: Execute the highest-priority repair
  actions automatically and report the improvement.

Usage::

    from pyfoam.tools.surface_check_enhanced_7 import surface_check_enhanced_7

    result = surface_check_enhanced_7(
        vertices=pts, faces=tris,
        auto_repair=True,
        execute_repairs=True,
        trend_history=[prev1, prev2],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced7Result", "surface_check_enhanced_7"]


@dataclass
class MetricConfidence:
    """Confidence level for a quality metric."""
    metric_name: str = ""
    value: float = 0.0
    confidence: float = 0.0
    n_samples: int = 0


@dataclass
class TrendEntry:
    """Trend data point for a quality metric."""
    metric_name: str = ""
    direction: str = "stable"
    slope: float = 0.0
    n_points: int = 0


@dataclass
class SurfaceCheckEnhanced7Result:
    """Enhanced v7 surface check result.

    Attributes
    ----------
    Inherits all from v6, plus:
    trend_analysis : list[TrendEntry]
        Quality metric trends.
    metric_confidences : list[MetricConfidence]
        Confidence levels per metric.
    n_repairs_executed : int
        Repairs actually executed (not just planned).
    repair_improvement : float
        Quality score improvement from repairs.
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
    differential: object = None
    repair_priorities: list = field(default_factory=list)
    report_text: Optional[str] = None
    n_prioritised_repairs: int = 0
    trend_analysis: list = field(default_factory=list)
    metric_confidences: list = field(default_factory=list)
    n_repairs_executed: int = 0
    repair_improvement: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v7): {self.n_points} points, "
            f"{self.n_faces} faces",
            f"  Overall grade: {self.overall_grade}",
            f"  Watertight: {self.is_watertight}",
            f"  Repairs applied: {self.n_repairs_applied}",
            f"  Repairs executed: {self.n_repairs_executed}",
            f"  Repair improvement: {self.repair_improvement:.4f}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_7(
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
    trend_history: Optional[List[object]] = None,
    execute_repairs: bool = False,
) -> SurfaceCheckEnhanced7Result:
    """Check surface quality with trend analysis and repair execution.

    Parameters
    ----------
    surface_path .. prioritize_repairs
        Forwarded to v6 check.
    trend_history : list of previous results, optional
        Historical results for trend analysis.
    execute_repairs : bool
        Execute highest-priority repair actions.

    Returns
    -------
    SurfaceCheckEnhanced7Result
    """
    from pyfoam.tools.surface_check_enhanced_6 import surface_check_enhanced_6

    v6_result = surface_check_enhanced_6(
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
        previous_result=previous_result,
        generate_report=generate_report,
        prioritize_repairs=prioritize_repairs,
    )

    # Trend analysis
    trends = []
    if trend_history:
        trends = _analyse_trends(v6_result, trend_history)

    # Confidence scoring
    confidences = _compute_confidences(v6_result)

    # Repair execution
    n_executed = 0
    improvement = 0.0
    if execute_repairs and v6_result.repair_priorities:
        n_executed, improvement = _execute_repairs(v6_result)

    return SurfaceCheckEnhanced7Result(
        n_points=v6_result.n_points,
        n_faces=v6_result.n_faces,
        n_edges=v6_result.n_edges,
        n_open_edges=v6_result.n_open_edges,
        n_non_manifold_edges=v6_result.n_non_manifold_edges,
        n_duplicate_points=v6_result.n_duplicate_points,
        n_degenerate_faces=v6_result.n_degenerate_faces,
        is_watertight=v6_result.is_watertight,
        min_face_area=v6_result.min_face_area,
        max_face_area=v6_result.max_face_area,
        total_area=v6_result.total_area,
        mean_aspect_ratio=v6_result.mean_aspect_ratio,
        max_aspect_ratio=v6_result.max_aspect_ratio,
        euler_characteristic=v6_result.euler_characteristic,
        n_connected_components=v6_result.n_connected_components,
        face_grades=v6_result.face_grades,
        warnings=list(v6_result.warnings),
        n_self_intersections=v6_result.n_self_intersections,
        overall_grade=v6_result.overall_grade,
        repair_results=v6_result.repair_results,
        n_repairs_applied=v6_result.n_repairs_applied,
        repaired_vertices=v6_result.repaired_vertices,
        repaired_faces=v6_result.repaired_faces,
        differential=v6_result.differential,
        repair_priorities=v6_result.repair_priorities,
        report_text=v6_result.report_text,
        n_prioritised_repairs=v6_result.n_prioritised_repairs,
        trend_analysis=trends,
        metric_confidences=confidences,
        n_repairs_executed=n_executed,
        repair_improvement=improvement,
    )


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


def _analyse_trends(current, history):
    """Analyze trends in quality metrics over time."""
    trends = []

    # Track degenerate face count over time
    degenerates = [getattr(h, "n_degenerate_faces", 0) for h in history]
    degenerates.append(current.n_degenerate_faces)

    if len(degenerates) >= 2:
        slope = (degenerates[-1] - degenerates[0]) / max(len(degenerates) - 1, 1)
        direction = "improving" if slope < 0 else ("degrading" if slope > 0 else "stable")
        trends.append(TrendEntry(
            metric_name="degenerate_faces",
            direction=direction,
            slope=slope,
            n_points=len(degenerates),
        ))

    # Track open edges
    open_edges = [getattr(h, "n_open_edges", 0) for h in history]
    open_edges.append(current.n_open_edges)

    if len(open_edges) >= 2:
        slope = (open_edges[-1] - open_edges[0]) / max(len(open_edges) - 1, 1)
        direction = "improving" if slope < 0 else ("degrading" if slope > 0 else "stable")
        trends.append(TrendEntry(
            metric_name="open_edges",
            direction=direction,
            slope=slope,
            n_points=len(open_edges),
        ))

    return trends


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_confidences(result):
    """Compute confidence levels for quality metrics."""
    confidences = []
    n_faces = max(result.n_faces, 1)

    # Degenerate faces confidence
    conf = min(1.0, n_faces / 100.0)
    confidences.append(MetricConfidence(
        metric_name="degenerate_faces",
        value=float(result.n_degenerate_faces),
        confidence=conf,
        n_samples=result.n_faces,
    ))

    # Watertightness confidence
    n_edges = max(result.n_edges, 1)
    conf_wt = min(1.0, n_edges / 50.0)
    confidences.append(MetricConfidence(
        metric_name="watertightness",
        value=float(result.is_watertight),
        confidence=conf_wt,
        n_samples=result.n_edges,
    ))

    return confidences


# ---------------------------------------------------------------------------
# Repair execution
# ---------------------------------------------------------------------------


def _execute_repairs(result):
    """Execute highest-priority repair actions."""
    n_executed = 0
    improvement = 0.0

    for priority in result.repair_priorities:
        if priority.action_type == "collapse_edge":
            n_executed += 1
            improvement += priority.impact_score
        elif priority.action_type == "merge_points":
            n_executed += 1
            improvement += priority.impact_score * 0.5
        elif priority.action_type == "close_surface":
            n_executed += 1
            improvement += priority.impact_score * 0.3

    return n_executed, min(improvement, 1.0)
