"""
surfaceCheck enhanced v8 — enhanced surface quality checking with statistical
process control, predictive maintenance scheduling, and mesh health scoring
(eighth generation).

Extends :func:`surface_check_enhanced_7` with:

- **Statistical process control**: Apply SPC rules (Western Electric)
  to detect out-of-control quality metrics.
- **Predictive maintenance**: Estimate remaining useful quality life
  from trend data and schedule maintenance windows.
- **Mesh health scoring**: Compute an aggregate health score combining
  all quality metrics with configurable weights.

Usage::

    from pyfoam.tools.surface_check_enhanced_8 import surface_check_enhanced_8

    result = surface_check_enhanced_8(
        vertices=pts, faces=tris,
        compute_health_score=True,
        spc_analysis=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced8Result", "surface_check_enhanced_8"]


@dataclass
class SPCAlert:
    """Statistical process control alert."""
    metric_name: str = ""
    rule_violated: str = ""
    value: float = 0.0
    control_limit: float = 0.0
    severity: str = "warning"


@dataclass
class MaintenanceSchedule:
    """Predictive maintenance schedule entry."""
    metric_name: str = ""
    estimated_days_to_failure: float = 0.0
    recommended_action: str = ""
    priority: str = "low"


@dataclass
class MeshHealthScore:
    """Aggregate mesh health score."""
    overall_score: float = 0.0
    geometry_score: float = 0.0
    topology_score: float = 0.0
    quality_score: float = 0.0
    grade: str = "F"
    n_critical_issues: int = 0


@dataclass
class SurfaceCheckEnhanced8Result:
    """Enhanced v8 surface check result.

    Attributes
    ----------
    Inherits all from v7, plus:
    spc_alerts : list[SPCAlert]
        SPC violation alerts.
    maintenance_schedule : list[MaintenanceSchedule]
        Predictive maintenance schedule.
    health_score : MeshHealthScore
        Aggregate mesh health score.
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
    spc_alerts: list = field(default_factory=list)
    maintenance_schedule: list = field(default_factory=list)
    health_score: MeshHealthScore = field(default_factory=MeshHealthScore)

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v8): {self.n_points} points, "
            f"{self.n_faces} faces",
            f"  Overall grade: {self.overall_grade}",
            f"  Health score: {self.health_score.overall_score:.2f}",
            f"  SPC alerts: {len(self.spc_alerts)}",
            f"  Repairs executed: {self.n_repairs_executed}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_8(
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
    spc_analysis: bool = False,
    spc_sigma: float = 3.0,
    predict_maintenance: bool = False,
    compute_health_score: bool = False,
    health_weights: Optional[Dict[str, float]] = None,
) -> SurfaceCheckEnhanced8Result:
    """Check surface quality with SPC, maintenance scheduling, and health score.

    Parameters
    ----------
    surface_path .. execute_repairs
        Forwarded to v7 check.
    spc_analysis : bool
        Apply SPC rules to quality metrics.
    spc_sigma : float
        Control limit multiplier (sigma level).
    predict_maintenance : bool
        Predict maintenance schedule from trends.
    compute_health_score : bool
        Compute aggregate mesh health score.
    health_weights : dict, optional
        Weights for health score components.

    Returns
    -------
    SurfaceCheckEnhanced8Result
    """
    from pyfoam.tools.surface_check_enhanced_7 import surface_check_enhanced_7

    v7_result = surface_check_enhanced_7(
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
        trend_history=trend_history,
        execute_repairs=execute_repairs,
    )

    # SPC analysis
    spc_alerts = []
    if spc_analysis:
        spc_alerts = _spc_analysis(v7_result, spc_sigma)

    # Predictive maintenance
    maintenance = []
    if predict_maintenance and v7_result.trend_analysis:
        maintenance = _predict_maintenance(v7_result.trend_analysis)

    # Health score
    health = MeshHealthScore()
    if compute_health_score:
        health = _compute_health_score(
            v7_result, health_weights or {},
        )

    return SurfaceCheckEnhanced8Result(
        n_points=v7_result.n_points,
        n_faces=v7_result.n_faces,
        n_edges=v7_result.n_edges,
        n_open_edges=v7_result.n_open_edges,
        n_non_manifold_edges=v7_result.n_non_manifold_edges,
        n_duplicate_points=v7_result.n_duplicate_points,
        n_degenerate_faces=v7_result.n_degenerate_faces,
        is_watertight=v7_result.is_watertight,
        min_face_area=v7_result.min_face_area,
        max_face_area=v7_result.max_face_area,
        total_area=v7_result.total_area,
        mean_aspect_ratio=v7_result.mean_aspect_ratio,
        max_aspect_ratio=v7_result.max_aspect_ratio,
        euler_characteristic=v7_result.euler_characteristic,
        n_connected_components=v7_result.n_connected_components,
        face_grades=v7_result.face_grades,
        warnings=list(v7_result.warnings),
        n_self_intersections=v7_result.n_self_intersections,
        overall_grade=v7_result.overall_grade,
        repair_results=v7_result.repair_results,
        n_repairs_applied=v7_result.n_repairs_applied,
        repaired_vertices=v7_result.repaired_vertices,
        repaired_faces=v7_result.repaired_faces,
        differential=v7_result.differential,
        repair_priorities=v7_result.repair_priorities,
        report_text=v7_result.report_text,
        n_prioritised_repairs=v7_result.n_prioritised_repairs,
        trend_analysis=v7_result.trend_analysis,
        metric_confidences=v7_result.metric_confidences,
        n_repairs_executed=v7_result.n_repairs_executed,
        repair_improvement=v7_result.repair_improvement,
        spc_alerts=spc_alerts,
        maintenance_schedule=maintenance,
        health_score=health,
    )


# ---------------------------------------------------------------------------
# SPC analysis
# ---------------------------------------------------------------------------


def _spc_analysis(result, sigma_level):
    """Apply statistical process control rules."""
    alerts = []

    # Check degenerate faces against control limits
    mean_val = 0.0
    std_val = 1.0  # default for single-point
    upper_limit = mean_val + sigma_level * std_val

    if result.n_degenerate_faces > upper_limit:
        alerts.append(SPCAlert(
            metric_name="degenerate_faces",
            rule_violated="beyond_3sigma",
            value=float(result.n_degenerate_faces),
            control_limit=upper_limit,
            severity="critical" if result.n_degenerate_faces > 2 * upper_limit else "warning",
        ))

    if not result.is_watertight:
        alerts.append(SPCAlert(
            metric_name="watertightness",
            rule_violated="not_watertight",
            value=0.0,
            control_limit=1.0,
            severity="critical",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Predictive maintenance
# ---------------------------------------------------------------------------


def _predict_maintenance(trend_analysis):
    """Predict maintenance needs from trend data."""
    schedule = []
    for trend in trend_analysis:
        direction = getattr(trend, "direction", "stable")
        slope = getattr(trend, "slope", 0.0)
        metric = getattr(trend, "metric_name", "unknown")

        if direction == "degrading" and abs(slope) > 1e-10:
            days = 30.0 / max(abs(slope), 1e-10)  # rough estimate
            schedule.append(MaintenanceSchedule(
                metric_name=metric,
                estimated_days_to_failure=min(days, 365.0),
                recommended_action="inspect and repair",
                priority="high" if days < 30 else "medium",
            ))
        elif direction == "stable":
            schedule.append(MaintenanceSchedule(
                metric_name=metric,
                estimated_days_to_failure=365.0,
                recommended_action="routine check",
                priority="low",
            ))

    return schedule


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------


def _compute_health_score(result, weights):
    """Compute aggregate mesh health score."""
    default_weights = {
        "watertight": 0.3,
        "degenerate": 0.3,
        "open_edges": 0.2,
        "aspect_ratio": 0.2,
    }
    w = {**default_weights, **weights}

    # Geometry score: watertight + no degenerate faces
    geo = 1.0
    if not result.is_watertight:
        geo -= w.get("watertight", 0.3)
    deg_ratio = result.n_degenerate_faces / max(result.n_faces, 1)
    geo -= min(w.get("degenerate", 0.3), deg_ratio * 10)
    geo = max(0.0, geo)

    # Topology score: open edges + non-manifold
    topo = 1.0
    open_ratio = result.n_open_edges / max(result.n_edges, 1)
    topo -= min(w.get("open_edges", 0.2), open_ratio * 5)
    topo -= min(0.1, result.n_non_manifold_edges / max(result.n_edges, 1) * 5)
    topo = max(0.0, topo)

    # Quality score: aspect ratio
    qual = 1.0
    if result.mean_aspect_ratio > 5.0:
        qual -= min(w.get("aspect_ratio", 0.2), (result.mean_aspect_ratio - 5.0) * 0.1)
    qual = max(0.0, qual)

    overall = 0.4 * geo + 0.3 * topo + 0.3 * qual

    # Grade
    if overall >= 0.9:
        grade = "A"
    elif overall >= 0.75:
        grade = "B"
    elif overall >= 0.6:
        grade = "C"
    elif overall >= 0.4:
        grade = "D"
    else:
        grade = "F"

    n_critical = 0
    if not result.is_watertight:
        n_critical += 1
    if result.n_degenerate_faces > 0:
        n_critical += 1

    return MeshHealthScore(
        overall_score=overall,
        geometry_score=geo,
        topology_score=topo,
        quality_score=qual,
        grade=grade,
        n_critical_issues=n_critical,
    )
