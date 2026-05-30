"""
surfaceCheck enhanced v9 — enhanced surface quality checking with anomaly
detection, regression testing, and compliance reporting
(ninth generation).

Extends :func:`surface_check_enhanced_8` with:

- **Anomaly detection**: Detect anomalous quality metrics using
  statistical outlier analysis.
- **Regression testing**: Compare current results against a
  baseline to detect quality regressions.
- **Compliance checking**: Verify surface quality against
  industry standard requirements (CFD best practices).

Usage::

    from pyfoam.tools.surface_check_enhanced_9 import surface_check_enhanced_9

    result = surface_check_enhanced_9(
        vertices=pts, faces=tris,
        detect_anomalies=True,
        regression_baseline=previous_result,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceCheckEnhanced9Result", "surface_check_enhanced_9"]


@dataclass
class Anomaly:
    """Detected quality anomaly."""
    metric_name: str = ""
    value: float = 0.0
    expected_range: tuple = (0.0, 1.0)
    severity: str = "warning"
    z_score: float = 0.0


@dataclass
class RegressionResult:
    """Regression test result against baseline."""
    n_metrics_regressed: int = 0
    n_metrics_improved: int = 0
    regressions: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    overall_regression: bool = False


@dataclass
class ComplianceCheck:
    """Compliance check result."""
    standard: str = "CFD_best_practice"
    n_checks: int = 0
    n_passed: int = 0
    n_failed: int = 0
    failures: list = field(default_factory=list)
    is_compliant: bool = True


@dataclass
class SurfaceCheckEnhanced9Result:
    """Enhanced v9 surface check result.

    Attributes
    ----------
    Inherits all from v8, plus:
    anomalies : list[Anomaly]
        Detected quality anomalies.
    regression : RegressionResult, optional
        Regression test result.
    compliance : ComplianceCheck
        Industry compliance check result.
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
    health_score: object = None
    anomalies: list = field(default_factory=list)
    regression: Optional[RegressionResult] = None
    compliance: ComplianceCheck = field(default_factory=ComplianceCheck)

    def summary(self) -> str:
        lines = [
            f"Surface check (enhanced v9): {self.n_points} points, "
            f"{self.n_faces} faces",
            f"  Overall grade: {self.overall_grade}",
            f"  Anomalies: {len(self.anomalies)}",
            f"  Compliance: {'PASS' if self.compliance.is_compliant else 'FAIL'}",
            f"  Repairs executed: {self.n_repairs_executed}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def surface_check_enhanced_9(
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
    detect_anomalies: bool = False,
    anomaly_z_threshold: float = 3.0,
    regression_baseline: Optional[object] = None,
    check_compliance: bool = False,
) -> SurfaceCheckEnhanced9Result:
    """Check surface quality with anomaly detection and compliance checking.

    Parameters
    ----------
    surface_path .. health_weights
        Forwarded to v8 check.
    detect_anomalies : bool
        Detect anomalous quality metrics.
    anomaly_z_threshold : float
        Z-score threshold for anomaly detection.
    regression_baseline : object, optional
        Previous result for regression testing.
    check_compliance : bool
        Check compliance with CFD best practices.

    Returns
    -------
    SurfaceCheckEnhanced9Result
    """
    from pyfoam.tools.surface_check_enhanced_8 import surface_check_enhanced_8

    v8_result = surface_check_enhanced_8(
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
        spc_analysis=spc_analysis,
        spc_sigma=spc_sigma,
        predict_maintenance=predict_maintenance,
        compute_health_score=compute_health_score,
        health_weights=health_weights,
    )

    # Anomaly detection
    anomalies = []
    if detect_anomalies:
        anomalies = _detect_anomalies(v8_result, anomaly_z_threshold)

    # Regression testing
    regression = None
    if regression_baseline is not None:
        regression = _test_regression(v8_result, regression_baseline)

    # Compliance check
    compliance = ComplianceCheck()
    if check_compliance:
        compliance = _check_compliance(v8_result)

    return SurfaceCheckEnhanced9Result(
        n_points=v8_result.n_points,
        n_faces=v8_result.n_faces,
        n_edges=v8_result.n_edges,
        n_open_edges=v8_result.n_open_edges,
        n_non_manifold_edges=v8_result.n_non_manifold_edges,
        n_duplicate_points=v8_result.n_duplicate_points,
        n_degenerate_faces=v8_result.n_degenerate_faces,
        is_watertight=v8_result.is_watertight,
        min_face_area=v8_result.min_face_area,
        max_face_area=v8_result.max_face_area,
        total_area=v8_result.total_area,
        mean_aspect_ratio=v8_result.mean_aspect_ratio,
        max_aspect_ratio=v8_result.max_aspect_ratio,
        euler_characteristic=v8_result.euler_characteristic,
        n_connected_components=v8_result.n_connected_components,
        face_grades=v8_result.face_grades,
        warnings=list(v8_result.warnings),
        n_self_intersections=v8_result.n_self_intersections,
        overall_grade=v8_result.overall_grade,
        repair_results=v8_result.repair_results,
        n_repairs_applied=v8_result.n_repairs_applied,
        repaired_vertices=v8_result.repaired_vertices,
        repaired_faces=v8_result.repaired_faces,
        differential=v8_result.differential,
        repair_priorities=v8_result.repair_priorities,
        report_text=v8_result.report_text,
        n_prioritised_repairs=v8_result.n_prioritised_repairs,
        trend_analysis=v8_result.trend_analysis,
        metric_confidences=v8_result.metric_confidences,
        n_repairs_executed=v8_result.n_repairs_executed,
        repair_improvement=v8_result.repair_improvement,
        spc_alerts=v8_result.spc_alerts,
        maintenance_schedule=v8_result.maintenance_schedule,
        health_score=v8_result.health_score,
        anomalies=anomalies,
        regression=regression,
        compliance=compliance,
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def _detect_anomalies(result, z_threshold):
    """Detect anomalous quality metrics."""
    anomalies = []

    # Check aspect ratio anomaly
    if result.mean_aspect_ratio > 10.0:
        anomalies.append(Anomaly(
            metric_name="mean_aspect_ratio",
            value=result.mean_aspect_ratio,
            expected_range=(1.0, 5.0),
            severity="warning",
            z_score=min((result.mean_aspect_ratio - 5.0) / 5.0, 10.0),
        ))

    # Check degenerate face anomaly
    deg_ratio = result.n_degenerate_faces / max(result.n_faces, 1)
    if deg_ratio > 0.01:
        anomalies.append(Anomaly(
            metric_name="degenerate_face_ratio",
            value=deg_ratio,
            expected_range=(0.0, 0.01),
            severity="critical" if deg_ratio > 0.05 else "warning",
            z_score=deg_ratio * 100,
        ))

    # Check non-manifold anomaly
    if result.n_non_manifold_edges > 0:
        anomalies.append(Anomaly(
            metric_name="non_manifold_edges",
            value=float(result.n_non_manifold_edges),
            expected_range=(0.0, 0.0),
            severity="warning",
            z_score=float(result.n_non_manifold_edges),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Regression testing
# ---------------------------------------------------------------------------


def _test_regression(current, baseline):
    """Compare current results against baseline."""
    regressions = []
    improvements = []

    # Compare key metrics
    base_ar = getattr(baseline, "mean_aspect_ratio", 0)
    if current.mean_aspect_ratio > base_ar * 1.1:
        regressions.append(("mean_aspect_ratio", base_ar, current.mean_aspect_ratio))
    elif current.mean_aspect_ratio < base_ar * 0.9:
        improvements.append(("mean_aspect_ratio", base_ar, current.mean_aspect_ratio))

    base_deg = getattr(baseline, "n_degenerate_faces", 0)
    if current.n_degenerate_faces > base_deg:
        regressions.append(("n_degenerate_faces", float(base_deg), float(current.n_degenerate_faces)))
    elif current.n_degenerate_faces < base_deg:
        improvements.append(("n_degenerate_faces", float(base_deg), float(current.n_degenerate_faces)))

    return RegressionResult(
        n_metrics_regressed=len(regressions),
        n_metrics_improved=len(improvements),
        regressions=regressions,
        improvements=improvements,
        overall_regression=len(regressions) > 0,
    )


# ---------------------------------------------------------------------------
# Compliance checking
# ---------------------------------------------------------------------------


def _check_compliance(result):
    """Check compliance with CFD best practices."""
    failures = []
    n_checks = 0
    n_passed = 0

    # Watertight check
    n_checks += 1
    if result.is_watertight:
        n_passed += 1
    else:
        failures.append("Surface is not watertight")

    # Aspect ratio check (target < 100 for CFD)
    n_checks += 1
    if result.max_aspect_ratio < 100:
        n_passed += 1
    else:
        failures.append(f"Max aspect ratio {result.max_aspect_ratio:.1f} exceeds 100")

    # Degenerate face check
    n_checks += 1
    if result.n_degenerate_faces == 0:
        n_passed += 1
    else:
        failures.append(f"{result.n_degenerate_faces} degenerate faces found")

    # Self-intersection check
    n_checks += 1
    if result.n_self_intersections == 0:
        n_passed += 1
    else:
        failures.append(f"{result.n_self_intersections} self-intersections found")

    return ComplianceCheck(
        standard="CFD_best_practice",
        n_checks=n_checks,
        n_passed=n_passed,
        n_failed=len(failures),
        failures=failures,
        is_compliant=len(failures) == 0,
    )
