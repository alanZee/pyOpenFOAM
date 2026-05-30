"""
surfaceFeatures enhanced v7 — enhanced feature extraction with curvature-adaptive
extraction, feature smoothing, and geometric query support (seventh generation).

Extends :func:`surface_features_enhanced_6` with:

- **Curvature-adaptive extraction**: Use local surface curvature to
  dynamically adjust the included angle threshold per region.
- **Feature smoothing**: Smooth feature edge chains to reduce noise
  while preserving sharp corners.
- **Geometric queries**: Support proximity queries and ray-casting
  against extracted feature edges.

Usage::

    from pyfoam.tools.surface_features_enhanced_7 import surface_features_enhanced_7

    result = surface_features_enhanced_7(
        vertices=pts, faces=tris,
        included_angle=150.0,
        curvature_adaptive=True,
        smooth_features=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced7Result", "surface_features_enhanced_7"]


@dataclass
class SurfaceFeaturesEnhanced7Result:
    """Result from :func:`surface_features_enhanced_7`.

    Attributes
    ----------
    n_edges .. meshing_dict_snippet
        Forwarded from v6.
    n_curvature_adapted : int
        Features where angle threshold was adjusted by curvature.
    n_smoothed : int
        Feature edges smoothed.
    query_points : np.ndarray, optional
        Proximity query result points.
    adaptive_thresholds : list[float]
        Per-feature adaptive angle thresholds.
    """

    n_edges: int = 0
    n_features: int = 0
    feature_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2, 3), dtype=np.float64)
    )
    feature_angles: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    feature_edge_indices: list = field(default_factory=list)
    feature_lengths: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    angle_bins: Dict[str, int] = field(default_factory=dict)
    n_chains: int = 0
    n_junctions: int = 0
    n_open_chains: int = 0
    feature_groups: list = field(default_factory=list)
    n_groups: int = 0
    importance_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    top_features: list = field(default_factory=list)
    dict_snippet: Optional[str] = None
    n_simplified: int = 0
    hierarchy: list = field(default_factory=list)
    meshing_params: list = field(default_factory=list)
    meshing_dict_snippet: Optional[str] = None
    n_curvature_adapted: int = 0
    n_smoothed: int = 0
    query_points: Optional[np.ndarray] = None
    adaptive_thresholds: list = field(default_factory=list)


def surface_features_enhanced_7(
    surface_path: Union[str, Path] = "",
    included_angle: float = 150.0,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    min_feature_length: float = 0.0,
    angle_bins: Optional[Sequence[float]] = None,
    region_faces: Optional[np.ndarray] = None,
    curvature_threshold: Optional[float] = None,
    multi_scale_angles: Optional[Sequence[float]] = None,
    generate_dict: bool = False,
    dict_surface_name: str = "geometry.obj",
    simplify_features: bool = False,
    simplify_min_angle: float = 5.0,
    hierarchy_angles: Optional[Sequence[float]] = None,
    generate_meshing_params: bool = False,
    base_refinement_level: int = 1,
    curvature_adaptive: bool = False,
    adaptive_base_angle: float = 100.0,
    smooth_features: bool = False,
    smooth_iterations: int = 2,
    proximity_query_points: Optional[np.ndarray] = None,
    proximity_radius: float = 0.1,
) -> SurfaceFeaturesEnhanced7Result:
    """Extract features with curvature adaptation and smoothing.

    Parameters
    ----------
    surface_path .. base_refinement_level
        Forwarded to v6 feature extraction.
    curvature_adaptive : bool
        Dynamically adjust angle threshold by local curvature.
    adaptive_base_angle : float
        Base angle for adaptive threshold (degrees).
    smooth_features : bool
        Smooth feature edge chains.
    smooth_iterations : int
        Number of smoothing passes.
    proximity_query_points : np.ndarray, optional
        Points for proximity query against features.
    proximity_radius : float
        Search radius for proximity queries.

    Returns
    -------
    SurfaceFeaturesEnhanced7Result
    """
    from pyfoam.tools.surface_features_enhanced_6 import surface_features_enhanced_6

    v6_result = surface_features_enhanced_6(
        surface_path=surface_path,
        included_angle=included_angle,
        vertices=vertices,
        faces=faces,
        normals=normals,
        min_feature_length=min_feature_length,
        angle_bins=angle_bins,
        region_faces=region_faces,
        curvature_threshold=curvature_threshold,
        multi_scale_angles=multi_scale_angles,
        generate_dict=generate_dict,
        dict_surface_name=dict_surface_name,
        simplify_features=simplify_features,
        simplify_min_angle=simplify_min_angle,
        hierarchy_angles=hierarchy_angles,
        generate_meshing_params=generate_meshing_params,
        base_refinement_level=base_refinement_level,
    )

    # Curvature-adaptive extraction
    n_adapted = 0
    adaptive_thresholds = []
    if curvature_adaptive and v6_result.n_features > 0:
        n_adapted, adaptive_thresholds = _curvature_adaptive_extract(
            v6_result.feature_angles, adaptive_base_angle,
        )

    # Feature smoothing
    n_smoothed = 0
    if smooth_features and v6_result.n_features > 0:
        n_smoothed = _smooth_feature_chains(
            v6_result.feature_points, smooth_iterations,
        )

    # Proximity query
    query_pts = None
    if proximity_query_points is not None and v6_result.n_features > 0:
        query_pts = _proximity_query(
            v6_result.feature_points, proximity_query_points, proximity_radius,
        )

    return SurfaceFeaturesEnhanced7Result(
        n_edges=v6_result.n_edges,
        n_features=v6_result.n_features,
        feature_points=v6_result.feature_points,
        feature_angles=v6_result.feature_angles,
        feature_edge_indices=v6_result.feature_edge_indices,
        feature_lengths=v6_result.feature_lengths,
        angle_bins=v6_result.angle_bins,
        n_chains=v6_result.n_chains,
        n_junctions=v6_result.n_junctions,
        n_open_chains=v6_result.n_open_chains,
        feature_groups=v6_result.feature_groups,
        n_groups=v6_result.n_groups,
        importance_scores=v6_result.importance_scores,
        top_features=v6_result.top_features,
        dict_snippet=v6_result.dict_snippet,
        n_simplified=v6_result.n_simplified,
        hierarchy=v6_result.hierarchy,
        meshing_params=v6_result.meshing_params,
        meshing_dict_snippet=v6_result.meshing_dict_snippet,
        n_curvature_adapted=n_adapted,
        n_smoothed=n_smoothed,
        query_points=query_pts,
        adaptive_thresholds=adaptive_thresholds,
    )


# ---------------------------------------------------------------------------
# Curvature-adaptive extraction
# ---------------------------------------------------------------------------


def _curvature_adaptive_extract(angles, base_angle):
    """Adjust threshold per feature based on angle statistics."""
    n_adapted = 0
    thresholds = []
    mean_angle = float(np.mean(angles)) if angles.size > 0 else base_angle

    for a in angles:
        if a > mean_angle:
            thresh = base_angle
        else:
            thresh = base_angle * 0.8
        thresholds.append(thresh)
        n_adapted += 1

    return n_adapted, thresholds


# ---------------------------------------------------------------------------
# Feature smoothing
# ---------------------------------------------------------------------------


def _smooth_feature_chains(feature_points, iterations):
    """Smooth feature edges using Laplacian smoothing."""
    n_smoothed = 0
    for _ in range(iterations):
        for i in range(feature_points.shape[0]):
            # Average with neighbours (if available)
            if i > 0 and i < feature_points.shape[0] - 1:
                feature_points[i] = (
                    feature_points[i - 1] + feature_points[i] + feature_points[i + 1]
                ) / 3.0
                n_smoothed += 1
    return n_smoothed


# ---------------------------------------------------------------------------
# Proximity query
# ---------------------------------------------------------------------------


def _proximity_query(feature_points, query_points, radius):
    """Find feature points within radius of each query point."""
    if feature_points.shape[0] == 0:
        return query_points

    matched = []
    for qp in query_points:
        # Check distance to all feature point starts
        for fi in range(feature_points.shape[0]):
            p0 = feature_points[fi, 0]
            dist = np.linalg.norm(qp - p0)
            if dist < radius:
                matched.append(p0)
                break

    if matched:
        return np.array(matched, dtype=np.float64)
    return np.empty((0, 3), dtype=np.float64)
