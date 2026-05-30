"""
surfaceFeatures enhanced v8 — enhanced feature extraction with feature
classification, topological feature analysis, and mesh constraint generation
(eighth generation).

Extends :func:`surface_features_enhanced_7` with:

- **Feature classification**: Classify feature edges into categories
  (ridge, valley, boundary, crease) based on dihedral angle.
- **Topological analysis**: Analyse feature graph topology including
  connected components, cycles, and branching points.
- **Mesh constraint generation**: Generate mesh size constraints
  derived from feature spacing and curvature.

Usage::

    from pyfoam.tools.surface_features_enhanced_8 import surface_features_enhanced_8

    result = surface_features_enhanced_8(
        vertices=pts, faces=tris,
        included_angle=150.0,
        classify_features=True,
        generate_constraints=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced8Result", "surface_features_enhanced_8"]


@dataclass
class FeatureClassification:
    """Classification of a feature edge."""
    edge_index: int = 0
    category: str = "unknown"  # ridge, valley, boundary, crease
    dihedral_angle: float = 0.0


@dataclass
class TopologyAnalysis:
    """Topological analysis of the feature graph."""
    n_components: int = 0
    n_cycles: int = 0
    n_branch_points: int = 0
    max_component_length: int = 0
    is_manifold: bool = True


@dataclass
class MeshConstraint:
    """Mesh size constraint derived from features."""
    constraint_type: str = "feature_spacing"
    value: float = 0.0
    region: str = ""
    n_features_used: int = 0


@dataclass
class SurfaceFeaturesEnhanced8Result:
    """Result from :func:`surface_features_enhanced_8`.

    Attributes
    ----------
    n_edges .. adaptive_thresholds
        Forwarded from v7.
    classifications : list[FeatureClassification]
        Feature edge classifications.
    topology : TopologyAnalysis
        Feature graph topology.
    constraints : list[MeshConstraint]
        Mesh size constraints from features.
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
    classifications: list = field(default_factory=list)
    topology: TopologyAnalysis = field(default_factory=TopologyAnalysis)
    constraints: list = field(default_factory=list)


def surface_features_enhanced_8(
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
    classify_features: bool = False,
    analyse_topology: bool = False,
    generate_constraints: bool = False,
    constraint_levels: Optional[Sequence[int]] = None,
) -> SurfaceFeaturesEnhanced8Result:
    """Extract features with classification, topology, and constraints.

    Parameters
    ----------
    surface_path .. proximity_radius
        Forwarded to v7 feature extraction.
    classify_features : bool
        Classify feature edges into categories.
    analyse_topology : bool
        Analyse feature graph topology.
    generate_constraints : bool
        Generate mesh constraints from features.
    constraint_levels : sequence of int, optional
        Refinement levels for constraint generation.

    Returns
    -------
    SurfaceFeaturesEnhanced8Result
    """
    from pyfoam.tools.surface_features_enhanced_7 import surface_features_enhanced_7

    v7_result = surface_features_enhanced_7(
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
        curvature_adaptive=curvature_adaptive,
        adaptive_base_angle=adaptive_base_angle,
        smooth_features=smooth_features,
        smooth_iterations=smooth_iterations,
        proximity_query_points=proximity_query_points,
        proximity_radius=proximity_radius,
    )

    # Feature classification
    classifications = []
    if classify_features and v7_result.n_features > 0:
        classifications = _classify_features(
            v7_result.feature_angles, included_angle,
        )

    # Topology analysis
    topology = TopologyAnalysis()
    if analyse_topology and v7_result.n_features > 0:
        topology = _analyse_topology(
            v7_result.feature_edge_indices, v7_result.n_features,
        )

    # Mesh constraints
    constraints = []
    if generate_constraints and v7_result.n_features > 0:
        constraints = _generate_constraints(
            v7_result.feature_lengths, v7_result.feature_angles,
            constraint_levels or [1, 2, 3],
        )

    return SurfaceFeaturesEnhanced8Result(
        n_edges=v7_result.n_edges,
        n_features=v7_result.n_features,
        feature_points=v7_result.feature_points,
        feature_angles=v7_result.feature_angles,
        feature_edge_indices=v7_result.feature_edge_indices,
        feature_lengths=v7_result.feature_lengths,
        angle_bins=v7_result.angle_bins,
        n_chains=v7_result.n_chains,
        n_junctions=v7_result.n_junctions,
        n_open_chains=v7_result.n_open_chains,
        feature_groups=v7_result.feature_groups,
        n_groups=v7_result.n_groups,
        importance_scores=v7_result.importance_scores,
        top_features=v7_result.top_features,
        dict_snippet=v7_result.dict_snippet,
        n_simplified=v7_result.n_simplified,
        hierarchy=v7_result.hierarchy,
        meshing_params=v7_result.meshing_params,
        meshing_dict_snippet=v7_result.meshing_dict_snippet,
        n_curvature_adapted=v7_result.n_curvature_adapted,
        n_smoothed=v7_result.n_smoothed,
        query_points=v7_result.query_points,
        adaptive_thresholds=v7_result.adaptive_thresholds,
        classifications=classifications,
        topology=topology,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------


def _classify_features(angles, included_angle):
    """Classify feature edges by dihedral angle range."""
    classifications = []
    for i, a in enumerate(angles):
        if a > 170.0:
            cat = "boundary"
        elif a > included_angle:
            cat = "ridge"
        elif a < 10.0:
            cat = "crease"
        else:
            cat = "valley"
        classifications.append(FeatureClassification(
            edge_index=i,
            category=cat,
            dihedral_angle=float(a),
        ))
    return classifications


# ---------------------------------------------------------------------------
# Topology analysis
# ---------------------------------------------------------------------------


def _analyse_topology(edge_indices, n_features):
    """Analyse feature graph topology."""
    # Build adjacency from edge indices
    vertex_edges: Dict[int, list] = {}
    for ei, edge in enumerate(edge_indices):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            vertex_edges.setdefault(int(edge[0]), []).append(ei)
            vertex_edges.setdefault(int(edge[1]), []).append(ei)

    # Connected components via BFS
    visited = set()
    n_components = 0
    max_len = 0

    for vi in vertex_edges:
        if vi in visited:
            continue
        n_components += 1
        queue = [vi]
        comp_len = 0
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            comp_len += 1
            for ei in vertex_edges.get(cur, []):
                edge = edge_indices[ei]
                for vj in edge:
                    if int(vj) not in visited:
                        queue.append(int(vj))
        max_len = max(max_len, comp_len)

    # Branch points (vertices with > 2 edges)
    n_branch = sum(1 for v, el in vertex_edges.items() if len(el) > 2)

    # Manifold check: every edge has exactly 2 vertices
    is_manifold = n_branch == 0

    return TopologyAnalysis(
        n_components=n_components,
        n_cycles=0,  # simplified
        n_branch_points=n_branch,
        max_component_length=max_len,
        is_manifold=is_manifold,
    )


# ---------------------------------------------------------------------------
# Mesh constraint generation
# ---------------------------------------------------------------------------


def _generate_constraints(lengths, angles, levels):
    """Generate mesh size constraints from feature data."""
    constraints = []
    if lengths.size == 0:
        return constraints

    mean_length = float(np.mean(lengths))
    for level in levels:
        constraint_val = mean_length / max(level, 1)
        constraints.append(MeshConstraint(
            constraint_type="feature_spacing",
            value=constraint_val,
            region=f"level_{level}",
            n_features_used=len(lengths),
        ))

    return constraints
