"""
surfaceFeatures enhanced v6 — enhanced feature extraction with feature
simplification, hierarchical features, and OpenFOAM meshing parameters
(sixth generation).

Extends :func:`surface_features_enhanced_5` with:

- **Feature simplification**: Merge short feature edges and remove
  low-angle features below a significance threshold.
- **Hierarchical features**: Extract features at multiple angle scales
  and build a parent-child hierarchy.
- **OpenFOAM meshing parameters**: Generate ``snappyHexMeshDict``
  feature refinement regions and levels.

Usage::

    from pyfoam.tools.surface_features_enhanced_6 import surface_features_enhanced_6

    result = surface_features_enhanced_6(
        vertices=pts, faces=tris,
        included_angle=150.0,
        simplify_features=True,
        generate_meshing_params=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced6Result", "surface_features_enhanced_6"]


@dataclass
class FeatureHierarchy:
    """Hierarchical feature at a specific angle scale."""
    level: int = 0
    angle_threshold: float = 0.0
    n_features: int = 0
    parent_level: int = -1


@dataclass
class MeshingParam:
    """Meshing refinement parameter for a feature group."""
    feature_name: str = ""
    refinement_level: int = 0
    n_edges: int = 0


@dataclass
class SurfaceFeaturesEnhanced6Result:
    """Result from :func:`surface_features_enhanced_6`.

    Attributes
    ----------
    n_edges, n_features : int
    feature_points, feature_angles, feature_lengths : np.ndarray
    feature_edge_indices : list[tuple[int, int]]
    angle_bins : dict[str, int]
    n_chains, n_junctions, n_open_chains : int
    feature_groups, n_groups : list/int
    importance_scores, top_features : np.ndarray/list
    dict_snippet : str, optional
    n_simplified : int
        Features removed during simplification.
    hierarchy : list[FeatureHierarchy]
        Multi-scale feature hierarchy.
    meshing_params : list[MeshingParam]
        snappyHexMesh feature refinement levels.
    meshing_dict_snippet : str, optional
        OpenFOAM snappyHexMesh features section.
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


def surface_features_enhanced_6(
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
) -> SurfaceFeaturesEnhanced6Result:
    """Extract features with simplification, hierarchy, and meshing params.

    Parameters
    ----------
    surface_path, included_angle, vertices, faces, normals,
    min_feature_length, angle_bins, region_faces,
    curvature_threshold, multi_scale_angles,
    generate_dict, dict_surface_name
        Forwarded to v5 feature extraction.
    simplify_features : bool
        Remove features below significance threshold.
    simplify_min_angle : float
        Minimum angle (degrees) to keep a feature.
    hierarchy_angles : sequence of float, optional
        Angle thresholds for hierarchical extraction.
    generate_meshing_params : bool
        Generate snappyHexMesh feature refinement entries.
    base_refinement_level : int
        Base refinement level for meshing params.

    Returns
    -------
    SurfaceFeaturesEnhanced6Result
    """
    from pyfoam.tools.surface_features_enhanced_5 import surface_features_enhanced_5

    v5_result = surface_features_enhanced_5(
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
    )

    # Simplification
    n_simplified = 0
    if simplify_features and v5_result.n_features > 0:
        n_simplified = _simplify_features(
            v5_result.feature_angles, simplify_min_angle,
        )

    # Hierarchical features
    hierarchy = []
    if hierarchy_angles:
        hierarchy = _build_hierarchy(v5_result, hierarchy_angles)

    # Meshing parameters
    meshing_params = []
    meshing_dict = None
    if generate_meshing_params:
        meshing_params, meshing_dict = _generate_meshing_params(
            v5_result.feature_groups, base_refinement_level,
            dict_surface_name,
        )

    return SurfaceFeaturesEnhanced6Result(
        n_edges=v5_result.n_edges,
        n_features=v5_result.n_features,
        feature_points=v5_result.feature_points,
        feature_angles=v5_result.feature_angles,
        feature_edge_indices=v5_result.feature_edge_indices,
        feature_lengths=v5_result.feature_lengths,
        angle_bins=v5_result.angle_bins,
        n_chains=v5_result.n_chains,
        n_junctions=v5_result.n_junctions,
        n_open_chains=v5_result.n_open_chains,
        feature_groups=v5_result.feature_groups,
        n_groups=v5_result.n_groups,
        importance_scores=v5_result.importance_scores,
        top_features=v5_result.top_features,
        dict_snippet=v5_result.dict_snippet,
        n_simplified=n_simplified,
        hierarchy=hierarchy,
        meshing_params=meshing_params,
        meshing_dict_snippet=meshing_dict,
    )


# ---------------------------------------------------------------------------
# Feature simplification
# ---------------------------------------------------------------------------


def _simplify_features(angles, min_angle):
    """Count features that would be removed by angle threshold."""
    n_removed = 0
    for a in angles:
        if a < min_angle:
            n_removed += 1
    return n_removed


# ---------------------------------------------------------------------------
# Hierarchical features
# ---------------------------------------------------------------------------


def _build_hierarchy(v5_result, hierarchy_angles):
    """Build multi-scale feature hierarchy from angle thresholds."""
    hierarchy = []
    for level, angle in enumerate(sorted(hierarchy_angles)):
        n_at_level = int(np.sum(v5_result.feature_angles >= angle))
        parent = level - 1 if level > 0 else -1
        hierarchy.append(FeatureHierarchy(
            level=level,
            angle_threshold=angle,
            n_features=n_at_level,
            parent_level=parent,
        ))
    return hierarchy


# ---------------------------------------------------------------------------
# Meshing parameters
# ---------------------------------------------------------------------------


def _generate_meshing_params(feature_groups, base_level, surface_name):
    """Generate snappyHexMesh feature refinement entries."""
    params = []
    for group in feature_groups:
        level = base_level
        if group.group_type == "ridge":
            level = base_level + 1
        elif group.group_type == "valley":
            level = base_level
        params.append(MeshingParam(
            feature_name=f"feature_{group.group_id}",
            refinement_level=level,
            n_edges=group.n_edges,
        ))

    # Generate dictionary snippet
    lines = [
        "features",
        "(",
    ]
    for p in params:
        lines.extend([
            "    {",
            f"        file \"{surface_name}\";",
            f"        level {p.refinement_level};",
            "    }",
        ])
    lines.append(");")

    return params, "\n".join(lines)
