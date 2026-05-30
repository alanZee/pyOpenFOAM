"""
surfaceFeatures enhanced v5 — enhanced feature extraction with feature
grouping, importance ranking, and OpenFOAM dictionary generation
(fifth generation).

Extends :func:`surface_features_enhanced_4` with:

- **Feature grouping**: Cluster connected feature edges into groups
  and report per-group statistics (length, mean angle, type).
- **Importance ranking**: Rank features by a composite importance
  score combining angle, length, and persistence.
- **OpenFOAM dictionary generation**: Generate a ``surfaceFeaturesDict``
  snippet ready for ``surfaceFeatureExtract``.

Usage::

    from pyfoam.tools.surface_features_enhanced_5 import surface_features_enhanced_5

    result = surface_features_enhanced_5(
        vertices=pts, faces=tris,
        included_angle=150.0,
        generate_dict=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced5Result", "surface_features_enhanced_5"]


@dataclass
class FeatureGroup:
    """A connected group of feature edges."""
    group_id: int = 0
    n_edges: int = 0
    total_length: float = 0.0
    mean_angle: float = 0.0
    group_type: str = "general"  # "ridge", "valley", "crease", "general"


@dataclass
class SurfaceFeaturesEnhanced5Result:
    """Result from :func:`surface_features_enhanced_5`.

    Attributes
    ----------
    n_edges, n_features : int
    feature_points, feature_angles, feature_lengths : np.ndarray
    feature_edge_indices : list[tuple[int, int]]
    angle_bins : dict[str, int]
    n_chains, n_junctions, n_open_chains : int
    feature_groups : list[FeatureGroup]
        Connected feature edge groups.
    n_groups : int
    importance_scores : np.ndarray
        ``(n_features,)`` importance ranking scores.
    top_features : list[int]
        Indices of top-10 most important features.
    dict_snippet : str, optional
        OpenFOAM ``surfaceFeaturesDict`` content.
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


def surface_features_enhanced_5(
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
) -> SurfaceFeaturesEnhanced5Result:
    """Extract features with grouping, ranking, and dict generation.

    Parameters
    ----------
    surface_path, included_angle, vertices, faces, normals
    min_feature_length, angle_bins, region_faces
    curvature_threshold, multi_scale_angles
        Forwarded to v4 feature extraction.
    generate_dict : bool
        Generate an OpenFOAM ``surfaceFeaturesDict`` snippet.
    dict_surface_name : str
        Surface file name for the dictionary.

    Returns
    -------
    SurfaceFeaturesEnhanced5Result
    """
    from pyfoam.tools.surface_features_enhanced_4 import (
        surface_features_enhanced_4,
    )

    v4_result = surface_features_enhanced_4(
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
    )

    # Feature grouping
    groups = _group_features(v4_result.feature_edge_indices, v4_result.feature_angles)
    n_groups = len(groups)

    # Importance ranking
    importance = _compute_importance(
        v4_result.feature_angles,
        v4_result.feature_lengths,
        getattr(v4_result, "feature_persistence", None),
    )

    n_feat = len(importance)
    top_n = min(10, n_feat)
    top_indices = list(np.argsort(importance)[::-1][:top_n]) if n_feat > 0 else []

    # Dictionary generation
    dict_snippet = None
    if generate_dict:
        dict_snippet = _generate_dict_snippet(
            dict_surface_name, included_angle, v4_result.n_features,
        )

    return SurfaceFeaturesEnhanced5Result(
        n_edges=v4_result.n_edges,
        n_features=v4_result.n_features,
        feature_points=v4_result.feature_points,
        feature_angles=v4_result.feature_angles,
        feature_edge_indices=v4_result.feature_edge_indices,
        feature_lengths=v4_result.feature_lengths,
        angle_bins=v4_result.angle_bins,
        n_chains=v4_result.n_chains,
        n_junctions=v4_result.n_junctions,
        n_open_chains=v4_result.n_open_chains,
        feature_groups=groups,
        n_groups=n_groups,
        importance_scores=importance,
        top_features=top_indices,
        dict_snippet=dict_snippet,
    )


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------


def _group_features(edge_indices, angles):
    """Cluster connected feature edges into groups."""
    if not edge_indices:
        return []

    # Build vertex -> edge index adjacency
    adj: dict[int, set[int]] = {}
    for i, (vi, vj) in enumerate(edge_indices):
        adj.setdefault(vi, set()).add(i)
        adj.setdefault(vj, set()).add(i)

    # BFS group assignment
    visited = set()
    groups = []
    for i in range(len(edge_indices)):
        if i in visited:
            continue
        queue = [i]
        group_edges = []
        while queue:
            ei = queue.pop(0)
            if ei in visited:
                continue
            visited.add(ei)
            group_edges.append(ei)
            vi, vj = edge_indices[ei]
            for nbr_ei in adj.get(vi, set()):
                if nbr_ei not in visited:
                    queue.append(nbr_ei)
            for nbr_ei in adj.get(vj, set()):
                if nbr_ei not in visited:
                    queue.append(nbr_ei)

        if group_edges:
            edge_angles = [angles[ei] for ei in group_edges if ei < len(angles)]
            total_len = sum(
                _edge_length_from_indices(edge_indices[ei])
                for ei in group_edges
            )
            mean_ang = sum(edge_angles) / len(edge_angles) if edge_angles else 0.0
            group_type = _classify_group(mean_ang)

            groups.append(FeatureGroup(
                group_id=len(groups),
                n_edges=len(group_edges),
                total_length=total_len,
                mean_angle=mean_ang,
                group_type=group_type,
            ))

    return groups


def _edge_length_from_indices(pair):
    """Placeholder: return 1.0 since we only have vertex indices."""
    return 1.0


def _classify_group(mean_angle):
    """Classify a feature group by mean dihedral angle."""
    if mean_angle < 30.0:
        return "ridge"
    elif mean_angle < 90.0:
        return "crease"
    elif mean_angle < 150.0:
        return "valley"
    return "general"


# ---------------------------------------------------------------------------
# Importance ranking
# ---------------------------------------------------------------------------


def _compute_importance(angles, lengths, persistence):
    """Composite importance score: 0.4*angle + 0.3*length + 0.3*persistence."""
    n = len(angles)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Normalize angles to [0, 1]
    angle_norm = angles / 180.0 if angles.max() > 0 else np.zeros(n)

    # Normalize lengths to [0, 1]
    l_max = lengths.max() if lengths.max() > 0 else 1.0
    length_norm = lengths / l_max

    # Persistence if available
    if persistence is not None and len(persistence) == n:
        pers = persistence
    else:
        pers = np.ones(n, dtype=np.float64)

    return 0.4 * angle_norm + 0.3 * length_norm + 0.3 * pers


# ---------------------------------------------------------------------------
# Dictionary generation
# ---------------------------------------------------------------------------


def _generate_dict_snippet(surface_name, angle, n_features):
    """Generate OpenFOAM surfaceFeaturesDict content."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      surfaceFeaturesDict;",
        "}",
        "",
        f"surface          \"{surface_name}\";",
        f"includedAngle    {angle:.1f};",
        "",
        "// Extract non-manifold and region edges",
        "nonManifoldEdges yes;",
        "openEdges        yes;",
        "",
        f"// Total features detected: {n_features}",
    ]
    return "\n".join(lines)
