"""
surfaceFeatures enhanced v4 — enhanced feature edge extraction with
multi-scale detection and feature persistence scoring (fourth generation).

Extends :func:`surface_features_enhanced_3` with:

- **Multi-scale feature detection**: Run feature extraction at multiple
  angle thresholds and classify features by scale persistence.
- **Hierarchical classification**: Group features into primary,
  secondary, and tertiary categories based on angle and connectivity.
- **Feature persistence score**: Assign each feature a persistence
  score reflecting how consistently it appears across scales.

Usage::

    from pyfoam.tools.surface_features_enhanced_4 import surface_features_enhanced_4

    result = surface_features_enhanced_4(
        vertices=pts, faces=tris,
        included_angle=150.0,
        multi_scale_angles=[120.0, 150.0, 170.0],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced4Result", "surface_features_enhanced_4"]


@dataclass
class SurfaceFeaturesEnhanced4Result:
    """Result from :func:`surface_features_enhanced_4`.

    Attributes
    ----------
    n_edges, n_features : int
    feature_points : np.ndarray
    feature_angles : np.ndarray
    feature_edge_indices : list[tuple[int, int]]
    feature_lengths : np.ndarray
    angle_bins : dict[str, int]
    region_ids : np.ndarray, optional
    n_chains, n_junctions, n_open_chains : int
    curvature_features : int
    feature_sharpness : np.ndarray
    per_region_features : dict[int, int]
    weighted_mean_angle : float
    feature_persistence : np.ndarray
        ``(n_features,)`` persistence score (0-1) across scales.
    primary_features : int
        Number of primary (coarsest-scale) features.
    secondary_features : int
        Number of secondary features.
    tertiary_features : int
        Number of tertiary (finest-scale) features.
    per_scale_counts : dict[float, int]
        ``{angle_threshold: n_features}`` at each scale.
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
    region_ids: Optional[np.ndarray] = None
    n_chains: int = 0
    n_junctions: int = 0
    n_open_chains: int = 0
    curvature_features: int = 0
    feature_sharpness: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    per_region_features: Dict[int, int] = field(default_factory=dict)
    weighted_mean_angle: float = 0.0
    feature_persistence: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    primary_features: int = 0
    secondary_features: int = 0
    tertiary_features: int = 0
    per_scale_counts: Dict[float, int] = field(default_factory=dict)


def surface_features_enhanced_4(
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
) -> SurfaceFeaturesEnhanced4Result:
    """Extract features with multi-scale detection and persistence scoring.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file.
    included_angle : float
        Dihedral angle threshold in degrees.
    vertices, faces, normals : np.ndarray, optional
        Geometry arrays.
    min_feature_length : float
        Minimum edge length to keep as a feature.
    angle_bins : sequence of float, optional
        Angle boundaries for classification.
    region_faces : np.ndarray, optional
        ``(n_faces,)`` integer region ID per face.
    curvature_threshold : float, optional
        If set, also detect features by curvature criterion.
    multi_scale_angles : sequence of float, optional
        Run detection at these angle thresholds for persistence analysis.

    Returns
    -------
    SurfaceFeaturesEnhanced4Result
        Feature edge data with multi-scale persistence analysis.
    """
    # Obtain geometry
    if vertices is not None and faces is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = np.asarray(faces, dtype=np.int32)
        norms = (
            np.asarray(normals, dtype=np.float64)
            if normals is not None
            else _compute_normals(verts, facs)
        )
    else:
        from pyfoam.tools.surface_convert import _rs, _df
        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, norms, facs = _rs(p, fmt)

    if facs.shape[0] == 0:
        raise ValueError("Surface has no faces.")

    n_faces = facs.shape[0]
    region_arr = np.asarray(region_faces, dtype=np.int32) if region_faces is not None else None

    # Build edge -> face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ei = (int(tri[a]), int(tri[b]))
            key = (min(ei), max(ei))
            edge_faces.setdefault(key, []).append(fi)

    # Multi-scale analysis
    scales = sorted(multi_scale_angles) if multi_scale_angles else [included_angle]
    per_scale_features: dict[float, set] = {}  # angle -> set of edge keys

    for angle_thresh in scales:
        scale_edges = set()
        for (vi, vj), adj in edge_faces.items():
            angle = _compute_edge_angle(adj, norms, verts, vi, vj)
            if angle is not None and angle <= angle_thresh:
                edge_len = np.linalg.norm(verts[vi] - verts[vj])
                if edge_len >= min_feature_length:
                    scale_edges.add((min(vi, vj), max(vi, vj)))
        per_scale_features[angle_thresh] = scale_edges

    per_scale_counts = {a: len(s) for a, s in per_scale_features.items()}

    # Primary extraction at the main included_angle
    feat_pts: list = []
    feat_angles: list = []
    feat_indices: list = []
    feat_lengths: list = []
    feat_regions: list = []
    feat_sharpness: list = []
    feat_persistence: list = []
    curvature_count = 0

    total_weighted_angle = 0.0
    total_weight = 0.0

    # Build persistence lookup
    primary_edges = per_scale_features.get(included_angle, set())
    all_scale_edges = set()
    for s in per_scale_features.values():
        all_scale_edges.update(s)

    for (vi, vj), adj in edge_faces.items():
        angle = _compute_edge_angle(adj, norms, verts, vi, vj)
        if angle is None:
            continue

        included = 180.0 - angle
        is_curvature_feat = False
        if included < included_angle:
            if curvature_threshold is not None and angle >= curvature_threshold:
                is_curvature_feat = True
            else:
                continue

        edge_len = np.linalg.norm(verts[vi] - verts[vj])
        if edge_len < min_feature_length:
            continue

        region_id = -1
        if region_arr is not None:
            if len(adj) == 1:
                region_id = int(region_arr[adj[0]])
            elif len(adj) == 2:
                r0, r1 = int(region_arr[adj[0]]), int(region_arr[adj[1]])
                region_id = r0 if r0 == r1 else -1

        feat_pts.append([verts[vi], verts[vj]])
        feat_angles.append(angle)
        feat_indices.append((vi, vj))
        feat_lengths.append(edge_len)
        feat_regions.append(region_id)
        if is_curvature_feat:
            curvature_count += 1

        angle_score = angle / 180.0
        sharpness = 0.7 * angle_score + 0.3 * min(edge_len * 10.0, 1.0)
        feat_sharpness.append(sharpness)

        # Persistence: fraction of scales that detect this edge
        edge_key = (min(vi, vj), max(vi, vj))
        n_detected = sum(1 for s_edges in per_scale_features.values() if edge_key in s_edges)
        persistence = n_detected / len(scales) if scales else 1.0
        feat_persistence.append(persistence)

        total_weighted_angle += angle * edge_len
        total_weight += edge_len

    # Angle binning
    bin_counts: dict[str, int] = {}
    if angle_bins is not None:
        boundaries = sorted(angle_bins)
        for a in feat_angles:
            label = _bin_label(a, boundaries)
            bin_counts[label] = bin_counts.get(label, 0) + 1
    else:
        for a in feat_angles:
            if a < 30:
                label = "[0, 30)"
            elif a < 90:
                label = "[30, 90)"
            elif a < 150:
                label = "[90, 150)"
            else:
                label = "[150, 180]"
            bin_counts[label] = bin_counts.get(label, 0) + 1

    n_chains, n_junctions, n_open = _analyze_feature_topology(feat_indices)

    per_region: dict[int, int] = {}
    if region_arr is not None:
        for ri in feat_regions:
            if ri >= 0:
                per_region[ri] = per_region.get(ri, 0) + 1

    n_feat = len(feat_indices)
    weighted_mean = total_weighted_angle / total_weight if total_weight > 0 else 0.0

    # Hierarchical classification by persistence
    primary = sum(1 for p in feat_persistence if p >= 0.9)
    secondary = sum(1 for p in feat_persistence if 0.5 <= p < 0.9)
    tertiary = sum(1 for p in feat_persistence if p < 0.5)

    return SurfaceFeaturesEnhanced4Result(
        n_edges=len(edge_faces),
        n_features=n_feat,
        feature_points=np.array(feat_pts, dtype=np.float64) if feat_pts else np.empty((0, 2, 3), dtype=np.float64),
        feature_angles=np.array(feat_angles, dtype=np.float64) if feat_angles else np.empty(0, dtype=np.float64),
        feature_edge_indices=feat_indices,
        feature_lengths=np.array(feat_lengths, dtype=np.float64) if feat_lengths else np.empty(0, dtype=np.float64),
        angle_bins=bin_counts,
        region_ids=np.array(feat_regions, dtype=np.int32) if feat_regions and region_arr is not None else None,
        n_chains=n_chains,
        n_junctions=n_junctions,
        n_open_chains=n_open,
        curvature_features=curvature_count,
        feature_sharpness=np.array(feat_sharpness, dtype=np.float64) if feat_sharpness else np.empty(0, dtype=np.float64),
        per_region_features=per_region,
        weighted_mean_angle=weighted_mean,
        feature_persistence=np.array(feat_persistence, dtype=np.float64) if feat_persistence else np.empty(0, dtype=np.float64),
        primary_features=primary,
        secondary_features=secondary,
        tertiary_features=tertiary,
        per_scale_counts=per_scale_counts,
    )


# ---------------------------------------------------------------------------
# Edge angle computation
# ---------------------------------------------------------------------------


def _compute_edge_angle(adj, norms, verts, vi, vj):
    """Compute dihedral angle for an edge."""
    if len(adj) == 1:
        return 180.0
    elif len(adj) == 2:
        n0 = norms[adj[0]]
        n1 = norms[adj[1]]
        cos_angle = np.clip(np.dot(n0, n1), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    return None


# ---------------------------------------------------------------------------
# Topological analysis
# ---------------------------------------------------------------------------


def _analyze_feature_topology(feature_indices):
    if not feature_indices:
        return 0, 0, 0

    adj: dict[int, set[int]] = {}
    for vi, vj in feature_indices:
        adj.setdefault(vi, set()).add(vj)
        adj.setdefault(vj, set()).add(vi)

    n_junctions = sum(1 for v, nbrs in adj.items() if len(nbrs) > 2)

    visited_edges: set[tuple[int, int]] = set()
    n_chains = 0
    n_open = 0

    for vi, vj in feature_indices:
        edge = (min(vi, vj), max(vi, vj))
        if edge in visited_edges:
            continue
        n_chains += 1
        is_open = True
        stack = [edge]
        while stack:
            e = stack.pop()
            if e in visited_edges:
                continue
            visited_edges.add(e)
            a, b = e
            for c in adj.get(b, set()):
                e2 = (min(b, c), max(b, c))
                if e2 not in visited_edges:
                    stack.append(e2)
        for vi2, vj2 in feature_indices:
            e2 = (min(vi2, vj2), max(vi2, vj2))
            if e2 in visited_edges:
                if len(adj.get(vi2, set())) == 1 or len(adj.get(vj2, set())) == 1:
                    is_open = True
                    break
        if is_open:
            n_open += 1

    return n_chains, n_junctions, n_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _bin_label(angle, boundaries):
    for i, b in enumerate(boundaries):
        if angle < b:
            lower = boundaries[i - 1] if i > 0 else 0
            return f"[{lower}, {b})"
    return f"[{boundaries[-1]}, 180]"
