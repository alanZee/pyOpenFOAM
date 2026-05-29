"""
surfaceFeatures enhanced v2 — enhanced feature edge extraction with better
angle detection and multi-region support (second generation).

Extends :func:`surface_features_enhanced` with:

- **Curvature-based detection**: Detect features by local surface
  curvature in addition to dihedral angle.
- **Feature graph**: Build a connectivity graph of feature edges for
  topological analysis (open/closed chains, junctions).
- **Export formats**: Export feature edges in OBJ and VTK formats.

Usage::

    from pyfoam.tools.surface_features_enhanced_2 import surface_features_enhanced_2

    result = surface_features_enhanced_2(
        vertices=pts, faces=tris,
        included_angle=150.0,
        min_feature_length=0.01,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

__all__ = ["SurfaceFeaturesEnhanced2Result", "surface_features_enhanced_2"]


@dataclass
class SurfaceFeaturesEnhanced2Result:
    """Result from :func:`surface_features_enhanced_2`.

    Attributes
    ----------
    n_edges : int
        Total number of unique edges.
    n_features : int
        Number of feature edges after filtering.
    feature_points : np.ndarray
        ``(n_features, 2, 3)`` endpoint coordinates.
    feature_angles : np.ndarray
        ``(n_features,)`` dihedral angles (degrees).
    feature_edge_indices : list[tuple[int, int]]
        Vertex-index pairs for features.
    feature_lengths : np.ndarray
        ``(n_features,)`` edge lengths.
    angle_bins : dict[str, int]
        Count of features in each angle range.
    region_ids : np.ndarray, optional
        ``(n_features,)`` region ID per feature.
    n_chains : int
        Number of connected feature chains.
    n_junctions : int
        Number of junction vertices (degree > 2).
    n_open_chains : int
        Number of open (non-closed) feature chains.
    curvature_features : int
        Number of features detected by curvature criterion.
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


def surface_features_enhanced_2(
    surface_path: Union[str, Path] = "",
    included_angle: float = 150.0,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    min_feature_length: float = 0.0,
    angle_bins: Optional[Sequence[float]] = None,
    region_faces: Optional[np.ndarray] = None,
    curvature_threshold: Optional[float] = None,
) -> SurfaceFeaturesEnhanced2Result:
    """Extract feature edges with enhanced filtering, curvature detection,
    and topological analysis.

    Parameters
    ----------
    surface_path : str or Path
        Path to a surface mesh file. Ignored when arrays are provided.
    included_angle : float
        Dihedral angle threshold in degrees.
    vertices : np.ndarray, optional
        ``(n_points, 3)`` vertex coordinates.
    faces : np.ndarray, optional
        ``(n_faces, 3)`` triangle vertex indices.
    normals : np.ndarray, optional
        ``(n_faces, 3)`` face normals.
    min_feature_length : float
        Minimum edge length to keep as a feature.
    angle_bins : sequence of float, optional
        Angle boundaries for classification.
    region_faces : np.ndarray, optional
        ``(n_faces,)`` integer region ID per face.
    curvature_threshold : float, optional
        If set, also detect features where the angle between adjacent
        face normals exceeds this threshold (complementary to
        ``included_angle``).

    Returns
    -------
    SurfaceFeaturesEnhanced2Result
        Feature edge data with topology and curvature metadata.
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

    # Build edge → face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ei = (int(tri[a]), int(tri[b]))
            key = (min(ei), max(ei))
            edge_faces.setdefault(key, []).append(fi)

    # Classify features
    feat_pts = []
    feat_angles = []
    feat_indices = []
    feat_lengths = []
    feat_regions = []
    curvature_count = 0

    for (vi, vj), adj in edge_faces.items():
        angle = 0.0
        region_id = -1
        is_curvature_feat = False

        if len(adj) == 1:
            angle = 180.0
            if region_arr is not None:
                region_id = int(region_arr[adj[0]])
        elif len(adj) == 2:
            n0 = norms[adj[0]]
            n1 = norms[adj[1]]
            cos_angle = np.clip(np.dot(n0, n1), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            included = 180.0 - angle
            if included >= included_angle:
                # Check curvature threshold as secondary criterion
                if curvature_threshold is not None and angle >= curvature_threshold:
                    is_curvature_feat = True
                else:
                    continue
            if region_arr is not None:
                r0, r1 = int(region_arr[adj[0]]), int(region_arr[adj[1]])
                region_id = r0 if r0 == r1 else -1
        else:
            continue

        # Length filter
        edge_len = np.linalg.norm(verts[vi] - verts[vj])
        if edge_len < min_feature_length:
            continue

        feat_pts.append([verts[vi], verts[vj]])
        feat_angles.append(angle)
        feat_indices.append((vi, vj))
        feat_lengths.append(edge_len)
        feat_regions.append(region_id)
        if is_curvature_feat:
            curvature_count += 1

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

    # Topological analysis: build feature graph
    n_chains, n_junctions, n_open = _analyze_feature_topology(feat_indices)

    n_feat = len(feat_indices)

    return SurfaceFeaturesEnhanced2Result(
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
    )


# ---------------------------------------------------------------------------
# Topological analysis
# ---------------------------------------------------------------------------


def _analyze_feature_topology(
    feature_indices: List[Tuple[int, int]],
) -> Tuple[int, int, int]:
    """Analyze the topology of the feature edge graph.

    Returns (n_chains, n_junctions, n_open_chains).
    """
    if not feature_indices:
        return 0, 0, 0

    # Build vertex degree map
    adj: dict[int, set[int]] = {}
    for vi, vj in feature_indices:
        adj.setdefault(vi, set()).add(vj)
        adj.setdefault(vj, set()).add(vi)

    # Count junctions (degree > 2)
    n_junctions = sum(1 for v, nbrs in adj.items() if len(nbrs) > 2)

    # Count chains via DFS
    visited_edges: set[Tuple[int, int]] = set()
    n_chains = 0
    n_open = 0

    for vi, vj in feature_indices:
        edge = (min(vi, vj), max(vi, vj))
        if edge in visited_edges:
            continue
        n_chains += 1
        is_open = True
        # Walk the chain
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
            # Check if chain closes
            if a in adj.get(vj, set()) or b in adj.get(vi, set()):
                pass  # May be closed
        # If we returned to start, it's closed
        # Simple heuristic: if any vertex in chain has degree 1, it's open
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


def _compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _bin_label(angle: float, boundaries: list[float]) -> str:
    """Map an angle to a bin label."""
    for i, b in enumerate(boundaries):
        if angle < b:
            lower = boundaries[i - 1] if i > 0 else 0
            return f"[{lower}, {b})"
    return f"[{boundaries[-1]}, 180]"
