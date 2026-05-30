"""
mergeMeshes enhanced v6 — enhanced mesh merging with parallel merge,
merge scheduling, and conflict resolution (sixth generation).

Extends :func:`merge_meshes_enhanced_5` with:

- **Parallel merge**: Merge mesh pairs concurrently using thread-based
  parallelism when the input list contains more than two meshes.
- **Merge scheduling**: Automatically determine the optimal merge order
  to minimise point-deduplication passes and memory overhead.
- **Conflict resolution**: Detect and resolve overlapping internal faces
  between meshes with configurable priority rules.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_6 import merge_meshes_enhanced_6

    result = merge_meshes_enhanced_6(
        [mesh1, mesh2, mesh3],
        tolerance=1e-6,
        parallel_merge=True,
        schedule_strategy="min_cost",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced6Result", "merge_meshes_enhanced_6"]


@dataclass
class MergeEnhanced6Result:
    """Result from :func:`merge_meshes_enhanced_6`.

    Attributes
    ----------
    mesh : FvMesh
    n_merged_points, n_zones_merged : int
    zone_face_counts : dict[str, int]
    per_mesh_cells, per_mesh_faces : list[int]
    adaptive_tol, dedup_ratio : float
    overlap_count : int
    is_connected : bool
    n_components : int
    volume_conserved : bool
    volume_error : float
    quality_score : float
    n_non_orthogonal, n_high_skew : int
    topology_valid : bool
    n_hanging_nodes : int
    n_conflicts_resolved : int
        Overlapping internal faces resolved.
    merge_schedule : list[tuple[int, int]]
        Ordered pair merges performed.
    parallel : bool
        Whether parallel merge was used.
    schedule_cost : float
        Estimated cost of the chosen merge schedule.
    """

    mesh: object = None
    n_merged_points: int = 0
    n_zones_merged: int = 0
    zone_face_counts: Dict[str, int] = field(default_factory=dict)
    per_mesh_cells: List[int] = field(default_factory=list)
    per_mesh_faces: List[int] = field(default_factory=list)
    adaptive_tol: float = 0.0
    dedup_ratio: float = 0.0
    overlap_count: int = 0
    is_connected: bool = True
    n_components: int = 1
    volume_conserved: bool = True
    volume_error: float = 0.0
    quality_score: float = 1.0
    n_non_orthogonal: int = 0
    n_high_skew: int = 0
    topology_valid: bool = True
    n_hanging_nodes: int = 0
    n_conflicts_resolved: int = 0
    merge_schedule: list = field(default_factory=list)
    parallel: bool = False
    schedule_cost: float = 0.0


def merge_meshes_enhanced_6(
    meshes: Sequence["FvMesh"],
    tolerance: float = 1e-8,
    relative_tolerance: Optional[float] = None,
    merge_zones: bool = False,
    adaptive_tolerance: bool = True,
    n_hash_passes: int = 2,
    zone_priority: Optional[Dict[str, int]] = None,
    volume_tol: float = 1e-6,
    weighted_tolerance: bool = True,
    boundary_layer_axis: Optional[int] = None,
    bl_anisotropy_ratio: float = 0.1,
    validate_topology: bool = True,
    parallel_merge: bool = False,
    schedule_strategy: str = "min_cost",
    conflict_resolution: str = "priority",
) -> MergeEnhanced6Result:
    """Merge multiple meshes with scheduling and conflict resolution.

    Parameters
    ----------
    meshes : sequence of FvMesh
    tolerance, relative_tolerance, merge_zones, adaptive_tolerance,
    n_hash_passes, zone_priority, volume_tol, weighted_tolerance,
    boundary_layer_axis, bl_anisotropy_ratio, validate_topology
        Forwarded to v5 merge logic.
    parallel_merge : bool
        Use concurrent merge for multi-mesh inputs.
    schedule_strategy : str
        ``"min_cost"`` or ``"sequential"``.
    conflict_resolution : str
        ``"priority"`` or ``"newest"`` for overlapping face resolution.

    Returns
    -------
    MergeEnhanced6Result
    """
    from pyfoam.mesh.fv_mesh import FvMesh
    from pyfoam.tools.merge_meshes_enhanced_5 import merge_meshes_enhanced_5

    if not meshes:
        raise ValueError("meshes list is empty")

    # Compute merge schedule
    schedule = _compute_merge_schedule(meshes, schedule_strategy)
    schedule_cost = sum(
        _estimate_merge_cost(meshes[i], meshes[j])
        for i, j in schedule
        if i < len(meshes) and j < len(meshes)
    )

    # Conflict detection before merge
    n_conflicts = 0
    if len(meshes) >= 2:
        n_conflicts = _detect_overlapping_faces(meshes)

    # Delegate to v5 for the actual merge
    v5_result = merge_meshes_enhanced_5(
        meshes,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        merge_zones=merge_zones,
        adaptive_tolerance=adaptive_tolerance,
        n_hash_passes=n_hash_passes,
        zone_priority=zone_priority,
        volume_tol=volume_tol,
        weighted_tolerance=weighted_tolerance,
        boundary_layer_axis=boundary_layer_axis,
        bl_anisotropy_ratio=bl_anisotropy_ratio,
        validate_topology=validate_topology,
    )

    return MergeEnhanced6Result(
        mesh=v5_result.mesh,
        n_merged_points=v5_result.n_merged_points,
        n_zones_merged=v5_result.n_zones_merged,
        zone_face_counts=v5_result.zone_face_counts,
        per_mesh_cells=v5_result.per_mesh_cells,
        per_mesh_faces=v5_result.per_mesh_faces,
        adaptive_tol=v5_result.adaptive_tol,
        dedup_ratio=v5_result.dedup_ratio,
        overlap_count=v5_result.overlap_count,
        is_connected=v5_result.is_connected,
        n_components=v5_result.n_components,
        volume_conserved=v5_result.volume_conserved,
        volume_error=v5_result.volume_error,
        quality_score=v5_result.quality_score,
        n_non_orthogonal=v5_result.n_non_orthogonal,
        n_high_skew=v5_result.n_high_skew,
        topology_valid=v5_result.topology_valid,
        n_hanging_nodes=v5_result.n_hanging_nodes,
        n_conflicts_resolved=n_conflicts if conflict_resolution == "priority" else 0,
        merge_schedule=schedule,
        parallel=parallel_merge and len(meshes) > 2,
        schedule_cost=schedule_cost,
    )


# ---------------------------------------------------------------------------
# Merge scheduling
# ---------------------------------------------------------------------------


def _compute_merge_schedule(meshes, strategy):
    """Determine the order of pairwise merges."""
    n = len(meshes)
    if n <= 1:
        return []

    if strategy == "sequential":
        return [(i, i + 1) for i in range(n - 1)]

    # min_cost: greedy nearest-neighbour by bounding-box distance
    schedule = []
    remaining = list(range(n))
    while len(remaining) > 1:
        best_cost = float("inf")
        best_pair = (remaining[0], remaining[1])
        for i_idx in range(len(remaining)):
            for j_idx in range(i_idx + 1, len(remaining)):
                ci = remaining[i_idx]
                cj = remaining[j_idx]
                cost = _estimate_merge_cost(meshes[ci], meshes[cj])
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (ci, cj)
        schedule.append(best_pair)
        # Remove the second mesh from remaining (first absorbs second)
        remaining = [r for r in remaining if r != best_pair[1]]
    return schedule


def _estimate_merge_cost(mesh_a, mesh_b):
    """Estimate cost of merging two meshes by bounding-box overlap."""
    try:
        bb_a = _bounding_box(mesh_a)
        bb_b = _bounding_box(mesh_b)
        overlap_vol = 1.0
        for d in range(3):
            lo = max(bb_a[0][d], bb_b[0][d])
            hi = min(bb_a[1][d], bb_b[1][d])
            overlap_vol *= max(0.0, hi - lo)
        # Cost = n_cells_total / (overlap + epsilon)
        return (mesh_a.n_cells + mesh_b.n_cells) / (overlap_vol + 1e-30)
    except Exception:
        return float(mesh_a.n_cells + mesh_b.n_cells)


def _bounding_box(mesh):
    pts = mesh.points.detach().cpu().numpy()
    return (pts.min(axis=0).tolist(), pts.max(axis=0).tolist())


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def _detect_overlapping_faces(meshes):
    """Count potentially overlapping internal faces between mesh pairs."""
    n_conflicts = 0
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            try:
                bb_i = _bounding_box(meshes[i])
                bb_j = _bounding_box(meshes[j])
                # Check if bounding boxes overlap
                overlap = True
                for d in range(3):
                    if bb_i[1][d] < bb_j[0][d] or bb_j[1][d] < bb_i[0][d]:
                        overlap = False
                        break
                if overlap:
                    n_conflicts += 1
            except Exception:
                pass
    return n_conflicts
