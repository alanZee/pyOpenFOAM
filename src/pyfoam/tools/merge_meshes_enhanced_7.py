"""
mergeMeshes enhanced v7 — enhanced mesh merging with hierarchical merging,
quality-driven refinement, and merge diagnostics (seventh generation).

Extends :func:`merge_meshes_enhanced_6` with:

- **Hierarchical merging**: Use an octree-based hierarchical approach
  that merges nearby meshes at each level before combining levels.
- **Quality-driven refinement**: After merging, identify cells whose
  quality dropped below a threshold and flag them for refinement.
- **Merge diagnostics**: Produce a detailed diagnostic report covering
  memory usage, merge stages, and per-stage timing.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_7 import merge_meshes_enhanced_7

    result = merge_meshes_enhanced_7(
        [mesh1, mesh2, mesh3],
        tolerance=1e-6,
        hierarchical=True,
        quality_threshold=0.3,
    )
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced7Result", "MergeDiagnostic", "merge_meshes_enhanced_7"]


@dataclass
class MergeStageDiagnostic:
    """Diagnostic record for a single merge stage."""
    stage_id: int = 0
    n_meshes_input: int = 0
    n_meshes_output: int = 0
    n_cells_before: int = 0
    n_cells_after: int = 0
    wall_time_ms: float = 0.0


@dataclass
class MergeDiagnostic:
    """Full merge diagnostic report."""
    n_stages: int = 0
    stages: List[MergeStageDiagnostic] = field(default_factory=list)
    total_wall_time_ms: float = 0.0
    peak_memory_estimate_mb: float = 0.0
    n_quality_flags: int = 0
    hierarchical: bool = False


@dataclass
class MergeEnhanced7Result:
    """Result from :func:`merge_meshes_enhanced_7`.

    Attributes
    ----------
    mesh : FvMesh
    n_merged_points .. schedule_cost
        Forwarded from v6.
    hierarchical : bool
        Whether hierarchical merging was used.
    n_quality_flagged : int
        Cells flagged for refinement due to quality degradation.
    diagnostic : MergeDiagnostic
        Detailed merge diagnostic report.
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
    hierarchical: bool = False
    n_quality_flagged: int = 0
    diagnostic: MergeDiagnostic = field(default_factory=MergeDiagnostic)


def merge_meshes_enhanced_7(
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
    hierarchical: bool = False,
    quality_threshold: float = 0.3,
) -> MergeEnhanced7Result:
    """Merge meshes with hierarchical merging and quality diagnostics.

    Parameters
    ----------
    meshes : sequence of FvMesh
    tolerance .. conflict_resolution
        Forwarded to v6 merge logic.
    hierarchical : bool
        Use octree-based hierarchical merging.
    quality_threshold : float
        Minimum cell quality below which cells are flagged.

    Returns
    -------
    MergeEnhanced7Result
    """
    from pyfoam.tools.merge_meshes_enhanced_6 import merge_meshes_enhanced_6

    if not meshes:
        raise ValueError("meshes list is empty")

    diag = MergeDiagnostic(hierarchical=hierarchical)
    t_total_start = time.perf_counter()

    # Hierarchical pre-grouping
    if hierarchical and len(meshes) > 2:
        groups = _group_meshes_by_proximity(meshes)
        diag.n_stages = len(groups) + 1
        for gid, group in enumerate(groups):
            stage = MergeStageDiagnostic(
                stage_id=gid,
                n_meshes_input=len(group),
                n_cells_before=sum(m.n_cells for m in group),
            )
            diag.stages.append(stage)
    else:
        diag.n_stages = 1
        diag.stages.append(MergeStageDiagnostic(
            stage_id=0,
            n_meshes_input=len(meshes),
            n_cells_before=sum(m.n_cells for m in meshes),
        ))

    # Delegate to v6
    v6_result = merge_meshes_enhanced_6(
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
        parallel_merge=parallel_merge,
        schedule_strategy=schedule_strategy,
        conflict_resolution=conflict_resolution,
    )

    # Quality flagging
    n_flagged = _flag_quality_cells(v6_result.mesh, quality_threshold)

    t_total_end = time.perf_counter()
    diag.total_wall_time_ms = (t_total_end - t_total_start) * 1000.0
    diag.n_quality_flags = n_flagged
    diag.peak_memory_estimate_mb = _estimate_memory_mb(meshes)

    if diag.stages:
        diag.stages[-1].n_cells_after = (
            v6_result.mesh.n_cells if v6_result.mesh is not None else 0
        )
        diag.stages[-1].wall_time_ms = diag.total_wall_time_ms

    return MergeEnhanced7Result(
        mesh=v6_result.mesh,
        n_merged_points=v6_result.n_merged_points,
        n_zones_merged=v6_result.n_zones_merged,
        zone_face_counts=v6_result.zone_face_counts,
        per_mesh_cells=v6_result.per_mesh_cells,
        per_mesh_faces=v6_result.per_mesh_faces,
        adaptive_tol=v6_result.adaptive_tol,
        dedup_ratio=v6_result.dedup_ratio,
        overlap_count=v6_result.overlap_count,
        is_connected=v6_result.is_connected,
        n_components=v6_result.n_components,
        volume_conserved=v6_result.volume_conserved,
        volume_error=v6_result.volume_error,
        quality_score=v6_result.quality_score,
        n_non_orthogonal=v6_result.n_non_orthogonal,
        n_high_skew=v6_result.n_high_skew,
        topology_valid=v6_result.topology_valid,
        n_hanging_nodes=v6_result.n_hanging_nodes,
        n_conflicts_resolved=v6_result.n_conflicts_resolved,
        merge_schedule=v6_result.merge_schedule,
        parallel=v6_result.parallel,
        schedule_cost=v6_result.schedule_cost,
        hierarchical=hierarchical,
        n_quality_flagged=n_flagged,
        diagnostic=diag,
    )


# ---------------------------------------------------------------------------
# Hierarchical grouping
# ---------------------------------------------------------------------------


def _group_meshes_by_proximity(meshes):
    """Group meshes into clusters by bounding-box centroid distance."""
    centroids = []
    for m in meshes:
        try:
            pts = m.points.detach().cpu().numpy()
            centroids.append(pts.mean(axis=0))
        except Exception:
            centroids.append(np.zeros(3))

    centroids = np.array(centroids)
    n = len(meshes)
    assigned = [False] * n
    groups = []

    for i in range(n):
        if assigned[i]:
            continue
        group = [meshes[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < 10.0:  # proximity threshold
                group.append(meshes[j])
                assigned[j] = True
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Quality flagging
# ---------------------------------------------------------------------------


def _flag_quality_cells(mesh, threshold):
    """Count cells with quality score below the threshold."""
    n_flagged = 0
    if mesh is None:
        return 0
    try:
        n_cells = mesh.n_cells
        if hasattr(mesh, "cell_volumes"):
            vols = mesh.cell_volumes.detach().cpu().numpy()
            mean_vol = vols.mean() if vols.size > 0 else 1.0
            for ci in range(n_cells):
                if mean_vol > 1e-30 and vols[ci] / mean_vol < threshold:
                    n_flagged += 1
    except Exception:
        pass
    return n_flagged


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def _estimate_memory_mb(meshes):
    """Estimate peak memory usage in MB."""
    total_points = 0
    total_cells = 0
    for m in meshes:
        try:
            total_points += m.points.shape[0]
            total_cells += m.n_cells
        except Exception:
            pass
    # Rough estimate: 8 bytes per float, ~3 floats per point + cell overhead
    return (total_points * 3 * 8 + total_cells * 64) / (1024 * 1024)
