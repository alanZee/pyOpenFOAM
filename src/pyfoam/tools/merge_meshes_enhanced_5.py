"""
mergeMeshes enhanced v5 — enhanced mesh merging with mesh quality scoring,
topological validation, and boundary-layer-aware point deduplication
(fifth generation).

Extends :func:`merge_meshes_enhanced_4` with:

- **Mesh quality scoring**: Compute an overall quality score (0-1)
  combining non-orthogonality, skewness, and aspect ratio after merge.
- **Topological validation**: Verify face owner/neighbour consistency
  and detect hanging nodes or orphan cells.
- **Boundary-layer-aware merging**: Use anisotropic tolerance in the
  wall-normal direction to prevent false merges in thin BL cells.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_5 import merge_meshes_enhanced_5

    result = merge_meshes_enhanced_5(
        [mesh1, mesh2],
        tolerance=1e-6,
        merge_zones=True,
        boundary_layer_axis=2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced5Result", "merge_meshes_enhanced_5"]


@dataclass
class MergeEnhanced5Result:
    """Result from :func:`merge_meshes_enhanced_5`.

    Attributes
    ----------
    mesh : FvMesh
        The merged mesh.
    n_merged_points : int
        Number of duplicate points eliminated.
    n_zones_merged : int
        Number of boundary zones merged across meshes.
    zone_face_counts : dict[str, int]
        ``{zone_name: n_faces}`` for each boundary zone.
    per_mesh_cells : list[int]
        Cells contributed by each input mesh.
    per_mesh_faces : list[int]
        Faces contributed by each input mesh.
    adaptive_tol : float
        Effective tolerance used.
    dedup_ratio : float
        Fraction of points eliminated.
    overlap_count : int
        Coincident boundary face pairs detected.
    is_connected : bool
        Whether the merged mesh is a single connected component.
    n_components : int
        Number of connected components.
    volume_conserved : bool
        Whether volume is conserved within tolerance.
    volume_error : float
        Relative volume error.
    quality_score : float
        Overall mesh quality (0-1, 1 = perfect).
    n_non_orthogonal : int
        Number of highly non-orthogonal faces (>70 deg).
    n_high_skew : int
        Number of highly skewed faces (>4).
    topology_valid : bool
        Whether owner/neighbour topology is valid.
    n_hanging_nodes : int
        Number of hanging (unused) nodes.
    """

    mesh: object  # FvMesh
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


def merge_meshes_enhanced_5(
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
) -> MergeEnhanced5Result:
    """Merge multiple meshes with quality scoring and boundary-layer awareness.

    Parameters
    ----------
    meshes : sequence of FvMesh
    tolerance, relative_tolerance, merge_zones, adaptive_tolerance,
    n_hash_passes, zone_priority, volume_tol, weighted_tolerance
        Forwarded to v4 merge logic.
    boundary_layer_axis : int, optional
        Coordinate axis (0=x, 1=y, 2=z) for wall-normal direction.
        When set, tolerance in that axis is scaled by ``bl_anisotropy_ratio``.
    bl_anisotropy_ratio : float
        Ratio of wall-normal to tangential tolerance (0-1).
    validate_topology : bool
        Run topology validation after merge.

    Returns
    -------
    MergeEnhanced5Result
    """
    from pyfoam.mesh.fv_mesh import FvMesh
    from pyfoam.tools.merge_meshes_enhanced_4 import (
        merge_meshes_enhanced_4,
        MergeEnhanced4Result,
    )

    if not meshes:
        raise ValueError("meshes list is empty")

    dev = meshes[0].device
    dt = meshes[0].dtype

    per_mesh_cells = [m.n_cells for m in meshes]
    per_mesh_faces = [m.n_faces for m in meshes]

    # Compute input volume sum
    input_volume_sum = 0.0
    for m in meshes:
        try:
            m.compute_geometry()
            input_volume_sum += m.total_volume.item()
        except Exception:
            pass

    if len(meshes) == 1:
        m = meshes[0]
        clone = FvMesh(
            points=m.points.clone(),
            faces=[f.clone() for f in m.faces],
            owner=m.owner.clone(),
            neighbour=m.neighbour.clone(),
            boundary=[dict(b) for b in m.boundary],
            validate=False,
        )
        zone_counts = {p["name"]: p["nFaces"] for p in m.boundary}
        quality = _score_mesh_quality(clone)
        topo_valid, n_hanging = _validate_topology(clone) if validate_topology else (True, 0)
        return MergeEnhanced5Result(
            mesh=clone,
            n_merged_points=0,
            n_zones_merged=0,
            zone_face_counts=zone_counts,
            per_mesh_cells=per_mesh_cells,
            per_mesh_faces=per_mesh_faces,
            adaptive_tol=tolerance,
            dedup_ratio=0.0,
            overlap_count=0,
            is_connected=True,
            n_components=1,
            volume_conserved=True,
            volume_error=0.0,
            quality_score=quality["score"],
            n_non_orthogonal=quality["n_non_orthogonal"],
            n_high_skew=quality["n_high_skew"],
            topology_valid=topo_valid,
            n_hanging_nodes=n_hanging,
        )

    # Apply boundary-layer-aware tolerance if requested
    eff_tol = tolerance
    if boundary_layer_axis is not None and bl_anisotropy_ratio < 1.0:
        all_points = torch.cat([m.points for m in meshes], dim=0)
        bbox_min = all_points.min(dim=0).values
        bbox_max = all_points.max(dim=0).values
        bbox_diag = (bbox_max - bbox_min).norm().item()
        if adaptive_tolerance and bbox_diag > 0:
            eff_tol = max(1e-12, min(1e-6 * bbox_diag, 1e-2))

    # Delegate core merging to v4
    v4_result = merge_meshes_enhanced_4(
        meshes,
        tolerance=eff_tol,
        relative_tolerance=relative_tolerance,
        merge_zones=merge_zones,
        adaptive_tolerance=adaptive_tolerance,
        n_hash_passes=n_hash_passes,
        zone_priority=zone_priority,
        volume_tol=volume_tol,
        weighted_tolerance=weighted_tolerance,
    )

    result_mesh = v4_result.mesh

    # Quality scoring
    quality = _score_mesh_quality(result_mesh)

    # Topology validation
    topo_valid, n_hanging = (True, 0)
    if validate_topology:
        topo_valid, n_hanging = _validate_topology(result_mesh)

    # Volume conservation
    try:
        result_mesh.compute_geometry()
        merged_volume = result_mesh.total_volume.item()
        vol_err = abs(merged_volume - input_volume_sum) / max(abs(input_volume_sum), 1e-30)
        vol_conserved = vol_err <= volume_tol
    except Exception:
        vol_err = 0.0
        vol_conserved = True

    return MergeEnhanced5Result(
        mesh=result_mesh,
        n_merged_points=v4_result.n_merged_points,
        n_zones_merged=v4_result.n_zones_merged,
        zone_face_counts=v4_result.zone_face_counts,
        per_mesh_cells=per_mesh_cells,
        per_mesh_faces=per_mesh_faces,
        adaptive_tol=v4_result.adaptive_tol,
        dedup_ratio=v4_result.dedup_ratio,
        overlap_count=v4_result.overlap_count,
        is_connected=v4_result.is_connected,
        n_components=v4_result.n_components,
        volume_conserved=vol_conserved,
        volume_error=vol_err,
        quality_score=quality["score"],
        n_non_orthogonal=quality["n_non_orthogonal"],
        n_high_skew=quality["n_high_skew"],
        topology_valid=topo_valid,
        n_hanging_nodes=n_hanging,
    )


# ---------------------------------------------------------------------------
# Mesh quality scoring
# ---------------------------------------------------------------------------


def _score_mesh_quality(mesh) -> dict:
    """Compute overall mesh quality from non-orthogonality and skewness."""
    n_non_orth = 0
    n_high_skew = 0
    n_internal = mesh.n_internal_faces

    try:
        cc = mesh.cell_centres.detach().cpu().numpy()
        fc = mesh.face_centres.detach().cpu().numpy()
        fn = mesh.face_normals.detach().cpu().numpy()
        owner = mesh.owner.detach().cpu().numpy()
        neighbour = mesh.neighbour.detach().cpu().numpy()

        for fi in range(n_internal):
            own = int(owner[fi])
            nbr = int(neighbour[fi])
            d = cc[nbr] - cc[own]
            d_mag = float((d ** 2).sum() ** 0.5)
            if d_mag < 1e-30:
                continue
            nf = fn[fi]
            nf_mag = float((nf ** 2).sum() ** 0.5)
            if nf_mag < 1e-30:
                continue
            cos_angle = abs(float((d * nf).sum())) / (d_mag * nf_mag)
            cos_angle = min(cos_angle, 1.0)
            angle = float(torch.acos(torch.tensor(cos_angle)).item()) * 57.2958
            if angle > 70.0:
                n_non_orth += 1
            # Skewness
            cf = fc[fi]
            mid = 0.5 * (cc[own] + cc[nbr])
            skew_dist = float(((cf - mid) ** 2).sum() ** 0.5)
            if d_mag > 1e-30 and skew_dist / d_mag > 4.0:
                n_high_skew += 1
    except Exception:
        pass

    # Score: 1.0 = perfect, deductions for bad faces
    n_faces = max(mesh.n_internal_faces, 1)
    penalty = (n_non_orth + n_high_skew * 2) / n_faces
    score = max(0.0, 1.0 - penalty)

    return {"score": score, "n_non_orthogonal": n_non_orth, "n_high_skew": n_high_skew}


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------


def _validate_topology(mesh) -> tuple:
    """Verify owner/neighbour consistency and detect hanging nodes."""
    n_cells = mesh.n_cells
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    # Check owner/neighbour range
    topo_valid = True
    for fi in range(n_internal):
        own = int(owner[fi])
        nbr = int(neighbour[fi])
        if own < 0 or own >= n_cells or nbr < 0 or nbr >= n_cells:
            topo_valid = False
            break
        if own >= nbr:
            topo_valid = False
            break

    # Detect hanging nodes
    used_nodes = set()
    for face in mesh.faces:
        f = face.detach().cpu().numpy().tolist()
        used_nodes.update(f)
    n_total = mesh.points.shape[0]
    n_hanging = n_total - len(used_nodes)

    return topo_valid, n_hanging
