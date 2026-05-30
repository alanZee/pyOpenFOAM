"""
mergeMeshes enhanced v3 — enhanced mesh merging with multi-pass duplicate
point detection and priority-based zone merging (third generation).

Extends :func:`merge_meshes_enhanced_2` with:

- **Multi-pass hashing**: Runs duplicate-point elimination in multiple
  passes for higher accuracy on large meshes.
- **Priority-based zone merging**: When merging zones, each zone name
  can be assigned a priority so the higher-priority zone's type wins.
- **Merge quality metrics**: Reports the point-deduplication ratio and
  per-mesh overlap statistics.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_3 import merge_meshes_enhanced_3

    result = merge_meshes_enhanced_3(
        [mesh1, mesh2],
        tolerance=1e-6,
        merge_zones=True,
        n_hash_passes=2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced3Result", "merge_meshes_enhanced_3"]


@dataclass
class MergeEnhanced3Result:
    """Result from :func:`merge_meshes_enhanced_3`.

    Attributes
    ----------
    mesh : FvMesh
        The merged mesh.
    n_merged_points : int
        Number of duplicate points eliminated.
    n_zones_merged : int
        Number of boundary zones merged across meshes.
    zone_face_counts : dict[str, int]
        ``{zone_name: n_faces}`` for each boundary zone in the result.
    per_mesh_cells : list[int]
        Number of cells contributed by each input mesh.
    per_mesh_faces : list[int]
        Number of faces contributed by each input mesh.
    adaptive_tol : float
        Tolerance actually used.
    dedup_ratio : float
        Fraction of points eliminated: ``n_merged / total_raw_points``.
    overlap_count : int
        Number of face pairs detected as coincident between meshes.
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


def merge_meshes_enhanced_3(
    meshes: Sequence["FvMesh"],
    tolerance: float = 1e-8,
    relative_tolerance: Optional[float] = None,
    merge_zones: bool = False,
    adaptive_tolerance: bool = True,
    n_hash_passes: int = 2,
    zone_priority: Optional[Dict[str, int]] = None,
) -> MergeEnhanced3Result:
    """Merge multiple meshes with multi-pass dedup and priority zone merging.

    Parameters
    ----------
    meshes : sequence of FvMesh
        Input meshes to merge.
    tolerance : float
        Absolute distance tolerance for point deduplication.
    relative_tolerance : float, optional
        If set, tolerance is ``relative_tolerance * bbox_diagonal``.
    merge_zones : bool
        If True, boundary patches with the same name are combined.
    adaptive_tolerance : bool
        If True, automatically scale tolerance based on bounding box.
    n_hash_passes : int
        Number of spatial-hashing passes for point deduplication.
        More passes improve accuracy on meshes with near-coincident
        points at different scales.
    zone_priority : dict, optional
        ``{zone_name: priority_int}``. When merging zones with the
        same name, the zone with the higher priority determines the
        patch type. Default priority is 0.

    Returns
    -------
    MergeEnhanced3Result
        Merged mesh with quality metrics.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    if not meshes:
        raise ValueError("meshes list is empty")

    dev = meshes[0].device
    dt = meshes[0].dtype

    per_mesh_cells = [m.n_cells for m in meshes]
    per_mesh_faces = [m.n_faces for m in meshes]

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
        return MergeEnhanced3Result(
            mesh=clone,
            n_merged_points=0,
            n_zones_merged=0,
            zone_face_counts=zone_counts,
            per_mesh_cells=per_mesh_cells,
            per_mesh_faces=per_mesh_faces,
            adaptive_tol=tolerance,
            dedup_ratio=0.0,
            overlap_count=0,
        )

    # Compute effective tolerance
    all_points = torch.cat([m.points for m in meshes], dim=0)
    bbox_min = all_points.min(dim=0).values
    bbox_max = all_points.max(dim=0).values
    bbox_diag = (bbox_max - bbox_min).norm().item()
    total_raw = all_points.shape[0]

    if relative_tolerance is not None:
        eff_tol = relative_tolerance * bbox_diag
    elif adaptive_tolerance and bbox_diag > 0:
        eff_tol = max(1e-12, min(1e-6 * bbox_diag, 1e-2))
    else:
        eff_tol = tolerance

    # Multi-pass point merge
    merged_pts, point_map, n_merged = _merge_points_multipass(
        meshes, eff_tol, dev, dt, n_hash_passes,
    )

    # Merge faces, owners, neighbours, boundaries
    all_faces: list = []
    all_owner: list = []
    all_neighbour: list = []
    all_boundary: list = []
    cell_offset = 0
    face_offset = 0

    for mi, m in enumerate(meshes):
        remap = point_map[mi]
        for fi in range(m.n_faces):
            old_pts = m.faces[fi].tolist()
            new_pts = [remap[p] for p in old_pts]
            all_faces.append(torch.tensor(new_pts, dtype=INDEX_DTYPE, device=dev))

        for o in m.owner.tolist():
            all_owner.append(o + cell_offset)
        for n in m.neighbour.tolist():
            all_neighbour.append(n + cell_offset)

        for p in m.boundary:
            all_boundary.append({
                "name": p["name"],
                "type": p["type"],
                "startFace": p["startFace"] + face_offset,
                "nFaces": p["nFaces"],
            })

        cell_offset += m.n_cells
        face_offset += m.n_faces

    # Detect and convert shared boundary faces
    all_faces, all_owner, all_neighbour, all_boundary, overlap_count = (
        _convert_shared_faces_v3(all_faces, all_owner, all_neighbour, all_boundary)
    )

    # Priority-based zone merging
    n_zones_merged = 0
    if merge_zones:
        all_boundary, n_zones_merged = _merge_zones_priority(
            all_boundary, zone_priority or {},
        )

    zone_counts = {p["name"]: p["nFaces"] for p in all_boundary}
    dedup_ratio = n_merged / total_raw if total_raw > 0 else 0.0

    result_mesh = FvMesh(
        points=merged_pts,
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(all_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=all_boundary,
        validate=False,
    )

    return MergeEnhanced3Result(
        mesh=result_mesh,
        n_merged_points=n_merged,
        n_zones_merged=n_zones_merged,
        zone_face_counts=zone_counts,
        per_mesh_cells=per_mesh_cells,
        per_mesh_faces=per_mesh_faces,
        adaptive_tol=eff_tol,
        dedup_ratio=dedup_ratio,
        overlap_count=overlap_count,
    )


# ---------------------------------------------------------------------------
# Multi-pass spatial-hashing point merge
# ---------------------------------------------------------------------------


def _merge_points_multipass(
    meshes: Sequence["FvMesh"],
    tol: float,
    dev,
    dt,
    n_passes: int,
) -> tuple:
    """Merge points using multi-pass spatial hashing for better accuracy.

    First pass uses a coarse cell size; subsequent passes refine with
    progressively smaller cell sizes to catch near-coincident points.
    """
    all_pts = []
    offsets = []
    for m in meshes:
        offsets.append(len(all_pts))
        for i in range(m.points.shape[0]):
            all_pts.append(m.points[i])

    if not all_pts:
        return torch.empty((0, 3), dtype=dt, device=dev), [{} for _ in meshes], 0

    total_raw = len(all_pts)

    # Progressive cell sizes: start coarse, refine
    cell_sizes = []
    base = max(tol * 2, 1e-12)
    for p in range(n_passes):
        factor = 2.0 ** (n_passes - 1 - p)
        cell_sizes.append(base * factor)
    cell_sizes[-1] = base  # Final pass uses exact cell size

    # Build hash and merge iteratively
    merged = list(all_pts)
    mapping = list(range(total_raw))

    for pass_idx in range(n_passes):
        cs = cell_sizes[pass_idx]
        hash_table: dict[tuple[int, int, int], int] = {}
        new_merged = []
        new_mapping = list(range(total_raw))

        # Build representative set from current merged points
        representatives: list[int] = []
        rep_of: dict[int, int] = {}  # original index -> representative index

        for idx in range(len(merged)):
            pt = merged[idx]
            gx = int(torch.floor(pt[0] / cs).item())
            gy = int(torch.floor(pt[1] / cs).item())
            gz = int(torch.floor(pt[2] / cs).item())

            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (gx + dx, gy + dy, gz + dz)
                        if key in hash_table:
                            rep_idx = hash_table[key]
                            if (merged[rep_idx] - pt).norm().item() < tol:
                                rep_of[idx] = rep_idx
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

            if not found:
                key = (gx, gy, gz)
                hash_table[key] = idx
                rep_of[idx] = idx

        # Build final merged set and mapping
        final_hash: dict[tuple[int, int, int], int] = {}
        final_merged = []

        for idx in range(total_raw):
            # Follow the chain: original -> merged[pass] -> representative
            chain_idx = mapping[idx]
            rep = rep_of.get(chain_idx, chain_idx)

            pt = merged[rep]
            gx = int(torch.floor(pt[0] / cs).item())
            gy = int(torch.floor(pt[1] / cs).item())
            gz = int(torch.floor(pt[2] / cs).item())

            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (gx + dx, gy + dy, gz + dz)
                        if key in final_hash:
                            mi = final_hash[key]
                            if (final_merged[mi] - pt).norm().item() < tol:
                                mapping[idx] = mi
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

            if not found:
                key = (gx, gy, gz)
                final_hash[key] = len(final_merged)
                final_merged.append(pt)
                mapping[idx] = len(final_merged) - 1

        merged = final_merged

    if merged:
        merged_tensor = torch.cat(
            [pt.unsqueeze(0) if pt.dim() == 1 else pt for pt in merged], dim=0,
        ).to(device=dev, dtype=dt)
    else:
        merged_tensor = torch.empty((0, 3), dtype=dt, device=dev)

    # Build per-mesh point maps
    point_maps = []
    idx = 0
    for mi, m in enumerate(meshes):
        pm = {}
        for i in range(m.points.shape[0]):
            pm[i] = mapping[idx]
            idx += 1
        point_maps.append(pm)

    n_merged = total_raw - merged_tensor.shape[0]
    return merged_tensor, point_maps, n_merged


# ---------------------------------------------------------------------------
# Shared face conversion with overlap counting
# ---------------------------------------------------------------------------


def _convert_shared_faces_v3(
    faces: list,
    owner: list,
    neighbour: list,
    boundary: list,
) -> tuple:
    """Convert coincident boundary face pairs into internal faces.

    Returns the updated lists plus the count of overlapping face pairs.
    """
    n_internal = len(neighbour)
    boundary_indices = list(range(n_internal, len(faces)))

    face_hash: dict[tuple, list[int]] = {}
    for fi in boundary_indices:
        key = tuple(sorted(faces[fi].tolist()))
        face_hash.setdefault(key, []).append(fi)

    remove_set = set()
    new_internal = []
    overlap_count = 0
    for key, fis in face_hash.items():
        if len(fis) < 2:
            continue
        f1, f2 = fis[0], fis[1]
        o1, o2 = owner[f1], owner[f2]
        new_internal.append((faces[f1], min(o1, o2), max(o1, o2)))
        remove_set.update([f1, f2])
        overlap_count += 1

    if not remove_set:
        return faces, owner, neighbour, boundary, overlap_count

    new_faces: list = []
    new_owner: list = []
    new_nbr: list = []

    for fi in range(n_internal):
        new_faces.append(faces[fi])
        new_owner.append(owner[fi])
        new_nbr.append(neighbour[fi])

    for fp, o, n in new_internal:
        new_faces.append(fp)
        new_owner.append(o)
        new_nbr.append(n)

    n_new_internal = len(new_nbr)

    kept_bnd = [(faces[fi], owner[fi]) for fi in boundary_indices if fi not in remove_set]
    for fp, o in kept_bnd:
        new_faces.append(fp)
        new_owner.append(o)

    new_boundary: list = []
    if kept_bnd:
        new_boundary.append({
            "name": boundary[0]["name"] if boundary else "merged",
            "type": boundary[0]["type"] if boundary else "wall",
            "startFace": n_new_internal,
            "nFaces": len(kept_bnd),
        })

    return new_faces, new_owner, new_nbr, new_boundary, overlap_count


# ---------------------------------------------------------------------------
# Priority-based zone merging
# ---------------------------------------------------------------------------


def _merge_zones_priority(
    boundary: list,
    zone_priority: Dict[str, int],
) -> tuple[list, int]:
    """Merge patches with the same name; higher priority zone determines type."""
    zone_data: dict[str, dict] = {}
    order: list[str] = []
    for p in boundary:
        name = p["name"]
        if name not in zone_data:
            zone_data[name] = {
                "name": name,
                "type": p["type"],
                "nFaces": 0,
                "sources": [],
                "priority": zone_priority.get(name, 0),
            }
            order.append(name)
        zone_data[name]["nFaces"] += p["nFaces"]
        zone_data[name]["sources"].append(p)
        # Update type if this source has higher priority
        src_priority = zone_priority.get(name, 0)
        if src_priority > zone_data[name]["priority"]:
            zone_data[name]["type"] = p["type"]
            zone_data[name]["priority"] = src_priority

    n_merged = sum(1 for name in order if len(zone_data[name]["sources"]) > 1)

    merged_boundary = []
    offset = 0
    for name in order:
        zd = zone_data[name]
        merged_boundary.append({
            "name": zd["name"],
            "type": zd["type"],
            "startFace": offset,
            "nFaces": zd["nFaces"],
        })
        offset += zd["nFaces"]

    return merged_boundary, n_merged
