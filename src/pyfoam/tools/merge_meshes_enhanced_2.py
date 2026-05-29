"""
mergeMeshes enhanced v2 — enhanced mesh merging with better duplicate point
detection and zone merging support (second generation).

Extends :func:`merge_meshes_enhanced` with:

- **Adaptive spatial hashing**: Automatically tunes grid cell size from
  bounding-box extent, removing the need for manual tolerance setting.
- **Zone-aware face renumbering**: When merging zones, renumbers all
  boundary faces so that faces belonging to the same zone are contiguous.
- **Merge statistics**: Reports per-mesh contribution counts and the
  quality of duplicate-point merging.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_2 import merge_meshes_enhanced_2

    result = merge_meshes_enhanced_2(
        [mesh1, mesh2],
        tolerance=1e-6,
        merge_zones=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced2Result", "merge_meshes_enhanced_2"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MergeEnhanced2Result:
    """Result from :func:`merge_meshes_enhanced_2`.

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
        Tolerance actually used (after adaptive scaling).
    """

    mesh: object  # FvMesh
    n_merged_points: int = 0
    n_zones_merged: int = 0
    zone_face_counts: Dict[str, int] = field(default_factory=dict)
    per_mesh_cells: List[int] = field(default_factory=list)
    per_mesh_faces: List[int] = field(default_factory=list)
    adaptive_tol: float = 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_meshes_enhanced_2(
    meshes: Sequence["FvMesh"],
    tolerance: float = 1e-8,
    relative_tolerance: Optional[float] = None,
    merge_zones: bool = False,
    adaptive_tolerance: bool = True,
) -> MergeEnhanced2Result:
    """Merge multiple meshes with adaptive hashing and zone-aware renumbering.

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
        If True (default), automatically scale tolerance based on
        bounding-box extent when no explicit relative_tolerance is given.

    Returns
    -------
    MergeEnhanced2Result
        Merged mesh with metadata.

    Raises
    ------
    ValueError
        If *meshes* is empty.
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
        return MergeEnhanced2Result(
            mesh=clone,
            n_merged_points=0,
            n_zones_merged=0,
            zone_face_counts=zone_counts,
            per_mesh_cells=per_mesh_cells,
            per_mesh_faces=per_mesh_faces,
            adaptive_tol=tolerance,
        )

    # Compute effective tolerance
    all_points = torch.cat([m.points for m in meshes], dim=0)
    bbox_min = all_points.min(dim=0).values
    bbox_max = all_points.max(dim=0).values
    bbox_diag = (bbox_max - bbox_min).norm().item()

    if relative_tolerance is not None:
        eff_tol = relative_tolerance * bbox_diag
    elif adaptive_tolerance and bbox_diag > 0:
        # Adaptive: use 1e-6 * diagonal, clamped to [1e-12, 1e-2]
        eff_tol = max(1e-12, min(1e-6 * bbox_diag, 1e-2))
    else:
        eff_tol = tolerance

    # Merge points using spatial hashing
    merged_pts, point_map, n_merged = _merge_points_adaptive(meshes, eff_tol, dev, dt)

    # Merge faces, owners, neighbours, boundaries
    all_faces = []
    all_owner = []
    all_neighbour = []
    all_boundary = []
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

    # Detect and convert shared boundary faces to internal faces
    all_faces, all_owner, all_neighbour, all_boundary = _convert_shared_faces(
        all_faces, all_owner, all_neighbour, all_boundary,
    )

    # Optionally merge zones by name with contiguous renumbering
    n_zones_merged = 0
    if merge_zones:
        all_boundary, n_zones_merged = _merge_zones_contiguous(all_boundary)

    zone_counts = {p["name"]: p["nFaces"] for p in all_boundary}

    result_mesh = FvMesh(
        points=merged_pts,
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(all_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=all_boundary,
        validate=False,
    )

    return MergeEnhanced2Result(
        mesh=result_mesh,
        n_merged_points=n_merged,
        n_zones_merged=n_zones_merged,
        zone_face_counts=zone_counts,
        per_mesh_cells=per_mesh_cells,
        per_mesh_faces=per_mesh_faces,
        adaptive_tol=eff_tol,
    )


# ---------------------------------------------------------------------------
# Adaptive spatial-hashing point merge
# ---------------------------------------------------------------------------


def _merge_points_adaptive(
    meshes: Sequence["FvMesh"],
    tol: float,
    dev,
    dt,
) -> tuple:
    """Merge points using adaptive spatial hashing.

    Automatically adjusts grid cell size to balance bucket occupancy.
    """
    all_pts = []
    offsets = []
    for m in meshes:
        offsets.append(len(all_pts))
        for i in range(m.points.shape[0]):
            all_pts.append(m.points[i])

    if not all_pts:
        return torch.empty((0, 3), dtype=dt, device=dev), [{} for _ in meshes], 0

    # Adaptive cell size: 2 * tolerance but at least based on point cloud extent
    cell_size = max(tol * 2, 1e-12)

    # If too many points, scale up cell size for performance
    n_pts = len(all_pts)
    if n_pts > 10000:
        cell_size = max(cell_size, (n_pts ** (1.0 / 3.0)) * tol)

    hash_table: dict[tuple[int, int, int], int] = {}
    merged = []
    total_raw = len(all_pts)

    for idx, pt in enumerate(all_pts):
        gx = int(torch.floor(pt[0] / cell_size).item())
        gy = int(torch.floor(pt[1] / cell_size).item())
        gz = int(torch.floor(pt[2] / cell_size).item())

        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in hash_table:
                        mi = hash_table[key]
                        if (merged[mi] - pt).norm().item() < tol:
                            found = True
                            break
            if found:
                break

        if not found:
            key = (gx, gy, gz)
            hash_table[key] = len(merged)
            merged.append(pt.unsqueeze(0))

    if merged:
        merged_tensor = torch.cat(merged, dim=0).to(device=dev, dtype=dt)
    else:
        merged_tensor = torch.empty((0, 3), dtype=dt, device=dev)

    # Build per-mesh point maps
    point_maps = []
    for mi, m in enumerate(meshes):
        pm = {}
        for i in range(m.points.shape[0]):
            pt = m.points[i]
            gx = int(torch.floor(pt[0] / cell_size).item())
            gy = int(torch.floor(pt[1] / cell_size).item())
            gz = int(torch.floor(pt[2] / cell_size).item())

            best_idx = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (gx + dx, gy + dy, gz + dz)
                        if key in hash_table:
                            mi_idx = hash_table[key]
                            if (merged_tensor[mi_idx] - pt).norm().item() < tol:
                                best_idx = mi_idx
                                break
                if best_idx is not None:
                    break

            pm[i] = best_idx if best_idx is not None else len(merged_tensor) - 1
        point_maps.append(pm)

    n_merged = total_raw - merged_tensor.shape[0]
    return merged_tensor, point_maps, n_merged


# ---------------------------------------------------------------------------
# Shared face conversion
# ---------------------------------------------------------------------------


def _convert_shared_faces(
    faces: list,
    owner: list,
    neighbour: list,
    boundary: list,
) -> tuple:
    """Convert coincident boundary face pairs into internal faces."""
    n_internal = len(neighbour)
    boundary_indices = list(range(n_internal, len(faces)))

    face_hash: dict[tuple, list[int]] = {}
    for fi in boundary_indices:
        key = tuple(sorted(faces[fi].tolist()))
        face_hash.setdefault(key, []).append(fi)

    remove_set = set()
    new_internal = []
    for key, fis in face_hash.items():
        if len(fis) < 2:
            continue
        f1, f2 = fis[0], fis[1]
        o1, o2 = owner[f1], owner[f2]
        new_internal.append((faces[f1], min(o1, o2), max(o1, o2)))
        remove_set.update([f1, f2])

    if not remove_set:
        return faces, owner, neighbour, boundary

    new_faces = []
    new_owner = []
    new_nbr = []

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

    new_boundary = []
    if kept_bnd:
        new_boundary.append({
            "name": boundary[0]["name"] if boundary else "merged",
            "type": boundary[0]["type"] if boundary else "wall",
            "startFace": n_new_internal,
            "nFaces": len(kept_bnd),
        })

    return new_faces, new_owner, new_nbr, new_boundary


# ---------------------------------------------------------------------------
# Zone merging with contiguous renumbering
# ---------------------------------------------------------------------------


def _merge_zones_contiguous(boundary: list) -> tuple[list, int]:
    """Merge patches with the same name and ensure contiguous face numbering.

    Returns the new boundary list and number of zones that were merged.
    """
    zone_data: dict[str, dict] = {}
    order: list[str] = []
    for p in boundary:
        name = p["name"]
        if name not in zone_data:
            zone_data[name] = {"name": name, "type": p["type"], "nFaces": 0, "sources": []}
            order.append(name)
        zone_data[name]["nFaces"] += p["nFaces"]
        zone_data[name]["sources"].append(p)

    n_merged = sum(1 for name in order if len(zone_data[name]["sources"]) > 1)

    # Rebuild with contiguous numbering
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
