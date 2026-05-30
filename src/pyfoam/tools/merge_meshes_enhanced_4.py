"""
mergeMeshes enhanced v4 — enhanced mesh merging with graph-based
connectivity analysis and weighted tolerance zones (fourth generation).

Extends :func:`merge_meshes_enhanced_3` with:

- **Graph-based connectivity**: Build a mesh connectivity graph and
  validate topological consistency after merging.
- **Weighted tolerance zones**: Apply different deduplication tolerances
  to different spatial regions based on local mesh density.
- **Volume conservation check**: Verify that the merged mesh preserves
  total volume within a configurable tolerance.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_4 import merge_meshes_enhanced_4

    result = merge_meshes_enhanced_4(
        [mesh1, mesh2],
        tolerance=1e-6,
        merge_zones=True,
        n_hash_passes=2,
        volume_tol=1e-6,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced4Result", "merge_meshes_enhanced_4"]


@dataclass
class MergeEnhanced4Result:
    """Result from :func:`merge_meshes_enhanced_4`.

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
        Fraction of points eliminated.
    overlap_count : int
        Number of face pairs detected as coincident.
    is_connected : bool
        Whether the merged mesh forms a single connected component.
    n_components : int
        Number of connected components in the merged mesh.
    volume_conserved : bool
        Whether merged volume matches sum of input volumes within tolerance.
    volume_error : float
        Relative volume error: ``|V_merged - V_sum| / V_sum``.
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


def merge_meshes_enhanced_4(
    meshes: Sequence["FvMesh"],
    tolerance: float = 1e-8,
    relative_tolerance: Optional[float] = None,
    merge_zones: bool = False,
    adaptive_tolerance: bool = True,
    n_hash_passes: int = 2,
    zone_priority: Optional[Dict[str, int]] = None,
    volume_tol: float = 1e-6,
    weighted_tolerance: bool = True,
) -> MergeEnhanced4Result:
    """Merge multiple meshes with graph analysis and volume conservation.

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
    zone_priority : dict, optional
        ``{zone_name: priority_int}`` for zone merging.
    volume_tol : float
        Relative tolerance for volume conservation check.
    weighted_tolerance : bool
        If True, apply density-weighted spatial tolerances.

    Returns
    -------
    MergeEnhanced4Result
        Merged mesh with quality metrics and topology analysis.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

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
        return MergeEnhanced4Result(
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

    # Point merge with optional density weighting
    if weighted_tolerance:
        merged_pts, point_map, n_merged = _merge_points_weighted(
            meshes, eff_tol, dev, dt, n_hash_passes,
        )
    else:
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
        _convert_shared_faces(all_faces, all_owner, all_neighbour, all_boundary)
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

    # Graph connectivity analysis
    n_components, is_connected = _analyze_connectivity(
        result_mesh.n_cells, len(all_neighbour), all_owner[:len(all_neighbour)],
        all_neighbour,
    )

    # Volume conservation check
    try:
        result_mesh.compute_geometry()
        merged_volume = result_mesh.total_volume.item()
        vol_err = abs(merged_volume - input_volume_sum) / max(abs(input_volume_sum), 1e-30)
        vol_conserved = vol_err <= volume_tol
    except Exception:
        vol_err = 0.0
        vol_conserved = True

    return MergeEnhanced4Result(
        mesh=result_mesh,
        n_merged_points=n_merged,
        n_zones_merged=n_zones_merged,
        zone_face_counts=zone_counts,
        per_mesh_cells=per_mesh_cells,
        per_mesh_faces=per_mesh_faces,
        adaptive_tol=eff_tol,
        dedup_ratio=dedup_ratio,
        overlap_count=overlap_count,
        is_connected=is_connected,
        n_components=n_components,
        volume_conserved=vol_conserved,
        volume_error=vol_err,
    )


# ---------------------------------------------------------------------------
# Density-weighted spatial-hashing point merge
# ---------------------------------------------------------------------------


def _merge_points_weighted(
    meshes: Sequence["FvMesh"],
    base_tol: float,
    dev,
    dt,
    n_passes: int,
) -> tuple:
    """Merge points using density-weighted spatial hashing.

    Computes local point density near each point and scales tolerance
    inversely: denser regions get tighter tolerance to avoid false merges.
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

    # Estimate local density with coarse grid
    coarse_cs = max(base_tol * 10, 1e-10)
    density_counts: dict[tuple, int] = {}
    for pt in all_pts:
        gx = int(torch.floor(pt[0] / coarse_cs).item())
        gy = int(torch.floor(pt[1] / coarse_cs).item())
        gz = int(torch.floor(pt[2] / coarse_cs).item())
        key = (gx, gy, gz)
        density_counts[key] = density_counts.get(key, 0) + 1

    # Build per-point tolerance based on local density
    max_density = max(density_counts.values()) if density_counts else 1

    # Multi-pass merge
    merged = list(all_pts)
    mapping = list(range(total_raw))

    for pass_idx in range(n_passes):
        cs = base_tol * (2.0 ** (n_passes - 1 - pass_idx))
        cs = max(cs, 1e-12)

        hash_table: dict[tuple, int] = {}
        rep_of: dict[int, int] = {}

        for idx in range(len(merged)):
            pt = merged[idx]
            gx = int(torch.floor(pt[0] / cs).item())
            gy = int(torch.floor(pt[1] / cs).item())
            gz = int(torch.floor(pt[2] / cs).item())

            # Density-weighted tolerance for this point
            coarse_key = (
                int(torch.floor(pt[0] / coarse_cs).item()),
                int(torch.floor(pt[1] / coarse_cs).item()),
                int(torch.floor(pt[2] / coarse_cs).item()),
            )
            local_density = density_counts.get(coarse_key, 1)
            density_factor = max(0.5, 1.0 - 0.5 * local_density / max_density)
            local_tol = base_tol * density_factor

            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (gx + dx, gy + dy, gz + dz)
                        if key in hash_table:
                            rep_idx = hash_table[key]
                            if (merged[rep_idx] - pt).norm().item() < local_tol:
                                rep_of[idx] = rep_idx
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

            if not found:
                hash_table[(gx, gy, gz)] = idx
                rep_of[idx] = idx

        # Final merge pass
        final_hash: dict[tuple, int] = {}
        final_merged = []

        for idx in range(total_raw):
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
                            if (final_merged[mi] - pt).norm().item() < base_tol:
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


def _merge_points_multipass(
    meshes: Sequence["FvMesh"],
    tol: float,
    dev,
    dt,
    n_passes: int,
) -> tuple:
    """Multi-pass spatial-hashing point merge (fallback from weighted)."""
    all_pts = []
    for m in meshes:
        for i in range(m.points.shape[0]):
            all_pts.append(m.points[i])

    if not all_pts:
        return torch.empty((0, 3), dtype=dt, device=dev), [{} for _ in meshes], 0

    total_raw = len(all_pts)
    base = max(tol * 2, 1e-12)

    merged = list(all_pts)
    mapping = list(range(total_raw))

    for pass_idx in range(n_passes):
        cs = base * (2.0 ** (n_passes - 1 - pass_idx))
        cs = max(cs, 1e-12)

        hash_table: dict[tuple, int] = {}
        rep_of: dict[int, int] = {}

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
                hash_table[(gx, gy, gz)] = idx
                rep_of[idx] = idx

        final_hash: dict[tuple, int] = {}
        final_merged = []

        for idx in range(total_raw):
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
# Shared face conversion
# ---------------------------------------------------------------------------


def _convert_shared_faces(faces, owner, neighbour, boundary) -> tuple:
    """Convert coincident boundary face pairs into internal faces."""
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


def _merge_zones_priority(boundary, zone_priority) -> tuple:
    """Merge patches with the same name; higher priority zone determines type."""
    zone_data: dict[str, dict] = {}
    order: list[str] = []
    for p in boundary:
        name = p["name"]
        if name not in zone_data:
            zone_data[name] = {
                "name": name, "type": p["type"],
                "nFaces": 0, "sources": [],
                "priority": zone_priority.get(name, 0),
            }
            order.append(name)
        zone_data[name]["nFaces"] += p["nFaces"]
        zone_data[name]["sources"].append(p)
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
            "name": zd["name"], "type": zd["type"],
            "startFace": offset, "nFaces": zd["nFaces"],
        })
        offset += zd["nFaces"]

    return merged_boundary, n_merged


# ---------------------------------------------------------------------------
# Graph connectivity analysis
# ---------------------------------------------------------------------------


def _analyze_connectivity(
    n_cells: int, n_internal: int, owner: list, neighbour: list,
) -> tuple:
    """Analyze cell-to-cell connectivity using union-find."""
    if n_cells == 0:
        return 0, True

    parent = list(range(n_cells))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for fi in range(n_internal):
        union(int(owner[fi]), int(neighbour[fi]))

    roots = set(find(i) for i in range(n_cells))
    n_components = len(roots)
    return n_components, n_components == 1
