"""
splitMeshRegions -- split mesh into sub-meshes by cellZone.
"""
from __future__ import annotations
import torch
from pyfoam.core.dtype import INDEX_DTYPE
__all__ = ["split_mesh_regions"]

def split_mesh_regions(mesh, zones):
    from pyfoam.mesh.fv_mesh import FvMesh
    zone_cells = {}
    for name, idxs in zones.items():
        if isinstance(idxs, torch.Tensor): zone_cells[name] = set(int(i) for i in idxs.tolist())
        else: zone_cells[name] = set(int(i) for i in idxs)
    seen = set()
    for name, cs in zone_cells.items():
        overlap = seen & cs
        if overlap: raise ValueError(f"Zones overlap at cells: {sorted(overlap)}")
        seen |= cs
    if not zone_cells: return {}
    result = {}
    for zone_name, cells_set in zone_cells.items():
        if not cells_set: continue
        result[zone_name] = _extract_zone(mesh, zone_name, cells_set)
    return result

def _extract_zone(mesh, zone_name, cells_set):
    from pyfoam.mesh.fv_mesh import FvMesh
    device = mesh.device; dtype = mesh.dtype
    sorted_cells = sorted(cells_set)
    cell_remap = {old: new for new, old in enumerate(sorted_cells)}
    int_faces = []; int_owner = []; int_nbr = []
    bnd_faces = []; bnd_owner = []
    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        nbr = int(mesh.neighbour[fi].item()) if fi < mesh.n_internal_faces else -1
        own_in = own in cells_set; nbr_in = nbr >= 0 and nbr in cells_set
        if own_in and nbr_in:
            if cell_remap[own] > cell_remap[nbr]:
                int_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
                int_owner.append(cell_remap[nbr]); int_nbr.append(cell_remap[own])
            else:
                int_faces.append(mesh.faces[fi].clone())
                int_owner.append(cell_remap[own]); int_nbr.append(cell_remap[nbr])
        elif own_in:
            bnd_faces.append(mesh.faces[fi].clone()); bnd_owner.append(cell_remap[own])
        elif nbr_in:
            bnd_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone()); bnd_owner.append(cell_remap[nbr])
    sub_faces = int_faces + bnd_faces; sub_owner = int_owner + bnd_owner; sub_neighbour = int_nbr
    used_points = set()
    for face in sub_faces:
        for pidx in face.tolist(): used_points.add(pidx)
    sorted_points = sorted(used_points)
    point_remap = {old: new for new, old in enumerate(sorted_points)}
    new_points = mesh.points[sorted_points].to(device=device, dtype=dtype)
    new_faces = [torch.tensor([point_remap[p] for p in face.tolist()], dtype=INDEX_DTYPE, device=device) for face in sub_faces]
    owner_t = torch.tensor(sub_owner, dtype=INDEX_DTYPE, device=device)
    nbr_t = torch.tensor(sub_neighbour, dtype=INDEX_DTYPE, device=device)
    n_internal = nbr_t.shape[0]; n_total = len(new_faces); n_boundary = n_total - n_internal
    boundary = []
    if n_boundary > 0:
        boundary.append({"name": zone_name, "type": "wall", "startFace": n_internal, "nFaces": n_boundary})
    return FvMesh(points=new_points, faces=new_faces, owner=owner_t, neighbour=nbr_t, boundary=boundary, validate=False)
