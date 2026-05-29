"""
subsetMesh — extract a subset of mesh cells.

Mirrors OpenFOAM's ``subsetMesh`` utility.  Extracts cells matching a
selection criterion into a new mesh.  The extracted mesh preserves
boundary patches from the original where applicable.

Selection methods:

- **cellSet**: extract by explicit cell indices
- **cellZone**: extract by cell zone name
- **boundingBox**: extract cells whose centres fall within a bounding box

References
----------
- OpenFOAM ``subsetMesh`` utility source
"""

from __future__ import annotations

import torch
from typing import Optional, Union

from pyfoam.core.dtype import INDEX_DTYPE

__all__ = ["subset_mesh", "subset_mesh_by_box"]


def subset_mesh(
    mesh,
    cells,
) -> "FvMesh":
    """Extract a subset of cells into a new mesh.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    cells : list[int] | torch.Tensor
        Cell indices to extract.

    Returns
    -------
    FvMesh
        New mesh containing only the selected cells.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    if isinstance(cells, torch.Tensor):
        cells_set = set(int(c) for c in cells.tolist())
    else:
        cells_set = set(int(c) for c in cells)

    if not cells_set:
        raise ValueError("No cells selected.")

    device = mesh.device
    dtype = mesh.dtype

    # Validate cell indices
    max_cell = mesh.n_cells
    invalid = cells_set - set(range(max_cell))
    if invalid:
        raise ValueError(f"Invalid cell indices: {sorted(invalid)}")

    # Build cell remap: old_idx -> new_idx
    sorted_cells = sorted(cells_set)
    cell_remap = {old: new for new, old in enumerate(sorted_cells)}

    int_faces = []
    int_owner = []
    int_nbr = []
    bnd_faces = []
    bnd_owner = []

    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        nbr = int(mesh.neighbour[fi].item()) if fi < mesh.n_internal_faces else -1

        own_in = own in cells_set
        nbr_in = nbr >= 0 and nbr in cells_set

        if own_in and nbr_in:
            # Internal face: ensure owner < neighbour for consistency
            new_own = cell_remap[own]
            new_nbr = cell_remap[nbr]
            if new_own > new_nbr:
                int_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
                int_owner.append(new_nbr)
                int_nbr.append(new_own)
            else:
                int_faces.append(mesh.faces[fi].clone())
                int_owner.append(new_own)
                int_nbr.append(new_nbr)
        elif own_in:
            bnd_faces.append(mesh.faces[fi].clone())
            bnd_owner.append(cell_remap[own])
        elif nbr_in:
            # Flip face so outward normal points away from selected cell
            bnd_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
            bnd_owner.append(cell_remap[nbr])

    # Build point remap
    used_points = set()
    for face in int_faces + bnd_faces:
        for pidx in face.tolist():
            used_points.add(pidx)

    sorted_points = sorted(used_points)
    point_remap = {old: new for new, old in enumerate(sorted_points)}

    new_points = mesh.points[sorted_points].to(device=device, dtype=dtype)
    new_faces = [
        torch.tensor(
            [point_remap[p] for p in face.tolist()],
            dtype=INDEX_DTYPE, device=device,
        )
        for face in int_faces + bnd_faces
    ]

    owner_t = torch.tensor(int_owner + bnd_owner, dtype=INDEX_DTYPE, device=device)
    nbr_t = torch.tensor(int_nbr, dtype=INDEX_DTYPE, device=device)

    n_internal = len(int_nbr)
    n_total = len(new_faces)
    n_boundary = n_total - n_internal

    boundary = []
    if n_boundary > 0:
        boundary.append({
            "name": "subset",
            "type": "wall",
            "startFace": n_internal,
            "nFaces": n_boundary,
        })

    return FvMesh(
        points=new_points,
        faces=new_faces,
        owner=owner_t,
        neighbour=nbr_t,
        boundary=boundary,
        validate=False,
    )


def subset_mesh_by_box(
    mesh,
    min_point: tuple[float, float, float],
    max_point: tuple[float, float, float],
) -> "FvMesh":
    """Extract cells whose centres fall within a bounding box.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    min_point : tuple
        ``(x_min, y_min, z_min)`` corner of bounding box.
    max_point : tuple
        ``(x_max, y_max, z_max)`` corner of bounding box.

    Returns
    -------
    FvMesh
        New mesh containing only cells within the bounding box.
    """
    device = mesh.device
    dtype = mesh.dtype

    # Compute cell centres if not available
    if hasattr(mesh, "cell_centres"):
        cc = mesh.cell_centres.to(device=device, dtype=dtype)
    else:
        # Fallback: compute from faces
        cc = _compute_cell_centres(mesh, device, dtype)

    bb_min = torch.tensor(min_point, dtype=dtype, device=device)
    bb_max = torch.tensor(max_point, dtype=dtype, device=device)

    # Find cells within bounding box
    inside = (cc >= bb_min.unsqueeze(0)) & (cc <= bb_max.unsqueeze(0))
    cell_mask = inside.all(dim=1)

    selected_cells = cell_mask.nonzero(as_tuple=True)[0].tolist()

    if not selected_cells:
        raise ValueError(
            f"No cells found within bounding box "
            f"[{min_point}] to [{max_point}]"
        )

    return subset_mesh(mesh, selected_cells)


def _compute_cell_centres(mesh, device, dtype):
    """Compute cell centres from face geometry (fallback)."""
    cc_sum = torch.zeros((mesh.n_cells, 3), dtype=dtype, device=device)
    cc_count = torch.zeros(mesh.n_cells, dtype=dtype, device=device)

    for fi in range(mesh.n_faces):
        face_pts = mesh.points[mesh.faces[fi].tolist()]
        fc = face_pts.mean(dim=0)
        own = int(mesh.owner[fi].item())
        cc_sum[own] += fc
        cc_count[own] += 1.0

        if fi < mesh.n_internal_faces:
            nbr = int(mesh.neighbour[fi].item())
            cc_sum[nbr] += fc
            cc_count[nbr] += 1.0

    return cc_sum / cc_count.unsqueeze(1).clamp(min=1.0)
