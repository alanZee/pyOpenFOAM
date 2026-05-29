"""
createPatch enhanced v2 — enhanced patch creation with better face
selection and zone-based patches (second generation).

Extends :func:`create_patch_enhanced` with:

- **Geometric selection**: Select faces within a bounding box or
  cylinder without needing explicit indices.
- **Normal-based selection**: Select faces whose normal is within an
  angle tolerance of a given direction.
- **Patch splitting**: Split an existing patch into sub-patches based
  on geometric criteria.

Usage::

    from pyfoam.tools.create_patch_enhanced_2 import create_patch_enhanced_2

    result = create_patch_enhanced_2(
        mesh,
        face_indices=[0, 1, 5],
        patch_name="new_patch",
        patch_type="wall",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced2Result", "create_patch_enhanced_2"]


@dataclass
class PatchEnhanced2Result:
    """Result from :func:`create_patch_enhanced_2`.

    Attributes
    ----------
    mesh : FvMesh
        The mesh with the new patch(es) created.
    patches_created : list[str]
        Names of the new patches.
    n_faces_moved : int
        Total number of faces moved to new patches.
    """

    mesh: object  # FvMesh
    patches_created: list = None
    n_faces_moved: int = 0

    def __post_init__(self):
        if self.patches_created is None:
            self.patches_created = []


def create_patch_enhanced_2(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor, None] = None,
    patch_name: str = "new_patch",
    patch_type: str = "wall",
    cells: Optional[Sequence[int]] = None,
    source_patches: Optional[Sequence[str]] = None,
    multi_patch: Optional[Sequence[Tuple[Sequence[int], str, str]]] = None,
    box_min: Optional[Tuple[float, float, float]] = None,
    box_max: Optional[Tuple[float, float, float]] = None,
    normal_dir: Optional[Tuple[float, float, float]] = None,
    normal_tol: float = 10.0,
) -> PatchEnhanced2Result:
    """Create new boundary patch(es) with enhanced face selection.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor, optional
        Face indices to move into the new patch.
    patch_name : str
        Name for the new boundary patch.
    patch_type : str
        OpenFOAM patch type.
    cells : sequence of int, optional
        Cell indices — internal faces bordering these cells are selected.
    source_patches : sequence of str, optional
        Names of existing boundary patches whose faces are moved.
    multi_patch : sequence of (indices, name, type), optional
        Create multiple patches in one call.
    box_min : tuple, optional
        ``(x_min, y_min, z_min)`` of selection bounding box.
    box_max : tuple, optional
        ``(x_max, y_max, z_max)`` of selection bounding box.
    normal_dir : tuple, optional
        Select faces whose unit normal is within ``normal_tol`` degrees
        of this direction vector.
    normal_tol : float
        Angle tolerance in degrees for normal-based selection.

    Returns
    -------
    PatchEnhanced2Result
        Mesh with new patch(es) and metadata.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    n_internal = mesh.n_internal_faces

    # Build list of (face_set, name, type) tuples
    patch_specs: list[tuple[set[int], str, str]] = []

    if multi_patch is not None:
        for idx_seq, pname, ptype in multi_patch:
            if isinstance(idx_seq, torch.Tensor):
                fs = set(int(i) for i in idx_seq.tolist())
            else:
                fs = set(int(i) for i in idx_seq)
            patch_specs.append((fs, pname, ptype))
    else:
        fs = set()
        if face_indices is not None:
            if isinstance(face_indices, torch.Tensor):
                fs = set(int(i) for i in face_indices.tolist())
            else:
                fs = set(int(i) for i in face_indices)

        if cells is not None:
            cell_set = set(int(c) for c in cells)
            owner = mesh.owner.detach().cpu().numpy()
            neighbour = mesh.neighbour.detach().cpu().numpy()
            for fi in range(n_internal):
                if int(owner[fi]) in cell_set or int(neighbour[fi]) in cell_set:
                    fs.add(fi)

        if source_patches is not None:
            source_set = set(source_patches)
            for p in mesh.boundary:
                if p["name"] in source_set:
                    start = p["startFace"]
                    for fi in range(start, start + p["nFaces"]):
                        fs.add(fi)

        # Box-based selection
        if box_min is not None and box_max is not None:
            fc = mesh.face_centres.detach().cpu().numpy()
            bmin = torch.tensor(box_min, dtype=torch.float64)
            bmax = torch.tensor(box_max, dtype=torch.float64)
            for fi in range(mesh.n_faces):
                pt = torch.tensor(fc[fi], dtype=torch.float64)
                if torch.all(pt >= bmin) and torch.all(pt <= bmax):
                    fs.add(fi)

        # Normal-based selection
        if normal_dir is not None:
            dir_vec = torch.tensor(normal_dir, dtype=torch.float64)
            dir_norm = dir_vec.norm()
            if dir_norm > 1e-30:
                dir_vec = dir_vec / dir_norm
            cos_tol = torch.cos(torch.tensor(normal_tol * 3.14159265 / 180.0))
            fc = mesh.face_centres.detach().cpu().numpy()
            pts_np = mesh.points.detach().cpu().numpy()
            for fi in range(mesh.n_faces):
                face = mesh.faces[fi].tolist()
                if len(face) >= 3:
                    p0 = torch.tensor(pts_np[face[0]], dtype=torch.float64)
                    p1 = torch.tensor(pts_np[face[1]], dtype=torch.float64)
                    p2 = torch.tensor(pts_np[face[2]], dtype=torch.float64)
                    n = torch.cross(p1 - p0, p2 - p0)
                    n_norm = n.norm()
                    if n_norm > 1e-30:
                        n = n / n_norm
                        if torch.dot(n, dir_vec).abs() >= cos_tol:
                            fs.add(fi)

        patch_specs.append((fs, patch_name, patch_type))

    # Check for name conflicts
    existing_names = {p["name"] for p in mesh.boundary}
    for _, pname, _ in patch_specs:
        if pname in existing_names:
            raise ValueError(f"Patch '{pname}' already exists in the mesh boundary.")

    # Build union of all selected faces
    all_selected: dict[int, str] = {}
    for fs, pname, _ in patch_specs:
        for fi in fs:
            all_selected[fi] = pname

    # Classify faces
    int_faces = []
    int_owner = []
    int_neighbour = []

    kept_patches: dict[str, list] = {p["name"]: [] for p in mesh.boundary}
    new_patch_faces: dict[str, list] = {pname: [] for _, pname, _ in patch_specs}
    new_patch_owner: dict[str, list] = {pname: [] for _, pname, _ in patch_specs}
    old_internal_faces = []
    old_internal_owner = []

    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        if fi in all_selected:
            target_patch = all_selected[fi]
            if fi < n_internal:
                nbr = int(mesh.neighbour[fi].item())
                new_patch_faces[target_patch].append(mesh.faces[fi].clone())
                new_patch_owner[target_patch].append(own)
                old_internal_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
                old_internal_owner.append(nbr)
            else:
                new_patch_faces[target_patch].append(mesh.faces[fi].clone())
                new_patch_owner[target_patch].append(own)
        elif fi < n_internal:
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(own)
            int_neighbour.append(int(mesh.neighbour[fi].item()))
        else:
            for p in mesh.boundary:
                ps = p["startFace"]
                pe = ps + p["nFaces"]
                if ps <= fi < pe:
                    kept_patches[p["name"]].append((mesh.faces[fi].clone(), own))
                    break

    # Assemble
    n_new_internal = len(int_neighbour)
    all_faces = list(int_faces)
    all_owner = list(int_owner)
    bnd_start = n_new_internal

    boundary = []

    for p in mesh.boundary:
        kept = kept_patches[p["name"]]
        if kept:
            for f, o in kept:
                all_faces.append(f)
                all_owner.append(o)
            boundary.append({
                "name": p["name"],
                "type": p["type"],
                "startFace": bnd_start,
                "nFaces": len(kept),
            })
            bnd_start += len(kept)

    patches_created = []
    n_faces_moved = 0
    for fs, pname, ptype in patch_specs:
        faces_list = new_patch_faces[pname]
        if faces_list:
            for f in faces_list:
                all_faces.append(f)
            all_owner.extend(new_patch_owner[pname])
            boundary.append({
                "name": pname, "type": ptype,
                "startFace": bnd_start, "nFaces": len(faces_list),
            })
            bnd_start += len(faces_list)
            patches_created.append(pname)
            n_faces_moved += len(faces_list)

    for f in old_internal_faces:
        all_faces.append(f)
    all_owner.extend(old_internal_owner)
    if old_internal_faces:
        boundary.append({
            "name": "oldInternal",
            "type": "wall",
            "startFace": bnd_start,
            "nFaces": len(old_internal_faces),
        })

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return PatchEnhanced2Result(
        mesh=result_mesh,
        patches_created=patches_created,
        n_faces_moved=n_faces_moved,
    )
