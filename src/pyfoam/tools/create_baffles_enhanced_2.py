"""
createBaffles enhanced v2 — enhanced baffle creation with better face
selection and zone-based baffle support (second generation).

Extends :func:`create_baffles_enhanced` with:

- **Zone-based source patches**: Create baffles from all faces in
  specified existing patches (by name).
- **Minimum area filter**: Only create baffles for faces above a
  minimum area threshold.
- **Triangulation option**: Optionally triangulate non-triangular
  baffle faces for better solver compatibility.

Usage::

    from pyfoam.tools.create_baffles_enhanced_2 import create_baffles_enhanced_2

    result = create_baffles_enhanced_2(
        mesh,
        face_indices=[0, 1, 2],
        patch_name="baffle",
        dual_patches=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced2Result", "create_baffles_enhanced_2"]


@dataclass
class BaffleEnhanced2Result:
    """Result from :func:`create_baffles_enhanced_2`.

    Attributes
    ----------
    mesh : FvMesh
        The mesh with baffles created.
    n_baffles : int
        Number of baffle face pairs created.
    baffle_patches : list[str]
        Names of the baffle patches created.
    n_filtered : int
        Number of faces filtered out by area threshold.
    """

    mesh: object  # FvMesh
    n_baffles: int = 0
    baffle_patches: list = None
    n_filtered: int = 0

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced_2(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor, None] = None,
    cells: Optional[Sequence[int]] = None,
    source_patches: Optional[Sequence[str]] = None,
    patch_name: str = "baffle",
    patch_type: str = "wall",
    dual_patches: bool = False,
    min_area: float = 0.0,
    triangulate: bool = False,
) -> BaffleEnhanced2Result:
    """Create baffles with zone-based selection and area filtering.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor, optional
        Indices of internal faces to convert to baffles.
    cells : sequence of int, optional
        Cell indices — all internal faces of these cells become baffles.
    source_patches : sequence of str, optional
        Names of existing boundary patches whose faces become baffles.
    patch_name : str
        Base name for the baffle patch.
    patch_type : str
        OpenFOAM patch type.
    dual_patches : bool
        If True, create separate ``"{patch_name}_left"`` and
        ``"{patch_name}_right"`` patches.
    min_area : float
        Minimum face area to create a baffle. Faces below this are skipped.
    triangulate : bool
        If True, triangulate non-triangular baffle faces.

    Returns
    -------
    BaffleEnhanced2Result
        Mesh with baffles and metadata.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    n_internal = mesh.n_internal_faces

    # Determine face set
    if face_indices is not None:
        if isinstance(face_indices, torch.Tensor):
            baffle_set = set(int(i) for i in face_indices.tolist())
        else:
            baffle_set = set(int(i) for i in face_indices)
    elif cells is not None:
        cell_set = set(int(c) for c in cells)
        baffle_set = set()
        owner = mesh.owner.detach().cpu().numpy()
        neighbour = mesh.neighbour.detach().cpu().numpy()
        for fi in range(n_internal):
            if int(owner[fi]) in cell_set or int(neighbour[fi]) in cell_set:
                baffle_set.add(fi)
    elif source_patches is not None:
        source_set = set(source_patches)
        baffle_set = set()
        for p in mesh.boundary:
            if p["name"] in source_set:
                start = p["startFace"]
                for fi in range(start, start + p["nFaces"]):
                    baffle_set.add(fi)
    else:
        raise ValueError("One of 'face_indices', 'cells', or 'source_patches' must be provided.")

    for fi in baffle_set:
        if fi >= n_internal:
            raise ValueError(
                f"Face {fi} is not an internal face (n_internal_faces={n_internal})."
            )

    # Area filtering
    n_filtered = 0
    if min_area > 0:
        filtered = set()
        for fi in baffle_set:
            pts = mesh.points[mesh.faces[fi]].float()
            if pts.shape[0] >= 3:
                cross = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
                area = 0.5 * cross.norm().item()
                if area >= min_area:
                    filtered.add(fi)
                else:
                    n_filtered += 1
            else:
                filtered.add(fi)
        baffle_set = filtered

    if not baffle_set:
        clone = FvMesh(
            points=mesh.points.clone(),
            faces=[f.clone() for f in mesh.faces],
            owner=mesh.owner.clone(),
            neighbour=mesh.neighbour.clone(),
            boundary=[dict(b) for b in mesh.boundary],
            validate=False,
        )
        return BaffleEnhanced2Result(mesh=clone, n_baffles=0, baffle_patches=[],
                                     n_filtered=n_filtered)

    # Build baffles
    new_int_faces = []
    new_int_owner = []
    new_int_neighbour = []
    new_bnd_faces = []
    new_bnd_owner = []

    for fi in range(mesh.n_faces):
        if fi < n_internal:
            if fi in baffle_set:
                own = int(mesh.owner[fi].item())
                nbr = int(mesh.neighbour[fi].item())
                face = mesh.faces[fi]
                if triangulate and face.shape[0] > 3:
                    # Fan triangulation
                    pts_list = face.tolist()
                    for k in range(1, len(pts_list) - 1):
                        tri = torch.tensor([pts_list[0], pts_list[k], pts_list[k + 1]],
                                           dtype=INDEX_DTYPE, device=dev)
                        new_bnd_faces.append(tri)
                        new_bnd_owner.append(own)
                        new_bnd_faces.append(torch.flip(tri, dims=[0]))
                        new_bnd_owner.append(nbr)
                else:
                    new_bnd_faces.append(face.clone())
                    new_bnd_owner.append(own)
                    new_bnd_faces.append(torch.flip(face, dims=[0]).clone())
                    new_bnd_owner.append(nbr)
            else:
                new_int_faces.append(mesh.faces[fi].clone())
                new_int_owner.append(int(mesh.owner[fi].item()))
                new_int_neighbour.append(int(mesh.neighbour[fi].item()))
        else:
            new_bnd_faces.append(mesh.faces[fi].clone())
            new_bnd_owner.append(int(mesh.owner[fi].item()))

    n_new_internal = len(new_int_neighbour)
    all_faces = new_int_faces + new_bnd_faces
    all_owner = new_int_owner + new_bnd_owner

    boundary = []
    bnd_start = n_new_internal
    n_baffle_faces = len(new_bnd_faces) - sum(p["nFaces"] for p in mesh.boundary)
    baffle_patch_names = []

    if dual_patches:
        n_per_side = n_baffle_faces // 2
        left_name = f"{patch_name}_left"
        right_name = f"{patch_name}_right"
        boundary.append({
            "name": left_name, "type": patch_type,
            "startFace": bnd_start, "nFaces": n_per_side,
        })
        bnd_start += n_per_side
        boundary.append({
            "name": right_name, "type": patch_type,
            "startFace": bnd_start, "nFaces": n_per_side,
        })
        bnd_start += n_per_side
        baffle_patch_names = [left_name, right_name]
    else:
        if n_baffle_faces > 0:
            boundary.append({
                "name": patch_name, "type": patch_type,
                "startFace": bnd_start, "nFaces": n_baffle_faces,
            })
            bnd_start += n_baffle_faces
            baffle_patch_names = [patch_name]

    for patch in mesh.boundary:
        orig_start = patch["startFace"]
        orig_end = orig_start + patch["nFaces"]
        removed = sum(1 for fi in range(orig_start, orig_end) if fi in baffle_set)
        new_n = patch["nFaces"] - removed
        if new_n > 0:
            boundary.append({
                "name": patch["name"],
                "type": patch["type"],
                "startFace": bnd_start,
                "nFaces": new_n,
            })
            bnd_start += new_n

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(new_int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return BaffleEnhanced2Result(
        mesh=result_mesh,
        n_baffles=len(baffle_set),
        baffle_patches=baffle_patch_names,
        n_filtered=n_filtered,
    )
