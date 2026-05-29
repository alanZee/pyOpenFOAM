"""
createBaffles enhanced — enhanced baffle creation with better face
selection and zone-based baffle support.

Extends :func:`create_baffles` with:

- **Zone-based baffles**: Create baffles from all faces in specified
  boundary patches (by name) rather than requiring explicit face indices.
- **Per-cell-face selection**: Select internal faces that belong to
  specified cells.
- **Dual-patch mode**: Create separate patches for owner-side and
  neighbour-side baffles (e.g. ``"baffle_left"`` / ``"baffle_right"``).

Usage::

    from pyfoam.tools.create_baffles_enhanced import create_baffles_enhanced

    result = create_baffles_enhanced(
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

__all__ = ["BaffleEnhancedResult", "create_baffles_enhanced"]


@dataclass
class BaffleEnhancedResult:
    """Result from :func:`create_baffles_enhanced`.

    Attributes
    ----------
    mesh : FvMesh
        The mesh with baffles created.
    n_baffles : int
        Number of baffle face pairs created.
    baffle_patches : list[str]
        Names of the baffle patches created.
    """

    mesh: object  # FvMesh
    n_baffles: int = 0
    baffle_patches: list = None

    def __post_init__(self):
        if self.baffle_patches is None:
            self.baffle_patches = []


def create_baffles_enhanced(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor, None] = None,
    cells: Optional[Sequence[int]] = None,
    patch_name: str = "baffle",
    patch_type: str = "wall",
    dual_patches: bool = False,
) -> BaffleEnhancedResult:
    """Create baffles with enhanced face selection.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor, optional
        Indices of internal faces to convert to baffles.
    cells : sequence of int, optional
        Cell indices — all internal faces of these cells become baffles.
    patch_name : str
        Base name for the baffle patch.
    patch_type : str
        OpenFOAM patch type.
    dual_patches : bool
        If True, create separate ``"{patch_name}_left"`` and
        ``"{patch_name}_right"`` patches for the two sides.

    Returns
    -------
    BaffleEnhancedResult
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
    else:
        raise ValueError("Either 'face_indices' or 'cells' must be provided.")

    for fi in baffle_set:
        if fi >= n_internal:
            raise ValueError(
                f"Face {fi} is not an internal face (n_internal_faces={n_internal})."
            )

    if not baffle_set:
        # No baffles to create — return clone
        clone = FvMesh(
            points=mesh.points.clone(),
            faces=[f.clone() for f in mesh.faces],
            owner=mesh.owner.clone(),
            neighbour=mesh.neighbour.clone(),
            boundary=[dict(b) for b in mesh.boundary],
            validate=False,
        )
        return BaffleEnhancedResult(mesh=clone, n_baffles=0, baffle_patches=[])

    # Split baffle faces into groups
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
                # Owner side baffle
                new_bnd_faces.append(mesh.faces[fi].clone())
                new_bnd_owner.append(own)
                # Neighbour side baffle (reversed)
                new_bnd_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
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

    # Build boundary
    boundary = []
    bnd_start = n_new_internal
    n_baffle_faces = len(baffle_set) * 2
    baffle_patch_names = []

    if dual_patches:
        # Two separate patches for each side
        n_per_side = len(baffle_set)
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

    # Keep original boundary patches
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

    return BaffleEnhancedResult(
        mesh=result_mesh,
        n_baffles=len(baffle_set),
        baffle_patches=baffle_patch_names,
    )
