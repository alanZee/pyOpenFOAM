"""
createBaffles — create baffle faces from internal faces.

Mirrors OpenFOAM's ``createBaffles`` utility.  Converts selected internal
faces into boundary (baffle) faces by duplicating them and assigning each
copy to a new boundary patch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["create_baffles"]


def create_baffles(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor],
    patch_name: str = "baffle",
    patch_type: str = "wall",
) -> "FvMesh":
    """Create baffle faces from selected internal faces.

    Each selected internal face is converted into two boundary faces (one for
    each side of the original face).  The resulting mesh has no internal face
    at those locations; instead, thin-wall baffle patches are created.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor
        Indices of internal faces to convert to baffles.  Indices must be
        less than ``mesh.n_internal_faces``.
    patch_name : str
        Name for the new baffle boundary patch.
    patch_type : str
        OpenFOAM patch type for the baffle (default ``"wall"``).

    Returns
    -------
    FvMesh
        New mesh with baffles created.

    Raises
    ------
    ValueError
        If any face index is not an internal face.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    dt = mesh.dtype

    if isinstance(face_indices, torch.Tensor):
        baffle_set = set(int(i) for i in face_indices.tolist())
    else:
        baffle_set = set(int(i) for i in face_indices)

    n_internal = mesh.n_internal_faces
    for fi in baffle_set:
        if fi >= n_internal:
            raise ValueError(
                f"Face {fi} is not an internal face (n_internal_faces={n_internal})."
            )

    new_int_faces = []
    new_int_owner = []
    new_int_neighbour = []
    new_bnd_faces = []
    new_bnd_owner = []

    for fi in range(mesh.n_faces):
        if fi < n_internal:
            if fi in baffle_set:
                # 转为两个边界面：owner 侧和 neighbour 侧
                own = int(mesh.owner[fi].item())
                nbr = int(mesh.neighbour[fi].item())
                # owner 侧 baffle（保持原方向）
                new_bnd_faces.append(mesh.faces[fi].clone())
                new_bnd_owner.append(own)
                # neighbour 侧 baffle（反转方向）
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

    # 构建 boundary 描述
    boundary = []
    bnd_start = n_new_internal
    n_baffle_faces = len(baffle_set) * 2

    # baffle patch 排在最前面
    if n_baffle_faces > 0:
        boundary.append({
            "name": patch_name,
            "type": patch_type,
            "startFace": bnd_start,
            "nFaces": n_baffle_faces,
        })
        bnd_start += n_baffle_faces

    # 原有 boundary patches（更新 startFace 和 nFaces）
    for patch in mesh.boundary:
        orig_start = patch["startFace"]
        orig_end = orig_start + patch["nFaces"]
        # 计算该 patch 中被移除的面数
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

    return FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(new_int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )
