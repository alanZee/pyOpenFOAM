"""
createPatch — create a new boundary patch from selected faces.

Mirrors OpenFOAM's ``createPatch`` utility.  Moves faces from their current
patch (or converts internal faces) into a new named boundary patch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["create_patch"]


def create_patch(
    mesh: "FvMesh",
    face_indices: Union[Sequence[int], torch.Tensor],
    patch_name: str,
    patch_type: str = "wall",
) -> "FvMesh":
    """Create a new boundary patch from selected faces.

    Selected faces are removed from their current location (internal or
    boundary) and re-assigned to a new boundary patch named *patch_name*.
    Internal faces are split: the owner-side copy goes to the new patch
    while the neighbour-side copy goes to a separate ``oldInternal`` patch
    (mirroring OpenFOAM behaviour).

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    face_indices : sequence of int or Tensor
        Face indices to move into the new patch.
    patch_name : str
        Name for the new boundary patch.
    patch_type : str
        OpenFOAM patch type (default ``"wall"``).

    Returns
    -------
    FvMesh
        New mesh with the face set moved to a new boundary patch.

    Raises
    ------
    ValueError
        If *patch_name* already exists in the mesh.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    dt = mesh.dtype

    # 检查名称冲突
    for p in mesh.boundary:
        if p["name"] == patch_name:
            raise ValueError(
                f"Patch '{patch_name}' already exists in the mesh boundary."
            )

    if isinstance(face_indices, torch.Tensor):
        selected = set(int(i) for i in face_indices.tolist())
    else:
        selected = set(int(i) for i in face_indices)

    n_internal = mesh.n_internal_faces

    # 分类输出面
    int_faces = []
    int_owner = []
    int_neighbour = []

    # 保留的原有 boundary faces，按 patch 分组
    kept_patches = {p["name"]: [] for p in mesh.boundary}

    # 新 patch 的 faces
    new_patch_faces = []
    new_patch_owner = []

    # 如果选中了 internal face，需要在 owner 侧留下边界，在 neighbour 侧加入 oldInternal
    old_internal_faces = []
    old_internal_owner = []

    for fi in range(mesh.n_faces):
        own = int(mesh.owner[fi].item())
        if fi in selected:
            if fi < n_internal:
                nbr = int(mesh.neighbour[fi].item())
                # owner 侧 → 新 patch
                new_patch_faces.append(mesh.faces[fi].clone())
                new_patch_owner.append(own)
                # neighbour 侧 → oldInternal
                old_internal_faces.append(torch.flip(mesh.faces[fi], dims=[0]).clone())
                old_internal_owner.append(nbr)
            else:
                # 边界面 → 移动到新 patch
                new_patch_faces.append(mesh.faces[fi].clone())
                new_patch_owner.append(own)
        elif fi < n_internal:
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(own)
            int_neighbour.append(int(mesh.neighbour[fi].item()))
        else:
            # 归属原始 patch
            for p in mesh.boundary:
                ps = p["startFace"]
                pe = ps + p["nFaces"]
                if ps <= fi < pe:
                    kept_patches[p["name"]].append((mesh.faces[fi].clone(), own))
                    break

    # 组装
    n_new_internal = len(int_neighbour)
    all_faces = list(int_faces)
    all_owner = list(int_owner)
    bnd_start = n_new_internal

    boundary = []

    # 遍历原有 patches（保持顺序）
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

    # new patch
    for f in new_patch_faces:
        all_faces.append(f)
    all_owner.extend(new_patch_owner)
    if new_patch_faces:
        boundary.append({
            "name": patch_name,
            "type": patch_type,
            "startFace": bnd_start,
            "nFaces": len(new_patch_faces),
        })
        bnd_start += len(new_patch_faces)

    # oldInternal（如有从 internal 转来的面）
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

    return FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )
