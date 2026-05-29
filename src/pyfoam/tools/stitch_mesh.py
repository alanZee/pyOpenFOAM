"""
stitchMesh — stitch two boundary patches together by merging coincident faces.

Mirrors OpenFOAM's ``stitchMesh`` utility.  Identifies face pairs across two
patches whose vertices are within a geometric tolerance, then converts those
boundary faces into internal faces (owner–neighbour pairs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["stitch_mesh"]


def stitch_mesh(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
) -> "FvMesh":
    """Stitch two boundary patches into internal faces.

    For each boundary face on *patch1*, the algorithm searches *patch2* for a
    matching face whose vertex positions coincide (within *tolerance*).  Matched
    pairs are converted to internal faces; unmatched boundary faces are kept.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    patch1_name : str
        Name of the first boundary patch.
    patch2_name : str
        Name of the second boundary patch.
    tolerance : float
        Maximum distance between corresponding vertices for a match.

    Returns
    -------
    FvMesh
        New mesh with stitched patches converted to internal faces.

    Raises
    ------
    ValueError
        If either patch name is not found in the mesh boundary.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    dt = mesh.dtype

    # 定位两个 patch
    p1 = _find_patch(mesh, patch1_name)
    p2 = _find_patch(mesh, patch2_name)

    p1_start = p1["startFace"]
    p1_end = p1_start + p1["nFaces"]
    p2_start = p2["startFace"]
    p2_end = p2_start + p2["nFaces"]

    # 构建 patch2 面中心索引，用于快速查找
    p2_face_centres = torch.stack([
        mesh.points[mesh.faces[fi]].float().mean(dim=0) for fi in range(p2_start, p2_end)
    ])

    # 匹配 patch1 → patch2
    matched_p2 = set()
    stitch_pairs = []  # (p1_face_idx, p2_face_idx)

    for fi1 in range(p1_start, p1_end):
        pts1 = mesh.points[mesh.faces[fi1]].float()
        fc1 = pts1.mean(dim=0)
        # 计算与 patch2 所有面中心的距离
        dists = (p2_face_centres - fc1.unsqueeze(0)).norm(dim=1)
        candidates = (dists < tolerance).nonzero(as_tuple=False).squeeze(1)

        for ci in candidates:
            fi2 = p2_start + int(ci.item())
            if fi2 in matched_p2:
                continue
            if _faces_coincide(mesh, fi1, fi2, tolerance):
                stitch_pairs.append((fi1, fi2))
                matched_p2.add(fi2)
                break

    # 分类面：internal、stitched（新 internal）、boundary（保留）
    int_faces = []
    int_owner = []
    int_neighbour = []
    bnd_faces = []
    bnd_owner = []

    stitched_set_p1 = {p for p, _ in stitch_pairs}
    stitched_set_p2 = {p for _, p in stitch_pairs}

    for fi in range(mesh.n_faces):
        if fi < mesh.n_internal_faces:
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(int(mesh.owner[fi].item()))
            int_neighbour.append(int(mesh.neighbour[fi].item()))
        elif fi in stitched_set_p1 or fi in stitched_set_p2:
            # stitched 面不加入 boundary（将在后面转为 internal）
            continue
        else:
            bnd_faces.append(mesh.faces[fi].clone())
            bnd_owner.append(int(mesh.owner[fi].item()))

    # 添加新 internal faces（stitched 对）
    for fi1, fi2 in stitch_pairs:
        own = int(mesh.owner[fi1].item())
        nbr = int(mesh.owner[fi2].item())
        int_faces.append(mesh.faces[fi1].clone())
        if own > nbr:
            own, nbr = nbr, own
        int_owner.append(own)
        int_neighbour.append(nbr)

    n_internal = len(int_neighbour)
    all_faces = int_faces + bnd_faces
    all_owner = int_owner + bnd_owner

    # 构建 boundary 描述
    boundary = []
    existing_patches = [p for p in mesh.boundary
                        if p["name"] != patch1_name and p["name"] != patch2_name]
    bnd_start = n_internal
    for patch in existing_patches:
        # 统计该 patch 保留了多少面
        orig_start = patch["startFace"]
        orig_end = orig_start + patch["nFaces"]
        count = sum(1 for fi in range(orig_start, orig_end)
                    if fi not in stitched_set_p1 and fi not in stitched_set_p2)
        if count > 0:
            boundary.append({
                "name": patch["name"],
                "type": patch["type"],
                "startFace": bnd_start,
                "nFaces": count,
            })
            bnd_start += count

    return FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )


def _find_patch(mesh: "FvMesh", name: str) -> dict:
    """查找指定名称的 boundary patch。"""
    for patch in mesh.boundary:
        if patch["name"] == name:
            return patch
    raise ValueError(f"Patch '{name}' not found in mesh boundary.")


def _faces_coincide(mesh: "FvMesh", fi1: int, fi2: int, tol: float) -> bool:
    """判断两个面的顶点是否在容差内一致。"""
    pts1 = mesh.points[mesh.faces[fi1]].float()
    pts2 = mesh.points[mesh.faces[fi2]].float()

    if pts1.shape[0] != pts2.shape[0]:
        return False

    # 尝试所有循环偏移和正/反转匹配
    n = pts1.shape[0]
    for offset in range(n):
        for sign in [1, -1]:
            if sign == 1:
                reordered = torch.roll(pts2, shifts=offset, dims=0)
            else:
                reordered = torch.roll(pts2.flip(0), shifts=offset, dims=0)
            dist = (pts1 - reordered).norm(dim=1).max().item()
            if dist < tol:
                return True
    return False
