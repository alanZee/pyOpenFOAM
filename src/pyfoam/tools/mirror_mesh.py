"""
mirrorMesh — mirror a mesh about a plane.

Mirrors OpenFOAM's ``mirrorMesh`` utility.  Reflects all mesh points about a
plane defined by a point on the plane and its normal vector.  The mesh
topology is duplicated and flipped to produce a valid merged mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["mirror_mesh"]


def mirror_mesh(
    mesh: "FvMesh",
    plane_normal: Union[Sequence[float], torch.Tensor],
    plane_point: Union[Sequence[float], torch.Tensor],
) -> "FvMesh":
    """Mirror a mesh about a plane.

    The original mesh is combined with its mirror image.  For each face in
    the mirror copy, the vertex ordering is reversed to maintain the
    owner–neighbour convention.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    plane_normal : sequence of 3 floats or Tensor
        Normal vector of the mirror plane.
    plane_point : sequence of 3 floats or Tensor
        A point on the mirror plane.

    Returns
    -------
    FvMesh
        New mesh containing the original and mirrored halves.

    Raises
    ------
    ValueError
        If ``plane_normal`` is a zero vector.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    dt = mesh.dtype

    normal = torch.as_tensor(plane_normal, dtype=torch.float64, device="cpu").flatten()
    n_mag = normal.norm()
    if n_mag < 1e-30:
        raise ValueError("plane_normal must be a non-zero vector")
    normal = normal / n_mag

    point = torch.as_tensor(plane_point, dtype=torch.float64, device="cpu").flatten()
    if point.numel() != 3:
        raise ValueError("plane_point must have exactly 3 elements")

    # --- 反射原始点 ---
    orig_pts = mesh.points.clone().cpu().to(torch.float64)
    # 反射公式: p' = p - 2 * dot(p - plane_point, normal) * normal
    dp = orig_pts - point.unsqueeze(0)
    dist = (dp * normal.unsqueeze(0)).sum(dim=1, keepdim=True)
    mirrored_pts = orig_pts - 2.0 * dist * normal.unsqueeze(0)

    # --- 合并点 ---
    n_orig = orig_pts.shape[0]
    all_points = torch.cat([orig_pts, mirrored_pts], dim=0).to(device=dev, dtype=dt)

    # --- 构建镜像面 ---
    n_cells_orig = mesh.n_cells
    n_internal_orig = mesh.n_internal_faces

    orig_faces = [f.clone() for f in mesh.faces]
    orig_owner = mesh.owner.clone()
    orig_neighbour = mesh.neighbour.clone()

    # 镜像面：反转顶点顺序，cell index 偏移
    mirror_faces = []
    mirror_owner_list = []
    mirror_neighbour_list = []
    mirror_bnd_faces = []
    mirror_bnd_owner = []

    for fi in range(mesh.n_faces):
        # 镜像面的点索引 = 原始索引 + n_orig
        mirrored_face = (mesh.faces[fi] + n_orig).clone()
        # 反转顶点顺序以保持法线方向正确
        mirrored_face = torch.flip(mirrored_face, dims=[0])
        own = int(mesh.owner[fi].item()) + n_cells_orig

        if fi < n_internal_orig:
            nbr = int(mesh.neighbour[fi].item()) + n_cells_orig
            # 确保 owner < neighbour
            if own > nbr:
                mirrored_face = torch.flip(mirrored_face, dims=[0])
                own, nbr = nbr, own
            mirror_faces.append(mirrored_face)
            mirror_owner_list.append(own)
            mirror_neighbour_list.append(nbr)
        else:
            mirror_bnd_faces.append(mirrored_face)
            mirror_bnd_owner.append(own)

    # --- 合并 internal faces ---
    int_faces = []
    int_owner = []
    int_neighbour = []

    # 原始 internal faces
    for fi in range(n_internal_orig):
        int_faces.append(orig_faces[fi])
        int_owner.append(int(orig_owner[fi].item()))
        int_neighbour.append(int(orig_neighbour[fi].item()))

    # 镜像 internal faces
    int_faces.extend(mirror_faces)
    int_owner.extend(mirror_owner_list)
    int_neighbour.extend(mirror_neighbour_list)

    # --- 查找匹配的对称面并生成新 internal faces ---
    # 对于原始 boundary faces 中面心在镜像平面上的那些，与镜像副本配对
    sym_tolerance = 1e-6
    bnd_faces_orig = []
    bnd_owner_orig = []
    used_mirror_bnd = set()

    for fi in range(mesh.n_faces):
        if fi >= n_internal_orig:
            bnd_faces_orig.append(fi)
            bnd_owner_orig.append(int(mesh.owner[fi].item()))

    # 找对称配对
    new_internal_pairs = []
    mirror_bnd_pts = []
    for i, (mf, mo) in enumerate(zip(mirror_bnd_faces, mirror_bnd_owner)):
        mirror_bnd_pts.append(mesh.points.cpu().to(torch.float64)[mf - n_orig].mean(dim=0))
    mirror_bnd_pts_tensor = torch.stack(mirror_bnd_pts) if mirror_bnd_pts else torch.empty((0, 3))

    for fi in bnd_faces_orig:
        orig_fc = mesh.points.cpu().to(torch.float64)[mesh.faces[fi]].mean(dim=0)
        # 反射面中心
        dp2 = orig_fc - point
        dist2 = (dp2 * normal).sum()
        reflected_fc = orig_fc - 2.0 * dist2 * normal

        if mirror_bnd_pts_tensor.numel() > 0:
            dists = (mirror_bnd_pts_tensor - reflected_fc.unsqueeze(0)).norm(dim=1)
            best = dists.min()
            best_idx = int(dists.argmin().item())
            if best.item() < sym_tolerance and best_idx not in used_mirror_bnd:
                used_mirror_bnd.add(best_idx)
                orig_own = int(mesh.owner[fi].item())
                mir_own = mirror_bnd_owner[best_idx]
                new_internal_pairs.append((fi, best_idx, orig_own, mir_own))

    for orig_fi, mir_idx, orig_own, mir_own in new_internal_pairs:
        int_faces.append(orig_faces[orig_fi])
        o, n = min(orig_own, mir_own), max(orig_own, mir_own)
        int_owner.append(o)
        int_neighbour.append(n)

    # --- 合并 boundary faces ---
    all_bnd_faces = []
    all_bnd_owner = []

    for fi in bnd_faces_orig:
        skip = False
        for ofi, _, _, _ in new_internal_pairs:
            if ofi == fi:
                skip = True
                break
        if not skip:
            all_bnd_faces.append(orig_faces[fi])
            all_bnd_owner.append(int(mesh.owner[fi].item()))

    for i, (mf, mo) in enumerate(zip(mirror_bnd_faces, mirror_bnd_owner)):
        if i not in used_mirror_bnd:
            all_bnd_faces.append(mf)
            all_bnd_owner.append(mo)

    # --- 组装 ---
    n_new_internal = len(int_neighbour)
    all_faces = int_faces + all_bnd_faces
    all_owner = int_owner + all_bnd_owner

    boundary = []
    bnd_start = n_new_internal
    if all_bnd_faces:
        # 按原始 patch 名分组 —— 简化为单个 patch
        boundary.append({
            "name": "mirror_boundary",
            "type": "wall",
            "startFace": bnd_start,
            "nFaces": len(all_bnd_faces),
        })

    return FvMesh(
        points=all_points,
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )
