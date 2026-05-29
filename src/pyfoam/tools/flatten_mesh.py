"""
flattenMesh — convert a 3D mesh to 2D by collapsing the z-direction.

Mirrors a common pre-processing step in OpenFOAM workflows.  Projects all
points onto a plane by averaging z-coordinates within a tolerance, then
removes degenerate (zero-area) faces that result from the collapse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["flatten_mesh"]


def flatten_mesh(
    mesh: "FvMesh",
    z_tolerance: float = 1e-6,
) -> "FvMesh":
    """Convert a 3D mesh to 2D by collapsing z-direction variation.

    Points whose z-coordinates differ by less than *z_tolerance* are mapped
    to the same z-level (their mean).  This effectively flattens thin 3D
    geometries into true 2D representations.

    After collapsing, any faces that become degenerate (duplicate vertices)
    are cleaned up.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh (typically a single-cell-thick extruded mesh).
    z_tolerance : float
        Maximum z-range within a group of points that will be collapsed
        to the same z-level.

    Returns
    -------
    FvMesh
        New mesh with z-coordinates flattened.

    Raises
    ------
    ValueError
        If the mesh has no points.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device
    dt = mesh.dtype

    pts = mesh.points.clone().cpu()
    z_vals = pts[:, 2]

    # 对 z 坐标进行聚类
    z_unique = _cluster_z(z_vals, z_tolerance)

    # 创建映射：每个点映射到聚类后的 z 值
    new_pts = pts.clone()
    for i in range(pts.shape[0]):
        new_pts[i, 2] = z_unique[i]

    new_pts = new_pts.to(device=dev, dtype=dt)

    # 清理退化面（重复顶点）
    clean_faces = []
    degenerate = set()
    for fi in range(mesh.n_faces):
        face = mesh.faces[fi]
        unique_verts = torch.unique(face)
        if unique_verts.numel() < 3:
            degenerate.add(fi)
            continue
        clean_faces.append(fi)

    # 如果没有退化面，直接返回
    if not degenerate:
        return FvMesh(
            points=new_pts,
            faces=[f.clone() for f in mesh.faces],
            owner=mesh.owner.clone(),
            neighbour=mesh.neighbour.clone(),
            boundary=[dict(b) for b in mesh.boundary],
            validate=False,
        )

    # 重建拓扑（移除退化面）
    int_faces = []
    int_owner = []
    int_neighbour = []
    bnd_faces = []
    bnd_owner = []

    for fi in clean_faces:
        own = int(mesh.owner[fi].item())
        if fi < mesh.n_internal_faces:
            nbr = int(mesh.neighbour[fi].item())
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(own)
            int_neighbour.append(nbr)
        else:
            bnd_faces.append(mesh.faces[fi].clone())
            bnd_owner.append(own)

    n_new_internal = len(int_neighbour)
    all_faces = int_faces + bnd_faces
    all_owner = int_owner + bnd_owner

    # 更新 boundary 描述
    boundary = []
    bnd_start = n_new_internal
    for patch in mesh.boundary:
        ps = patch["startFace"]
        pe = ps + patch["nFaces"]
        count = sum(1 for fi in range(ps, pe) if fi in clean_faces)
        if count > 0:
            boundary.append({
                "name": patch["name"],
                "type": patch["type"],
                "startFace": bnd_start,
                "nFaces": count,
            })
            bnd_start += count

    return FvMesh(
        points=new_pts,
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )


def _cluster_z(z_vals: torch.Tensor, tol: float) -> torch.Tensor:
    """将 z 坐标按容差聚类，返回每点的代表 z 值。"""
    n = z_vals.shape[0]
    result = z_vals.clone()

    visited = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if visited[i]:
            continue
        # 找出与 z_vals[i] 在容差内的所有点
        cluster_mask = (z_vals - z_vals[i]).abs() <= tol
        cluster_mask = cluster_mask & ~visited
        cluster_z = z_vals[cluster_mask]
        mean_z = cluster_z.mean()
        # 全局赋值（避免 indexed assignment 的 size 问题）
        indices = cluster_mask.nonzero(as_tuple=False).squeeze(1)
        for idx in indices:
            result[int(idx.item())] = mean_z
        visited[indices] = True

    return result
