"""Shared fixtures for tools tests."""

from __future__ import annotations

import pytest
import torch

from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.core.dtype import INDEX_DTYPE
from tests.unit.mesh.conftest import make_fv_mesh


@pytest.fixture
def fv_mesh():
    """2-cell hex mesh with geometry computed."""
    return make_fv_mesh()


def make_4x4_hex_mesh(dtype=torch.float64):
    """Create a 4x4x1 hex mesh (16 cells) for more thorough testing.

    Returns an FvMesh with geometry computed.
    """
    nx, ny, nz = 4, 4, 1
    dx, dy, dz = 1.0, 1.0, 1.0

    # 生成顶点
    points = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append([i * dx, j * dy, k * dz])
    points = torch.tensor(points, dtype=dtype)

    def pt_idx(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    faces = []
    owner = []
    neighbour = []

    # x 方向内部面（(i,j) 和 (i+1,j) 之间）
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                f = [
                    pt_idx(i + 1, j, k),
                    pt_idx(i + 1, j + 1, k),
                    pt_idx(i + 1, j + 1, k + 1),
                    pt_idx(i + 1, j, k + 1),
                ]
                faces.append(f)
                owner.append(k * ny * nx + j * nx + i)
                neighbour.append(k * ny * nx + j * nx + i + 1)

    # y 方向内部面（(i,j) 和 (i,j+1) 之间）
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                f = [
                    pt_idx(i, j + 1, k),
                    pt_idx(i + 1, j + 1, k),
                    pt_idx(i + 1, j + 1, k + 1),
                    pt_idx(i, j + 1, k + 1),
                ]
                faces.append(f)
                owner.append(k * ny * nx + j * nx + i)
                neighbour.append(k * ny * nx + (j + 1) * nx + i)

    n_internal = len(faces)

    # 边界面（6 个 patch）
    # bottom (z=0)
    for j in range(ny):
        for i in range(nx):
            faces.append([
                pt_idx(i, j, 0),
                pt_idx(i + 1, j, 0),
                pt_idx(i + 1, j + 1, 0),
                pt_idx(i, j + 1, 0),
            ])
            owner.append(j * nx + i)

    # top (z=1)
    for j in range(ny):
        for i in range(nx):
            faces.append([
                pt_idx(i, j, 1),
                pt_idx(i, j + 1, 1),
                pt_idx(i + 1, j + 1, 1),
                pt_idx(i + 1, j, 1),
            ])
            owner.append(j * nx + i)

    # front (y=0)
    for k in range(nz):
        for i in range(nx):
            faces.append([
                pt_idx(i, 0, k),
                pt_idx(i, 0, k + 1),
                pt_idx(i + 1, 0, k + 1),
                pt_idx(i + 1, 0, k),
            ])
            owner.append(k * ny * nx + i)

    # back (y=ny)
    for k in range(nz):
        for i in range(nx):
            faces.append([
                pt_idx(i, ny, k),
                pt_idx(i + 1, ny, k),
                pt_idx(i + 1, ny, k + 1),
                pt_idx(i, ny, k + 1),
            ])
            owner.append(k * ny * nx + (ny - 1) * nx + i)

    # left (x=0)
    for k in range(nz):
        for j in range(ny):
            faces.append([
                pt_idx(0, j, k),
                pt_idx(0, j + 1, k),
                pt_idx(0, j + 1, k + 1),
                pt_idx(0, j, k + 1),
            ])
            owner.append(k * ny * nx + j * nx)

    # right (x=nx)
    for k in range(nz):
        for j in range(ny):
            faces.append([
                pt_idx(nx, j, k),
                pt_idx(nx, j, k + 1),
                pt_idx(nx, j + 1, k + 1),
                pt_idx(nx, j + 1, k),
            ])
            owner.append(k * ny * nx + j * nx + (nx - 1))

    # 边界 patch 描述
    n_bnd_xy = nx * ny  # 16
    n_bnd_xz = nx * nz  # 4
    n_bnd_yz = ny * nz  # 4

    boundary = [
        {"name": "bottom", "type": "wall", "startFace": n_internal, "nFaces": n_bnd_xy},
        {"name": "top", "type": "wall", "startFace": n_internal + n_bnd_xy, "nFaces": n_bnd_xy},
        {"name": "front", "type": "wall", "startFace": n_internal + 2 * n_bnd_xy, "nFaces": n_bnd_xz},
        {"name": "back", "type": "wall", "startFace": n_internal + 2 * n_bnd_xy + n_bnd_xz, "nFaces": n_bnd_xz},
        {"name": "left", "type": "wall", "startFace": n_internal + 2 * n_bnd_xy + 2 * n_bnd_xz, "nFaces": n_bnd_yz},
        {"name": "right", "type": "wall", "startFace": n_internal + 2 * n_bnd_xy + 2 * n_bnd_xz + n_bnd_yz, "nFaces": n_bnd_yz},
    ]

    faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in faces]
    owner_t = torch.tensor(owner, dtype=INDEX_DTYPE)
    neighbour_t = torch.tensor(neighbour, dtype=INDEX_DTYPE)

    mesh = FvMesh(
        points=points,
        faces=faces_t,
        owner=owner_t,
        neighbour=neighbour_t,
        boundary=boundary,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def large_mesh():
    """4x4x1 hex mesh (16 cells) with geometry computed."""
    return make_4x4_hex_mesh()
