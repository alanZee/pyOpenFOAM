"""Shared fixtures for application tests."""

import pytest
import torch

from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.core.dtype import INDEX_DTYPE


def make_fv_mesh(dtype=torch.float64):
    """Create a minimal 2-cell hex mesh for testing.

    Returns an FvMesh with geometry computed.
    """
    nx, ny, nz = 2, 1, 1
    dx, dy, dz = 1.0, 1.0, 1.0

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

    # Internal face between cell 0 and cell 1 (x-direction)
    f = [pt_idx(1, 0, 0), pt_idx(1, 1, 0), pt_idx(1, 1, 1), pt_idx(1, 0, 1)]
    faces.append(f)
    owner.append(0)
    neighbour.append(1)

    n_internal = len(faces)

    # Boundary faces
    # bottom (z=0)
    for j in range(ny):
        for i in range(nx):
            faces.append([pt_idx(i, j, 0), pt_idx(i + 1, j, 0), pt_idx(i + 1, j + 1, 0), pt_idx(i, j + 1, 0)])
            owner.append(j * nx + i)

    # top (z=1)
    for j in range(ny):
        for i in range(nx):
            faces.append([pt_idx(i, j, 1), pt_idx(i, j + 1, 1), pt_idx(i + 1, j + 1, 1), pt_idx(i + 1, j, 1)])
            owner.append(j * nx + i)

    # front (y=0)
    for k in range(nz):
        for i in range(nx):
            faces.append([pt_idx(i, 0, k), pt_idx(i, 0, k + 1), pt_idx(i + 1, 0, k + 1), pt_idx(i + 1, 0, k)])
            owner.append(k * ny * nx + i)

    # back (y=1)
    for k in range(nz):
        for i in range(nx):
            faces.append([pt_idx(i, ny, k), pt_idx(i + 1, ny, k), pt_idx(i + 1, ny, k + 1), pt_idx(i, ny, k + 1)])
            owner.append(k * ny * nx + (ny - 1) * nx + i)

    # left (x=0)
    for k in range(nz):
        for j in range(ny):
            faces.append([pt_idx(0, j, k), pt_idx(0, j + 1, k), pt_idx(0, j + 1, k + 1), pt_idx(0, j, k + 1)])
            owner.append(k * ny * nx + j * nx)

    # right (x=2)
    for k in range(nz):
        for j in range(ny):
            faces.append([pt_idx(nx, j, k), pt_idx(nx, j, k + 1), pt_idx(nx, j + 1, k + 1), pt_idx(nx, j + 1, k)])
            owner.append(k * ny * nx + j * nx + (nx - 1))

    n_bnd = 2 * ny * nz + 2 * nx * nz + 2 * nx * ny

    boundary = [
        {"name": "bottom", "type": "wall", "startFace": n_internal, "nFaces": nx * ny},
        {"name": "top", "type": "wall", "startFace": n_internal + nx * ny, "nFaces": nx * ny},
        {"name": "front", "type": "wall", "startFace": n_internal + 2 * nx * ny, "nFaces": nx * nz},
        {"name": "back", "type": "wall", "startFace": n_internal + 2 * nx * ny + nx * nz, "nFaces": nx * nz},
        {"name": "left", "type": "wall", "startFace": n_internal + 2 * nx * ny + 2 * nx * nz, "nFaces": ny * nz},
        {"name": "right", "type": "wall", "startFace": n_internal + 2 * nx * ny + 2 * nx * nz + ny * nz, "nFaces": ny * nz},
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
def fv_mesh():
    """2-cell hex mesh with geometry computed."""
    return make_fv_mesh()
