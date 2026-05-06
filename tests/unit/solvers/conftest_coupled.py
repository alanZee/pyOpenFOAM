"""Shared fixtures for coupled solver tests (SIMPLE, PISO, PIMPLE).

Provides a lid-driven cavity mesh setup for testing pressure-velocity
coupling algorithms. The mesh is a 2D square cavity with:
- 4 internal cells (2x2 grid)
- 4 boundary patches (top moving wall, bottom, left, right)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh


def make_cavity_mesh(n_cells_x: int = 2, n_cells_y: int = 2):
    """Create a 2D lid-driven cavity mesh.

    Creates a unit square [0,1] x [0,1] divided into n_cells_x × n_cells_y
    cells. The mesh is 2D (z=0 plane) with one cell depth in z.

    Args:
        n_cells_x: Number of cells in x-direction.
        n_cells_y: Number of cells in y-direction.

    Returns:
        FvMesh with computed geometry.
    """
    device = "cpu"
    dtype = CFD_DTYPE

    # Create points for a unit square mesh
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y

    # Points in a grid (x, y, z=0)
    points = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points.append([i * dx, j * dy, 0.0])

    # Add z=1 points for 3D mesh
    n_base = len(points)
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points.append([i * dx, j * dy, 1.0])

    points = torch.tensor(points, dtype=dtype, device=device)

    # Create faces (quads)
    faces = []
    owner = []
    neighbour = []

    # Internal faces (horizontal between cells)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            # Vertical internal face
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append([p0, p1, p2, p3])

            cell_left = j * n_cells_x + i
            cell_right = j * n_cells_x + i + 1
            owner.append(cell_left)
            neighbour.append(cell_right)

    # Internal faces (vertical between cells)
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            # Horizontal internal face
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append([p0, p1, p2, p3])

            cell_below = j * n_cells_x + i
            cell_above = (j + 1) * n_cells_x + i
            owner.append(cell_below)
            neighbour.append(cell_above)

    n_internal = len(neighbour)

    # Boundary faces
    # Bottom (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append([p0, p1, p2, p3])
        owner.append(i)  # bottom cell

    # Top (y=1) - moving wall
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append([p0, p1, p2, p3])
        owner.append((n_cells_y - 1) * n_cells_x + i)

    # Left (x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append([p0, p1, p2, p3])
        owner.append(j * n_cells_x)

    # Right (x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append([p0, p1, p2, p3])
        owner.append(j * n_cells_x + n_cells_x - 1)

    # Create face tensors
    face_tensors = [torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in faces]
    owner_tensor = torch.tensor(owner, dtype=INDEX_DTYPE, device=device)
    neighbour_tensor = torch.tensor(neighbour, dtype=INDEX_DTYPE, device=device)

    # Boundary patches
    n_bottom = n_cells_x
    n_top = n_cells_x
    n_left = n_cells_y
    n_right = n_cells_y

    boundary = [
        {"name": "bottom", "type": "wall", "startFace": n_internal, "nFaces": n_bottom},
        {"name": "top", "type": "wall", "startFace": n_internal + n_bottom, "nFaces": n_top},
        {"name": "left", "type": "wall", "startFace": n_internal + n_bottom + n_top, "nFaces": n_left},
        {"name": "right", "type": "wall", "startFace": n_internal + n_bottom + n_top + n_left, "nFaces": n_right},
    ]

    mesh = FvMesh(
        points=points,
        faces=face_tensors,
        owner=owner_tensor,
        neighbour=neighbour_tensor,
        boundary=boundary,
    )
    mesh.compute_geometry()

    return mesh


@pytest.fixture
def cavity_mesh():
    """2x2 lid-driven cavity mesh."""
    return make_cavity_mesh(2, 2)


@pytest.fixture
def cavity_mesh_4x4():
    """4x4 lid-driven cavity mesh."""
    return make_cavity_mesh(4, 4)


@pytest.fixture
def zero_velocity(cavity_mesh):
    """Zero velocity field for the cavity mesh."""
    return torch.zeros(cavity_mesh.n_cells, 3, dtype=CFD_DTYPE)


@pytest.fixture
def zero_pressure(cavity_mesh):
    """Zero pressure field for the cavity mesh."""
    return torch.zeros(cavity_mesh.n_cells, dtype=CFD_DTYPE)


@pytest.fixture
def zero_flux(cavity_mesh):
    """Zero face flux for the cavity mesh."""
    return torch.zeros(cavity_mesh.n_faces, dtype=CFD_DTYPE)
