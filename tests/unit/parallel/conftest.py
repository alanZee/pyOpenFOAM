"""Shared fixtures for parallel tests — 8-cell mesh (2x2x2 grid).

Creates a mesh with 8 unit cubes arranged in a 2x2x2 grid, suitable
for testing domain decomposition into 2 or 4 subdomains.

Layout (z increases upward, y into page, x right):

    Layer z=1..2:  cells 4,5,6,7
    Layer z=0..1:  cells 0,1,2,3

    y=1: [2,3] [6,7]
    y=0: [0,1] [4,5]
         x=0    x=1
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# 8-cell mesh data
# ---------------------------------------------------------------------------

# 27 points for a 3x3x3 grid (2 intervals per axis)
_POINTS = []
for k in range(3):
    for j in range(3):
        for i in range(3):
            _POINTS.append([float(i), float(j), float(k)])

# Cell ordering: cell = ix + iy*2 + iz*4
# Neighbouring cells share faces.

# We define faces as (owner, neighbour) pairs first, then boundary faces.
# Internal faces (between cells):
#   x-direction: (0,1), (2,3), (4,5), (6,7) — 4 faces
#   y-direction: (0,2), (1,3), (4,6), (5,7) — 4 faces
#   z-direction: (0,4), (1,5), (2,6), (3,7) — 4 faces
# Total internal faces: 12

# Helper to get point index in the 3x3x3 grid
def _pt(ix, iy, iz):
    return ix + iy * 3 + iz * 9


# Face vertex lists (quads)
_FACES = []
_OWNER = []
_NEIGHBOUR = []

# --- x-direction internal faces (normal = +x) ---
# Between cell 0 (ix=0,iy=0,iz=0) and cell 1 (ix=1,iy=0,iz=0)
_FACES.append([_pt(1,0,0), _pt(1,1,0), _pt(1,1,1), _pt(1,0,1)])
_OWNER.append(0); _NEIGHBOUR.append(1)

# Between cell 2 (ix=0,iy=1,iz=0) and cell 3 (ix=1,iy=1,iz=0)
_FACES.append([_pt(1,1,0), _pt(1,2,0), _pt(1,2,1), _pt(1,1,1)])
_OWNER.append(2); _NEIGHBOUR.append(3)

# Between cell 4 (ix=0,iy=0,iz=1) and cell 5 (ix=1,iy=0,iz=1)
_FACES.append([_pt(1,0,1), _pt(1,1,1), _pt(1,1,2), _pt(1,0,2)])
_OWNER.append(4); _NEIGHBOUR.append(5)

# Between cell 6 (ix=0,iy=1,iz=1) and cell 7 (ix=1,iy=1,iz=1)
_FACES.append([_pt(1,1,1), _pt(1,2,1), _pt(1,2,2), _pt(1,1,2)])
_OWNER.append(6); _NEIGHBOUR.append(7)

# --- y-direction internal faces (normal = +y) ---
# Between cell 0 and cell 2
_FACES.append([_pt(0,1,0), _pt(1,1,0), _pt(1,1,1), _pt(0,1,1)])
_OWNER.append(0); _NEIGHBOUR.append(2)

# Between cell 1 and cell 3
_FACES.append([_pt(1,1,0), _pt(2,1,0), _pt(2,1,1), _pt(1,1,1)])
_OWNER.append(1); _NEIGHBOUR.append(3)

# Between cell 4 and cell 6
_FACES.append([_pt(0,1,1), _pt(1,1,1), _pt(1,1,2), _pt(0,1,2)])
_OWNER.append(4); _NEIGHBOUR.append(6)

# Between cell 5 and cell 7
_FACES.append([_pt(1,1,1), _pt(2,1,1), _pt(2,1,2), _pt(1,1,2)])
_OWNER.append(5); _NEIGHBOUR.append(7)

# --- z-direction internal faces (normal = +z) ---
# Between cell 0 and cell 4
_FACES.append([_pt(0,0,1), _pt(1,0,1), _pt(1,1,1), _pt(0,1,1)])
_OWNER.append(0); _NEIGHBOUR.append(4)

# Between cell 1 and cell 5
_FACES.append([_pt(1,0,1), _pt(2,0,1), _pt(2,1,1), _pt(1,1,1)])
_OWNER.append(1); _NEIGHBOUR.append(5)

# Between cell 2 and cell 6
_FACES.append([_pt(0,1,1), _pt(1,1,1), _pt(1,2,1), _pt(0,2,1)])
_OWNER.append(2); _NEIGHBOUR.append(6)

# Between cell 3 and cell 7
_FACES.append([_pt(1,1,1), _pt(2,1,1), _pt(2,2,1), _pt(1,2,1)])
_OWNER.append(3); _NEIGHBOUR.append(7)

# --- Boundary faces ---
# For each cell, add the faces that are on the domain boundary.
# Bottom (z=0) faces
boundary_patches = []
boundary_start = len(_FACES)

# Bottom faces (z=0, normal = -z)
bottom_faces = []
for cell_idx, (ix, iy) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    _FACES.append([_pt(ix,iy,0), _pt(ix+1,iy,0), _pt(ix+1,iy+1,0), _pt(ix,iy+1,0)])
    _OWNER.append(cell_idx)
    bottom_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "bottom",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(bottom_faces),
})
boundary_start += len(bottom_faces)

# Top faces (z=2, normal = +z)
top_faces = []
for cell_idx, (ix, iy) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    _FACES.append([_pt(ix,iy,2), _pt(ix+1,iy,2), _pt(ix+1,iy+1,2), _pt(ix,iy+1,2)])
    _OWNER.append(cell_idx + 4)
    top_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "top",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(top_faces),
})
boundary_start += len(top_faces)

# Front faces (y=0, normal = -y)
front_faces = []
for cell_idx, (ix, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = ix + iz * 4
    _FACES.append([_pt(ix,0,iz), _pt(ix+1,0,iz), _pt(ix+1,0,iz+1), _pt(ix,0,iz+1)])
    _OWNER.append(c)
    front_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "front",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(front_faces),
})
boundary_start += len(front_faces)

# Back faces (y=2, normal = +y)
back_faces = []
for cell_idx, (ix, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = ix + 2 + iz * 4
    _FACES.append([_pt(ix,2,iz), _pt(ix+1,2,iz), _pt(ix+1,2,iz+1), _pt(ix,2,iz+1)])
    _OWNER.append(c)
    back_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "back",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(back_faces),
})
boundary_start += len(back_faces)

# Left faces (x=0, normal = -x)
left_faces = []
for cell_idx, (iy, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = iy + iz * 4
    _FACES.append([_pt(0,iy,iz), _pt(0,iy+1,iz), _pt(0,iy+1,iz+1), _pt(0,iy,iz+1)])
    _OWNER.append(c)
    left_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "left",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(left_faces),
})
boundary_start += len(left_faces)

# Right faces (x=2, normal = +x)
right_faces = []
for cell_idx, (iy, iz) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
    c = 1 + iy + iz * 4
    _FACES.append([_pt(2,iy,iz), _pt(2,iy+1,iz), _pt(2,iy+1,iz+1), _pt(2,iy,iz+1)])
    _OWNER.append(c)
    right_faces.append(len(_FACES) - 1)

boundary_patches.append({
    "name": "right",
    "type": "wall",
    "startFace": boundary_start,
    "nFaces": len(right_faces),
})


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_8cell_raw_mesh():
    """Return raw Python lists for the 8-cell mesh."""
    return _POINTS, _FACES, _OWNER, _NEIGHBOUR, boundary_patches


def make_8cell_poly_mesh(device="cpu", dtype=torch.float64):
    """Return a PolyMesh of the 8-cell mesh."""
    from pyfoam.mesh.poly_mesh import PolyMesh

    points = torch.tensor(_POINTS, dtype=dtype, device=device)
    faces = [
        torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES
    ]
    owner = torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device)
    neighbour = torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device)

    return PolyMesh(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary_patches,
    )


def make_8cell_fv_mesh(device="cpu", dtype=torch.float64):
    """Return an FvMesh of the 8-cell mesh with geometry computed."""
    from pyfoam.mesh.fv_mesh import FvMesh

    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES
        ],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=boundary_patches,
    )
    mesh.compute_geometry()
    return mesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_8cell_raw():
    """Raw mesh data for the 8-cell mesh."""
    return make_8cell_raw_mesh()


@pytest.fixture
def poly_mesh_8cell():
    """PolyMesh of the 8-cell mesh."""
    return make_8cell_poly_mesh()


@pytest.fixture
def fv_mesh_8cell():
    """FvMesh of the 8-cell mesh with geometry."""
    return make_8cell_fv_mesh()
