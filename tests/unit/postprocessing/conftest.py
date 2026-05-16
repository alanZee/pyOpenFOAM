"""Shared fixtures for postprocessing tests — 2-cell hex mesh."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE


# ---------------------------------------------------------------------------
# 2-cell hex mesh (two unit cubes stacked in z)
# ---------------------------------------------------------------------------
#
#   Cell 1: z=1..2     11------10
#                      /|      /|
#                     / |     / |
#                    8------9   |
#                    |  7---|--6
#                    | /    | /
#                    |/     |/
#   Cell 0: z=0..1   4------5
#                    |      |
#                    3------2
#                    /|     /|
#                   / |    / |
#                  0------1   |
#
# Internal face (z=1): (4,5,6,7), owner=0, neighbour=1
# Boundary: 5 faces per cell (10 total)

_POINTS = [
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1
    [1.0, 1.0, 0.0],  # 2
    [0.0, 1.0, 0.0],  # 3
    [0.0, 0.0, 1.0],  # 4
    [1.0, 0.0, 1.0],  # 5
    [1.0, 1.0, 1.0],  # 6
    [0.0, 1.0, 1.0],  # 7
    [0.0, 0.0, 2.0],  # 8
    [1.0, 0.0, 2.0],  # 9
    [1.0, 1.0, 2.0],  # 10
    [0.0, 1.0, 2.0],  # 11
]

_FACES = [
    [4, 5, 6, 7],     # 0: internal face at z=1
    [0, 3, 2, 1],     # 1: bottom of cell 0 (z=0)
    [0, 1, 5, 4],     # 2: front of cell 0 (y=0)
    [3, 7, 6, 2],     # 3: back of cell 0 (y=1)
    [0, 4, 7, 3],     # 4: left of cell 0 (x=0)
    [1, 2, 6, 5],     # 5: right of cell 0 (x=1)
    [8, 9, 10, 11],   # 6: top of cell 1 (z=2)
    [4, 5, 9, 8],     # 7: front of cell 1 (y=0)
    [7, 11, 10, 6],   # 8: back of cell 1 (y=1)
    [4, 8, 11, 7],    # 9: left of cell 1 (x=0)
    [5, 6, 10, 9],    # 10: right of cell 1 (x=1)
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


def make_fv_mesh(device="cpu", dtype=torch.float64):
    """Return an FvMesh of the 2-cell hex mesh with geometry computed."""
    from pyfoam.mesh.fv_mesh import FvMesh

    mesh = FvMesh(
        points=torch.tensor(_POINTS, dtype=dtype, device=device),
        faces=[
            torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in _FACES
        ],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE, device=device),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE, device=device),
        boundary=_BOUNDARY,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def fv_mesh():
    """Fixture providing an FvMesh of the 2-cell hex mesh."""
    return make_fv_mesh()


@pytest.fixture
def sample_fields(fv_mesh):
    """Fixture providing sample fields for testing."""
    from pyfoam.fields.vol_fields import volScalarField, volVectorField

    device = fv_mesh.device
    dtype = fv_mesh.dtype

    # Scalar pressure field
    p = volScalarField(fv_mesh, "p")
    p.assign(torch.tensor([101325.0, 101300.0], dtype=dtype, device=device))

    # Vector velocity field
    U = volVectorField(fv_mesh, "U")
    U.assign(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
    ], dtype=dtype, device=device))

    return {"p": p, "U": U}
