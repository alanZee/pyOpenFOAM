"""Shared fixtures for field tests — uses the 2-cell hex mesh from mesh tests."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.boundary.boundary_condition import Patch
from pyfoam.boundary.boundary_field import BoundaryField
from pyfoam.boundary.fixed_value import FixedValueBC
from pyfoam.boundary.zero_gradient import ZeroGradientBC


# ---------------------------------------------------------------------------
# Re-use the 2-cell hex mesh from mesh/conftest.py
# ---------------------------------------------------------------------------

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
def bottom_patch() -> Patch:
    """Bottom boundary patch (5 faces)."""
    return Patch(
        name="bottom",
        face_indices=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
        face_normals=torch.tensor([
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64),
    )


@pytest.fixture
def top_patch() -> Patch:
    """Top boundary patch (5 faces)."""
    return Patch(
        name="top",
        face_indices=torch.tensor([6, 7, 8, 9, 10], dtype=torch.int64),
        face_normals=torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64),
    )


@pytest.fixture
def boundary_field(bottom_patch, top_patch) -> BoundaryField:
    """Boundary field with fixedValue on bottom and zeroGradient on top."""
    bf = BoundaryField()
    bf.add(FixedValueBC(bottom_patch, {"value": 0.0}))
    bf.add(ZeroGradientBC(top_patch))
    return bf
