"""Shared fixtures for boundary condition tests."""

import pytest
import torch

from pyfoam.boundary.boundary_condition import Patch


@pytest.fixture
def simple_patch() -> Patch:
    """A simple 3-face patch for testing.

    Geometry:
        - 3 faces, all facing +x direction
        - Face areas = 1.0
        - deltaCoeffs = 2.0 (distance = 0.5)
        - Owner cells: [0, 1, 2]
    """
    return Patch(
        name="testPatch",
        face_indices=torch.tensor([10, 11, 12]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


@pytest.fixture
def two_face_patch() -> Patch:
    """A minimal 2-face patch."""
    return Patch(
        name="twoFace",
        face_indices=torch.tensor([5, 6]),
        face_normals=torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([0.5, 0.5], dtype=torch.float64),
        delta_coeffs=torch.tensor([4.0, 4.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1]),
    )


@pytest.fixture
def cyclic_pair() -> tuple[Patch, Patch]:
    """Two coupled patches for cyclic BC testing."""
    patch_a = Patch(
        name="cyclic_half1",
        face_indices=torch.tensor([20, 21]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1]),
        neighbour_patch="cyclic_half2",
    )
    patch_b = Patch(
        name="cyclic_half2",
        face_indices=torch.tensor([22, 23]),
        face_normals=torch.tensor([
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([2, 3]),
        neighbour_patch="cyclic_half1",
    )
    return patch_a, patch_b


@pytest.fixture
def wall_patch() -> Patch:
    """A wall patch for wall-function tests."""
    return Patch(
        name="wall",
        face_indices=torch.tensor([30, 31, 32]),
        face_normals=torch.tensor([
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )
