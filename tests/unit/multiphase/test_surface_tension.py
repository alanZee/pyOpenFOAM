"""
Unit tests for CSF surface tension model.

Tests cover:
- SurfaceTensionModel initialisation and properties
- compute_force output shape and zero-sigma shortcut
- compute_force on uniform alpha (no interface → zero force)
- compute_curvature output shape and finiteness
- __repr__ string
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Minimal mesh (same as test_vof.py)
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
    """Minimal 2D Cartesian mesh for testing."""

    n_cells: int
    n_internal_faces: int
    n_faces: int
    owner: torch.Tensor
    neighbour: torch.Tensor
    face_areas: torch.Tensor
    cell_volumes: torch.Tensor
    face_weights: torch.Tensor
    delta_coefficients: torch.Tensor


def make_2d_mesh(
    n_x: int = 4,
    n_y: int = 4,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 0.1,
    device: str = "cpu",
    dtype: torch.dtype = CFD_DTYPE,
) -> SimpleMesh:
    """Create a 2D Cartesian mesh for testing."""
    n_cells = n_x * n_y

    owners = []
    neighbours = []

    # Vertical faces (x-direction)
    for j in range(n_y):
        for i in range(n_x - 1):
            owners.append(j * n_x + i)
            neighbours.append(j * n_x + i + 1)

    # Horizontal faces (y-direction)
    for j in range(n_y - 1):
        for i in range(n_x):
            owners.append(j * n_x + i)
            neighbours.append((j + 1) * n_x + i)

    n_internal = len(neighbours)

    # Boundary faces
    for i in range(n_x):
        owners.append(i)                      # bottom
    for i in range(n_x):
        owners.append((n_y - 1) * n_x + i)   # top
    for j in range(n_y):
        owners.append(j * n_x)               # left
    for j in range(n_y):
        owners.append(j * n_x + n_x - 1)    # right

    n_boundary = 2 * n_x + 2 * n_y
    n_faces = n_internal + n_boundary

    owner = torch.tensor(owners, dtype=torch.long, device=device)
    neigh = torch.tensor(neighbours, dtype=torch.long, device=device)

    face_areas_list = []
    for j in range(n_y):
        for i in range(n_x - 1):
            face_areas_list.append([dy * dz, 0.0, 0.0])
    for j in range(n_y - 1):
        for i in range(n_x):
            face_areas_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_x):
        face_areas_list.append([0.0, -dx * dz, 0.0])
    for _ in range(n_x):
        face_areas_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_y):
        face_areas_list.append([-dy * dz, 0.0, 0.0])
    for _ in range(n_y):
        face_areas_list.append([dy * dz, 0.0, 0.0])

    face_areas = torch.tensor(face_areas_list, dtype=dtype, device=device)

    vol = dx * dy * dz
    cell_volumes = torch.full((n_cells,), vol, dtype=dtype, device=device)

    face_weights = torch.full((n_internal,), 0.5, dtype=dtype, device=device)

    delta_list = []
    for j in range(n_y):
        for i in range(n_x - 1):
            delta_list.append(1.0 / dx)
    for j in range(n_y - 1):
        for i in range(n_x):
            delta_list.append(1.0 / dy)
    delta_coeffs = torch.tensor(delta_list, dtype=dtype, device=device)

    return SimpleMesh(
        n_cells=n_cells,
        n_internal_faces=n_internal,
        n_faces=n_faces,
        owner=owner,
        neighbour=neigh,
        face_areas=face_areas,
        cell_volumes=cell_volumes,
        face_weights=face_weights,
        delta_coefficients=delta_coeffs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_4x4():
    """4x4 mesh."""
    return make_2d_mesh(4, 4)


@pytest.fixture
def mesh_8x8():
    """8x8 mesh."""
    return make_2d_mesh(8, 8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSurfaceTensionModel:
    """Tests for SurfaceTensionModel."""

    def test_initialization(self, mesh_4x4):
        """Model stores sigma and mesh correctly."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4, n_smooth=2)
        assert st.sigma == 0.07
        assert st._n_smooth == 2
        assert st._mesh is mesh_4x4

    def test_sigma_property(self, mesh_4x4):
        """sigma property returns the stored value."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        st = SurfaceTensionModel(sigma=0.0, mesh=mesh_4x4)
        assert st.sigma == 0.0

        st2 = SurfaceTensionModel(sigma=1.23, mesh=mesh_4x4)
        assert st2.sigma == 1.23

    def test_zero_sigma_returns_zeros(self, mesh_4x4):
        """sigma=0 yields a zero force tensor of the correct shape."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        st = SurfaceTensionModel(sigma=0.0, mesh=mesh_4x4)
        alpha = torch.rand(n_cells, dtype=CFD_DTYPE)

        F = st.compute_force(alpha)
        assert F.shape == (n_cells, 3)
        assert torch.allclose(F, torch.zeros_like(F))

    def test_force_shape(self, mesh_4x4):
        """compute_force returns (n_cells, 3)."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4)
        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.0, 1.0)

        F = st.compute_force(alpha)
        assert F.shape == (n_cells, 3)

    def test_uniform_alpha_zero_force(self, mesh_4x4):
        """Uniform alpha (no interface) produces zero force."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4, n_smooth=1)

        # Fully wet / fully dry → interface mask is empty
        for val in [0.0, 1.0]:
            alpha = torch.full((n_cells,), val, dtype=CFD_DTYPE)
            F = st.compute_force(alpha)
            assert torch.allclose(F, torch.zeros_like(F), atol=1e-20), (
                f"Expected zero force for uniform alpha={val}"
            )

    def test_force_finite_at_interface(self, mesh_8x8):
        """Force is finite (no NaN/Inf) with a step-function interface."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_8x8.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_8x8, n_smooth=2)

        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(8):
            for i in range(4):
                alpha[j * 8 + i] = 1.0

        F = st.compute_force(alpha)
        assert torch.isfinite(F).all(), "Force contains NaN or Inf"

    def test_force_scales_with_sigma(self, mesh_4x4):
        """Force magnitude scales linearly with sigma."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 1.0

        st1 = SurfaceTensionModel(sigma=0.1, mesh=mesh_4x4, n_smooth=1)
        st2 = SurfaceTensionModel(sigma=0.2, mesh=mesh_4x4, n_smooth=1)

        F1 = st1.compute_force(alpha)
        F2 = st2.compute_force(alpha)

        # F2 should be approximately 2x F1 (same alpha, double sigma)
        nonzero = F1.norm(dim=1) > 1e-30
        if nonzero.any():
            ratio = F2[nonzero].norm(dim=1) / F1[nonzero].norm(dim=1)
            assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=0.1)

    def test_compute_curvature_shape(self, mesh_4x4):
        """compute_curvature returns (n_cells,)."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4)

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.0, 1.0)
        kappa = st.compute_curvature(alpha)

        assert kappa.shape == (n_cells,)

    def test_compute_curvature_finite(self, mesh_8x8):
        """Curvature is finite at a step-function interface."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_8x8.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_8x8, n_smooth=2)

        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(8):
            for i in range(4):
                alpha[j * 8 + i] = 1.0

        kappa = st.compute_curvature(alpha)
        assert torch.isfinite(kappa).all()

    def test_curvature_zero_for_no_interface(self, mesh_4x4):
        """Curvature is zero when no cells are in the interface band."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        n_cells = mesh_4x4.n_cells
        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4, n_smooth=1)

        # alpha=0 or alpha=1: interface mask (0.01, 0.99) excludes all cells
        for val in [0.0, 1.0]:
            alpha = torch.full((n_cells,), val, dtype=CFD_DTYPE)
            kappa = st.compute_curvature(alpha)
            assert torch.allclose(kappa, torch.zeros_like(kappa), atol=1e-20), (
                f"Expected zero curvature for uniform alpha={val}"
            )

    def test_repr(self, mesh_4x4):
        """__repr__ includes sigma, n_smooth, and n_cells."""
        from pyfoam.multiphase.surface_tension import SurfaceTensionModel

        st = SurfaceTensionModel(sigma=0.07, mesh=mesh_4x4, n_smooth=3)
        r = repr(st)
        assert "SurfaceTensionModel" in r
        assert "0.07" in r
        assert "3" in r       # n_smooth
        assert "16" in r      # 4*4 = 16 cells
