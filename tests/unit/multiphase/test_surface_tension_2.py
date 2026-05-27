"""Tests for CSFSurfaceTension (enhanced CSF model with curvature smoothing)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Minimal 2D mesh for testing (same as test_vof.py)
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
    """Minimal mesh for testing."""
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

    # Vertical internal faces (x-direction neighbours)
    for j in range(n_y):
        for i in range(n_x - 1):
            owners.append(j * n_x + i)
            neighbours.append(j * n_x + i + 1)

    # Horizontal internal faces (y-direction neighbours)
    for j in range(n_y - 1):
        for i in range(n_x):
            owners.append(j * n_x + i)
            neighbours.append((j + 1) * n_x + i)

    n_internal = len(neighbours)

    # Boundary faces
    for i in range(n_x):  # bottom
        owners.append(i)
    for i in range(n_x):  # top
        owners.append((n_y - 1) * n_x + i)
    for j in range(n_y):  # left
        owners.append(j * n_x)
    for j in range(n_y):  # right
        owners.append(j * n_x + n_x - 1)

    n_boundary = 2 * n_x + 2 * n_y
    n_faces = n_internal + n_boundary

    owner = torch.tensor(owners, dtype=torch.long, device=device)
    neigh = torch.tensor(neighbours, dtype=torch.long, device=device)

    # Face areas
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

    delta_x = dx
    delta_y = dy
    delta_list = []
    for j in range(n_y):
        for i in range(n_x - 1):
            delta_list.append(1.0 / delta_x)
    for j in range(n_y - 1):
        for i in range(n_x):
            delta_list.append(1.0 / delta_y)
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
def mesh_8x8():
    """8x8 mesh for testing."""
    return make_2d_mesh(8, 8)


@pytest.fixture
def mesh_4x4():
    """4x4 mesh for testing."""
    return make_2d_mesh(4, 4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCSFSurfaceTension:
    """Tests for the CSFSurfaceTension model."""

    def test_initialization(self, mesh_4x4):
        """CSFSurfaceTension creates correctly."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        assert model.sigma == 0.07
        assert model.n_alpha_smooth == 1
        assert model.n_curvature_smooth == 2

    def test_custom_smoothing_params(self, mesh_4x4):
        """Custom smoothing parameters are stored."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(
            sigma=0.07, mesh=mesh_4x4,
            n_alpha_smooth=3, n_curvature_smooth=5,
        )
        assert model.n_alpha_smooth == 3
        assert model.n_curvature_smooth == 5

    def test_sigma_setter(self, mesh_4x4):
        """Sigma can be updated."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        model.sigma = 0.1
        assert model.sigma == 0.1

    def test_force_shape(self, mesh_4x4):
        """Force has correct shape."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        n_cells = mesh_4x4.n_cells
        # Smooth sigmoid interface with sufficient width for 4x4 mesh
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                cc_x[j * 4 + i] = (i + 0.5) / 4.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        F_st = model.compute_force(alpha)
        assert F_st.shape == (n_cells, 3)

    def test_zero_sigma_zero_force(self, mesh_4x4):
        """Zero sigma produces zero force."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.0, mesh=mesh_4x4)
        n_cells = mesh_4x4.n_cells
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                cc_x[j * 4 + i] = (i + 0.5) / 4.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        F_st = model.compute_force(alpha)
        assert torch.allclose(F_st, torch.zeros(n_cells, 3, dtype=CFD_DTYPE))

    def test_uniform_alpha_zero_force(self, mesh_4x4):
        """Zero alpha (no interface) produces zero force."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        n_cells = mesh_4x4.n_cells
        # Zero alpha: entirely below the interface threshold, gradient is exactly zero
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)

        F_st = model.compute_force(alpha)
        assert torch.allclose(F_st, torch.zeros(n_cells, 3, dtype=CFD_DTYPE), atol=1e-15)

    def test_curvature_shape(self, mesh_4x4):
        """Curvature has correct shape."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        n_cells = mesh_4x4.n_cells
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                cc_x[j * 4 + i] = (i + 0.5) / 4.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        kappa = model.compute_curvature(alpha)
        assert kappa.shape == (n_cells,)

    def test_curvature_nonzero_at_interface(self, mesh_4x4):
        """Curvature is nonzero near the interface."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4)
        n_cells = mesh_4x4.n_cells
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                cc_x[j * 4 + i] = (i + 0.5) / 4.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        kappa = model.compute_curvature(alpha)
        # At least some cells near the interface should have nonzero curvature
        assert kappa.abs().sum() > 0

    def test_force_nonzero_at_interface(self, mesh_4x4):
        """Surface tension force is nonzero near a smooth interface."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4, n_curvature_smooth=1)
        n_cells = mesh_4x4.n_cells
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                cc_x[j * 4 + i] = (i + 0.5) / 4.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        F_st = model.compute_force(alpha)
        # Force should be nonzero somewhere at the interface
        assert F_st.abs().sum() > 0

    def test_repr(self, mesh_4x4):
        """__repr__ includes key parameters."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh_4x4, n_curvature_smooth=3)
        r = repr(model)
        assert "CSFSurfaceTension" in r
        assert "0.07" in r
        assert "n_curvature_smooth=3" in r

    def test_smoothing_reduces_curvature_magnitude(self, mesh_8x8):
        """More curvature smoothing reduces curvature peaks."""
        from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

        n_cells = mesh_8x8.n_cells
        cc_x = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(8):
            for i in range(8):
                cc_x[j * 8 + i] = (i + 0.5) / 8.0
        alpha = 0.5 * (1.0 - torch.tanh((cc_x - 0.5) / 0.2))

        model_low = CSFSurfaceTension(
            sigma=0.07, mesh=mesh_8x8,
            n_alpha_smooth=0, n_curvature_smooth=0,
        )
        model_high = CSFSurfaceTension(
            sigma=0.07, mesh=mesh_8x8,
            n_alpha_smooth=2, n_curvature_smooth=5,
        )

        kappa_low = model_low.compute_curvature(alpha)
        kappa_high = model_high.compute_curvature(alpha)

        # Peak curvature should be reduced by smoothing
        assert kappa_high.abs().max() <= kappa_low.abs().max() + 1e-10
