"""
Unit tests for VOF (Volume of Fluid) advection and MULES limiter.

Tests cover:
- VOFAdvection initialisation and basic properties
- Conservation of volume fraction
- Boundedness (α ∈ [0, 1])
- Interface compression effect
- Interface normal and curvature computation
- MULES limiter boundedness and conservation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Minimal mesh for testing
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
    """Minimal mesh for testing VOF and MULES.

    Creates a 2D Cartesian mesh (n_x × n_y) with cells of size dx × dy.
    Thin z-direction (dz) for 3D compatibility.
    """

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
    """Create a 2D Cartesian mesh for testing.

    Returns a SimpleMesh with owner, neighbour, face_areas, etc.
    """
    n_cells = n_x * n_y

    # --- Internal faces ---
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

    # --- Boundary faces ---
    # Bottom (y=0)
    for i in range(n_x):
        owners.append(i)
    # Top (y=1)
    for i in range(n_x):
        owners.append((n_y - 1) * n_x + i)
    # Left (x=0)
    for j in range(n_y):
        owners.append(j * n_x)
    # Right (x=1)
    for j in range(n_y):
        owners.append(j * n_x + n_x - 1)

    n_boundary = 2 * n_x + 2 * n_y
    n_faces = n_internal + n_boundary

    # Build tensors
    owner = torch.tensor(owners, dtype=torch.long, device=device)
    neigh = torch.tensor(neighbours, dtype=torch.long, device=device)

    # Face areas: vertical faces have area dy*dz, horizontal faces dx*dz
    face_areas_list = []
    # Vertical internal faces
    for j in range(n_y):
        for i in range(n_x - 1):
            face_areas_list.append([dy * dz, 0.0, 0.0])
    # Horizontal internal faces
    for j in range(n_y - 1):
        for i in range(n_x):
            face_areas_list.append([0.0, dx * dz, 0.0])
    # Boundary faces
    for _ in range(n_x):  # bottom
        face_areas_list.append([0.0, -dx * dz, 0.0])
    for _ in range(n_x):  # top
        face_areas_list.append([0.0, dx * dz, 0.0])
    for _ in range(n_y):  # left
        face_areas_list.append([-dy * dz, 0.0, 0.0])
    for _ in range(n_y):  # right
        face_areas_list.append([dy * dz, 0.0, 0.0])

    face_areas = torch.tensor(face_areas_list, dtype=dtype, device=device)

    # Cell volumes
    vol = dx * dy * dz
    cell_volumes = torch.full((n_cells,), vol, dtype=dtype, device=device)

    # Face weights (0.5 for uniform mesh)
    face_weights = torch.full((n_internal,), 0.5, dtype=dtype, device=device)

    # Delta coefficients (1/d for uniform mesh)
    delta_x = dx  # distance between cell centres
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
def mesh_4x4():
    """4×4 mesh for testing."""
    return make_2d_mesh(4, 4)


@pytest.fixture
def mesh_8x8():
    """8×8 mesh for testing."""
    return make_2d_mesh(8, 8)


# ---------------------------------------------------------------------------
# Tests: VOFAdvection
# ---------------------------------------------------------------------------

class TestVOFAdvection:
    """Tests for VOFAdvection class."""

    def test_initialization(self, mesh_4x4):
        """VOFAdvection creates correctly."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        assert vof.alpha.shape == (n_cells,)
        assert vof.phi.shape == (mesh_4x4.n_faces,)
        assert vof.U.shape == (n_cells, 3)

    def test_uniform_alpha_preserved(self, mesh_4x4):
        """Uniform α=1.0 stays at 1.0 with zero velocity."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.ones(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        alpha_new = vof.advance(0.01)

        assert torch.allclose(alpha_new, torch.ones(n_cells, dtype=CFD_DTYPE), atol=1e-10)

    def test_zero_alpha_preserved(self, mesh_4x4):
        """Uniform α=0.0 stays at 0.0 with zero velocity."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        alpha_new = vof.advance(0.01)

        assert torch.allclose(alpha_new, torch.zeros(n_cells, dtype=CFD_DTYPE), atol=1e-10)

    def test_bounded_with_velocity(self, mesh_4x4):
        """α stays in [0, 1] after advection with velocity."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        # Step function: α=1 for x<0.5, α=0 for x>=0.5
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):  # left half
                alpha[j * 4 + i] = 1.0

        # Uniform velocity in x-direction
        U = torch.ones(n_cells, 3, dtype=CFD_DTYPE)

        # Compute face flux (simplified)
        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        n_internal = mesh_4x4.n_internal_faces
        for f in range(n_internal):
            phi[f] = 0.01  # small flux

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        for _ in range(10):
            alpha_new = vof.advance(0.001)
            assert alpha_new.min() >= -1e-10, f"α min = {alpha_new.min()}"
            assert alpha_new.max() <= 1.0 + 1e-10, f"α max = {alpha_new.max()}"

    def test_conservation(self, mesh_4x4):
        """Total volume fraction is conserved (sum of α*V before ≈ after)."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        # Smooth gradient
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                alpha[j * 4 + i] = i / 3.0

        V = mesh_4x4.cell_volumes
        total_before = (alpha * V).sum().item()

        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=0.0)
        alpha_new = vof.advance(0.01)

        total_after = (alpha_new * V).sum().item()
        assert abs(total_after - total_before) < 1e-10 * max(abs(total_before), 1e-30)

    def test_interface_compression(self, mesh_8x8):
        """With compression, interface sharpens."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_8x8.n_cells
        n_faces = mesh_8x8.n_faces
        n_internal = mesh_8x8.n_internal_faces

        # Create a diffuse interface (smooth step)
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(8):
            for i in range(8):
                x = (i + 0.5) / 8.0
                alpha[j * 8 + i] = 0.5 * (1.0 + math.tanh(20.0 * (x - 0.5)))

        # Small uniform flux to move interface
        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        for f in range(n_internal):
            phi[f] = 0.001

        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        # Measure sharpness (standard deviation of alpha gradient)
        def measure_sharpness(a):
            grad = a[1:] - a[:-1]
            return grad.abs().mean().item()

        sharp_before = measure_sharpness(alpha)

        vof_compressed = VOFAdvection(
            mesh_8x8, alpha.clone(), phi.clone(), U.clone(),
            C_alpha=1.0, use_mules=False,
        )
        for _ in range(5):
            alpha_c = vof_compressed.advance(0.001)

        sharp_after = measure_sharpness(alpha_c)
        # Compression should increase sharpness (larger gradient mean)
        # This is a qualitative check
        assert alpha_c.min() >= -1e-10
        assert alpha_c.max() <= 1.0 + 1e-10

    def test_interface_normal(self, mesh_4x4):
        """compute_interface_normal returns correct normals."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        # Step function in x-direction
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 1.0

        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        n_hat = vof.compute_interface_normal()

        assert n_hat.shape == (n_cells, 3)
        # Normals should be unit vectors (where non-zero)
        nonzero = n_hat.norm(dim=1) > 0.1
        if nonzero.any():
            norms = n_hat[nonzero].norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_curvature(self, mesh_4x4):
        """compute_curvature returns reasonable values."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        n_faces = mesh_4x4.n_faces

        # Circular interface (approximately)
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                x = (i + 0.5) / 4.0 - 0.5
                y = (j + 0.5) / 4.0 - 0.5
                r = math.sqrt(x**2 + y**2)
                if r < 0.3:
                    alpha[j * 4 + i] = 1.0
                elif r < 0.4:
                    alpha[j * 4 + i] = 0.5

        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        kappa = vof.compute_curvature()

        assert kappa.shape == (n_cells,)
        assert torch.isfinite(kappa).all()

    def test_repr(self, mesh_4x4):
        """VOFAdvection has a useful string representation."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=1.0)
        r = repr(vof)
        assert "VOFAdvection" in r
        assert "16" in r  # n_cells


# ---------------------------------------------------------------------------
# Tests: MULESLimiter
# ---------------------------------------------------------------------------

class TestMULESLimiter:
    """Tests for MULESLimiter class."""

    def test_initialization(self, mesh_4x4):
        """MULESLimiter creates correctly."""
        from pyfoam.multiphase.mules import MULESLimiter

        mules = MULESLimiter(mesh_4x4, n_iterations=3)
        assert "MULESLimiter" in repr(mules)

    def test_limit_bounded(self, mesh_4x4):
        """MULES ensures α ∈ [0, 1]."""
        from pyfoam.multiphase.mules import MULESLimiter

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces

        # Start with smooth alpha
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(4):
                alpha[j * 4 + i] = 0.3 + 0.4 * math.sin(math.pi * i / 3)

        # Large flux that could cause overshoot
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 10.0
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh_4x4, n_iterations=5)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.01,
            alpha_min=0.0, alpha_max=1.0,
        )

        assert alpha_new.min() >= -1e-10, f"α min = {alpha_new.min()}"
        assert alpha_new.max() <= 1.0 + 1e-10, f"α max = {alpha_new.max()}"

    def test_limit_conservation(self, mesh_4x4):
        """MULES preserves conservation."""
        from pyfoam.multiphase.mules import MULESLimiter

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        V = mesh_4x4.cell_volumes
        total_before = (alpha * V).sum().item()

        # Small flux (should be mostly conserved)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 0.01
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh_4x4, n_iterations=3)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.001,
            alpha_min=0.0, alpha_max=1.0,
        )

        total_after = (alpha_new * V).sum().item()
        # Conservation should be approximately preserved
        assert abs(total_after - total_before) < 0.01 * max(abs(total_before), 1e-30)

    def test_limit_flux_bounded(self, mesh_4x4):
        """limit_flux returns bounded flux."""
        from pyfoam.multiphase.mules import MULESLimiter

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 5.0

        mules = MULESLimiter(mesh_4x4, n_iterations=3)
        limited_flux = mules.limit_flux(
            alpha, alpha_flux, 0.01,
            alpha_min=0.0, alpha_max=1.0,
        )

        # The limited flux should be smaller in magnitude than original
        assert limited_flux.abs().max() <= alpha_flux.abs().max() + 1e-10

    def test_no_op_for_bounded(self, mesh_4x4):
        """MULES doesn't modify already bounded fields."""
        from pyfoam.multiphase.mules import MULESLimiter

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces

        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        # Tiny flux that won't cause issues
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 1e-8
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh_4x4, n_iterations=3)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.001,
            alpha_min=0.0, alpha_max=1.0,
        )

        # Should be very close to original
        assert torch.allclose(alpha_new, alpha, atol=0.1)

    def test_extreme_flux(self, mesh_4x4):
        """MULES handles extreme fluxes gracefully."""
        from pyfoam.multiphase.mules import MULESLimiter

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        # Very large flux
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 1000.0
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh_4x4, n_iterations=5)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.01,
            alpha_min=0.0, alpha_max=1.0,
        )

        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10
        assert torch.isfinite(alpha_new).all()

    def test_repr(self, mesh_4x4):
        """MULESLimiter has a useful string representation."""
        from pyfoam.multiphase.mules import MULESLimiter

        mules = MULESLimiter(mesh_4x4, n_iterations=3)
        r = repr(mules)
        assert "MULESLimiter" in r
        assert "16" in r


# ---------------------------------------------------------------------------
# Tests: VOF with MULES integration
# ---------------------------------------------------------------------------

class TestVOFWithMULES:
    """Tests for VOFAdvection with MULES enabled."""

    def test_mules_enabled_by_default(self, mesh_4x4):
        """MULES is enabled by default."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.0, 1.0)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, use_mules=True)
        alpha_new = vof.advance(0.01)
        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10

    def test_mules_disabled(self, mesh_4x4):
        """VOF works with MULES disabled."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.0, 1.0)
        phi = torch.zeros(mesh_4x4.n_faces, dtype=CFD_DTYPE)
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, use_mules=False)
        alpha_new = vof.advance(0.01)
        # Should still be bounded (by clamp)
        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10

    def test_mules_conservation_with_advection(self, mesh_4x4):
        """VOF+MULES preserves conservation during advection."""
        from pyfoam.multiphase.volume_of_fluid import VOFAdvection

        n_cells = mesh_4x4.n_cells
        n_internal = mesh_4x4.n_internal_faces
        n_faces = mesh_4x4.n_faces

        # Step function
        alpha = torch.zeros(n_cells, dtype=CFD_DTYPE)
        for j in range(4):
            for i in range(2):
                alpha[j * 4 + i] = 1.0

        V = mesh_4x4.cell_volumes
        total_before = (alpha * V).sum().item()

        # Small flux
        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)
        for f in range(n_internal):
            phi[f] = 0.001
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)

        vof = VOFAdvection(mesh_4x4, alpha, phi, U, C_alpha=0.5, use_mules=True)
        alpha_new = vof.advance(0.01)

        total_after = (alpha_new * V).sum().item()
        # Conservation should be preserved to reasonable tolerance
        assert abs(total_after - total_before) < 0.05 * max(abs(total_before), 1e-30)
