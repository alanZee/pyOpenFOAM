"""
Unit tests for MULES (Multidimensional Universal Limiter with Explicit Solution).

Tests cover:
- Initialization (default and custom n_iterations)
- Alpha bounds enforcement for limit()
- Alpha bounds enforcement for limit_flux()
- Conservation under limiting
- Per-cell alpha_min/alpha_max fields
- Edge cases (zero flux, uniform alpha, extreme flux)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.multiphase.mules import MULESLimiter


# ---------------------------------------------------------------------------
# Minimal mesh fixture (reuse pattern from test_vof.py)
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
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
) -> SimpleMesh:
    """Create a 2D Cartesian mesh for testing."""
    n_cells = n_x * n_y
    owners, neighbours = [], []

    for j in range(n_y):
        for i in range(n_x - 1):
            owners.append(j * n_x + i)
            neighbours.append(j * n_x + i + 1)
    for j in range(n_y - 1):
        for i in range(n_x):
            owners.append(j * n_x + i)
            neighbours.append((j + 1) * n_x + i)

    n_internal = len(neighbours)

    # boundary owners
    for i in range(n_x):
        owners.append(i)
    for i in range(n_x):
        owners.append((n_y - 1) * n_x + i)
    for j in range(n_y):
        owners.append(j * n_x)
    for j in range(n_y):
        owners.append(j * n_x + n_x - 1)

    n_faces = len(owners)

    owner = torch.tensor(owners, dtype=torch.long)
    neigh = torch.tensor(neighbours, dtype=torch.long)

    vol = dx * dy * dz
    cell_volumes = torch.full((n_cells,), vol, dtype=CFD_DTYPE)
    face_areas = torch.zeros(n_faces, 3, dtype=CFD_DTYPE)
    face_weights = torch.full((n_internal,), 0.5, dtype=CFD_DTYPE)
    delta_coeffs = torch.ones(n_internal, dtype=CFD_DTYPE)

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


@pytest.fixture
def mesh():
    return make_2d_mesh(4, 4)


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

class TestMULESInit:
    """MULESLimiter initialization tests."""

    def test_default_iterations(self, mesh):
        """Default n_iterations is 3."""
        mules = MULESLimiter(mesh)
        assert mules._n_iterations == 3

    def test_custom_iterations(self, mesh):
        """Custom n_iterations is stored."""
        mules = MULESLimiter(mesh, n_iterations=7)
        assert mules._n_iterations == 7

    def test_repr(self, mesh):
        """repr contains class name and n_cells."""
        mules = MULESLimiter(mesh)
        r = repr(mules)
        assert "MULESLimiter" in r
        assert str(mesh.n_cells) in r


# ---------------------------------------------------------------------------
# Tests: limit() — boundedness
# ---------------------------------------------------------------------------

class TestMULESLimitBounded:
    """Tests that limit() keeps alpha within [alpha_min, alpha_max]."""

    def test_bounded_default_range(self, mesh):
        """Alpha stays in [0, 1] with default bounds."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        # Large flux to stress-test the limiter
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 10.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.01)

        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10

    def test_bounded_custom_range(self, mesh):
        """Alpha stays in custom [0.2, 0.8] bounds."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 5.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.01,
            alpha_min=0.2, alpha_max=0.8,
        )

        assert alpha_new.min() >= 0.2 - 1e-10
        assert alpha_new.max() <= 0.8 + 1e-10

    def test_bounded_extreme_flux(self, mesh):
        """Alpha stays bounded even with extreme fluxes."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 1000.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.01)

        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10
        assert torch.isfinite(alpha_new).all()

    def test_bounded_mixed_sign_flux(self, mesh):
        """Alpha stays bounded with random positive/negative fluxes."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 5.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.005)

        assert alpha_new.min() >= -1e-10
        assert alpha_new.max() <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Tests: limit() — per-cell bounds
# ---------------------------------------------------------------------------

class TestMULESPerCellBounds:
    """Tests for per-cell alpha_min_field / alpha_max_field."""

    def test_per_cell_bounds(self, mesh):
        """Per-cell bounds override global defaults."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 5.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        # Narrow per-cell bounds
        alpha_min_field = torch.full((n_cells,), 0.3, dtype=CFD_DTYPE)
        alpha_max_field = torch.full((n_cells,), 0.7, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(
            alpha, phi, alpha_flux, 0.005,
            alpha_min_field=alpha_min_field,
            alpha_max_field=alpha_max_field,
        )

        assert alpha_new.min() >= 0.3 - 1e-10
        assert alpha_new.max() <= 0.7 + 1e-10


# ---------------------------------------------------------------------------
# Tests: limit() — conservation
# ---------------------------------------------------------------------------

class TestMULESConservation:
    """Tests that limiting preserves total scalar (approximately)."""

    def test_conservation_small_flux(self, mesh):
        """Total alpha*V conserved for small fluxes."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        V = mesh.cell_volumes
        total_before = (alpha * V).sum().item()

        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 0.001
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=3)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.001)

        total_after = (alpha_new * V).sum().item()
        assert abs(total_after - total_before) < 0.01 * max(abs(total_before), 1e-30)


# ---------------------------------------------------------------------------
# Tests: limit_flux()
# ---------------------------------------------------------------------------

class TestMULESLimitFlux:
    """Tests for the limit_flux() method."""

    def test_flux_bounded(self, mesh):
        """Limited flux magnitude is no larger than original."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 5.0

        mules = MULESLimiter(mesh, n_iterations=3)
        limited = mules.limit_flux(alpha, alpha_flux, 0.01)

        assert limited.abs().max() <= alpha_flux.abs().max() + 1e-10

    def test_flux_shape(self, mesh):
        """limit_flux returns tensor with shape (n_internal,)."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh)
        limited = mules.limit_flux(alpha, alpha_flux, 0.01)

        assert limited.shape == (n_internal,)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestMULESEdgeCases:
    """Edge-case tests."""

    def test_zero_flux_no_change(self, mesh):
        """Zero flux leaves alpha unchanged."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE).clamp(0.1, 0.9)
        alpha_flux = torch.zeros(n_internal, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=3)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.01)

        assert torch.allclose(alpha_new, alpha, atol=1e-10)

    def test_uniform_alpha_preserved(self, mesh):
        """Uniform alpha with zero flux is unchanged."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        alpha_flux = torch.zeros(n_internal, dtype=CFD_DTYPE)
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=3)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.01)

        assert torch.allclose(alpha_new, alpha, atol=1e-10)

    def test_no_op_for_tiny_flux(self, mesh):
        """Tiny fluxes barely modify alpha."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.full((n_cells,), 0.5, dtype=CFD_DTYPE)
        alpha_flux = torch.ones(n_internal, dtype=CFD_DTYPE) * 1e-10
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=3)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.001)

        assert torch.allclose(alpha_new, alpha, atol=0.01)

    def test_output_is_finite(self, mesh):
        """Output never contains NaN or Inf."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        alpha = torch.rand(n_cells, dtype=CFD_DTYPE)
        alpha_flux = torch.randn(n_internal, dtype=CFD_DTYPE) * 50.0
        phi = torch.zeros(mesh.n_faces, dtype=CFD_DTYPE)

        mules = MULESLimiter(mesh, n_iterations=5)
        alpha_new = mules.limit(alpha, phi, alpha_flux, 0.001)

        assert torch.isfinite(alpha_new).all()
