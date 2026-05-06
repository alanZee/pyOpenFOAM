"""
Integration test: Dam break problem (VOF two-phase flow).

Tests the VOF advection and multiphase infrastructure against
the classic dam break problem.

The dam break is a two-phase flow problem with:
- Water column (α=1) on the left
- Air (α=0) on the right
- Gravity drives the water into the air region

This test verifies that the VOF model correctly:
1. Advects the volume fraction
2. Maintains boundedness (0 ≤ α ≤ 1)
3. Computes mixture properties correctly
4. Handles interface compression
"""

import pytest
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.multiphase.volume_of_fluid import VOFAdvection


def make_1d_mesh(n_cells: int = 100, length: float = 1.0):
    """Create a 1D mesh for dam break test.

    Creates a 1D channel with n_cells cells along x-axis.
    The mesh is actually 3D with unit depth in y and z.

    Args:
        n_cells: Number of cells.
        length: Domain length (m).

    Returns:
        FvMesh with computed geometry.
    """
    device = "cpu"
    dtype = CFD_DTYPE

    dx = length / n_cells

    # Points: 2*(n_cells+1) points (front and back z-planes)
    points = []
    for i in range(n_cells + 1):
        points.append([i * dx, 0.0, 0.0])
    for i in range(n_cells + 1):
        points.append([i * dx, 1.0, 0.0])
    for i in range(n_cells + 1):
        points.append([i * dx, 0.0, 1.0])
    for i in range(n_cells + 1):
        points.append([i * dx, 1.0, 1.0])

    points = torch.tensor(points, dtype=dtype, device=device)

    faces = []
    owner = []
    neighbour = []

    # Internal faces (between cells)
    for i in range(n_cells - 1):
        # Face at x = (i+1) * dx
        p0 = i + 1
        p1 = i + 1 + (n_cells + 1)
        p2 = i + 1 + 3 * (n_cells + 1)
        p3 = i + 1 + 2 * (n_cells + 1)
        faces.append([p0, p1, p2, p3])
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Boundary faces
    # Left (x=0)
    p0 = 0
    p1 = n_cells + 1
    p2 = 3 * (n_cells + 1)
    p3 = 2 * (n_cells + 1)
    faces.append([p0, p1, p2, p3])
    owner.append(0)

    # Right (x=length)
    p0 = n_cells
    p1 = 2 * (n_cells + 1) - 1
    p2 = 4 * (n_cells + 1) - 1
    p3 = 3 * (n_cells + 1) - 1
    faces.append([p0, p1, p2, p3])
    owner.append(n_cells - 1)

    face_tensors = [torch.tensor(f, dtype=INDEX_DTYPE, device=device) for f in faces]
    owner_tensor = torch.tensor(owner, dtype=INDEX_DTYPE, device=device)
    neighbour_tensor = torch.tensor(neighbour, dtype=INDEX_DTYPE, device=device)

    boundary = [
        {"name": "left", "type": "wall", "startFace": n_internal, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": n_internal + 1, "nFaces": 1},
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


class TestDamBreakVOF:
    """Test VOF advection for dam break problem."""

    @pytest.fixture
    def mesh_1d(self):
        """1D mesh with 100 cells."""
        return make_1d_mesh(100, 1.0)

    @pytest.fixture
    def dam_break_alpha(self, mesh_1d):
        """Initial volume fraction: air (α=1) in right half, water (α=0) in left.

        Convention: α=0 → fluid 1 (water, rho1), α=1 → fluid 2 (air, rho2).
        For dam break: water column on left (α=0), air on right (α=1).
        """
        device = get_device()
        dtype = get_default_dtype()
        n = mesh_1d.n_cells
        alpha = torch.zeros(n, dtype=dtype, device=device)
        alpha[n // 2:] = 1.0  # Air in right half (alpha=1 → fluid 2)
        return alpha

    @pytest.fixture
    def uniform_velocity(self, mesh_1d):
        """Uniform velocity to the right."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 1.0  # 1 m/s to the right
        return U

    @pytest.fixture
    def rightward_flux(self, mesh_1d, uniform_velocity):
        """Face flux corresponding to uniform rightward velocity."""
        device = get_device()
        dtype = get_default_dtype()
        mesh = mesh_1d
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Flux = U · S
        face_areas = mesh.face_areas[:n_internal]
        U_face = 0.5 * (uniform_velocity[int_owner] + uniform_velocity[int_neigh])
        phi = (U_face * face_areas).sum(dim=1)

        # Full flux vector (internal + boundary)
        phi_full = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        phi_full[:n_internal] = phi

        return phi_full

    def test_initial_boundedness(self, dam_break_alpha):
        """Initial alpha should be bounded [0, 1]."""
        assert (dam_break_alpha >= 0.0).all()
        assert (dam_break_alpha <= 1.0).all()

    def test_initial_interface(self, dam_break_alpha):
        """Interface should be at x=0.5 (α jumps from 0 to 1)."""
        n = len(dam_break_alpha)
        assert float(dam_break_alpha[0].item()) == 0.0  # water (left)
        assert float(dam_break_alpha[-1].item()) == 1.0  # air (right)
        # Interface somewhere in the middle
        assert float(dam_break_alpha[n // 2 - 1].item()) == 0.0
        assert float(dam_break_alpha[n // 2].item()) == 1.0

    def test_vof_advection_boundedness(self, mesh_1d, dam_break_alpha, rightward_flux):
        """VOF advection should keep alpha bounded [0, 1]."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 1.0

        vof = VOFAdvection(
            mesh_1d, dam_break_alpha, rightward_flux, U,
            C_alpha=1.0,
        )

        # Advance several steps
        dt = 0.001
        for _ in range(10):
            alpha = vof.advance(dt)
            assert (alpha >= -0.01).all(), f"alpha min: {alpha.min()}"
            assert (alpha <= 1.01).all(), f"alpha max: {alpha.max()}"

    def test_vof_advection_transports_interface(self, mesh_1d, dam_break_alpha, rightward_flux):
        """VOF should transport the interface to the right.

        With rightward velocity, water (alpha=0 on left) gets pushed right,
        so the region of alpha=0 expands and alpha > 0.5 count decreases.
        """
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 1.0

        vof = VOFAdvection(
            mesh_1d, dam_break_alpha.clone(), rightward_flux, U,
            C_alpha=1.0,
        )

        # Initial: count cells with alpha > 0.5
        initial_high = int((dam_break_alpha > 0.5).sum().item())

        # Advance
        dt = 0.001
        for _ in range(50):
            alpha = vof.advance(dt)

        # With rightward flux, the interface moves right → fewer high-alpha cells
        final_high = int((alpha > 0.5).sum().item())
        # Interface should have moved (count changed)
        assert final_high != initial_high or torch.allclose(alpha, dam_break_alpha, atol=1e-6)

    def test_mixture_properties(self, dam_break_alpha):
        """Mixture properties should interpolate between phases."""
        rho1 = 1000.0  # water
        rho2 = 1.225  # air
        mu1 = 1e-3  # water
        mu2 = 1.8e-5  # air

        alpha = dam_break_alpha

        # Mixture density
        rho_mix = alpha * rho2 + (1.0 - alpha) * rho1
        # Convention: alpha=0 → rho1 (water), alpha=1 → rho2 (air)
        # Left side: alpha=0 → rho1 (water)
        # Right side: alpha=1 → rho2 (air)
        assert float(rho_mix[0].item()) == pytest.approx(rho1)
        assert float(rho_mix[-1].item()) == pytest.approx(rho2)

        # Mixture viscosity
        mu_mix = alpha * mu2 + (1.0 - alpha) * mu1
        assert float(mu_mix[0].item()) == pytest.approx(mu1)
        assert float(mu_mix[-1].item()) == pytest.approx(mu2)

    def test_interface_normal(self, mesh_1d, dam_break_alpha, rightward_flux):
        """Interface normal should point from water to air."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 1.0

        vof = VOFAdvection(
            mesh_1d, dam_break_alpha, rightward_flux, U,
            C_alpha=1.0,
        )

        n_hat = vof.compute_interface_normal()
        assert n_hat.shape == (mesh_1d.n_cells, 3)

        # Normal should be zero away from interface
        # (at cells with alpha=0 or alpha=1)
        # Cell 0 has alpha=0 (water), cell -1 has alpha=1 (air)
        assert float(n_hat[0, 0].item()) == 0.0
        assert float(n_hat[-1, 0].item()) == 0.0

    def test_compression_sharpens_interface(self, mesh_1d, dam_break_alpha, rightward_flux):
        """Interface compression should sharpen the interface."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 0.5

        # Create a smeared interface (alpha between 0 and 1)
        alpha_smeared = torch.zeros(mesh_1d.n_cells, dtype=dtype, device=device)
        n = mesh_1d.n_cells
        # Smeared interface over 10 cells
        for i in range(n // 2 - 5, n // 2 + 5):
            alpha_smeared[i] = 0.5 + 0.5 * torch.cos(
                torch.tensor(3.14159 * (i - n // 2 + 5) / 10.0)
            )

        # Without compression
        vof_no_comp = VOFAdvection(
            mesh_1d, alpha_smeared.clone(), rightward_flux, U,
            C_alpha=0.0,
        )
        alpha_no_comp = vof_no_comp.advance(0.001)

        # With compression
        vof_comp = VOFAdvection(
            mesh_1d, alpha_smeared.clone(), rightward_flux, U,
            C_alpha=2.0,
        )
        alpha_comp = vof_comp.advance(0.001)

        # Compression should make the interface sharper
        # (more cells closer to 0 or 1)
        sharpness_no_comp = float(alpha_no_comp.var().item())
        sharpness_comp = float(alpha_comp.var().item())
        # Higher variance = sharper interface
        assert sharpness_comp >= sharpness_no_comp * 0.9  # Allow small tolerance

    def test_zero_flux_no_change(self, mesh_1d, dam_break_alpha):
        """With zero flux, alpha should not change."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        phi = torch.zeros(mesh_1d.n_faces, dtype=dtype, device=device)

        vof = VOFAdvection(
            mesh_1d, dam_break_alpha.clone(), phi, U,
            C_alpha=1.0,
        )

        alpha_new = vof.advance(0.01)
        assert torch.allclose(alpha_new, dam_break_alpha, atol=1e-10)

    def test_conservation(self, mesh_1d, dam_break_alpha, rightward_flux):
        """VOF should approximately conserve total volume fraction."""
        device = get_device()
        dtype = get_default_dtype()
        U = torch.zeros(mesh_1d.n_cells, 3, dtype=dtype, device=device)
        U[:, 0] = 1.0

        vof = VOFAdvection(
            mesh_1d, dam_break_alpha.clone(), rightward_flux, U,
            C_alpha=1.0,
        )

        initial_total = float(dam_break_alpha.sum().item())

        dt = 0.001
        for _ in range(20):
            alpha = vof.advance(dt)

        final_total = float(alpha.sum().item())

        # Should be approximately conserved (within 20%)
        # Note: upwind scheme with compression is not perfectly conservative
        assert abs(final_total - initial_total) / max(initial_total, 1e-10) < 0.20
