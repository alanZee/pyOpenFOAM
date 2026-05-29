"""Tests for enhanced PLIC (Piecewise Linear Interface Calculation).

Tests cover:
- PLICReconstruction constructor
- Interface cell identification
- Normal estimation (Youngs method)
- Plane constant computation (bisection)
- Full reconstruction (normal + plane constant)
- Volume flux computation
"""

import pytest
import torch

from pyfoam.multiphase.pllic import PLICReconstruction


class TestPLICReconstructionInit:
    """Constructor tests."""

    def test_defaults(self):
        plic = PLICReconstruction()
        assert plic.alpha_tol == pytest.approx(1e-6)
        assert plic.max_bisection_iter == 40

    def test_custom_params(self):
        plic = PLICReconstruction(alpha_tol=1e-4, max_bisection_iter=20)
        assert plic.alpha_tol == pytest.approx(1e-4)
        assert plic.max_bisection_iter == 20


class TestInterfaceCells:
    """Interface cell identification tests."""

    def test_full_and_empty_not_interface(self):
        plic = PLICReconstruction(alpha_tol=1e-6)
        alpha = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        mask = plic.interface_cells(alpha)
        assert not mask.any()

    def test_half_is_interface(self):
        plic = PLICReconstruction()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        mask = plic.interface_cells(alpha)
        assert mask[0]

    def test_near_zero_not_interface(self):
        plic = PLICReconstruction(alpha_tol=1e-6)
        alpha = torch.tensor([1e-8], dtype=torch.float64)
        mask = plic.interface_cells(alpha)
        assert not mask[0]

    def test_near_one_not_interface(self):
        plic = PLICReconstruction(alpha_tol=1e-6)
        alpha = torch.tensor([1.0 - 1e-8], dtype=torch.float64)
        mask = plic.interface_cells(alpha)
        assert not mask[0]

    def test_multiple_interface_cells(self):
        plic = PLICReconstruction()
        alpha = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
        mask = plic.interface_cells(alpha)
        assert mask[1] and mask[2] and mask[3]
        assert not mask[0] and not mask[4]


class TestNormalEstimation:
    """Normal estimation tests."""

    def test_shape(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals = plic.estimate_normal(alpha)
        assert normals.shape == (10, 3)

    def test_unit_length_for_interface_cells(self):
        plic = PLICReconstruction()
        alpha = torch.rand(20, dtype=torch.float64) * 0.8 + 0.1
        normals = plic.estimate_normal(alpha)
        for i in range(20):
            mag = normals[i].norm().item()
            assert abs(mag - 1.0) < 1e-6 or mag < 1e-10

    def test_zero_for_non_interface(self):
        plic = PLICReconstruction()
        alpha = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0], dtype=torch.float64)
        normals = plic.estimate_normal(alpha)
        assert torch.allclose(normals[0], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(normals[1], torch.zeros(3, dtype=torch.float64))
        assert normals[2].norm() > 0.5  # interface cell has non-zero normal

    def test_custom_gradient(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        grad = torch.randn(10, 3, dtype=torch.float64)
        normals = plic.estimate_normal(alpha, grad_alpha=grad)
        assert normals.shape == (10, 3)
        # Should match normalised gradient for interface cells
        for i in range(10):
            if 1e-6 < alpha[i] < 1 - 1e-6:
                expected = grad[i] / grad[i].norm()
                assert torch.allclose(normals[i], expected, atol=1e-10)


class TestPlaneConstant:
    """Plane constant computation tests."""

    def test_shape(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals = torch.randn(10, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=1, keepdim=True)
        d = plic.compute_plane_constant(alpha, normals)
        assert d.shape == (10,)

    def test_zero_for_non_interface(self):
        plic = PLICReconstruction()
        alpha = torch.tensor([0.0, 1.0, 0.5, 0.0], dtype=torch.float64)
        normals = torch.randn(4, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=1, keepdim=True)
        d = plic.compute_plane_constant(alpha, normals)
        assert d[0] == pytest.approx(0.0)
        assert d[1] == pytest.approx(0.0)
        assert d[3] == pytest.approx(0.0)

    def test_nonzero_for_interface_cells(self):
        plic = PLICReconstruction()
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        normals = torch.zeros(3, 3, dtype=torch.float64)
        normals[1, 0] = 1.0  # normal in x-direction
        d = plic.compute_plane_constant(alpha, normals)
        assert d[1] != 0.0

    def test_with_cell_volumes(self):
        """Should work with explicit cell volumes."""
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals = torch.randn(10, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=1, keepdim=True)
        V = torch.rand(10, dtype=torch.float64) * 1e-3 + 1e-6
        d = plic.compute_plane_constant(alpha, normals, cell_volumes=V)
        assert d.shape == (10,)
        assert torch.isfinite(d).all()

    def test_finite(self):
        plic = PLICReconstruction()
        alpha = torch.rand(50, dtype=torch.float64) * 0.8 + 0.1
        normals = torch.randn(50, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=1, keepdim=True)
        d = plic.compute_plane_constant(alpha, normals)
        assert torch.isfinite(d).all()


class TestReconstruct:
    """Full reconstruction tests."""

    def test_returns_tuple(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals, d = plic.reconstruct(alpha)
        assert normals.shape == (10, 3)
        assert d.shape == (10,)

    def test_consistency(self):
        """reconstruct() should give same result as individual calls."""
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals1, d1 = plic.reconstruct(alpha)
        normals2 = plic.estimate_normal(alpha)
        d2 = plic.compute_plane_constant(alpha, normals2)
        assert torch.allclose(normals1, normals2)
        assert torch.allclose(d1, d2)


class TestVolumeFlux:
    """Face volume flux computation tests."""

    def test_shape(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals, d = plic.reconstruct(alpha)
        U_face = torch.randn(15, 3, dtype=torch.float64) * 0.1
        A = torch.randn(15, 3, dtype=torch.float64) * 0.01
        flux = plic.compute_face_volume_flux(alpha, normals, d, U_face, A, dt=0.001)
        assert flux.shape == (15,)

    def test_finite(self):
        plic = PLICReconstruction()
        alpha = torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        normals, d = plic.reconstruct(alpha)
        U_face = torch.randn(20, 3, dtype=torch.float64) * 0.1
        A = torch.randn(20, 3, dtype=torch.float64) * 0.01
        flux = plic.compute_face_volume_flux(alpha, normals, d, U_face, A, dt=0.001)
        assert torch.isfinite(flux).all()
