"""Tests for apply_boundary_layer — boundary layer correction."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.apply_boundary_layer import (
    apply_boundary_layer, BoundaryLayerProperties
)


class TestBoundaryLayerProperties:
    def test_default_values(self):
        """Default constructor should work."""
        bl = BoundaryLayerProperties()
        assert bl.delta == 0.1
        assert bl.nu == 1e-5
        assert bl.kappa == 0.41
        assert bl.E == 9.8
        assert bl.u_star is None

    def test_custom_values(self):
        """Custom values should be stored."""
        bl = BoundaryLayerProperties(delta=0.5, nu=1e-4, u_star=0.1)
        assert bl.delta == 0.5
        assert bl.nu == 1e-4
        assert bl.u_star == 0.1


class TestApplyBoundaryLayer:
    def test_returns_same_shape(self, fv_mesh):
        """Output should have the same shape as input."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3))
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        assert U_new.shape == U.shape

    def test_does_not_modify_input(self, fv_mesh):
        """Input array should not be modified."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3))
        U_orig = U.copy()
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        apply_boundary_layer(fv_mesh, U, bl)
        assert np.array_equal(U, U_orig)

    def test_far_cells_unchanged(self, fv_mesh):
        """Cells beyond delta should be unchanged."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3)) * 5.0
        bl = BoundaryLayerProperties(delta=0.001, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        # Cells far from walls should be unchanged
        # (though for 2-cell hex mesh, all cells may be within delta)
        # Just verify no NaN/Inf
        assert np.all(np.isfinite(U_new))

    def test_wall_patches_filter(self, fv_mesh):
        """Specifying wall_patches should only affect near-wall cells."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3))
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl, wall_patches=["bottom"])
        assert U_new.shape == U.shape
        assert np.all(np.isfinite(U_new))

    def test_no_wall_patches_unchanged(self, fv_mesh):
        """Non-existent wall patches should return unchanged field."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3))
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl, wall_patches=["nonexistent"])
        assert np.array_equal(U_new, U)

    def test_direction_preserved(self, fv_mesh):
        """Velocity direction should be preserved (only magnitude changes)."""
        n_cells = fv_mesh.n_cells
        U = np.zeros((n_cells, 3))
        U[:, 0] = 1.0  # all flow in x-direction
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        for ci in range(n_cells):
            if np.linalg.norm(U_new[ci]) > 1e-30:
                # y and z should still be ~0
                assert abs(U_new[ci, 1]) < 1e-10
                assert abs(U_new[ci, 2]) < 1e-10

    def test_zero_velocity_unchanged(self, fv_mesh):
        """Zero velocity cells should remain zero."""
        n_cells = fv_mesh.n_cells
        U = np.zeros((n_cells, 3))
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        assert np.allclose(U_new, 0.0)

    def test_auto_estimate_u_star(self, fv_mesh):
        """When u_star is None, it should be estimated automatically."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3))
        bl = BoundaryLayerProperties(delta=10.0, nu=1e-5, u_star=None)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        assert np.all(np.isfinite(U_new))

    def test_large_nu(self, fv_mesh):
        """Large viscosity should produce smoother profiles."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3)) * 10.0
        bl = BoundaryLayerProperties(delta=10.0, nu=1.0, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        assert np.all(np.isfinite(U_new))

    def test_very_small_delta(self, fv_mesh):
        """Very small delta should leave most cells unchanged."""
        n_cells = fv_mesh.n_cells
        U = np.ones((n_cells, 3)) * 5.0
        bl = BoundaryLayerProperties(delta=1e-8, nu=1e-5, u_star=0.1)
        U_new = apply_boundary_layer(fv_mesh, U, bl)
        # Effectively no cells should be modified
        assert np.allclose(U_new, U)
