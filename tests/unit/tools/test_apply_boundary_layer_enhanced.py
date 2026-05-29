"""Tests for apply_boundary_layer_enhanced — enhanced BL application."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced import (
    EnhancedBLProperties,
    EnhancedBLResult,
    apply_boundary_layer_enhanced,
)


class TestEnhancedBLProperties:
    def test_default_values(self):
        bl = EnhancedBLProperties()
        assert bl.delta == 0.1
        assert bl.nu == 1e-5
        assert bl.z0_rough == 0.0
        assert bl.blend_width == 0.2

    def test_custom_values(self):
        bl = EnhancedBLProperties(delta=0.5, nu=1e-3, z0_rough=0.001)
        assert bl.delta == 0.5
        assert bl.z0_rough == 0.001


class TestApplyBoundaryLayerEnhanced:
    def test_returns_result_type(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert isinstance(r, EnhancedBLResult)

    def test_velocity_shape_preserved(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert r.velocity.shape == U.shape

    def test_u_star_estimated(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3)) * 5.0
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert r.u_star_used > 0

    def test_explicit_u_star(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5, u_star=0.1)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert r.u_star_used == 0.1

    def test_k_field_correction(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        k = np.ones(fv_mesh.n_cells) * 0.01
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl, k_field=k)
        assert r.k is not None
        assert r.k.shape == k.shape

    def test_epsilon_field_correction(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        eps = np.ones(fv_mesh.n_cells) * 0.001
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl, epsilon_field=eps)
        assert r.epsilon is not None

    def test_omega_field_correction(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        omg = np.ones(fv_mesh.n_cells) * 10.0
        k = np.ones(fv_mesh.n_cells) * 0.01
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl, k_field=k, omega_field=omg)
        assert r.omega is not None

    def test_rough_wall(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5, z0_rough=0.01)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert r.velocity.shape == U.shape

    def test_no_wall_patches(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3))
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        r = apply_boundary_layer_enhanced(fv_mesh, U, bl, wall_patches=["nonexistent"])
        assert r.u_star_used == 0.0

    def test_does_not_modify_input(self, fv_mesh):
        U = np.ones((fv_mesh.n_cells, 3)) * 5.0
        U_copy = U.copy()
        bl = EnhancedBLProperties(delta=10.0, nu=1e-5)
        apply_boundary_layer_enhanced(fv_mesh, U, bl)
        assert np.allclose(U, U_copy)
