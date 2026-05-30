"""Tests for apply_boundary_layer_enhanced_3 — enhanced BL application v3."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced_3 import (
    EnhancedBL3Properties,
    EnhancedBL3Result,
    apply_boundary_layer_enhanced_3,
)


class TestEnhancedBL3Properties:
    def test_default_values(self):
        p = EnhancedBL3Properties()
        assert p.delta == 0.1
        assert p.nu == 1e-5
        assert p.wall_function == "standard"
        assert p.reference_U is None

    def test_wall_function_options(self):
        for wf in ["standard", "rough", "multi_layer"]:
            p = EnhancedBL3Properties(wall_function=wf)
            assert p.wall_function == wf


class TestApplyBoundaryLayerEnhanced3:
    def test_returns_result_type(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        assert isinstance(r, EnhancedBL3Result)

    def test_velocity_shape_preserved(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        assert r.velocity.shape == vel.shape

    def test_multi_layer_function(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5, wall_function="multi_layer")
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        assert r.n_cells_modified >= 0

    def test_shape_factor_reported(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        # Shape factor H = delta_star / theta
        assert isinstance(r.shape_factor, float)

    def test_displacement_thickness(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        assert isinstance(r.displacement_thickness, float)

    def test_momentum_thickness(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["bottom"])
        assert isinstance(r.momentum_thickness, float)

    def test_no_wall_patches_returns_early(self, fv_mesh):
        bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
        vel = np.ones((fv_mesh.n_cells, 3)) * 10.0
        r = apply_boundary_layer_enhanced_3(fv_mesh, vel, bl, wall_patches=["nonexistent"])
        assert r.n_cells_modified == 0
