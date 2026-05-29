"""Tests for apply_boundary_layer_enhanced_2 — enhanced BL application v2."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced_2 import (
    EnhancedBL2Properties,
    EnhancedBL2Result,
    apply_boundary_layer_enhanced_2,
)


class TestEnhancedBL2Properties:
    def test_default_values(self):
        p = EnhancedBL2Properties()
        assert p.delta == 0.1
        assert p.nu == 1e-5
        assert p.wall_function == "standard"

    def test_wall_function_options(self):
        for wf in ["standard", "rough"]:
            p = EnhancedBL2Properties(wall_function=wf)
            assert p.wall_function == wf


class TestApplyBoundaryLayerEnhanced2:
    def test_returns_result_type(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert isinstance(r, EnhancedBL2Result)

    def test_velocity_shape_preserved(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert r.velocity.shape == (n, 3)

    def test_max_y_plus(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert r.max_y_plus >= 0

    def test_n_cells_modified(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert r.n_cells_modified >= 0

    def test_rough_wall_function(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5, wall_function="rough", z0_rough=0.001)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert r.velocity.shape == (n, 3)

    def test_pressure_gradient_correction(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        bl = EnhancedBL2Properties(delta=0.5, dp_dx=-100.0)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl)
        assert r.velocity.shape == (n, 3)

    def test_turbulence_fields(self, fv_mesh):
        n = fv_mesh.n_cells
        U = np.ones((n, 3)) * 10.0
        k_in = np.ones(n) * 0.01
        eps_in = np.ones(n) * 0.001
        omg_in = np.ones(n) * 1.0
        bl = EnhancedBL2Properties(delta=0.5)
        r = apply_boundary_layer_enhanced_2(fv_mesh, U, bl,
                                             k_field=k_in, epsilon_field=eps_in,
                                             omega_field=omg_in)
        assert r.k is not None
        assert r.epsilon is not None
        assert r.omega is not None
