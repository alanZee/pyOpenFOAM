"""Tests for set_atm_boundary_layer_enhanced — enhanced ABL."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced import (
    EnhancedABLProperties,
    EnhancedABLResult,
    set_atm_boundary_layer_enhanced,
)


class TestEnhancedABLProperties:
    def test_default_values(self):
        abl = EnhancedABLProperties()
        assert abl.u_star == 0.5
        assert abl.z0 == 0.01
        assert abl.model == "neutral"

    def test_model_options(self):
        for m in ["neutral", "stable", "unstable"]:
            abl = EnhancedABLProperties(model=m)
            assert abl.model == m


class TestSetAtmBoundaryLayerEnhanced:
    def test_returns_result_type(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert isinstance(r, EnhancedABLResult)

    def test_velocity_shape(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_tke_shape(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert r.k.shape == (fv_mesh.n_cells,)
        assert np.all(r.k > 0)

    def test_epsilon_positive(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert np.all(r.epsilon > 0)

    def test_omega_positive(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert np.all(r.omega > 0)

    def test_length_scale_positive(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert np.all(r.length_scale > 0)

    def test_stable_model(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01, model="stable", L_Monin=100.0)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_unstable_model(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01, model="unstable", L_Monin=-50.0)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_custom_direction(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01, direction=(0.0, 1.0, 0.0))
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl)
        # Velocity should be mainly in y-direction
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_custom_z_axis(self, fv_mesh):
        abl = EnhancedABLProperties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced(fv_mesh, abl, z_axis=2)
        assert r.U.shape == (fv_mesh.n_cells, 3)
