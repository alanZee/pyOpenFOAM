"""Tests for set_atm_boundary_layer_enhanced_3 — enhanced ABL v3."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced_3 import (
    EnhancedABL3Properties,
    EnhancedABL3Result,
    set_atm_boundary_layer_enhanced_3,
)


class TestEnhancedABL3Properties:
    def test_default_values(self):
        p = EnhancedABL3Properties()
        assert p.u_star == 0.5
        assert p.model == "neutral"
        assert p.coriolis_parameter == 1e-4
        assert p.geostrophic_height == 1000.0

    def test_model_options(self):
        for m in ["neutral", "stable", "unstable", "power", "deaves_harris", "ekman"]:
            p = EnhancedABL3Properties(model=m)
            assert p.model == m


class TestSetAtmBoundaryLayerEnhanced3:
    def test_returns_result_type(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert isinstance(r, EnhancedABL3Result)

    def test_U_shape(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_k_shape(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.k.shape == (fv_mesh.n_cells,)

    def test_boundary_layer_height(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.boundary_layer_height > 0

    def test_geostrophic_wind(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.geostrophic_wind > 0

    def test_profile_quality(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert 0.0 <= r.profile_quality <= 1.0

    def test_deaves_harris_model(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01, model="deaves_harris")
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_ekman_model(self, fv_mesh):
        abl = EnhancedABL3Properties(
            u_star=0.5, z0=0.01, model="ekman", coriolis_parameter=1e-4,
        )
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_reynolds_stress(self, fv_mesh):
        abl = EnhancedABL3Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_3(fv_mesh, abl, compute_reynolds_stress=True)
        assert r.reynolds_stress is not None
        assert r.reynolds_stress.shape == (fv_mesh.n_cells, 6)
