"""Tests for set_atm_boundary_layer_enhanced_2 — enhanced ABL v2."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced_2 import (
    EnhancedABL2Properties,
    EnhancedABL2Result,
    set_atm_boundary_layer_enhanced_2,
)


class TestEnhancedABL2Properties:
    def test_default_values(self):
        p = EnhancedABL2Properties()
        assert p.u_star == 0.5
        assert p.z0 == 0.01
        assert p.model == "neutral"

    def test_model_options(self):
        for m in ["neutral", "stable", "unstable", "power"]:
            p = EnhancedABL2Properties(model=m)
            assert p.model == m

    def test_power_exponent(self):
        p = EnhancedABL2Properties(model="power", power_exponent=0.2)
        assert p.power_exponent == 0.2


class TestSetAtmBoundaryLayerEnhanced2:
    def test_returns_result_type(self, fv_mesh):
        abl = EnhancedABL2Properties()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert isinstance(r, EnhancedABL2Result)

    def test_U_shape(self, fv_mesh):
        abl = EnhancedABL2Properties()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_k_shape(self, fv_mesh):
        abl = EnhancedABL2Properties()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert r.k.shape == (fv_mesh.n_cells,)

    def test_intensity_shape(self, fv_mesh):
        abl = EnhancedABL2Properties()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert r.intensity.shape == (fv_mesh.n_cells,)
        assert np.all(r.intensity >= 0)

    def test_stable_model(self, fv_mesh):
        abl = EnhancedABLProperties_stable()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_power_model(self, fv_mesh):
        abl = EnhancedABL2Properties(model="power", U_ref=10.0, z_ref=10.0)
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl)
        assert r.U.shape == (fv_mesh.n_cells, 3)

    def test_reynolds_stress(self, fv_mesh):
        abl = EnhancedABL2Properties()
        r = set_atm_boundary_layer_enhanced_2(fv_mesh, abl, compute_reynolds_stress=True)
        assert r.reynolds_stress is not None
        assert r.reynolds_stress.shape == (fv_mesh.n_cells, 6)


def EnhancedABLProperties_stable():
    return EnhancedABL2Properties(
        model="stable", L_Monin=100.0,
    )
