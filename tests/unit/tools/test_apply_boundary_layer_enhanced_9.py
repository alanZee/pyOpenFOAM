"""Tests for apply_boundary_layer_enhanced_9 — enhanced BL application v9."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_9 import (
    EnhancedBL9Properties, EnhancedBL9Result, AdaptiveRefinement,
    ScalarCoupling, ActiveControlResult,
    apply_boundary_layer_enhanced_9,
)


def _single_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestApplyBoundaryLayerEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        U = np.zeros((1, 3))
        bl = EnhancedBL9Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_9(m, U, bl)
        assert isinstance(r, EnhancedBL9Result)

    def test_adaptive_refinement(self):
        m = _single_hex()
        U = np.zeros((1, 3))
        bl = EnhancedBL9Properties(delta=0.1, nu=1e-5, adaptive_refinement=True, target_y_plus=1.0)
        r = apply_boundary_layer_enhanced_9(m, U, bl)
        assert isinstance(r.refinement, AdaptiveRefinement)

    def test_scalar_coupling(self):
        m = _single_hex()
        U = np.zeros((1, 3))
        bl = EnhancedBL9Properties(delta=0.1, nu=1e-5, scalar_transport=True, scalar_name="T")
        r = apply_boundary_layer_enhanced_9(m, U, bl)
        assert isinstance(r.scalar, ScalarCoupling)
        assert r.scalar.scalar_name == "T"

    def test_active_control(self):
        m = _single_hex()
        U = np.zeros((1, 3))
        bl = EnhancedBL9Properties(
            delta=0.1, nu=1e-5, active_control=True,
            control_type="suction", control_velocity=2.0,
        )
        r = apply_boundary_layer_enhanced_9(m, U, bl)
        assert isinstance(r.active_control, ActiveControlResult)
        assert r.active_control.control_type == "suction"
