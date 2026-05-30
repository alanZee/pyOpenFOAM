"""Tests for stitch_mesh_enhanced_8 — enhanced mesh stitching v8."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_8 import StitchEnhanced8Result, StitchStrength, stitch_mesh_enhanced_8


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


class TestStitchEnhanced8:
    def test_returns_result_type(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_8(m, "all", "all")
        assert isinstance(r, StitchEnhanced8Result)

    def test_pattern_matching(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_8(m, "all", "all", pattern_matching=True)
        assert isinstance(r.patterns, list)

    def test_strength_analysis(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_8(m, "all", "all", analyze_strength=True)
        assert isinstance(r.strength, StitchStrength)
        assert r.strength.strength_score >= 0

    def test_topology_score(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_8(m, "all", "all", topology_aware=True)
        assert 0.0 <= r.topology_score <= 1.0
