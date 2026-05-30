"""Tests for stitch_mesh_enhanced_9 — enhanced mesh stitching v9."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_9 import (
    StitchEnhanced9Result, SymmetryInfo, RecoveryRecord, StitchOptimization,
    stitch_mesh_enhanced_9,
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


class TestStitchEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_9(m, "all", "all")
        assert isinstance(r, StitchEnhanced9Result)

    def test_symmetry_detection(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_9(m, "all", "all", detect_symmetry=True)
        assert isinstance(r.symmetry, SymmetryInfo)

    def test_recovery(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_9(m, "all", "all", enable_recovery=True)
        assert isinstance(r.recovery, RecoveryRecord)

    def test_optimization(self):
        m = _single_hex()
        r = stitch_mesh_enhanced_9(
            m, "all", "all", optimize_order=True,
            multi_patch_pairs=[("a", "b"), ("c", "d")],
        )
        assert isinstance(r.optimisation, StitchOptimization)
