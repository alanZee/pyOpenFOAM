"""Tests for create_patch_enhanced_9 — enhanced patch creation v9."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_9 import (
    PatchEnhanced9Result, PatchVersion, DependencyNode, RepairReport,
    create_patch_enhanced_9,
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


class TestPatchEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_9(m, patch_name="test")
        assert isinstance(r, PatchEnhanced9Result)

    def test_versioning(self):
        m = _single_hex()
        r = create_patch_enhanced_9(m, patch_name="test", enable_versioning=True)
        assert isinstance(r.versions, list)

    def test_dependency_graph(self):
        m = _single_hex()
        r = create_patch_enhanced_9(m, patch_name="test", build_dependency_graph=True)
        assert isinstance(r.dependency_graph, list)

    def test_auto_repair(self):
        m = _single_hex()
        r = create_patch_enhanced_9(m, patch_name="test", auto_repair=True)
        assert isinstance(r.repair_report, RepairReport)
