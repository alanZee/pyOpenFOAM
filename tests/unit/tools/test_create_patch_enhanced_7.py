"""Tests for create_patch_enhanced_7 — enhanced patch creation v7."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_7 import PatchEnhanced7Result, QualityImpact, PatchDependency, create_patch_enhanced_7


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


class TestPatchEnhanced7:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0, 1], patch_name="test")
        assert isinstance(r, PatchEnhanced7Result)

    def test_quality_impact(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0], analyze_quality=True)
        assert isinstance(r.quality_impact, QualityImpact)

    def test_bc_hints(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0], template="inlet", suggest_bc=True)
        assert len(r.bc_hints) > 0
        assert any("fixedValue" in h for h in r.bc_hints)

    def test_bc_hints_empty_without_template(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0], suggest_bc=True)
        assert len(r.bc_hints) == 0

    def test_dependencies(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0, 1], patch_name="dep_test")
        assert isinstance(r.dependencies, list)

    def test_quality_no_impact_by_default(self):
        m = _single_hex()
        r = create_patch_enhanced_7(m, face_indices=[0])
        assert isinstance(r.quality_impact, QualityImpact)
        assert r.quality_impact.n_degraded_cells >= 0
