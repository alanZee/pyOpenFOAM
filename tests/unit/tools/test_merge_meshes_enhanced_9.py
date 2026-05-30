"""Tests for merge_meshes_enhanced_9 — enhanced mesh merging v9."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_9 import (
    MergeEnhanced9Result, QualityCertificate, MergeReport,
    merge_meshes_enhanced_9,
)


def _single_hex(offset=(0, 0, 0)):
    ox, oy, oz = offset
    pts = [[ox,oy,oz],[ox+1,oy,oz],[ox+1,oy+1,oz],[ox,oy+1,oz],
           [ox,oy,oz+1],[ox+1,oy,oz+1],[ox+1,oy+1,oz+1],[ox,oy+1,oz+1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestMergeEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        r = merge_meshes_enhanced_9([m])
        assert isinstance(r, MergeEnhanced9Result)

    def test_quality_certification(self):
        m = _single_hex()
        r = merge_meshes_enhanced_9([m], certify_quality=True)
        assert isinstance(r.certificate, QualityCertificate)
        assert r.certificate.n_criteria_checked > 0

    def test_merge_report(self):
        m = _single_hex()
        r = merge_meshes_enhanced_9([m], generate_report=True)
        assert isinstance(r.report, MergeReport)
        assert r.report.n_input_meshes == 1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_9([])

    def test_incremental_merge(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((5, 0, 0))
        r = merge_meshes_enhanced_9([m2], incremental_base=m1)
        assert r.incremental is True

    def test_custom_certification_criteria(self):
        m = _single_hex()
        r = merge_meshes_enhanced_9(
            [m], certify_quality=True,
            certification_criteria={"min_quality_score": 0.9},
        )
        assert isinstance(r.certificate, QualityCertificate)
