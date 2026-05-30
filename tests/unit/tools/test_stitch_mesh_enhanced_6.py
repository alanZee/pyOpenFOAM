"""Tests for stitch_mesh_enhanced_6 — enhanced mesh stitching v6."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_6 import StitchEnhanced6Result, stitch_mesh_enhanced_6


def _two_patch_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[
                   {"name": "left", "type": "wall", "startFace": 0, "nFaces": 1},
                   {"name": "right", "type": "wall", "startFace": 1, "nFaces": 1},
                   {"name": "rest", "type": "wall", "startFace": 2, "nFaces": 4},
               ],
               validate=False)
    m.compute_geometry()
    return m


class TestStitchEnhanced6:
    def test_returns_result_type(self):
        m = _two_patch_hex()
        r = stitch_mesh_enhanced_6(m, "left", "right")
        assert isinstance(r, StitchEnhanced6Result)

    def test_stitch_verification(self):
        m = _two_patch_hex()
        r = stitch_mesh_enhanced_6(m, "left", "right", verify_stitch=True)
        assert isinstance(r.stitch_verified, bool)

    def test_bl_preservation(self):
        m = _two_patch_hex()
        r = stitch_mesh_enhanced_6(m, "left", "right", preserve_bl=True)
        assert r.bl_cells_preserved >= 0

    def test_multi_patch_pairs(self):
        m = _two_patch_hex()
        r = stitch_mesh_enhanced_6(
            m, "left", "right",
            multi_patch_pairs=[],
        )
        assert r.multi_stitch_pairs == 0

    def test_quality_score(self):
        m = _two_patch_hex()
        r = stitch_mesh_enhanced_6(m, "left", "right")
        assert 0.0 <= r.mean_quality <= 1.0

    def test_patch_not_found_raises(self):
        m = _two_patch_hex()
        with pytest.raises(ValueError, match="not found"):
            stitch_mesh_enhanced_6(m, "left", "nonexistent")
