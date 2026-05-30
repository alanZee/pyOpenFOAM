"""Tests for stitch_mesh_enhanced_3 — enhanced mesh stitching v3."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_3 import StitchEnhanced3Result, stitch_mesh_enhanced_3


def _two_cell_mesh_with_patches():
    pts = [
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ]
    int_faces = [[1,2,6,5]]
    owner_i = [0]
    nbr_i = [1]

    bnd_faces = [
        [0,3,2,1], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,4,7,3],
        [1,8,10,5], [6,7,11,10], [8,9,11,10],
    ]
    bnd_owner = [0,0,0,0,0,1,1,1]

    faces = [torch.tensor(f, dtype=INDEX_DTYPE) for f in int_faces + bnd_faces]
    owner = torch.tensor(owner_i + bnd_owner, dtype=INDEX_DTYPE)
    neighbour = torch.tensor(nbr_i, dtype=INDEX_DTYPE)

    boundary = [
        {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 1},
        {"name": "top", "type": "wall", "startFace": 2, "nFaces": 1},
        {"name": "front", "type": "wall", "startFace": 3, "nFaces": 1},
        {"name": "back0", "type": "wall", "startFace": 4, "nFaces": 1},
        {"name": "left", "type": "wall", "startFace": 5, "nFaces": 1},
        {"name": "patch1", "type": "wall", "startFace": 6, "nFaces": 1},
        {"name": "patch2", "type": "wall", "startFace": 7, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": 8, "nFaces": 1},
    ]

    m = FvMesh(
        points=torch.tensor(pts, dtype=torch.float64),
        faces=faces, owner=owner, neighbour=neighbour,
        boundary=boundary, validate=False,
    )
    m.compute_geometry()
    return m


class TestStitchEnhanced3:
    def test_returns_result_type(self):
        m = _two_cell_mesh_with_patches()
        r = stitch_mesh_enhanced_3(m, "patch1", "patch2")
        assert isinstance(r, StitchEnhanced3Result)

    def test_mean_match_quality_range(self):
        m = _two_cell_mesh_with_patches()
        r = stitch_mesh_enhanced_3(m, "patch1", "patch2")
        if r.n_stitched > 0:
            assert 0.0 <= r.mean_match_quality <= 1.0

    def test_bb_filtered_non_negative(self):
        m = _two_cell_mesh_with_patches()
        r = stitch_mesh_enhanced_3(m, "patch1", "patch2")
        assert r.n_bb_filtered >= 0

    def test_non_conformal_mode(self):
        m = _two_cell_mesh_with_patches()
        r = stitch_mesh_enhanced_3(m, "patch1", "patch2", non_conformal=True)
        assert isinstance(r, StitchEnhanced3Result)

    def test_auto_tolerance(self):
        m = _two_cell_mesh_with_patches()
        r = stitch_mesh_enhanced_3(m, "patch1", "patch2", auto_tolerance=True)
        assert r.used_tolerance > 0

    def test_patch_not_found_raises(self):
        m = _two_cell_mesh_with_patches()
        with pytest.raises(ValueError, match="not found"):
            stitch_mesh_enhanced_3(m, "nonexistent", "patch2")
