"""Tests for stitch_mesh_enhanced_7 — enhanced mesh stitching v7."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_7 import StitchEnhanced7Result, StitchQualityGrade, stitch_mesh_enhanced_7


def _two_cell_hex():
    pts2 = torch.tensor([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ], dtype=torch.float64)
    faces2 = [
        torch.tensor([1,2,6,5], dtype=INDEX_DTYPE),
        torch.tensor([0,3,2,1], dtype=INDEX_DTYPE),
        torch.tensor([4,5,6,7], dtype=INDEX_DTYPE),
        torch.tensor([0,1,5,4], dtype=INDEX_DTYPE),
        torch.tensor([3,7,6,2], dtype=INDEX_DTYPE),
        torch.tensor([0,4,7,3], dtype=INDEX_DTYPE),
        torch.tensor([1,5,10,8], dtype=INDEX_DTYPE),
        torch.tensor([2,9,11,6], dtype=INDEX_DTYPE),
        torch.tensor([1,8,9,2], dtype=INDEX_DTYPE),
        torch.tensor([8,10,11,9], dtype=INDEX_DTYPE),
    ]
    owner2 = torch.tensor([0,0,0,0,0,0,1,1,1,1], dtype=INDEX_DTYPE)
    neighbour2 = torch.tensor([1], dtype=INDEX_DTYPE)
    boundary2 = [
        {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 1},
        {"name": "top", "type": "wall", "startFace": 2, "nFaces": 1},
        {"name": "front0", "type": "wall", "startFace": 3, "nFaces": 1},
        {"name": "back0", "type": "wall", "startFace": 4, "nFaces": 1},
        {"name": "left", "type": "wall", "startFace": 5, "nFaces": 1},
        {"name": "front1", "type": "wall", "startFace": 6, "nFaces": 1},
        {"name": "back1", "type": "wall", "startFace": 7, "nFaces": 1},
        {"name": "face1", "type": "wall", "startFace": 8, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": 9, "nFaces": 1},
    ]
    m = FvMesh(points=pts2, faces=faces2, owner=owner2,
               neighbour=neighbour2, boundary=boundary2, validate=False)
    m.compute_geometry()
    return m


class TestStitchEnhanced7:
    def test_returns_result_type(self):
        m = _two_cell_hex()
        r = stitch_mesh_enhanced_7(m, "bottom", "top")
        assert isinstance(r, StitchEnhanced7Result)

    def test_quality_grade(self):
        m = _two_cell_hex()
        r = stitch_mesh_enhanced_7(m, "bottom", "top")
        assert isinstance(r.quality_grade, StitchQualityGrade)
        assert r.quality_grade.grade in ("A", "B", "C", "D", "F")

    def test_gap_bridge(self):
        m = _two_cell_hex()
        r = stitch_mesh_enhanced_7(m, "bottom", "top", gap_bridge=True)
        assert r.n_bridged >= 0

    def test_adaptive_schedule(self):
        m = _two_cell_hex()
        r = stitch_mesh_enhanced_7(
            m, "bottom", "top",
            adaptive_schedule=True, schedule_passes=3,
        )
        assert r.n_adaptive_passes == 3
        assert r.final_tolerance > 0

    def test_grade_quality_stitch(self):
        m = _two_cell_hex()
        r = stitch_mesh_enhanced_7(m, "bottom", "top")
        assert r.quality_grade.match_ratio >= 0.0
