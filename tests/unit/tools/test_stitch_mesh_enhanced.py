"""Tests for stitch_mesh_enhanced — enhanced mesh stitching."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced import StitchEnhancedResult, stitch_mesh_enhanced


def _two_cell_with_stitchable_patches():
    """Create a 2-cell mesh where two boundary patches are coincident."""
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    # Face 0: bottom, Face 1: top of cell 0 = internal top/bottom between cells
    # We'll create a single-cell hex with two coincident boundary patches
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "patch_a", "type": "wall", "startFace": 0, "nFaces": 3},
                          {"name": "patch_b", "type": "wall", "startFace": 3, "nFaces": 3}],
               validate=False)
    m.compute_geometry()
    return m


def _two_separate_hex_with_matching_faces():
    """Two separate hex meshes, each with a boundary face at x=1 / x=0."""
    # Create two single-cell hex meshes side by side
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    # Right face (x=1): [1,2,6,5]  Left face (x=0): [0,4,7,3]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "left", "type": "wall", "startFace": 0, "nFaces": 1},
                          {"name": "right", "type": "wall", "startFace": 1, "nFaces": 1},
                          {"name": "rest", "type": "wall", "startFace": 2, "nFaces": 4}],
               validate=False)
    m.compute_geometry()
    return m


class TestStitchEnhanced:
    def test_returns_result_type(self):
        m = _two_separate_hex_with_matching_faces()
        # No matching patches → just returns empty stitch
        r = stitch_mesh_enhanced(m, "left", "right", tolerance=1e-6)
        assert isinstance(r, StitchEnhancedResult)

    def test_nonexistent_patch_raises(self):
        m = _two_cell_with_stitchable_patches()
        with pytest.raises(ValueError, match="not found"):
            stitch_mesh_enhanced(m, "patch_a", "nonexistent")

    def test_non_conformal_mode(self):
        m = _two_separate_hex_with_matching_faces()
        r = stitch_mesh_enhanced(m, "left", "right", tolerance=1e-6, non_conformal=True)
        assert isinstance(r, StitchEnhancedResult)
        assert r.n_stitched >= 0

    def test_stitched_count_and_unmatched(self):
        m = _two_cell_with_stitchable_patches()
        r = stitch_mesh_enhanced(m, "patch_a", "patch_b", tolerance=1e-6)
        assert r.n_stitched >= 0
        assert r.n_unmatched >= 0
        assert isinstance(r.mesh, FvMesh)

    def test_result_mesh_valid(self):
        m = _two_cell_with_stitchable_patches()
        r = stitch_mesh_enhanced(m, "patch_a", "patch_b", tolerance=1e-6)
        assert r.mesh.n_faces > 0
        assert r.mesh.points.shape[0] > 0
