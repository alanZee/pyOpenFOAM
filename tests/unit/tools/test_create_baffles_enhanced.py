"""Tests for create_baffles_enhanced — enhanced baffle creation."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced import BaffleEnhancedResult, create_baffles_enhanced


def _two_cell_mesh():
    """Two-cell mesh with one internal face."""
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1],
           [0,0,2],[1,0,2],[1,1,2],[0,1,2]]
    fc = [[4,5,6,7],[0,3,2,1],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5],
          [8,9,10,11],[4,5,9,8],[7,11,10,6],[4,8,11,7],[5,6,10,9]]
    m = FvMesh(
        points=torch.tensor(pts, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
        owner=torch.tensor([0,0,0,0,0,0,1,1,1,1,1], dtype=INDEX_DTYPE),
        neighbour=torch.tensor([1], dtype=INDEX_DTYPE),
        boundary=[{"name":"bottom","type":"wall","startFace":1,"nFaces":5},
                  {"name":"top","type":"wall","startFace":6,"nFaces":5}],
        validate=False,
    )
    m.compute_geometry()
    return m


class TestCreateBafflesEnhanced:
    def test_returns_result_type(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[0])
        assert isinstance(r, BaffleEnhancedResult)

    def test_creates_baffle_from_internal(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[0])
        assert r.n_baffles == 1
        assert "baffle" in r.baffle_patches

    def test_no_baffles_returns_clone(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[])
        assert r.n_baffles == 0

    def test_dual_patches(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[0], dual_patches=True)
        assert len(r.baffle_patches) == 2
        assert "baffle_left" in r.baffle_patches
        assert "baffle_right" in r.baffle_patches

    def test_cell_based_selection(self):
        m = _two_cell_mesh()
        # Cell 0 has internal face 0
        r = create_baffles_enhanced(m, cells=[0])
        assert r.n_baffles >= 1

    def test_non_internal_face_raises(self):
        m = _two_cell_mesh()
        # Face 1 is boundary, not internal
        with pytest.raises(ValueError, match="not an internal"):
            create_baffles_enhanced(m, face_indices=[1])

    def test_no_selection_raises(self):
        m = _two_cell_mesh()
        with pytest.raises(ValueError, match="Either"):
            create_baffles_enhanced(m)

    def test_preserves_original_boundary(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[0])
        patch_names = {p["name"] for p in r.mesh.boundary}
        assert "bottom" in patch_names or "top" in patch_names

    def test_custom_patch_name(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced(m, face_indices=[0], patch_name="my_baffle")
        assert "my_baffle" in r.baffle_patches
