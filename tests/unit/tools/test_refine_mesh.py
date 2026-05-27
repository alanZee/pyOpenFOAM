"""Tests for refine_mesh."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.refine_mesh import refine_mesh, split_face, _PointManager

def _hex1():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name":"all","type":"wall","startFace":0,"nFaces":6}])
    m.compute_geometry(); return m

def _hex2z():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,2],[1,0,2],[1,1,2],[0,1,2]]
    fc = [[4,5,6,7],[0,3,2,1],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5],[8,9,10,11],[4,5,9,8],[7,11,10,6],[4,8,11,7],[5,6,10,9]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.tensor([0,0,0,0,0,0,1,1,1,1,1], dtype=INDEX_DTYPE),
               neighbour=torch.tensor([1], dtype=INDEX_DTYPE),
               boundary=[{"name":"bottom","type":"wall","startFace":1,"nFaces":5},{"name":"top","type":"wall","startFace":6,"nFaces":5}])
    m.compute_geometry(); return m

class TestBasic:
    def test_returns_fv_mesh(self): assert isinstance(refine_mesh(_hex1(), [0], "all"), FvMesh)
    def test_all_8(self): assert refine_mesh(_hex1(), [0], "all").n_cells == 8
    def test_x_2(self): assert refine_mesh(_hex1(), [0], "x").n_cells == 2
    def test_y_2(self): assert refine_mesh(_hex1(), [0], "y").n_cells == 2
    def test_z_2(self): assert refine_mesh(_hex1(), [0], "z").n_cells == 2
    def test_empty(self):
        m = _hex1(); r = refine_mesh(m, [], "all")
        assert r.n_cells == m.n_cells and r.n_faces == m.n_faces
    def test_bad_dir(self):
        with pytest.raises(ValueError): refine_mesh(_hex1(), [0], "diag")

class TestTopology:
    def test_owner_lt_nbr_all(self):
        r = refine_mesh(_hex1(), [0], "all")
        for i in range(r.n_internal_faces):
            assert r.owner[i].item() < r.neighbour[i].item()
    def test_owner_lt_nbr_x(self):
        r = refine_mesh(_hex1(), [0], "x")
        for i in range(r.n_internal_faces):
            assert r.owner[i].item() < r.neighbour[i].item()
    def test_indices_valid(self):
        r = refine_mesh(_hex1(), [0], "all")
        np_ = r.points.shape[0]
        for fi in range(r.n_faces):
            assert r.faces[fi].min().item() >= 0 and r.faces[fi].max().item() < np_
        assert r.owner.min().item() >= 0 and r.owner.max().item() < r.n_cells

class TestVolume:
    def test_preserved_all(self):
        m = _hex1(); v0 = m.total_volume.item()
        r = refine_mesh(m, [0], "all"); r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8
    def test_preserved_x(self):
        m = _hex1(); v0 = m.total_volume.item()
        r = refine_mesh(m, [0], "x"); r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8
    def test_sub_cell_equal(self):
        r = refine_mesh(_hex1(), [0], "all"); r.compute_geometry()
        vols = r.cell_volumes.detach().cpu().numpy()
        for v in vols: assert abs(v - 0.125) < 1e-8
    def test_multi_cell(self):
        m = _hex2z(); v0 = m.total_volume.item()
        r = refine_mesh(m, [0, 1], "all"); r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8

class TestMultiCell:
    def test_one_of_two(self):
        assert refine_mesh(_hex2z(), [0], "all").n_cells == 9
    def test_both(self):
        assert refine_mesh(_hex2z(), [0, 1], "all").n_cells == 16
    def test_tensor_input(self):
        assert refine_mesh(_hex1(), torch.tensor([0], dtype=INDEX_DTYPE), "x").n_cells == 2
    def test_points_increased(self):
        assert refine_mesh(_hex1(), [0], "all").points.shape[0] > 8
    def test_original_unchanged(self):
        m = _hex1(); nc = m.n_cells; refine_mesh(m, [0], "all"); assert m.n_cells == nc

class TestSplitFace:
    def test_no_split(self):
        pts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float64)
        fi = torch.tensor([0,1,2,3], dtype=INDEX_DTYPE)
        pm = _PointManager(torch.zeros((10,3), dtype=torch.float64))
        assert len(split_face(pts, False, False, False, fi, pm)) == 1
    def test_split_xy(self):
        pts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float64)
        fi = torch.tensor([0,1,2,3], dtype=INDEX_DTYPE)
        pm = _PointManager(torch.zeros((10,3), dtype=torch.float64))
        assert len(split_face(pts, True, True, False, fi, pm)) == 4
