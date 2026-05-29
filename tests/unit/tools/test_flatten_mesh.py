"""Tests for flatten_mesh — convert 3D mesh to 2D by collapsing z-direction."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.flatten_mesh import flatten_mesh
from tests.unit.mesh.conftest import make_fv_mesh


def _thin_hex(z_thick=1e-7):
    """一个 z 方向极薄的六面体（z=0..z_thick）。"""
    z = z_thick
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,z],[1,0,z],[1,1,z],[0,1,z]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestBasic:
    def test_returns_fv_mesh(self):
        m = make_fv_mesh()
        r = flatten_mesh(m)
        assert isinstance(r, FvMesh)

    def test_preserves_cell_count(self):
        """flatten 不改变 cell 数。"""
        m = make_fv_mesh()
        r = flatten_mesh(m)
        assert r.n_cells == m.n_cells

    def test_preserves_face_count(self):
        m = make_fv_mesh()
        r = flatten_mesh(m)
        assert r.n_faces == m.n_faces


class TestZCollapse:
    def test_thin_mesh_collapses_to_plane(self):
        """极薄的 mesh 应该被压平到 z=0。"""
        m = _thin_hex(1e-7)
        r = flatten_mesh(m, z_tolerance=1e-6)
        z_vals = r.points[:, 2]
        # 所有 z 值应相同（在容差内）
        assert z_vals.max().item() - z_vals.min().item() < 1e-10

    def test_normal_mesh_unchanged(self):
        """正常厚度的 mesh 不应被改变。"""
        m = make_fv_mesh()
        r = flatten_mesh(m, z_tolerance=1e-6)
        z_vals = r.points[:, 2]
        # z 值应保持不同（0 和 1）
        assert z_vals.max().item() - z_vals.min().item() > 0.5

    def test_custom_tolerance(self):
        """使用自定义容差。"""
        m = _thin_hex(0.005)
        # tolerance=0.01 > z_thick=0.005 → 应被压平
        r = flatten_mesh(m, z_tolerance=0.01)
        z_vals = r.points[:, 2]
        assert z_vals.max().item() - z_vals.min().item() < 1e-10

    def test_x_y_unchanged(self):
        """x 和 y 坐标应保持不变。"""
        m = _thin_hex(1e-7)
        r = flatten_mesh(m, z_tolerance=1e-6)
        assert torch.allclose(r.points[:, :2], m.points[:, :2], atol=1e-12)


class TestTopology:
    def test_owner_lt_neighbour(self):
        m = make_fv_mesh()
        r = flatten_mesh(m, z_tolerance=1e-6)
        for i in range(r.n_internal_faces):
            assert r.owner[i].item() < r.neighbour[i].item()

    def test_indices_valid(self):
        m = make_fv_mesh()
        r = flatten_mesh(m, z_tolerance=1e-6)
        np_ = r.points.shape[0]
        for fi in range(r.n_faces):
            assert r.faces[fi].min().item() >= 0
            assert r.faces[fi].max().item() < np_


class TestEdgeCases:
    def test_does_not_modify_original(self):
        m = make_fv_mesh()
        orig_z = m.points[:, 2].clone()
        flatten_mesh(m, z_tolerance=1e-6)
        assert torch.allclose(m.points[:, 2], orig_z)

    def test_boundary_preserved(self):
        """边界 patch 应被保留。"""
        m = make_fv_mesh()
        r = flatten_mesh(m, z_tolerance=1e-6)
        assert len(r.boundary) > 0
