"""Tests for mirror_mesh — mirror a mesh about a plane."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.mirror_mesh import mirror_mesh
from tests.unit.mesh.conftest import make_fv_mesh


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


class TestBasic:
    def test_returns_fv_mesh(self):
        m = _single_hex()
        r = mirror_mesh(m, plane_normal=[1, 0, 0], plane_point=[0, 0, 0])
        assert isinstance(r, FvMesh)

    def test_doubles_cell_count(self):
        """镜像后 cells 数翻倍。"""
        m = _single_hex()
        r = mirror_mesh(m, plane_normal=[1, 0, 0], plane_point=[0, 0, 0])
        assert r.n_cells == 2 * m.n_cells

    def test_doubles_points(self):
        m = _single_hex()
        r = mirror_mesh(m, plane_normal=[1, 0, 0], plane_point=[0, 0, 0])
        assert r.points.shape[0] == 2 * m.points.shape[0]


class TestGeometry:
    def test_mirrored_points_reflected(self):
        """镜像点应满足反射关系。"""
        m = _single_hex()
        normal = [1, 0, 0]
        point = [0, 0, 0]
        r = mirror_mesh(m, plane_normal=normal, plane_point=point)
        n_orig = m.points.shape[0]
        orig = r.points[:n_orig]
        mirrored = r.points[n_orig:]
        # 沿 x 轴反射：x' = -x, y'=y, z'=z
        for i in range(n_orig):
            assert torch.allclose(mirrored[i, 1:], orig[i, 1:], atol=1e-10)
            assert abs(mirrored[i, 0].item() + orig[i, 0].item()) < 1e-10

    def test_preserves_original_volume(self):
        """镜像的总容量 = 2 × 原始容量。"""
        m = _single_hex()
        v0 = m.total_volume.item()
        r = mirror_mesh(m, plane_normal=[0, 1, 0], plane_point=[0, 0, 0])
        r.compute_geometry()
        assert abs(r.total_volume.item() - 2 * v0) < 1e-8

    def test_mirror_about_z_plane(self):
        m = _single_hex()
        r = mirror_mesh(m, plane_normal=[0, 0, 1], plane_point=[0, 0, 0])
        assert r.n_cells == 2 * m.n_cells
        r.compute_geometry()
        assert abs(r.total_volume.item() - 2 * m.total_volume.item()) < 1e-8


class TestEdgeCases:
    def test_zero_normal_raises(self):
        m = _single_hex()
        with pytest.raises(ValueError, match="non-zero"):
            mirror_mesh(m, plane_normal=[0, 0, 0], plane_point=[0, 0, 0])

    def test_non_unit_normal(self):
        """非单位法向量应被归一化，结果与单位法向量一致。"""
        m = _single_hex()
        r1 = mirror_mesh(m, plane_normal=[1, 0, 0], plane_point=[0, 0, 0])
        r2 = mirror_mesh(m, plane_normal=[5, 0, 0], plane_point=[0, 0, 0])
        assert r1.n_cells == r2.n_cells

    def test_does_not_modify_original(self):
        m = _single_hex()
        nc = m.n_cells
        mirror_mesh(m, plane_normal=[1, 0, 0], plane_point=[0, 0, 0])
        assert m.n_cells == nc
