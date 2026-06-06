"""
Tests for surf_mesh module.
"""
import math

import pytest
import torch

from pyfoam.surf_mesh import SurfMesh, SurfZone, SurfScalarField, SurfVectorField


def _make_triangle_mesh() -> SurfMesh:
    """创建简单三角形表面网格（2 个三角形）。"""
    pts = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [1.5, 1, 0],
    ], dtype=torch.float64)
    faces = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 3, 2]),
    ]
    return SurfMesh(points=pts, faces=faces)


def _make_quad_mesh() -> SurfMesh:
    """创建四边形表面网格。"""
    pts = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=torch.float64)
    faces = [torch.tensor([0, 1, 2, 3])]
    return SurfMesh(points=pts, faces=faces)


class TestSurfMeshBasic:
    """基本属性测试。"""

    def test_triangle_mesh(self):
        mesh = _make_triangle_mesh()
        assert mesh.n_points == 4
        assert mesh.n_faces == 2
        assert mesh.n_zones == 1

    def test_quad_mesh(self):
        mesh = _make_quad_mesh()
        assert mesh.n_points == 4
        assert mesh.n_faces == 1

    def test_zones_default(self):
        mesh = _make_triangle_mesh()
        assert mesh.zones[0].name == "default"
        assert mesh.zones[0].n_faces == 2

    def test_custom_zones(self):
        mesh = _make_triangle_mesh()
        zones = [SurfZone("zone1", 0, 1), SurfZone("zone2", 1, 1)]
        mesh2 = SurfMesh(points=mesh.points, faces=mesh.faces, zones=zones)
        assert mesh2.n_zones == 2

    def test_repr(self):
        mesh = _make_triangle_mesh()
        assert "SurfMesh" in repr(mesh)
        assert "n_points=4" in repr(mesh)


class TestSurfMeshGeometry:
    """几何计算测试。"""

    def test_face_centres_triangle(self):
        mesh = _make_triangle_mesh()
        centres = mesh.face_centres()
        assert centres.shape == (2, 3)
        # 第一个三角形中心
        expected_x = (0 + 1 + 0.5) / 3
        assert abs(centres[0, 0].item() - expected_x) < 1e-10

    def test_face_areas_triangle(self):
        mesh = _make_triangle_mesh()
        areas = mesh.face_areas()
        assert areas.shape == (2, 3)
        # 三角形面积 = 0.5 * base * height = 0.5 * 1 * 1 = 0.5
        mag = areas.norm(dim=1)
        assert abs(mag[0].item() - 0.5) < 1e-10

    def test_face_areas_quad(self):
        mesh = _make_quad_mesh()
        areas = mesh.face_areas()
        mag = areas.norm(dim=1)
        assert abs(mag[0].item() - 1.0) < 1e-10

    def test_face_mag_areas(self):
        mesh = _make_triangle_mesh()
        mag = mesh.face_mag_areas()
        assert mag.shape == (2,)
        assert (mag > 0).all()

    def test_face_type_counts(self):
        mesh = _make_triangle_mesh()
        counts = mesh.face_type_counts()
        assert counts[3] == 2  # 2 个三角形

    def test_quad_face_type_counts(self):
        mesh = _make_quad_mesh()
        counts = mesh.face_type_counts()
        assert counts[4] == 1  # 1 个四边形


class TestSurfFields:
    """表面场测试。"""

    def test_scalar_field(self):
        mesh = _make_triangle_mesh()
        field = SurfScalarField(mesh, "p")
        assert len(field) == 2
        assert field[0] == 0.0

    def test_vector_field(self):
        mesh = _make_triangle_mesh()
        field = SurfVectorField(mesh, "U")
        assert len(field) == 2
        assert field.data.shape == (2, 3)

    def test_field_setitem(self):
        mesh = _make_triangle_mesh()
        field = SurfScalarField(mesh, "p")
        field[0] = 1.5
        assert field[0] == 1.5

    def test_field_with_data(self):
        mesh = _make_triangle_mesh()
        data = torch.tensor([1.0, 2.0])
        field = SurfScalarField(mesh, "p", data=data)
        assert field[0] == 1.0
        assert field[1] == 2.0

    def test_field_repr(self):
        mesh = _make_triangle_mesh()
        field = SurfScalarField(mesh, "pressure")
        assert "SurfScalarField" in repr(field)
        assert "pressure" in repr(field)
