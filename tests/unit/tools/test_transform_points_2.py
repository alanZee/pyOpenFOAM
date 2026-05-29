"""Tests for transform_points_enhanced — enhanced coordinate transformation."""
from __future__ import annotations
import math
import pytest
import torch
from pyfoam.tools.transform_points_2 import transform_points_enhanced


def _t(*args, **kwargs):
    kwargs.setdefault("dtype", torch.float64)
    return torch.tensor(*args, **kwargs)


def _unit_cube_points():
    """8 个立方体顶点。"""
    return _t([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ])


class TestTranslation:
    def test_basic_translation(self):
        pts = _unit_cube_points()
        r = transform_points_enhanced(pts, translation=[1, 2, 3])
        expected = pts + _t([1, 2, 3])
        assert torch.allclose(r, expected, atol=1e-12)

    def test_translation_does_not_modify_input(self):
        pts = _unit_cube_points()
        orig = pts.clone()
        transform_points_enhanced(pts, translation=[10, 10, 10])
        assert torch.allclose(pts, orig)


class TestRotationAxisAngle:
    def test_90_deg_z(self):
        pts = _t([[1.0, 0.0, 0.0]])
        r = transform_points_enhanced(pts, rotation_axis=[0, 0, 1], rotation_angle=90.0)
        assert torch.allclose(r[0], _t([0.0, 1.0, 0.0]), atol=1e-10)

    def test_360_deg_identity(self):
        pts = _unit_cube_points()
        r = transform_points_enhanced(pts, rotation_axis=[0, 0, 1], rotation_angle=360.0)
        assert torch.allclose(r, pts, atol=1e-10)

    def test_requires_both_axis_and_angle(self):
        pts = _unit_cube_points()
        with pytest.raises(ValueError, match="must both be provided"):
            transform_points_enhanced(pts, rotation_axis=[0, 0, 1])
        with pytest.raises(ValueError, match="must both be provided"):
            transform_points_enhanced(pts, rotation_angle=45.0)

    def test_zero_axis_raises(self):
        pts = _unit_cube_points()
        with pytest.raises(ValueError, match="non-zero"):
            transform_points_enhanced(pts, rotation_axis=[0, 0, 0], rotation_angle=45.0)


class TestExplicitRotationMatrix:
    def test_identity_matrix(self):
        pts = _unit_cube_points()
        I = torch.eye(3, dtype=torch.float64)
        r = transform_points_enhanced(pts, rotation_matrix=I)
        assert torch.allclose(r, pts, atol=1e-12)

    def test_90_deg_z_matrix(self):
        """绕 z 轴旋转 90 度的旋转矩阵。"""
        pts = _t([[1.0, 0.0, 0.0]])
        R = _t([[0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]])
        r = transform_points_enhanced(pts, rotation_matrix=R)
        assert torch.allclose(r[0], _t([0.0, 1.0, 0.0]), atol=1e-10)

    def test_bad_shape_raises(self):
        pts = _unit_cube_points()
        with pytest.raises(ValueError, match="3x3"):
            transform_points_enhanced(pts, rotation_matrix=torch.eye(4, dtype=torch.float64))


class TestScaling:
    def test_uniform_scale(self):
        pts = _unit_cube_points()
        r = transform_points_enhanced(pts, scale=2.0)
        assert torch.allclose(r, pts * 2.0, atol=1e-12)

    def test_anisotropic_scale(self):
        pts = _unit_cube_points()
        r = transform_points_enhanced(pts, scale=[2, 1, 0.5])
        s = _t([2, 1, 0.5])
        assert torch.allclose(r, pts * s, atol=1e-12)


class TestCombined:
    def test_scale_rotate_translate(self):
        """scale=2, rotate 90 about z, translate (1,0,0) → (0,1,0)→(2,0,0)→(0,2,0)→(1,2,0)。"""
        pts = _t([[1.0, 0.0, 0.0]])
        r = transform_points_enhanced(
            pts, scale=2.0, rotation_axis=[0, 0, 1], rotation_angle=90.0, translation=[1, 0, 0]
        )
        assert torch.allclose(r[0], _t([1.0, 2.0, 0.0]), atol=1e-10)

    def test_axis_angle_then_explicit_matrix(self):
        """先 axis-angle 再 explicit matrix。"""
        pts = _t([[1.0, 0.0, 0.0]])
        # 先绕 z 轴转 90° → (0,1,0)
        # 再绕 x 轴转 90° 矩阵 → (0,0,1)
        Rx = _t([[1, 0, 0],
                 [0, 0, -1],
                 [0, 1, 0]])
        r = transform_points_enhanced(
            pts, rotation_axis=[0, 0, 1], rotation_angle=90.0, rotation_matrix=Rx
        )
        assert torch.allclose(r[0], _t([0.0, 0.0, 1.0]), atol=1e-10)

    def test_no_transform_returns_clone(self):
        pts = _unit_cube_points()
        r = transform_points_enhanced(pts)
        assert torch.allclose(r, pts)
        assert r is not pts
