"""Tests for transform_points — mesh coordinate transformation."""

import math

import pytest
import torch

from pyfoam.tools.transform_points import transform_points
from tests.unit.mesh.conftest import make_poly_mesh, make_fv_mesh


def _t(*args, **kwargs):
    kwargs.setdefault("dtype", torch.float64)
    return torch.tensor(*args, **kwargs)


class TestTranslation:
    """Translate mesh points."""

    def test_translate_returns_correct_shape(self):
        mesh = make_poly_mesh()
        pts = transform_points(mesh, translation=[1.0, 2.0, 3.0])
        assert pts.shape == mesh.points.shape

    def test_translate_does_not_modify_original(self):
        mesh = make_poly_mesh()
        original = mesh.points.clone()
        transform_points(mesh, translation=[10.0, 10.0, 10.0])
        assert torch.allclose(mesh.points, original)

    def test_translate_single_point(self):
        mesh = make_poly_mesh()
        pts = transform_points(mesh, translation=[1.0, 0.0, 0.0])
        # Point 0 was (0,0,0) → should now be (1,0,0)
        assert torch.allclose(pts[0], _t([1.0, 0.0, 0.0]), atol=1e-12)

    def test_translate_all_points(self):
        mesh = make_poly_mesh()
        t = [1.0, 2.0, 3.0]
        pts = transform_points(mesh, translation=t)
        expected = mesh.points + _t(t)
        assert torch.allclose(pts, expected, atol=1e-12)

    def test_translate_as_tensor(self):
        mesh = make_poly_mesh()
        t = torch.tensor([5.0, -1.0, 0.5], dtype=torch.float64)
        pts = transform_points(mesh, translation=t)
        expected = mesh.points + t.unsqueeze(0)
        assert torch.allclose(pts, expected, atol=1e-12)


class TestRotation:
    """Rotate mesh points."""

    def test_rotate_90_deg_z_axis(self):
        """90 deg around z: (1,0,0) → (0,1,0)."""
        mesh = make_poly_mesh()
        pts = transform_points(mesh, rotation_axis=[0, 0, 1], rotation_angle=90.0)
        # Point 1 was (1,0,0) → (0,1,0)
        assert torch.allclose(pts[1], _t([0.0, 1.0, 0.0]), atol=1e-10)

    def test_rotate_180_deg_z_axis(self):
        """180 deg around z: (1,0,0) → (-1,0,0)."""
        mesh = make_poly_mesh()
        pts = transform_points(mesh, rotation_axis=[0, 0, 1], rotation_angle=180.0)
        assert torch.allclose(pts[1], _t([-1.0, 0.0, 0.0]), atol=1e-10)

    def test_rotate_360_deg_identity(self):
        """Full rotation returns original points."""
        mesh = make_poly_mesh()
        pts = transform_points(mesh, rotation_axis=[0, 0, 1], rotation_angle=360.0)
        assert torch.allclose(pts, mesh.points, atol=1e-10)

    def test_rotate_does_not_modify_original(self):
        mesh = make_poly_mesh()
        original = mesh.points.clone()
        transform_points(mesh, rotation_axis=[1, 0, 0], rotation_angle=45.0)
        assert torch.allclose(mesh.points, original)

    def test_rotation_requires_both_axis_and_angle(self):
        mesh = make_poly_mesh()
        with pytest.raises(ValueError, match="must both be provided"):
            transform_points(mesh, rotation_axis=[0, 0, 1])
        with pytest.raises(ValueError, match="must both be provided"):
            transform_points(mesh, rotation_angle=45.0)

    def test_rotation_rejects_zero_axis(self):
        mesh = make_poly_mesh()
        with pytest.raises(ValueError, match="non-zero"):
            transform_points(mesh, rotation_axis=[0, 0, 0], rotation_angle=45.0)

    def test_non_unit_axis_is_normalized(self):
        """Non-unit axis should be normalized before rotation."""
        mesh = make_poly_mesh()
        pts_unit = transform_points(mesh, rotation_axis=[0, 0, 1], rotation_angle=90.0)
        pts_scaled = transform_points(mesh, rotation_axis=[0, 0, 5], rotation_angle=90.0)
        assert torch.allclose(pts_unit, pts_scaled, atol=1e-10)


class TestScaling:
    """Scale mesh points."""

    def test_uniform_scale(self):
        mesh = make_poly_mesh()
        pts = transform_points(mesh, scale=2.0)
        expected = mesh.points * 2.0
        assert torch.allclose(pts, expected, atol=1e-12)

    def test_anisotropic_scale(self):
        mesh = make_poly_mesh()
        pts = transform_points(mesh, scale=[2.0, 1.0, 0.5])
        s = _t([2.0, 1.0, 0.5])
        expected = mesh.points * s.unsqueeze(0)
        assert torch.allclose(pts, expected, atol=1e-12)

    def test_scale_does_not_modify_original(self):
        mesh = make_poly_mesh()
        original = mesh.points.clone()
        transform_points(mesh, scale=10.0)
        assert torch.allclose(mesh.points, original)

    def test_zero_scale(self):
        """Scale=0 collapses all points to origin."""
        mesh = make_poly_mesh()
        pts = transform_points(mesh, scale=0.0)
        assert torch.allclose(pts, torch.zeros_like(pts), atol=1e-12)


class TestCombinedTransformations:
    """Combined scale, rotate, translate."""

    def test_scale_then_rotate_then_translate(self):
        """Scale by 2, rotate 90 deg about z, then translate (1,0,0)."""
        mesh = make_poly_mesh()
        pts = transform_points(
            mesh,
            scale=2.0,
            rotation_axis=[0, 0, 1],
            rotation_angle=90.0,
            translation=[1.0, 0.0, 0.0],
        )
        # Point 1: (1,0,0) → scale → (2,0,0) → rotate → (0,2,0) → translate → (1,2,0)
        assert torch.allclose(pts[1], _t([1.0, 2.0, 0.0]), atol=1e-10)

    def test_no_transform_returns_clone(self):
        """No arguments → cloned copy equal to original."""
        mesh = make_poly_mesh()
        pts = transform_points(mesh)
        assert torch.allclose(pts, mesh.points)
        assert pts is not mesh.points  # different tensor object


class TestFvMeshCompatibility:
    """transform_points works on FvMesh (subclass of PolyMesh)."""

    def test_fv_mesh_points_transformed(self):
        mesh = make_fv_mesh()
        pts = transform_points(mesh, translation=[1.0, 0.0, 0.0])
        assert pts.shape == mesh.points.shape
        assert torch.allclose(pts[0], mesh.points[0] + _t([1.0, 0.0, 0.0]), atol=1e-12)
