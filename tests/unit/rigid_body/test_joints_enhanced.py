"""Tests for enhanced joint types."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced import (
    CylindricalJoint,
    PlanarJoint,
    UniversalJoint,
    FreeJoint,
)


class TestCylindricalJoint:
    """Test cylindrical joint."""

    def test_n_dof(self):
        joint = CylindricalJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        assert joint.n_dof == 2

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            CylindricalJoint(torch.zeros(3, dtype=torch.float64))

    def test_allows_translation_along_axis(self):
        joint = CylindricalJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_vel = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed, child_vel)

    def test_allows_rotation_about_axis(self):
        joint = CylindricalJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_omega = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed, child_omega)

    def test_constrains_perpendicular_translation(self):
        joint = CylindricalJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_vel = torch.tensor([5.0, 3.0, 0.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert abs(allowed[0].item()) < 1e-12
        assert abs(allowed[1].item()) < 1e-12

    def test_allowed_axes_shape(self):
        joint = CylindricalJoint(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        axes = joint.allowed_axes()
        assert axes.shape == (2, 3)


class TestPlanarJoint:
    """Test planar joint."""

    def test_n_dof(self):
        joint = PlanarJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        assert joint.n_dof == 3

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            PlanarJoint(torch.zeros(3, dtype=torch.float64))

    def test_allows_in_plane_translation(self):
        joint = PlanarJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_vel = torch.tensor([3.0, 5.0, 0.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert abs(allowed[0].item() - 3.0) < 1e-12
        assert abs(allowed[1].item() - 5.0) < 1e-12
        assert abs(allowed[2].item()) < 1e-12

    def test_constrains_out_of_plane_translation(self):
        joint = PlanarJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_vel = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed, torch.zeros(3, dtype=torch.float64))

    def test_allows_rotation_about_normal(self):
        joint = PlanarJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_omega = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed, child_omega)

    def test_constrains_in_plane_rotation(self):
        joint = PlanarJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        child_omega = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed, torch.zeros(3, dtype=torch.float64))

    def test_custom_in_plane_axis(self):
        """Custom in-plane axis gets orthogonalized."""
        joint = PlanarJoint(
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            in_plane_axis=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        )
        assert joint.n_dof == 3

    def test_parallel_in_plane_axis_raises(self):
        """In-plane axis parallel to normal raises."""
        with pytest.raises(ValueError, match="parallel"):
            PlanarJoint(
                normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                in_plane_axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            )


class TestUniversalJoint:
    """Test universal (Cardan) joint."""

    def test_n_dof(self):
        joint = UniversalJoint(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        assert joint.n_dof == 2

    def test_zero_axis1_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            UniversalJoint(
                torch.zeros(3, dtype=torch.float64),
                torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            )

    def test_parallel_axes_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            UniversalJoint(
                torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
            )

    def test_allows_rotation_about_both_axes(self):
        joint = UniversalJoint(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        omega = torch.tensor([3.0, 5.0, 0.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), omega,
        )
        assert torch.allclose(allowed, omega)

    def test_constrains_third_axis_rotation(self):
        joint = UniversalJoint(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), omega,
        )
        assert torch.allclose(allowed, torch.zeros(3, dtype=torch.float64))

    def test_translation_constrained(self):
        joint = UniversalJoint(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed, torch.zeros(3, dtype=torch.float64))


class TestFreeJoint:
    """Test free (unconstrained) joint."""

    def test_n_dof(self):
        joint = FreeJoint()
        assert joint.n_dof == 6

    def test_all_translation_allowed(self):
        joint = FreeJoint()
        vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        allowed, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed, vel)

    def test_all_rotation_allowed(self):
        joint = FreeJoint()
        omega = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        _, allowed = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), omega,
        )
        assert torch.allclose(allowed, omega)
