"""Tests for box_turb — turbulence box generator."""

import numpy as np
import pytest

from pyfoam.tools.box_turb import box_turb


class TestBoxTurb:
    """Test the box_turb function."""

    def test_output_shape(self, large_mesh):
        """Output should have shape (n_cells, 3)."""
        velocity = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=42)
        assert velocity.shape == (large_mesh.n_cells, 3)

    def test_target_turbulent_kinetic_energy(self, large_mesh):
        """Mean TKE should approximately match the target k."""
        k_target = 2.5
        velocity = box_turb(large_mesh, k=k_target, epsilon=1.0, seed=42)
        k_actual = 0.5 * np.mean(np.sum(velocity ** 2, axis=1))
        np.testing.assert_allclose(k_actual, k_target, rtol=0.05)

    def test_zero_mean_velocity(self, large_mesh):
        """Mean velocity should be approximately zero."""
        velocity = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=42)
        mean_vel = velocity.mean(axis=0)
        np.testing.assert_allclose(mean_vel, 0.0, atol=1e-10)

    def test_reproducibility_with_seed(self, large_mesh):
        """Same seed should produce identical fields."""
        vel1 = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=123)
        vel2 = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=123)
        np.testing.assert_array_equal(vel1, vel2)

    def test_different_seeds_differ(self, large_mesh):
        """Different seeds should produce different fields."""
        vel1 = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=1)
        vel2 = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=2)
        assert not np.allclose(vel1, vel2)

    def test_nonpositive_k_raises(self, large_mesh):
        """Non-positive k should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            box_turb(large_mesh, k=0.0, epsilon=1.0)

    def test_nonpositive_epsilon_raises(self, large_mesh):
        """Non-positive epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            box_turb(large_mesh, k=1.0, epsilon=-1.0)

    def test_with_viscosity(self, large_mesh):
        """Specifying viscosity should use Von Karman spectrum."""
        velocity = box_turb(large_mesh, k=1.0, epsilon=1.0, nu=1e-3, seed=42)
        assert velocity.shape == (large_mesh.n_cells, 3)
        k_actual = 0.5 * np.mean(np.sum(velocity ** 2, axis=1))
        np.testing.assert_allclose(k_actual, 1.0, rtol=0.05)

    def test_n_harmonics_minimum(self, large_mesh):
        """n_harmonics=1 should still produce valid output."""
        velocity = box_turb(large_mesh, k=1.0, epsilon=1.0, n_harmonics=1, seed=42)
        assert velocity.shape == (large_mesh.n_cells, 3)
        assert np.all(np.isfinite(velocity))

    def test_more_harmonics_increases_detail(self, large_mesh):
        """More harmonics should produce a different (finer) field."""
        vel_few = box_turb(large_mesh, k=1.0, epsilon=1.0, n_harmonics=4, seed=42)
        vel_many = box_turb(large_mesh, k=1.0, epsilon=1.0, n_harmonics=32, seed=42)
        # Fields should differ (more harmonics → more small-scale content)
        assert not np.allclose(vel_few, vel_many)

    def test_output_is_finite(self, large_mesh):
        """All values should be finite."""
        velocity = box_turb(large_mesh, k=1.0, epsilon=1.0, seed=42)
        assert np.all(np.isfinite(velocity))
