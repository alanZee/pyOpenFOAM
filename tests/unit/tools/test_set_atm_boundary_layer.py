"""Tests for set_atm_boundary_layer — atmospheric boundary layer profiles."""
from __future__ import annotations
import math
import numpy as np
import pytest
from pyfoam.tools.set_atm_boundary_layer import (
    set_atm_boundary_layer, ABLProperties, compute_u_star
)


class TestABLProperties:
    def test_default_values(self):
        """Default constructor should work."""
        abl = ABLProperties()
        assert abl.u_star == 0.5
        assert abl.z0 == 0.01
        assert abl.kappa == 0.41
        assert abl.Cmu == 0.09

    def test_custom_values(self):
        """Custom values should be stored."""
        abl = ABLProperties(u_star=1.0, z0=0.1, displacement_height=5.0)
        assert abl.u_star == 1.0
        assert abl.z0 == 0.1
        assert abl.displacement_height == 5.0


class TestSetAtmBoundaryLayer:
    def test_returns_arrays(self, fv_mesh):
        """Should return U, k, epsilon arrays."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        U, k, eps = set_atm_boundary_layer(fv_mesh, abl)
        n_cells = fv_mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert k.shape == (n_cells,)
        assert eps.shape == (n_cells,)

    def test_velocity_direction(self, fv_mesh):
        """Velocity should be aligned with wind direction."""
        abl = ABLProperties(u_star=0.5, z0=0.01, direction=(1.0, 0.0, 0.0))
        U, _, _ = set_atm_boundary_layer(fv_mesh, abl)
        for ci in range(fv_mesh.n_cells):
            u_mag = np.linalg.norm(U[ci])
            if u_mag > 1e-30:
                # y and z components should be ~0
                assert abs(U[ci, 1]) < 1e-10
                assert abs(U[ci, 2]) < 1e-10

    def test_velocity_increases_with_height(self, large_mesh):
        """Log-law: velocity should increase with height."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        U, _, _ = set_atm_boundary_layer(large_mesh, abl)
        cell_centres = large_mesh.cell_centres.detach().cpu().numpy()
        # Compare cells at different heights
        velocities = []
        for ci in range(large_mesh.n_cells):
            z = cell_centres[ci, 2]
            u_mag = np.linalg.norm(U[ci])
            velocities.append((z, u_mag))
        velocities.sort()
        # At least at the highest z, velocity should be larger
        low_z = [v for z, v in velocities if z < 0.5]
        high_z = [v for z, v in velocities if z > 0.5]
        if low_z and high_z:
            assert max(high_z) > max(low_z)

    def test_k_positive(self, fv_mesh):
        """TKE should be positive everywhere."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        _, k, _ = set_atm_boundary_layer(fv_mesh, abl)
        assert np.all(k > 0)

    def test_epsilon_positive(self, fv_mesh):
        """Dissipation rate should be positive everywhere."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        _, _, eps = set_atm_boundary_layer(fv_mesh, abl)
        assert np.all(eps > 0)

    def test_k_independent_of_height(self, fv_mesh):
        """TKE = u*^2 / sqrt(Cmu), should be uniform."""
        abl = ABLProperties(u_star=0.5, z0=0.01, Cmu=0.09)
        _, k, _ = set_atm_boundary_layer(fv_mesh, abl)
        expected = 0.5**2 / math.sqrt(0.09)
        assert np.allclose(k, expected, rtol=1e-10)

    def test_epsilon_decreases_with_height(self, large_mesh):
        """Epsilon = u*^3/(kappa*z), should decrease with height."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        _, _, eps = set_atm_boundary_layer(large_mesh, abl)
        cell_centres = large_mesh.cell_centres.detach().cpu().numpy()
        # Check monotonic trend
        pairs = [(cell_centres[ci, 2], eps[ci]) for ci in range(large_mesh.n_cells)]
        pairs.sort()
        # First should have higher epsilon than last
        if len(pairs) > 1:
            assert pairs[0][1] > pairs[-1][1]

    def test_different_z_axis(self, fv_mesh):
        """Custom z_axis should work."""
        abl = ABLProperties(u_star=0.5, z0=0.01, direction=(0.0, 1.0, 0.0))
        U, k, eps = set_atm_boundary_layer(fv_mesh, abl, z_axis=2)
        assert U.shape == (fv_mesh.n_cells, 3)

    def test_ground_level_offset(self, fv_mesh):
        """Setting free_surface_z should shift the profile."""
        abl = ABLProperties(u_star=0.5, z0=0.01)
        U1, _, _ = set_atm_boundary_layer(fv_mesh, abl, free_surface_z=0.0)
        U2, _, _ = set_atm_boundary_layer(fv_mesh, abl, free_surface_z=-5.0)
        # With lower ground, cells are effectively higher → larger velocity
        for ci in range(fv_mesh.n_cells):
            assert np.linalg.norm(U2[ci]) >= np.linalg.norm(U1[ci]) - 1e-10


class TestComputeUStar:
    def test_basic(self):
        """Basic friction velocity computation."""
        u_star = compute_u_star(U_ref=10.0, z_ref=10.0, z0=0.01)
        assert u_star > 0

    def test_roundtrip(self):
        """compute_u_star → set_atm_boundary_layer should recover U_ref."""
        z_ref = 10.0
        U_ref = 10.0
        z0 = 0.01
        u_star = compute_u_star(U_ref, z_ref, z0)
        kappa = 0.41
        U_calc = (u_star / kappa) * math.log((z_ref - 0.0) / z0)
        assert abs(U_calc - U_ref) < 1e-8

    def test_invalid_z_ref_raises(self):
        """Reference height below roughness should raise."""
        with pytest.raises(ValueError):
            compute_u_star(U_ref=10.0, z_ref=0.001, z0=0.01)

    def test_displacement_height(self):
        """Displacement height should reduce effective height."""
        u_star_no_d = compute_u_star(10.0, 10.0, 0.01)
        u_star_with_d = compute_u_star(10.0, 10.0, 0.01, displacement_height=5.0)
        # With displacement, effective height is smaller → higher gradient → higher u*
        assert u_star_with_d > u_star_no_d
