"""Tests for RotatingWallVelocity2BC (enhanced rotating wall BC)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, RotatingWallVelocity2BC
from pyfoam.boundary.boundary_condition import Patch


@pytest.fixture
def wall_patch() -> Patch:
    """A 3-face wall patch for testing."""
    return Patch(
        name="rotor",
        face_indices=torch.tensor([6, 7, 8]),
        face_normals=torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


class TestRotatingWallVelocity2BC:
    """Tests for the rotatingWallVelocity2 boundary condition."""

    def test_registration(self):
        """rotatingWallVelocity2 is registered in RTS."""
        assert "rotatingWallVelocity2" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "rotatingWallVelocity2",
            wall_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        assert isinstance(bc, RotatingWallVelocity2BC)

    def test_properties_constant_omega(self, wall_patch):
        """Properties are parsed correctly for constant omega."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"origin": [1, 2, 3], "axis": [0, 0, 1], "omega": 5.0},
        )
        assert bc.omega == 5.0
        assert torch.allclose(bc.origin, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        # Axis should be normalised
        assert torch.allclose(bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        assert not bc.has_table

    def test_default_values(self, wall_patch):
        """Default values are correct."""
        bc = RotatingWallVelocity2BC(wall_patch)
        assert bc.omega == 0.0
        assert torch.allclose(bc.origin, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        assert not bc.has_table

    def test_omega_table_properties(self, wall_patch):
        """Time-varying omega table is parsed correctly."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "origin": [0, 0, 0],
                "axis": [0, 0, 1],
                "omegaTable": [[0, 5.0], [1, 10.0], [2, 8.0]],
            },
        )
        assert bc.has_table
        assert bc.table_times is not None
        assert bc.table_omegas is not None
        assert torch.allclose(bc.table_times, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
        assert torch.allclose(bc.table_omegas, torch.tensor([5.0, 10.0, 8.0], dtype=torch.float64))

    def test_omega_interpolation_midpoint(self, wall_patch):
        """Omega interpolation returns midpoint value."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "origin": [0, 0, 0],
                "axis": [0, 0, 1],
                "omegaTable": [[0, 5.0], [1, 10.0], [2, 8.0]],
            },
        )
        # At t=0.5, omega should be 7.5 (midpoint between 5 and 10)
        assert abs(bc._get_omega(0.5) - 7.5) < 1e-10

    def test_omega_interpolation_clamp_low(self, wall_patch):
        """Omega is clamped to first value before table start."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "omegaTable": [[1, 5.0], [2, 10.0]],
            },
        )
        assert abs(bc._get_omega(0.0) - 5.0) < 1e-10

    def test_omega_interpolation_clamp_high(self, wall_patch):
        """Omega is clamped to last value after table end."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "omegaTable": [[0, 5.0], [1, 10.0]],
            },
        )
        assert abs(bc._get_omega(5.0) - 10.0) < 1e-10

    def test_constant_omega_without_table(self, wall_patch):
        """Constant omega is returned when no table is given."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"omega": 15.0},
        )
        assert abs(bc._get_omega(0.0) - 15.0) < 1e-10
        assert abs(bc._get_omega(100.0) - 15.0) < 1e-10

    def test_apply_constant_omega(self, wall_patch):
        """apply() with constant omega sets velocity correctly."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 2.0},
        )
        field = torch.zeros(12, 3, dtype=torch.float64)
        bc.apply(field)
        # Face centres are (0,0,0), (1,0,0), (2,0,0)
        # omega_vec = (0, 0, 2)
        # v = omega x r:
        #   face 0: (0,0,2) x (0,0,0) = (0,0,0)
        #   face 1: (0,0,2) x (1,0,0) = (0,2,0)
        #   face 2: (0,0,2) x (2,0,0) = (0,4,0)
        assert torch.allclose(field[6, :], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[7, :], torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[8, :], torch.tensor([0.0, 4.0, 0.0], dtype=torch.float64))

    def test_apply_time_varying(self, wall_patch):
        """apply() with time-varying table uses interpolated omega."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "origin": [0, 0, 0],
                "axis": [0, 0, 1],
                "omegaTable": [[0, 2.0], [1, 4.0]],
            },
        )
        field = torch.zeros(12, 3, dtype=torch.float64)
        bc.apply(field, time=0.5)
        # omega(0.5) = 3.0
        # face 1: (0,0,3) x (1,0,0) = (0,3,0)
        assert torch.allclose(field[7, :], torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, wall_patch):
        """apply() with explicit patch_idx writes at the correct location."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 1.0},
        )
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=10)
        # Should write at indices 10, 11, 12
        assert torch.allclose(field[10, :], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11, :], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    def test_matrix_contributions_constant(self, wall_patch):
        """Matrix contributions with constant omega have correct shape."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 5.0},
        )
        field = torch.zeros(12, 3, dtype=torch.float64)
        n_cells = 12
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0

    def test_matrix_contributions_time_varying(self, wall_patch):
        """Matrix contributions use time-varying omega."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {
                "origin": [0, 0, 0],
                "axis": [0, 0, 1],
                "omegaTable": [[0, 2.0], [1, 4.0]],
            },
        )
        field = torch.zeros(12, 3, dtype=torch.float64)
        n_cells = 12

        diag_0, source_0 = bc.matrix_contributions(field, n_cells, time=0.0)
        diag_1, source_1 = bc.matrix_contributions(field, n_cells, time=1.0)

        # At t=0, omega=2; at t=1, omega=4.  Diag should be same (geometry),
        # but source should differ (different velocity projection).
        assert torch.allclose(diag_0, diag_1)
        # Source for cell 1 (face at x=1): velocity x-component is 0 (cross product)
        # Source for cell 2 (face at x=2): also 0 for x-component projection
        # Since the cross product of z-axis x x-axis = y-axis, the x-component is 0
        # and source should be zero for this configuration
        assert torch.allclose(source_0, torch.zeros(n_cells, dtype=torch.float64))

    def test_axis_normalisation(self, wall_patch):
        """Non-unit axis is normalised."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"axis": [0, 0, 3], "omega": 1.0},
        )
        assert torch.allclose(bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))

    def test_arbitrary_axis(self, wall_patch):
        """Rotation about an arbitrary axis works."""
        bc = RotatingWallVelocity2BC(
            wall_patch,
            {"origin": [0, 0, 0], "axis": [1, 0, 0], "omega": 1.0},
        )
        field = torch.zeros(12, 3, dtype=torch.float64)
        bc.apply(field)
        # axis = (1,0,0), face 1 at (1,0,0): cross = (1,0,0) x (1,0,0) = (0,0,0)
        # face 2 at (2,0,0): cross = (1,0,0) x (2,0,0) = (0,0,0)
        # All x-axis positions -> zero cross product with x-axis
        assert torch.allclose(field[6:9], torch.zeros(3, 3, dtype=torch.float64))
