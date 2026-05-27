"""Tests for empty, advective, uniformFixedValue, surfaceNormalFixedValue BCs."""

import math

import pytest
import torch

from pyfoam.boundary import (
    AdvectiveBC,
    BoundaryCondition,
    EmptyBC,
    SurfaceNormalFixedValueBC,
    UniformFixedValueBC,
)


# =========================================================================
# EmptyBC
# =========================================================================


class TestEmptyBC:
    """Test the empty boundary condition (critical for 2D simulations)."""

    def test_registration(self):
        """empty is registered in the RTS registry."""
        assert "empty" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("empty", simple_patch)
        assert isinstance(bc, EmptyBC)

    def test_apply_is_noop(self, simple_patch):
        """apply() does not modify the field."""
        bc = EmptyBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field, original)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with patch_idx is still a no-op."""
        bc = EmptyBC(simple_patch)
        field = torch.arange(20, dtype=torch.float64)
        original = field.clone()
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field, original)

    def test_zero_matrix_contribution(self, simple_patch):
        """matrix_contributions() returns zeros."""
        bc = EmptyBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=5)
        assert torch.allclose(diag, torch.zeros(5, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(5, dtype=torch.float64))

    def test_accumulates_into_existing(self, simple_patch):
        """matrix_contributions() preserves pre-existing values."""
        bc = EmptyBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag_existing = torch.ones(5, dtype=torch.float64)
        source_existing = torch.ones(5, dtype=torch.float64) * 2.0
        diag, source = bc.matrix_contributions(field, 5, diag_existing, source_existing)
        assert torch.allclose(diag, torch.ones(5, dtype=torch.float64))
        assert torch.allclose(source, torch.full((5,), 2.0, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns 'empty'."""
        bc = EmptyBC(simple_patch)
        assert bc.type_name == "empty"


# =========================================================================
# AdvectiveBC
# =========================================================================


class TestAdvectiveBC:
    """Test the advective boundary condition."""

    def test_registration(self):
        """advective is registered."""
        assert "advective" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("advective", simple_patch)
        assert isinstance(bc, AdvectiveBC)

    def test_apply_no_phi_is_zero_gradient(self, simple_patch):
        """Without phi, apply() behaves as zero-gradient (copies owners)."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_with_flux(self, simple_patch):
        """apply() with face flux still copies owner values for outflow."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 10.0
        field[2] = 15.0
        # Positive flux = outflow
        phi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, phi=phi, dt=0.1)
        # For outflow, face value is still the owner value
        assert torch.allclose(field[10], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(15.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 7.0
        field[1] = 8.0
        field[2] = 9.0
        bc.apply(field, patch_idx=3)
        assert torch.allclose(field[3], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[4], torch.tensor(8.0, dtype=torch.float64))
        assert torch.allclose(field[5], torch.tensor(9.0, dtype=torch.float64))

    def test_matrix_contributions_no_phi(self, simple_patch):
        """No phi → zero matrix contribution."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_outflow(self, simple_patch):
        """Outflow flux adds to source of owner cells."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        phi = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, phi=phi)
        # diag stays zero (pure advective, no implicit part)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        # source = phi values (all positive = outflow)
        expected = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        assert torch.allclose(source, expected)

    def test_matrix_contributions_inflow_no_contribution(self, simple_patch):
        """Inflow (negative phi) should not contribute."""
        bc = AdvectiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        phi = torch.tensor([-2.0, -4.0, -6.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, phi=phi)
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns 'advective'."""
        bc = AdvectiveBC(simple_patch)
        assert bc.type_name == "advective"


# =========================================================================
# UniformFixedValueBC
# =========================================================================


class TestUniformFixedValueBC:
    """Test the uniformFixedValue boundary condition."""

    def test_registration(self):
        """uniformFixedValue is registered."""
        assert "uniformFixedValue" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "uniformFixedValue", simple_patch, {"uniformValue": 5.0}
        )
        assert isinstance(bc, UniformFixedValueBC)

    def test_constant_value(self, simple_patch):
        """Plain number → constant value at all times."""
        bc = UniformFixedValueBC(simple_patch, {"uniformValue": 3.14})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, t=0.0)
        assert torch.allclose(field[10], torch.tensor(3.14, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(3.14, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.14, dtype=torch.float64))

    def test_constant_value_same_at_different_times(self, simple_patch):
        """Constant value is the same regardless of time."""
        bc = UniformFixedValueBC(simple_patch, {"uniformValue": 2.0})
        v0 = bc.evaluate(0.0)
        v5 = bc.evaluate(5.0)
        assert torch.allclose(v0, v5)

    def test_ramp_function(self, simple_patch):
        """Ramp function: linear interpolation."""
        bc = UniformFixedValueBC(
            simple_patch,
            {"uniformValue": "ramp", "start": 0.0, "end": 10.0, "duration": 1.0},
        )
        v0 = bc.evaluate(0.0)
        v05 = bc.evaluate(0.5)
        v1 = bc.evaluate(1.0)
        assert torch.allclose(v0, torch.full((3,), 0.0, dtype=torch.float64))
        assert torch.allclose(v05, torch.full((3,), 5.0, dtype=torch.float64))
        assert torch.allclose(v1, torch.full((3,), 10.0, dtype=torch.float64))

    def test_ramp_clamps(self, simple_patch):
        """Ramp function clamps beyond duration."""
        bc = UniformFixedValueBC(
            simple_patch,
            {"uniformValue": "ramp", "start": 0.0, "end": 10.0, "duration": 1.0},
        )
        v_neg = bc.evaluate(-0.5)
        v_over = bc.evaluate(2.0)
        assert torch.allclose(v_neg, torch.full((3,), 0.0, dtype=torch.float64))
        assert torch.allclose(v_over, torch.full((3,), 10.0, dtype=torch.float64))

    def test_sine_function(self, simple_patch):
        """Sine function oscillates correctly."""
        bc = UniformFixedValueBC(
            simple_patch,
            {
                "uniformValue": "sine",
                "amplitude": 2.0,
                "frequency": 1.0,
                "phase": 0.0,
                "offset": 0.0,
            },
        )
        # sin(0) = 0
        v0 = bc.evaluate(0.0)
        assert torch.allclose(v0, torch.zeros(3, dtype=torch.float64), atol=1e-10)
        # sin(pi/2) with freq=1 → t = 0.25
        v_peak = bc.evaluate(0.25)
        assert torch.allclose(
            v_peak,
            torch.full((3,), 2.0, dtype=torch.float64),
            atol=1e-10,
        )

    def test_callable_function(self, simple_patch):
        """Custom callable is accepted."""
        bc = UniformFixedValueBC(
            simple_patch, {"uniformValue": lambda t: t * 10.0}
        )
        v = bc.evaluate(0.3)
        assert torch.allclose(v, torch.full((3,), 3.0, dtype=torch.float64))

    def test_apply_at_time(self, simple_patch):
        """apply() sets field values from the time function."""
        bc = UniformFixedValueBC(
            simple_patch,
            {"uniformValue": "ramp", "start": 0.0, "end": 100.0, "duration": 1.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, t=0.5)
        assert torch.allclose(field[10], torch.tensor(50.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with patch_idx."""
        bc = UniformFixedValueBC(simple_patch, {"uniformValue": 7.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, t=0.0)
        assert torch.allclose(field[5], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(7.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diagonal and source for constant value."""
        bc = UniformFixedValueBC(simple_patch, {"uniformValue": 4.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, t=0.0)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        # owner [0,1,2] → diag = [2.0, 2.0, 2.0]
        expected_diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        expected_src = torch.tensor([8.0, 8.0, 8.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        assert torch.allclose(source, expected_src)

    def test_type_name(self, simple_patch):
        """type_name returns 'uniformFixedValue'."""
        bc = UniformFixedValueBC(simple_patch, {"uniformValue": 1.0})
        assert bc.type_name == "uniformFixedValue"


# =========================================================================
# SurfaceNormalFixedValueBC
# =========================================================================


class TestSurfaceNormalFixedValueBC:
    """Test the surfaceNormalFixedValue boundary condition."""

    def test_registration(self):
        """surfaceNormalFixedValue is registered."""
        assert "surfaceNormalFixedValue" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "surfaceNormalFixedValue", simple_patch, {"value": 5.0}
        )
        assert isinstance(bc, SurfaceNormalFixedValueBC)

    def test_scalar_field_sets_magnitude(self, simple_patch):
        """For a scalar field, apply() sets the magnitude."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 3.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(3.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(3.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.0, dtype=torch.float64))

    def test_vector_field_sets_velocity(self, simple_patch):
        """For a vector field, apply() sets magnitude * normal."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 5.0})
        # (15, 3) vector field
        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # All normals are (1,0,0) → velocity = (5,0,0)
        expected = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_patch_idx_scalar(self, simple_patch):
        """apply() with patch_idx for scalar field."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 2.5})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(2.5, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(2.5, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(2.5, dtype=torch.float64))

    def test_apply_with_patch_idx_vector(self, simple_patch):
        """apply() with patch_idx for vector field."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 4.0})
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        expected = torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_nonuniform_normals(self, two_face_patch):
        """Velocity direction follows face normals."""
        # two_face_patch normals are (0,1,0)
        bc = SurfaceNormalFixedValueBC(two_face_patch, {"value": 3.0})
        field = torch.zeros(10, 3, dtype=torch.float64)
        bc.apply(field)
        expected = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)
        assert torch.allclose(field[6], expected)

    def test_magnitude_property(self, simple_patch):
        """magnitude property is accessible and settable."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 1.0})
        assert torch.allclose(bc.magnitude, torch.tensor(1.0, dtype=torch.float64))
        bc.magnitude = 10.0
        assert torch.allclose(bc.magnitude, torch.tensor(10.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diagonal and source from magnitude."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 6.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        expected_diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        expected_src = torch.tensor([12.0, 12.0, 12.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        assert torch.allclose(source, expected_src)

    def test_type_name(self, simple_patch):
        """type_name returns 'surfaceNormalFixedValue'."""
        bc = SurfaceNormalFixedValueBC(simple_patch, {"value": 1.0})
        assert bc.type_name == "surfaceNormalFixedValue"
