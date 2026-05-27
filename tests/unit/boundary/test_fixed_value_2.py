"""Tests for fixedValue2 boundary condition."""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.fixed_value_2 import FixedValue2BC


class TestFixedValue2BC:
    """Test the fixedValue2 boundary condition."""

    def test_registration(self):
        """fixedValue2 is registered in the RTS registry."""
        assert "fixedValue2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "fixedValue2", simple_patch,
            {"baseValue": 1.0, "timeFunction": "constant"},
        )
        assert isinstance(bc, FixedValue2BC)

    def test_constant_time_function(self, simple_patch):
        """constant time function returns factor=1 at all times."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 5.0,
            "timeFunction": "constant",
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.0)
        assert torch.allclose(field[10:13], torch.full((3,), 5.0, dtype=torch.float64))

        field2 = torch.zeros(15, dtype=torch.float64)
        bc.apply(field2, time=999.0)
        assert torch.allclose(field2[10:13], torch.full((3,), 5.0, dtype=torch.float64))

    def test_sine_time_function(self, simple_patch):
        """sine time function: value = base * amplitude * sin(2*pi*f*t)."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 10.0,
            "timeFunction": "sine",
            "amplitude": 0.5,
            "frequency": 1.0,
        })
        # t=0.25 -> sin(2*pi*0.25) = sin(pi/2) = 1.0
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.25)
        expected = 10.0 * 0.5 * 1.0  # 5.0
        assert torch.allclose(
            field[10:13], torch.full((3,), expected, dtype=torch.float64), atol=1e-10
        )

    def test_cosine_time_function(self, simple_patch):
        """cosine time function at t=0 -> amplitude."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 2.0,
            "timeFunction": "cosine",
            "amplitude": 3.0,
            "frequency": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.0)
        # cos(0) = 1 -> factor = 3.0, value = 2.0 * 3.0 = 6.0
        assert torch.allclose(
            field[10:13], torch.full((3,), 6.0, dtype=torch.float64), atol=1e-10
        )

    def test_linear_time_function(self, simple_patch):
        """linear time function: factor = amplitude * t."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 4.0,
            "timeFunction": "linear",
            "amplitude": 2.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=3.0)
        # factor = 2.0 * 3.0 = 6.0, value = 4.0 * 6.0 = 24.0
        assert torch.allclose(
            field[10:13], torch.full((3,), 24.0, dtype=torch.float64), atol=1e-10
        )

    def test_step_time_function(self, simple_patch):
        """step function: 0 before onset, amplitude after."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 1.0,
            "timeFunction": "step",
            "amplitude": 5.0,
            "phase": 1.0,  # onset time
        })
        # Before onset
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.5)
        assert torch.allclose(field[10:13], torch.zeros(3, dtype=torch.float64))

        # After onset
        field2 = torch.zeros(15, dtype=torch.float64)
        bc.apply(field2, time=2.0)
        # factor = 5.0, value = 1.0 * 5.0 = 5.0
        assert torch.allclose(field2[10:13], torch.full((3,), 5.0, dtype=torch.float64))

    def test_default_value_is_zero(self, simple_patch):
        """Default baseValue is 0 when no coefficient is given."""
        bc = FixedValue2BC(simple_patch)
        assert torch.allclose(bc.base_value, torch.zeros(3, dtype=torch.float64))

    def test_default_time_function_is_constant(self, simple_patch):
        """Default timeFunction is constant."""
        bc = FixedValue2BC(simple_patch)
        assert bc.time_function == "constant"

    def test_unknown_time_function_raises(self, simple_patch):
        """Invalid time function name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time function"):
            FixedValue2BC(simple_patch, {"timeFunction": "invalid"})

    def test_tensor_base_value(self, simple_patch):
        """Per-face tensor baseValue is used directly."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = FixedValue2BC(simple_patch, {"baseValue": vals})
        assert torch.allclose(bc.base_value, vals)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 10.0,
            "timeFunction": "constant",
        })
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, time=0.0)
        assert torch.allclose(field[5:8], torch.full((3,), 10.0, dtype=torch.float64))

    def test_matrix_contributions_use_base_value(self, simple_patch):
        """Matrix contributions use base_value (not time-dependent)."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 10.0,
            "timeFunction": "sine",
            "amplitude": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # diag = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * base_value = 2.0 * 10.0 = 20.0 per face
        assert torch.allclose(source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64))

    def test_phase_offset_sine(self, simple_patch):
        """sine function with phase offset: sin(2*pi*f*t + phase)."""
        bc = FixedValue2BC(simple_patch, {
            "baseValue": 1.0,
            "timeFunction": "sine",
            "amplitude": 1.0,
            "frequency": 1.0,
            "phase": math.pi / 2.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.0)
        # sin(0 + pi/2) = 1.0
        assert torch.allclose(
            field[10:13], torch.full((3,), 1.0, dtype=torch.float64), atol=1e-10
        )

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = FixedValue2BC(simple_patch, {"baseValue": 1.0})
        r = repr(bc)
        assert "FixedValue2BC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = FixedValue2BC(simple_patch, {"baseValue": 1.0})
        assert bc.type_name == "fixedValue2"
