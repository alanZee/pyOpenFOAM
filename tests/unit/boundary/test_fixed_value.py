"""Tests for fixedValue boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, FixedValueBC


class TestFixedValueBC:
    """Test the fixedValue boundary condition."""

    def test_registration(self):
        """fixedValue is registered in the RTS registry."""
        assert "fixedValue" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("fixedValue", simple_patch, {"value": 5.0})
        assert isinstance(bc, FixedValueBC)

    def test_uniform_value(self, simple_patch):
        """Uniform value is broadcast to all faces."""
        bc = FixedValueBC(simple_patch, {"value": 3.14})
        assert bc.value.shape == (3,)
        assert torch.allclose(bc.value, torch.full((3,), 3.14, dtype=torch.float64))

    def test_tensor_value(self, simple_patch):
        """Per-face tensor value is used directly."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = FixedValueBC(simple_patch, {"value": vals})
        assert torch.allclose(bc.value, vals)

    def test_default_value_is_zero(self, simple_patch):
        """Default value is 0 when no coefficient is given."""
        bc = FixedValueBC(simple_patch)
        assert torch.allclose(bc.value, torch.zeros(3, dtype=torch.float64))

    def test_apply_sets_face_values(self, simple_patch):
        """apply() sets the boundary-face values in the field."""
        bc = FixedValueBC(simple_patch, {"value": 7.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # Faces at indices [10, 11, 12] should be 7.0
        assert torch.allclose(field[10:13], torch.full((3,), 7.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        bc = FixedValueBC(simple_patch, {"value": 42.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 42.0, dtype=torch.float64))

    def test_matrix_contributions_penalty_method(self, simple_patch):
        """Matrix contributions use penalty method: large diag + source."""
        bc = FixedValueBC(simple_patch, {"value": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # diag[c] += deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 each get one face contribution
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * value = 2.0 * 10.0 = 20.0 per face
        assert torch.allclose(source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing diag/source."""
        bc = FixedValueBC(simple_patch, {"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # diag should be 1.0 + 2.0 = 3.0
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        # source should be 1.0 + 10.0 = 11.0
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_value_setter(self, simple_patch):
        """Value can be updated after construction."""
        bc = FixedValueBC(simple_patch, {"value": 1.0})
        bc.value = 99.0
        assert torch.allclose(bc.value, torch.full((3,), 99.0, dtype=torch.float64))

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = FixedValueBC(simple_patch, {"value": 1.0})
        r = repr(bc)
        assert "FixedValueBC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = FixedValueBC(simple_patch, {"value": 1.0})
        assert bc.type_name == "fixedValue"
