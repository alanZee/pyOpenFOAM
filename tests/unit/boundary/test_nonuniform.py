"""Tests for nonUniform boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.nonuniform import NonUniformBC


class TestNonUniformBC:
    """Test the nonUniform boundary condition."""

    def test_registration(self):
        """nonUniform is registered in the RTS registry."""
        assert "nonUniform" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = BoundaryCondition.create("nonUniform", simple_patch, {"value": vals})
        assert isinstance(bc, NonUniformBC)

    def test_tensor_value(self, simple_patch):
        """Per-face tensor value is used directly."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        assert torch.allclose(bc.value, vals)

    def test_list_value(self, simple_patch):
        """Per-face list value is converted to tensor."""
        bc = NonUniformBC(simple_patch, {"value": [10.0, 20.0, 30.0]})
        expected = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        assert torch.allclose(bc.value, expected)

    def test_default_value_is_zero(self, simple_patch):
        """Default value is zeros when no coefficient is given."""
        bc = NonUniformBC(simple_patch)
        assert torch.allclose(bc.value, torch.zeros(3, dtype=torch.float64))

    def test_wrong_length_raises(self, simple_patch):
        """Value with wrong number of elements raises ValueError."""
        vals = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="2 elements"):
            NonUniformBC(simple_patch, {"value": vals})

    def test_apply_sets_face_values(self, simple_patch):
        """apply() sets the boundary-face values in the field."""
        vals = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], vals)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        vals = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], vals)

    def test_matrix_contributions_penalty_method(self, simple_patch):
        """Matrix contributions use penalty method."""
        vals = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # diag[c] += deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source[c] = deltaCoeff * area * value
        expected_source = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing diag/source."""
        vals = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_value_setter(self, simple_patch):
        """Value can be updated after construction."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = NonUniformBC(simple_patch, {"value": vals})
        new_vals = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        bc.value = new_vals
        assert torch.allclose(bc.value, new_vals)

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = NonUniformBC(simple_patch)
        r = repr(bc)
        assert "NonUniformBC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = NonUniformBC(simple_patch)
        assert bc.type_name == "nonUniform"
