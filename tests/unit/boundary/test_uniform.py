"""Tests for uniform boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.uniform import UniformBC


class TestUniformBC:
    """Test the uniform boundary condition."""

    def test_registration(self):
        """uniform is registered in the RTS registry."""
        assert "uniform" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("uniform", simple_patch, {"value": 5.0})
        assert isinstance(bc, UniformBC)

    def test_scalar_value(self, simple_patch):
        """Scalar value is stored as a float."""
        bc = UniformBC(simple_patch, {"value": 3.14})
        assert bc.value == pytest.approx(3.14)

    def test_default_value_is_zero(self, simple_patch):
        """Default value is 0 when no coefficient is given."""
        bc = UniformBC(simple_patch)
        assert bc.value == pytest.approx(0.0)

    def test_apply_sets_face_values(self, simple_patch):
        """apply() broadcasts scalar to all boundary faces."""
        bc = UniformBC(simple_patch, {"value": 7.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 7.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        bc = UniformBC(simple_patch, {"value": 42.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 42.0, dtype=torch.float64))

    def test_matrix_contributions_penalty_method(self, simple_patch):
        """Matrix contributions use penalty method."""
        bc = UniformBC(simple_patch, {"value": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # diag[c] += deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * value = 2.0 * 10.0 = 20.0 per face
        assert torch.allclose(source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing diag/source."""
        bc = UniformBC(simple_patch, {"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_value_setter(self, simple_patch):
        """Value can be updated after construction."""
        bc = UniformBC(simple_patch, {"value": 1.0})
        bc.value = 99.0
        assert bc.value == pytest.approx(99.0)

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = UniformBC(simple_patch, {"value": 1.0})
        r = repr(bc)
        assert "UniformBC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = UniformBC(simple_patch, {"value": 1.0})
        assert bc.type_name == "uniform"
