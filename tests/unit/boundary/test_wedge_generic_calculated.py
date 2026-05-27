"""Tests for wedge, generic, and calculated boundary conditions."""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    WedgeBC,
    GenericBC,
    CalculatedBC,
)


# ---- WedgeBC ----


class TestWedgeBC:
    """Test the wedge boundary condition."""

    def test_registration(self):
        """wedge is registered in the RTS registry."""
        assert "wedge" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("wedge", simple_patch)
        assert isinstance(bc, WedgeBC)

    def test_apply_noop(self, simple_patch):
        """apply() does not modify the field."""
        bc = WedgeBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.equal(field, original)

    def test_apply_with_patch_idx_noop(self, simple_patch):
        """apply() with patch_idx is still a no-op."""
        bc = WedgeBC(simple_patch)
        field = torch.arange(20, dtype=torch.float64)
        original = field.clone()
        bc.apply(field, patch_idx=5)
        assert torch.equal(field, original)

    def test_matrix_contributions_zero(self, simple_patch):
        """Matrix contributions are zero."""
        bc = WedgeBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Zero contribution does not change pre-existing diag/source."""
        bc = WedgeBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        assert torch.allclose(diag, torch.ones(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.ones(n_cells, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = WedgeBC(simple_patch)
        assert bc.type_name == "wedge"

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = WedgeBC(simple_patch)
        r = repr(bc)
        assert "WedgeBC" in r
        assert "testPatch" in r


# ---- GenericBC ----


class TestGenericBC:
    """Test the generic boundary condition."""

    def test_registration(self):
        """generic is registered in the RTS registry."""
        assert "generic" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("generic", simple_patch, {
            "value": 10.0,
            "gradient": 0.0,
            "valueFraction": 0.5,
        })
        assert isinstance(bc, GenericBC)

    def test_default_coeffs(self, simple_patch):
        """Default coefficients: value=0, gradient=0, valueFraction=1."""
        bc = GenericBC(simple_patch)
        assert torch.allclose(bc.value, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.gradient, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.value_fraction, torch.ones(3, dtype=torch.float64))

    def test_pure_fixed_value(self, simple_patch):
        """valueFraction=1 behaves as pure fixed value."""
        bc = GenericBC(simple_patch, {
            "value": 5.0,
            "valueFraction": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 5.0, dtype=torch.float64))

    def test_pure_zero_gradient(self, simple_patch):
        """valueFraction=0 behaves as pure zero gradient (copy owner values)."""
        bc = GenericBC(simple_patch, {
            "value": 100.0,
            "valueFraction": 0.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        # Set owner cell values: cells 0, 1, 2
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)
        # Faces at indices [10, 11, 12] should match owner cells
        assert torch.allclose(field[10], torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.0, dtype=torch.float64))

    def test_blended_apply(self, simple_patch):
        """Intermediate valueFraction blends fixed value and owner value."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.8,
        })
        field = torch.zeros(15, dtype=torch.float64)
        # Owner cells: 0, 1, 2 all have value 0
        # face = 0.8 * 10 + 0.2 * 0 = 8.0
        bc.apply(field)
        assert torch.allclose(
            field[10:13],
            torch.full((3,), 8.0, dtype=torch.float64),
        )

    def test_blended_apply_nonzero_owner(self, simple_patch):
        """Blend with non-zero owner values."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 2.0
        field[1] = 4.0
        field[2] = 6.0
        bc.apply(field)
        # face = 0.5 * 10 + 0.5 * owner
        assert torch.allclose(field[10], torch.tensor(6.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(8.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.8,
        })
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(
            field[5:8],
            torch.full((3,), 8.0, dtype=torch.float64),
        )

    def test_matrix_contributions_weighted(self, simple_patch):
        """Matrix contributions are weighted by valueFraction."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # coeff = deltaCoeff * area * valueFraction = 2.0 * 1.0 * 0.5 = 1.0
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
        # source = coeff * value = 1.0 * 10.0 = 10.0
        assert torch.allclose(source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))

    def test_matrix_contributions_pure_fixed_value(self, simple_patch):
        """valueFraction=1 gives same matrix contributions as fixedValue."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # coeff = 2.0 * 1.0 * 1.0 = 2.0 (same as fixedValue)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64))

    def test_matrix_contributions_zero_fraction(self, simple_patch):
        """valueFraction=0 gives zero matrix contributions."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing diag/source."""
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # diag = 1.0 + 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = 1.0 + 10.0 = 11.0
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_tensor_value_fraction(self, simple_patch):
        """Per-face valueFraction tensor is supported."""
        vf = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        bc = GenericBC(simple_patch, {
            "value": 10.0,
            "valueFraction": vf,
        })
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 0.0
        field[1] = 0.0
        field[2] = 0.0
        bc.apply(field)
        # face[0]: 0.0 * 10 + 1.0 * 0 = 0
        # face[1]: 0.5 * 10 + 0.5 * 0 = 5
        # face[2]: 1.0 * 10 + 0.0 * 0 = 10
        assert torch.allclose(field[10], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(10.0, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = GenericBC(simple_patch, {"value": 1.0, "valueFraction": 0.5})
        assert bc.type_name == "generic"

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = GenericBC(simple_patch, {"value": 1.0})
        r = repr(bc)
        assert "GenericBC" in r
        assert "testPatch" in r


# ---- CalculatedBC ----


class TestCalculatedBC:
    """Test the calculated boundary condition."""

    def test_registration(self):
        """calculated is registered in the RTS registry."""
        assert "calculated" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("calculated", simple_patch)
        assert isinstance(bc, CalculatedBC)

    def test_apply_copies_owner_values(self, simple_patch):
        """apply() copies owner cell values to boundary faces."""
        bc = CalculatedBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        # Set owner cell values: cells 0, 1, 2
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)
        # Faces at indices [10, 11, 12] should match owner cells
        assert torch.allclose(field[10], torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx copies owner values at offset."""
        bc = CalculatedBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 7.0
        field[1] = 8.0
        field[2] = 9.0
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(8.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(9.0, dtype=torch.float64))

    def test_matrix_contributions_zero(self, simple_patch):
        """Matrix contributions are zero."""
        bc = CalculatedBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Zero contribution does not change pre-existing diag/source."""
        bc = CalculatedBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        assert torch.allclose(diag, torch.ones(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.ones(n_cells, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = CalculatedBC(simple_patch)
        assert bc.type_name == "calculated"

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = CalculatedBC(simple_patch)
        r = repr(bc)
        assert "CalculatedBC" in r
        assert "testPatch" in r
