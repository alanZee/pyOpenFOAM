"""Tests for fixedNormal, slip, and pressureInletOutlet boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.fixed_normal import FixedNormalBC
from pyfoam.boundary.slip import SlipBC
from pyfoam.boundary.inlet_outlet_2 import PressureInletOutletBC


# ======================================================================
# FixedNormalBC
# ======================================================================


class TestFixedNormalBC:
    """Test the fixedNormal boundary condition."""

    def test_registration(self):
        """fixedNormal is registered in the RTS registry."""
        assert "fixedNormal" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("fixedNormal", simple_patch, coeffs={"value": 5.0})
        assert isinstance(bc, FixedNormalBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 1.0})
        assert bc.type_name == "fixedNormal"

    def test_default_value(self, simple_patch):
        """Default value is 0.0."""
        bc = FixedNormalBC(simple_patch)
        assert torch.allclose(bc.value, torch.tensor(0.0, dtype=torch.float64))

    def test_custom_value(self, simple_patch):
        """Value is parsed from the 'value' coefficient."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 3.5})
        assert torch.allclose(bc.value, torch.tensor(3.5, dtype=torch.float64))

    def test_apply_scalar_field(self, simple_patch):
        """For scalar fields, applies fixed value."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 42.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(42.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(42.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(42.0, dtype=torch.float64))

    def test_apply_vector_sets_normal_preserves_tangential(self, simple_patch):
        """For vector fields, sets normal component and preserves tangential."""
        # simple_patch normals face +x direction
        bc = FixedNormalBC(simple_patch, coeffs={"value": 10.0})
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Owner cell values: [5, 3, 0], [10, -2, 1], [1, 0, 4]
        field[0] = torch.tensor([5.0, 3.0, 0.0])
        field[1] = torch.tensor([10.0, -2.0, 1.0])
        field[2] = torch.tensor([1.0, 0.0, 4.0])
        bc.apply(field)
        # Normal (+x) = 10.0, tangential preserved from owner
        assert torch.allclose(field[10], torch.tensor([10.0, 3.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([10.0, -2.0, 1.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([10.0, 0.0, 4.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 7.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(7.0, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 1.0})
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        )

    def test_value_setter(self, simple_patch):
        """Value can be updated via setter."""
        bc = FixedNormalBC(simple_patch, coeffs={"value": 1.0})
        bc.value = 99.0
        assert torch.allclose(bc.value, torch.tensor(99.0, dtype=torch.float64))


# ======================================================================
# SlipBC
# ======================================================================


class TestSlipBC:
    """Test the slip boundary condition."""

    def test_registration(self):
        """slip is registered in the RTS registry."""
        assert "slip" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("slip", simple_patch)
        assert isinstance(bc, SlipBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = SlipBC(simple_patch)
        assert bc.type_name == "slip"

    def test_apply_scalar_copies_owner_values(self, simple_patch):
        """For scalar fields, apply() behaves as zeroGradient."""
        bc = SlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_vector_removes_normal_component(self, simple_patch):
        """For vector fields, apply() removes the normal component (free slip)."""
        bc = SlipBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Normals face +x; set owner values with both normal and tangential
        field[0] = torch.tensor([5.0, 3.0, 0.0])
        field[1] = torch.tensor([10.0, -2.0, 1.0])
        field[2] = torch.tensor([1.0, 0.0, 4.0])
        bc.apply(field)
        # Normal component removed, tangential preserved
        assert torch.allclose(field[10], torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = SlipBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 6.0
        field[2] = 7.0
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(6.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(7.0, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = SlipBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_zero(self, simple_patch):
        """Free-slip has zero matrix contribution (no flux coupling)."""
        bc = SlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_preserves_existing(self, simple_patch):
        """matrix_contributions preserves pre-existing diag/source values."""
        bc = SlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 5
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 2.0
        new_diag, new_source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        assert torch.allclose(new_diag, torch.ones(n_cells, dtype=torch.float64))
        assert torch.allclose(new_source, torch.ones(n_cells, dtype=torch.float64) * 2.0)


# ======================================================================
# PressureInletOutletBC
# ======================================================================


class TestPressureInletOutletBC:
    """Test the pressureInletOutlet boundary condition."""

    def test_registration(self):
        """pressureInletOutlet is registered in the RTS registry."""
        assert "pressureInletOutlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "pressureInletOutlet", simple_patch, coeffs={"p0": 101325.0}
        )
        assert isinstance(bc, PressureInletOutletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = PressureInletOutletBC(simple_patch)
        assert bc.type_name == "pressureInletOutlet"

    def test_p0_property(self, simple_patch):
        """p0 returns the total pressure."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 101325.0})
        assert bc.p0 == 101325.0

    def test_default_p0(self, simple_patch):
        """Default p0 is 0.0."""
        bc = PressureInletOutletBC(simple_patch)
        assert bc.p0 == 0.0

    def test_apply_no_flux_acts_as_zero_gradient(self, simple_patch):
        """Without flux/velocity, apply() copies owner values (all outflow)."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 100.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_inflow_uses_p0(self, simple_patch):
        """Inflow (negative flux) applies total pressure p0."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        # Negative flux = inflow
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(field[10], torch.tensor(101325.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(101325.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(101325.0, dtype=torch.float64))

    def test_apply_outflow_copies_owner_values(self, simple_patch):
        """Outflow (positive flux) applies zero gradient."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(field[10], torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(200.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(300.0, dtype=torch.float64))

    def test_apply_mixed_flux(self, simple_patch):
        """Mixed flux: some faces inflow, some outflow."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 500.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        # face 0: negative flux → inflow → p0 = 500
        # face 1: positive flux → outflow → owner[1] = 20
        # face 2: negative flux → inflow → p0 = 500
        flux = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(field[10], torch.tensor(500.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(500.0, dtype=torch.float64))

    def test_apply_with_velocity_fallback(self, simple_patch):
        """Velocity can be used as fallback for direction detection."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 1000.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 6.0
        field[2] = 7.0
        # Normals face +x; velocity -x => v·n < 0 => inflow
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        bc.apply(field, velocity=velocity)
        assert torch.allclose(field[10], torch.tensor(1000.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(1000.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(1000.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 50.0})
        field = torch.zeros(20, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        bc.apply(field, patch_idx=5, flux=flux)
        assert torch.allclose(field[5], torch.tensor(50.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(50.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(50.0, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 100.0})
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_inflow(self, simple_patch):
        """Inflow faces contribute to diagonal and source."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, flux=flux)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        )

    def test_matrix_contributions_outflow(self, simple_patch):
        """Outflow faces contribute nothing."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, flux=flux)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_no_flux(self, simple_patch):
        """Without flux, no matrix contributions (all outflow assumed)."""
        bc = PressureInletOutletBC(simple_patch, coeffs={"p0": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))
