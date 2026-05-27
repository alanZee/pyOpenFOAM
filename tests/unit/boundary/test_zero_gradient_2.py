"""Tests for zeroGradient2 boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.zero_gradient_2 import ZeroGradient2BC


class TestZeroGradient2BC:
    """Test the zeroGradient2 boundary condition."""

    def test_registration(self):
        """zeroGradient2 is registered in the RTS registry."""
        assert "zeroGradient2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("zeroGradient2", simple_patch)
        assert isinstance(bc, ZeroGradient2BC)

    def test_apply_copies_owner_values(self, simple_patch):
        """apply() without correction copies owner-cell values (plain zero gradient)."""
        bc = ZeroGradient2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = ZeroGradient2BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 15.0
        field[2] = 25.0
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(15.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(25.0, dtype=torch.float64))

    def test_correction_disabled_by_default(self, simple_patch):
        """Correction is disabled by default."""
        bc = ZeroGradient2BC(simple_patch)
        assert bc.correction_enabled is False

    def test_correction_enabled(self, simple_patch):
        """Correction can be enabled via coefficients."""
        bc = ZeroGradient2BC(simple_patch, {"nonOrthogonalCorrection": True})
        assert bc.correction_enabled is True

    def test_default_correction_factor(self, simple_patch):
        """Default correction factor is 1.0."""
        bc = ZeroGradient2BC(simple_patch)
        assert bc.correction_factor == 1.0

    def test_custom_correction_factor(self, simple_patch):
        """Correction factor can be set via coefficients."""
        bc = ZeroGradient2BC(simple_patch, {"correctionFactor": 0.5})
        assert bc.correction_factor == 0.5

    def test_apply_with_correction(self, simple_patch):
        """apply() adds correction when enabled and data is set."""
        bc = ZeroGradient2BC(simple_patch, {
            "nonOrthogonalCorrection": True,
            "correctionFactor": 1.0,
        })
        correction = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float64)
        bc.set_correction(correction)

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        # face = owner + factor * correction
        assert torch.allclose(field[10], torch.tensor(10.5, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(19.5, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(31.0, dtype=torch.float64))

    def test_apply_with_partial_correction_factor(self, simple_patch):
        """Correction is scaled by correction_factor."""
        bc = ZeroGradient2BC(simple_patch, {
            "nonOrthogonalCorrection": True,
            "correctionFactor": 0.5,
        })
        correction = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        bc.set_correction(correction)

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        # face = owner + 0.5 * correction
        assert torch.allclose(field[10], torch.tensor(11.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(22.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(33.0, dtype=torch.float64))

    def test_correction_disabled_ignores_correction_data(self, simple_patch):
        """When correction is disabled, correction data is ignored."""
        bc = ZeroGradient2BC(simple_patch, {
            "nonOrthogonalCorrection": False,
        })
        correction = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.set_correction(correction)

        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)
        # Without correction enabled, should be plain zero gradient
        assert torch.allclose(field[10], torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(3.0, dtype=torch.float64))

    def test_correction_enabled_no_data_set(self, simple_patch):
        """Correction enabled but no data set -> falls back to plain zero gradient."""
        bc = ZeroGradient2BC(simple_patch, {
            "nonOrthogonalCorrection": True,
        })
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 7.0
        field[1] = 8.0
        field[2] = 9.0
        bc.apply(field)
        # No correction data -> plain zero gradient
        assert torch.allclose(field[10], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(8.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(9.0, dtype=torch.float64))

    def test_no_matrix_contribution(self, simple_patch):
        """zeroGradient2 has zero matrix contribution (same as zeroGradient)."""
        bc = ZeroGradient2BC(simple_patch, {
            "nonOrthogonalCorrection": True,
            "correctionFactor": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = ZeroGradient2BC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = ZeroGradient2BC(simple_patch)
        r = repr(bc)
        assert "ZeroGradient2BC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = ZeroGradient2BC(simple_patch)
        assert bc.type_name == "zeroGradient2"
