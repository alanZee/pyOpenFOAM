"""Tests for turbulent kinetic energy boundary conditions.

Tests cover TurbulentIntensityKEBC and FixedTurbulentKEBC:
- RTS registration
- Factory creation
- Property access
- apply() with and without velocity
- matrix_contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_bcs import (
    TurbulentIntensityKEBC,
    FixedTurbulentKEBC,
)


# ============================================================================
# TurbulentIntensityKEBC
# ============================================================================


class TestTurbulentIntensityKEBC:
    """turbulentIntensityKE boundary condition tests."""

    def test_registration(self):
        """turbulentIntensityKE is registered in RTS."""
        assert "turbulentIntensityKE" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityKEBC(simple_patch)
        assert bc.type_name == "turbulentIntensityKE"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityKE", simple_patch, {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityKEBC)

    def test_default_intensity(self, simple_patch):
        bc = TurbulentIntensityKEBC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)

    def test_custom_intensity(self, simple_patch):
        bc = TurbulentIntensityKEBC(simple_patch, {"intensity": 0.10})
        assert bc.intensity == pytest.approx(0.10)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2."""
        bc = TurbulentIntensityKEBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, velocity=velocity)

        # k = 1.5 * (0.05 * 10)^2 = 1.5 * 0.25 = 0.375
        expected = torch.full((3,), 0.375, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_without_velocity(self, simple_patch):
        """Default k = 0.01 when no velocity provided."""
        bc = TurbulentIntensityKEBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        expected = torch.full((3,), 0.01, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityKEBC(simple_patch, {"intensity": 0.10})
        velocity = torch.tensor([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        field = bc.apply(field, patch_idx=5, velocity=velocity)

        # k = 1.5 * (0.10 * 5)^2 = 1.5 * 0.25 = 0.375
        expected = torch.full((3,), 0.375, dtype=torch.float64)
        assert torch.allclose(field[5:8], expected)

    def test_apply_nonuniform_velocity(self, simple_patch):
        """Different velocities produce different k values."""
        bc = TurbulentIntensityKEBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, velocity=velocity)

        expected = torch.tensor([
            1.5 * (0.05 * 1.0) ** 2,
            1.5 * (0.05 * 2.0) ** 2,
            1.5 * (0.05 * 3.0) ** 2,
        ], dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag = delta * area, source = coeff * k_default."""
        bc = TurbulentIntensityKEBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))


# ============================================================================
# FixedTurbulentKEBC
# ============================================================================


class TestFixedTurbulentKEBC:
    """fixedTurbulentKE boundary condition tests."""

    def test_registration(self):
        """fixedTurbulentKE is registered in RTS."""
        assert "fixedTurbulentKE" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = FixedTurbulentKEBC(simple_patch)
        assert bc.type_name == "fixedTurbulentKE"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "fixedTurbulentKE", simple_patch, {"k": 0.05},
        )
        assert isinstance(bc, FixedTurbulentKEBC)

    def test_default_k_value(self, simple_patch):
        """Default k value is 0.01."""
        bc = FixedTurbulentKEBC(simple_patch)
        assert bc.k_value == pytest.approx(0.01)

    def test_custom_k_value(self, simple_patch):
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.1})
        assert bc.k_value == pytest.approx(0.1)

    def test_value_alias(self, simple_patch):
        """'value' coefficient is used as alias for 'k'."""
        bc = FixedTurbulentKEBC(simple_patch, {"value": 0.5})
        assert bc.k_value == pytest.approx(0.5)

    def test_k_takes_precedence_over_value(self, simple_patch):
        """'k' coefficient takes precedence over 'value'."""
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.3, "value": 0.5})
        assert bc.k_value == pytest.approx(0.3)

    def test_apply_uniform(self, simple_patch):
        """Uniform k is set on all patch faces."""
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        expected = torch.full((3,), 0.05, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.2})
        field = torch.zeros(20, dtype=torch.float64)
        field = bc.apply(field, patch_idx=5)

        expected = torch.full((3,), 0.2, dtype=torch.float64)
        assert torch.allclose(field[5:8], expected)

    def test_apply_tensor_value(self, simple_patch):
        """Per-face tensor k value."""
        vals = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc = FixedTurbulentKEBC(simple_patch, {"k": vals})
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        assert torch.allclose(field[10:13], vals)

    def test_k_value_setter(self, simple_patch):
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.01})
        bc.k_value = 0.5
        assert bc.k_value == pytest.approx(0.5)

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag = delta * area, source = coeff * k."""
        bc = FixedTurbulentKEBC(simple_patch, {"k": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * k = 2.0 * 0.05 = 0.1
        assert torch.allclose(source, torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64))

    def test_matrix_contributions_tensor(self, simple_patch):
        """Matrix contributions use mean of tensor k value."""
        vals = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc = FixedTurbulentKEBC(simple_patch, {"k": vals})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        # k_mean = 0.2, source = 2.0 * 0.2 = 0.4
        expected_source = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
        assert torch.allclose(source, expected_source)
