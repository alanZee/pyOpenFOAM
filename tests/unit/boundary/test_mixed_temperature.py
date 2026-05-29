"""Tests for mixed temperature boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mixed_temperature import MixedTemperatureBC


class TestMixedTemperatureBC:
    """Test the mixedTemperature boundary condition."""

    def test_registration(self):
        assert "mixedTemperature" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mixedTemperature", simple_patch, {"T_ref": 300.0, "alpha": 10.0},
        )
        assert isinstance(bc, MixedTemperatureBC)

    def test_type_name(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch)
        assert bc.type_name == "mixedTemperature"

    def test_default_t_ref(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch)
        assert bc.t_ref == pytest.approx(300.0)

    def test_custom_t_ref(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 350.0})
        assert bc.t_ref == pytest.approx(350.0)

    def test_default_alpha(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch)
        assert bc.alpha == pytest.approx(10.0)

    def test_custom_alpha(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch, {"alpha": 25.0})
        assert bc.alpha == pytest.approx(25.0)

    def test_apply_robin_blend(self, simple_patch):
        """Robin blend: T = (h*T_ref + delta*T_int) / (h + delta)."""
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 300.0, "alpha": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 350.0  # owner cell 0
        bc.apply(field)
        # delta = 2.0, h = 10.0
        expected = (10.0 * 300.0 + 2.0 * 350.0) / (10.0 + 2.0)
        assert field[10] == pytest.approx(expected)

    def test_apply_high_alpha_fixed_value(self, simple_patch):
        """High alpha: approaches fixed value (T_ref)."""
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 300.0, "alpha": 1e12})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 500.0
        bc.apply(field)
        assert field[10] == pytest.approx(300.0, rel=1e-4)

    def test_apply_low_alpha_zero_gradient(self, simple_patch):
        """Low alpha: approaches zero-gradient (T_interior)."""
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 300.0, "alpha": 1e-12})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 500.0
        bc.apply(field)
        assert field[10] == pytest.approx(500.0, rel=1e-4)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 300.0, "alpha": 10.0})
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 400.0
        field[1] = 400.0
        field[2] = 400.0
        bc.apply(field, patch_idx=5)
        expected = (10.0 * 300.0 + 2.0 * 400.0) / (10.0 + 2.0)
        assert field[5] == pytest.approx(expected)

    def test_matrix_contributions(self, simple_patch):
        bc = MixedTemperatureBC(simple_patch, {"T_ref": 300.0, "alpha": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = h * A / (h + delta) = 10 * 1 / (10 + 2) = 10/12
        expected_coeff = 10.0 * 1.0 / (10.0 + 2.0)
        assert diag[0] == pytest.approx(expected_coeff)
        assert source[0] == pytest.approx(expected_coeff * 300.0)
