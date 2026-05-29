"""Tests for enhanced wave transmissive pressure boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_wave_transmissive_2 import PressureWaveTransmissive2BC


class TestPressureWaveTransmissive2BC:
    """Test the pressureWaveTransmissive2 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive2", simple_patch,
            {"fieldInf": 101325.0, "blending": 0.05},
        )
        assert isinstance(bc, PressureWaveTransmissive2BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive2"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.blending == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch, {
            "fieldInf": 100000.0,
            "lInf": 2.0,
            "gamma": 1.67,
            "blending": 0.2,
        })
        assert bc.field_inf == pytest.approx(100000.0)
        assert bc.l_inf == pytest.approx(2.0)
        assert bc.gamma == pytest.approx(1.67)
        assert bc.blending == pytest.approx(0.2)

    def test_apply_basic(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch, {
            "fieldInf": 101325.0,
            "blending": 0.0,
        })

        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        bc.apply(field, velocity=velocity, rho=1.225, c=343.0)

        # Should produce values (even if same as far field with zero dp)
        assert field.shape == (15,)

    def test_apply_with_blending(self, simple_patch):
        """Blending should pull pressure toward far-field value."""
        bc = PressureWaveTransmissive2BC(simple_patch, {
            "fieldInf": 101325.0,
            "blending": 0.5,
        })

        field = torch.full((15,), 105000.0, dtype=torch.float64)
        velocity = torch.zeros((3, 3), dtype=torch.float64)

        bc.apply(field, velocity=velocity)

        # With non-zero dp and blending, value should be modified
        assert field[10] != 105000.0

    def test_apply_with_tensor_rho(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        rho = torch.full((3,), 1.225, dtype=torch.float64)
        velocity = torch.zeros((3, 3), dtype=torch.float64)

        bc.apply(field, velocity=velocity, rho=rho)
        assert field.shape == (15,)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch)
        field = torch.full((20,), 101325.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field.shape == (20,)

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Diag should be positive (relaxation + blending)
        assert torch.all(diag > 0)
