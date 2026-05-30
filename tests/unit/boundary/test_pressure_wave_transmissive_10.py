"""Tests for v10 enhanced pressure wave transmissive boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_wave_transmissive_10 import PressureWaveTransmissive10BC


class TestPressureWaveTransmissive10BC:
    """Test the pressureWaveTransmissive10 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive10", simple_patch,
            {"fieldInf": 101325.0},
        )
        assert isinstance(bc, PressureWaveTransmissive10BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive10"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.imp_coeff == pytest.approx(0.1)
        assert bc.grad_damp_coeff == pytest.approx(0.05)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch, {
            "impCoeff": 0.2, "gradDampCoeff": 0.1, "fieldInf": 200000.0,
        })
        assert bc.imp_coeff == pytest.approx(0.2)
        assert bc.grad_damp_coeff == pytest.approx(0.1)
        assert bc.field_inf == pytest.approx(200000.0)

    def test_apply_basic(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, c=343.0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_k_damping(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, c=343.0, k=k)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch, {
            "fieldInf": 101325.0, "gradDampCoeff": 0.1,
        })
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        pg = torch.tensor([100.0, -50.0, 200.0], dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, pressure_gradient=pg)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch)
        field = torch.full((20,), 101325.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive10BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
