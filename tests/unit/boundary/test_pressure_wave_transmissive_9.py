"""Tests for v9 enhanced pressure wave transmissive boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_wave_transmissive_9 import PressureWaveTransmissive9BC


class TestPressureWaveTransmissive9BC:
    """Test the pressureWaveTransmissive9 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive9", simple_patch,
            {"fieldInf": 101325.0},
        )
        assert isinstance(bc, PressureWaveTransmissive9BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive9"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.visc_coeff == pytest.approx(0.1)
        assert bc.visc_Re_ref == pytest.approx(1000.0)
        assert bc.mu == pytest.approx(1.81e-5)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch, {
            "fieldInf": 100000.0, "viscCoeff": 0.2, "viscReRef": 500.0, "mu": 1e-5,
        })
        assert bc.field_inf == pytest.approx(100000.0)
        assert bc.visc_coeff == pytest.approx(0.2)
        assert bc.visc_Re_ref == pytest.approx(500.0)

    def test_apply_basic(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, c=343.0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_k_damping(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch, {"damping": 0.1})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, k=k)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_temperature(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch, {"entropyCoeff": 0.05})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, T_ref=300.0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_without_args(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        field = torch.full((20,), 101325.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.isfinite(field[5])

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive9BC(simple_patch)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
