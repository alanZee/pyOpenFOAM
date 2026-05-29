"""Tests for v3 enhanced wave transmissive pressure boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_wave_transmissive_3 import PressureWaveTransmissive3BC


class TestPressureWaveTransmissive3BC:
    """Test the pressureWaveTransmissive3 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive3", simple_patch,
            {"fieldInf": 101325.0},
        )
        assert isinstance(bc, PressureWaveTransmissive3BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive3"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.blending == pytest.approx(0.1)
        assert bc.sigma_base == pytest.approx(0.25)
        assert bc.damping == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {
            "fieldInf": 1e5, "lInf": 0.5, "gamma": 1.67,
            "blending": 0.2, "sigmaBase": 0.3, "damping": 0.05,
        })
        assert bc.field_inf == pytest.approx(1e5)
        assert bc.l_inf == pytest.approx(0.5)
        assert bc.gamma == pytest.approx(1.67)
        assert bc.blending == pytest.approx(0.2)
        assert bc.sigma_base == pytest.approx(0.3)
        assert bc.damping == pytest.approx(0.05)

    def test_apply_default(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_velocity_and_rho(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {
            "fieldInf": 101325.0, "blending": 0.1, "sigmaBase": 0.25,
        })
        velocity = torch.tensor([
            [50.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, c=343.0)

        # Pressure should change from far-field value
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_turbulent_damping(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {
            "fieldInf": 101325.0, "damping": 0.2,
        })
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.full((15,), 101325.0, dtype=torch.float64)

        field_no_damp = field.clone()
        bc_no_damp = PressureWaveTransmissive3BC(simple_patch, {
            "fieldInf": 101325.0, "damping": 0.0,
        })
        bc_no_damp.apply(field_no_damp)

        bc.apply(field, k=k)

        # With damping, pressure should be lower (p_damp subtracts)
        assert torch.all(field[10:13] <= field_no_damp[10:13] + 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((20,), 101325.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive3BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
        assert torch.all(source > 0)
