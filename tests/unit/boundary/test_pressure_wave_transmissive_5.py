"""Tests for v5 enhanced pressure wave transmissive boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_wave_transmissive_5 import PressureWaveTransmissive5BC


class TestPressureWaveTransmissive5BC:
    """Test the pressureWaveTransmissive5 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive5", simple_patch,
            {"fieldInf": 101325.0},
        )
        assert isinstance(bc, PressureWaveTransmissive5BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive5"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.blending == pytest.approx(0.1)
        assert bc.sigma_base == pytest.approx(0.25)
        assert bc.damping == pytest.approx(0.1)
        assert bc.R_specific == pytest.approx(287.05)
        assert bc.Cp == pytest.approx(1005.0)
        assert bc.f_cutoff == pytest.approx(1000.0)
        assert bc.beta_sigma == pytest.approx(0.01)
        assert bc.Re_t_ref == pytest.approx(100.0)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {
            "fieldInf": 100000.0, "lInf": 2.0, "gamma": 1.3,
            "blending": 0.2, "sigmaBase": 0.3, "damping": 0.2,
            "betaSigma": 0.05, "ReTRef": 50.0,
        })
        assert bc.field_inf == pytest.approx(100000.0)
        assert bc.l_inf == pytest.approx(2.0)
        assert bc.gamma == pytest.approx(1.3)
        assert bc.beta_sigma == pytest.approx(0.05)
        assert bc.Re_t_ref == pytest.approx(50.0)

    def test_apply_basic(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 102000.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(torch.isfinite(field[10:13]))
        # Pressure should be modified from initial value
        assert not torch.allclose(field[10:13], torch.full((3,), 102000.0, dtype=torch.float64))

    def test_apply_with_k_damping(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {
            "fieldInf": 101325.0, "damping": 0.5,
        })
        field = torch.full((15,), 102000.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field_no_k = field.clone()
        field_with_k = field.clone()

        bc.apply(field_no_k, velocity=velocity)
        bc.apply(field_with_k, velocity=velocity, k=k)

        # With k damping, pressure should be lower
        assert torch.all(field_with_k[10:13] <= field_no_k[10:13])

    def test_apply_with_temperature_correction(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 102000.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        bc.apply(field, velocity=velocity, T_ref=600.0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_nu_adaptive_sigma(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {
            "fieldInf": 101325.0, "betaSigma": 0.05,
        })
        field = torch.full((15,), 102000.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        bc.apply(field, velocity=velocity, k=k, nu=1e-5)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((20,), 102000.0, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5] != 102000.0

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive5BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
        assert torch.all(source > 0)
