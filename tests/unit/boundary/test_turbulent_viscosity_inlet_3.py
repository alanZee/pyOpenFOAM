"""Tests for v3 enhanced turbulent viscosity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_3 import TurbulentViscosityInlet3BC


class TestTurbulentViscosityInlet3BC:
    """Test the turbulentViscosityInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentViscosityInlet3", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentViscosityInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.nut_min == pytest.approx(1e-10)
        assert bc.nut_max == pytest.approx(1e4)
        assert bc.alpha == pytest.approx(1.0)
        assert bc.nut_ratio_ref == pytest.approx(10.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch, {
            "Cmu": 0.1, "intensity": 0.10, "lengthScale": 0.05,
            "nutMin": 1e-8, "nutMax": 1e3, "alpha": 0.8, "nutRatioRef": 50.0,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.intensity == pytest.approx(0.10)
        assert bc.length_scale == pytest.approx(0.05)
        assert bc.nut_min == pytest.approx(1e-8)
        assert bc.nut_max == pytest.approx(1e3)
        assert bc.alpha == pytest.approx(0.8)
        assert bc.nut_ratio_ref == pytest.approx(50.0)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        expected = 0.09 * 1.0 ** 2 / 0.1
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_ratio_blending(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch, {
            "alpha": 0.5, "nutRatioRef": 10.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5)

        nut_computed = 0.09 * 1.0 / 0.1
        nut_ref = 10.0 * 1e-5
        expected = 0.5 * nut_computed + 0.5 * nut_ref
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(0.001)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.002, 0.002, 0.002], dtype=torch.float64))
