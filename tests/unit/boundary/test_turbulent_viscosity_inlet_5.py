"""Tests for v5 enhanced turbulent viscosity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_5 import TurbulentViscosityInlet5BC


class TestTurbulentViscosityInlet5BC:
    """Test the turbulentViscosityInlet5 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentViscosityInlet5", simple_patch,
            {"Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentViscosityInlet5BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet5"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.nut_min == pytest.approx(1e-10)
        assert bc.nut_max == pytest.approx(1e4)
        assert bc.alpha == pytest.approx(0.9)
        assert bc.nut_ratio_ref == pytest.approx(10.0)
        assert bc.wall_dist == pytest.approx(0.01)
        assert bc.y_plus_low == pytest.approx(5.0)
        assert bc.y_plus_high == pytest.approx(30.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch, {
            "Cmu": 0.1, "intensity": 0.10, "lengthScale": 0.05,
            "kappa": 0.38, "wallDist": 0.001,
            "yPlusLow": 3.0, "yPlusHigh": 50.0,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.kappa == pytest.approx(0.38)
        assert bc.wall_dist == pytest.approx(0.001)
        assert bc.y_plus_low == pytest.approx(3.0)
        assert bc.y_plus_high == pytest.approx(50.0)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] > 0)

    def test_apply_with_wall_blending(self, simple_patch):
        """Wall-distance blending should affect nut values."""
        bc = TurbulentViscosityInlet5BC(simple_patch, {
            "wallDist": 0.01, "kappa": 0.41,
            "yPlusLow": 5.0, "yPlusHigh": 30.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch, {
            "nutMin": 0.01, "nutMax": 10.0,
        })
        k = torch.tensor([1e-10, 1e6, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        assert torch.all(field[10:13] >= 0.01 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k, epsilon=epsilon)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.002, 0.002, 0.002], dtype=torch.float64))
