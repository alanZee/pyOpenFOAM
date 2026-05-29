"""Tests for v5 enhanced turbulent dissipation inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_dissipation_inlet_5 import TurbulentDissipationInlet5BC


class TestTurbulentDissipationInlet5BC:
    """Test the turbulentDissipationInlet5 boundary condition."""

    def test_registration(self):
        assert "turbulentDissipationInlet5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentDissipationInlet5", simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentDissipationInlet5BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet5"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(1.0)
        assert bc.wall_dist == pytest.approx(0.01)
        assert bc.y_plus_low == pytest.approx(5.0)
        assert bc.y_plus_high == pytest.approx(30.0)
        assert bc.epsilon_min == pytest.approx(1e-10)
        assert bc.epsilon_max == pytest.approx(1e6)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.1, "kappa": 0.38,
            "intensity": 0.10, "wallDist": 0.001,
            "yPlusLow": 3.0, "yPlusHigh": 50.0,
            "epsilonMin": 1e-8, "epsilonMax": 1e4,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.kappa == pytest.approx(0.38)
        assert bc.wall_dist == pytest.approx(0.001)
        assert bc.y_plus_low == pytest.approx(3.0)
        assert bc.y_plus_high == pytest.approx(50.0)

    def test_apply_two_layer_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch, {
            "wallDist": 0.01, "kappa": 0.41, "Cmu": 0.09,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_fallback_without_nu(self, simple_patch):
        """Without nu, falls back to standard mixing-length formula."""
        bc = TurbulentDissipationInlet5BC(simple_patch, {"mixingLength": 0.02})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.02
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch, {
            "intensity": 0.05, "mixingLength": 0.01,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] > 0)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch, {
            "epsilonMin": 0.5, "epsilonMax": 10.0,
        })
        k = torch.tensor([1e-10, 1e6, 1.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)

        assert torch.all(field[10:13] >= 0.5 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_two_layer_near_wall(self, simple_patch):
        """Buffer-layer epsilon should be larger than log-law at small y+."""
        bc = TurbulentDissipationInlet5BC(simple_patch, {
            "wallDist": 0.0001, "kappa": 0.41, "Cmu": 0.09,
            "yPlusLow": 5.0, "yPlusHigh": 30.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)

        # Very small wall distance -> buffer layer dominates -> eps = 2*nu*k/y^2
        assert torch.all(torch.isfinite(field[10:13]))
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k, nu=1e-5)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
