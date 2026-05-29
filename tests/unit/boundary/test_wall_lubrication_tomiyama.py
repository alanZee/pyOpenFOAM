"""Tests for enhanced Tomiyama wall lubrication BC (version 2).

Tests cover:
- RTS registration
- Factory creation
- Property access
- Eotvos number and f(Eo) correlation
- Combined Eo-dependent + distance-dependent coefficient
- apply() and matrix_contributions
"""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_lubrication_tomiyama import TomiyamaWallLubrication2BC


class TestTomiyamaWallLubrication2BC:
    """tomiyamaWallLubrication2 boundary condition tests."""

    def test_registration(self):
        """tomiyamaWallLubrication2 is registered in RTS."""
        assert "tomiyamaWallLubrication2" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        assert bc.type_name == "tomiyamaWallLubrication2"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "tomiyamaWallLubrication2", simple_patch,
            {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "sigma": 0.072},
        )
        assert isinstance(bc, TomiyamaWallLubrication2BC)

    def test_default_coefficients(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        assert bc.Cw0 == pytest.approx(0.05)
        assert bc.Dp == pytest.approx(0.003)
        assert bc.CwMax == pytest.approx(10.0)
        assert bc.sigma == pytest.approx(0.072)
        assert bc.rho_c == pytest.approx(1000.0)
        assert bc.rho_d == pytest.approx(1.225)

    def test_custom_coefficients(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(
            simple_patch,
            {"Cw": 0.1, "Dp": 0.005, "CwMax": 5.0, "sigma": 0.05,
             "rho_c": 900.0, "rho_d": 10.0},
        )
        assert bc.Cw0 == pytest.approx(0.1)
        assert bc.Dp == pytest.approx(0.005)
        assert bc.CwMax == pytest.approx(5.0)
        assert bc.sigma == pytest.approx(0.05)
        assert bc.rho_c == pytest.approx(900.0)
        assert bc.rho_d == pytest.approx(10.0)

    def test_eotvos_number(self, simple_patch):
        """Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        bc = TomiyamaWallLubrication2BC(
            simple_patch, {"Dp": 0.003, "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        Eo = bc.eotvos_number()
        expected = 9.81 * abs(1000.0 - 1.225) * 0.003 ** 2 / 0.072
        assert Eo == pytest.approx(expected, rel=1e-6)

    def test_f_eotvos_small_eo(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        assert bc.f_eotvos(0.5) == pytest.approx(0.0)

    def test_f_eotvos_moderate_eo(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        f = bc.f_eotvos(3.0)
        expected = 0.474 * (1.0 - math.exp(-0.0183 * 3.0)) * math.exp(1.48 * 3.0)
        assert f == pytest.approx(expected, rel=1e-6)

    def test_f_eotvos_large_eo(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        assert bc.f_eotvos(33.0) == pytest.approx(0.474)

    def test_effective_coefficient_distance_dependent(self, simple_patch):
        """C_w_eff depends on wall distance (unlike version 1)."""
        bc = TomiyamaWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0,
                           "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        # Wall distance: closer to wall → larger coefficient
        dist_close = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
        dist_far = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        Cw_close = bc.effective_coefficient(dist_close)
        Cw_far = bc.effective_coefficient(dist_far)

        assert (Cw_close > Cw_far).all()

    def test_effective_coefficient_capped(self, simple_patch):
        """C_w_eff is capped at CwMax."""
        bc = TomiyamaWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 1.0,
                           "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        # Very small distance → large raw coefficient → should be capped
        dist = torch.tensor([1e-10, 1e-10, 1e-10], dtype=torch.float64)
        Cw = bc.effective_coefficient(dist)
        assert (Cw <= 1.0 + 1e-10).all()

    def test_apply_sets_zero_gradient(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        field = bc.apply(field)
        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 50.0
        field = bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(50.0)

    def test_matrix_contributions_shape(self, simple_patch):
        bc = TomiyamaWallLubrication2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)

    def test_matrix_contributions_nonzero_force(self, simple_patch):
        """With air-water densities, force should be nonzero."""
        bc = TomiyamaWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003,
                           "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        field = torch.zeros(15, dtype=torch.float64)
        _, source = bc.matrix_contributions(field, 3, alpha=0.1)
        assert (source > 0).all()
