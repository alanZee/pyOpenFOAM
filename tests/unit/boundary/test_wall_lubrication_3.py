"""Tests for Tomiyama wall lubrication BC.

Tests cover TomiyamaWallLubricationBC:
- RTS registration
- Factory creation
- Property access
- Eötvös number and f(Eo) correlation
- Distance-dependent effective coefficient
- apply() and matrix_contributions
"""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_lubrication_3 import TomiyamaWallLubricationBC


class TestTomiyamaWallLubricationBC:
    """tomiyamaWallLubrication boundary condition tests."""

    def test_registration(self):
        """tomiyamaWallLubrication is registered in RTS."""
        assert "tomiyamaWallLubrication" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.type_name == "tomiyamaWallLubrication"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "tomiyamaWallLubrication", simple_patch,
            {"Cw": 0.05, "Dp": 0.003, "sigma": 0.072},
        )
        assert isinstance(bc, TomiyamaWallLubricationBC)

    def test_default_coefficients(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.Cw0 == pytest.approx(0.05)
        assert bc.Dp == pytest.approx(0.003)
        assert bc.sigma == pytest.approx(0.072)
        assert bc.rho_c == pytest.approx(1000.0)
        assert bc.rho_d == pytest.approx(1.225)

    def test_custom_coefficients(self, simple_patch):
        bc = TomiyamaWallLubricationBC(
            simple_patch,
            {"Cw": 0.1, "Dp": 0.005, "sigma": 0.05, "rho_c": 900.0, "rho_d": 10.0},
        )
        assert bc.Cw0 == pytest.approx(0.1)
        assert bc.Dp == pytest.approx(0.005)
        assert bc.sigma == pytest.approx(0.05)
        assert bc.rho_c == pytest.approx(900.0)
        assert bc.rho_d == pytest.approx(10.0)

    def test_alpha_name_default(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.alpha_name == "alpha.d"

    def test_eotvos_number(self, simple_patch):
        """Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        bc = TomiyamaWallLubricationBC(
            simple_patch, {"Dp": 0.003, "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        Eo = bc.eotvos_number()
        expected = 9.81 * abs(1000.0 - 1.225) * 0.003 ** 2 / 0.072
        assert Eo == pytest.approx(expected, rel=1e-6)

    def test_f_eotvos_small_eo(self, simple_patch):
        """f(Eo) = 0 for Eo < 1."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.f_eotvos(0.5) == pytest.approx(0.0)
        assert bc.f_eotvos(0.0) == pytest.approx(0.0)

    def test_f_eotvos_moderate_eo(self, simple_patch):
        """f(Eo) in range [1, 5)."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        f = bc.f_eotvos(3.0)
        expected = 0.474 * (1.0 - math.exp(-0.0183 * 3.0)) * math.exp(1.48 * 3.0)
        assert f == pytest.approx(expected, rel=1e-6)

    def test_f_eotvos_intermediate_eo(self, simple_patch):
        """f(Eo) in range [5, 33)."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.f_eotvos(10.0) == pytest.approx(0.0219 * 10.0, rel=1e-6)
        assert bc.f_eotvos(20.0) == pytest.approx(0.0219 * 20.0, rel=1e-6)

    def test_f_eotvos_large_eo(self, simple_patch):
        """f(Eo) = 0.474 for Eo >= 33."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        assert bc.f_eotvos(33.0) == pytest.approx(0.474)
        assert bc.f_eotvos(100.0) == pytest.approx(0.474)

    def test_effective_coefficient(self, simple_patch):
        """C_w_eff = Cw0 * f(Eo) / Dp."""
        bc = TomiyamaWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        wall_dist = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)

        Eo = bc.eotvos_number()
        f_eo = bc.f_eotvos(Eo)
        expected = 0.05 * f_eo / 0.003
        assert torch.allclose(Cw, torch.full((3,), expected, dtype=torch.float64))

    def test_apply_sets_zero_gradient(self, simple_patch):
        """apply() sets BC face values to owner cell values."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        field = bc.apply(field)

        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 50.0
        field[1] = 60.0
        field[2] = 70.0

        field = bc.apply(field, patch_idx=5)

        assert field[5] == pytest.approx(50.0)
        assert field[6] == pytest.approx(60.0)
        assert field[7] == pytest.approx(70.0)

    def test_matrix_contributions_shape(self, simple_patch):
        """Matrix contributions have correct shape."""
        bc = TomiyamaWallLubricationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert (source >= 0).all()

    def test_matrix_contributions_with_alpha(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, alpha=0.3)
        assert source.shape == (n_cells,)

    def test_matrix_contributions_tensor_alpha(self, simple_patch):
        bc = TomiyamaWallLubricationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, alpha=alpha)
        assert source.shape == (n_cells,)

    def test_matrix_contributions_nonzero_force(self, simple_patch):
        """With air-water densities, force should be nonzero (Eo > 1)."""
        bc = TomiyamaWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "rho_c": 1000.0, "rho_d": 1.225, "sigma": 0.072},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        _, source = bc.matrix_contributions(field, n_cells, alpha=0.1)
        # Eo >> 1 for air-water at 3mm → f(Eo) > 0 → force > 0
        assert (source > 0).all()
