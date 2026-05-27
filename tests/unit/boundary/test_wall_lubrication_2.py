"""Tests for Antal wall lubrication boundary condition.

Tests cover AntalWallLubricationBC:
- RTS registration
- Factory creation
- Property access (Cw0, Dp, CwMax)
- Distance-dependent effective coefficient
- apply() and matrix_contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_lubrication_2 import AntalWallLubricationBC


class TestAntalWallLubricationBC:
    """antalWallLubrication boundary condition tests."""

    def test_registration(self):
        """antalWallLubrication is registered in RTS."""
        assert "antalWallLubrication" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = AntalWallLubricationBC(simple_patch)
        assert bc.type_name == "antalWallLubrication"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "antalWallLubrication", simple_patch,
            {"Cw": 0.05, "Dp": 0.003},
        )
        assert isinstance(bc, AntalWallLubricationBC)

    def test_default_coefficients(self, simple_patch):
        bc = AntalWallLubricationBC(simple_patch)
        assert bc.Cw0 == pytest.approx(0.05)
        assert bc.Dp == pytest.approx(0.003)
        assert bc.CwMax == pytest.approx(10.0)

    def test_custom_coefficients(self, simple_patch):
        bc = AntalWallLubricationBC(
            simple_patch,
            {"Cw": 0.1, "Dp": 0.005, "CwMax": 20.0},
        )
        assert bc.Cw0 == pytest.approx(0.1)
        assert bc.Dp == pytest.approx(0.005)
        assert bc.CwMax == pytest.approx(20.0)

    def test_alpha_name_default(self, simple_patch):
        bc = AntalWallLubricationBC(simple_patch)
        assert bc.alpha_name == "alpha.d"

    def test_rho_name_custom(self, simple_patch):
        bc = AntalWallLubricationBC(
            simple_patch, {"alpha": "alpha.air", "rho": "rho.air"},
        )
        assert bc.alpha_name == "alpha.air"
        assert bc.rho_name == "rho.air"

    def test_effective_coefficient_far_from_wall(self, simple_patch):
        """Far from wall: Cw_eff = Cw0 * Dp / y → small value."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        # Large wall distance → small coefficient
        wall_dist = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)

        # Cw_eff = 0.05 * 0.003 / 1.0 = 0.00015
        expected = torch.full((3,), 0.00015, dtype=torch.float64)
        assert torch.allclose(Cw, expected)

    def test_effective_coefficient_near_wall(self, simple_patch):
        """Near wall: Cw_eff is capped by CwMax."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        # Very small wall distance → coefficient is capped
        wall_dist = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)

        # Cw_raw = 0.05 * 0.003 / 1e-6 = 150 → capped at 10
        expected = torch.full((3,), 10.0, dtype=torch.float64)
        assert torch.allclose(Cw, expected)

    def test_effective_coefficient_intermediate(self, simple_patch):
        """Intermediate distance: Cw_eff = Cw0 * Dp / y (not capped)."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        # y = 0.01 → Cw = 0.05 * 0.003 / 0.01 = 0.015
        wall_dist = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)

        expected = torch.full((3,), 0.015, dtype=torch.float64)
        assert torch.allclose(Cw, expected)

    def test_apply_sets_zero_gradient(self, simple_patch):
        """apply() sets BC face values to owner cell values."""
        bc = AntalWallLubricationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        # Set owner cell values
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        field = bc.apply(field)

        # Owner cells [0, 1, 2] → faces [10, 11, 12]
        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = AntalWallLubricationBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 50.0
        field[1] = 60.0
        field[2] = 70.0

        field = bc.apply(field, patch_idx=5)

        assert field[5] == pytest.approx(50.0)
        assert field[6] == pytest.approx(60.0)
        assert field[7] == pytest.approx(70.0)

    def test_matrix_contributions_default(self, simple_patch):
        """Matrix contributions with default alpha and rho."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # diag = 0 (wall lubrication does not add diagonal)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        # source > 0 (repulsive force)
        assert (source > 0).all()

    def test_matrix_contributions_with_alpha_and_rho(self, simple_patch):
        """Matrix contributions with explicit alpha and rho."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(
            field, n_cells, alpha=0.3, rho=900.0,
        )

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert (source > 0).all()

    def test_matrix_contributions_tensor_alpha(self, simple_patch):
        """Matrix contributions with per-cell tensor alpha."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, alpha=alpha)

        assert source.shape == (n_cells,)
        assert (source > 0).all()

    def test_distance_dependence_is_monotone(self, simple_patch):
        """Closer to wall → larger coefficient → larger force."""
        bc = AntalWallLubricationBC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 100.0},
        )

        far = torch.tensor([0.1], dtype=torch.float64)
        mid = torch.tensor([0.01], dtype=torch.float64)
        near = torch.tensor([0.001], dtype=torch.float64)

        Cw_far = bc.effective_coefficient(far)
        Cw_mid = bc.effective_coefficient(mid)
        Cw_near = bc.effective_coefficient(near)

        assert Cw_near > Cw_mid > Cw_far
