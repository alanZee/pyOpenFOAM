"""Tests for enhanced Antal wall lubrication boundary condition (version 2).

Tests cover AntalWallLubrication2BC:
- RTS registration
- Factory creation
- Property access (Cw0, Dp, CwMax, exponent, damping)
- Distance-dependent effective coefficient with exponent
- Interface damping factor
- apply() and matrix_contributions
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_lubrication_antal import AntalWallLubrication2BC


class TestAntalWallLubrication2BC:
    """antalWallLubrication2 boundary condition tests."""

    def test_registration(self):
        """antalWallLubrication2 is registered in RTS."""
        assert "antalWallLubrication2" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = AntalWallLubrication2BC(simple_patch)
        assert bc.type_name == "antalWallLubrication2"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "antalWallLubrication2", simple_patch,
            {"Cw": 0.05, "Dp": 0.003},
        )
        assert isinstance(bc, AntalWallLubrication2BC)

    def test_default_coefficients(self, simple_patch):
        bc = AntalWallLubrication2BC(simple_patch)
        assert bc.Cw0 == pytest.approx(0.05)
        assert bc.Dp == pytest.approx(0.003)
        assert bc.CwMax == pytest.approx(10.0)
        assert bc.exponent == pytest.approx(1.0)
        assert bc.damping == pytest.approx(0.0)

    def test_custom_coefficients(self, simple_patch):
        bc = AntalWallLubrication2BC(
            simple_patch,
            {"Cw": 0.1, "Dp": 0.005, "CwMax": 20.0, "exponent": 1.5, "damping": 0.8},
        )
        assert bc.Cw0 == pytest.approx(0.1)
        assert bc.Dp == pytest.approx(0.005)
        assert bc.CwMax == pytest.approx(20.0)
        assert bc.exponent == pytest.approx(1.5)
        assert bc.damping == pytest.approx(0.8)

    def test_alpha_rho_name_defaults(self, simple_patch):
        bc = AntalWallLubrication2BC(simple_patch)
        assert bc.alpha_name == "alpha.d"
        assert bc.rho_name == "rho.d"

    def test_alpha_rho_name_custom(self, simple_patch):
        bc = AntalWallLubrication2BC(
            simple_patch, {"alpha": "alpha.air", "rho": "rho.air"},
        )
        assert bc.alpha_name == "alpha.air"
        assert bc.rho_name == "rho.air"

    def test_effective_coefficient_far_from_wall(self, simple_patch):
        """Far from wall: Cw_eff = Cw0 * Dp^exp / y → small value."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "exponent": 1.0},
        )
        wall_dist = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)
        expected = torch.full((3,), 0.00015, dtype=torch.float64)
        assert torch.allclose(Cw, expected)

    def test_effective_coefficient_near_wall_capped(self, simple_patch):
        """Near wall: Cw_eff is capped by CwMax."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        wall_dist = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)
        expected = torch.full((3,), 10.0, dtype=torch.float64)
        assert torch.allclose(Cw, expected)

    def test_effective_coefficient_with_exponent(self, simple_patch):
        """Exponent modifies the Dp power: Dp^1.5 instead of Dp."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 100.0, "exponent": 1.5},
        )
        wall_dist = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)
        # Dp^1.5 = 0.003^1.5 ≈ 0.0001643
        # Cw_eff = 0.05 * 0.0001643 / 0.01 ≈ 0.0008216
        expected_val = 0.05 * (0.003 ** 1.5) / 0.01
        expected = torch.full((3,), expected_val, dtype=torch.float64)
        assert torch.allclose(Cw, expected, rtol=1e-6)

    def test_effective_coefficient_exponent_1_equals_standard(self, simple_patch):
        """exponent=1.0 matches standard Antal model."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "exponent": 1.0},
        )
        wall_dist = torch.tensor([0.01], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist)
        expected = 0.05 * 0.003 / 0.01  # = 0.015
        assert Cw[0].item() == pytest.approx(expected)

    def test_interface_damping_pure_phases(self, simple_patch):
        """Pure phases (alpha=0 or 1) have no damping."""
        bc = AntalWallLubrication2BC(simple_patch, {"damping": 0.9})
        alpha = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        f = bc.interface_damping_factor(alpha)
        assert torch.allclose(f, torch.ones(4, dtype=torch.float64), atol=1e-10)

    def test_interface_damping_at_interface(self, simple_patch):
        """alpha=0.5 → f_d = 1 - damping * 1.0."""
        bc = AntalWallLubrication2BC(simple_patch, {"damping": 0.8})
        alpha = torch.tensor([0.5], dtype=torch.float64)
        f = bc.interface_damping_factor(alpha)
        # f = 1 - 0.8 * 4 * 0.5 * 0.5 = 1 - 0.8 = 0.2
        assert f[0].item() == pytest.approx(0.2)

    def test_interface_damping_zero_strength(self, simple_patch):
        """damping=0 → no effect regardless of alpha."""
        bc = AntalWallLubrication2BC(simple_patch, {"damping": 0.0})
        alpha = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        f = bc.interface_damping_factor(alpha)
        assert torch.allclose(f, torch.ones(3, dtype=torch.float64))

    def test_effective_coefficient_with_interface_damping(self, simple_patch):
        """Interface damping reduces coefficient near alpha=0.5."""
        bc = AntalWallLubrication2BC(
            simple_patch,
            {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "damping": 0.9},
        )
        wall_dist = torch.tensor([0.01, 0.01], dtype=torch.float64)
        alpha = torch.tensor([0.0, 0.5], dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist, alpha)
        # alpha=0: no damping → Cw = 0.015
        # alpha=0.5: f_d = 1 - 0.9 = 0.1 → Cw = 0.015 * 0.1 = 0.0015
        assert Cw[0].item() == pytest.approx(0.015)
        assert Cw[1].item() == pytest.approx(0.0015, rel=1e-6)

    def test_apply_sets_zero_gradient(self, simple_patch):
        """apply() sets BC face values to owner cell values."""
        bc = AntalWallLubrication2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        field = bc.apply(field)
        assert field[10] == pytest.approx(100.0)
        assert field[11] == pytest.approx(200.0)
        assert field[12] == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = AntalWallLubrication2BC(simple_patch)
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
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert (source > 0).all()

    def test_matrix_contributions_with_alpha_and_rho(self, simple_patch):
        bc = AntalWallLubrication2BC(
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

    def test_matrix_contributions_with_damping_reduces_force(self, simple_patch):
        """Interface damping reduces the matrix source near interface."""
        bc_no_damp = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "damping": 0.0},
        )
        bc_damp = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "damping": 0.9},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        _, source_no_damp = bc_no_damp.matrix_contributions(field, n_cells, alpha=alpha)
        _, source_damp = bc_damp.matrix_contributions(field, n_cells, alpha=alpha)
        assert (source_damp < source_no_damp).all()

    def test_distance_dependence_is_monotone(self, simple_patch):
        """Closer to wall → larger coefficient (before cap)."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 100.0},
        )
        far = torch.tensor([0.1], dtype=torch.float64)
        mid = torch.tensor([0.01], dtype=torch.float64)
        near = torch.tensor([0.001], dtype=torch.float64)
        Cw_far = bc.effective_coefficient(far)
        Cw_mid = bc.effective_coefficient(mid)
        Cw_near = bc.effective_coefficient(near)
        assert Cw_near > Cw_mid > Cw_far

    def test_batch_processing(self, simple_patch):
        """Batch processing with many faces."""
        bc = AntalWallLubrication2BC(
            simple_patch, {"Cw": 0.05, "Dp": 0.003, "CwMax": 10.0, "damping": 0.5},
        )
        n = 50
        wall_dist = torch.rand(n, dtype=torch.float64) * 0.1 + 0.001
        alpha = torch.rand(n, dtype=torch.float64)
        Cw = bc.effective_coefficient(wall_dist, alpha)
        assert Cw.shape == (n,)
        assert torch.isfinite(Cw).all()
        assert (Cw >= 0).all()
