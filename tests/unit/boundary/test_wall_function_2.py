"""Tests for EnhancedWallFunctionBC — enhanced wall function with y+ switching."""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_function_2 import EnhancedWallFunctionBC


class TestEnhancedWallFunctionBC:
    """Test the enhanced wall function boundary condition."""

    def test_registration(self):
        """enhancedWallFunction is registered in the RTS registry."""
        assert "enhancedWallFunction" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("enhancedWallFunction", wall_patch)
        assert isinstance(bc, EnhancedWallFunctionBC)

    def test_type_name(self, wall_patch):
        """type_name returns the registered name."""
        bc = EnhancedWallFunctionBC(wall_patch)
        assert bc.type_name == "enhancedWallFunction"

    def test_default_coefficients(self, wall_patch):
        """Default coefficients match OpenFOAM values."""
        bc = EnhancedWallFunctionBC(wall_patch)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.E == pytest.approx(9.8)

    def test_custom_coefficients(self, wall_patch):
        """Custom coefficients override defaults."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.38, "E": 8.5})
        assert bc.kappa == pytest.approx(0.38)
        assert bc.E == pytest.approx(8.5)

    def test_yplus_transition(self, wall_patch):
        """Transition y+ is at 11.225."""
        bc = EnhancedWallFunctionBC(wall_patch)
        assert bc.yplus_transition == pytest.approx(11.225)

    def test_compute_nut_sublayer(self, wall_patch):
        """In the viscous sublayer (y+ < 11.225), nut = 0."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8, "Cmu": 0.09})
        # Very small k and y to get y+ < 11.225
        k = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
        y = torch.tensor([1e-5, 1e-5, 1e-5], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert torch.allclose(nut, torch.zeros_like(nut), atol=1e-10)

    def test_compute_nut_loglaw(self, wall_patch):
        """In the log-law region (y+ > 11.225), nut > 0."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8, "Cmu": 0.09})
        # Larger k and y to get y+ > 11.225
        k = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert (nut > 0).all()

    def test_compute_nut_mixed(self, wall_patch):
        """Mixed y+ regime: sublayer cells get nut=0, log-law cells get nut>0."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8, "Cmu": 0.09})
        # Two cells: one sublayer, one log-law
        k = torch.tensor([0.001, 10.0, 10.0], dtype=torch.float64)
        y = torch.tensor([1e-5, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert nut[0].item() == pytest.approx(0.0, abs=1e-10)
        assert nut[1].item() > 0.0

    def test_compute_u_plus_sublayer(self, wall_patch):
        """In the sublayer, u+ = y+."""
        bc = EnhancedWallFunctionBC(wall_patch)
        y_plus = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64)
        u_plus = bc.compute_u_plus(y_plus)
        # Should match y_plus for sublayer values
        assert torch.allclose(u_plus, y_plus, atol=0.1)

    def test_compute_u_plus_loglaw(self, wall_patch):
        """In the log-law region, u+ follows (1/kappa) * ln(E * y+)."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8})
        y_plus = torch.tensor([30.0, 100.0, 300.0], dtype=torch.float64)
        u_plus = bc.compute_u_plus(y_plus)
        # Expected: (1/0.41) * ln(9.8 * y+)
        expected = torch.log(9.8 * y_plus) / 0.41
        assert torch.allclose(u_plus, expected, atol=0.5)

    def test_compute_u_plus_transition(self, wall_patch):
        """u+ is continuous at the transition y+."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8})
        # Just below and just above transition
        y_below = torch.tensor([11.0], dtype=torch.float64)
        y_above = torch.tensor([11.5], dtype=torch.float64)
        u_below = bc.compute_u_plus(y_below)
        u_above = bc.compute_u_plus(y_above)
        # Should be close (within 10%)
        assert abs(u_below.item() - u_above.item()) / max(u_below.item(), 1e-10) < 0.1

    def test_spalding_u_plus_sublayer(self, wall_patch):
        """Spalding's law in sublayer returns u+ ~ y+."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8})
        y_plus = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        u_plus = bc.spalding_u_plus(y_plus)
        # Should be close to y_plus for small values
        assert torch.allclose(u_plus, y_plus, atol=1.0)

    def test_spalding_u_plus_loglaw(self, wall_patch):
        """Spalding's law in log-law matches analytical formula."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8})
        y_plus = torch.tensor([100.0], dtype=torch.float64)
        u_plus = bc.spalding_u_plus(y_plus)
        expected = math.log(9.8 * 100.0) / 0.41
        # Spalding's law is an implicit relation; allow larger tolerance
        assert abs(u_plus.item() - expected) < 5.0

    def test_apply_sets_value(self, wall_patch):
        """apply() sets face values when value coefficient is provided."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"value": 0.5})
        field = torch.zeros(40, dtype=torch.float64)
        bc.apply(field)
        # Wall patch has 3 faces at indices [30, 31, 32]
        assert field[30].item() == pytest.approx(0.5)
        assert field[31].item() == pytest.approx(0.5)
        assert field[32].item() == pytest.approx(0.5)

    def test_apply_no_value(self, wall_patch):
        """Without value coefficient, field is unchanged."""
        bc = EnhancedWallFunctionBC(wall_patch)
        field = torch.arange(40, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field, original)

    def test_apply_with_patch_idx(self, wall_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"value": 3.14})
        field = torch.zeros(50, dtype=torch.float64)
        bc.apply(field, patch_idx=10)
        assert field[10].item() == pytest.approx(3.14)
        assert field[11].item() == pytest.approx(3.14)
        assert field[12].item() == pytest.approx(3.14)

    def test_matrix_contributions_zero(self, wall_patch):
        """Wall functions have zero matrix contribution."""
        bc = EnhancedWallFunctionBC(wall_patch)
        field = torch.zeros(40, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_repr(self, wall_patch):
        """__repr__ includes patch name and constants."""
        bc = EnhancedWallFunctionBC(wall_patch, coeffs={"kappa": 0.41, "E": 9.8})
        r = repr(bc)
        assert "wall" in r
        assert "0.41" in r
