"""Tests for wall function boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, KqRWallFunctionBC, NutkWallFunctionBC


class TestNutkWallFunctionBC:
    """Test the nutkWallFunction boundary condition."""

    def test_registration(self):
        """nutkWallFunction is registered in the RTS registry."""
        assert "nutkWallFunction" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("nutkWallFunction", wall_patch)
        assert isinstance(bc, NutkWallFunctionBC)

    def test_compute_nut_basic(self, wall_patch):
        """compute_nut returns positive values."""
        bc = NutkWallFunctionBC(wall_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert nut.shape == (3,)
        assert (nut > 0).all()

    def test_compute_nut_scales_with_k(self, wall_patch):
        """ν_t increases with turbulent kinetic energy."""
        bc = NutkWallFunctionBC(wall_patch)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        k_low = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        k_high = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        nut_low = bc.compute_nut(k_low, y, nu=1e-5)
        nut_high = bc.compute_nut(k_high, y, nu=1e-5)
        assert (nut_high > nut_low).all()

    def test_compute_nut_scales_with_y(self, wall_patch):
        """ν_t increases with wall distance."""
        bc = NutkWallFunctionBC(wall_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y_near = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
        y_far = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        nut_near = bc.compute_nut(k, y_near, nu=1e-5)
        nut_far = bc.compute_nut(k, y_far, nu=1e-5)
        assert (nut_far > nut_near).all()

    def test_compute_nut_handles_zero_k(self, wall_patch):
        """compute_nut handles zero k without NaN."""
        bc = NutkWallFunctionBC(wall_patch)
        k = torch.zeros(3, dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert not torch.isnan(nut).any()
        assert not torch.isinf(nut).any()

    def test_apply_sets_value(self, wall_patch):
        """apply() sets face values from coefficient."""
        bc = NutkWallFunctionBC(wall_patch, {"value": 0.1})
        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[30:33], torch.full((3,), 0.1, dtype=torch.float64))

    def test_apply_no_value_coeff(self, wall_patch):
        """apply() leaves field unchanged when no value coefficient."""
        bc = NutkWallFunctionBC(wall_patch)
        field = torch.ones(35, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field, original)

    def test_no_matrix_contribution(self, wall_patch):
        """Wall functions have zero matrix contribution."""
        bc = NutkWallFunctionBC(wall_patch)
        field = torch.zeros(35, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_custom_coefficients(self, wall_patch):
        """Custom Cmu, kappa, E are respected."""
        bc = NutkWallFunctionBC(
            wall_patch,
            {"Cmu": 0.08, "kappa": 0.4, "E": 9.0},
        )
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        # Should still produce valid positive values
        assert (nut > 0).all()

    def test_type_name(self, wall_patch):
        """type_name returns the registered name."""
        bc = NutkWallFunctionBC(wall_patch)
        assert bc.type_name == "nutkWallFunction"


class TestKqRWallFunctionBC:
    """Test the kqRWallFunction boundary condition."""

    def test_registration(self):
        """kqRWallFunction is registered in the RTS registry."""
        assert "kqRWallFunction" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("kqRWallFunction", wall_patch)
        assert isinstance(bc, KqRWallFunctionBC)

    def test_compute_k_wall(self, wall_patch):
        """k at wall is computed from friction velocity."""
        bc = KqRWallFunctionBC(wall_patch)
        u_tau = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        k = bc.compute_k_wall(u_tau)
        assert k.shape == (3,)
        assert (k > 0).all()
        # k = u_tau^2 / sqrt(Cmu)
        import math
        expected = u_tau**2 / math.sqrt(0.09)
        assert torch.allclose(k, expected)

    def test_apply_sets_value(self, wall_patch):
        """apply() sets face values from coefficient."""
        bc = KqRWallFunctionBC(wall_patch, {"value": 0.5})
        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[30:33], torch.full((3,), 0.5, dtype=torch.float64))

    def test_no_matrix_contribution(self, wall_patch):
        """Wall functions have zero matrix contribution."""
        bc = KqRWallFunctionBC(wall_patch)
        field = torch.zeros(35, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_type_name(self, wall_patch):
        """type_name returns the registered name."""
        bc = KqRWallFunctionBC(wall_patch)
        assert bc.type_name == "kqRWallFunction"
