"""Tests for wall lubrication model ABC hierarchy.

Tests cover:
- WallLubricationModel ABC RTS registry
- AntalWallLubrication model
- TomiyamaWallLubrication model
- Factory creation and unknown model errors
- Force computation (direction, magnitude, monotonicity)
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.multiphase.wall_lubrication_models import (
    WallLubricationModel,
    AntalWallLubrication,
    TomiyamaWallLubrication,
)


# ============================================================================
# WallLubricationModel ABC
# ============================================================================


class TestWallLubricationModelABC:
    """WallLubricationModel abstract base class tests."""

    def test_rts_registry_contains_models(self):
        """Antal and Tomiyama models are registered."""
        types = WallLubricationModel.available_types()
        assert "antal" in types
        assert "tomiyama" in types

    def test_factory_create_antal(self):
        model = WallLubricationModel.create(
            "antal", d=1e-3, rho_c=998.0, rho_d=1.225,
        )
        assert isinstance(model, AntalWallLubrication)

    def test_factory_create_tomiyama(self):
        model = WallLubricationModel.create(
            "tomiyama", d=1e-3, rho_c=998.0, rho_d=1.225,
        )
        assert isinstance(model, TomiyamaWallLubrication)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown wall lubrication model"):
            WallLubricationModel.create(
                "nonexistent", d=1e-3, rho_c=998.0, rho_d=1.225,
            )

    def test_is_abstract(self):
        """WallLubricationModel cannot be directly instantiated."""
        with pytest.raises(TypeError):
            WallLubricationModel(d=1e-3, rho_c=998.0, rho_d=1.225)


# ============================================================================
# AntalWallLubrication
# ============================================================================


class TestAntalWallLubrication:
    """Antal wall lubrication model tests."""

    def test_init(self):
        model = AntalWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        assert model.d == 1e-3
        assert model.rho_c == 998.0
        assert model.rho_d == 1.225

    def test_custom_coefficients(self):
        model = AntalWallLubrication(
            d=2e-3, rho_c=1000.0, rho_d=1.0, C_w0=0.1, C_w_max=20.0,
        )
        assert model.C_w0 == 0.1
        assert model.C_w_max == 20.0

    def test_compute_returns_positive_force(self):
        """Force should be positive (away from wall) for positive wall normal."""
        model = AntalWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((n, 3), 0.1, dtype=CFD_DTYPE)
        wall_dist = torch.full((n,), 0.01, dtype=CFD_DTYPE)
        wall_normal = torch.zeros((n, 3), dtype=CFD_DTYPE)
        wall_normal[:, 1] = 1.0  # pointing in +y direction

        F = model.compute(alpha, U_rel, wall_dist, wall_normal)
        assert F.shape == (n, 3)
        # Force should be in wall-normal direction (positive y)
        assert (F[:, 1] > 0).all()
        assert torch.isfinite(F).all()

    def test_compute_force_increases_closer_to_wall(self):
        """Force should be larger closer to the wall (Antal model)."""
        model = AntalWallLubrication(
            d=1e-3, rho_c=998.0, rho_d=1.225, C_w0=0.05, C_w_max=100.0,
        )
        alpha = torch.tensor([0.3], dtype=CFD_DTYPE)
        U_rel = torch.tensor([[0.1, 0.0, 0.0]], dtype=CFD_DTYPE)
        wall_normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        dist_far = torch.tensor([0.1], dtype=CFD_DTYPE)
        dist_near = torch.tensor([0.001], dtype=CFD_DTYPE)

        F_far = model.compute(alpha, U_rel, dist_far, wall_normal)
        F_near = model.compute(alpha, U_rel, dist_near, wall_normal)

        assert F_near.norm() > F_far.norm()

    def test_compute_force_capped_near_wall(self):
        """Force should be capped at C_w_max near the wall."""
        model = AntalWallLubrication(
            d=1e-3, rho_c=998.0, rho_d=1.225, C_w0=0.05, C_w_max=5.0,
        )
        alpha = torch.tensor([0.3], dtype=CFD_DTYPE)
        U_rel = torch.tensor([[0.1, 0.0, 0.0]], dtype=CFD_DTYPE)
        wall_normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        dist_very_near = torch.tensor([1e-8], dtype=CFD_DTYPE)
        dist_near = torch.tensor([1e-5], dtype=CFD_DTYPE)

        F_very_near = model.compute(alpha, U_rel, dist_very_near, wall_normal)
        F_near = model.compute(alpha, U_rel, dist_near, wall_normal)

        # Both should be capped → similar magnitude
        assert torch.allclose(
            F_very_near.norm(), F_near.norm(), rtol=0.1,
        )

    def test_compute_increases_with_alpha(self):
        """Force proportional to alpha."""
        model = AntalWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        U_rel = torch.full((5, 3), 0.1, dtype=CFD_DTYPE)
        wall_dist = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        wall_normal = torch.zeros((5, 3), dtype=CFD_DTYPE)
        wall_normal[:, 1] = 1.0

        alpha_low = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        alpha_high = torch.full((5,), 0.5, dtype=CFD_DTYPE)

        F_low = model.compute(alpha_low, U_rel, wall_dist, wall_normal)
        F_high = model.compute(alpha_high, U_rel, wall_dist, wall_normal)

        assert F_high.norm() > F_low.norm()


# ============================================================================
# TomiyamaWallLubrication
# ============================================================================


class TestTomiyamaWallLubrication:
    """Tomiyama wall lubrication model tests."""

    def test_init(self):
        model = TomiyamaWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        assert model.d == 1e-3
        assert model.sigma == 0.072

    def test_eotvos_number(self):
        model = TomiyamaWallLubrication(
            d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072,
        )
        Eo = model.eotvos_number
        # Eo = 9.81 * |998 - 1.225| * (1e-3)^2 / 0.072
        expected = 9.81 * abs(998.0 - 1.225) * (1e-3) ** 2 / 0.072
        assert Eo == pytest.approx(expected, rel=1e-6)

    def test_coefficient_small_bubble(self):
        """For small Eo (small bubble): C_wl uses polynomial."""
        model = TomiyamaWallLubrication(
            d=1e-4, rho_c=998.0, rho_d=1.225, sigma=0.072,
        )
        C = model._wall_lubrication_coefficient()
        Eo = model.eotvos_number
        assert Eo < 1.0
        expected = -0.0063 * Eo ** 2 + 0.078 * Eo + 0.0035
        assert C == pytest.approx(expected, rel=1e-6)

    def test_coefficient_large_bubble(self):
        """For large Eo (large bubble): C_wl = 0.0035."""
        model = TomiyamaWallLubrication(
            d=1e-2, rho_c=998.0, rho_d=1.225, sigma=0.001,
        )
        C = model._wall_lubrication_coefficient()
        Eo = model.eotvos_number
        assert Eo >= 1.0
        assert C == pytest.approx(0.0035)

    def test_compute_returns_correct_shape(self):
        model = TomiyamaWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((n, 3), 0.1, dtype=CFD_DTYPE)
        wall_dist = torch.full((n,), 0.01, dtype=CFD_DTYPE)
        wall_normal = torch.zeros((n, 3), dtype=CFD_DTYPE)
        wall_normal[:, 1] = 1.0

        F = model.compute(alpha, U_rel, wall_dist, wall_normal)
        assert F.shape == (n, 3)
        assert torch.isfinite(F).all()

    def test_compute_force_direction(self):
        """Force should be in the wall-normal direction."""
        model = TomiyamaWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        alpha = torch.tensor([0.3], dtype=CFD_DTYPE)
        U_rel = torch.tensor([[0.1, 0.0, 0.0]], dtype=CFD_DTYPE)
        wall_dist = torch.tensor([0.01], dtype=CFD_DTYPE)
        wall_normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        F = model.compute(alpha, U_rel, wall_dist, wall_normal)
        # Force should be primarily in y-direction
        assert abs(F[0, 1]) > abs(F[0, 0])
        assert abs(F[0, 1]) > abs(F[0, 2])

    def test_compute_force_increases_closer_to_wall(self):
        """Force increases closer to the wall (y_w in denominator)."""
        model = TomiyamaWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225)
        alpha = torch.tensor([0.3], dtype=CFD_DTYPE)
        U_rel = torch.tensor([[0.1, 0.0, 0.0]], dtype=CFD_DTYPE)
        wall_normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        dist_far = torch.tensor([0.1], dtype=CFD_DTYPE)
        dist_near = torch.tensor([0.001], dtype=CFD_DTYPE)

        F_far = model.compute(alpha, U_rel, dist_far, wall_normal)
        F_near = model.compute(alpha, U_rel, dist_near, wall_normal)

        assert F_near.norm() > F_far.norm()

    def test_custom_C_wl_override(self):
        """Override coefficient produces expected force."""
        model = TomiyamaWallLubrication(
            d=1e-3, rho_c=998.0, rho_d=1.225, C_wl=0.01,
        )
        assert model._wall_lubrication_coefficient() == pytest.approx(0.01)

    def test_force_proportional_to_U_rel_squared(self):
        """Force is proportional to |U_rel|^2."""
        model = TomiyamaWallLubrication(d=1e-3, rho_c=998.0, rho_d=1.225, C_wl=0.01)
        alpha = torch.tensor([0.3], dtype=CFD_DTYPE)
        wall_dist = torch.tensor([0.01], dtype=CFD_DTYPE)
        wall_normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=CFD_DTYPE)

        U1 = torch.tensor([[0.1, 0.0, 0.0]], dtype=CFD_DTYPE)
        U2 = torch.tensor([[0.2, 0.0, 0.0]], dtype=CFD_DTYPE)

        F1 = model.compute(alpha, U1, wall_dist, wall_normal)
        F2 = model.compute(alpha, U2, wall_dist, wall_normal)

        # F2 / F1 should be (0.2/0.1)^2 = 4.0
        ratio = F2.norm() / F1.norm()
        assert ratio == pytest.approx(4.0, rel=1e-6)
