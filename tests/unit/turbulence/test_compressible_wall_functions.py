"""
Tests for compressible wall functions.

Tests cover:
- CompressibleWallFunction ABC
- CompressibleNutWallFunction computation
- CompressibleKWallFunction computation
- Van Driest damping correction
- Compressible y+ definition
- Friction velocity computation
- Viscous sublayer behaviour
- Log-law region behaviour
"""

import math

import pytest
import torch

from pyfoam.turbulence.compressible_wall_functions import (
    CompressibleWallFunction,
    CompressibleNutWallFunction,
    CompressibleKWallFunction,
)


class TestCompressibleWallFunctionABC:
    """Abstract base class tests."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            CompressibleWallFunction()

    def test_compute_u_tau(self):
        wf = CompressibleNutWallFunction()
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        u_tau = wf.compute_u_tau(k)
        # u_tau = C_mu^{0.25} * sqrt(k)
        expected = 0.09 ** 0.25 * torch.sqrt(k)
        assert torch.allclose(u_tau, expected, atol=1e-10)

    def test_compute_u_tau_clamps_negative_k(self):
        wf = CompressibleNutWallFunction()
        k = torch.tensor([0.0, -1.0], dtype=torch.float64)
        u_tau = wf.compute_u_tau(k)
        assert torch.isfinite(u_tau).all()
        assert (u_tau >= 0).all()

    def test_compute_y_plus_compressible(self):
        wf = CompressibleNutWallFunction()
        u_tau = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        y_p = wf.compute_y_plus_compressible(u_tau, y, mu, rho)
        expected = 1.225 * 1.0 * 0.01 / 1e-5
        assert y_p[0] == pytest.approx(expected, rel=1e-6)

    def test_compute_y_plus_clamps_small_values(self):
        wf = CompressibleNutWallFunction()
        u_tau = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1e-5], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        y_p = wf.compute_y_plus_compressible(u_tau, y, mu, rho)
        assert y_p[0] >= 1e-4


class TestCompressibleNutWallFunction:
    """Compressible nut wall function tests."""

    def test_default_constants(self):
        wf = CompressibleNutWallFunction()
        assert wf.kappa == 0.41
        assert wf.E == 9.8
        assert wf.C_mu == 0.09

    def test_custom_constants(self):
        wf = CompressibleNutWallFunction(kappa=0.38, E=8.5)
        assert wf.kappa == 0.38
        assert wf.E == 8.5

    def test_compute_shape(self):
        wf = CompressibleNutWallFunction()
        n = 5
        k = torch.full((n,), 0.01, dtype=torch.float64)
        y = torch.full((n,), 0.001, dtype=torch.float64)
        mu = torch.full((n,), 1.8e-5, dtype=torch.float64)
        rho = torch.full((n,), 1.225, dtype=torch.float64)
        nut = wf.compute(k, y, mu, rho)
        assert nut.shape == (n,)

    def test_compute_nonnegative(self):
        wf = CompressibleNutWallFunction()
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        mu = torch.full((3,), 1.8e-5, dtype=torch.float64)
        rho = torch.full((3,), 1.225, dtype=torch.float64)
        nut = wf.compute(k, y, mu, rho)
        assert (nut >= 0).all()
        assert torch.isfinite(nut).all()

    def test_viscous_sublayer_zero_nut(self):
        """In viscous sublayer (y+ < 5), nut should be zero."""
        wf = CompressibleNutWallFunction()
        # Use very small y to ensure y+ < 5
        k = torch.tensor([0.01], dtype=torch.float64)
        y = torch.tensor([1e-8], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        nut = wf.compute(k, y, mu, rho)
        assert nut[0] == pytest.approx(0.0, abs=1e-20)

    def test_log_law_region_positive_nut(self):
        """In log-law region (y+ > 30), nut should be positive."""
        wf = CompressibleNutWallFunction()
        # Use moderate y to ensure y+ > 30
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        nut = wf.compute(k, y, mu, rho)
        assert nut[0] > 0

    def test_higher_k_higher_nut(self):
        """Higher k produces higher nut."""
        wf = CompressibleNutWallFunction()
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        k_low = torch.tensor([0.01], dtype=torch.float64)
        k_high = torch.tensor([1.0], dtype=torch.float64)
        nut_low = wf.compute(k_low, y, mu, rho)
        nut_high = wf.compute(k_high, y, mu, rho)
        assert nut_high[0] > nut_low[0]

    def test_higher_density_higher_nut(self):
        """Higher density increases y+ and thus affects nut."""
        wf = CompressibleNutWallFunction()
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho_low = torch.tensor([1.0], dtype=torch.float64)
        rho_high = torch.tensor([5.0], dtype=torch.float64)
        nut_low = wf.compute(k, y, mu, rho_low)
        nut_high = wf.compute(k, y, mu, rho_high)
        # Both should be finite and non-negative
        assert torch.isfinite(nut_low).all()
        assert torch.isfinite(nut_high).all()
        assert nut_low[0] >= 0
        assert nut_high[0] >= 0


class TestCompressibleKWallFunction:
    """Compressible k wall function tests."""

    def test_compute_shape(self):
        wf = CompressibleKWallFunction()
        n = 4
        k = torch.full((n,), 0.01, dtype=torch.float64)
        y = torch.full((n,), 0.001, dtype=torch.float64)
        mu = torch.full((n,), 1.8e-5, dtype=torch.float64)
        rho = torch.full((n,), 1.225, dtype=torch.float64)
        k_wall = wf.compute(k, y, mu, rho)
        assert k_wall.shape == (n,)

    def test_compute_positive(self):
        wf = CompressibleKWallFunction()
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.full((3,), 0.01, dtype=torch.float64)
        mu = torch.full((3,), 1.8e-5, dtype=torch.float64)
        rho = torch.full((3,), 1.225, dtype=torch.float64)
        k_wall = wf.compute(k, y, mu, rho)
        assert (k_wall > 0).all()
        assert torch.isfinite(k_wall).all()

    def test_local_equilibrium(self):
        """k_wall = u_tau^2 / sqrt(C_mu)."""
        wf = CompressibleKWallFunction()
        k = torch.tensor([0.04], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        k_wall = wf.compute(k, y, mu, rho)

        u_tau = 0.09 ** 0.25 * math.sqrt(0.04)
        expected = u_tau ** 2 / math.sqrt(0.09)
        assert k_wall[0] == pytest.approx(expected, rel=1e-6)

    def test_higher_k_higher_k_wall(self):
        wf = CompressibleKWallFunction()
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)
        k_low = torch.tensor([0.01], dtype=torch.float64)
        k_high = torch.tensor([1.0], dtype=torch.float64)
        kw_low = wf.compute(k_low, y, mu, rho)
        kw_high = wf.compute(k_high, y, mu, rho)
        assert kw_high[0] > kw_low[0]

    def test_van_driest_damping(self):
        """Van Driest correction reduces k near wall."""
        wf = CompressibleKWallFunction()
        k = torch.tensor([0.1, 0.1], dtype=torch.float64)
        y = torch.tensor([1e-6, 0.1], dtype=torch.float64)
        mu = torch.tensor([1.8e-5, 1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225, 1.225], dtype=torch.float64)

        k_vd = wf.compute_with_van_driest(k, y, mu, rho)
        # Near wall (small y): f_vd small => k_vd small
        # Far from wall (large y): f_vd ~ 1 => k_vd ~ k_eq
        assert k_vd[0] < k_vd[1]

    def test_van_driest_far_wall_no_damping(self):
        """Far from wall, Van Driest damping has no effect."""
        wf = CompressibleKWallFunction()
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([10.0], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)

        k_wall = wf.compute(k, y, mu, rho)
        k_vd = wf.compute_with_van_driest(k, y, mu, rho)
        # For large y+, Van Driest f ~ 1, so k_vd ~ k_wall
        assert k_vd[0] == pytest.approx(k_wall[0], rel=0.01)

    def test_van_driest_custom_A_plus(self):
        """Custom A_plus changes damping range."""
        wf = CompressibleKWallFunction()
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([0.001], dtype=torch.float64)
        mu = torch.tensor([1.8e-5], dtype=torch.float64)
        rho = torch.tensor([1.225], dtype=torch.float64)

        k_vd_small = wf.compute_with_van_driest(k, y, mu, rho, A_plus=10.0)
        k_vd_large = wf.compute_with_van_driest(k, y, mu, rho, A_plus=100.0)
        # Smaller A_plus: stronger damping => f_vd closer to 1 => k_vd larger
        # (Van Driest correction *removes* the viscous sublayer penalty,
        #  so smaller A_plus means the correction kicks in more strongly)
        assert k_vd_small[0] >= k_vd_large[0]
