"""Tests for Xi premixed combustion model.

Tests cover:
- XiFluid constructor and parameter validation
- Flame efficiency function (Meneveau & Poinsot)
- Xi production and destruction source terms
- Reaction progress (b) source term
- Turbulent flame speed
- Xi and b clamping
- solve_timestep convenience method
"""

import math

import pytest
import torch

from pyfoam.multiphase.xi_fluid import XiFluid


class TestXiFluidInit:
    """Constructor and default parameter tests."""

    def test_defaults(self):
        model = XiFluid()
        assert model.S_L == pytest.approx(0.3)
        assert model.sigma_y == pytest.approx(2.0)
        assert model.Xi_min == pytest.approx(1.0)
        assert model.Xi_max == pytest.approx(10.0)
        assert model.b_min == pytest.approx(0.0)
        assert model.D_Xi == pytest.approx(1.0)

    def test_custom_params(self):
        model = XiFluid(S_L=0.5, sigma_y=3.0, Xi_max=20.0)
        assert model.S_L == pytest.approx(0.5)
        assert model.sigma_y == pytest.approx(3.0)
        assert model.Xi_max == pytest.approx(20.0)


class TestFlameEfficiency:
    """Flame efficiency function tests."""

    def test_shape(self):
        model = XiFluid()
        Xi = torch.ones(10, dtype=torch.float64) * 2.0
        u_prime = torch.rand(10, dtype=torch.float64)
        E = model.flame_efficiency(Xi, u_prime)
        assert E.shape == (10,)

    def test_planar_flame(self):
        """At Xi=1 (planar) and u'=0, E = 1."""
        model = XiFluid(S_L=0.3)
        Xi = torch.ones(5, dtype=torch.float64)
        u_prime = torch.zeros(5, dtype=torch.float64)
        E = model.flame_efficiency(Xi, u_prime)
        assert torch.allclose(E, torch.ones(5, dtype=torch.float64))

    def test_increases_with_wrinkling(self):
        """E increases with Xi for fixed u'."""
        model = XiFluid(S_L=0.3)
        u_prime = torch.full((5,), 0.5, dtype=torch.float64)
        Xi_low = torch.full((5,), 1.0, dtype=torch.float64)
        Xi_high = torch.full((5,), 3.0, dtype=torch.float64)
        E_low = model.flame_efficiency(Xi_low, u_prime)
        E_high = model.flame_efficiency(Xi_high, u_prime)
        assert (E_high > E_low).all()

    def test_increases_with_u_prime(self):
        """E increases with u' for fixed Xi > 1."""
        model = XiFluid(S_L=0.3)
        Xi = torch.full((5,), 2.0, dtype=torch.float64)
        u_low = torch.full((5,), 0.1, dtype=torch.float64)
        u_high = torch.full((5,), 1.0, dtype=torch.float64)
        E_low = model.flame_efficiency(Xi, u_low)
        E_high = model.flame_efficiency(Xi, u_high)
        assert (E_high > E_low).all()

    def test_finite(self):
        model = XiFluid()
        Xi = torch.rand(20, dtype=torch.float64) * 5 + 1
        u_prime = torch.rand(20, dtype=torch.float64) * 5
        E = model.flame_efficiency(Xi, u_prime)
        assert torch.isfinite(E).all()


class TestXiSourceTerms:
    """Xi production and destruction tests."""

    def test_production_shape(self):
        model = XiFluid()
        Xi = torch.ones(10, dtype=torch.float64) * 2.0
        u_prime = torch.rand(10, dtype=torch.float64)
        l_t = torch.rand(10, dtype=torch.float64) * 0.01
        P = model.xi_production(Xi, u_prime, l_t)
        assert P.shape == (10,)

    def test_production_zero_at_planar(self):
        """Production is zero when Xi=1 (planar flame)."""
        model = XiFluid()
        Xi = torch.ones(5, dtype=torch.float64)
        u_prime = torch.rand(5, dtype=torch.float64)
        l_t = torch.rand(5, dtype=torch.float64) * 0.01
        P = model.xi_production(Xi, u_prime, l_t)
        assert torch.allclose(P, torch.zeros(5, dtype=torch.float64))

    def test_production_positive_above_planar(self):
        """Production is positive when Xi > 1."""
        model = XiFluid()
        Xi = torch.full((5,), 2.0, dtype=torch.float64)
        u_prime = torch.full((5,), 1.0, dtype=torch.float64)
        l_t = torch.full((5,), 0.01, dtype=torch.float64)
        P = model.xi_production(Xi, u_prime, l_t)
        assert (P > 0).all()

    def test_destruction_shape(self):
        model = XiFluid()
        Xi = torch.ones(10, dtype=torch.float64) * 2.0
        u_prime = torch.rand(10, dtype=torch.float64)
        l_t = torch.rand(10, dtype=torch.float64) * 0.01
        D = model.xi_destruction(Xi, u_prime, l_t)
        assert D.shape == (10,)

    def test_destruction_zero_at_planar(self):
        """Destruction is zero when Xi=1."""
        model = XiFluid()
        Xi = torch.ones(5, dtype=torch.float64)
        u_prime = torch.rand(5, dtype=torch.float64)
        l_t = torch.rand(5, dtype=torch.float64) * 0.01
        D = model.xi_destruction(Xi, u_prime, l_t)
        assert torch.allclose(D, torch.zeros(5, dtype=torch.float64))

    def test_destruction_positive_above_planar(self):
        """Destruction is positive when Xi > 1."""
        model = XiFluid()
        Xi = torch.full((5,), 3.0, dtype=torch.float64)
        u_prime = torch.rand(5, dtype=torch.float64)
        l_t = torch.rand(5, dtype=torch.float64) * 0.01
        D = model.xi_destruction(Xi, u_prime, l_t)
        assert (D > 0).all()

    def test_finite(self):
        model = XiFluid()
        Xi = torch.rand(20, dtype=torch.float64) * 5 + 1
        u_prime = torch.rand(20, dtype=torch.float64) * 5
        l_t = torch.rand(20, dtype=torch.float64) * 0.1 + 1e-6
        assert torch.isfinite(model.xi_production(Xi, u_prime, l_t)).all()
        assert torch.isfinite(model.xi_destruction(Xi, u_prime, l_t)).all()


class TestBSource:
    """Reaction progress source term tests."""

    def test_shape(self):
        model = XiFluid()
        b = torch.rand(10, dtype=torch.float64)
        Xi = torch.ones(10, dtype=torch.float64) * 2.0
        rho = torch.ones(10, dtype=torch.float64) * 1.225
        omega = model.b_source(b, Xi, rho)
        assert omega.shape == (10,)

    def test_zero_at_full_burnt(self):
        """Source is zero when b=1 (fully burnt)."""
        model = XiFluid()
        b = torch.ones(5, dtype=torch.float64)
        Xi = torch.ones(5, dtype=torch.float64) * 2.0
        rho = torch.ones(5, dtype=torch.float64)
        omega = model.b_source(b, Xi, rho)
        assert torch.allclose(omega, torch.zeros(5, dtype=torch.float64))

    def test_positive_when_unburnt(self):
        """Source is positive when b < 1."""
        model = XiFluid()
        b = torch.zeros(5, dtype=torch.float64)
        Xi = torch.ones(5, dtype=torch.float64) * 2.0
        rho = torch.ones(5, dtype=torch.float64)
        omega = model.b_source(b, Xi, rho)
        assert (omega > 0).all()

    def test_increases_with_xi(self):
        """Higher Xi gives higher reaction rate."""
        model = XiFluid()
        b = torch.full((5,), 0.5, dtype=torch.float64)
        rho = torch.ones(5, dtype=torch.float64)
        Xi_low = torch.full((5,), 1.0, dtype=torch.float64)
        Xi_high = torch.full((5,), 5.0, dtype=torch.float64)
        omega_low = model.b_source(b, Xi_low, rho)
        omega_high = model.b_source(b, Xi_high, rho)
        assert (omega_high > omega_low).all()

    def test_finite(self):
        model = XiFluid()
        b = torch.rand(20, dtype=torch.float64)
        Xi = torch.rand(20, dtype=torch.float64) * 5 + 1
        rho = torch.rand(20, dtype=torch.float64) * 2 + 0.5
        omega = model.b_source(b, Xi, rho)
        assert torch.isfinite(omega).all()


class TestTurbulentFlameSpeed:
    """Turbulent flame speed tests."""

    def test_formula(self):
        model = XiFluid(S_L=0.3)
        Xi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        S_t = model.turbulent_flame_speed(Xi)
        expected = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64)
        assert torch.allclose(S_t, expected)

    def test_min_xi_enforced(self):
        """Xi < Xi_min is clamped to Xi_min."""
        model = XiFluid(S_L=0.3, Xi_min=1.0)
        Xi = torch.tensor([0.5], dtype=torch.float64)
        S_t = model.turbulent_flame_speed(Xi)
        assert S_t[0] == pytest.approx(0.3)  # S_L * Xi_min


class TestClamping:
    """Clamping tests."""

    def test_clamp_xi(self):
        model = XiFluid(Xi_min=1.0, Xi_max=10.0)
        Xi = torch.tensor([0.5, 2.0, 15.0], dtype=torch.float64)
        clamped = model.clamp_xi(Xi)
        assert clamped[0] == pytest.approx(1.0)
        assert clamped[1] == pytest.approx(2.0)
        assert clamped[2] == pytest.approx(10.0)

    def test_clamp_b(self):
        model = XiFluid(b_min=0.0)
        b = torch.tensor([-0.1, 0.5, 1.5], dtype=torch.float64)
        clamped = model.clamp_b(b)
        assert clamped[0] == pytest.approx(0.0)
        assert clamped[1] == pytest.approx(0.5)
        assert clamped[2] == pytest.approx(1.0)


class TestSolveTimestep:
    """Convenience solve_timestep tests."""

    def test_returns_all_keys(self):
        model = XiFluid()
        Xi = torch.rand(10, dtype=torch.float64) * 3 + 1
        b = torch.rand(10, dtype=torch.float64)
        rho = torch.ones(10, dtype=torch.float64) * 1.225
        u_prime = torch.rand(10, dtype=torch.float64)
        l_t = torch.rand(10, dtype=torch.float64) * 0.01 + 1e-6
        result = model.solve_timestep(Xi, b, rho, u_prime, l_t)
        expected_keys = {"Xi_clamped", "b_clamped", "P_Xi", "D_Xi", "omega_b", "S_t", "E"}
        assert set(result.keys()) == expected_keys

    def test_all_finite(self):
        model = XiFluid()
        Xi = torch.rand(20, dtype=torch.float64) * 5 + 1
        b = torch.rand(20, dtype=torch.float64)
        rho = torch.ones(20, dtype=torch.float64) * 1.225
        u_prime = torch.rand(20, dtype=torch.float64) * 2
        l_t = torch.rand(20, dtype=torch.float64) * 0.1 + 1e-6
        result = model.solve_timestep(Xi, b, rho, u_prime, l_t)
        for v in result.values():
            assert torch.isfinite(v).all()
