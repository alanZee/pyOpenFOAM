"""Tests for thin film flow model.

Tests cover:
- FilmModel constructor and properties
- Film pressure from curvature
- Gravity source term (tangential component)
- Capillary pressure
- Wall shear stress (Nusselt film theory)
- Film Reynolds and Weber numbers
- solve_timestep convenience method
"""

import math

import pytest
import torch

from pyfoam.multiphase.film_model import FilmModel


class TestFilmModelInit:
    """Constructor and property tests."""

    def test_defaults(self):
        model = FilmModel()
        assert model.rho_film == pytest.approx(998.0)
        assert model.mu_film == pytest.approx(1.002e-3)
        assert model.sigma == pytest.approx(0.072)
        assert model.contact_angle == pytest.approx(90.0)
        assert model.delta_min == pytest.approx(1e-7)

    def test_custom_params(self):
        model = FilmModel(rho_film=800.0, sigma=0.05, contact_angle=45.0)
        assert model.rho_film == pytest.approx(800.0)
        assert model.sigma == pytest.approx(0.05)
        assert model.contact_angle == pytest.approx(45.0)

    def test_contact_angle_properties(self):
        model = FilmModel(contact_angle=60.0)
        assert model.contact_angle_rad == pytest.approx(math.radians(60.0))
        assert model.cos_theta == pytest.approx(math.cos(math.radians(60.0)))


class TestFilmPressure:
    """Film pressure from curvature tests."""

    def test_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4
        curvature = torch.rand(10, dtype=torch.float64)
        p = model.film_pressure(delta, curvature)
        assert p.shape == (10,)

    def test_increases_with_curvature(self):
        model = FilmModel(sigma=0.072, contact_angle=90.0)
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        kappa = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        p = model.film_pressure(delta, kappa)
        # At theta=90, cos(theta)=0, so all pressures should be ~0
        assert torch.allclose(p, torch.zeros_like(p), atol=1e-10)

    def test_wetting_angle_gives_negative_pressure(self):
        """Wetting angle < 90 should give negative (suction) pressure."""
        model = FilmModel(sigma=0.072, contact_angle=30.0)
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        kappa = torch.full((5,), 100.0, dtype=torch.float64)
        p = model.film_pressure(delta, kappa)
        # cos(30) > 0, curvature > 0 => p < 0
        assert (p < 0).all()

    def test_finite(self):
        model = FilmModel()
        delta = torch.rand(20, dtype=torch.float64) * 1e-3 + 1e-7
        curvature = torch.randn(20, dtype=torch.float64) * 100
        p = model.film_pressure(delta, curvature)
        assert torch.isfinite(p).all()


class TestGravitySource:
    """Gravity source term tests."""

    def test_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4
        n_wall = torch.zeros(10, 3, dtype=torch.float64)
        n_wall[:, 2] = 1.0  # wall normal pointing up
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        S = model.gravity_source(delta, n_wall, gravity)
        assert S.shape == (10, 3)

    def test_tangential_only(self):
        """Gravity source should have zero component along wall normal."""
        model = FilmModel()
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        n_wall = torch.zeros(5, 3, dtype=torch.float64)
        n_wall[:, 2] = 1.0
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        S = model.gravity_source(delta, n_wall, gravity)
        # Normal component should be zero (or near-zero)
        g_dot_n = (S * n_wall).sum(dim=1)
        assert torch.allclose(g_dot_n, torch.zeros(5, dtype=torch.float64), atol=1e-10)

    def test_full_gravity_tangential(self):
        """When wall is horizontal, entire gravity is tangential."""
        model = FilmModel()
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        n_wall = torch.zeros(5, 3, dtype=torch.float64)
        n_wall[:, 0] = 1.0  # wall normal in x
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        S = model.gravity_source(delta, n_wall, gravity)
        # Gravity is entirely tangential to this wall
        assert torch.allclose(S[:, 0], torch.zeros(5, dtype=torch.float64), atol=1e-10)
        assert (S[:, 2] != 0).all()  # z-component should be nonzero


class TestCapillaryPressure:
    """Capillary pressure tests."""

    def test_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-3
        p = model.capillary_pressure(delta)
        assert p.shape == (10,)

    def test_inversely_proportional_to_thickness(self):
        """p_cap = sigma * cos(theta) / delta: smaller delta -> larger |p|."""
        model = FilmModel(sigma=0.072, contact_angle=30.0)
        delta_thin = torch.tensor([1e-6], dtype=torch.float64)
        delta_thick = torch.tensor([1e-3], dtype=torch.float64)
        p_thin = model.capillary_pressure(delta_thin)
        p_thick = model.capillary_pressure(delta_thick)
        assert abs(p_thin[0].item()) > abs(p_thick[0].item())


class TestWallShearStress:
    """Wall shear stress (Nusselt) tests."""

    def test_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        tau = model.wall_shear_stress(delta, U)
        assert tau.shape == (10, 3)

    def test_proportional_to_velocity(self):
        """tau_w = mu * U / delta: proportional to U."""
        model = FilmModel(mu_film=1e-3)
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        U = torch.ones(5, 3, dtype=torch.float64) * 0.5
        tau = model.wall_shear_stress(delta, U)
        expected = 1e-3 * 0.5 / 1e-4
        assert torch.allclose(tau[:, 0], torch.full((5,), expected, dtype=torch.float64))

    def test_inversely_proportional_to_thickness(self):
        """Thinner films have higher shear stress."""
        model = FilmModel(mu_film=1e-3)
        U = torch.ones(5, 3, dtype=torch.float64)
        delta_thin = torch.full((5,), 1e-6, dtype=torch.float64)
        delta_thick = torch.full((5,), 1e-3, dtype=torch.float64)
        tau_thin = model.wall_shear_stress(delta_thin, U)
        tau_thick = model.wall_shear_stress(delta_thick, U)
        assert tau_thin.norm() > tau_thick.norm()


class TestFilmNumbers:
    """Reynolds and Weber number tests."""

    def test_reynolds_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        Re = model.film_reynolds_number(delta, U)
        assert Re.shape == (10,)

    def test_weber_shape(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        We = model.film_weber_number(delta, U)
        assert We.shape == (10,)

    def test_reynolds_formula(self):
        model = FilmModel(rho_film=1000.0, mu_film=1e-3)
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        U = torch.ones(5, 3, dtype=torch.float64)
        U[:, 1:] = 0  # |U| = 1.0
        Re = model.film_reynolds_number(delta, U)
        expected = 1000.0 * 1.0 * 1e-4 / 1e-3  # = 100
        assert torch.allclose(Re, torch.full((5,), expected, dtype=torch.float64))

    def test_weber_formula(self):
        model = FilmModel(rho_film=1000.0, sigma=0.072)
        delta = torch.full((5,), 1e-4, dtype=torch.float64)
        U = torch.ones(5, 3, dtype=torch.float64) * 2.0
        U[:, 1:] = 0  # |U| = 2.0
        We = model.film_weber_number(delta, U)
        expected = 1000.0 * 4.0 * 1e-4 / 0.072
        assert torch.allclose(We, torch.full((5,), expected, dtype=torch.float64))

    def test_finite(self):
        model = FilmModel()
        delta = torch.rand(20, dtype=torch.float64) * 1e-3 + 1e-7
        U = torch.randn(20, 3, dtype=torch.float64) * 0.5
        assert torch.isfinite(model.film_reynolds_number(delta, U)).all()
        assert torch.isfinite(model.film_weber_number(delta, U)).all()


class TestSolveTimestep:
    """Convenience solve_timestep tests."""

    def test_returns_all_keys(self):
        model = FilmModel()
        delta = torch.rand(10, dtype=torch.float64) * 1e-4 + 1e-7
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        n_wall = torch.zeros(10, 3, dtype=torch.float64)
        n_wall[:, 2] = 1.0
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        kappa = torch.randn(10, dtype=torch.float64) * 10
        result = model.solve_timestep(delta, U, n_wall, gravity, kappa)
        assert set(result.keys()) == {"p_film", "p_cap", "S_grav", "tau_w", "Re_film", "We_film"}

    def test_all_finite(self):
        model = FilmModel()
        delta = torch.rand(20, dtype=torch.float64) * 1e-3 + 1e-7
        U = torch.randn(20, 3, dtype=torch.float64) * 0.1
        n_wall = torch.zeros(20, 3, dtype=torch.float64)
        n_wall[:, 2] = 1.0
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        kappa = torch.randn(20, dtype=torch.float64) * 10
        result = model.solve_timestep(delta, U, n_wall, gravity, kappa)
        for v in result.values():
            assert torch.isfinite(v).all()
