"""
Unit tests for relative velocity models.

Tests cover:
- ManninenRelativeVelocity: algebraic slip model
- GraceRelativeVelocity: Grace drag correlation
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestManninenRelativeVelocity:
    """Tests for Manninen et al. algebraic slip model."""

    def test_init(self):
        from pyfoam.multiphase.relative_velocity import ManninenRelativeVelocity

        model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3)
        assert model.rho_d == 1.225
        assert model.rho_c == 998.0
        assert model.d == 1e-3

    def test_particle_relaxation_time(self):
        from pyfoam.multiphase.relative_velocity import ManninenRelativeVelocity

        model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3)
        tau = model.particle_relaxation_time
        # tau = rho_d * d^2 / (18 * mu_c)
        expected = 1.225 * (1e-3) ** 2 / (18.0 * 1e-3)
        assert tau == pytest.approx(expected, rel=1e-6)

    def test_compute_shape(self):
        from pyfoam.multiphase.relative_velocity import ManninenRelativeVelocity

        model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3)
        alpha_d = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        U_slip = model.compute(alpha_d, U_mix)
        assert U_slip.shape == (10, 3)
        assert torch.isfinite(U_slip).all()

    def test_compute_gravity_driven(self):
        """Slip velocity opposes gravity (light particle rises)."""
        from pyfoam.multiphase.relative_velocity import ManninenRelativeVelocity

        model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3)
        alpha_d = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        g = torch.tensor([0.0, 0.0, -9.81], dtype=CFD_DTYPE)
        U_slip = model.compute(alpha_d, U_mix, g=g)

        # Light particle (rho_d < rho_c) should have positive z-slip (rises)
        assert (U_slip[:, 2] > 0).all(), "Light particle should rise (positive z-slip)"

    def test_compute_alpha_dependence(self):
        """Slip velocity magnitude increases with volume fraction."""
        from pyfoam.multiphase.relative_velocity import ManninenRelativeVelocity

        model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3)
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        alpha_low = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        alpha_high = torch.full((10,), 0.3, dtype=CFD_DTYPE)

        U_slip_low = model.compute(alpha_low, U_mix)
        U_slip_high = model.compute(alpha_high, U_mix)

        # Higher alpha -> higher slip (due to hindered settling effect in model)
        assert U_slip_high[:, 2].mean() > U_slip_low[:, 2].mean()


class TestGraceRelativeVelocity:
    """Tests for Grace drag correlation model."""

    def test_init(self):
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        assert model.rho_d == 1.225
        assert model.sigma == 0.072

    def test_eotvos_number(self):
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        Eo = model.eotvos_number
        assert Eo > 0
        assert isinstance(Eo, float)

    def test_morton_number(self):
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        Mo = model.morton_number
        assert Mo > 0
        assert isinstance(Mo, float)

    def test_compute_shape(self):
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        alpha_d = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        U_slip = model.compute(alpha_d, U_mix)
        assert U_slip.shape == (10, 3)
        assert torch.isfinite(U_slip).all()

    def test_compute_gravity_direction(self):
        """Light particles rise, heavy particles sink."""
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        alpha_d = torch.full((10,), 0.05, dtype=CFD_DTYPE)
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        U_slip = model.compute(alpha_d, U_mix)
        # Light particle should rise (positive z-component)
        assert (U_slip[:, 2] > 0).all()

    def test_hindered_settling(self):
        """Higher volume fraction reduces slip velocity."""
        from pyfoam.multiphase.relative_velocity import GraceRelativeVelocity

        model = GraceRelativeVelocity(
            rho_d=1.225, rho_c=998.0, d=1e-3, mu_c=1e-3, sigma=0.072,
        )
        U_mix = torch.zeros(10, 3, dtype=CFD_DTYPE)

        alpha_low = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        alpha_high = torch.full((10,), 0.4, dtype=CFD_DTYPE)

        U_slip_low = model.compute(alpha_low, U_mix)
        U_slip_high = model.compute(alpha_high, U_mix)

        # Hindered settling: higher alpha -> lower slip
        assert U_slip_low[:, 2].mean() > U_slip_high[:, 2].mean()
