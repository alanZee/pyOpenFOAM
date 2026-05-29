"""Tests for Kato-Launder turbulence production limiter.

Tests cover:
- KatoLaunderDamping init
- damp() method
- damp_from_P_k() method
- damping_factor() method
- Behaviour at stagnation points and shear layers
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.turbulence.turbulence_kato_launder import KatoLaunderDamping


class TestKatoLaunderDampingInit:
    """Initialisation tests."""

    def test_init_default(self):
        d = KatoLaunderDamping()
        assert d.nu_t == pytest.approx(1e-3)

    def test_init_custom_scalar(self):
        d = KatoLaunderDamping(nu_t=0.01)
        assert d.nu_t == pytest.approx(0.01)

    def test_init_tensor(self):
        nu_t = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        d = KatoLaunderDamping(nu_t=nu_t)
        assert isinstance(d.nu_t, torch.Tensor)


class TestKatoLaunderDampingDamp:
    """Tests for the damp() method."""

    def test_damp_shape(self):
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((10,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((10,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((10,), 50.0, dtype=CFD_DTYPE)
        P_damped = d.damp(P_k, strain, vort)
        assert P_damped.shape == (10,)

    def test_damp_formula(self):
        """P_KL = 2 * nu_t * |S| * |Omega|."""
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        P_damped = d.damp(P_k, strain, vort)
        expected = 2.0 * 0.01 * 100.0 * 50.0  # = 100.0
        assert torch.allclose(
            P_damped, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-6,
        )

    def test_damp_at_stagnation(self):
        """At stagnation: |Omega| ~ 0 → P_KL ~ 0."""
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 0.001, dtype=CFD_DTYPE)
        P_damped = d.damp(P_k, strain, vort)
        # P_KL = 2 * 0.01 * 100 * 0.001 = 0.002
        assert (P_damped < P_k).all()
        assert P_damped.mean() < 1.0

    def test_damp_in_shear(self):
        """In shear: |S| ~ |Omega| → P_KL ~ standard P_k."""
        nu_t = 0.01
        d = KatoLaunderDamping(nu_t=nu_t)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        P_k = torch.full((5,), 2.0 * nu_t * 100.0 * 100.0, dtype=CFD_DTYPE)
        P_damped = d.damp(P_k, strain, vort)
        # P_KL = 2 * 0.01 * 100 * 100 = 200 = P_k
        assert torch.allclose(P_damped, P_k, rtol=1e-6)

    def test_damp_with_tensor_nu_t(self):
        """Per-cell nu_t works."""
        nu_t = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        d = KatoLaunderDamping(nu_t=nu_t)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        P_damped = d.damp(P_k, strain, vort)
        expected = 2.0 * 0.01 * 100.0 * 50.0
        assert torch.allclose(
            P_damped, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-6,
        )

    def test_damp_finite(self):
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.randn(20, dtype=CFD_DTYPE).abs() * 100
        strain = torch.randn(20, dtype=CFD_DTYPE).abs() + 1.0
        vort = torch.randn(20, dtype=CFD_DTYPE).abs() + 1.0
        P_damped = d.damp(P_k, strain, vort)
        assert torch.isfinite(P_damped).all()


class TestKatoLaunderDampingFromPK:
    """Tests for the damp_from_P_k() method."""

    def test_damp_from_pk_formula(self):
        """P_KL = P_k * |Omega| / |S|."""
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        P_damped = d.damp_from_P_k(P_k, strain, vort)
        expected = 200.0 * 50.0 / 100.0  # = 100.0
        assert torch.allclose(
            P_damped, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-6,
        )

    def test_damp_from_pk_in_shear(self):
        """|S| ~ |Omega| → P_KL ~ P_k."""
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        P_damped = d.damp_from_P_k(P_k, strain, vort)
        assert torch.allclose(P_damped, P_k, rtol=1e-6)

    def test_damp_from_pk_at_stagnation(self):
        """|Omega| << |S| → P_KL << P_k."""
        d = KatoLaunderDamping(nu_t=0.01)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        P_damped = d.damp_from_P_k(P_k, strain, vort)
        # ratio = 1/100 = 0.01 → P_KL = 200 * 0.01 = 2.0
        assert (P_damped < P_k).all()
        assert torch.allclose(
            P_damped, torch.full((5,), 2.0, dtype=CFD_DTYPE), rtol=1e-6,
        )


class TestKatoLaunderDampingFactor:
    """Tests for the damping_factor() method."""

    def test_factor_in_shear(self):
        d = KatoLaunderDamping()
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        factor = d.damping_factor(strain, vort)
        assert torch.allclose(factor, torch.ones(5, dtype=CFD_DTYPE), rtol=1e-6)

    def test_factor_at_stagnation(self):
        d = KatoLaunderDamping()
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        factor = d.damping_factor(strain, vort)
        # factor = 1/100 = 0.01
        assert torch.allclose(factor, torch.full((5,), 0.01, dtype=CFD_DTYPE), rtol=1e-6)

    def test_factor_capped_at_one(self):
        """Factor is clamped to max 1.0."""
        d = KatoLaunderDamping()
        strain = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        vort = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        factor = d.damping_factor(strain, vort)
        assert (factor <= 1.0 + 1e-10).all()

    def test_factor_between_zero_and_one(self):
        d = KatoLaunderDamping()
        strain = torch.rand(20, dtype=CFD_DTYPE).abs() + 1.0
        vort = torch.rand(20, dtype=CFD_DTYPE).abs() + 0.001
        factor = d.damping_factor(strain, vort)
        assert (factor >= 0).all()
        assert (factor <= 1.0 + 1e-10).all()
