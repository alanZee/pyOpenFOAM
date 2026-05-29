"""Tests for dense particle-laden flow model.

Tests cover:
- DenseParticleFluid constructor and parameter validation
- Gidaspow drag coefficient blend
- Drag force computation
- Solids pressure and radial distribution function
- Packing correction
- solve_timestep convenience method
"""

import pytest
import torch

from pyfoam.multiphase.dense_particle_fluid import DenseParticleFluid


class TestDenseParticleFluidInit:
    """Constructor and validation tests."""

    def test_defaults(self):
        model = DenseParticleFluid()
        assert model.d_p == pytest.approx(1e-4)
        assert model.rho_p == pytest.approx(2500.0)
        assert model.rho_f == pytest.approx(1.225)
        assert model.mu_f == pytest.approx(1.8e-5)
        assert model.alpha_max == pytest.approx(0.63)
        assert model.e_p == pytest.approx(0.9)

    def test_custom_params(self):
        model = DenseParticleFluid(d_p=1e-3, rho_p=2000.0, alpha_max=0.55)
        assert model.d_p == pytest.approx(1e-3)
        assert model.rho_p == pytest.approx(2000.0)
        assert model.alpha_max == pytest.approx(0.55)

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DenseParticleFluid(d_p=0.0)

    def test_invalid_alpha_max_raises(self):
        with pytest.raises(ValueError, match="alpha_max"):
            DenseParticleFluid(alpha_max=1.5)


class TestDragCoefficient:
    """Gidaspow drag coefficient tests."""

    def test_shape(self):
        model = DenseParticleFluid()
        alpha_p = torch.full((10,), 0.3, dtype=torch.float64)
        U_slip = torch.randn(10, 3, dtype=torch.float64) * 0.1
        beta = model.drag_coefficient(alpha_p, U_slip)
        assert beta.shape == (10,)

    def test_positive_output(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(20, dtype=torch.float64).clamp(0.01, 0.5)
        U_slip = torch.randn(20, 3, dtype=torch.float64) * 0.5
        beta = model.drag_coefficient(alpha_p, U_slip)
        assert (beta >= 0).all()

    def test_zero_slip_low_drag_in_dilute(self):
        """Zero slip in dilute regime should give near-zero drag (Wen-Yu)."""
        model = DenseParticleFluid()
        alpha_p = torch.full((5,), 0.1, dtype=torch.float64)  # dilute: alpha < 0.2
        U_slip = torch.zeros(5, 3, dtype=torch.float64)
        beta = model.drag_coefficient(alpha_p, U_slip)
        # At zero slip, Re_p ~ 0 => Wen-Yu gives near-zero beta
        assert (beta < 1e-5).all()

    def test_dilute_vs_dense_blend(self):
        """Drag should be higher in dense regime (Ergun) than dilute (Wen-Yu)."""
        model = DenseParticleFluid()
        U_slip = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        alpha_dilute = torch.tensor([0.1], dtype=torch.float64)
        alpha_dense = torch.tensor([0.5], dtype=torch.float64)
        beta_dilute = model.drag_coefficient(alpha_dilute, U_slip)
        beta_dense = model.drag_coefficient(alpha_dense, U_slip)
        assert beta_dense[0] > beta_dilute[0]

    def test_finite_output(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(20, dtype=torch.float64).clamp(0.01, 0.6)
        U_slip = torch.randn(20, 3, dtype=torch.float64)
        beta = model.drag_coefficient(alpha_p, U_slip)
        assert torch.isfinite(beta).all()


class TestDragForce:
    """Drag force computation tests."""

    def test_shape(self):
        model = DenseParticleFluid()
        alpha_p = torch.full((10,), 0.3, dtype=torch.float64)
        U_p = torch.zeros(10, 3, dtype=torch.float64)
        U_f = torch.ones(10, 3, dtype=torch.float64) * 0.5
        F = model.compute_drag_force(alpha_p, U_p, U_f)
        assert F.shape == (10, 3)

    def test_direction_follows_slip(self):
        """Drag force direction should match slip velocity direction."""
        model = DenseParticleFluid()
        alpha_p = torch.full((5,), 0.2, dtype=torch.float64)
        U_p = torch.zeros(5, 3, dtype=torch.float64)
        U_f = torch.zeros(5, 3, dtype=torch.float64)
        U_f[:, 2] = 1.0  # fluid moving upward
        F = model.compute_drag_force(alpha_p, U_p, U_f)
        # Drag on particle should be in fluid direction (upward)
        assert (F[:, 2] > 0).all()

    def test_zero_slip_zero_force(self):
        model = DenseParticleFluid()
        alpha_p = torch.full((5,), 0.1, dtype=torch.float64)
        U = torch.randn(5, 3, dtype=torch.float64)
        F = model.compute_drag_force(alpha_p, U, U)
        # Very small because beta approaches 0 at zero slip
        assert F.norm() < 1e-10


class TestSolidsPressure:
    """Solids pressure tests."""

    def test_shape(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(10, dtype=torch.float64) * 0.5
        p_s = model.solids_pressure(alpha_p)
        assert p_s.shape == (10,)

    def test_positive(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(20, dtype=torch.float64) * 0.5
        p_s = model.solids_pressure(alpha_p)
        assert (p_s >= 0).all()

    def test_increases_near_packing(self):
        """Solids pressure should increase steeply near alpha_max."""
        model = DenseParticleFluid(alpha_max=0.63)
        alpha_low = torch.tensor([0.1], dtype=torch.float64)
        alpha_high = torch.tensor([0.6], dtype=torch.float64)
        p_low = model.solids_pressure(alpha_low)
        p_high = model.solids_pressure(alpha_high)
        assert p_high[0] > p_low[0] * 10  # much higher near packing

    def test_zero_at_zero_alpha(self):
        model = DenseParticleFluid()
        alpha_p = torch.zeros(5, dtype=torch.float64)
        p_s = model.solids_pressure(alpha_p)
        assert torch.allclose(p_s, torch.zeros_like(p_s))

    def test_finite(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(20, dtype=torch.float64) * 0.6
        p_s = model.solids_pressure(alpha_p)
        assert torch.isfinite(p_s).all()


class TestRadialDistribution:
    """Radial distribution function tests."""

    def test_shape(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(10, dtype=torch.float64) * 0.5
        g_0 = model.radial_distribution(alpha_p)
        assert g_0.shape == (10,)

    def test_g0_at_zero_alpha(self):
        """g_0 should be 1.0 at alpha=0 (Carnahan-Starling limit)."""
        model = DenseParticleFluid()
        alpha_p = torch.zeros(5, dtype=torch.float64)
        g_0 = model.radial_distribution(alpha_p)
        # g_0(0) = (2 - 0) / (2 * 1^3) = 1.0
        assert torch.allclose(g_0, torch.ones(5, dtype=torch.float64))

    def test_g0_increases_with_alpha(self):
        """g_0 increases as particles get closer together."""
        model = DenseParticleFluid()
        alpha_low = torch.tensor([0.01], dtype=torch.float64)
        alpha_high = torch.tensor([0.5], dtype=torch.float64)
        g_low = model.radial_distribution(alpha_low)
        g_high = model.radial_distribution(alpha_high)
        assert g_high[0] > g_low[0]

    def test_finite(self):
        model = DenseParticleFluid()
        alpha_p = torch.rand(20, dtype=torch.float64) * 0.6
        g_0 = model.radial_distribution(alpha_p)
        assert torch.isfinite(g_0).all()


class TestPackingCorrection:
    """Packing correction tests."""

    def test_clamps_below_zero(self):
        model = DenseParticleFluid()
        alpha = torch.tensor([-0.1, 0.3], dtype=torch.float64)
        corrected = model.correct_packing(alpha)
        assert corrected[0] == pytest.approx(0.0)
        assert corrected[1] == pytest.approx(0.3)

    def test_clamps_above_alpha_max(self):
        model = DenseParticleFluid(alpha_max=0.63)
        alpha = torch.tensor([0.7], dtype=torch.float64)
        corrected = model.correct_packing(alpha)
        assert corrected[0] == pytest.approx(0.63)


class TestSolveTimestep:
    """Convenience solve_timestep tests."""

    def test_returns_all_keys(self):
        model = DenseParticleFluid()
        alpha = torch.rand(10, dtype=torch.float64).clamp(0.01, 0.5)
        U_p = torch.zeros(10, 3, dtype=torch.float64)
        U_f = torch.ones(10, 3, dtype=torch.float64) * 0.5
        result = model.solve_timestep(alpha, U_p, U_f)
        assert set(result.keys()) == {"F_drag", "p_s", "g_0", "alpha_corrected"}

    def test_all_finite(self):
        model = DenseParticleFluid()
        alpha = torch.rand(20, dtype=torch.float64).clamp(0.01, 0.5)
        U_p = torch.randn(20, 3, dtype=torch.float64) * 0.1
        U_f = torch.randn(20, 3, dtype=torch.float64) * 0.1
        result = model.solve_timestep(alpha, U_p, U_f)
        for v in result.values():
            assert torch.isfinite(v).all()
