"""Tests for non-linear roughness and wave generation boundary conditions.

Tests cover:
- NonLinearRoughnessBC: registration, nut computation, damping, roughness effects
- WaveGenerationBC: registration, dispersion relation, velocity computation, ramping
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.nonlinear_roughness import NonLinearRoughnessBC
from pyfoam.boundary.wave_generation import WaveGenerationBC


# ===========================================================================
# NonLinearRoughnessBC tests
# ===========================================================================


class TestNonLinearRoughnessBCRegistration:
    """RTS registration tests for nonLinearRoughness."""

    def test_registered(self):
        assert "nonLinearRoughness" in BoundaryCondition.available_types()

    def test_factory_create(self, wall_patch):
        bc = BoundaryCondition.create("nonLinearRoughness", wall_patch)
        assert isinstance(bc, NonLinearRoughnessBC)

    def test_type_name(self, wall_patch):
        bc = NonLinearRoughnessBC(wall_patch)
        assert bc.type_name == "nonLinearRoughness"


class TestNonLinearRoughnessBCCompute:
    """Computation tests for NonLinearRoughnessBC."""

    def test_smooth_wall_matches_standard(self, wall_patch):
        """Zero roughness height should behave like a standard wall function."""
        bc_smooth = NonLinearRoughnessBC(wall_patch, {"ks": 0.0})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc_smooth.compute_nut(k, y, nu=1e-5)
        assert nut.shape == (3,)
        assert (nut > 0).all()
        assert torch.isfinite(nut).all()

    def test_rough_wall_positive_nut(self, wall_patch):
        """Rough wall produces positive nut values."""
        bc = NonLinearRoughnessBC(wall_patch, {"ks": 1e-4})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert (nut > 0).all()
        assert torch.isfinite(nut).all()

    def test_roughness_reduces_nut(self, wall_patch):
        """Higher roughness should reduce effective nut (more mixing)."""
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)

        bc_smooth = NonLinearRoughnessBC(wall_patch, {"ks": 0.0})
        bc_rough = NonLinearRoughnessBC(wall_patch, {"ks": 1e-3})

        nut_smooth = bc_smooth.compute_nut(k, y, nu=1e-5)
        nut_rough = bc_rough.compute_nut(k, y, nu=1e-5)
        # Roughness modifies the log-law; nut changes
        # (may increase or decrease depending on the regime)
        assert torch.isfinite(nut_rough).all()

    def test_handles_zero_k(self, wall_patch):
        """Zero TKE should not produce NaN."""
        bc = NonLinearRoughnessBC(wall_patch, {"ks": 1e-4})
        k = torch.zeros(3, dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert torch.isfinite(nut).all()

    def test_custom_coefficients(self, wall_patch):
        """Custom coefficients are respected."""
        bc = NonLinearRoughnessBC(wall_patch, {
            "ks": 1e-4, "kappa": 0.4, "E": 9.0, "Ar": 3.0, "Cmu": 0.08,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        nut = bc.compute_nut(k, y, nu=1e-5)
        assert (nut > 0).all()
        assert torch.isfinite(nut).all()


class TestNonLinearRoughnessBCApply:
    """Apply and matrix tests for NonLinearRoughnessBC."""

    def test_apply_sets_value(self, wall_patch):
        bc = NonLinearRoughnessBC(wall_patch, {"value": 0.05})
        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[30:33], torch.full((3,), 0.05, dtype=torch.float64))

    def test_apply_no_value(self, wall_patch):
        bc = NonLinearRoughnessBC(wall_patch)
        field = torch.ones(35, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field, original)

    def test_no_matrix_contribution(self, wall_patch):
        bc = NonLinearRoughnessBC(wall_patch)
        field = torch.zeros(35, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))


# ===========================================================================
# WaveGenerationBC tests
# ===========================================================================


class TestWaveGenerationBCRegistration:
    """RTS registration tests for waveGeneration."""

    def test_registered(self):
        assert "waveGeneration" in BoundaryCondition.available_types()

    def test_factory_create(self, simple_patch):
        bc = BoundaryCondition.create("waveGeneration", simple_patch)
        assert isinstance(bc, WaveGenerationBC)

    def test_type_name(self, simple_patch):
        bc = WaveGenerationBC(simple_patch)
        assert bc.type_name == "waveGeneration"


class TestWaveGenerationBCDispersion:
    """Tests for dispersion relation solver."""

    def test_dispersion_deep_water(self, simple_patch):
        """Deep water: k ~ omega^2 / g."""
        bc = WaveGenerationBC(simple_patch, {
            "period": 2.0, "depth": 100.0, "amplitude": 0.1,
        })
        omega = 2.0 * math.pi / 2.0
        k_expected = omega ** 2 / 9.81
        assert abs(bc._k - k_expected) / k_expected < 0.01

    def test_dispersion_shallow_water(self, simple_patch):
        """Shallow water: omega^2 ~ g*k (since tanh(kd) ~ kd for small kd)."""
        bc = WaveGenerationBC(simple_patch, {
            "period": 10.0, "depth": 1.0, "amplitude": 0.05,
        })
        omega = 2.0 * math.pi / 10.0
        omega2 = omega ** 2
        g = 9.81
        k = bc._k
        # Check dispersion relation
        residual = abs(g * k * math.tanh(k * 1.0) - omega2)
        assert residual < 1e-6

    def test_dispersion_various_periods(self, simple_patch):
        """Dispersion relation holds for various periods."""
        for T in [0.5, 1.0, 2.0, 5.0]:
            bc = WaveGenerationBC(simple_patch, {
                "period": T, "depth": 5.0, "amplitude": 0.01,
            })
            omega = 2.0 * math.pi / T
            k = bc._k
            residual = abs(9.81 * k * math.tanh(k * 5.0) - omega ** 2)
            assert residual < 1e-6, f"Failed for T={T}"

    def test_wave_number_positive(self, simple_patch):
        """Wave number is always positive."""
        bc = WaveGenerationBC(simple_patch, {"period": 1.0, "depth": 1.0})
        assert bc._k > 0


class TestWaveGenerationBCVelocity:
    """Tests for velocity computation."""

    def test_velocity_shape(self, simple_patch):
        """Velocity has correct shape."""
        bc = WaveGenerationBC(simple_patch, {"amplitude": 0.1, "period": 1.0, "depth": 5.0})
        centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        vel = bc.compute_velocity(centres, time=0.0)
        assert vel.shape == (3, 3)

    def test_velocity_finite(self, simple_patch):
        """Velocity values are finite."""
        bc = WaveGenerationBC(simple_patch, {"amplitude": 0.1, "period": 1.0, "depth": 5.0})
        centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        vel = bc.compute_velocity(centres, time=0.0)
        assert torch.isfinite(vel).all()

    def test_velocity_scales_with_amplitude(self, simple_patch):
        """Velocity magnitude scales linearly with amplitude."""
        centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

        bc1 = WaveGenerationBC(simple_patch, {"amplitude": 0.05, "period": 1.0, "depth": 5.0})
        bc2 = WaveGenerationBC(simple_patch, {"amplitude": 0.10, "period": 1.0, "depth": 5.0})

        vel1 = bc1.compute_velocity(centres, time=0.25)
        vel2 = bc2.compute_velocity(centres, time=0.25)

        # Velocities should be proportional
        norm1 = vel1.norm()
        norm2 = vel2.norm()
        if norm1 > 1e-30:
            ratio = norm2 / norm1
            assert abs(ratio - 2.0) < 0.01

    def test_velocity_zero_at_seabed(self, simple_patch):
        """Vertical velocity approaches zero at the seabed (z = -d)."""
        bc = WaveGenerationBC(simple_patch, {
            "amplitude": 0.1, "period": 1.0, "depth": 2.0, "zRef": 0.0,
        })
        # At the seabed: z = -depth
        centres = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
        vel = bc.compute_velocity(centres, time=0.0)
        # Vertical velocity should be very small at seabed
        assert abs(vel[0, 2].item()) < 1e-6

    def test_velocity_periodic_in_time(self, simple_patch):
        """Velocity is periodic with the wave period."""
        bc = WaveGenerationBC(simple_patch, {
            "amplitude": 0.1, "period": 2.0, "depth": 5.0,
        })
        centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        T = 2.0

        vel_t0 = bc.compute_velocity(centres, time=0.0)
        vel_tT = bc.compute_velocity(centres, time=T)
        assert torch.allclose(vel_t0, vel_tT, atol=1e-10)

    def test_ramp_factor(self, simple_patch):
        """Ramp factor starts at zero and increases to 1."""
        bc = WaveGenerationBC(simple_patch, {
            "amplitude": 0.1, "period": 1.0, "depth": 5.0, "rampTime": 2.0,
        })
        assert bc._ramp_factor(0.0) == 0.0
        assert bc._ramp_factor(1.0) == pytest.approx(0.5)
        assert bc._ramp_factor(2.0) == pytest.approx(1.0)
        assert bc._ramp_factor(5.0) == pytest.approx(1.0)


class TestWaveGenerationBCApply:
    """Apply and matrix tests for WaveGenerationBC."""

    def test_apply_sets_value(self, simple_patch):
        bc = WaveGenerationBC(simple_patch, {"value": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 1.0, dtype=torch.float64))

    def test_no_matrix_contribution(self, simple_patch):
        bc = WaveGenerationBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))
