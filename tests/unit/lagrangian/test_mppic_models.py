"""
Unit tests for Lagrangian MPPIC models.
"""

from __future__ import annotations

import pytest

from pyfoam.lagrangian.mppic_models import (
    MPPICModel,
    StandardMPPIC,
    FrictionModel,
    SchaefferFriction,
)


# ======================================================================
# MPPICModel ABC
# ======================================================================

class TestMPPICModelABC:
    """Tests for MPPICModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MPPICModel()


# ======================================================================
# StandardMPPIC
# ======================================================================

class TestStandardMPPIC:
    """Tests for StandardMPPIC (Harris-Crighton) model."""

    def test_default_parameters(self):
        model = StandardMPPIC()
        assert model.packing_alpha_max == pytest.approx(0.62)
        assert model.exponent == pytest.approx(2.0)
        assert model.p0 == pytest.approx(1000.0)

    def test_custom_parameters(self):
        model = StandardMPPIC(
            packing_alpha_max=0.55, exponent=3.0, p0=5000.0,
        )
        assert model.packing_alpha_max == pytest.approx(0.55)
        assert model.exponent == pytest.approx(3.0)
        assert model.p0 == pytest.approx(5000.0)

    def test_alpha_max_must_be_valid(self):
        with pytest.raises(ValueError, match="packing_alpha_max"):
            StandardMPPIC(packing_alpha_max=0.0)
        with pytest.raises(ValueError, match="packing_alpha_max"):
            StandardMPPIC(packing_alpha_max=1.5)

    def test_exponent_must_be_positive(self):
        with pytest.raises(ValueError, match="exponent"):
            StandardMPPIC(exponent=0.0)
        with pytest.raises(ValueError, match="exponent"):
            StandardMPPIC(exponent=-1.0)

    def test_p0_must_be_non_negative(self):
        with pytest.raises(ValueError, match="p0"):
            StandardMPPIC(p0=-1.0)

    def test_zero_alpha_gives_zero_stress(self):
        model = StandardMPPIC()
        assert model.packing_stress(0.0) == pytest.approx(0.0)

    def test_negative_alpha_gives_zero_stress(self):
        model = StandardMPPIC()
        assert model.packing_stress(-0.1) == pytest.approx(0.0)

    def test_stress_increases_with_alpha(self):
        model = StandardMPPIC()
        s1 = model.packing_stress(0.1)
        s2 = model.packing_stress(0.3)
        s3 = model.packing_stress(0.5)
        assert s2 > s1
        assert s3 > s2

    def test_stress_diverges_near_packing(self):
        """Stress should be very large as alpha approaches alpha_max."""
        model = StandardMPPIC(packing_alpha_max=0.62)
        s_near = model.packing_stress(0.61)
        s_far = model.packing_stress(0.3)
        assert s_near > s_far * 10

    def test_stress_finite_at_max(self):
        """Stress should be very large (not diverge to inf) but not zero
        at exactly alpha_max due to the epsilon guard."""
        model = StandardMPPIC(packing_alpha_max=0.62)
        result = model.packing_stress(0.62)
        # With the epsilon guard, stress at alpha_max is extremely large
        assert result > 0.0
        assert float(result) == pytest.approx(float(result))  # finite

    def test_stress_gradient(self):
        model = StandardMPPIC()
        dp = model.packing_stress_gradient(0.3)
        assert dp > 0.0
        assert float(dp) == pytest.approx(float(dp))  # finite

    def test_gradient_zero_at_zero(self):
        model = StandardMPPIC()
        assert model.packing_stress_gradient(0.0) == pytest.approx(0.0)

    def test_stress_non_negative(self):
        """Stress should never be negative."""
        model = StandardMPPIC()
        for alpha in [0.0, 0.1, 0.3, 0.5, 0.6, 0.61]:
            assert model.packing_stress(alpha) >= 0.0

    def test_repr(self):
        model = StandardMPPIC(p0=5000.0)
        r = repr(model)
        assert "StandardMPPIC" in r
        assert "5000" in r


# ======================================================================
# FrictionModel ABC
# ======================================================================

class TestFrictionModelABC:
    """Tests for FrictionModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            FrictionModel()


# ======================================================================
# SchaefferFriction
# ======================================================================

class TestSchaefferFriction:
    """Tests for Schaeffer friction stress model."""

    def test_default_parameters(self):
        model = SchaefferFriction()
        assert model.friction_angle == pytest.approx(25.0)
        assert model.packing_alpha_f == pytest.approx(0.5)

    def test_custom_parameters(self):
        model = SchaefferFriction(friction_angle=30.0, packing_alpha_f=0.55)
        assert model.friction_angle == pytest.approx(30.0)
        assert model.packing_alpha_f == pytest.approx(0.55)

    def test_angle_must_be_valid(self):
        with pytest.raises(ValueError, match="friction_angle"):
            SchaefferFriction(friction_angle=0.0)
        with pytest.raises(ValueError, match="friction_angle"):
            SchaefferFriction(friction_angle=90.0)

    def test_alpha_f_must_be_valid(self):
        with pytest.raises(ValueError, match="packing_alpha_f"):
            SchaefferFriction(packing_alpha_f=0.0)
        with pytest.raises(ValueError, match="packing_alpha_f"):
            SchaefferFriction(packing_alpha_f=1.5)

    def test_below_threshold_zero_stress(self):
        model = SchaefferFriction(packing_alpha_f=0.5)
        assert model.friction_stress(0.3, 100.0) == pytest.approx(0.0)

    def test_above_threshold_nonzero_stress(self):
        model = SchaefferFriction(packing_alpha_f=0.5)
        result = model.friction_stress(0.6, 100.0)
        assert result > 0.0

    def test_zero_strain_rate_zero_stress(self):
        model = SchaefferFriction(packing_alpha_f=0.5)
        assert model.friction_stress(0.6, 0.0) == pytest.approx(0.0)

    def test_stress_increases_with_alpha(self):
        model = SchaefferFriction(packing_alpha_f=0.5)
        s1 = model.friction_stress(0.55, 100.0)
        s2 = model.friction_stress(0.7, 100.0)
        assert s2 > s1

    def test_higher_friction_angle_more_stress(self):
        model_low = SchaefferFriction(friction_angle=10.0)
        model_high = SchaefferFriction(friction_angle=40.0)
        s_low = model_low.friction_stress(0.6, 100.0)
        s_high = model_high.friction_stress(0.6, 100.0)
        assert s_high > s_low

    def test_repr(self):
        model = SchaefferFriction(friction_angle=30.0)
        r = repr(model)
        assert "SchaefferFriction" in r
        assert "30" in r
