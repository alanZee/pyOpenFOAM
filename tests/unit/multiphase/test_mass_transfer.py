"""Tests for mass transfer models (LeeMassTransfer, ThermalPhaseChange).

Tests cover:
- RTS registration and factory creation
- LeeMassTransfer: evaporation/condensation, equilibrium, rate controls
- ThermalPhaseChange: interface heat transfer, masking, under-relaxation
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestMassTransferRegistry:
    """Tests for RTS registry."""

    def test_lee_registered(self):
        from pyfoam.multiphase.mass_transfer import MassTransferModel

        assert "Lee" in MassTransferModel.available_types()

    def test_thermal_phase_change_registered(self):
        from pyfoam.multiphase.mass_transfer import MassTransferModel

        assert "ThermalPhaseChange" in MassTransferModel.available_types()

    def test_factory_lee(self):
        from pyfoam.multiphase.mass_transfer import MassTransferModel, LeeMassTransfer

        model = MassTransferModel.create("Lee", T_sat=373.15)
        assert isinstance(model, LeeMassTransfer)

    def test_factory_thermal_phase_change(self):
        from pyfoam.multiphase.mass_transfer import (
            MassTransferModel,
            ThermalPhaseChange,
        )

        model = MassTransferModel.create("ThermalPhaseChange")
        assert isinstance(model, ThermalPhaseChange)

    def test_factory_unknown_raises(self):
        from pyfoam.multiphase.mass_transfer import MassTransferModel

        with pytest.raises(KeyError, match="Unknown mass transfer model"):
            MassTransferModel.create("NonExistent")

    def test_duplicate_registration_raises(self):
        from pyfoam.multiphase.mass_transfer import MassTransferModel

        with pytest.raises(ValueError, match="already registered"):
            MassTransferModel.register("Lee")(
                type("Dup", (MassTransferModel,), {
                    "compute": lambda self, *a: None,
                })
            )


class TestLeeMassTransfer:
    """Tests for Lee mass transfer model."""

    def test_init(self):
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15, r_evap=0.1, r_cond=0.2)
        assert model.T_sat == 373.15
        assert model.r_evap == 0.1
        assert model.r_cond == 0.2

    def test_evaporation_above_saturation(self):
        """T > T_sat: positive mass transfer (evaporation)."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_dot > 0).all(), "Evaporation: T > T_sat -> m_dot > 0"

    def test_condensation_below_saturation(self):
        """T < T_sat: negative mass transfer (condensation)."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 350.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_dot < 0).all(), "Condensation: T < T_sat -> m_dot < 0"

    def test_equilibrium_at_saturation(self):
        """T = T_sat: zero mass transfer."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 373.15, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert torch.allclose(m_dot, torch.zeros_like(m_dot), atol=1e-20)

    def test_rate_increases_with_superheat(self):
        """Larger superheat -> larger evaporation rate."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        T_low = torch.full((10,), 380.0, dtype=CFD_DTYPE)
        T_high = torch.full((10,), 400.0, dtype=CFD_DTYPE)

        m_low = model.compute(alpha, T_low, p, 1000.0, 0.6)
        m_high = model.compute(alpha, T_high, p, 1000.0, 0.6)
        assert (m_high > m_low).all()

    def test_custom_rates(self):
        """Custom r_evap/r_cond scale the result."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model_slow = LeeMassTransfer(T_sat=373.15, r_evap=0.01)
        model_fast = LeeMassTransfer(T_sat=373.15, r_evap=1.0)

        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((10,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        m_slow = model_slow.compute(alpha, T, p, 1000.0, 0.6)
        m_fast = model_fast.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_fast > m_slow).all()

    def test_finite_output(self):
        """Output is always finite."""
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=373.15)
        alpha = torch.rand(20, dtype=CFD_DTYPE)
        T = torch.randn(20, dtype=CFD_DTYPE) * 50 + 373.15
        p = torch.full((20,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert torch.isfinite(m_dot).all()

    def test_repr(self):
        from pyfoam.multiphase.mass_transfer import LeeMassTransfer

        model = LeeMassTransfer(T_sat=400.0)
        r = repr(model)
        assert "LeeMassTransfer" in r
        assert "400" in r


class TestThermalPhaseChange:
    """Tests for thermal phase change model."""

    def test_init(self):
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15, h_lv=2257e3, k_eff=0.6)
        assert model.T_sat == 373.15
        assert model.h_lv == pytest.approx(2257e3)
        assert model.k_eff == 0.6

    def test_evaporation_above_saturation(self):
        """T > T_sat at interface: positive mass transfer."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15, h_lv=2257e3, k_eff=0.6)
        n = 10
        alpha = torch.full((n,), 0.5, dtype=CFD_DTYPE)  # at interface
        T = torch.full((n,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_dot > 0).all(), "T > T_sat at interface -> evaporation"

    def test_condensation_below_saturation(self):
        """T < T_sat at interface: negative mass transfer."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15, h_lv=2257e3, k_eff=0.6)
        n = 10
        alpha = torch.full((n,), 0.5, dtype=CFD_DTYPE)
        T = torch.full((n,), 350.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_dot < 0).all(), "T < T_sat at interface -> condensation"

    def test_interface_masking(self):
        """Only cells at the interface (0 < alpha < 1) should have non-zero m_dot."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15)
        n = 5
        # Cell 0: pure liquid (alpha=0), Cell 4: pure vapour (alpha=1)
        alpha = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=CFD_DTYPE)
        T = torch.full((n,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        # Cells 0 and 4 (pure phases) should have zero m_dot
        assert m_dot[0].abs() < 1e-20
        assert m_dot[4].abs() < 1e-20
        # Cells 1, 2, 3 (interface) should have non-zero m_dot
        assert m_dot[1].abs() > 0
        assert m_dot[2].abs() > 0
        assert m_dot[3].abs() > 0

    def test_rate_increases_with_conductivity(self):
        """Higher conductivity -> higher mass transfer rate."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model_low = ThermalPhaseChange(T_sat=373.15, k_eff=0.1)
        model_high = ThermalPhaseChange(T_sat=373.15, k_eff=1.0)

        alpha = torch.full((10,), 0.5, dtype=CFD_DTYPE)
        T = torch.full((10,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        m_low = model_low.compute(alpha, T, p, 1000.0, 0.6)
        m_high = model_high.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_high.abs() > m_low.abs()).all()

    def test_rate_inversely_proportional_to_latent_heat(self):
        """Higher latent heat -> lower mass transfer rate."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model_low_hlv = ThermalPhaseChange(T_sat=373.15, h_lv=1e3)
        model_high_hlv = ThermalPhaseChange(T_sat=373.15, h_lv=1e6)

        alpha = torch.full((10,), 0.5, dtype=CFD_DTYPE)
        T = torch.full((10,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        m_low = model_low_hlv.compute(alpha, T, p, 1000.0, 0.6)
        m_high = model_high_hlv.compute(alpha, T, p, 1000.0, 0.6)
        assert (m_low.abs() > m_high.abs()).all()

    def test_under_relaxation(self):
        """Under-relaxation should blend new and old values."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15, relaxation=0.5)
        n = 10
        alpha = torch.full((n,), 0.5, dtype=CFD_DTYPE)
        T = torch.full((n,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        # First call (no old value) -> returns new value
        m1 = model.compute(alpha, T, p, 1000.0, 0.6)

        # Second call with same input -> should be relaxed
        m2 = model.compute(alpha, T, p, 1000.0, 0.6)
        # With relaxation=0.5: m2 = 0.5 * m1 + 0.5 * m1 = m1 (same input)
        assert torch.allclose(m2, m1)

    def test_reset(self):
        """reset() clears stored old value."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=373.15, relaxation=0.5)
        alpha = torch.full((10,), 0.5, dtype=CFD_DTYPE)
        T = torch.full((10,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        model.compute(alpha, T, p, 1000.0, 0.6)
        assert model._m_dot_old is not None

        model.reset()
        assert model._m_dot_old is None

    def test_finite_output(self):
        """Output is always finite for random inputs."""
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange()
        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        T = torch.randn(20, dtype=CFD_DTYPE) * 50 + 373.15
        p = torch.full((20,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute(alpha, T, p, 1000.0, 0.6)
        assert torch.isfinite(m_dot).all()

    def test_repr(self):
        from pyfoam.multiphase.mass_transfer import ThermalPhaseChange

        model = ThermalPhaseChange(T_sat=400.0, h_lv=2000e3)
        r = repr(model)
        assert "ThermalPhaseChange" in r
        assert "400" in r
        assert "2000000" in r or "2e+06" in r or "2000000.0" in r

    def test_export_availability(self):
        """Models are importable from multiphase."""
        from pyfoam.multiphase import MassTransferModel, LeeMassTransfer, ThermalPhaseChange

        assert MassTransferModel is not None
        assert LeeMassTransfer is not None
        assert ThermalPhaseChange is not None
