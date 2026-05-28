"""Tests for compressibility corrections (Sarkar, Zeman).

Tests cover:
- RTS registration and factory creation
- SarkarModel: dissipation correction, turbulent Mach number, constants
- ZemanModel: threshold behaviour, dissipation correction, constants
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestCompressibilityCorrectionRegistry:
    """Tests for RTS registry."""

    def test_sarkar_registered(self):
        from pyfoam.turbulence.compressibility_corrections import CompressibilityCorrection

        assert "Sarkar" in CompressibilityCorrection.available_types()

    def test_zeman_registered(self):
        from pyfoam.turbulence.compressibility_corrections import CompressibilityCorrection

        assert "Zeman" in CompressibilityCorrection.available_types()

    def test_factory_sarkar(self):
        from pyfoam.turbulence.compressibility_corrections import (
            CompressibilityCorrection,
            SarkarModel,
        )

        model = CompressibilityCorrection.create("Sarkar")
        assert isinstance(model, SarkarModel)

    def test_factory_zeman(self):
        from pyfoam.turbulence.compressibility_corrections import (
            CompressibilityCorrection,
            ZemanModel,
        )

        model = CompressibilityCorrection.create("Zeman")
        assert isinstance(model, ZemanModel)

    def test_factory_unknown_raises(self):
        from pyfoam.turbulence.compressibility_corrections import CompressibilityCorrection

        with pytest.raises(KeyError, match="Unknown compressibility correction"):
            CompressibilityCorrection.create("NonExistent")

    def test_duplicate_registration_raises(self):
        from pyfoam.turbulence.compressibility_corrections import CompressibilityCorrection

        with pytest.raises(ValueError, match="already registered"):
            CompressibilityCorrection.register("Sarkar")(
                type("Dup", (CompressibilityCorrection,), {
                    "correct_dissipation": lambda self, *a: None,
                    "turbulent_mach_number": lambda self, *a: None,
                })
            )


class TestSarkarModel:
    """Tests for Sarkar compressibility correction."""

    def test_init(self):
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel(alpha_1=1.0, alpha_2=0.5)
        assert model.alpha_1 == 1.0
        assert model.alpha_2 == 0.5

    def test_default_constants(self):
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel()
        assert model.alpha_1 == 1.0
        assert model.alpha_2 == 0.5

    def test_turbulent_mach_number(self):
        """M_t = sqrt(2k) / a."""
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel()
        k = torch.tensor([100.0], dtype=CFD_DTYPE)
        a = torch.tensor([340.0], dtype=CFD_DTYPE)

        M_t = model.turbulent_mach_number(k, a)
        import math
        expected = math.sqrt(200.0) / 340.0
        assert torch.allclose(M_t, torch.tensor([expected], dtype=CFD_DTYPE), atol=1e-10)

    def test_correction_increases_dissipation(self):
        """Compressibility correction should increase effective dissipation."""
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel()
        n = 10
        epsilon = torch.full((n,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((n,), 50.0, dtype=CFD_DTYPE)
        nut = torch.full((n,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((n,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        # Correction adds positive terms -> eps_corrected >= epsilon
        assert (eps_corrected >= epsilon).all()

    def test_low_k_no_correction(self):
        """Very low k -> negligible correction."""
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel()
        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((10,), 1e-10, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        assert torch.allclose(eps_corrected, epsilon, atol=1e-6)

    def test_finite_output(self):
        """Output is always finite."""
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel()
        epsilon = torch.rand(20, dtype=CFD_DTYPE).clamp(0.1, 100.0)
        k = torch.rand(20, dtype=CFD_DTYPE).clamp(0.1, 100.0)
        nut = torch.rand(20, dtype=CFD_DTYPE).clamp(1e-5, 1.0)
        rho = torch.full((20,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        assert torch.isfinite(eps_corrected).all()

    def test_custom_constants(self):
        """Custom constants scale the correction."""
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model_weak = SarkarModel(alpha_1=0.1, alpha_2=0.01)
        model_strong = SarkarModel(alpha_1=5.0, alpha_2=2.0)

        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((10,), 100.0, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        eps_weak = model_weak.correct_dissipation(epsilon, k, nut, rho)
        eps_strong = model_strong.correct_dissipation(epsilon, k, nut, rho)

        # Stronger model -> more correction
        assert (eps_strong > eps_weak).all()

    def test_repr(self):
        from pyfoam.turbulence.compressibility_corrections import SarkarModel

        model = SarkarModel(alpha_1=1.0, alpha_2=0.5)
        r = repr(model)
        assert "SarkarModel" in r
        assert "1.0" in r
        assert "0.5" in r


class TestZemanModel:
    """Tests for Zeman compressibility correction."""

    def test_init(self):
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel(alpha_3=0.75, M_t0=0.1)
        assert model.alpha_3 == 0.75
        assert model.M_t0 == 0.1

    def test_default_constants(self):
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel()
        assert model.alpha_3 == 0.75
        assert model.M_t0 == 0.1

    def test_turbulent_mach_number(self):
        """M_t = sqrt(2k) / a."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel()
        k = torch.tensor([50.0], dtype=CFD_DTYPE)
        a = torch.tensor([340.0], dtype=CFD_DTYPE)

        M_t = model.turbulent_mach_number(k, a)
        import math
        expected = math.sqrt(100.0) / 340.0
        assert torch.allclose(M_t, torch.tensor([expected], dtype=CFD_DTYPE), atol=1e-10)

    def test_below_threshold_no_correction(self):
        """Below M_t0 -> no correction (excess clamped to zero)."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel(M_t0=0.5)
        # Very low k -> M_t << M_t0
        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((10,), 1e-6, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        assert torch.allclose(eps_corrected, epsilon, atol=1e-6)

    def test_above_threshold_correction_applied(self):
        """Above M_t0 -> correction applied."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel(M_t0=0.01)  # Very low threshold
        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((10,), 100.0, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        # M_t = sqrt(200)/340 ≈ 0.0416 > 0.01 -> correction applied
        assert (eps_corrected >= epsilon).all()

    def test_correction_increases_with_k(self):
        """Higher k -> higher turbulent Mach -> more correction."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel(M_t0=0.01)
        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        k_low = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k_high = torch.full((10,), 200.0, dtype=CFD_DTYPE)

        eps_low = model.correct_dissipation(epsilon, k_low, nut, rho)
        eps_high = model.correct_dissipation(epsilon, k_high, nut, rho)

        # Higher k -> more correction
        assert (eps_high >= eps_low).all()

    def test_finite_output(self):
        """Output is always finite."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel()
        epsilon = torch.rand(20, dtype=CFD_DTYPE).clamp(0.1, 100.0)
        k = torch.rand(20, dtype=CFD_DTYPE).clamp(0.001, 100.0)
        nut = torch.rand(20, dtype=CFD_DTYPE).clamp(1e-5, 1.0)
        rho = torch.full((20,), 1.225, dtype=CFD_DTYPE)

        eps_corrected = model.correct_dissipation(epsilon, k, nut, rho)
        assert torch.isfinite(eps_corrected).all()

    def test_custom_constants(self):
        """Custom constants scale the correction."""
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model_weak = ZemanModel(alpha_3=0.1, M_t0=0.5)
        model_strong = ZemanModel(alpha_3=5.0, M_t0=0.01)

        epsilon = torch.full((10,), 10.0, dtype=CFD_DTYPE)
        k = torch.full((10,), 100.0, dtype=CFD_DTYPE)
        nut = torch.full((10,), 0.01, dtype=CFD_DTYPE)
        rho = torch.full((10,), 1.225, dtype=CFD_DTYPE)

        eps_weak = model_weak.correct_dissipation(epsilon, k, nut, rho)
        eps_strong = model_strong.correct_dissipation(epsilon, k, nut, rho)

        assert (eps_strong >= eps_weak).all()

    def test_repr(self):
        from pyfoam.turbulence.compressibility_corrections import ZemanModel

        model = ZemanModel(alpha_3=0.75, M_t0=0.1)
        r = repr(model)
        assert "ZemanModel" in r
        assert "0.75" in r
        assert "0.1" in r

    def test_export_availability(self):
        """Models are importable from turbulence."""
        from pyfoam.turbulence import (
            CompressibilityCorrection,
            SarkarModel,
            ZemanModel,
        )

        assert CompressibilityCorrection is not None
        assert SarkarModel is not None
        assert ZemanModel is not None
