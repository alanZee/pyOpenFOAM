"""Tests for enhanced cavitation models (ZGBModel, MerkleModel)."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestZGBModel:
    """Tests for enhanced ZGB cavitation model."""

    def test_init_defaults(self):
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel()
        assert model.base.C_evap == pytest.approx(0.02)
        assert model.base.C_cond == pytest.approx(0.01)
        assert model.relaxation == pytest.approx(1.0)
        assert model.alpha_clip == pytest.approx(1e-6)

    def test_init_custom(self):
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel(C_evap=0.05, relaxation=0.3, alpha_clip=1e-8)
        assert model.base.C_evap == pytest.approx(0.05)
        assert model.relaxation == pytest.approx(0.3)

    def test_evaporation_at_low_pressure(self):
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 1000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot >= 0).all(), "Evaporation should be positive"

    def test_condensation_at_high_pressure(self):
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 5000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot <= 0).all(), "Condensation should be negative"

    def test_pressure_limiting(self):
        """Pressure clipping prevents extreme mass transfer values."""
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model_clipped = ZGBModel(p_v=2300.0, p_clip=1e3)
        model_noclip = ZGBModel(p_v=2300.0, p_clip=1e10)

        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 1e6, dtype=CFD_DTYPE)  # very high

        m_clipped = model_clipped.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        m_noclip = model_noclip.compute_mass_transfer(alpha, p, 1000.0, 0.02)

        # Clipped should be smaller in magnitude
        assert m_clipped.abs().mean() <= m_noclip.abs().mean()

    def test_finite_output(self):
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel()
        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        p = torch.randn(20, dtype=CFD_DTYPE) * 1000 + 2300

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_relax_no_relaxation(self):
        """With relaxation=1.0, relax returns m_dot_new."""
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel(relaxation=1.0)
        m_new = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        m_old = torch.zeros(3, dtype=CFD_DTYPE)

        result = model.relax(m_new, m_old)
        assert torch.allclose(result, m_new)

    def test_relax_with_factor(self):
        """With relaxation=0.5, result is average of new and old."""
        from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel

        model = ZGBModel(relaxation=0.5)
        m_new = torch.tensor([2.0, 4.0, 6.0], dtype=CFD_DTYPE)
        m_old = torch.tensor([0.0, 0.0, 0.0], dtype=CFD_DTYPE)

        result = model.relax(m_new, m_old)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)


class TestMerkleModel:
    """Tests for enhanced Merkle cavitation model."""

    def test_init_defaults(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel()
        assert model.base.C_evap == pytest.approx(1.0)
        assert model.base.C_cond == pytest.approx(1.0)
        assert model.relaxation == pytest.approx(1.0)

    def test_init_custom(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel(C_evap=2.0, relaxation=0.7)
        assert model.base.C_evap == pytest.approx(2.0)
        assert model.relaxation == pytest.approx(0.7)

    def test_evaporation_at_low_pressure(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 1000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot >= 0).all(), "Evaporation should be positive"

    def test_condensation_at_high_pressure(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 5000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot <= 0).all(), "Condensation should be negative"

    def test_pressure_limiting(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model_clipped = MerkleModel(p_v=2300.0, p_clip=1e3)
        model_noclip = MerkleModel(p_v=2300.0, p_clip=1e10)

        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 1e6, dtype=CFD_DTYPE)

        m_clipped = model_clipped.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        m_noclip = model_noclip.compute_mass_transfer(alpha, p, 1000.0, 0.02)

        assert m_clipped.abs().mean() <= m_noclip.abs().mean()

    def test_finite_output(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel()
        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        p = torch.randn(20, dtype=CFD_DTYPE) * 1000 + 2300

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_relax(self):
        from pyfoam.multiphase.cavitation_models_enhanced import MerkleModel

        model = MerkleModel(relaxation=0.3)
        m_new = torch.tensor([10.0, 20.0], dtype=CFD_DTYPE)
        m_old = torch.tensor([0.0, 0.0], dtype=CFD_DTYPE)

        result = model.relax(m_new, m_old)
        expected = torch.tensor([3.0, 6.0], dtype=CFD_DTYPE)
        assert torch.allclose(result, expected)
