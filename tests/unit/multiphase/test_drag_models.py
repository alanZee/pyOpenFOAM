"""Tests for drag model ABC hierarchy."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.multiphase.drag_models import (
    DragModel,
    SchillerNaumannDrag,
    WenYuDrag,
    GidaspowDrag,
)


# ============================================================================
# DragModel ABC
# ============================================================================


class TestDragModelABC:
    """DragModel 抽象基类测试。"""

    def test_rts_registry_contains_models(self):
        """三个拖曳模型已注册。"""
        types = DragModel.available_types()
        assert "schillerNaumann" in types
        assert "wenYu" in types
        assert "gidaspow" in types

    def test_factory_create_schiller_naumann(self):
        model = DragModel.create("schillerNaumann", d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert isinstance(model, SchillerNaumannDrag)

    def test_factory_create_wen_yu(self):
        model = DragModel.create("wenYu", d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert isinstance(model, WenYuDrag)

    def test_factory_create_gidaspow(self):
        model = DragModel.create("gidaspow", d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert isinstance(model, GidaspowDrag)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown drag model"):
            DragModel.create("nonexistent", d=1e-3, rho_c=1.225, mu_c=1.8e-5)

    def test_is_abstract(self):
        """DragModel 不能直接实例化。"""
        with pytest.raises(TypeError):
            DragModel(d=1e-3, rho_c=1.225, mu_c=1.8e-5)


# ============================================================================
# SchillerNaumannDrag (from drag_models)
# ============================================================================


class TestSchillerNaumannDrag:
    """Schiller-Naumann 拖曳模型测试。"""

    def test_init(self):
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert drag.d == 1e-3
        assert drag.rho_c == 1.225
        assert drag.mu_c == 1.8e-5

    def test_compute_returns_positive(self):
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()

    def test_compute_shape(self):
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((5,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        assert K.shape == (5,)

    def test_drag_increases_with_velocity(self):
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        K_low = drag.compute(alpha, torch.full((10,), 0.01, dtype=CFD_DTYPE))
        K_high = drag.compute(alpha, torch.full((10,), 1.0, dtype=CFD_DTYPE))
        assert K_high.mean() > K_low.mean()

    def test_drag_increases_with_alpha(self):
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K_low = drag.compute(torch.full((10,), 0.1, dtype=CFD_DTYPE), U_rel)
        K_high = drag.compute(torch.full((10,), 0.5, dtype=CFD_DTYPE), U_rel)
        assert K_high.mean() > K_low.mean()

    def test_zero_velocity(self):
        """零相对速度时 Re -> 0，Cd -> 24/Re（大但有限）。"""
        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((5,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.zeros(5, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        assert torch.isfinite(K).all()


# ============================================================================
# WenYuDrag (from drag_models)
# ============================================================================


class TestWenYuDrag:
    """Wen-Yu 拖曳模型测试。"""

    def test_init(self):
        drag = WenYuDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert drag.d == 1e-3

    def test_compute_returns_positive(self):
        drag = WenYuDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()

    def test_dilute_correction(self):
        """Wen-Yu 的 void-fraction 校正使 K > Schiller-Naumann（低 alpha）。"""
        sn = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        wy = WenYuDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.05, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K_sn = sn.compute(alpha, U_rel)
        K_wy = wy.compute(alpha, U_rel)
        # Wen-Yu 有 (1-alpha)^(-2.65) 校正因子 > 1
        assert K_wy.mean() > K_sn.mean()


# ============================================================================
# GidaspowDrag (from drag_models)
# ============================================================================


class TestGidaspowDrag:
    """Gidaspow 拖曳模型测试。"""

    def test_init(self):
        drag = GidaspowDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert drag.d == 1e-3

    def test_compute_returns_positive(self):
        drag = GidaspowDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()

    def test_dilute_uses_wen_yu(self):
        """低 alpha（alpha_c > 0.8）时使用 Wen-Yu 模型。"""
        gidaspow = GidaspowDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        wy = WenYuDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)  # alpha_c = 0.9 > 0.8
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K_gid = gidaspow.compute(alpha, U_rel)
        K_wy = wy.compute(alpha, U_rel)
        assert torch.allclose(K_gid, K_wy, rtol=1e-10)

    def test_dense_uses_ergun(self):
        """高 alpha（alpha_c < 0.8）时使用 Ergun 方程。"""
        drag = GidaspowDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.5, dtype=CFD_DTYPE)  # alpha_c = 0.5 < 0.8
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        K = drag.compute(alpha, U_rel)
        # Ergun: 150*alpha*mu/(d^2*alpha_c) + 1.75*rho*U/d*alpha/alpha_c
        alpha_c = 0.5
        K_ergun_expected = (
            150.0 * 0.5 * 1.8e-5 / (1e-3 ** 2 * alpha_c)
            + 1.75 * 1.225 * 0.1 / 1e-3 * 0.5 / alpha_c
        )
        assert torch.allclose(
            K, torch.full((10,), K_ergun_expected, dtype=CFD_DTYPE), rtol=1e-6,
        )
