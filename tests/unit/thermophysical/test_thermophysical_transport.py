"""
Unit tests for thermophysical transport models.

Covers ThermophysicalTransportModel, FourierTransport, and FickianTransport.
"""

from __future__ import annotations

import math
import pytest
import torch

from pyfoam.thermophysical.thermophysical_transport import (
    ThermophysicalTransportModel,
    FourierTransport,
    FickianTransport,
)


# ======================================================================
# 抽象基类
# ======================================================================

class TestThermophysicalTransportModelABC:
    """Tests for the ThermophysicalTransportModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ThermophysicalTransportModel()


# ======================================================================
# FourierTransport
# ======================================================================

class TestFourierTransport:
    """Tests for FourierTransport (Fourier law: q = -k * grad(T))."""

    def test_default_kappa(self):
        model = FourierTransport()
        assert model.kappa == pytest.approx(0.026)

    def test_custom_kappa(self):
        model = FourierTransport(kappa=0.6)
        assert model.kappa == pytest.approx(0.6)

    def test_flux_list_input(self):
        """Should accept a list and return a tensor."""
        model = FourierTransport(kappa=0.026)
        q = model.flux(grad_X=[100.0, 0.0, 0.0])
        assert isinstance(q, torch.Tensor)
        assert q[0].item() == pytest.approx(-2.6)
        assert q[1].item() == pytest.approx(0.0)
        assert q[2].item() == pytest.approx(0.0)

    def test_flux_tensor_input(self):
        """Should accept a tensor and return a tensor."""
        model = FourierTransport(kappa=0.5)
        grad = torch.tensor([200.0, 100.0, 50.0])
        q = model.flux(grad_X=grad)
        assert q.shape == (3,)
        assert q[0].item() == pytest.approx(-100.0)
        assert q[1].item() == pytest.approx(-50.0)
        assert q[2].item() == pytest.approx(-25.0)

    def test_batch_flux(self):
        """Should work with batched input (n, 3)."""
        model = FourierTransport(kappa=0.026)
        grad = torch.tensor([
            [100.0, 0.0, 0.0],
            [0.0, 200.0, 0.0],
        ])
        q = model.flux(grad_X=grad)
        assert q.shape == (2, 3)
        assert q[0, 0].item() == pytest.approx(-2.6)
        assert q[1, 1].item() == pytest.approx(-5.2)

    def test_negative_gradient_positive_flux_component(self):
        """Negative gradient should give positive flux."""
        model = FourierTransport(kappa=1.0)
        q = model.flux(grad_X=[-10.0, 0.0, 0.0])
        assert q[0].item() == pytest.approx(10.0)

    def test_zero_gradient_zero_flux(self):
        model = FourierTransport(kappa=0.026)
        q = model.flux(grad_X=[0.0, 0.0, 0.0])
        assert torch.allclose(q, torch.zeros(3, dtype=q.dtype))

    def test_flux_linearity(self):
        """Flux should scale linearly with gradient."""
        model = FourierTransport(kappa=2.0)
        grad1 = torch.tensor([10.0, 20.0, 30.0])
        grad2 = torch.tensor([20.0, 40.0, 60.0])
        q1 = model.flux(grad_X=grad1)
        q2 = model.flux(grad_X=grad2)
        assert torch.allclose(q2, 2.0 * q1)

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FourierTransport(kappa=-0.026)

    def test_zero_kappa_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FourierTransport(kappa=0.0)

    def test_repr(self):
        model = FourierTransport(kappa=0.5)
        r = repr(model)
        assert "FourierTransport" in r
        assert "0.5" in r


# ======================================================================
# FickianTransport
# ======================================================================

class TestFickianTransport:
    """Tests for FickianTransport (Fick's law: j = -rho*D*grad(Y))."""

    def test_default_params(self):
        model = FickianTransport()
        assert model.D == pytest.approx(2.1e-5)
        assert model.rho == pytest.approx(1.225)

    def test_custom_params(self):
        model = FickianTransport(D=1e-4, rho=1000.0)
        assert model.D == pytest.approx(1e-4)
        assert model.rho == pytest.approx(1000.0)

    def test_flux_list_input(self):
        """Should accept a list and return a tensor."""
        model = FickianTransport(D=1e-4, rho=1000.0)
        j = model.flux(grad_X=[0.01, 0.0, 0.0])
        assert isinstance(j, torch.Tensor)
        # j = -rho * D * grad = -1000 * 1e-4 * 0.01 = -0.001
        assert j[0].item() == pytest.approx(-0.001)
        assert j[1].item() == pytest.approx(0.0)
        assert j[2].item() == pytest.approx(0.0)

    def test_flux_tensor_input(self):
        """Should accept a tensor and return a tensor."""
        model = FickianTransport(D=2.0, rho=0.5)
        grad = torch.tensor([1.0, 2.0, 3.0])
        j = model.flux(grad_X=grad)
        # j = -0.5 * 2.0 * grad = -1.0 * grad
        assert j[0].item() == pytest.approx(-1.0)
        assert j[1].item() == pytest.approx(-2.0)
        assert j[2].item() == pytest.approx(-3.0)

    def test_batch_flux(self):
        """Should work with batched input (n, 3)."""
        model = FickianTransport(D=1.0, rho=1.0)
        grad = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ])
        j = model.flux(grad_X=grad)
        assert j.shape == (2, 3)
        assert j[0, 0].item() == pytest.approx(-1.0)
        assert j[1, 1].item() == pytest.approx(-0.5)

    def test_zero_gradient_zero_flux(self):
        model = FickianTransport(D=2.1e-5, rho=1.225)
        j = model.flux(grad_X=[0.0, 0.0, 0.0])
        assert torch.allclose(j, torch.zeros(3, dtype=j.dtype))

    def test_flux_linearity(self):
        """Flux should scale linearly with gradient."""
        model = FickianTransport(D=1.5, rho=2.0)
        grad1 = torch.tensor([10.0, 20.0, 30.0])
        grad2 = torch.tensor([30.0, 60.0, 90.0])
        j1 = model.flux(grad_X=grad1)
        j2 = model.flux(grad_X=grad2)
        assert torch.allclose(j2, 3.0 * j1)

    def test_negative_D_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FickianTransport(D=-1e-5)

    def test_zero_D_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FickianTransport(D=0.0)

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FickianTransport(D=1e-5, rho=-1.0)

    def test_zero_rho_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FickianTransport(D=1e-5, rho=0.0)

    def test_repr(self):
        model = FickianTransport(D=1e-4, rho=1000.0)
        r = repr(model)
        assert "FickianTransport" in r
        assert "0.0001" in r
        assert "1000" in r
