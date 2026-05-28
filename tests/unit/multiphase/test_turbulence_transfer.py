"""Tests for multiphase turbulence transfer models."""

import pytest
import torch

from pyfoam.multiphase.turbulence_transfer import (
    TurbulenceTransferModel,
    ContinuousTurbulenceTransfer,
    DispersedTurbulenceTransfer,
)


class TestContinuousTurbulenceTransfer:
    """Test the ContinuousTurbulenceTransfer model."""

    def test_registration(self):
        """continuousTurbulenceTransfer is registered."""
        assert "continuousTurbulenceTransfer" in TurbulenceTransferModel.available_types()

    def test_factory_creation(self):
        """Model can be created via factory."""
        model = TurbulenceTransferModel.create("continuousTurbulenceTransfer")
        assert isinstance(model, ContinuousTurbulenceTransfer)

    def test_default_params(self):
        """Default parameters: C_t=1.0, sigma_t=0.9."""
        model = ContinuousTurbulenceTransfer()
        assert model.C_t == 1.0
        assert model.sigma_t == 0.9

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        model = ContinuousTurbulenceTransfer(C_t=2.0, sigma_t=0.7)
        assert model.C_t == 2.0
        assert model.sigma_t == 0.7

    def test_k_transfer_zero_alpha(self):
        """Zero dispersed fraction yields zero transfer."""
        model = ContinuousTurbulenceTransfer()
        alpha_d = torch.zeros(4, dtype=torch.float64)
        k_c = torch.ones(4, dtype=torch.float64) * 10.0
        k_d = torch.ones(4, dtype=torch.float64) * 5.0
        U_slip = torch.ones(4, dtype=torch.float64) * 2.0
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert torch.allclose(transfer, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_k_transfer_positive(self):
        """Positive transfer with non-zero alpha and slip."""
        model = ContinuousTurbulenceTransfer()
        alpha_d = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        k_c = torch.ones(3, dtype=torch.float64) * 10.0
        k_d = torch.ones(3, dtype=torch.float64) * 5.0
        U_slip = torch.ones(3, dtype=torch.float64) * 2.0
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert (transfer > 0).all()

    def test_k_transfer_scales_with_slip(self):
        """Transfer scales with U_slip^2."""
        model = ContinuousTurbulenceTransfer()
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([10.0], dtype=torch.float64)
        k_d = torch.tensor([5.0], dtype=torch.float64)

        U1 = torch.tensor([1.0], dtype=torch.float64)
        U2 = torch.tensor([2.0], dtype=torch.float64)

        t1 = model.compute_k_transfer(alpha_d, k_c, k_d, U1)
        t2 = model.compute_k_transfer(alpha_d, k_c, k_d, U2)

        # t2/t1 should be 4.0 (U^2 scaling)
        assert torch.allclose(t2 / t1, torch.tensor([4.0], dtype=torch.float64), rtol=1e-6)

    def test_epsilon_transfer_shape(self):
        """Dissipation transfer has correct shape."""
        model = ContinuousTurbulenceTransfer()
        n = 10
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.3
        k_c = torch.ones(n, dtype=torch.float64) * 10.0
        k_d = torch.ones(n, dtype=torch.float64) * 5.0
        eps_c = torch.ones(n, dtype=torch.float64) * 100.0
        U_slip = torch.ones(n, dtype=torch.float64) * 2.0

        eps_transfer = model.compute_epsilon_transfer(alpha_d, eps_c, k_c, k_d, U_slip)
        assert eps_transfer.shape == (n,)
        assert (eps_transfer >= 0).all()

    def test_zero_slip_velocity(self):
        """Zero slip velocity yields zero transfer."""
        model = ContinuousTurbulenceTransfer()
        alpha_d = torch.tensor([0.3], dtype=torch.float64)
        k_c = torch.tensor([10.0], dtype=torch.float64)
        k_d = torch.tensor([5.0], dtype=torch.float64)
        U_slip = torch.tensor([0.0], dtype=torch.float64)
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert torch.allclose(transfer, torch.tensor([0.0], dtype=torch.float64), atol=1e-10)


class TestDispersedTurbulenceTransfer:
    """Test the DispersedTurbulenceTransfer model."""

    def test_registration(self):
        """dispersedTurbulenceTransfer is registered."""
        assert "dispersedTurbulenceTransfer" in TurbulenceTransferModel.available_types()

    def test_factory_creation(self):
        """Model can be created via factory."""
        model = TurbulenceTransferModel.create("dispersedTurbulenceTransfer")
        assert isinstance(model, DispersedTurbulenceTransfer)

    def test_default_params(self):
        """Default parameters: C_t=1.0, sigma_t=0.9."""
        model = DispersedTurbulenceTransfer()
        assert model.C_t == 1.0
        assert model.sigma_t == 0.9

    def test_k_transfer_when_kc_greater(self):
        """Transfer is positive when k_c > k_d."""
        model = DispersedTurbulenceTransfer()
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([10.0], dtype=torch.float64)
        k_d = torch.tensor([2.0], dtype=torch.float64)
        U_slip = torch.tensor([1.0], dtype=torch.float64)
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert transfer.item() > 0

    def test_k_transfer_when_kd_greater(self):
        """Transfer is zero when k_d > k_c (no reverse transfer)."""
        model = DispersedTurbulenceTransfer()
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([2.0], dtype=torch.float64)
        k_d = torch.tensor([10.0], dtype=torch.float64)
        U_slip = torch.tensor([1.0], dtype=torch.float64)
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert torch.allclose(transfer, torch.tensor([0.0], dtype=torch.float64), atol=1e-10)

    def test_k_transfer_zero_alpha(self):
        """Zero dispersed fraction yields zero transfer."""
        model = DispersedTurbulenceTransfer()
        alpha_d = torch.zeros(4, dtype=torch.float64)
        k_c = torch.ones(4, dtype=torch.float64) * 10.0
        k_d = torch.ones(4, dtype=torch.float64) * 5.0
        U_slip = torch.ones(4, dtype=torch.float64) * 2.0
        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert torch.allclose(transfer, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_epsilon_transfer_shape(self):
        """Dissipation transfer has correct shape."""
        model = DispersedTurbulenceTransfer()
        n = 8
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.3
        k_c = torch.ones(n, dtype=torch.float64) * 10.0
        k_d = torch.ones(n, dtype=torch.float64) * 5.0
        eps_c = torch.ones(n, dtype=torch.float64) * 100.0
        U_slip = torch.ones(n, dtype=torch.float64) * 1.0

        eps_transfer = model.compute_epsilon_transfer(alpha_d, eps_c, k_c, k_d, U_slip)
        assert eps_transfer.shape == (n,)
        assert (eps_transfer >= 0).all()

    def test_batch_processing(self):
        """Works correctly with batched tensors."""
        model = DispersedTurbulenceTransfer()
        n = 100
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.3
        k_c = torch.rand(n, dtype=torch.float64) * 10.0 + 0.1
        k_d = torch.rand(n, dtype=torch.float64) * 5.0 + 0.1
        U_slip = torch.rand(n, dtype=torch.float64) * 3.0

        transfer = model.compute_k_transfer(alpha_d, k_c, k_d, U_slip)
        assert transfer.shape == (n,)
        assert (transfer >= 0).all()
