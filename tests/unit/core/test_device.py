"""Tests for DeviceManager, TensorConfig, and device_context."""

import pytest
import torch

from pyfoam.core.device import (
    DeviceCapabilities,
    DeviceManager,
    TensorConfig,
    device_context,
    get_default_dtype,
    get_device,
)


# ---------------------------------------------------------------------------
# DeviceCapabilities
# ---------------------------------------------------------------------------


class TestDeviceCapabilities:
    def test_cpu_always_available(self):
        caps = DeviceCapabilities()
        assert caps.cpu is True
        assert "cpu" in caps.available_devices

    def test_available_devices_list(self):
        caps = DeviceCapabilities(cpu=True, cuda=False, mps=False)
        assert caps.available_devices == ["cpu"]

    def test_with_cuda(self):
        caps = DeviceCapabilities(cuda=True, cuda_devices=2)
        assert "cuda" in caps.available_devices

    def test_with_mps(self):
        caps = DeviceCapabilities(mps=True)
        assert "mps" in caps.available_devices

    def test_frozen(self):
        caps = DeviceCapabilities()
        with pytest.raises(AttributeError):
            caps.cpu = False


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------


class TestDeviceManager:
    def test_singleton(self):
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        assert dm1 is dm2

    def test_default_device_is_cpu_if_no_accelerator(self):
        dm = DeviceManager()
        # On CI without GPU, device should be cpu
        if not torch.cuda.is_available():
            assert dm.device == torch.device("cpu")

    def test_capabilities_detected(self):
        dm = DeviceManager()
        caps = dm.capabilities
        assert isinstance(caps, DeviceCapabilities)
        assert caps.cpu is True
        # CUDA detection matches torch
        assert caps.cuda == torch.cuda.is_available()

    def test_set_device_to_cpu(self):
        dm = DeviceManager()
        dm.device = "cpu"
        assert dm.device == torch.device("cpu")

    def test_set_device_invalid_raises(self):
        dm = DeviceManager()
        with pytest.raises(ValueError):
            dm.device = "nonexistent_device"

    def test_is_available_cpu(self):
        dm = DeviceManager()
        assert dm.is_available("cpu") is True

    def test_is_available_cuda_matches_torch(self):
        dm = DeviceManager()
        assert dm.is_available("cuda") == torch.cuda.is_available()

    def test_repr(self):
        dm = DeviceManager()
        r = repr(dm)
        assert "DeviceManager" in r
        assert "device=" in r


# ---------------------------------------------------------------------------
# TensorConfig
# ---------------------------------------------------------------------------


class TestTensorConfig:
    def test_default_dtype_is_float64(self):
        cfg = TensorConfig()
        assert cfg.dtype == torch.float64

    def test_default_device_follows_manager(self):
        cfg = TensorConfig()
        dm = DeviceManager()
        assert cfg.device == dm.device

    def test_custom_dtype(self):
        cfg = TensorConfig(dtype=torch.float32)
        assert cfg.dtype == torch.float32

    def test_set_dtype(self):
        cfg = TensorConfig()
        cfg.dtype = torch.float32
        assert cfg.dtype == torch.float32

    def test_set_device(self):
        cfg = TensorConfig()
        cfg.device = "cpu"
        assert cfg.device == torch.device("cpu")

    def test_zeros_creates_tensor(self):
        cfg = TensorConfig()
        t = cfg.zeros(3, 4)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float64
        assert t.device == cfg.device

    def test_ones_creates_tensor(self):
        cfg = TensorConfig()
        t = cfg.ones(2, 5)
        assert t.shape == (2, 5)
        assert (t == 1).all()

    def test_empty_creates_tensor(self):
        cfg = TensorConfig()
        t = cfg.empty(10)
        assert t.shape == (10,)
        assert t.dtype == torch.float64

    def test_full_creates_tensor(self):
        cfg = TensorConfig()
        t = cfg.full(3, 3, fill_value=42.0)
        assert (t == 42.0).all()

    def test_tensor_from_list(self):
        cfg = TensorConfig()
        t = cfg.tensor([1.0, 2.0, 3.0])
        assert t.dtype == torch.float64
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_override_dtype(self):
        cfg = TensorConfig()
        assert cfg.dtype == torch.float64
        with cfg.override(dtype=torch.float32):
            assert cfg.dtype == torch.float32
            t = cfg.zeros(3)
            assert t.dtype == torch.float32
        assert cfg.dtype == torch.float64

    def test_override_device(self):
        cfg = TensorConfig()
        with cfg.override(device="cpu"):
            assert cfg.device == torch.device("cpu")

    def test_override_restores_on_exception(self):
        cfg = TensorConfig()
        original_dtype = cfg.dtype
        with pytest.raises(RuntimeError):
            with cfg.override(dtype=torch.float32):
                raise RuntimeError("test")
        assert cfg.dtype == original_dtype

    def test_repr(self):
        cfg = TensorConfig()
        r = repr(cfg)
        assert "TensorConfig" in r
        assert "float64" in r


# ---------------------------------------------------------------------------
# device_context
# ---------------------------------------------------------------------------


class TestDeviceContext:
    def test_context_sets_dtype(self):
        original = get_default_dtype()
        with device_context(dtype=torch.float32):
            assert get_default_dtype() == torch.float32
        assert get_default_dtype() == original

    def test_context_sets_device(self):
        with device_context(device="cpu"):
            assert get_device() == torch.device("cpu")

    def test_context_restores_on_exception(self):
        original = get_default_dtype()
        with pytest.raises(ValueError):
            with device_context(dtype=torch.float32):
                raise ValueError("boom")
        assert get_default_dtype() == original


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_get_device_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_get_default_dtype_returns_float64(self):
        assert get_default_dtype() == torch.float64
