"""
Device management and tensor configuration for pyOpenFOAM.

Provides automatic device detection (CPU/CUDA/MPS), default dtype configuration
for CFD precision, and context managers for temporary overrides.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Generator

import torch

__all__ = [
    "DeviceManager",
    "TensorConfig",
    "device_context",
    "get_device",
    "get_default_dtype",
]


@dataclass(frozen=True)
class DeviceCapabilities:
    """Immutable snapshot of available hardware devices."""

    cpu: bool = True
    cuda: bool = False
    mps: bool = False
    cuda_devices: int = 0

    @property
    def available_devices(self) -> list[str]:
        """Return list of available device names."""
        devices = ["cpu"]
        if self.cuda:
            devices.append("cuda")
        if self.mps:
            devices.append("mps")
        return devices


class DeviceManager:
    """Manages device detection and selection for tensor operations.

    Detects available hardware (CPU, CUDA, MPS) and provides a clean interface
    for device selection. Thread-safe singleton pattern via class-level state.

    Usage::

        dm = DeviceManager()
        device = dm.device  # auto-selected best device
        print(dm.capabilities)  # DeviceCapabilities(cpu=True, cuda=False, ...)
    """

    _instance: DeviceManager | None = None

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._capabilities = self._detect()
        self._device = self._select_best()

    @staticmethod
    def _detect() -> DeviceCapabilities:
        """Detect available hardware devices."""
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        return DeviceCapabilities(
            cpu=True,
            cuda=cuda_available,
            mps=mps_available,
            cuda_devices=cuda_count,
        )

    @staticmethod
    def _select_best() -> torch.device:
        """Select the best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Return detected hardware capabilities."""
        return self._capabilities

    @property
    def device(self) -> torch.device:
        """Return the currently selected device."""
        return self._device

    @device.setter
    def device(self, device: str | torch.device) -> None:
        """Set the active device.

        Args:
            device: Device string ('cpu', 'cuda', 'mps') or torch.device.

        Raises:
            ValueError: If the requested device is not available.
        """
        try:
            device = torch.device(device)
        except RuntimeError as e:
            raise ValueError(str(e)) from e
        available = self._capabilities.available_devices
        if device.type not in available:
            raise ValueError(
                f"Device '{device.type}' is not available. "
                f"Available: {available}"
            )
        self._device = device

    def is_available(self, device: str | torch.device) -> bool:
        """Check if a specific device type is available."""
        device = torch.device(device)
        return device.type in self._capabilities.available_devices

    def __repr__(self) -> str:
        return (
            f"DeviceManager(device={self._device}, "
            f"capabilities={self._capabilities})"
        )


class TensorConfig:
    """Global tensor configuration for CFD operations.

    Manages default dtype and device for tensor creation. Defaults to float64
    for CFD numerical precision (float32 causes divergence in iterative solvers).

    Usage::

        config = TensorConfig()
        t = config.zeros(3, 3)  # float64 on best device
        with config.override(dtype=torch.float32):
            t32 = config.zeros(3, 3)  # float32
    """

    def __init__(
        self,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self._default_dtype = dtype or torch.float64
        self._device_manager = DeviceManager()
        if device is not None:
            self._device_manager.device = device

    @property
    def dtype(self) -> torch.dtype:
        """Return the default dtype (float64 for CFD precision)."""
        return self._default_dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype) -> None:
        """Set the default dtype."""
        self._default_dtype = dtype

    @property
    def device(self) -> torch.device:
        """Return the current device."""
        return self._device_manager.device

    @device.setter
    def device(self, device: str | torch.device) -> None:
        """Set the current device."""
        self._device_manager.device = device

    @property
    def device_manager(self) -> DeviceManager:
        """Return the underlying DeviceManager."""
        return self._device_manager

    def tensor(self, data: torch.Tensor | list, **kwargs) -> torch.Tensor:
        """Create a tensor with default dtype and device.

        Args:
            data: Input data.
            **kwargs: Override dtype, device, etc.

        Returns:
            Tensor on the configured device with configured dtype.
        """
        kwargs.setdefault("dtype", self._default_dtype)
        kwargs.setdefault("device", self._device_manager.device)
        return torch.tensor(data, **kwargs)

    def zeros(self, *size: int, **kwargs) -> torch.Tensor:
        """Create a zeros tensor with default dtype and device."""
        kwargs.setdefault("dtype", self._default_dtype)
        kwargs.setdefault("device", self._device_manager.device)
        return torch.zeros(*size, **kwargs)

    def ones(self, *size: int, **kwargs) -> torch.Tensor:
        """Create a ones tensor with default dtype and device."""
        kwargs.setdefault("dtype", self._default_dtype)
        kwargs.setdefault("device", self._device_manager.device)
        return torch.ones(*size, **kwargs)

    def empty(self, *size: int, **kwargs) -> torch.Tensor:
        """Create an empty tensor with default dtype and device."""
        kwargs.setdefault("dtype", self._default_dtype)
        kwargs.setdefault("device", self._device_manager.device)
        return torch.empty(*size, **kwargs)

    def full(self, *size: int, fill_value: float = 0.0, **kwargs) -> torch.Tensor:
        """Create a filled tensor with default dtype and device."""
        kwargs.setdefault("dtype", self._default_dtype)
        kwargs.setdefault("device", self._device_manager.device)
        return torch.full(size, fill_value, **kwargs)

    @contextlib.contextmanager
    def override(
        self,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for temporary dtype/device overrides.

        Args:
            dtype: Temporary dtype override.
            device: Temporary device override.

        Usage::

            with config.override(dtype=torch.float32, device='cpu'):
                t = config.zeros(3)  # float32 on CPU
            # Back to defaults after
        """
        old_dtype = self._default_dtype
        old_device = self._device_manager.device
        try:
            if dtype is not None:
                self._default_dtype = dtype
            if device is not None:
                self._device_manager.device = device
            yield
        finally:
            self._default_dtype = old_dtype
            self._device_manager.device = old_device

    def __repr__(self) -> str:
        return (
            f"TensorConfig(dtype={self._default_dtype}, "
            f"device={self._device_manager.device})"
        )


# Module-level convenience instances (must be defined before device_context)
_device_manager = DeviceManager()
_tensor_config = TensorConfig()


def get_device() -> torch.device:
    """Return the current default device."""
    return _tensor_config.device


def get_default_dtype() -> torch.dtype:
    """Return the current default dtype."""
    return _tensor_config.dtype


@contextlib.contextmanager
def device_context(
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Generator[None, None, None]:
    """Module-level context manager for temporary device/dtype overrides.

    This is a convenience wrapper that modifies the global TensorConfig.

    Args:
        device: Temporary device override.
        dtype: Temporary dtype override.

    Usage::

        from pyfoam.core.device import device_context

        with device_context(device='cpu', dtype=torch.float32):
            # All operations use CPU + float32
            pass
    """
    with _tensor_config.override(dtype=dtype, device=device):
        yield
