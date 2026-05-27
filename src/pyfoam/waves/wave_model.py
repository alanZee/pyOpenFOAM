"""
Base wave model with RTS (Run-Time Selection) registry.

In OpenFOAM, wave models are selected at run-time from a dictionary
via the ``type`` keyword. This module provides an equivalent mechanism
in Python using a class-level registry and decorator pattern.

All wave models compute:

- Free-surface elevation: :math:`\\eta(x, t)`
- Velocity field: :math:`\\mathbf{u}(x, t, z)`

Usage::

    @WaveModel.register("airly")
    class AiryWave(WaveModel):
        ...

    wave = WaveModel.create("airy", amplitude=1.0, depth=10.0, period=8.0)

References:
    OpenFOAM ``waveModels`` — wave generation and absorption for coastal/offshore.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["WaveModel"]

# 标准重力加速度 (m/s²)
GRAVITY = 9.81


class WaveModel(ABC):
    """Abstract base class for wave models.

    Subclasses must implement :meth:`wave_elevation` and :meth:`velocity`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        wave = WaveModel.create("airy", amplitude=1.0, depth=10.0, period=8.0)

    Parameters
    ----------
    amplitude : float
        Wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    """

    # 类级 RTS 注册表：name -> class
    _registry: ClassVar[dict[str, Type[WaveModel]]] = {}

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
    ) -> None:
        self._amplitude = amplitude
        self._depth = depth
        self._period = period
        self._device = get_device()
        self._dtype = get_default_dtype()

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a wave model class under *name*.

        Usage::

            @WaveModel.register("airy")
            class AiryWave(WaveModel):
                ...
        """

        def decorator(model_cls: Type[WaveModel]) -> Type[WaveModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Wave model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> WaveModel:
        """Factory: create a wave model instance by registered *name*.

        Args:
            name: Registered wave model type name (e.g. ``"airy"``).
            **kwargs: Parameters forwarded to the wave model constructor.

        Returns:
            Instantiated wave model.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown wave model type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered wave model type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def amplitude(self) -> float:
        """Wave amplitude A (m)."""
        return self._amplitude

    @property
    def depth(self) -> float:
        """Water depth d (m)."""
        return self._depth

    @property
    def period(self) -> float:
        """Wave period T (s)."""
        return self._period

    @property
    def angular_frequency(self) -> float:
        """Angular frequency omega = 2*pi/T (rad/s)."""
        return 2.0 * math.pi / self._period

    @property
    def wavenumber(self) -> float:
        """Wavenumber k via linear dispersion relation.

        Solves: omega^2 = g*k*tanh(k*d) iteratively.
        """
        return self._solve_dispersion()

    # ------------------------------------------------------------------
    # Linear dispersion relation solver
    # ------------------------------------------------------------------

    def _solve_dispersion(self, max_iter: int = 50, tol: float = 1e-12) -> float:
        """Solve linear dispersion relation for wavenumber k.

        omega^2 = g * k * tanh(k * d)

        Uses Newton-Raphson iteration starting from deep-water approximation.

        Returns:
            Wavenumber k (1/m).
        """
        omega = self.angular_frequency
        d = self._depth
        g = GRAVITY

        # 深水近似作为初值
        k = omega**2 / g

        for _ in range(max_iter):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega**2
            df = g * (tanh_kd + k * d / math.cosh(k * d) ** 2)
            dk = f / df
            k -= dk
            if abs(dk) < tol:
                break

        return k

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute free-surface elevation eta(x, t).

        Parameters
        ----------
        x : torch.Tensor
            ``(n_points,)`` horizontal positions (m).
        t : float
            Time (s).

        Returns
        -------
        torch.Tensor
            ``(n_points,)`` wave elevation at each position (m).
        """
        ...

    @abstractmethod
    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity components (u, w) at given positions and depth.

        Parameters
        ----------
        x : torch.Tensor
            ``(n_points,)`` horizontal positions (m).
        t : float
            Time (s).
        z : torch.Tensor
            ``(n_points,)`` vertical positions above seabed (m), z in [0, d].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(u, w)`` — horizontal and vertical velocity components (m/s),
            each of shape ``(n_points,)``.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def type_name(self) -> str:
        """Return the registered type name for this wave model class."""
        for name, model_cls in self._registry.items():
            if isinstance(self, model_cls):
                return name
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"A={self._amplitude}, d={self._depth}, T={self._period})"
        )
