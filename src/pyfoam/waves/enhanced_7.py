"""
Enhanced wave models v7 — relaxation zone and wave generation base.

Extends the wave model framework with:

- :class:`RelaxationZone` — sponge layer with flexible blending profiles
- :class:`WaveGenerationModel` — abstract base for wave generation (wavemaker models)

References:
    OpenFOAM ``waveModels::relaxationZone``
    OpenFOAM ``waveModels::waveGeneration``
    Larsen & Dancy (1983). "Open boundaries in short wave simulations."

Usage::

    from pyfoam.waves.enhanced_7 import RelaxationZone, WaveGenerationModel

    zone = RelaxationZone(zone_length=20.0, depth=10.0, profile="cosine")
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.enhanced_5 import AbsorptionModel

__all__ = ["RelaxationZone", "WaveGenerationModel"]


# ---------------------------------------------------------------------------
# RelaxationZone
# ---------------------------------------------------------------------------

@AbsorptionModel.register("relaxationZone")
class RelaxationZone(AbsorptionModel):
    """Flexible relaxation zone (sponge layer) for wave absorption/generation.

    Implements multiple blending profiles for the relaxation function:
    - ``"cosine"``: w(s) = 0.5 * (1 - cos(pi * s))  [smooth, default]
    - ``"polynomial"``: w(s) = s^p  [power law, adjustable steepness]
    - ``"exponential"``: w(s) = (exp(alpha*s) - 1) / (exp(alpha) - 1)

    The relaxation zone blends between the computed wave field and a target
    state (still water or target wave). Weight 0 = full computed field,
    weight 1 = full target state.

    Parameters
    ----------
    zone_length : float
        Length of the relaxation zone (m).
    depth : float
        Water depth d (m).
    profile : str
        Blending profile: ``"cosine"`` (default), ``"polynomial"``, or ``"exponential"``.
    polynomial_power : float
        Power for polynomial profile (default 3.0).
    exponential_alpha : float
        Steepness for exponential profile (default 5.0).
    """

    def __init__(
        self,
        zone_length: float,
        depth: float,
        *,
        profile: str = "cosine",
        polynomial_power: float = 3.0,
        exponential_alpha: float = 5.0,
    ) -> None:
        super().__init__(zone_length, depth)
        self._profile = profile
        self._poly_power = polynomial_power
        self._exp_alpha = exponential_alpha

    @property
    def profile(self) -> str:
        """返回混合剖面类型。"""
        return self._profile

    def relaxation_weight(self, x: torch.Tensor, x_zone_start: float) -> torch.Tensor:
        """Compute relaxation blending weight based on selected profile.

        Parameters
        ----------
        x : torch.Tensor
            Position of each point.
        x_zone_start : float
            Start of the relaxation zone.

        Returns
        -------
        torch.Tensor
            Blending weight in [0, 1] for each point.
        """
        L = self._zone_length
        s = ((x - x_zone_start) / L).clamp(0.0, 1.0)

        if self._profile == "cosine":
            return 0.5 * (1.0 - torch.cos(math.pi * s))
        elif self._profile == "polynomial":
            return s.pow(self._poly_power)
        elif self._profile == "exponential":
            alpha = self._exp_alpha
            return (torch.exp(alpha * s) - 1.0) / (math.exp(alpha) - 1.0)
        else:
            # 退回 cosine
            return 0.5 * (1.0 - torch.cos(math.pi * s))

    def absorb(
        self,
        eta: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        x_zone_start: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply relaxation zone blending.

        Blends computed field toward still water (eta=0, u=0, w=0):
            eta_abs = eta * (1 - weight)
            u_abs = u * (1 - weight)
            w_abs = w * (1 - weight)

        Parameters
        ----------
        eta : torch.Tensor
            Wave elevation field.
        u : torch.Tensor
            Horizontal velocity field.
        w : torch.Tensor
            Vertical velocity field.
        x : torch.Tensor
            Position of each point (m).
        x_zone_start : float
            x-coordinate where the zone starts.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Blended (eta, u, w) fields.
        """
        wt = self.relaxation_weight(x, x_zone_start)
        factor = 1.0 - wt

        return eta * factor, u * factor, w * factor

    def __repr__(self) -> str:
        return (
            f"RelaxationZone(L_zone={self._zone_length}, "
            f"d={self._depth}, profile={self._profile!r})"
        )


# ---------------------------------------------------------------------------
# WaveGenerationModel (ABC)
# ---------------------------------------------------------------------------

class WaveGenerationModel(ABC):
    """Abstract base class for wave generation models (wavemakers).

    Wave generation models produce waves at domain boundaries using
    analytical wavemaker theory. They compute the paddle displacement
    or velocity needed to generate a target wave.

    Subclasses must implement :meth:`generate_elevation` and
    :meth:`generate_velocity`.

    RTS registry for generation models.

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    """

    _registry: ClassVar[dict[str, Type[WaveGenerationModel]]] = {}

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
    ) -> None:
        self._amplitude = amplitude
        self._depth = depth
        self._period = period

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a generation model under *name*."""

        def decorator(model_cls: Type[WaveGenerationModel]) -> Type[WaveGenerationModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Wave generation model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> WaveGenerationModel:
        """Factory: create a generation model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown wave generation model type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered generation model type names."""
        return sorted(cls._registry.keys())

    @property
    def amplitude(self) -> float:
        """目标波幅 (m)。"""
        return self._amplitude

    @property
    def depth(self) -> float:
        """水深 (m)。"""
        return self._depth

    @property
    def period(self) -> float:
        """波周期 (s)。"""
        return self._period

    @property
    def angular_frequency(self) -> float:
        """角频率 omega = 2*pi/T (rad/s)。"""
        return 2.0 * math.pi / self._period

    @abstractmethod
    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute generated wave elevation at domain boundary.

        Parameters
        ----------
        x : torch.Tensor
            ``(n_points,)`` horizontal positions (m).
        t : float
            Time (s).

        Returns
        -------
        torch.Tensor
            ``(n_points,)`` wave elevation (m).
        """
        ...

    @abstractmethod
    def generate_velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute generated wave velocity at domain boundary.

        Parameters
        ----------
        x : torch.Tensor
            ``(n_points,)`` horizontal positions (m).
        t : float
            Time (s).
        z : torch.Tensor
            ``(n_points,)`` vertical positions above seabed (m).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(u, w)`` — horizontal and vertical velocity (m/s).
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"A={self._amplitude}, d={self._depth}, T={self._period})"
        )
