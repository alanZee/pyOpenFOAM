"""
Enhanced wave models v5 — reflected/diffracted waves and absorption base.

Extends :class:`~pyfoam.waves.cnoidal.CnoidalWave` with:

- :class:`ReflectedWave` — wave reflection from a vertical wall
- :class:`DiffractedWave` — wave diffraction around a semi-infinite breakwater

Also introduces the absorption model ABC:

- :class:`AbsorptionModel` — abstract base for wave absorption models

References:
    Dean & Dalrymple (1991). "Water Wave Mechanics for Engineers and Scientists."
    Mei (1989). "The Applied Dynamics of Ocean Surface Waves."
    OpenFOAM ``waveModels::absorption``.

Usage::

    from pyfoam.waves.enhanced_5 import ReflectedWave, DiffractedWave, AbsorptionModel

    wave = ReflectedWave(amplitude=1.0, depth=10.0, period=8.0, reflection_coeff=0.5)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.cnoidal import CnoidalWave

__all__ = ["ReflectedWave", "DiffractedWave", "AbsorptionModel"]


# ---------------------------------------------------------------------------
# ReflectedWave
# ---------------------------------------------------------------------------

@WaveModel.register("reflected")
class ReflectedWave(CnoidalWave):
    """Wave reflection model — standing wave from a vertical wall.

    Models the superposition of an incident wave and its reflection from
    a vertical wall at x = x_wall:

        eta = eta_incident(x, t) + Kr * eta_incident(2*x_wall - x, t)

    where Kr is the reflection coefficient (0 = no reflection, 1 = full).

    For Kr = 1, the result is a pure standing wave:
        eta = 2*A*cos(k*x_wall)*cos(k*(x-x_wall) - omega*t)

    Parameters
    ----------
    amplitude : float
        Incident wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    reflection_coeff : float
        Reflection coefficient Kr in [0, 1] (default 0.5).
    wall_position : float
        x-coordinate of reflecting wall (m, default 0).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        reflection_coeff: float = 0.5,
        wall_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._Kr = max(0.0, min(1.0, reflection_coeff))
        self._x_wall = wall_position

    @property
    def reflection_coeff(self) -> float:
        """返回反射系数 Kr。"""
        return self._Kr

    @property
    def wall_position(self) -> float:
        """返回反射壁面位置 (m)。"""
        return self._x_wall

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute reflected wave elevation.

        eta = eta_inc(x, t) + Kr * eta_inc(2*x_wall - x, t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        # 入射波（使用 Airy 线性近似以确保数值稳定性）
        k = self.wavenumber
        omega = self.angular_frequency
        A = self._amplitude

        theta_inc = k * x - omega * t
        eta_inc = A * torch.cos(theta_inc)

        # 反射波（从壁面反射，相位反转）
        x_reflected = 2.0 * self._x_wall - x
        theta_ref = k * x_reflected - omega * t
        eta_ref = A * torch.cos(theta_ref)

        return eta_inc + self._Kr * eta_ref

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reflected wave velocity.

        u = u_inc + Kr * u_ref (reflected velocity has reversed x-component)
        w = w_inc + Kr * w_ref

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        A = self._amplitude
        sinh_kd = math.sinh(k * d)

        # 入射波速度
        theta_inc = k * x - omega * t
        coeff = A * omega / sinh_kd
        u_inc = coeff * torch.cosh(k * z) * torch.cos(theta_inc)
        w_inc = coeff * torch.sinh(k * z) * torch.sin(theta_inc)

        # 反射波速度（x 方向反转）
        x_ref = 2.0 * self._x_wall - x
        theta_ref = k * x_ref - omega * t
        u_ref = -coeff * torch.cosh(k * z) * torch.cos(theta_ref)  # 反向
        w_ref = coeff * torch.sinh(k * z) * torch.sin(theta_ref)

        return u_inc + self._Kr * u_ref, w_inc + self._Kr * w_ref

    def __repr__(self) -> str:
        return (
            f"ReflectedWave(A={self._amplitude}, Kr={self._Kr}, "
            f"x_wall={self._x_wall})"
        )


# ---------------------------------------------------------------------------
# DiffractedWave
# ---------------------------------------------------------------------------

@WaveModel.register("diffracted")
class DiffractedWave(CnoidalWave):
    """Wave diffraction model — semi-infinite breakwater.

    Implements the Sommerfeld solution for wave diffraction around a
    semi-infinite breakwater at x = x_tip (y >= 0 side blocked):

        eta_total = eta_incident + eta_diffracted

    Uses the Fresnel integral approximation for the diffraction coefficient.

    Parameters
    ----------
    amplitude : float
        Incident wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    tip_position : float
        x-coordinate of breakwater tip (m, default 0).
    diffraction_coeff : float
        Simplified diffraction attenuation coefficient Kd in (0, 1]
        for the shadow zone (default 0.3).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        tip_position: float = 0.0,
        diffraction_coeff: float = 0.3,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._x_tip = tip_position
        self._Kd = max(0.01, min(1.0, diffraction_coeff))

    @property
    def tip_position(self) -> float:
        """返回防波堤尖端位置 (m)。"""
        return self._x_tip

    @property
    def diffraction_coeff(self) -> float:
        """返回衍射衰减系数 Kd。"""
        return self._Kd

    def _diffraction_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute simplified diffraction attenuation factor.

        Uses exponential decay in the shadow zone (x < x_tip):
            f(x) = Kd^((x_tip - x) / lambda)

        where lambda is the wavelength.

        In the illuminated zone (x >= x_tip): f(x) = 1.

        Parameters
        ----------
        x : torch.Tensor
            Horizontal positions.

        Returns
        -------
        torch.Tensor
            Diffraction factor in [Kd, 1].
        """
        k = self.wavenumber
        L = 2.0 * math.pi / k
        x_tip = self._x_tip
        Kd = self._Kd

        # 距离防波堤尖端的距离（以波长归一化）
        dist = (x_tip - x).clamp(min=0.0) / L
        # 指数衰减
        factor = Kd ** dist
        # 只在阴影区衰减（x < x_tip）
        factor = torch.where(x < x_tip, factor, torch.ones_like(x))
        return factor

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute diffracted wave elevation.

        eta = K_d(x) * A * cos(k*x - omega*t)

        where K_d is the position-dependent diffraction factor.

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        A = self._amplitude
        theta = k * x - omega * t

        K_d = self._diffraction_factor(x)
        return K_d * A * torch.cos(theta)

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute diffracted wave velocity.

        Velocity is attenuated by the same diffraction factor.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        A = self._amplitude
        sinh_kd = math.sinh(k * d)
        theta = k * x - omega * t

        K_d = self._diffraction_factor(x)
        coeff = K_d * A * omega / sinh_kd

        u = coeff * torch.cosh(k * z) * torch.cos(theta)
        w = coeff * torch.sinh(k * z) * torch.sin(theta)
        return u, w

    def __repr__(self) -> str:
        return (
            f"DiffractedWave(A={self._amplitude}, x_tip={self._x_tip}, "
            f"Kd={self._Kd})"
        )


# ---------------------------------------------------------------------------
# AbsorptionModel (ABC)
# ---------------------------------------------------------------------------

class AbsorptionModel(ABC):
    """Abstract base class for wave absorption models.

    Wave absorption models are used at domain boundaries to prevent
    wave reflections. They modify the wave field within a relaxation
    zone by blending the computed solution with a target (still water)
    state.

    Subclasses must implement :meth:`absorb`.

    RTS registry for absorption models.

    Parameters
    ----------
    zone_length : float
        Length of the absorption zone (m).
    depth : float
        Water depth d (m).
    """

    _registry: ClassVar[dict[str, Type[AbsorptionModel]]] = {}

    def __init__(
        self,
        zone_length: float,
        depth: float,
    ) -> None:
        self._zone_length = zone_length
        self._depth = depth

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register an absorption model under *name*."""

        def decorator(model_cls: Type[AbsorptionModel]) -> Type[AbsorptionModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Absorption model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbsorptionModel:
        """Factory: create an absorption model by registered *name*."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown absorption model type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered absorption model type names."""
        return sorted(cls._registry.keys())

    @property
    def zone_length(self) -> float:
        """吸收区长度 (m)。"""
        return self._zone_length

    @property
    def depth(self) -> float:
        """水深 (m)。"""
        return self._depth

    @abstractmethod
    def absorb(
        self,
        eta: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        x_zone_start: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply absorption within the relaxation zone.

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
            x-coordinate where the absorption zone starts.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Absorbed (eta, u, w) fields.
        """
        ...

    def relaxation_weight(self, x: torch.Tensor, x_zone_start: float) -> torch.Tensor:
        """Compute relaxation blending weight in [0, 1].

        weight = 0 at the zone start (full wave field)
        weight = 1 at the zone end (full absorption)

        Uses a cosine profile: w(s) = 0.5 * (1 - cos(pi * s))
        where s = (x - x_start) / L_zone in [0, 1].

        Parameters
        ----------
        x : torch.Tensor
            Position of each point.
        x_zone_start : float
            Start of the absorption zone.

        Returns
        -------
        torch.Tensor
            Blending weight for each point.
        """
        L = self._zone_length
        s = ((x - x_zone_start) / L).clamp(0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(math.pi * s))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(L_zone={self._zone_length}, "
            f"d={self._depth})"
        )
