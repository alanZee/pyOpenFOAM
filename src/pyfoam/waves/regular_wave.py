"""
Regular wave model — multi-component superposition.

Implements a regular wave model that combines multiple Airy (linear) wave
components by superposition.  Each component has its own amplitude, period,
direction, and phase shift.

Free-surface elevation:
    eta(x, t) = sum_i [ A_i * cos(k_i * (x*cos(theta_i) + y*sin(theta_i)) - omega_i * t + phi_i) ]

Velocity field:
    u_x = sum_i [ A_i * omega_i * cos(theta_i) * cosh(k_i*z) / sinh(k_i*d) * cos(phase_i) ]
    u_y = sum_i [ A_i * omega_i * sin(theta_i) * cosh(k_i*z) / sinh(k_i*d) * cos(phase_i) ]
    w   = sum_i [ A_i * omega_i * sinh(k_i*z) / sinh(k_i*d) * sin(phase_i) ]

where:
    A_i = component amplitude
    k_i = component wavenumber (from dispersion relation)
    omega_i = 2*pi/T_i = component angular frequency
    theta_i = wave direction angle (radians, 0 = +x)
    phi_i = phase shift
    d = water depth
    z = vertical coordinate above seabed (z in [0, d])

Reference:
    OpenFOAM ``waveModels::irregular`` — multi-component wave superposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel

__all__ = ["RegularWave", "WaveComponent"]


@dataclass
class WaveComponent:
    """单个波浪分量参数。

    Attributes
    ----------
    amplitude : float
        波幅 A (m).
    period : float
        周期 T (s).
    direction : float
        波向角 theta (rad)，0 表示 +x 方向.
    phase : float
        相位偏移 phi (rad).
    """
    amplitude: float
    period: float
    direction: float = 0.0
    phase: float = 0.0

    @property
    def angular_frequency(self) -> float:
        """角频率 omega = 2*pi/T (rad/s)."""
        return 2.0 * math.pi / self.period


@WaveModel.register("regular")
class RegularWave(WaveModel):
    """Regular wave model — multi-component superposition.

    Combines multiple Airy (linear) wave components by superposition.
    Each component satisfies the linear dispersion relation independently.

    Parameters
    ----------
    amplitude : float
        Primary wave amplitude A (m). Used as the default single-component
        amplitude when ``components`` is not specified.
    depth : float
        Water depth d (m). Shared by all components.
    period : float
        Primary wave period T (s). Used as the default single-component
        period when ``components`` is not specified.
    components : list of dict, optional
        Explicit list of wave components. Each dict has keys:
        ``amplitude``, ``period``, ``direction`` (default 0), ``phase`` (default 0).
        When ``None``, a single component is created from the primary parameters.
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        components: Optional[List[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(amplitude, depth, period)

        if components is not None:
            self._components = [
                WaveComponent(
                    amplitude=c["amplitude"],
                    period=c["period"],
                    direction=c.get("direction", 0.0),
                    phase=c.get("phase", 0.0),
                )
                for c in components
            ]
        else:
            self._components = [
                WaveComponent(amplitude=amplitude, period=period)
            ]

    @property
    def components(self) -> List[WaveComponent]:
        """返回所有波浪分量列表。"""
        return list(self._components)

    @property
    def n_components(self) -> int:
        """分量数量。"""
        return len(self._components)

    def _solve_k(self, comp: WaveComponent) -> float:
        """为指定分量求解波数 k（线性弥散关系）。"""
        omega = comp.angular_frequency
        d = self._depth
        g = GRAVITY

        # 深水近似作为初值
        k = omega**2 / g

        for _ in range(50):
            tanh_kd = math.tanh(k * d)
            f = g * k * tanh_kd - omega**2
            df = g * (tanh_kd + k * d / math.cosh(k * d) ** 2)
            dk = f / df
            k -= dk
            if abs(dk) < 1e-12:
                break

        return k

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute wave elevation from multi-component superposition.

        eta(x, t) = sum_i A_i * cos(k_i * x * cos(theta_i) - omega_i * t + phi_i)

        Note: In the 1D interface (x only), the y-component is assumed zero,
        so the phase reduces to k_i * x * cos(theta_i).

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        eta = torch.zeros_like(x)
        dtype = x.dtype

        for comp in self._components:
            k = self._solve_k(comp)
            omega = comp.angular_frequency
            theta = comp.direction
            phase = k * x * math.cos(theta) - omega * t + comp.phase
            eta = eta + comp.amplitude * torch.cos(phase)

        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity from multi-component superposition.

        In the 1D interface (x only, y=0), returns::

            u = sum_i A_i * omega_i * cos(theta_i) * cosh(k_i*z) / sinh(k_i*d) * cos(phase_i)
            w = sum_i A_i * omega_i * sinh(k_i*z) / sinh(k_i*d) * sin(phase_i)

        where ``u`` is the velocity in the x-direction (projected from 2D).

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m), z in [0, d].

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        u = torch.zeros_like(x)
        w = torch.zeros_like(x)
        d = self._depth
        dtype = x.dtype

        for comp in self._components:
            k = self._solve_k(comp)
            omega = comp.angular_frequency
            theta = comp.direction
            phase = k * x * math.cos(theta) - omega * t + comp.phase

            sinh_kd = math.sinh(k * d)
            cosh_kz = torch.cosh(k * z)
            sinh_kz = torch.sinh(k * z)

            coeff = comp.amplitude * omega / sinh_kd
            cos_theta = math.cos(theta)

            u = u + coeff * cos_theta * cosh_kz * torch.cos(phase)
            w = w + coeff * sinh_kz * torch.sin(phase)

        return u, w

    def __repr__(self) -> str:
        return (
            f"RegularWave(n_components={self.n_components}, "
            f"d={self._depth})"
        )
