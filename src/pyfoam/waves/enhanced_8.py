"""
Enhanced wave models v8 — piston-type and flap-type wavemaker models.

Extends :class:`~pyfoam.waves.enhanced_7.WaveGenerationModel` with:

- :class:`PistonType` — piston-type wavemaker (horizontal plate motion)
- :class:`FlapType` — hinged flap-type wavemaker (rotating plate)

References:
    OpenFOAM ``waveModels:: piston`` and ``flap`` generation.
    Dean & Dalrymple (1991). "Water Wave Mechanics for Engineers and Scientists."
    Ursell et al. (1960). "The forced oscillations of a fluid cylinder."

Usage::

    from pyfoam.waves.enhanced_8 import PistonType, FlapType

    gen = PistonType(amplitude=1.0, depth=10.0, period=8.0)
    eta = gen.generate_elevation(x, t=0.0)
"""

from __future__ import annotations

import math

import torch

from pyfoam.waves.wave_model import GRAVITY, WaveModel
from pyfoam.waves.enhanced_7 import WaveGenerationModel

__all__ = ["PistonType", "FlapType"]


def _solve_dispersion(omega: float, d: float, max_iter: int = 50, tol: float = 1e-12) -> float:
    """Solve linear dispersion relation for wavenumber k.

    omega^2 = g * k * tanh(k * d)

    Args:
        omega: Angular frequency (rad/s).
        d: Water depth (m).
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance.

    Returns:
        Wavenumber k (1/m).
    """
    g = GRAVITY
    k = omega**2 / g  # 深水近似初值

    for _ in range(max_iter):
        tanh_kd = math.tanh(k * d)
        f = g * k * tanh_kd - omega**2
        df = g * (tanh_kd + k * d / math.cosh(k * d) ** 2)
        dk = f / df
        k -= dk
        if abs(dk) < tol:
            break
    return k


# ---------------------------------------------------------------------------
# PistonType
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("piston")
class PistonType(WaveGenerationModel):
    """Piston-type wavemaker.

    A piston wavemaker moves horizontally as a flat plate, generating
    waves by pushing the water column. The paddle displacement is:

        X(t) = X_0 * sin(omega * t)

    where X_0 is the paddle stroke amplitude, determined by the
    wavemaker transfer function:

        X_0 = A * sinh(k*d) / (k * d * (tanh(k*d) + k*d / cosh^2(k*d)))

    This is the most common wavemaker type in laboratory wave tanks.

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    paddle_position : float
        x-coordinate of the paddle (m, default 0).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        paddle_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._x_paddle = paddle_position

        # 预计算波数和 paddle 行程
        omega = self.angular_frequency
        self._k = _solve_dispersion(omega, depth)
        self._X0 = self._compute_paddle_amplitude()

    def _compute_paddle_amplitude(self) -> float:
        """Compute paddle stroke amplitude from wavemaker transfer function.

        X_0 = A * sinh(k*d) / (k * d * F(k, d))

        where F(k, d) = tanh(k*d) + k*d / cosh^2(k*d) (Ursell number correction).

        Returns:
            Paddle amplitude X_0 (m).
        """
        A = self._amplitude
        k = self._k
        d = self._depth

        sinh_kd = math.sinh(k * d)
        tanh_kd = math.tanh(k * d)

        # Ursell transfer function denominator
        F = tanh_kd + k * d / math.cosh(k * d) ** 2
        denom = k * d * F

        if abs(denom) < 1e-30:
            return A  # 退化情况

        return A * sinh_kd / denom

    @property
    def paddle_amplitude(self) -> float:
        """返回 paddle 行程振幅 X_0 (m)。"""
        return self._X0

    @property
    def paddle_position(self) -> float:
        """返回 paddle 位置 x_paddle (m)。"""
        return self._x_paddle

    def paddle_displacement(self, t: float) -> float:
        """Compute paddle displacement at time t.

        X(t) = X_0 * sin(omega * t)

        Args:
            t: Time (s).

        Returns:
            Paddle displacement (m).
        """
        return self._X0 * math.sin(self.angular_frequency * t)

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute generated wave elevation.

        eta = A * cos(k * (x - x_paddle) - omega * t)

        The wave propagates away from the paddle into the domain.

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self._k
        omega = self.angular_frequency
        phase = k * (x - self._x_paddle) - omega * t
        return self._amplitude * torch.cos(phase)

    def generate_velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute generated wave velocity.

        Uses Airy kinematics from the generated wave.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        k = self._k
        omega = self.angular_frequency
        d = self._depth
        phase = k * (x - self._x_paddle) - omega * t

        sinh_kd = math.sinh(k * d)
        coeff = self._amplitude * omega / sinh_kd

        u = coeff * torch.cosh(k * z) * torch.cos(phase)
        w = coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"PistonType(A={self._amplitude}, d={self._depth}, "
            f"T={self._period}, X0={self._X0:.4f})"
        )


# ---------------------------------------------------------------------------
# FlapType
# ---------------------------------------------------------------------------

@WaveGenerationModel.register("flap")
class FlapType(WaveGenerationModel):
    """Hinged flap-type wavemaker.

    A flap wavemaker rotates about a hinge point at the seabed (or
    below), generating waves with depth-dependent amplitude. The flap
    angle is:

        theta(t) = theta_0 * sin(omega * t)

    where theta_0 is the flap angular amplitude.

    The wave amplitude varies with depth as:
        A(z) = A_0 * cosh(k*(z+d)) / cosh(k*d)

    This produces more realistic deep-water waves compared to piston type.

    Parameters
    ----------
    amplitude : float
        Target wave amplitude A at the surface (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    hinge_depth : float
        Depth of hinge below seabed (m, default 0 = at seabed).
    paddle_position : float
        x-coordinate of the paddle (m, default 0).
    """

    def __init__(
        self,
        amplitude: float,
        depth: float,
        period: float,
        *,
        hinge_depth: float = 0.0,
        paddle_position: float = 0.0,
    ) -> None:
        super().__init__(amplitude, depth, period)
        self._hinge_depth = hinge_depth
        self._x_paddle = paddle_position

        omega = self.angular_frequency
        self._k = _solve_dispersion(omega, depth)

        # Flap 转角振幅
        self._theta0 = self._compute_flap_amplitude()

    def _compute_flap_amplitude(self) -> float:
        """Compute flap angular amplitude from wavemaker transfer function.

        For a flap hinged at the seabed:
            theta_0 = A * sinh(k*d) / (k * d * sinh(k*d) - cosh(k*d) + 1)

        Returns:
            Flap angular amplitude (rad).
        """
        A = self._amplitude
        k = self._k
        d = self._depth
        h = self._hinge_depth  # 铰链深度（海底以下）

        # 有效深度 = 水深 + 铰链深度
        d_eff = d + h

        sinh_kd = math.sinh(k * d_eff)
        cosh_kd = math.cosh(k * d_eff)

        # 转换函数：表面位移 -> 角振幅
        denom = k * d_eff * sinh_kd - cosh_kd + 1.0

        if abs(denom) < 1e-30:
            return A / d_eff if d_eff > 0 else 0.0

        return A * sinh_kd / denom

    @property
    def flap_amplitude(self) -> float:
        """返回 flap 角振幅 theta_0 (rad)。"""
        return self._theta0

    @property
    def hinge_depth(self) -> float:
        """返回铰链深度 (m)。"""
        return self._hinge_depth

    def flap_angle(self, t: float) -> float:
        """Compute flap angle at time t.

        theta(t) = theta_0 * sin(omega * t)

        Args:
            t: Time (s).

        Returns:
            Flap angle (rad).
        """
        return self._theta0 * math.sin(self.angular_frequency * t)

    def generate_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute generated wave elevation.

        eta = A * cos(k * (x - x_paddle) - omega * t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self._k
        omega = self.angular_frequency
        phase = k * (x - self._x_paddle) - omega * t
        return self._amplitude * torch.cos(phase)

    def generate_velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute generated wave velocity.

        Uses Airy kinematics from the generated wave.

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m).

        Returns:
            (u, w) — horizontal and vertical velocity (m/s).
        """
        k = self._k
        omega = self.angular_frequency
        d = self._depth
        phase = k * (x - self._x_paddle) - omega * t

        sinh_kd = math.sinh(k * d)
        coeff = self._amplitude * omega / sinh_kd

        u = coeff * torch.cosh(k * z) * torch.cos(phase)
        w = coeff * torch.sinh(k * z) * torch.sin(phase)
        return u, w

    def __repr__(self) -> str:
        return (
            f"FlapType(A={self._amplitude}, d={self._depth}, "
            f"T={self._period}, hinge_depth={self._hinge_depth})"
        )
