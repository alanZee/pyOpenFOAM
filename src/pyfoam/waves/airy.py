"""
Airy wave theory (linear wave theory).

Implements the first-order (Airy) wave solution:

- Free-surface elevation:
    eta = A * cos(k*x - omega*t)

- Horizontal velocity (u):
    u = A * omega * cosh(k*z) / sinh(k*d) * cos(k*x - omega*t)

- Vertical velocity (w):
    w = A * omega * sinh(k*z) / sinh(k*d) * sin(k*x - omega*t)

where:
    A = wave amplitude
    k = wavenumber (from linear dispersion relation)
    omega = 2*pi/T = angular frequency
    d = water depth
    z = vertical coordinate above seabed (z in [0, d])

The Airy wave theory is valid for small wave steepness: H/L << 1.

Reference:
    OpenFOAM ``waveModels::Airy``
"""

from __future__ import annotations

import math

import torch

from pyfoam.waves.wave_model import WaveModel

__all__ = ["AiryWave"]


@WaveModel.register("airy")
class AiryWave(WaveModel):
    """Linear Airy wave theory.

    First-order potential flow wave solution. Valid for small steepness
    (H/L << 1), i.e. deep water and small amplitude waves.

    Parameters
    ----------
    amplitude : float
        Wave amplitude A (m).
    depth : float
        Water depth d (m).
    period : float
        Wave period T (s).
    """

    def wave_elevation(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute Airy wave elevation.

        eta(x, t) = A * cos(k*x - omega*t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        phase = k * x - omega * t
        return self._amplitude * torch.cos(phase)

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Airy wave velocity field.

        u(x, t, z) = A * omega * cosh(k*z) / sinh(k*d) * cos(k*x - omega*t)
        w(x, t, z) = A * omega * sinh(k*z) / sinh(k*d) * sin(k*x - omega*t)

        Args:
            x: Horizontal positions (m).
            t: Time (s).
            z: Vertical positions above seabed (m), z in [0, d].

        Returns:
            (u, w) horizontal and vertical velocity (m/s).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        phase = k * x - omega * t

        sinh_kd = math.sinh(k * d)
        # z 从海底向上测量（z=0 海底，z=d 静水面）
        # 标准坐标 z_std = z - d，代入 cosh(k*(z_std+d)) = cosh(k*z)
        cosh_kz = torch.cosh(k * z)
        sinh_kz = torch.sinh(k * z)

        coeff = self._amplitude * omega / sinh_kd

        u = coeff * cosh_kz * torch.cos(phase)
        w = coeff * sinh_kz * torch.sin(phase)

        return u, w
