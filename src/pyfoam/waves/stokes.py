"""
Second-order Stokes wave theory.

Implements the 2nd-order Stokes wave correction:

Free-surface elevation:
    eta = A*cos(theta) + A2*cos(2*theta) + eta_mean

where:
    theta = k*x - omega*t
    A2 = k*A^2/2 * (3 - tanh^2(k*d)) / (4*tanh^3(k*d))
    eta_mean = k*A^2 / (2*sinh(2*k*d))  (mean water level setdown)

Horizontal velocity:
    u = A*omega*cosh(k*(z+d))/sinh(k*d)*cos(theta)
      + 3*k*A^2*omega*cosh(2*k*(z+d))/(4*sinh^4(k*d))*cos(2*theta)

Vertical velocity:
    w = A*omega*sinh(k*(z+d))/sinh(k*d)*sin(theta)
      + 3*k*A^2*omega*sinh(2*k*(z+d))/(4*sinh^4(k*d))*sin(2*theta)

Reference:
    OpenFOAM ``waveModels::Stokes2nd``
"""

from __future__ import annotations

import math

import torch

from pyfoam.waves.wave_model import WaveModel

__all__ = ["StokesWave"]


@WaveModel.register("stokes")
class StokesWave(WaveModel):
    """Second-order Stokes wave theory.

    Adds second-harmonic corrections to the Airy (linear) wave solution.
    Better representation of crest/trough asymmetry for moderate steepness.
    Valid for Ursell number Ur = H*L^2/d^3 not too large.

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
        """Compute 2nd-order Stokes wave elevation.

        eta = A*cos(theta) + A2*cos(2*theta) + eta_mean

        Args:
            x: Horizontal positions (m).
            t: Time (s).

        Returns:
            Wave elevation at each position (m).
        """
        k = self.wavenumber
        omega = self.angular_frequency
        d = self._depth
        A = self._amplitude
        theta = k * x - omega * t

        # 二阶 Stokes 修正系数
        tanh_kd = math.tanh(k * d)
        A2 = (k * A**2 / 2.0) * (3.0 - tanh_kd**2) / (4.0 * tanh_kd**3)
        eta_mean = (k * A**2) / (2.0 * math.sinh(2.0 * k * d))

        eta = A * torch.cos(theta) + A2 * torch.cos(2.0 * theta) + eta_mean
        return eta

    def velocity(
        self,
        x: torch.Tensor,
        t: float,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2nd-order Stokes wave velocity.

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
        A = self._amplitude
        theta = k * x - omega * t

        sinh_kd = math.sinh(k * d)
        sinh4_kd = sinh_kd**4

        # 一阶项（Airy）— z 从海底向上测量，需转换为 z_std = z - d
        cosh_kzd = torch.cosh(k * (z - d))
        sinh_kzd = torch.sinh(k * (z - d))
        coeff1 = A * omega / sinh_kd

        u1 = coeff1 * cosh_kzd * torch.cos(theta)
        w1 = coeff1 * sinh_kzd * torch.sin(theta)

        # 二阶项
        cosh_2kzd = torch.cosh(2.0 * k * (z - d))
        sinh_2kzd = torch.sinh(2.0 * k * (z - d))
        coeff2 = 3.0 * k * A**2 * omega / (4.0 * sinh4_kd)

        u2 = coeff2 * cosh_2kzd * torch.cos(2.0 * theta)
        w2 = coeff2 * sinh_2kzd * torch.sin(2.0 * theta)

        return u1 + u2, w1 + w2
